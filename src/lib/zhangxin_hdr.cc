/*
    Copyright (C) 2026 zhangxin
*/

#include "zhangxin_hdr.h"
#include "dcpomatic_assert.h"
#include "config.h"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <vector>
#include <mutex>
#include <memory>
#include <onnxruntime_cxx_api.h>

using std::shared_ptr;
using std::make_shared;
using std::min;
using std::max;
using std::string;
using std::vector;

// === Global ORT Session Management ===
struct OrtContext {
    Ort::Env env;
    std::unique_ptr<Ort::Session> session;
    Ort::RunOptions run_options;
    
    OrtContext() : env(ORT_LOGGING_LEVEL_WARNING, "ZhangxinHDR") {
        run_options = Ort::RunOptions{nullptr};
    }
};

static std::unique_ptr<OrtContext> g_ort_ctx;
static std::mutex g_ort_mutex;

static void init_ort_session(const std::string& model_path) {
    std::lock_guard<std::mutex> lock(g_ort_mutex);
    if (g_ort_ctx && g_ort_ctx->session) return; // Already initialized

    if (model_path.empty()) {
        std::cerr << "[ZHANGXIN_HDR] Error: Model path is empty!" << std::endl;
        return;
    }

    try {
        if (!g_ort_ctx) g_ort_ctx.reset(new OrtContext());
        
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(4); 
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        std::cout << "[ZHANGXIN_HDR] Loading ONNX model from " << model_path << " ..." << std::endl;
        g_ort_ctx->session.reset(new Ort::Session(g_ort_ctx->env, model_path.c_str(), session_options));
        std::cout << "[ZHANGXIN_HDR] Model loaded successfully." << std::endl;
    } catch (const Ort::Exception& e) {
        std::cerr << "[ZHANGXIN_HDR] ORT Exception: " << e.what() << std::endl;
        g_ort_ctx.reset(); 
    }
}

static float decode_transfer(float v, ZhangxinHDR::TransferFunction tf) {
    // Anti-NaN clipping in encoding domain.
    // NOTE: We clip negative values here for safety.
    v = std::max(0.0f, std::min(v, 1.0f));
    
    switch (tf) {
        case ZhangxinHDR::TransferFunction::REC709_SCENE_LINEAR: {
            // Rec.709 inverse OETF (Scene Linear)
            // L = V/4.5                   if V < 0.081
            // L = ((V+0.099)/1.099)^(1/0.45) otherwise
            const float k = 4.5f;
            const float t = 0.018f;        // Knee at 0.018 linear
            const float Vt = k * t;        // Knee at ~0.081 encoded
            const float a = 0.099f;
            const float b = 1.099f;
            const float gamma = 1.0f / 0.45f; // ~2.222
            
            if (v < Vt) return v / k;
            else return std::pow((v + a) / b, gamma); 
        }
        case ZhangxinHDR::TransferFunction::GAMMA_24:
            return std::pow(v, 2.4f);
        case ZhangxinHDR::TransferFunction::GAMMA_26:
            return std::pow(v, 2.6f);
        default:
            return std::pow(v, 2.4f);
    }
}

// === Color Matrices ===
static const float M_RGB_XYZ[9] = {
    0.4124564f, 0.3575761f, 0.1804375f,
    0.2126729f, 0.7151522f, 0.0721750f,
    0.0193339f, 0.1191920f, 0.9503041f
};

static const float M_XYZ_RGB[9] = {
     3.2404542f, -1.5371385f, -0.4985314f,
    -0.9692660f,  1.8760108f,  0.0415560f,
     0.0556434f, -0.2040259f,  1.0572252f
};

// Transform Utils


// From src/utils/colorspace.py in your repo (Copied values)
// _M_XYZ_TO_P3
static const float M_XYZ_P3[9] = {
     2.49349691f, -0.93138362f, -0.40271078f,
    -0.82948897f,  1.76266406f,  0.02362469f,
     0.03584583f, -0.07617239f,  0.95688452f
};
// _M_P3_TO_XYZ
static const float M_P3_XYZ[9] = {
    0.48657095f, 0.26566769f, 0.19821729f,
    0.22897457f, 0.69173852f, 0.07928691f,
    0.00000000f, 0.04511338f, 1.04394436f
};


void
ZhangxinHDR::log_stats(const string& tag, const Stats& s) {
    if (s.y_max >= 0.0f) {
        std::cout << "[ZHANGXIN_HDR] " << tag << " (Nits): "
                  << "Y_min=" << s.y_min << " "
                  << "Y_max=" << s.y_max << " "
                  << "Y_median=" << s.y_median << " "
                  << "Y_p99=" << s.y_p99 << std::endl;
    }
}

ZhangxinHDR::Config
ZhangxinHDR::Config::load_from_config()
{
    Config c;
    
    auto global_config = ::Config::instance();
    
    if (global_config->zhangxin_hdr_enable()) {
        c.enable = *global_config->zhangxin_hdr_enable();
    }
    
    if (global_config->zhangxin_hdr_model_path()) {
        c.model_path = *global_config->zhangxin_hdr_model_path();
    }
    
    if (global_config->zhangxin_hdr_hue_lock()) {
        c.hue_lock = *global_config->zhangxin_hdr_hue_lock();
    }
    
    // Validation
    if (c.enable && c.model_path.empty()) {
        std::cerr << "[ZHANGXIN_HDR] Warning: HDR enabled but model path not set. "
                  << "Please configure it in Edit -> Preferences -> Zhangxin HDR." << std::endl;
    }
    
    return c;
}

shared_ptr<Image>
ZhangxinHDR::process(shared_ptr<const Image> image, Config config)
{
    if (!config.enable) {
        return make_shared<Image>(*image);
    }
    
    // Init ORT if needed
    if (!g_ort_ctx || !g_ort_ctx->session) {
        init_ort_session(config.model_path);
    }
    
    // If Init failed, fallback to Passthrough
    if (!g_ort_ctx || !g_ort_ctx->session) {
        return make_shared<Image>(*image);
    }

    DCPOMATIC_ASSERT(image->pixel_format() == AV_PIX_FMT_RGB48LE);
    dcp::Size size = image->size();
    int w = size.width;
    int h = size.height;

    shared_ptr<Image> out = make_shared<Image>(image->pixel_format(), size, image->alignment());
    uint8_t* in_data = image->data()[0];
    uint8_t* out_data = out->data()[0];
    int stride = image->stride()[0];
    const float i_max = 65535.0f;

    // === Preview Generation (NOT for Mastering) ===
    // This path scales HDR Nits back to 0-1 range for simple visualization.
    // Do not use this output for DCP creation.

    // Allocate Input Tensor (NCHW, 1x3xHxW)
    size_t input_tensor_size = w * h * 3;
    std::vector<float> input_tensor_values(input_tensor_size);
    
    // Pre-Process Loop: 
    // 1. SDR RGB Gamma 2.4 -> Linear RGB
    // 2. Linear RGB -> XYZ Relative
    // 3. XYZ Relative -> P3 Relative
    // 4. P3 Relative -> P3 Nits (x 48.0) -> Tensor NCHW
    
    float* p_plane_0 = input_tensor_values.data();           // Channel 0
    float* p_plane_1 = input_tensor_values.data() + w * h;   // Channel 1
    float* p_plane_2 = input_tensor_values.data() + 2 * w * h; // Channel 2
    
    for (int y = 0; y < h; ++y) {
        uint16_t* p_in = reinterpret_cast<uint16_t*>(in_data + y * stride);
        for (int x = 0; x < w; ++x) {
            // SDR -> Linear RGB
            float r = decode_transfer(p_in[0] / i_max, config.transfer_function);
            float g = decode_transfer(p_in[1] / i_max, config.transfer_function);
            float b = decode_transfer(p_in[2] / i_max, config.transfer_function);
            
            // RGB -> XYZ
            float X = r * M_RGB_XYZ[0] + g * M_RGB_XYZ[1] + b * M_RGB_XYZ[2];
            float Y = r * M_RGB_XYZ[3] + g * M_RGB_XYZ[4] + b * M_RGB_XYZ[5];
            float Z = r * M_RGB_XYZ[6] + g * M_RGB_XYZ[7] + b * M_RGB_XYZ[8];

            // XYZ -> P3 (Model Expectations)
            float P3_R = X * M_XYZ_P3[0] + Y * M_XYZ_P3[1] + Z * M_XYZ_P3[2];
            float P3_G = X * M_XYZ_P3[3] + Y * M_XYZ_P3[4] + Z * M_XYZ_P3[5];
            float P3_B = X * M_XYZ_P3[6] + Y * M_XYZ_P3[7] + Z * M_XYZ_P3[8];
            
            // P3 Relative -> 48 Nits
            float P3_R_nit = P3_R * 48.0f;
            float P3_G_nit = P3_G * 48.0f;
            float P3_B_nit = P3_B * 48.0f;
            
            // Write to NCHW
            int plane_idx = y * w + x;
            p_plane_0[plane_idx] = P3_R_nit;
            p_plane_1[plane_idx] = P3_G_nit;
            p_plane_2[plane_idx] = P3_B_nit;

            p_in += 3;
        }
    }

    // Run Inference
    std::vector<int64_t> input_shape = {1, 3, h, w};
    
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_shape.data(), input_shape.size());

    const char* input_names[] = {"input"};
    const char* output_names[] = {"gain_map", "hdr_pred"}; // We want hdr_pred
    
    // We only care about hdr_pred (idx 1)
    auto output_tensors = g_ort_ctx->session->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 2);
    
    // Output 1 is hdr_pred
    float* floatarr = output_tensors[1].GetTensorMutableData<float>();
    
    // Post-Process Loop:
    // 1. Read NCHW Tensor (P3 Absolute Nits)
    // 2. P3 Nits -> XYZ Nits
    // 3. XYZ Nits -> XYZ Normalized (Safe Mode 0..1 for DCP Container)
    //    Note: Mapping 300 nits -> 1.0 (48 nits container) or 48->1.0 (Clip)?
    //    User wants to verify dynamic range.
    //    Let's do: XYZ_Norm = XYZ_Abs / 48.0 => Allows > 1.0 for HDR!
    //    DCP-o-matic internal Image is float? No, RGB48 uint16.
    //    So we clamp to 1.0. This means anything > 48 nits gets clipped.
    //    TO SEE HDR: We must attenuate. let's scale by 1.0/300.0 so 300 nits -> 1.0.
    //    This makes the image look "dark" in SDR but preserves data.
    float display_scale = 1.0f / 300.0f; 

    // Ptrs to output planes
    const float* out_p0 = floatarr;             // R
    const float* out_p1 = floatarr + w * h;     // G
    const float* out_p2 = floatarr + 2 * w * h; // B

    for (int y = 0; y < h; ++y) {
        uint16_t* p_out = reinterpret_cast<uint16_t*>(out_data + y * stride);
        for (int x = 0; x < w; ++x) {
            int plane_idx = y * w + x;
            float P3_R_nit = out_p0[plane_idx];
            float P3_G_nit = out_p1[plane_idx];
            float P3_B_nit = out_p2[plane_idx];
            
            // P3 Nits -> XYZ Nits
            float X_nit = P3_R_nit * M_P3_XYZ[0] + P3_G_nit * M_P3_XYZ[1] + P3_B_nit * M_P3_XYZ[2];
            float Y_nit = P3_R_nit * M_P3_XYZ[3] + P3_G_nit * M_P3_XYZ[4] + P3_B_nit * M_P3_XYZ[5];
            float Z_nit = P3_R_nit * M_P3_XYZ[6] + P3_G_nit * M_P3_XYZ[7] + P3_B_nit * M_P3_XYZ[8];
            
            // Scale and Clamp (0.005 nits -> ~0)
            float X = X_nit * display_scale;
            float Y = Y_nit * display_scale;
            float Z = Z_nit * display_scale;
            
            // XYZ -> Linear RGB
            float r = X * M_XYZ_RGB[0] + Y * M_XYZ_RGB[1] + Z * M_XYZ_RGB[2];
            float g = X * M_XYZ_RGB[3] + Y * M_XYZ_RGB[4] + Z * M_XYZ_RGB[5];
            float b = X * M_XYZ_RGB[6] + Y * M_XYZ_RGB[7] + Z * M_XYZ_RGB[8];

            p_out[0] = static_cast<uint16_t>(max(0.0f, min(1.0f, r)) * i_max);
            p_out[1] = static_cast<uint16_t>(max(0.0f, min(1.0f, g)) * i_max);
            p_out[2] = static_cast<uint16_t>(max(0.0f, min(1.0f, b)) * i_max);

            p_out += 3;
        }
    }

    return out;
}

// === New PQ-Ready HDR Pipeline ===
// NOTE: This HDR pipeline is EXPERIMENTAL until MXF TransferCharacteristic UL is set (DCI HDR Addendum).
ZhangxinHDR::HDRXYZResult
ZhangxinHDR::process_to_hdr_xyz(shared_ptr<const Image> image, Config config)
{
    HDRXYZResult result;
    DCPOMATIC_ASSERT(image->pixel_format() == AV_PIX_FMT_RGB48LE);
    dcp::Size size = image->size();
    int w = size.width;
    int h = size.height;
    
    if (!config.enable) {
        // Return empty result if not enabled
        result.width = 0;
        result.height = 0;
        return result;
    }
    
    // Init ORT if needed
    if (!g_ort_ctx || !g_ort_ctx->session) {
        init_ort_session(config.model_path);
    }
    
    // If Init failed, throw error. We do NOT use fallback in production or strict testing.
    if (!g_ort_ctx || !g_ort_ctx->session) {
        throw std::runtime_error("[ZHANGXIN_HDR] Critical Error: HDR Model not loaded! Configure model path in Edit -> Preferences -> Zhangxin HDR.");
    }



    result.width = w;
    result.height = h;
    result.x.resize(w * h);
    result.y.resize(w * h);
    result.z.resize(w * h);

    uint8_t* in_data = image->data()[0];
    int stride = image->stride()[0];
    const float i_max = 65535.0f;

    // === Pre-Process: SDR RGB -> P3 Nits (model input) ===
    size_t input_tensor_size = w * h * 3;
    std::vector<float> input_tensor_values(input_tensor_size);
    
    // For Debug Stats (SDR Input)
    std::vector<float> sdr_y_values;
    if (config.debug_mode) {
        sdr_y_values.reserve(w * h);
    }
    
    float* p_plane_0 = input_tensor_values.data();
    float* p_plane_1 = input_tensor_values.data() + w * h;
    float* p_plane_2 = input_tensor_values.data() + 2 * w * h;
    
    for (int y = 0; y < h; ++y) {
        uint16_t* p_in = reinterpret_cast<uint16_t*>(in_data + y * stride);
        for (int x = 0; x < w; ++x) {
            // SDR Transfer Function -> Linear RGB
            float r = decode_transfer(p_in[0] / i_max, config.transfer_function);
            float g = decode_transfer(p_in[1] / i_max, config.transfer_function);
            float b = decode_transfer(p_in[2] / i_max, config.transfer_function);
            
            // RGB -> XYZ (Relative)
            float X = r * M_RGB_XYZ[0] + g * M_RGB_XYZ[1] + b * M_RGB_XYZ[2];
            float Y = r * M_RGB_XYZ[3] + g * M_RGB_XYZ[4] + b * M_RGB_XYZ[5];
            float Z = r * M_RGB_XYZ[6] + g * M_RGB_XYZ[7] + b * M_RGB_XYZ[8];

            // XYZ -> P3 (Model Expects P3)
            float P3_R = X * M_XYZ_P3[0] + Y * M_XYZ_P3[1] + Z * M_XYZ_P3[2];
            float P3_G = X * M_XYZ_P3[3] + Y * M_XYZ_P3[4] + Z * M_XYZ_P3[5];
            float P3_B = X * M_XYZ_P3[6] + Y * M_XYZ_P3[7] + Z * M_XYZ_P3[8];
            
            // P3 Relative -> P3 48 Nits (Model Input Scale)
            float P3_R_nit = P3_R * 48.0f;
            float P3_G_nit = P3_G * 48.0f;
            float P3_B_nit = P3_B * 48.0f;
            
            if (config.debug_mode) {
                // Calculate SDR Y (Nits) for stats
                float Y_sdr = P3_R_nit * M_P3_XYZ[3] + P3_G_nit * M_P3_XYZ[4] + P3_B_nit * M_P3_XYZ[5];
                sdr_y_values.push_back(Y_sdr);
            }
            
            // Write to NCHW Tensor
            int plane_idx = y * w + x;
            p_plane_0[plane_idx] = P3_R_nit;
            p_plane_1[plane_idx] = P3_G_nit;
            p_plane_2[plane_idx] = P3_B_nit;

            p_in += 3;
        }
    }

    // Identify SDR Stats (printed before inference for real-time feedback)
    static long long g_sdr_frame_counter = 0;
    g_sdr_frame_counter++;
    
    if (config.debug_mode && !sdr_y_values.empty()) {
        std::sort(sdr_y_values.begin(), sdr_y_values.end());
        float min_y = sdr_y_values.front();
        float max_y = sdr_y_values.back();
        float median = sdr_y_values[sdr_y_values.size() / 2];
        float p99 = sdr_y_values[(size_t)(sdr_y_values.size() * 0.99)];

        std::cout << "[Frame " << g_sdr_frame_counter << "] "
                  << "SDR(med=" << median << " p99=" << p99 << " max=" << max_y << " nits)"
                  << std::endl;
    }

    // === Run ONNX Inference ===
    std::vector<int64_t> input_shape = {1, 3, h, w};
    
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_shape.data(), input_shape.size());

    const char* input_names[] = {"input"};
    const char* output_names[] = {"gain_map", "hdr_pred"}; 
    
    auto output_tensors = g_ort_ctx->session->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 2);
    
    // Output 1 is hdr_pred (P3 Nits)
    float* floatarr = output_tensors[1].GetTensorMutableData<float>();
    
    // === Post-Process: P3 Nits -> XYZ Nits (Absolute, cd/m²) ===
    const float* out_p0 = floatarr;             // P3 R
    const float* out_p1 = floatarr + w * h;     // P3 G
    const float* out_p2 = floatarr + 2 * w * h; // P3 B

    // For debug stats
    static long long g_frame_counter = 0;
    g_frame_counter++;
    
    std::vector<float> Y_values;
    std::vector<float> sdr_y_values_all;  // For SDR stats
    std::vector<float> hue_values;        // For P95 calculation
    double sum_hue_shift = 0;
    long long hue_count = 0;
    
    // Chroma stats
    double sum_chroma_sdr = 0, sum_chroma_hdr = 0;
    long long chroma_count = 0;
    
    // Luminance distribution (Cinema Semantic Zones)
    // SDR: DeepDark(<0.5), Shadows(0.5-5), Midtones(5-20), Highlights(20-48)
    // HDR: DeepDark(<3), Shadows(3-30), Midtones(30-120), Highlights(120-300)
    long long sdr_deep = 0, sdr_shadow = 0, sdr_mid = 0, sdr_hi = 0;
    long long hdr_deep = 0, hdr_shadow = 0, hdr_mid = 0, hdr_hi = 0;
    
    // Bucket Stats for Color Volume
    struct BucketStat {
        double sum_sat_sdr = 0;
        double sum_sat_hdr = 0;
        double sum_hue = 0;
        std::vector<float> hue_list;
        long long n = 0;
    };
    BucketStat buckets[4]; 
    const char* bucket_names[4] = {"Deep", "Shadow", "Mid", "Hi"};
    
    // Clamp Stats
    long long clamp_neg_count = 0; // Output pixels < 0
    
    auto sat01 = [](float r, float g, float b) {
        float mx = std::max({r,g,b});
        float mn = std::min({r,g,b});
        float d  = mx - mn;
        const float eps = 1e-6f;
        return (mx > eps) ? (d / (mx + eps)) : 0.0f;
    };
    
    auto bucket_idx_hdr = [](float Y) {
        if (Y < 3.0f) return 0;       // Deep
        if (Y < 30.0f) return 1;      // Shadow
        if (Y < 120.0f) return 2;     // Mid
        return 3;                     // Hi
    };

    if (config.debug_mode) {
        Y_values.reserve(w * h);
        sdr_y_values_all.reserve(w * h);
        hue_values.reserve(w * h / 4);  // Estimate: ~25% pixels have significant chroma
        for(int i=0; i<4; i++) buckets[i].hue_list.reserve(w*h/10);
    }

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int plane_idx = y * w + x;
            float P3_R_nit = out_p0[plane_idx];
            float P3_G_nit = out_p1[plane_idx];
            float P3_B_nit = out_p2[plane_idx];
            
            // === Hue Lock (Gain Map Mode) ===
            // Forces HDR output to match SDR Hue/Saturation, using Model only for Luminance gain.
            // Solves hue shift and temporal instability.
            if (config.hue_lock) {
                float sdr_r = p_plane_0[plane_idx];
                float sdr_g = p_plane_1[plane_idx];
                float sdr_b = p_plane_2[plane_idx];
                
                // Calculate Luminance (P3 Y approx)
                // Weights from M_P3_XYZ (R->Y, G->Y, B->Y)
                // 0.22897457f, 0.69173852f, 0.07928691f
                const float Wy_R = 0.22897457f;
                const float Wy_G = 0.69173852f;
                const float Wy_B = 0.07928691f;
                
                float y_sdr = sdr_r * Wy_R + sdr_g * Wy_G + sdr_b * Wy_B;
                float y_hdr = P3_R_nit * Wy_R + P3_G_nit * Wy_G + P3_B_nit * Wy_B;
                
                const float eps = 1e-4f;
                // Only applying gain if SDR has signal to avoid amplifying noise
                if (y_sdr > eps) {
                    float gain = y_hdr / y_sdr;
                    P3_R_nit = sdr_r * gain;
                    P3_G_nit = sdr_g * gain;
                    P3_B_nit = sdr_b * gain;
                }
            }
            
            if (config.debug_mode) {
                 float sdr_r = p_plane_0[plane_idx];
                 float sdr_g = p_plane_1[plane_idx];
                 float sdr_b = p_plane_2[plane_idx];
                 
                 // Inline Hue Calculation (P3 RGB)
                 float mn_sdr = min({sdr_r, sdr_g, sdr_b});
                 float mx_sdr = max({sdr_r, sdr_g, sdr_b});
                 float d_sdr = mx_sdr - mn_sdr;
                 
                 float mn_hdr = min({P3_R_nit, P3_G_nit, P3_B_nit});
                 float mx_hdr = max({P3_R_nit, P3_G_nit, P3_B_nit});
                 float d_hdr = mx_hdr - mn_hdr;

                 if (d_sdr > 1.0f && d_hdr > 1.0f) { // Threshold: 1.0 nit to filter low-saturation noise
                     auto calc_h = [](float r, float g, float b, float mx, float d) {
                        float h = 0;
                        if (mx == r) h = (g - b) / d + (g < b ? 6 : 0);
                        else if (mx == g) h = (b - r) / d + 2;
                        else h = (r - g) / d + 4;
                        return h * 60.0f;
                     };
                     float h_sdr = calc_h(sdr_r, sdr_g, sdr_b, mx_sdr, d_sdr);
                     float h_hdr = calc_h(P3_R_nit, P3_G_nit, P3_B_nit, mx_hdr, d_hdr);
                     float diff = abs(h_sdr - h_hdr);
                     if (diff > 180.0f) diff = 360.0f - diff;
                     
                     sum_hue_shift += diff;
                     hue_values.push_back(diff);
                     hue_count++;
                     
                     // Chroma: use d (max-min) as proxy
                     sum_chroma_sdr += d_sdr;
                     sum_chroma_hdr += d_hdr;
                     chroma_count++;
                     
                     // Bucket Stats Collection (New)
                     // Re-calculate Y for bucket index (Correct P3 Weights)
                     const float Wy_R = 0.22897457f;
                     const float Wy_G = 0.69173852f;
                     const float Wy_B = 0.07928691f;
                     float Y_now = P3_R_nit * Wy_R + P3_G_nit * Wy_G + P3_B_nit * Wy_B; 
                     int bi = bucket_idx_hdr(Y_now);
                     
                     float sat_sdr = sat01(sdr_r, sdr_g, sdr_b);
                     float sat_hdr = sat01(P3_R_nit, P3_G_nit, P3_B_nit);
                     
                     // Using sat threshold to filter noise
                     if (sat_sdr > 0.02f || sat_hdr > 0.02f) {
                         buckets[bi].sum_sat_sdr += sat_sdr;
                         buckets[bi].sum_sat_hdr += sat_hdr;
                         buckets[bi].sum_hue += diff;
                         buckets[bi].hue_list.push_back(diff);
                         buckets[bi].n++;
                     }
                 }
            }
            
            // P3 Nits -> XYZ Nits (Absolute luminance in cd/m²)
            float X_nit = P3_R_nit * M_P3_XYZ[0] + P3_G_nit * M_P3_XYZ[1] + P3_B_nit * M_P3_XYZ[2];
            float Y_nit = P3_R_nit * M_P3_XYZ[3] + P3_G_nit * M_P3_XYZ[4] + P3_B_nit * M_P3_XYZ[5];
            float Z_nit = P3_R_nit * M_P3_XYZ[6] + P3_G_nit * M_P3_XYZ[7] + P3_B_nit * M_P3_XYZ[8];
            
            // Clamp negative values (XYZ should be non-negative for PQ)
            // NOTE: Model may produce slight negatives due to matrix math
            if (config.debug_mode) {
                if (X_nit < 0 || Y_nit < 0 || Z_nit < 0) clamp_neg_count++;
            }
            
            result.x[plane_idx] = max(0.0f, X_nit);
            result.y[plane_idx] = max(0.0f, Y_nit);
            result.z[plane_idx] = max(0.0f, Z_nit);
            
            if (config.debug_mode) {
                Y_values.push_back(max(0.0f, Y_nit));
                
                // SDR luminance bins (use sdr_y from input tensor if available)
                // Actually we need SDR Y - let's compute it here
                float sdr_r = p_plane_0[plane_idx];
                float sdr_g = p_plane_1[plane_idx];
                float sdr_b = p_plane_2[plane_idx];
                float sdr_y = sdr_r * M_P3_XYZ[3] + sdr_g * M_P3_XYZ[4] + sdr_b * M_P3_XYZ[5];
                
                // SDR bins (Cinema Semantic)
                if (sdr_y < 0.5f) sdr_deep++;
                else if (sdr_y < 5.0f) sdr_shadow++;
                else if (sdr_y < 20.0f) sdr_mid++;
                else sdr_hi++;
                
                // HDR bins (Cinema Semantic)
                if (Y_nit < 3.0f) hdr_deep++;
                else if (Y_nit < 30.0f) hdr_shadow++;
                else if (Y_nit < 120.0f) hdr_mid++;
                else hdr_hi++;
            }
        }
    }

    // Compute debug stats
    if (config.debug_mode && !Y_values.empty()) {
        std::sort(Y_values.begin(), Y_values.end());
        result.Y_min = Y_values.front();
        result.Y_max = Y_values.back();
        result.Y_median = Y_values[Y_values.size() / 2];
        result.Y_p99 = Y_values[(size_t)(Y_values.size() * 0.99)];
        
        // Compute SDR stats from sdr_y_values (we need to collect them - simplified here using bins)
        // For SDR, we'll just use the already-collected sdr_y_values from Input SDR Stats (printed earlier)
        
        // Hue P95
        float hue_p95 = 0;
        float hue_mean = 0;
        if (!hue_values.empty()) {
            std::sort(hue_values.begin(), hue_values.end());
            hue_p95 = hue_values[(size_t)(hue_values.size() * 0.95)];
            hue_mean = sum_hue_shift / hue_count;
        }
        
        // Chroma ratio
        double chroma_ratio = 1.0;
        if (chroma_count > 0 && sum_chroma_sdr > 0.01) {
            chroma_ratio = sum_chroma_hdr / sum_chroma_sdr;
        }
        
        // Luminance distribution percentages
        long long total = w * h;
        auto pct = [total](long long n) { return 100.0 * n / total; };
        
        // Line 1: Luminance stats
        std::cout << "[Frame " << g_frame_counter << "] "
                  << "HDR(med=" << result.Y_median 
                  << " p99=" << result.Y_p99 
                  << " max=" << result.Y_max << " nits)"
                  << std::endl;
        
        // Line 2: Color quality stats
        std::cout << "[Frame " << g_frame_counter << "] "
                  << "Hue(mean=" << hue_mean << "° p95=" << hue_p95 << "°) "
                  << "Chroma(×" << chroma_ratio << ") "
                  << "ValidPx=" << hue_count
                  << std::endl;
        
        // Line 3: Distribution
        std::cout << "[Frame " << g_frame_counter << "] "
                  << "SDR(DD=" << (int)pct(sdr_deep) << "% Sh=" << (int)pct(sdr_shadow) 
                  << "% Mid=" << (int)pct(sdr_mid) << "% Hi=" << (int)pct(sdr_hi) << "%) "
                  << "HDR(DD=" << (int)pct(hdr_deep) << "% Sh=" << (int)pct(hdr_shadow) 
                  << "% Mid=" << (int)pct(hdr_mid) << "% Hi=" << (int)pct(hdr_hi) << "%) "
                  << "Negs=" << clamp_neg_count
                  << std::endl;
                  
        // Line 4: Vol Stats
        auto p95 = [](std::vector<float>& v)->float{
            if (v.empty()) return 0.0f;
            std::sort(v.begin(), v.end());
            return v[(size_t)(v.size() * 0.95)];
        };

        std::cout << "[Frame " << g_frame_counter << "] Vol: ";
        for (int i=0;i<4;i++) {
            if (buckets[i].n == 0) {
                // std::cout << bucket_names[i] << "(n=0) ";
                continue;
            }
            double as_sdr = buckets[i].sum_sat_sdr / buckets[i].n;
            double as_hdr = buckets[i].sum_sat_hdr / buckets[i].n;
            double ratio  = (as_sdr > 1e-6) ? (as_hdr / as_sdr) : 0.0;
            double ah     = buckets[i].sum_hue / buckets[i].n;
            float  h95    = p95(buckets[i].hue_list);
            
            std::cout << bucket_names[i]
                      << "(n=" << buckets[i].n
                      << " SatHDR=" << as_hdr
                      << " Ratio=" << ratio
                      << " Hue=" << ah << "°"
                      << " P95=" << h95 << "°) ";
        }
        std::cout << std::endl;
    }

    return result;
}
