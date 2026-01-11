/*
    Copyright (C) 2026 zhangxin
*/

#include "zhangxin_hdr.h"
#include "dcpomatic_assert.h"
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

static void init_ort_session() {
    std::lock_guard<std::mutex> lock(g_ort_mutex);
    if (g_ort_ctx && g_ort_ctx->session) return; // Already initialized

    const char* model_path = getenv("ZHANGXIN_HDR_MODEL");
    if (!model_path) {
        std::cerr << "[ZHANGXIN_HDR] Error: ZHANGXIN_HDR_MODEL environment variable not set!" << std::endl;
        return;
    }

    try {
        if (!g_ort_ctx) g_ort_ctx.reset(new OrtContext());
        
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(4); 
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        std::cout << "[ZHANGXIN_HDR] Loading ONNX model from " << model_path << " ..." << std::endl;
        g_ort_ctx->session.reset(new Ort::Session(g_ort_ctx->env, model_path, session_options));
        std::cout << "[ZHANGXIN_HDR] Model loaded successfully." << std::endl;
    } catch (const Ort::Exception& e) {
        std::cerr << "[ZHANGXIN_HDR] ORT Exception: " << e.what() << std::endl;
        g_ort_ctx.reset(); 
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

shared_ptr<Image>
ZhangxinHDR::process(shared_ptr<const Image> image, Config config)
{
    if (!config.enable) {
        return make_shared<Image>(*image);
    }
    
    // Init ORT if needed
    if (!g_ort_ctx || !g_ort_ctx->session) {
        init_ort_session();
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
            float r = pow(p_in[0] / i_max, config.sdr_gamma);
            float g = pow(p_in[1] / i_max, config.sdr_gamma);
            float b = pow(p_in[2] / i_max, config.sdr_gamma);
            
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
        init_ort_session();
    }
    
    // If Init failed, throw error. We do NOT use fallback in production or strict testing.
    if (!g_ort_ctx || !g_ort_ctx->session) {
        throw std::runtime_error("[ZHANGXIN_HDR] Critical Error: HDR Model not loaded! Set ZHANGXIN_HDR_MODEL env var.");
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
    
    float* p_plane_0 = input_tensor_values.data();
    float* p_plane_1 = input_tensor_values.data() + w * h;
    float* p_plane_2 = input_tensor_values.data() + 2 * w * h;
    
    for (int y = 0; y < h; ++y) {
        uint16_t* p_in = reinterpret_cast<uint16_t*>(in_data + y * stride);
        for (int x = 0; x < w; ++x) {
            // SDR Gamma 2.4 -> Linear RGB
            float r = pow(p_in[0] / i_max, config.sdr_gamma);
            float g = pow(p_in[1] / i_max, config.sdr_gamma);
            float b = pow(p_in[2] / i_max, config.sdr_gamma);
            
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
            
            // Write to NCHW Tensor
            int plane_idx = y * w + x;
            p_plane_0[plane_idx] = P3_R_nit;
            p_plane_1[plane_idx] = P3_G_nit;
            p_plane_2[plane_idx] = P3_B_nit;

            p_in += 3;
        }
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
    std::vector<float> Y_values;
    if (config.debug_mode) {
        Y_values.reserve(w * h);
    }

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int plane_idx = y * w + x;
            float P3_R_nit = out_p0[plane_idx];
            float P3_G_nit = out_p1[plane_idx];
            float P3_B_nit = out_p2[plane_idx];
            
            // P3 Nits -> XYZ Nits (Absolute luminance in cd/m²)
            float X_nit = P3_R_nit * M_P3_XYZ[0] + P3_G_nit * M_P3_XYZ[1] + P3_B_nit * M_P3_XYZ[2];
            float Y_nit = P3_R_nit * M_P3_XYZ[3] + P3_G_nit * M_P3_XYZ[4] + P3_B_nit * M_P3_XYZ[5];
            float Z_nit = P3_R_nit * M_P3_XYZ[6] + P3_G_nit * M_P3_XYZ[7] + P3_B_nit * M_P3_XYZ[8];
            
            // Clamp negative values (XYZ should be non-negative for PQ)
            // NOTE: Model may produce slight negatives due to matrix math
            result.x[plane_idx] = max(0.0f, X_nit);
            result.y[plane_idx] = max(0.0f, Y_nit);
            result.z[plane_idx] = max(0.0f, Z_nit);
            
            if (config.debug_mode) {
                Y_values.push_back(max(0.0f, Y_nit));
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
        
        std::cout << "[ZHANGXIN_HDR] HDR XYZ Stats (Nits): "
                  << "Y_min=" << result.Y_min 
                  << " Y_median=" << result.Y_median
                  << " Y_p99=" << result.Y_p99
                  << " Y_max=" << result.Y_max << std::endl;
    }

    return result;
}
