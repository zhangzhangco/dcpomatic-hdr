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

using std::shared_ptr;
using std::make_shared;
using std::min;
using std::max;
using std::string;
using std::vector;

// Standard sRGB/Rec709 (D65) to XYZ Matrix
// R     G     B
// 0.4124564  0.3575761  0.1804375
// 0.2126729  0.7151522  0.0721750
// 0.0193339  0.1191920  0.9503041
static const float M_RGB_XYZ[9] = {
    0.4124564f, 0.3575761f, 0.1804375f,
    0.2126729f, 0.7151522f, 0.0721750f,
    0.0193339f, 0.1191920f, 0.9503041f
};

void
ZhangxinHDR::log_stats(const string& tag, const Stats& s)
{
    // ... Simplified log for now
    std::cout << "[ZHANGXIN_HDR] " << tag << " Frame Stats:" << std::endl;
}

// Minimal NPY Writer (Version 1.0)
// Format Spec: https://numpy.org/neps/nep-0001-npy-format.html
static void save_npy_float(const std::vector<float>& data, int h, int w, int c, string path) {
    std::ofstream fs(path, std::ios::binary);
    if (!fs) return;

    const char magic[] = "\x93NUMPY";
    const uint8_t major = 1;
    const uint8_t minor = 0;
    fs.write(magic, 6);
    fs.write(reinterpret_cast<const char*>(&major), 1);
    fs.write(reinterpret_cast<const char*>(&minor), 1);

    // Create header dict string
    // {'descr': '<f4', 'fortran_order': False, 'shape': (h, w, c), }
    std::string header_dict = "{'descr': '<f4', 'fortran_order': False, 'shape': ("
                            + std::to_string(h) + ", " + std::to_string(w) + ", " + std::to_string(c) + "), }";
    
    // Pad header to 16-byte alignment (accounting for magic+ver+len = 10 bytes)
    int current_len = 10 + header_dict.size() + 1; // +1 for newline
    int padding = 16 - (current_len % 16);
    if (padding == 16) padding = 0;
    
    // Header length (unsigned short)
    uint16_t header_len = header_dict.size() + 1 + padding;
    fs.write(reinterpret_cast<const char*>(&header_len), 2);
    
    fs.write(header_dict.c_str(), header_dict.size());
    for(int i=0; i<padding; ++i) fs.put(' ');
    fs.put('\n');

    // Data
    fs.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    fs.close();
    std::cout << "[ZHANGXIN_HDR] Dumped NPY to " << path << std::endl;
}

static bool load_npy_float(string path, std::vector<float>& data, int& h, int& w, int& c) {
    std::ifstream fs(path, std::ios::binary);
    if (!fs) return false;

    // Skip magic (6) + ver (2)
    fs.seekg(8, std::ios::cur);
    uint16_t header_len;
    fs.read(reinterpret_cast<char*>(&header_len), 2);
    
    // Skip header
    fs.seekg(header_len, std::ios::cur);
    
    // We assume size matches (risky but okay for internal pipe)
    // To be safe, we should parse header, but let's trust the consistent pipeline for now.
    
    // Read Data
    size_t sz = data.size();
    if (sz == 0) return false; // Safety
    
    fs.read(reinterpret_cast<char*>(data.data()), sz * sizeof(float));
    return fs.good();
}

// Inverse Matrix (XYZ -> Rec.709 RGB D65)
// Calculated as inv(M_RGB_XYZ)
static const float M_XYZ_RGB[9] = {
     3.2404542f, -1.5371385f, -0.4985314f,
    -0.9692660f,  1.8760108f,  0.0415560f,
     0.0556434f, -0.2040259f,  1.0572252f
};

shared_ptr<Image>
ZhangxinHDR::process(shared_ptr<const Image> image, Config config)
{
    if (!config.enable) {
        return make_shared<Image>(*image);
    }
    
    DCPOMATIC_ASSERT(image->pixel_format() == AV_PIX_FMT_RGB48LE);
    dcp::Size size = image->size();
    shared_ptr<Image> out = make_shared<Image>(image->pixel_format(), size, image->alignment());
    // No copy needed, we fully rewrite out

    uint8_t*in_data = image->data()[0];
    uint8_t* out_data = out->data()[0];
    int stride = image->stride()[0];
    const float i_max = 65535.0f;

    std::vector<float> tensor_xyz;
    tensor_xyz.resize(size.width * size.height * 3);

    // 1. SDR -> Linear XYZ Tensor
    for (int y = 0; y < size.height; ++y) {
        uint16_t* p_in = reinterpret_cast<uint16_t*>(in_data + y * stride);
        for (int x = 0; x < size.width; ++x) {
            float r = pow(p_in[0] / i_max, config.sdr_gamma);
            float g = pow(p_in[1] / i_max, config.sdr_gamma);
            float b = pow(p_in[2] / i_max, config.sdr_gamma);
            
            float X = r * M_RGB_XYZ[0] + g * M_RGB_XYZ[1] + b * M_RGB_XYZ[2];
            float Y = r * M_RGB_XYZ[3] + g * M_RGB_XYZ[4] + b * M_RGB_XYZ[5];
            float Z = r * M_RGB_XYZ[6] + g * M_RGB_XYZ[7] + b * M_RGB_XYZ[8];

            size_t idx = (y * size.width + x) * 3;
            tensor_xyz[idx+0] = X;
            tensor_xyz[idx+1] = Y;
            tensor_xyz[idx+2] = Z;

            p_in += 3;
        }
    }

    // 2. Dump Input
    static int frame_idx = 0;
    string path_in, path_out;
    if (config.dump_debug_frames) {
        char filename[64];
        snprintf(filename, sizeof(filename), "/zx_in_%06d.npy", frame_idx);
        path_in = config.dump_path + filename;
        save_npy_float(tensor_xyz, size.height, size.width, 3, path_in);
        
        // Construct expected output path
        snprintf(filename, sizeof(filename), "/zx_out_%06d.npy", frame_idx);
        path_out = config.dump_path + filename;
    }

    // 3. Try Load Inference Output (Integration Point)
    bool loaded = false;
    if (config.dump_debug_frames) {
        int h=size.height, w=size.width, c=3; // Dummy, assume match
        // Reuse tensor_xyz buffer to save memory (Overwrite Input with Output)
        // BUT infer_out is Absolute Nits, logic below needs adaptation.
        loaded = load_npy_float(path_out, tensor_xyz, h, w, c);
        if (loaded) {
            std::cout << "[ZHANGXIN_HDR] Loaded inference output " << path_out << std::endl;
        }
    }

    // 4. XYZ -> Linear RGB -> Output
    for (int y = 0; y < size.height; ++y) {
        uint16_t* p_out = reinterpret_cast<uint16_t*>(out_data + y * stride);
        for (int x = 0; x < size.width; ++x) {
            size_t idx = (y * size.width + x) * 3;
            float X = tensor_xyz[idx+0];
            float Y = tensor_xyz[idx+1];
            float Z = tensor_xyz[idx+2];
            
            if (loaded) {
                // IMPORTANT: Model Output is Absolute Nits (e.g. 0.005 to 300.0)
                // DCP Standard: 1.0 = 48 nits (Usually)
                // We need to Normalize based on Target Peak or Containment.
                
                // OPTION A: Normalize 48 nits = 1.0 (Standard DCP)
                // Values > 48 nits (1.0) will be clipped by clamp(1.0) below unless we tone map.
                // This validates "Texture" and "Black" but clips highlights.
                float scale = 1.0f / 48.0f; 
                
                // OR OPTION B: Normalize 300 nits = 1.0 (For HDR verification without clip)
                // Then image will look dark on SDR player but full range is preserved.
                // float scale = 1.0f / 300.0f;
                
                X *= scale; 
                Y *= scale; 
                Z *= scale;
            }

            // XYZ -> Linear RGB
            float r = X * M_XYZ_RGB[0] + Y * M_XYZ_RGB[1] + Z * M_XYZ_RGB[2];
            float g = X * M_XYZ_RGB[3] + Y * M_XYZ_RGB[4] + Z * M_XYZ_RGB[5];
            float b = X * M_XYZ_RGB[6] + Y * M_XYZ_RGB[7] + Z * M_XYZ_RGB[8];
            
            // Output
            p_out[0] = static_cast<uint16_t>(max(0.0f, min(1.0f, r)) * i_max);
            p_out[1] = static_cast<uint16_t>(max(0.0f, min(1.0f, g)) * i_max);
            p_out[2] = static_cast<uint16_t>(max(0.0f, min(1.0f, b)) * i_max);

            p_out += 3;
        }
    }

    if (config.dump_debug_frames) {
        frame_idx++;
    }

    return out;
}
