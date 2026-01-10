/*
    Copyright (C) 2026 zhangxin
*/

#ifndef DCPOMATIC_ZHANGXIN_HDR_H
#define DCPOMATIC_ZHANGXIN_HDR_H

#include "image.h"
#include <memory>

class ZhangxinHDR
{
public:
    struct Config {
        // SDR 物理定义
        double sdr_white_nits = 48.0;
        double sdr_gamma = 2.4;

        // HDR 目标定义
        double hdr_black_nits = 0.024; // 黑位注入
        double hdr_white_nits = 300.0; // 峰值限制
        
        // 模式开关
        bool enable = false;
        bool debug_mode = false; 
        bool dump_debug_frames = false; // Dump input/output frames for analysis
        std::string dump_path = "/tmp";
    };

    struct Stats {
        float min_r=1.0f, max_r=0.0f, mean_r=0.0f;
        float min_g=1.0f, max_g=0.0f, mean_g=0.0f;
        float min_b=1.0f, max_b=0.0f, mean_b=0.0f;
        float clip_hi_rate = 0.0f;
        float clip_lo_rate = 0.0f; // 低于 SDR 黑位 (0) 的比例
        float black_inject_miss_rate = 0.0f; // 处理后低于目标黑位的比例
    };

    static std::shared_ptr<Image> process(std::shared_ptr<const Image> image, Config config);
    
    // Log helper
    static void log_stats(const std::string& tag, const Stats& s);

private:
    // 内部实现：纯像素处理
    static void process_pixel(float& r, float& g, float& b, const Config& c);
};

#endif
