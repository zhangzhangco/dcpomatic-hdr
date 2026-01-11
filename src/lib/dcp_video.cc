/*
    Copyright (C) 2012-2021 Carl Hetherington <cth@carlh.net>

    This file is part of DCP-o-matic.

    DCP-o-matic is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    DCP-o-matic is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with DCP-o-matic.  If not, see <http://www.gnu.org/licenses/>.

*/


/** @file  src/dcp_video_frame.cc
 *  @brief A single frame of video destined for a DCP.
 *
 *  Given an Image and some settings, this class knows how to encode
 *  the image to J2K either on the local host or on a remote server.
 *
 *  Objects of this class are used for the queue that we keep
 *  of images that require encoding.
 */


#include "config.h"
#include "cross.h"
#include "dcp_video.h"
#include "dcpomatic_log.h"
#include "dcpomatic_socket.h"
#include "encode_server_description.h"
#include "exceptions.h"
#include "image.h"
#include "log.h"
#include "player_video.h"
#include "rng.h"
#include "util.h"
#include <libcxml/cxml.h>
#include <dcp/openjpeg_image.h>
#include <dcp/rgb_xyz.h>
#include <dcp/j2k_transcode.h>
#include <dcp/warnings.h>
LIBDCP_DISABLE_WARNINGS
#include <libxml++/libxml++.h>
extern "C" {
#include <libavutil/pixdesc.h>
}
LIBDCP_ENABLE_WARNINGS
#include <fmt/format.h>
#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <stdint.h>
#include <iomanip>
#include <iostream>

#include "i18n.h"


using std::cout;
using std::make_shared;
using std::shared_ptr;
using std::string;
using dcp::ArrayData;
#if BOOST_VERSION >= 106100
using namespace boost::placeholders;
#endif


/** Construct a DCP video frame.
 *  @param frame Input frame.
 *  @param index Index of the frame within the DCP.
 *  @param bit_rate Video bit rate to use.
 */
DCPVideo::DCPVideo(
	shared_ptr<const PlayerVideo> frame, int index, int dcp_fps, int64_t bit_rate, Resolution r
	)
	: _frame(frame)
	, _index(index)
	, _frames_per_second(dcp_fps)
	, _video_bit_rate(bit_rate)
	, _resolution(r)
{

}

DCPVideo::DCPVideo(shared_ptr<const PlayerVideo> frame, shared_ptr<const cxml::Node> node)
	: _frame(frame)
{
	_index = node->number_child<int>("Index");
	_frames_per_second = node->number_child<int>("FramesPerSecond");
	_video_bit_rate = node->number_child<int64_t>("VideoBitRate");
	_resolution = Resolution(node->optional_number_child<int>("Resolution").get_value_or(static_cast<int>(Resolution::TWO_K)));
}

#include "zhangxin_hdr.h"
#include <dcp/gamma_transfer_function.h>
#include <dcp/chromaticity.h>
#include <algorithm>

// === ST 2084 PQ OETF (Encoding) ===
// Input: Linear luminance in cd/m² (nits)
// Output: PQ code value 0-4095 (12-bit full range)
// NOTE: This HDR pipeline is EXPERIMENTAL until MXF TransferCharacteristic UL is set (DCI HDR Addendum).
static inline int pq_encode_12bit(float L_nits) {
    // ST 2084 Constants
    const float c1 = 0.8359375f;        // 3424/4096
    const float c2 = 18.8515625f;       // 2413/128
    const float c3 = 18.6875f;          // 2392/128
    const float m1 = 0.1593017578125f;  // 1305/8192
    const float m2 = 78.84375f;         // 2523/32
    const float PQ_REF_NITS = 10000.0f; // PQ reference (hardcoded, not content peak!)
    
    // Clamp to non-negative and normalize
    float Y = std::max(0.0f, L_nits) / PQ_REF_NITS;
    
    // Apply PQ OETF
    float Ym = std::pow(Y, m1);
    float num = c1 + c2 * Ym;
    float den = 1.0f + c3 * Ym;
    float V = std::pow(num / den, m2);  // PQ non-linear value [0,1]
    
    // Quantize to 12-bit full range
    return std::clamp(static_cast<int>(V * 4095.0f + 0.5f), 0, 4095);
}

shared_ptr<dcp::OpenJPEGImage>
DCPVideo::convert_to_xyz(shared_ptr<const PlayerVideo> frame)
{
	shared_ptr<dcp::OpenJPEGImage> xyz;

	auto conversion = [](AVPixelFormat fmt) {
		return fmt == AV_PIX_FMT_XYZ12LE ? AV_PIX_FMT_XYZ12LE : AV_PIX_FMT_RGB48LE;
	};

	auto image = frame->image(conversion, VideoRange::FULL, false);

    // [ZHANGXIN] HDR Processor Insertion Point
    // NOTE: This HDR pipeline is EXPERIMENTAL until MXF TransferCharacteristic UL is set (DCI HDR Addendum).
    ZhangxinHDR::Config hdr_config;
    
    if (getenv("ZHANGXIN_HDR_ENABLE")) {
        hdr_config.enable = true;
        hdr_config.debug_mode = (getenv("ZHANGXIN_HDR_DEBUG") != nullptr);
        hdr_config.dump_debug_frames = (getenv("ZHANGXIN_HDR_DUMP") != nullptr);
        
        const char* gamma_env = getenv("ZHANGXIN_HDR_GAMMA");
        if (gamma_env) {
            std::string g(gamma_env);
            if (g == "2.6") hdr_config.transfer_function = ZhangxinHDR::TransferFunction::GAMMA_26;
            else if (g == "2.4") hdr_config.transfer_function = ZhangxinHDR::TransferFunction::GAMMA_24;
            else if (g == "REC709") hdr_config.transfer_function = ZhangxinHDR::TransferFunction::REC709_SCENE_LINEAR;
            // Default GAMMA_24 from Config constructor
        }
    }

    if (hdr_config.enable) {
        // === HDR Path: Direct PQ Encoding ===
        // 1. Run HDR model: outputs linear XYZ in cd/m² (absolute luminance)
        auto hdr_xyz = ZhangxinHDR::process_to_hdr_xyz(image, hdr_config);
        
        if (hdr_xyz.width == 0 || hdr_xyz.height == 0) {
            // Model failed, fallback to original SDR path
            LOG_WARNING_NC("ZhangxinHDR: Model processing failed, falling back to SDR path");
            goto sdr_path;
        }
        
        // 2. Create OpenJPEGImage and directly write PQ-encoded values
        dcp::Size size(hdr_xyz.width, hdr_xyz.height);
        xyz = make_shared<dcp::OpenJPEGImage>(size);
        
        int* x_data = xyz->data(0);  // X component
        int* y_data = xyz->data(1);  // Y component
        int* z_data = xyz->data(2);  // Z component
        
        // 3. PQ Encode all XYZ components (not just Y!)
        int pixel_count = hdr_xyz.width * hdr_xyz.height;
        for (int i = 0; i < pixel_count; ++i) {
            // PQ encode each component: Linear XYZ (cd/m²) -> 12-bit X″Y″Z″
            int pq_x = pq_encode_12bit(hdr_xyz.x[i]);
            int pq_y = pq_encode_12bit(hdr_xyz.y[i]);
            int pq_z = pq_encode_12bit(hdr_xyz.z[i]);
            
            x_data[i] = pq_x;
            y_data[i] = pq_y;
            z_data[i] = pq_z;
        }
        
        return xyz;
    }

sdr_path:
    // === SDR Path: Original Implementation ===
    if (frame->colour_conversion()) {
        xyz = dcp::rgb_to_xyz(
            image->data()[0],
            image->size(),
            image->stride()[0],
            frame->colour_conversion().get()
            );
    } else {
        xyz = make_shared<dcp::OpenJPEGImage>(image->data()[0], image->size(), image->stride()[0]);
    }

	return xyz;
}

dcp::Size
DCPVideo::get_size() const
{
	auto image = _frame->image(force(AV_PIX_FMT_RGB48LE), VideoRange::FULL, false);
	return image->size();
}


void
DCPVideo::convert_to_xyz(uint16_t* dst) const
{
	DCPOMATIC_ASSERT(_frame->colour_conversion());

	auto image = _frame->image(force(AV_PIX_FMT_RGB48LE), VideoRange::FULL, false);
	dcp::rgb_to_xyz(
		image->data()[0],
		dst,
		image->size(),
		image->stride()[0],
		_frame->colour_conversion().get()
		);
}


/** J2K-encode this frame on the local host.
 *  @return Encoded data.
 */
ArrayData
DCPVideo::encode_locally() const
{
	auto const comment = Config::instance()->dcp_j2k_comment();

	ArrayData enc = {};
	/* This was empirically derived by a user: see #1902 */
	int const minimum_size = 16384;
	LOG_DEBUG_ENCODE("Using minimum frame size {}", minimum_size);

	auto xyz = convert_to_xyz(_frame);
	int noise_amount = 2;
	int pixel_skip = 16;
	while (true) {
		enc = dcp::compress_j2k(
			xyz,
			_video_bit_rate,
			_frames_per_second,
			_frame->eyes() == Eyes::LEFT || _frame->eyes() == Eyes::RIGHT,
			_resolution == Resolution::FOUR_K,
			comment.empty() ? "libdcp" : comment
		);

		if (enc.size() >= minimum_size) {
			LOG_DEBUG_ENCODE(N_("Frame {} encoded size was OK ({})"), _index, enc.size());
			break;
		}

		LOG_GENERAL(N_("Frame {} encoded size was small ({}); adding noise at level {} with pixel skip {}"), _index, enc.size(), noise_amount, pixel_skip);

		/* The JPEG2000 is too low-bitrate for some decoders <cough>DSS200</cough> so add some noise
		 * and try again.  This is slow but hopefully won't happen too often.  We have to do
		 * convert_to_xyz() again because compress_j2k() corrupts its xyz parameter.
		 */

		xyz = convert_to_xyz(_frame);
		auto size = xyz->size();
		auto pixels = size.width * size.height;
		dcpomatic::RNG rng(42);
		for (auto c = 0; c < 3; ++c) {
			auto p = xyz->data(c);
			auto e = xyz->data(c) + pixels;
			while (p < e) {
				*p = std::min(4095, std::max(0, *p + (rng.get() % noise_amount)));
				p += pixel_skip;
			}
		}

		if (pixel_skip > 1) {
			--pixel_skip;
		} else {
			++noise_amount;
		}
		/* Something's gone badly wrong if this much noise doesn't help */
		DCPOMATIC_ASSERT(noise_amount < 16);
	}

	switch (_frame->eyes()) {
	case Eyes::BOTH:
		LOG_DEBUG_ENCODE(N_("Finished locally-encoded frame {} for mono"), _index);
		break;
	case Eyes::LEFT:
		LOG_DEBUG_ENCODE(N_("Finished locally-encoded frame {} for L"), _index);
		break;
	case Eyes::RIGHT:
		LOG_DEBUG_ENCODE(N_("Finished locally-encoded frame {} for R"), _index);
		break;
	default:
		break;
	}

	return enc;
}

/** Send this frame to a remote server for J2K encoding, then read the result.
 *  @param serv Server to send to.
 *  @param timeout timeout in seconds.
 *  @return Encoded data.
 */
ArrayData
DCPVideo::encode_remotely(EncodeServerDescription serv, int timeout) const
{
	auto socket = make_shared<Socket>(timeout);
	socket->set_send_buffer_size(512 * 1024);

	socket->connect(serv.host_name(), ENCODE_FRAME_PORT);

	/* Collect all XML metadata */
	xmlpp::Document doc;
	auto root = doc.create_root_node("EncodingRequest");
	cxml::add_text_child(root, "Version", fmt::to_string(SERVER_LINK_VERSION));
	add_metadata(root);

	LOG_DEBUG_ENCODE(N_("Sending frame {} to remote"), _index);

	{
		Socket::WriteDigestScope ds(socket);

		/* Send XML metadata */
		auto xml = doc.write_to_string("UTF-8");
		socket->write(xml.bytes() + 1);
		socket->write((uint8_t *) xml.c_str(), xml.bytes() + 1);

		/* Send binary data */
		LOG_TIMING("start-remote-send thread={}", thread_id());
		_frame->write_to_socket(socket);
	}

	/* Read the response (JPEG2000-encoded data); this blocks until the data
	   is ready and sent back.
	*/
	Socket::ReadDigestScope ds(socket);
	LOG_TIMING("start-remote-encode thread={}", thread_id());
	ArrayData e(socket->read_uint32());
	LOG_TIMING("start-remote-receive thread={}", thread_id());
	socket->read(e.data(), e.size());
	LOG_TIMING("finish-remote-receive thread={}", thread_id());
	if (!ds.check()) {
		throw NetworkError("Checksums do not match");
	}

	LOG_DEBUG_ENCODE(N_("Finished remotely-encoded frame {}"), _index);

	return e;
}

void
DCPVideo::add_metadata(xmlpp::Element* el) const
{
	cxml::add_text_child(el, "Index", fmt::to_string(_index));
	cxml::add_text_child(el, "FramesPerSecond", fmt::to_string(_frames_per_second));
	cxml::add_text_child(el, "VideoBitRate", fmt::to_string(_video_bit_rate));
	cxml::add_text_child(el, "Resolution", fmt::to_string(int(_resolution)));
	_frame->add_metadata(el);
}

Eyes
DCPVideo::eyes() const
{
	return _frame->eyes();
}

/** @return true if this DCPVideo is definitely the same as another;
 *  (apart from the frame index), false if it is probably not.
 */
bool
DCPVideo::same(shared_ptr<const DCPVideo> other) const
{
	if (_frames_per_second != other->_frames_per_second ||
	    _video_bit_rate != other->_video_bit_rate ||
	    _resolution != other->_resolution) {
		return false;
	}

	return _frame->same(other->_frame);
}
