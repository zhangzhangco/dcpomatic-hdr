/*
    Copyright (C) 2013-2021 Carl Hetherington <cth@carlh.net>

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


#include "content.h"
#include "film.h"
#include "image.h"
#include "image_proxy.h"
#include "j2k_image_proxy.h"
#include "player.h"
#include "player_video.h"
#include "video_content.h"
extern "C" {
#include <libavutil/pixfmt.h>
}
#include <libxml++/libxml++.h>
#include <fmt/format.h>
#include <iostream>


using std::cout;
using std::dynamic_pointer_cast;
using std::function;
using std::make_shared;
using std::shared_ptr;
using std::string;
using std::weak_ptr;
using boost::optional;
using dcp::Data;
using namespace dcpomatic;


static std::vector<int> _pq_to_gamma26_lut;
static boost::mutex _pq_to_gamma26_lut_mutex;

static void ensure_pq_to_gamma26_lut()
{
	boost::mutex::scoped_lock lm(_pq_to_gamma26_lut_mutex);
	if (!_pq_to_gamma26_lut.empty()) {
		return;
	}

	_pq_to_gamma26_lut.resize(4096);

	/* ST 2084 Constants */
	float const c1 = 0.8359375f;
	float const c2 = 18.8515625f;
	float const c3 = 18.6875f;
	float const m1 = 0.1593017578125f;
	float const m2 = 78.84375f;

	for (int i = 0; i < 4096; ++i) {
		float const N = i / 4095.0f;
		float const N_m2 = std::pow(N, 1.0f / m2);
		float const num = std::max(N_m2 - c1, 0.0f);
		float const den = c2 - c3 * N_m2;
		float const L_nits = std::pow(num / den, 1.0f / m1) * 10000.0f;

		/* Tone map: simple scaling to 300 nits peak */
		float const L_sdr = L_nits / 300.0f;

		/* Gamma 2.6 encode */
		float const v = std::pow(L_sdr, 1.0f / 2.6f);
		_pq_to_gamma26_lut[i] = std::clamp(static_cast<int>(v * 4095.0f + 0.5f), 0, 4095);
	}
}


PlayerVideo::PlayerVideo(
	shared_ptr<const ImageProxy> in,
	Crop crop,
	boost::optional<double> fade,
	dcp::Size inter_size,
	dcp::Size out_size,
	Eyes eyes,
	Part part,
	optional<ColourConversion> colour_conversion,
	VideoRange video_range,
	weak_ptr<Content> content,
	optional<ContentTime> video_time,
	bool error
	)
	: _in(in)
	, _crop(crop)
	, _fade(fade)
	, _inter_size(inter_size)
	, _out_size(out_size)
	, _eyes(eyes)
	, _part(part)
	, _colour_conversion(colour_conversion)
	, _video_range(video_range)
	, _content(content)
	, _video_time(video_time)
	, _error(error)
{

}


PlayerVideo::PlayerVideo(shared_ptr<cxml::Node> node, shared_ptr<Socket> socket)
{
	_crop = Crop(node);
	_fade = node->optional_number_child<double>("Fade");

	_inter_size = dcp::Size(node->number_child<int>("InterWidth"), node->number_child<int>("InterHeight"));
	_out_size = dcp::Size(node->number_child<int>("OutWidth"), node->number_child<int>("OutHeight"));
	_eyes = static_cast<Eyes>(node->number_child<int>("Eyes"));
	_part = static_cast<Part>(node->number_child<int>("Part"));
	_video_range = static_cast<VideoRange>(node->number_child<int>("VideoRange"));
	_error = node->optional_bool_child("Error").get_value_or(false);

	/* Assume that the ColourConversion uses the current state version */
	_colour_conversion = ColourConversion::from_xml(node, Film::current_state_version);

	_in = image_proxy_factory(node->node_child("In"), socket);

	if (node->optional_number_child<int>("SubtitleX")) {

		auto image = make_shared<Image>(
			AV_PIX_FMT_BGRA, dcp::Size(node->number_child<int>("SubtitleWidth"), node->number_child<int>("SubtitleHeight")), Image::Alignment::PADDED
			);

		image->read_from_socket(socket);

		_text = PositionImage(image, Position<int>(node->number_child<int>("SubtitleX"), node->number_child<int>("SubtitleY")));
	}
}


void
PlayerVideo::set_text(PositionImage image)
{
	_text = image;
}


shared_ptr<Image>
PlayerVideo::image(function<AVPixelFormat (AVPixelFormat)> pixel_format, VideoRange video_range, bool fast) const
{
	/* XXX: this assumes that image() and prepare() are only ever called with the same parameters (except crop, inter size, out size, fade) */

	boost::mutex::scoped_lock lm(_mutex);
	if (!_image || _crop != _image_crop || _inter_size != _image_inter_size || _out_size != _image_out_size || _fade != _image_fade) {
		make_image(pixel_format, video_range, fast);
	}
	return _image;
}


shared_ptr<const Image>
PlayerVideo::raw_image() const
{
	return _in->image(Image::Alignment::COMPACT, _inter_size).image;
}


/** Create an image for this frame.  A lock must be held on _mutex.
 *  @param pixel_format Functor returning output image pixel format for a given input pixel format.
 *  @param fast true to be fast at the expense of quality.
 */
void
PlayerVideo::make_image(function<AVPixelFormat (AVPixelFormat)> pixel_format, VideoRange video_range, bool fast) const
{
	_image_crop = _crop;
	_image_inter_size = _inter_size;
	_image_out_size = _out_size;
	_image_fade = _fade;

	auto prox = _in->image(Image::Alignment::PADDED, _inter_size);
	_error = prox.error;

	/* Handle HDR tone mapping if required */
	/* HDR Preview Pipeline: PQ XYZ (P3 D65) -> Linear XYZ -> Tone Map -> sRGB (Rec.709 D65) */
	if (_content.lock() && _content.lock()->video && _content.lock()->video->video_is_hdr() && prox.image->pixel_format() == AV_PIX_FMT_XYZ12LE) {
		int const w = prox.image->size().width;
		int const h = prox.image->size().height;
		int const stride_bytes = prox.image->stride()[0];

		/* Output as RGB48LE instead of XYZ12LE to bypass SDR color conversion pipeline */
		auto srgb_out = make_shared<Image>(AV_PIX_FMT_RGB48LE, prox.image->size(), prox.image->alignment());

		uint8_t* in_p = prox.image->data()[0];
		uint8_t* out_p = srgb_out->data()[0];
		int const out_stride = srgb_out->stride()[0];

		/* ST 2084 Constants for PQ EOTF (Inverse of OETF) */
		float const c1 = 0.8359375f;
		float const c2 = 18.8515625f;
		float const c3 = 18.6875f;
		float const m1 = 0.1593017578125f;
		float const m2 = 78.84375f;

		/* XYZ D65 -> Linear P3 D65 */
		float const M_XYZ_P3[9] = {
			 2.49349691f, -0.93138362f, -0.40271078f,
			-0.82948897f,  1.76266406f,  0.02362469f,
			 0.03584583f, -0.07617239f,  0.95688452f
		};

		/* P3 D65 -> Rec.709 D65 (same white point, only primaries change) */
		/* This matrix converts linear P3 to linear Rec.709 */
		float const M_P3_709[9] = {
			 1.2249f, -0.2249f,  0.0000f,
			-0.0420f,  1.0420f,  0.0000f,
			-0.0197f, -0.0786f,  1.0983f
		};

		for (int y = 0; y < h; ++y) {
			uint16_t* in_line = reinterpret_cast<uint16_t*>(in_p);
			uint16_t* out_line = reinterpret_cast<uint16_t*>(out_p);

			for (int x = 0; x < w; ++x) {
				/* Read XYZ12LE (12-bit in 16-bit, MSB aligned) */
				float X_pq = (in_line[x * 3 + 0] >> 4) / 4095.0f;
				float Y_pq = (in_line[x * 3 + 1] >> 4) / 4095.0f;
				float Z_pq = (in_line[x * 3 + 2] >> 4) / 4095.0f;

				/* PQ EOTF: Decode to linear luminance (cd/m²) */
				auto pq_eotf = [&](float N) -> float {
					if (N <= 0.0f) return 0.0f;
					float N_pow = std::pow(N, 1.0f / m2);
					float num = std::max(N_pow - c1, 0.0f);
					float den = c2 - c3 * N_pow;
					if (den <= 0.0f) return 0.0f;
					return std::pow(num / den, 1.0f / m1) * 10000.0f;
				};

				float X_nits = pq_eotf(X_pq);
				float Y_nits = pq_eotf(Y_pq);
				float Z_nits = pq_eotf(Z_pq);

				/* Tone Mapping: Simple linear scaling to 300 nits -> 1.0 */
				float const tone_scale = 1.0f / 300.0f;
				float X_lin = X_nits * tone_scale;
				float Y_lin = Y_nits * tone_scale;
				float Z_lin = Z_nits * tone_scale;

				/* XYZ D65 -> Linear P3 D65 */
				float P3_R = X_lin * M_XYZ_P3[0] + Y_lin * M_XYZ_P3[1] + Z_lin * M_XYZ_P3[2];
				float P3_G = X_lin * M_XYZ_P3[3] + Y_lin * M_XYZ_P3[4] + Z_lin * M_XYZ_P3[5];
				float P3_B = X_lin * M_XYZ_P3[6] + Y_lin * M_XYZ_P3[7] + Z_lin * M_XYZ_P3[8];

				/* P3 D65 -> Rec.709 D65 (Gamut Compression) */
				float R_709 = P3_R * M_P3_709[0] + P3_G * M_P3_709[1] + P3_B * M_P3_709[2];
				float G_709 = P3_R * M_P3_709[3] + P3_G * M_P3_709[4] + P3_B * M_P3_709[5];
				float B_709 = P3_R * M_P3_709[6] + P3_G * M_P3_709[7] + P3_B * M_P3_709[8];

				/* Clamp to [0, 1] (simple gamut clipping) */
				R_709 = std::max(0.0f, std::min(1.0f, R_709));
				G_709 = std::max(0.0f, std::min(1.0f, G_709));
				B_709 = std::max(0.0f, std::min(1.0f, B_709));

				/* sRGB OETF (IEC 61966-2-1) */
				auto srgb_oetf = [](float L) -> float {
					if (L <= 0.0031308f) {
						return L * 12.92f;
					} else {
						return 1.055f * std::pow(L, 1.0f / 2.4f) - 0.055f;
					}
				};

				float sR = srgb_oetf(R_709);
				float sG = srgb_oetf(G_709);
				float sB = srgb_oetf(B_709);

				/* Write RGB48LE (16-bit per channel) */
				out_line[x * 3 + 0] = static_cast<uint16_t>(sR * 65535.0f + 0.5f);
				out_line[x * 3 + 1] = static_cast<uint16_t>(sG * 65535.0f + 0.5f);
				out_line[x * 3 + 2] = static_cast<uint16_t>(sB * 65535.0f + 0.5f);
			}

			in_p += stride_bytes;
			out_p += out_stride;
		}

		/* Replace input with our sRGB output - bypass SDR color pipeline */
		prox.image = srgb_out;
	}

	auto total_crop = _crop;
	switch (_part) {
	case Part::LEFT_HALF:
		total_crop.right += prox.image->size().width / 2;
		break;
	case Part::RIGHT_HALF:
		total_crop.left += prox.image->size().width / 2;
		break;
	case Part::TOP_HALF:
		total_crop.bottom += prox.image->size().height / 2;
		break;
	case Part::BOTTOM_HALF:
		total_crop.top += prox.image->size().height / 2;
		break;
	default:
		break;
	}

	if (prox.log2_scaling > 0) {
		/* Scale the crop down to account for the scaling that has already happened in ImageProxy::image */
		int const r = pow(2, prox.log2_scaling);
		total_crop.left /= r;
		total_crop.right /= r;
		total_crop.top /= r;
		total_crop.bottom /= r;
	}

	dcp::YUVToRGB yuv_to_rgb = dcp::YUVToRGB::REC709;
	if (_colour_conversion) {
		yuv_to_rgb = _colour_conversion.get().yuv_to_rgb();
	}

	_image = prox.image->crop_scale_window(
		total_crop, _inter_size, _out_size, yuv_to_rgb, _video_range, pixel_format(prox.image->pixel_format()), video_range, Image::Alignment::COMPACT, fast
		);

	if (_text) {
		_image->alpha_blend(_text->image, _text->position);
	}

	if (_fade) {
		_image->fade(_fade.get());
	}
}


void
PlayerVideo::add_metadata(xmlpp::Element* element) const
{
	_crop.as_xml(element);
	if (_fade) {
		cxml::add_text_child(element, "Fade", fmt::to_string(_fade.get()));
	}
	_in->add_metadata(cxml::add_child(element, "In"));
	cxml::add_text_child(element, "InterWidth", fmt::to_string(_inter_size.width));
	cxml::add_text_child(element, "InterHeight", fmt::to_string(_inter_size.height));
	cxml::add_text_child(element, "OutWidth", fmt::to_string(_out_size.width));
	cxml::add_text_child(element, "OutHeight", fmt::to_string(_out_size.height));
	cxml::add_text_child(element, "Eyes", fmt::to_string(static_cast<int>(_eyes)));
	cxml::add_text_child(element, "Part", fmt::to_string(static_cast<int>(_part)));
	cxml::add_text_child(element, "VideoRange", fmt::to_string(static_cast<int>(_video_range)));
	cxml::add_text_child(element, "Error", _error ? "1" : "0");
	if (_colour_conversion) {
		_colour_conversion.get().as_xml(element);
	}
	if (_text) {
		cxml::add_text_child(element, "SubtitleWidth", fmt::to_string(_text->image->size().width));
		cxml::add_text_child(element, "SubtitleHeight", fmt::to_string(_text->image->size().height));
		cxml::add_text_child(element, "SubtitleX", fmt::to_string(_text->position.x));
		cxml::add_text_child(element, "SubtitleY", fmt::to_string(_text->position.y));
	}
}


void
PlayerVideo::write_to_socket(shared_ptr<Socket> socket) const
{
	_in->write_to_socket(socket);
	if (_text) {
		_text->image->write_to_socket(socket);
	}
}


bool
PlayerVideo::has_j2k() const
{
	/* XXX: maybe other things */

	auto j2k = dynamic_pointer_cast<const J2KImageProxy>(_in);
	if (!j2k) {
		return false;
	}

	return _crop == Crop() && _out_size == j2k->size() && _inter_size == j2k->size() && !_text && !_fade && !_colour_conversion;
}


shared_ptr<const dcp::Data>
PlayerVideo::j2k() const
{
	auto j2k = dynamic_pointer_cast<const J2KImageProxy>(_in);
	DCPOMATIC_ASSERT(j2k);
	return j2k->j2k();
}


Position<int>
PlayerVideo::inter_position() const
{
	return Position<int>((_out_size.width - _inter_size.width) / 2, (_out_size.height - _inter_size.height) / 2);
}


/** @return true if this PlayerVideo is definitely the same as another, false if it is probably not */
bool
PlayerVideo::same(shared_ptr<const PlayerVideo> other) const
{
	if (_crop != other->_crop ||
	    _fade != other->_fade ||
	    _inter_size != other->_inter_size ||
	    _out_size != other->_out_size ||
	    _eyes != other->_eyes ||
	    _part != other->_part ||
	    _colour_conversion != other->_colour_conversion ||
	    _video_range != other->_video_range) {
		return false;
	}

	if ((!_text && other->_text) || (_text && !other->_text)) {
		/* One has a text and the other doesn't */
		return false;
	}

	if (_text && other->_text && !_text->same(other->_text.get())) {
		/* They both have texts but they are different */
		return false;
	}

	/* Now neither has subtitles */

	return _in->same(other->_in);
}


void
PlayerVideo::prepare(AVPixelFormat pixel_format, VideoRange video_range, Image::Alignment alignment, bool fast, bool proxy_only)
{
	_in->prepare(alignment, _inter_size);
	boost::mutex::scoped_lock lm(_mutex);
	if (!_image && !proxy_only) {
		make_image(force(pixel_format), video_range, fast);
	}
}


size_t
PlayerVideo::memory_used() const
{
	return _in->memory_used();
}


/** @return Shallow copy of this; _in and _text are shared between the original and the copy */
shared_ptr<PlayerVideo>
PlayerVideo::shallow_copy() const
{
	return std::make_shared<PlayerVideo>(
		_in,
		_crop,
		_fade,
		_inter_size,
		_out_size,
		_eyes,
		_part,
		_colour_conversion,
		_video_range,
		_content,
		_video_time,
		_error
		);
}


/** Re-read crop, fade, inter/out size, colour conversion and video range from our content.
 *  @return true if this was possible, false if not.
 */
bool
PlayerVideo::reset_metadata(shared_ptr<const Film> film, dcp::Size player_video_container_size)
{
	auto content = _content.lock();
	if (!content || !_video_time) {
		return false;
	}

	_crop = content->video->actual_crop();
	_fade = content->video->fade(film, _video_time.get());
	auto const size = content->video->scaled_size(film->frame_size());
	if (!size) {
		return false;
	}

	_inter_size = scale_for_display(
		*size,
		player_video_container_size,
		film->frame_size(),
		content->video->pixel_quanta()
		);
	_out_size = player_video_container_size;
	_colour_conversion = content->video->colour_conversion();
	_video_range = content->video->range();

	return true;
}
