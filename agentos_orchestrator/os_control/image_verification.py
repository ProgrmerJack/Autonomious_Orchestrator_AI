from __future__ import annotations

import hashlib
import struct
import zlib
from dataclasses import dataclass


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(slots=True)
class PngImageSummary:
    width: int
    height: int
    bit_depth: int
    color_type: int
    sha256: str
    distinct_pixel_count: int
    non_background_pixel_count: int
    non_background_bounds: tuple[int, int, int, int] | None = None


@dataclass(slots=True)
class _PngPayload:
    width: int
    height: int
    bit_depth: int
    color_type: int
    interlace: int
    idat: bytes


def analyze_png(value: bytes) -> PngImageSummary:
    payload = _parse_png_payload(value)
    _validate_png_payload(payload)
    channels = _channel_count(payload.color_type)
    pixels = _decode_png_pixels(
        zlib.decompress(payload.idat),
        payload.width,
        payload.height,
        channels,
    )
    if not pixels:
        raise ValueError("PNG image contained no pixels")
    background = pixels[0]
    bounds = _non_background_bounds(pixels, payload.width, background)
    return PngImageSummary(
        width=payload.width,
        height=payload.height,
        bit_depth=payload.bit_depth,
        color_type=payload.color_type,
        sha256=hashlib.sha256(value).hexdigest(),
        distinct_pixel_count=len(set(pixels)),
        non_background_pixel_count=sum(
            1 for pixel in pixels if pixel != background
        ),
        non_background_bounds=bounds,
    )


def _non_background_bounds(
    pixels: list[bytes],
    width: int,
    background: bytes,
) -> tuple[int, int, int, int] | None:
    foreground = [
        (index % width, index // width)
        for index, pixel in enumerate(pixels)
        if pixel != background
    ]
    if not foreground:
        return None
    xs = [point[0] for point in foreground]
    ys = [point[1] for point in foreground]
    return min(xs), min(ys), max(xs), max(ys)


def _parse_png_payload(value: bytes) -> _PngPayload:
    if not value.startswith(PNG_SIGNATURE):
        raise ValueError("Saved Paint artifact is not a PNG file")
    width = height = bit_depth = color_type = interlace = 0
    idat_parts: list[bytes] = []
    for chunk_type, chunk_data in _png_chunks(value):
        if chunk_type == b"IHDR":
            (
                width,
                height,
                bit_depth,
                color_type,
                _compression,
                _filter,
                interlace,
            ) = struct.unpack(">IIBBBBB", chunk_data)
        elif chunk_type == b"IDAT":
            idat_parts.append(chunk_data)
        elif chunk_type == b"IEND":
            break
    return _PngPayload(
        width=width,
        height=height,
        bit_depth=bit_depth,
        color_type=color_type,
        interlace=interlace,
        idat=b"".join(idat_parts),
    )


def _png_chunks(value: bytes) -> list[tuple[bytes, bytes]]:
    chunks: list[tuple[bytes, bytes]] = []
    offset = len(PNG_SIGNATURE)
    while offset + 8 <= len(value):
        length = struct.unpack(">I", value[offset:offset + 4])[0]
        chunk_type = value[offset + 4:offset + 8]
        data_start = offset + 8
        data_end = data_start + length
        if data_end + 4 > len(value):
            raise ValueError("PNG chunk extends past end of file")
        chunks.append((chunk_type, value[data_start:data_end]))
        offset = data_end + 4
    return chunks


def _validate_png_payload(payload: _PngPayload) -> None:
    if payload.width <= 0 or payload.height <= 0:
        raise ValueError("PNG image dimensions were not found")
    if payload.interlace != 0:
        raise ValueError(
            "Interlaced PNG files are not supported for verification"
        )
    if payload.bit_depth != 8:
        raise ValueError(f"Unsupported PNG bit depth: {payload.bit_depth}")
    if not payload.idat:
        raise ValueError("PNG file did not contain image data")
    _channel_count(payload.color_type)


def _channel_count(color_type: int) -> int:
    channel_counts = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
    channels = channel_counts.get(color_type)
    if channels is None:
        raise ValueError(f"Unsupported PNG color type: {color_type}")
    return channels


def _decode_png_pixels(
    decompressed: bytes,
    width: int,
    height: int,
    channels: int,
) -> list[bytes]:
    row_length = width * channels
    offset = 0
    previous = bytearray(row_length)
    pixels: list[bytes] = []
    for _row in range(height):
        if offset >= len(decompressed):
            raise ValueError(
                "PNG image data ended before all rows were decoded"
            )
        filter_type = decompressed[offset]
        offset += 1
        row = bytearray(decompressed[offset:offset + row_length])
        if len(row) != row_length:
            raise ValueError("PNG scanline length did not match image width")
        offset += row_length
        _apply_png_filter(row, previous, channels, filter_type)
        pixels.extend(_row_pixels(row, channels))
        previous = row
    return pixels


def _row_pixels(row: bytearray, channels: int) -> list[bytes]:
    return [
        bytes(row[pixel_offset:pixel_offset + channels])
        for pixel_offset in range(0, len(row), channels)
    ]


def _apply_png_filter(
    row: bytearray,
    previous: bytearray,
    bytes_per_pixel: int,
    filter_type: int,
) -> None:
    handlers = {
        0: _filter_none,
        1: _filter_sub,
        2: _filter_up,
        3: _filter_average,
        4: _filter_paeth,
    }
    try:
        handlers[filter_type](row, previous, bytes_per_pixel)
    except KeyError as exc:
        raise ValueError(
            f"Unsupported PNG filter type: {filter_type}"
        ) from exc


def _filter_none(
    row: bytearray,
    previous: bytearray,
    bytes_per_pixel: int,
) -> None:
    _ = (row, previous, bytes_per_pixel)


def _filter_sub(
    row: bytearray,
    previous: bytearray,
    bytes_per_pixel: int,
) -> None:
    _ = previous
    for index, value in enumerate(row):
        row[index] = (value + _left(row, bytes_per_pixel, index)) & 0xFF


def _filter_up(
    row: bytearray,
    previous: bytearray,
    bytes_per_pixel: int,
) -> None:
    _ = bytes_per_pixel
    for index, value in enumerate(row):
        row[index] = (value + previous[index]) & 0xFF


def _filter_average(
    row: bytearray,
    previous: bytearray,
    bytes_per_pixel: int,
) -> None:
    for index, value in enumerate(row):
        average = (_left(row, bytes_per_pixel, index) + previous[index]) // 2
        row[index] = (value + average) & 0xFF


def _filter_paeth(
    row: bytearray,
    previous: bytearray,
    bytes_per_pixel: int,
) -> None:
    for index, value in enumerate(row):
        row[index] = (
            value + _paeth_predictor(
                _left(row, bytes_per_pixel, index),
                previous[index],
                _upper_left(previous, bytes_per_pixel, index),
            )
        ) & 0xFF


def _left(row: bytearray, bytes_per_pixel: int, index: int) -> int:
    if index < bytes_per_pixel:
        return 0
    return row[index - bytes_per_pixel]


def _upper_left(previous: bytearray, bytes_per_pixel: int, index: int) -> int:
    if index < bytes_per_pixel:
        return 0
    return previous[index - bytes_per_pixel]


def _paeth_predictor(left: int, up: int, upper_left: int) -> int:
    estimate = left + up - upper_left
    left_distance = abs(estimate - left)
    up_distance = abs(estimate - up)
    upper_left_distance = abs(estimate - upper_left)
    if left_distance <= up_distance and left_distance <= upper_left_distance:
        return left
    if up_distance <= upper_left_distance:
        return up
    return upper_left
