"""Synthetic protocol tests; no downloads or robot execution."""

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/advanced/predictive_async"))
from libero_rlds_contract import crc32c, masked_crc, parse_example, records, varint  # noqa: E402


def vi(x):
    result = bytearray()
    while x > 127:
        result.append((x & 127) | 128)
        x >>= 7
    result.append(x)
    return bytes(result)


def field(n, value):
    return vi(8 * n + 2) + vi(len(value)) + value


def example():
    def entry(key, kind, payload):
        return field(1, field(1, key.encode()) + field(2, field(kind, payload)))
    return field(1, entry("floats", 2, field(1, struct.pack("<3f", 1, 2, 3)))
                 + entry("bytes", 1, field(1, b"abc") + field(1, b"def"))
                 + entry("ints", 3, field(1, vi(0) + vi(150))))


def test_crc_reference_and_varint():
    assert crc32c(b"123456789") == 0xE3069283
    assert varint(vi(150), 0) == (150, 2)


def test_decoding_typed_features():
    v = parse_example(example())
    assert np.array_equal(v["floats"], np.array([1, 2, 3], dtype=np.float32))
    assert v["bytes"] == [b"abc", b"def"]
    assert np.array_equal(v["ints"], [0, 150])


def test_two_records_and_record_limit(tmp_path):
    data = example()
    head = struct.pack("<Q", len(data))
    raw = head + struct.pack("<I", masked_crc(head)) + data + struct.pack("<I", masked_crc(data))
    p = tmp_path / "data.tfrecord"
    p.write_bytes(raw * 2)
    assert list(records(p)) == [(0, data), (len(raw), data)]
    with pytest.raises(ValueError, match="budget"):
        list(records(p, max_records=1))


@pytest.mark.parametrize("at", [0, 8, 15, -1])
def test_corruption_is_rejected(tmp_path, at):
    data = example()
    head = struct.pack("<Q", len(data))
    raw = bytearray(head + struct.pack("<I", masked_crc(head)) + data + struct.pack("<I", masked_crc(data)))
    raw[at] ^= 1
    p = tmp_path / "bad"
    p.write_bytes(raw)
    with pytest.raises(ValueError):
        list(records(p))


@pytest.mark.parametrize("raw", [b"\x80", b"\x0a\x07a", b"\x00"])
def test_invalid_proto_rejected(raw):
    with pytest.raises(ValueError):
        parse_example(raw)
