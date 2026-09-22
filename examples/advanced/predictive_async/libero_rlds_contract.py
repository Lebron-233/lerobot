"""Bounded, read-only TensorFlow Example/TFRecord decoding for LIBERO provenance.

No TensorFlow installation, pickle, network, model or physics execution. Supports
only the wire types used by tf.train.Example and checks both TFRecord CRC32Cs.
The stored RLDS source path, not converted episode numbers, identifies a demo.
"""

import struct
from pathlib import Path

import numpy as np


def varint(data, pos):
    value = 0
    for shift in range(0, 70, 7):
        if pos >= len(data):
            raise ValueError("Truncated varint")
        byte = data[pos]
        pos += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, pos
    raise ValueError("Overlong varint")


def fields(data):
    pos = 0
    while pos < len(data):
        tag, pos = varint(data, pos)
        number, wire = tag >> 3, tag & 7
        if not number:
            raise ValueError("Zero field number")
        if wire == 0:
            value, pos = varint(data, pos)
        elif wire in (1, 2, 5):
            if wire == 2:
                size, pos = varint(data, pos)
            else:
                size = 8 if wire == 1 else 4
            if size > len(data) - pos:
                raise ValueError("Truncated field")
            value = data[pos : pos + size]
            pos += size
        else:
            raise ValueError("Unsupported wire type")
        yield number, wire, value


def _single(data, number):
    rows = list(fields(data))
    if len(rows) != 1 or rows[0][:2] != (number, 2):
        raise ValueError("Unexpected Example container")
    return rows[0][2]


def parse_example(data):
    result = {}
    for number, wire, entry in fields(_single(data, 1)):
        if (number, wire) != (1, 2):
            raise ValueError("Unexpected Features entry")
        items = {n: (w, v) for n, w, v in fields(entry)}
        if set(items) != {1, 2} or items[1][0] != 2 or items[2][0] != 2:
            raise ValueError("Invalid feature map")
        key = items[1][1].decode("utf-8")
        if key in result:
            raise ValueError("Duplicate feature")
        body = list(fields(items[2][1]))
        if len(body) != 1 or body[0][1] != 2:
            raise ValueError("Invalid Feature oneof")
        kind, _, packed = body[0]
        values = []
        for n, w, value in fields(packed):
            if n != 1:
                raise ValueError("Unexpected list field")
            if kind == 1 and w == 2:
                values.append(value)
            elif kind == 2 and w in (2, 5):
                if len(value) % 4:
                    raise ValueError("FloatList byte count")
                values.extend(np.frombuffer(value, dtype="<f4"))
            elif kind == 3 and w == 0:
                values.append(value if value < (1 << 63) else value - (1 << 64))
            elif kind == 3 and w == 2:
                pos = 0
                while pos < len(value):
                    x, pos = varint(value, pos)
                    values.append(x if x < (1 << 63) else x - (1 << 64))
            else:
                raise ValueError("Unsupported Feature list")
        result[key] = values if kind == 1 else np.asarray(values, dtype=np.float32 if kind == 2 else np.int64)
    return result


def _crc_table():
    table = []
    for i in range(256):
        c = i
        for _ in range(8):
            c = (c >> 1) ^ (0x82F63B78 if c & 1 else 0)
        table.append(c)
    return table


CRC_TABLE = _crc_table()


def crc32c(data):
    try:
        import google_crc32c
    except ImportError:
        c = 0xFFFFFFFF
        for byte in data:
            c = CRC_TABLE[(c ^ byte) & 255] ^ (c >> 8)
        return c ^ 0xFFFFFFFF
    return google_crc32c.value(data)


def masked_crc(data):
    c = crc32c(data)
    return (((c >> 15) | (c << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def records(path, max_records=32, max_record_bytes=32 * 1024 * 1024):
    """Yield (offset, verified protobuf payload); reject partial downloads."""
    with Path(path).open("rb") as f:
        for _ in range(max_records):
            offset = f.tell()
            header = f.read(12)
            if not header:
                return
            if len(header) != 12:
                raise ValueError("Truncated TFRecord header")
            n, checksum = struct.unpack("<QI", header)
            if not 0 < n <= max_record_bytes or masked_crc(header[:8]) != checksum:
                raise ValueError("Invalid TFRecord length/CRC")
            payload, tail = f.read(n), f.read(4)
            if len(payload) != n or len(tail) != 4:
                raise ValueError("Truncated TFRecord payload")
            if masked_crc(payload) != struct.unpack("<I", tail)[0]:
                raise ValueError("Payload CRC mismatch")
            yield offset, payload
        if f.read(1):
            raise ValueError("TFRecord record budget exceeded")


def decode_episode(features):
    action = features["steps/action"].reshape(-1, 7)
    n = len(action)
    state = features["steps/observation/state"].reshape(-1, 8)
    joint = features["steps/observation/joint_state"].reshape(-1, 7)
    paths = features["episode_metadata/file_path"]
    language = features["steps/language_instruction"]
    if len(paths) != 1 or len(language) != n or len(set(language)) != 1:
        raise ValueError("Non-unique demo identity/language")
    if state.shape != (n, 8) or joint.shape != (n, 7) or not all(np.isfinite(v).all() for v in (action, state, joint)):
        raise ValueError("State/action shape or values")
    first, last, terminal = (features["steps/" + key] for key in ("is_first", "is_last", "is_terminal"))
    if not (np.array_equal(first, np.arange(n) == 0) and np.array_equal(last, np.arange(n) == n - 1) and np.array_equal(last, terminal)):
        raise ValueError("Episode boundary flags differ")
    images, wrists = features["steps/observation/image"], features["steps/observation/wrist_image"]
    if len(images) != n or len(wrists) != n:
        raise ValueError("Image row count differs")
    return {"source_path": paths[0].decode(), "language": language[0].decode(), "actions": action,
            "states": state, "joints": joint, "images": images, "wrists": wrists}
