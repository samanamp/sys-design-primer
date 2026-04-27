---
title: KV store with custom serialization to disk
description: KV store with custom serialization to disk
---
## Question
Implement serialization and deserialization for a key-value store where both keys and values can contain any characters including delimiters; you can't use simple delimiters because they might appear in the data; most candidates land on length-prefix encoding (3:key5:value), the same pattern used in Redis protocol. Custom serialization/deserialization must be implemented (no Python built-in libraries like json).

```
class KVStore:
    def set(self, key: str, value: str): ...
    def get(self, key: str) -> str | None: ...
    def save(self, path: str): ...
    @classmethod
    def load(cls, path: str) -> "KVStore": ...
```
- **P1**: Length-prefix encoding. Show on paper why a delimiter approach (e.g. `key=value\n`) breaks for arbitrary-content keys/values.
- **P2**: Round-trip property test: random keys/values containing `\n`, `:`, `\0`, unicode — must survive.
- **P3**: Append-only log instead of full rewrite on every `save()`. Compaction strategy.
- **P4**: Crash mid-write. How do you detect a torn record on `load()`? (Length prefix + checksum, or length prefix + final-marker.)

---
Here's a version that hits all four marks, is correct, and is realistically writable in 70 minutes. I've kept the framing binary because it's actually *less* error-prone than ASCII parens-and-commas under pressure — `struct` does the work for you.

```python
import os
import struct
import zlib

# Record layout (big-endian):
#   op:    1 byte   ('S' = set, 'D' = delete)
#   klen:  4 bytes  (unsigned)
#   vlen:  4 bytes  (unsigned; 0 for delete)
#   key:   klen bytes (utf-8)
#   value: vlen bytes (utf-8)
#   crc32: 4 bytes over [op | klen | vlen | key | value]
#
# Length prefix → arbitrary bytes survive (P1).
# CRC → torn or corrupted tail detected at load (P4).

_HDR = struct.Struct(">cII")
_CRC = struct.Struct(">I")
_OP_SET = b"S"
_OP_DEL = b"D"


class KVStore:
    def __init__(self):
        self._data: dict[str, str] = {}

    # ---- public API ----

    def set(self, key: str, value: str) -> None:
        self._data[key] = value

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def delete(self, key: str) -> None:
        self._data.pop(key, None)

    def save(self, path: str) -> None:
        """Full snapshot. Atomic: tmp + fsync + rename."""
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            for k, v in self._data.items():
                f.write(self._encode(_OP_SET, k, v))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def append(self, path: str, key: str, value: str) -> None:
        """ THIS IS A SHITY SOLUTION- USE WATERMARK"""
        """Incremental write: append one record, no full rewrite (P3)."""
        self._data[key] = value
        with open(path, "ab") as f:
            f.write(self._encode(_OP_SET, key, value))
            f.flush()
            os.fsync(f.fileno())

    def compact(self, path: str) -> None:
        """Rewrite log as snapshot of live state. Trigger when
        log_size > 2 * live_bytes (Bitcask heuristic)."""
        self.save(path)

    @classmethod
    def load(cls, path: str) -> "KVStore":
        store = cls()
        if not os.path.exists(path):
            return store

        good_offset = 0
        with open(path, "rb") as f:
            while True:
                rec = cls._read_record(f)
                if rec is None:
                    break
                op, k, v = rec
                if op == _OP_SET:
                    store._data[k] = v
                elif op == _OP_DEL:
                    store._data.pop(k, None)
                good_offset = f.tell()

        # Heal torn tail so future appends start at a known-good offset.
        if good_offset < os.path.getsize(path):
            os.truncate(path, good_offset)
        return store

    # ---- framing ----

    @staticmethod
    def _encode(op: bytes, key: str, value: str = "") -> bytes:
        kb = key.encode("utf-8")
        vb = value.encode("utf-8")
        body = _HDR.pack(op, len(kb), len(vb)) + kb + vb
        return body + _CRC.pack(zlib.crc32(body))

    @staticmethod
    def _read_record(f):
        hdr = f.read(_HDR.size)
        if not hdr:
            return None                              # clean EOF
        if len(hdr) < _HDR.size:
            return None                              # torn header
        op, klen, vlen = _HDR.unpack(hdr)
        payload = f.read(klen + vlen)
        crc_bytes = f.read(_CRC.size)
        if len(payload) < klen + vlen or len(crc_bytes) < _CRC.size:
            return None                              # torn body
        if zlib.crc32(hdr + payload) != _CRC.unpack(crc_bytes)[0]:
            return None                              # corrupted
        return op, payload[:klen].decode("utf-8"), payload[klen:].decode("utf-8")
```

P2 — round-trip property test:

```python
def test_roundtrip():
    import random, string, tempfile
    pool = string.printable + "\n\0:," + "فارسیファ🦀"
    for _ in range(500):
        kv = KVStore()
        for _ in range(random.randint(0, 50)):
            k = "".join(random.choice(pool) for _ in range(random.randint(0, 20)))
            v = "".join(random.choice(pool) for _ in range(random.randint(0, 100)))
            kv.set(k, v)
        path = tempfile.NamedTemporaryFile(delete=False).name
        kv.save(path)
        assert kv._data == KVStore.load(path)._data
```

P4 — torn-tail test:

```python
def test_torn_tail():
    kv = KVStore(); kv.set("a", "1"); kv.set("b", "2")
    kv.save("/tmp/kv")
    with open("/tmp/kv", "ab") as f:
        f.write(b"S\x00\x00\x00\x05hell")  # truncated record, no CRC
    kv2 = KVStore.load("/tmp/kv")
    assert kv2.get("a") == "1" and kv2.get("b") == "2"
```

## How to actually budget the 70 minutes

| Minutes | What you produce |
|---|---|
| 0–5 | Clarify: persistence model? concurrency? Sketch record layout on the board, narrate why ASCII delimiters fail. |
| 5–25 | `set`, `get`, `save`, `load` with bytes-counted length prefix. No CRC yet. Get the basic round-trip running. |
| 25–35 | Run the property test in your head (or actually) with `\n`, `\0`, unicode. (P2) |
| 35–55 | Add CRC32 + the `_read_record` returns-None-on-anything-bad pattern + `os.truncate` heal. Demo torn-tail test. (P4) |
| 55–70 | Add `append` + `compact`, talk through compaction trigger (size ratio, or every N writes), and the tombstone op for deletes. (P3) |

## What to narrate while coding (the part that gets you the offer)

1. **"Length must be in bytes, not characters."** Encode once, count `len(kb)`. Mention UTF-8 explicitly. This is the single most common bug interviewers watch for after they accept your length-prefix idea.
2. **"Atomicity: tmp + fsync + rename."** Say the word `fsync` out loud. Mention that on Linux, `rename` is atomic within a filesystem. Bonus: `os.fsync` the *directory* fd if they push.
3. **"CRC vs final-marker."** Both detect truncation. Only CRC detects bit-flips and partial-page-write corruption. The cost is 4 bytes per record. Worth it.
4. **"Compaction trigger."** `log_size > k * live_bytes` for k≈2, or "every N writes," or "when stale fraction exceeds threshold." Reference Bitcask if asked.
5. **"Deletes need tombstones in append-only mode"** — otherwise compaction is the only way to actually reclaim space, and replays would resurrect deleted keys.

## Shortcuts that are fair game under pressure (and that I took above)

- No locking / no concurrency story. State the assumption: single-writer.
- No hint file / no in-memory index of offsets. State: "for >RAM datasets I'd add a Bitcask-style hint file mapping key → (file_id, offset, size)."
- No varint length encoding. Fixed 4-byte unsigned is fine; mention varint only if pushed on space.
- `os.fsync` on every `append` is slow. State: "in production I'd batch with group-commit on a flush interval."

The version above is ~80 lines of substance. That's the right size for 70 minutes — anything significantly longer means you over-engineered and probably didn't get to P3/P4 cleanly.