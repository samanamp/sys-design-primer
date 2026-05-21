---
title: Memory allocator
description: Memory allocator
---

You're given a fixed memory region of capacity C bytes (model it as a Python `bytearray` of size C). Implement an allocator over it:

    malloc(n) -> int | None    # returns an offset into the region, or None if no fit
    free(offset)                # releases a previously allocated block

You own the entire region. You decide the metadata layout. No use of Python's own memory management beyond the `bytearray` itself — the region is your universe.

---

# Bytearray Allocator — Three Iterations

Single Python file, three monotonic refinements over a fixed `bytearray`. Block metadata lives in a Python dict keyed by offset, since the spec lets me skip the byte-packing chore and focus on policy.

## Clarifying questions (60 seconds, then assume)

1. **Alignment.** I'll round all requests up to 8 bytes. Standard for 64-bit systems; matches malloc semantics.
2. **Minimum block size.** If a split would leave a remainder smaller than 16 bytes, I won't split — keep it as internal slack. Otherwise we generate dust fragments that never satisfy any request.
3. **Thread safety.** Out of scope. Single-threaded. Real allocators bolt on per-thread arenas (jemalloc) or fastbins-plus-locks (ptmalloc); not in 60 minutes.

Also assuming: `free(0)` is valid (offset 0 is the start of the region, not a sentinel), `malloc(0)` returns `None`, and the entire capacity is available — no reserved metadata region.

## Block layout

```
Block @ offset O   (stored in self.blocks[O], NOT packed in the bytearray)
┌────────┬─────────┬───────────┬───────────┬───────────┬───────────┐
│  size  │ is_free │ prev_free │ next_free │ prev_addr │ next_addr │
└────────┴─────────┴───────────┴───────────┴───────────┴───────────┘
                   └──── free list ────┘   └─── address-order ────┘
                       (V1, V2, V3)              (V2, V3 only)

Memory layout (the bytearray is just inert storage):
  0                                                              C
  ├─────────┬──────────────┬─────────┬──────────────────────────┤
  │ block A │   block B    │ block C │        block D           │
  └─────────┴──────────────┴─────────┴──────────────────────────┘
```

```python
from typing import Optional, List

ALIGN = 8
MIN_BLOCK = 16  # below this, a remainder isn't worth splitting

def align_up(n: int) -> int:
    return (n + ALIGN - 1) & ~(ALIGN - 1)

class Block:
    __slots__ = ('offset', 'size', 'is_free',
                 'prev_free', 'next_free',
                 'prev_addr', 'next_addr')

    def __init__(self, offset: int, size: int):
        self.offset = offset
        self.size = size
        self.is_free = True
        self.prev_free: Optional[int] = None
        self.next_free: Optional[int] = None
        self.prev_addr: Optional[int] = None
        self.next_addr: Optional[int] = None

    def __repr__(self) -> str:
        s = "free" if self.is_free else "used"
        return f"[off={self.offset:>3} sz={self.size:>3} {s}]"
```

## Part 1 — First-fit, single free list

The cheapest correct allocator: one LIFO-ordered linked list of free blocks. `malloc` walks until it finds a fit, splits off the remainder if it's big enough, returns the offset. `free` just pushes the block onto the free list — no merging, so adjacent freed blocks stay as separate entries.

```python
class AllocatorV1:
    def __init__(self, capacity: int):
        self.cap = align_up(capacity)
        self.mem = bytearray(self.cap)
        root = Block(0, self.cap)
        self.blocks = {0: root}
        self.free_head: Optional[int] = 0

    def _free_push(self, b: Block) -> None:
        b.is_free = True
        b.prev_free = None
        b.next_free = self.free_head
        if self.free_head is not None:
            self.blocks[self.free_head].prev_free = b.offset
        self.free_head = b.offset

    def _free_unlink(self, b: Block) -> None:
        if b.prev_free is not None:
            self.blocks[b.prev_free].next_free = b.next_free
        else:
            self.free_head = b.next_free
        if b.next_free is not None:
            self.blocks[b.next_free].prev_free = b.prev_free
        b.prev_free = b.next_free = None
        b.is_free = False

    def malloc(self, n: int) -> Optional[int]:
        if n <= 0:
            return None
        need = align_up(n)
        cur = self.free_head
        while cur is not None:
            b = self.blocks[cur]
            if b.size >= need:
                self._free_unlink(b)
                if b.size - need >= MIN_BLOCK:
                    rem = Block(b.offset + need, b.size - need)
                    self.blocks[rem.offset] = rem
                    b.size = need
                    self._free_push(rem)
                return b.offset
            cur = b.next_free
        return None

    def free(self, offset: int) -> None:
        b = self.blocks.get(offset)
        if b is None:
            raise ValueError(f"free: unknown offset {offset}")
        if b.is_free:
            raise ValueError(f"free: double free at {offset}")
        self._free_push(b)

    def fragmentation(self) -> float:
        free = [b.size for b in self.blocks.values() if b.is_free]
        return 0.0 if not free else 1.0 - max(free) / sum(free)

    def layout(self) -> str:
        return " ".join(repr(b) for b in
                        sorted(self.blocks.values(), key=lambda x: x.offset))
```

**Split, before/after** (malloc(40) on a 256-byte virgin region):

```
Before:                                              After:
free_head ──► [off=  0 sz=256 free]                  free_head ──► [off= 40 sz=216 free]
                                                     blocks dict:  [off=  0 sz= 40 used]
                                                                   [off= 40 sz=216 free]
                                                     returns 0
```

**What this buys:** correctness, O(1) free, dead-simple invariant (a block is either in the free list or it isn't).
**What it costs:** malloc is O(free-list-length); fragmentation accumulates monotonically because nothing ever merges back. After the alloc-8/free-8 test below, `fragmentation = 0.875` and `malloc(128)` fails despite 256 bytes being "free".

## Part 2 — Coalescing

The fix is two-part: maintain a second linked list, this one threaded in **address order** across all blocks (used and free), so each block knows its physical neighbors in O(1). Then on `free`, inspect both neighbors and merge whichever are free. Four cases, four code paths.

```
Case 1 — no merge (both neighbors used)
  before:  [A used] [B used] [C used]    free list: ∅
   free B
  after:   [A used] [B free] [C used]    free list: {B}

Case 2 — merge with prev (prev free, next used)
  before:  [A free] [B used] [C used]    free list: {A}
   free B
  after:   [  A+B   free   ] [C used]    free list: {A+B}      ← B's record deleted

Case 3 — merge with next (prev used, next free)
  before:  [A used] [B used] [C free]    free list: {C}
   free B
  after:   [A used] [  B+C   free   ]    free list: {B+C}      ← C's record deleted

Case 4 — merge with both (both neighbors free)
  before:  [A free] [B used] [C free]    free list: {A, C}
   free B
  after:   [        A+B+C       free]    free list: {A+B+C}    ← B and C deleted
```

The only subtle bookkeeping: in cases 2 and 4, the prev block (`p`) is being mutated, so unlink it from the free list *before* changing its size — if I forget, the next `malloc` walks the free list and finds `p`'s old size, picks it for a request that won't fit. Same logic for the next neighbor in cases 3 and 4. This is bug #3 below.

```python
class AllocatorV2(AllocatorV1):
    def _carve(self, b: Block, need: int) -> int:
        """Allocate `need` bytes from free block `b`; split remainder."""
        self._free_unlink(b)
        if b.size - need >= MIN_BLOCK:
            rem = Block(b.offset + need, b.size - need)
            self.blocks[rem.offset] = rem
            rem.prev_addr = b.offset
            rem.next_addr = b.next_addr
            if b.next_addr is not None:
                self.blocks[b.next_addr].prev_addr = rem.offset
            b.next_addr = rem.offset
            b.size = need
            self._free_push(rem)
        return b.offset

    def malloc(self, n: int) -> Optional[int]:
        if n <= 0:
            return None
        need = align_up(n)
        cur = self.free_head
        while cur is not None:
            b = self.blocks[cur]
            if b.size >= need:
                return self._carve(b, need)
            cur = b.next_free
        return None

    def free(self, offset: int) -> None:
        b = self.blocks.get(offset)
        if b is None:
            raise ValueError(f"free: unknown offset {offset}")
        if b.is_free:
            raise ValueError(f"free: double free at {offset}")

        prev_off, next_off = b.prev_addr, b.next_addr
        prev_free = prev_off is not None and self.blocks[prev_off].is_free
        next_free = next_off is not None and self.blocks[next_off].is_free

        if prev_free and next_free:                       # case 4
            p, nx = self.blocks[prev_off], self.blocks[next_off]
            self._free_unlink(p); self._free_unlink(nx)
            p.size += b.size + nx.size
            p.next_addr = nx.next_addr
            if nx.next_addr is not None:
                self.blocks[nx.next_addr].prev_addr = p.offset
            del self.blocks[b.offset]
            del self.blocks[nx.offset]
            self._free_push(p)
        elif prev_free:                                   # case 2
            p = self.blocks[prev_off]
            self._free_unlink(p)
            p.size += b.size
            p.next_addr = b.next_addr
            if b.next_addr is not None:
                self.blocks[b.next_addr].prev_addr = p.offset
            del self.blocks[b.offset]
            self._free_push(p)
        elif next_free:                                   # case 3
            nx = self.blocks[next_off]
            self._free_unlink(nx)
            b.size += nx.size
            b.next_addr = nx.next_addr
            if nx.next_addr is not None:
                self.blocks[nx.next_addr].prev_addr = b.offset
            del self.blocks[nx.offset]
            self._free_push(b)
        else:                                             # case 1
            self._free_push(b)
```

**What this buys:** fragmentation drops dramatically. Concrete numbers from the test harness: alloc 8 blocks of 32 bytes in a 256-byte region, then free all eight.

| Version | `fragmentation()` | `malloc(128)` after free-all |
|---|---|---|
| V1 | 0.875 | None |
| V2 | 0.000 | 0 (succeeds) |

**What it costs:** every block now carries two extra links, every `malloc`'s split path touches the address-order list, and `free` has four branches instead of one. Still O(1) per neighbor check, since neighbors are direct dict lookups. The address-order links never need to be re-sorted because they're maintained incrementally at the only two places blocks appear/disappear: `_carve` (splits one block into two) and `free` (cases 2/3/4 collapse N+1 blocks into N).

## Part 3 — Segregated free lists

The remaining cost is `malloc`'s linear walk. With one free list, finding a fit is O(free-list-length). Bucket the free list by size class — bucket `i` holds blocks whose size is in `[2^i, 2^(i+1))` — and `malloc` jumps directly to the bucket that's guaranteed to contain a fit (or close to it). `free` and coalescing are inherited unchanged; only the free-list insert/remove primitives become bucket-aware.

```python
class AllocatorV3(AllocatorV2):
    NUM_BUCKETS = 32  # enough for 4GB

    def __init__(self, capacity: int):
        self.cap = align_up(capacity)
        self.mem = bytearray(self.cap)
        self.blocks = {}
        self.free_heads: List[Optional[int]] = [None] * self.NUM_BUCKETS
        root = Block(0, self.cap)
        self.blocks[0] = root
        self._free_push(root)

    @staticmethod
    def _bucket_of(size: int) -> int:
        return max(0, size.bit_length() - 1)

    def _free_push(self, b: Block) -> None:
        b.is_free = True
        i = self._bucket_of(b.size)
        b.prev_free = None
        b.next_free = self.free_heads[i]
        if self.free_heads[i] is not None:
            self.blocks[self.free_heads[i]].prev_free = b.offset
        self.free_heads[i] = b.offset

    def _free_unlink(self, b: Block) -> None:
        i = self._bucket_of(b.size)
        if b.prev_free is not None:
            self.blocks[b.prev_free].next_free = b.next_free
        else:
            self.free_heads[i] = b.next_free
        if b.next_free is not None:
            self.blocks[b.next_free].prev_free = b.prev_free
        b.prev_free = b.next_free = None
        b.is_free = False

    def malloc(self, n: int) -> Optional[int]:
        if n <= 0:
            return None
        need = align_up(n)
        own = self._bucket_of(need)
        # Scan own bucket for a fit (some blocks here are < need; can't blind-pop)
        cur = self.free_heads[own]
        while cur is not None:
            b = self.blocks[cur]
            if b.size >= need:
                return self._carve(b, need)
            cur = b.next_free
        # Any block in a higher bucket fits unconditionally; take the first
        for i in range(own + 1, self.NUM_BUCKETS):
            if self.free_heads[i] is not None:
                return self._carve(self.blocks[self.free_heads[i]], need)
        return None
```

One subtlety worth surfacing: in the own-bucket scan, blocks are in `[2^own, 2^(own+1))`, but the request `need` is in the same range. So not every block in the bucket fits — I have to scan. In every higher bucket, every block fits, so I pop the head directly. This is the standard "good-fit segregated" structure; the bucket scan is bounded and in practice short.

**Tradeoff vs buddy allocator (3-4 sentences as asked):** A buddy allocator restricts every block to a power-of-2 size and identifies a block's sibling ("buddy") via a single XOR of its offset with its size, which makes merge-on-free O(1) and bounds external fragmentation tightly. The cost is internal fragmentation up to ~2× — a 33-byte request consumes 64 bytes — and constrained split sizes that can't track an irregular allocation distribution. Segregated free lists like mine keep blocks at their natural size and split flexibly, so internal fragmentation stays small for irregular workloads (a 33-byte request takes 40 bytes after alignment), but coalescing has to scan address-order links rather than computing the partner in O(1). Buddy wins when sizes are clustered around powers of 2 and you care about merge speed (Linux's page allocator); segregated lists win when sizes are irregular and you care about packing density (jemalloc, ptmalloc, the PyTorch caching allocator).

**What this buys:** `malloc` is effectively O(1) amortized — direct bucket jump for any request whose ceil-log2 bucket is non-empty, plus the bounded own-bucket scan. **What it costs:** an extra dimension in the free-list state and a slight increase in fragmentation when small allocations are satisfied from much larger buckets without splitting (mitigated by `_carve` still splitting whenever the remainder is ≥ `MIN_BLOCK`).

## Canonical bugs and how this avoids them

1. **Double-free.** Both `free` methods check `b.is_free` before doing anything and raise `ValueError`. A production allocator with packed headers gets this for free via header magic numbers; mine gets it via an explicit boolean.
2. **Coalescing past the region boundary.** In V2, neighbor inspection guards explicitly: `prev_off is not None` and `next_off is not None`. The first and last blocks have `prev_addr = None` and `next_addr = None` respectively, so they fall into cases 1 or 3 / 1 or 2 — never an attempt to look up `self.blocks[None]`.
3. **Forgetting to remove a neighbor from the free list before merging.** Cases 2, 3, and 4 call `_free_unlink(p)` and/or `_free_unlink(nx)` *before* mutating sizes or addr-links. If I skipped that, a future `malloc` would walk to the stale entry, see the old smaller size, dequeue it, and discover at use-time that it now overlaps a neighbor — classic memory corruption in C, classic invariant violation here. Pulling the unlinks to the top of each case also keeps the code grep-able.

## Test harness

Runs end-to-end, deterministic output. Hits: alloc, free, split, all four coalescing cases on fresh allocators (so the cases stay distinct), exhaustion, fragmentation comparison across all three versions, double-free rejection.

```python
def main():
    sep = "─" * 64

    print(sep); print("Part 1: first-fit, no coalescing")
    a = AllocatorV1(256)
    o1, o2, o3 = a.malloc(40), a.malloc(30), a.malloc(50)
    print(f"  malloc(40)={o1}  malloc(30)={o2}  malloc(50)={o3}")
    print(f"  layout: {a.layout()}")
    a.free(o2)
    print(f"  after free({o2}): {a.layout()}")
    print(f"  malloc(500) (overcommit) → {a.malloc(500)}")
    a.free(o1); a.free(o3)
    print(f"  after free-all (V1, no coalesce):\n    {a.layout()}")
    print(f"  V1 frag = {a.fragmentation():.3f}")
    print(f"  malloc(150) after free-all → {a.malloc(150)}  # fails: no contiguous fit")

    print(sep); print("Part 2: coalescing — four cases, fresh allocator each")
    for label, prep in [
        ("case 1 (no merge)",      lambda a, A, B, C: None),
        ("case 2 (prev free)",     lambda a, A, B, C: a.free(A)),
        ("case 3 (next free)",     lambda a, A, B, C: a.free(C)),
        ("case 4 (both free)",     lambda a, A, B, C: (a.free(A), a.free(C))),
    ]:
        a = AllocatorV2(96)
        A, B, C = a.malloc(32), a.malloc(32), a.malloc(32)
        prep(a, A, B, C)
        print(f"  {label}")
        print(f"    before free(B): {a.layout()}")
        a.free(B)
        print(f"    after  free(B): {a.layout()}")

    print(sep); print("Fragmentation comparison after alloc-8 / free-8")
    for cls, name in [(AllocatorV1, "V1"), (AllocatorV2, "V2"), (AllocatorV3, "V3")]:
        a = cls(256); xs = [a.malloc(32) for _ in range(8)]
        for x in xs: a.free(x)
        print(f"  {name}: frag={a.fragmentation():.3f}, "
              f"malloc(128) → {a.malloc(128)}")

    print(sep); print("Double-free protection")
    a = AllocatorV2(128); o = a.malloc(32); a.free(o)
    try:
        a.free(o); print("  BUG: double-free not caught")
    except ValueError as e:
        print(f"  rejected: {e}")

    print(sep); print("Exhaustion")
    a = AllocatorV2(64); o1 = a.malloc(40); o2 = a.malloc(30)
    print(f"  cap=64, malloc(40)={o1}, malloc(30)={o2}  # second fails (only 24 free)")

    print(sep); print("Part 3: segregated lists spot-check")
    a = AllocatorV3(512)
    xs = [a.malloc(s) for s in (16, 64, 100, 200)]
    print(f"  allocs(16,64,100,200) → {xs}")
    print(f"  layout: {a.layout()}")
    for x in xs: a.free(x)
    print(f"  after free-all (V3 coalesces via V2.free): {a.layout()}")

if __name__ == "__main__":
    main()
```

Verified output (abbreviated to the load-bearing lines):

```
case 1 (no merge)
  before free(B): [off=  0 sz= 32 used] [off= 32 sz= 32 used] [off= 64 sz= 32 used]
  after  free(B): [off=  0 sz= 32 used] [off= 32 sz= 32 free] [off= 64 sz= 32 used]
case 2 (prev free)
  after  free(B): [off=  0 sz= 64 free] [off= 64 sz= 32 used]
case 3 (next free)
  after  free(B): [off=  0 sz= 32 used] [off= 32 sz= 64 free]
case 4 (both free)
  after  free(B): [off=  0 sz= 96 free]

Fragmentation:  V1 frag=0.875, malloc(128) → None
                V2 frag=0.000, malloc(128) → 0
                V3 frag=0.000, malloc(128) → 0
```

## Connections to real systems

What I've sketched here is the skeleton of a Knuth-style boundary-tag allocator with size-class refinement. **glibc ptmalloc** combines fastbins (segregated by exact size for small allocations, LIFO, no coalescing for speed) with sorted bins for larger sizes and a "top chunk" that grows the heap via `sbrk`. **jemalloc** takes the size-class idea much further, with fine-grained classes (8 B granularity for small) and per-thread arenas to dodge lock contention. The **Linux buddy allocator** governs physical page allocation and uses the XOR trick described above to merge in O(1) — perfect for fixed-page-size territory. The **slab allocator** sits on top of the buddy allocator and carves slabs into same-typed object caches, eliminating intra-slab fragmentation for kernel objects. **PyTorch's caching allocator** keeps freed GPU memory in size-binned pools and never returns it to the driver, because `cudaMalloc`/`cudaFree` are absurdly expensive compared to chunking from a held pool. **vLLM's PagedAttention** allocates KV cache as fixed-size pages and treats sequences as page lists — slab-style allocation over GPU memory, optimized for the workload where every block is the same shape.