---
title: In-memory spreadsheet
description: In-memory spreadsheet
---

## 1. Problem statement

Implement an in-memory spreadsheet with two operations:

```
set_cell(addr, expr)   # expr is a number or "=A1+B2*3"
get_cell(addr) -> float
```

Four parts, building on each other:

1. **Part 1**: numeric values and `=`-expressions over `+ - * /` and cell refs. Lazy evaluation.
2. **Part 2**: `get_cell` is O(1). `set_cell` proactively updates downstream cells via a dependency graph.
3. **Part 3**: detect cycles. Reject the offending `set_cell` and preserve prior state — no half-updates.
4. **Part 4**: concurrent `set_cell` calls on cells in the same connected component must be safe. Minimal locking.

---

## 2. Clarifying questions

A few things I'd pin down with the interviewer before writing code, with the assumption I'd lock in if no answer comes:

- **Cell address format.** Assume `[A-Z]+[1-9]\d*` — letters then row number, no `$` absolute refs, no sheet qualifiers. `A1`, `AA10` valid; `A0`, `1A`, `A01` not.
- **Numeric type.** Float, not Decimal. The signature returns `float`, so the interviewer has implicitly answered. I'd flag that this loses precision on financial workloads; Decimal is a 1-line swap.
- **Operator precedence.** Standard math: `*`/`/` bind tighter than `+`/`-`, all left-associative. Unary minus tighter than `*`. Parentheses override. I'd bake precedence into a recursive-descent grammar — cleaner and more extensible than shunting-yard for the operator set we have.
- **Missing reference.** `get_cell("Z99")` on an unset cell returns `0.0` (Excel/Sheets behavior). The 0-default is friendlier when sheets are constructed top-down. The alternative is raising `KeyError`.
- **Division by zero.** Raise `ZeroDivisionError`. In a real product this surfaces as `#DIV/0!` and propagates through dependents — out of scope unless asked.
- **Integer vs float division.** Always float: `5/2 == 2.5`.
- **Self-reference.** `set_cell("A1", "=A1+1")` is a 1-cycle. Reject in Part 3.
- **Stale formulas pointing at deleted cells.** No `delete_cell` API was specified, so out of scope.
- **Whitespace.** Tolerated and skipped by the tokenizer.
- **Setting an existing formula cell to a number.** Replaces it, drops its outbound dep edges.

---

## 3. Part 1 — lazy evaluation

### Approach

The simplest correct thing. `set_cell` parses the expression once (catches syntax errors at write time) and stashes either a literal float or the AST. `get_cell` walks the AST, recursively resolving cell refs by re-entering `_eval`. A `stack` set carries the chain of in-flight cells so a cycle raises `RecursionError` instead of blowing the Python stack.

I'm using **recursive descent** rather than shunting-yard. The grammar is small, precedence falls out of the call hierarchy, and the AST is trivial to walk for both evaluation and ref-extraction (which I'll need in Part 2).

### Data model

```
cells: { addr -> ('val', 5.0)            # literal
                | ('expr', <AST node>) } # parsed formula

AST node = ('num', float)
         | ('ref', 'A1')
         | ('neg', node)
         | ('bin', op, left, right)

Example: "=A1+B2*3"
                 ('bin', '+')
                 /          \
           ('ref','A1')   ('bin','*')
                          /         \
                  ('ref','B2')    ('num', 3.0)
```

### Implementation

```python
import re
from typing import Callable, Optional, Union

CELL_RE = re.compile(r"^[A-Z]+[1-9]\d*$")

class Tokenizer:
    def __init__(self, s: str):
        self.tokens: list[tuple[str, str]] = []
        i = 0
        while i < len(s):
            c = s[i]
            if c.isspace():
                i += 1
            elif c in "+-*/()":
                self.tokens.append(("OP", c)); i += 1
            elif c.isdigit() or c == ".":
                j = i
                while j < len(s) and (s[j].isdigit() or s[j] == "."):
                    j += 1
                self.tokens.append(("NUM", s[i:j])); i = j
            elif c.isalpha():
                j = i
                while j < len(s) and s[j].isalpha(): j += 1
                k = j
                while k < len(s) and s[k].isdigit(): k += 1
                if k == j:
                    raise ValueError(f"Invalid cell ref at {i}")
                self.tokens.append(("REF", s[i:k])); i = k
            else:
                raise ValueError(f"Unexpected char {c!r} at {i}")
        self.pos = 0

    def peek(self) -> Optional[tuple[str, str]]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self) -> tuple[str, str]:
        t = self.tokens[self.pos]; self.pos += 1; return t


Node = tuple

class Parser:
    """
    expr   := term (('+'|'-') term)*
    term   := unary (('*'|'/') unary)*
    unary  := '-' unary | atom
    atom   := NUM | REF | '(' expr ')'
    """
    def __init__(self, src: str):
        self.tok = Tokenizer(src)

    def parse(self) -> Node:
        n = self.expr()
        if self.tok.peek() is not None:
            raise ValueError(f"Trailing tokens: {self.tok.peek()}")
        return n

    def expr(self) -> Node:
        left = self.term()
        while self.tok.peek() in (("OP", "+"), ("OP", "-")):
            op = self.tok.consume()[1]
            left = ("bin", op, left, self.term())
        return left

    def term(self) -> Node:
        left = self.unary()
        while self.tok.peek() in (("OP", "*"), ("OP", "/")):
            op = self.tok.consume()[1]
            left = ("bin", op, left, self.unary())
        return left

    def unary(self) -> Node:
        if self.tok.peek() == ("OP", "-"):
            self.tok.consume()
            return ("neg", self.unary())
        return self.atom()

    def atom(self) -> Node:
        t = self.tok.peek()
        if t is None: raise ValueError("Unexpected end of input")
        kind, val = t
        if kind == "NUM":
            self.tok.consume(); return ("num", float(val))
        if kind == "REF":
            self.tok.consume()
            if not CELL_RE.match(val):
                raise ValueError(f"Invalid cell ref: {val}")
            return ("ref", val)
        if t == ("OP", "("):
            self.tok.consume()
            inner = self.expr()
            if self.tok.peek() != ("OP", ")"):
                raise ValueError("Expected ')'")
            self.tok.consume()
            return inner
        raise ValueError(f"Unexpected token: {t}")


def extract_refs(node: Node) -> set[str]:
    k = node[0]
    if k == "num": return set()
    if k == "ref": return {node[1]}
    if k == "neg": return extract_refs(node[1])
    if k == "bin": return extract_refs(node[2]) | extract_refs(node[3])
    raise ValueError(node)


def evaluate_ast(node: Node, resolve: Callable[[str], float]) -> float:
    k = node[0]
    if k == "num": return node[1]
    if k == "ref": return resolve(node[1])
    if k == "neg": return -evaluate_ast(node[1], resolve)
    if k == "bin":
        _, op, l, r = node
        lv = evaluate_ast(l, resolve); rv = evaluate_ast(r, resolve)
        if op == "+": return lv + rv
        if op == "-": return lv - rv
        if op == "*": return lv * rv
        if op == "/":
            if rv == 0.0: raise ZeroDivisionError("division by zero")
            return lv / rv
    raise ValueError(node)


def parse_input(expr) -> tuple[bool, Union[float, Node]]:
    """Returns (is_formula, value). Float for literals, AST for formulas."""
    if isinstance(expr, (int, float)):
        return False, float(expr)
    if isinstance(expr, str):
        s = expr.strip()
        if s.startswith("="):
            return True, Parser(s[1:]).parse()
        try:
            return False, float(s)
        except ValueError:
            raise ValueError(f"Cannot parse {expr!r}; formulas need '='")
    raise TypeError(f"Unsupported expr type: {type(expr)}")


class SpreadsheetV1:
    def __init__(self):
        self.cells: dict[str, tuple] = {}

    def set_cell(self, addr: str, expr) -> None:
        is_formula, val = parse_input(expr)
        self.cells[addr] = ("expr", val) if is_formula else ("val", val)

    def get_cell(self, addr: str) -> float:
        return self._eval(addr, set())

    def _eval(self, addr: str, stack: set[str]) -> float:
        if addr in stack:
            raise RecursionError(f"Cycle through {addr}")
        c = self.cells.get(addr)
        if c is None:
            return 0.0
        kind, payload = c
        if kind == "val":
            return payload
        return evaluate_ast(payload, lambda a: self._eval(a, stack | {addr}))
```

### Tests

```python
def test_v1():
    s = SpreadsheetV1()
    s.set_cell("A1", 5)
    assert s.get_cell("A1") == 5.0                # numeric set/get
    s.set_cell("B1", "=A1+3")
    assert s.get_cell("B1") == 8.0                # simple expr
    s.set_cell("C1", "=B1*2")
    assert s.get_cell("C1") == 16.0               # nested ref
    s.set_cell("D1", "=2+3*4")
    assert s.get_cell("D1") == 14.0               # precedence
    s.set_cell("E1", "=(2+3)*4")
    assert s.get_cell("E1") == 20.0               # parens
    assert s.get_cell("Z99") == 0.0               # missing ref
    s.set_cell("F1", "=-A1+1")
    assert s.get_cell("F1") == -4.0               # unary minus
    s.set_cell("A1", 10)                          # lazy: picks up update
    assert s.get_cell("B1") == 13.0
test_v1()
```

### Complexity

- `set_cell`: **O(|expr|)** for parsing.
- `get_cell`: **O(N)** per call, where N is the size of the transitive dep set, because we re-evaluate the entire subtree every read. No caching.

### Commentary

Lazy eval is the wrong model for a workload with deep chains and frequent reads — every `get_cell` walks the chain again. A 100-cell chain read 1000 times is 100K AST evaluations. It's the right model when writes vastly outnumber reads, or when most cells are never read (sparse worksheets where 99% of `set_cell`s touch cells the user never queries).

Cycle handling here is reactive — we only notice when a `get_cell` recurses into itself. For the lazy version that's acceptable; for an eager system it isn't, because we'd recurse during `set_cell` itself. Part 3 makes cycle detection proactive.

---

## 4. Part 2 — eager propagation, O(1) reads

### What changes

Two structural shifts. First, `set_cell` does the work: parse, recompute, push results into a `value` cache. `get_cell` becomes a dict lookup. Second, we maintain two graphs:

- `forward[A1]` = the set of cells `A1` reads.
- `reverse[A1]` = the set of cells that read `A1`.

The reverse graph makes propagation efficient — when `A1` changes, walk reverse edges to find dependents in O(downstream).

### Forward vs reverse

```
Cells:   A1 = 5
         B1 = =A1+3        (reads A1)
         C1 = =B1*2        (reads B1)
         D1 = =A1+B1       (reads A1 and B1)

forward (deps):           reverse (dependents):
  A1 -> {}                  A1 -> {B1, D1}
  B1 -> {A1}                B1 -> {C1, D1}
  C1 -> {B1}                C1 -> {}
  D1 -> {A1, B1}            D1 -> {}

When A1 changes, walk reverse from A1 to find {B1, D1, C1}.
Recompute in topological order (forward edges within affected
set): A1 -> B1 -> C1, then D1 (depends on A1 and B1).
```

### Implementation

```python
# Reuses Tokenizer, Parser, extract_refs, evaluate_ast, parse_input from Part 1.
from collections import defaultdict, deque

class SpreadsheetV2:
    def __init__(self):
        self.cells: dict[str, tuple] = {}
        self.value: dict[str, float] = {}
        self.forward: dict[str, set[str]] = defaultdict(set)
        self.reverse: dict[str, set[str]] = defaultdict(set)

    def set_cell(self, addr: str, expr) -> None:
        is_formula, val = parse_input(expr)
        new_deps = extract_refs(val) if is_formula else set()
        old_deps = self.forward.get(addr, set())
        for d in old_deps - new_deps:
            self.reverse[d].discard(addr)
        for d in new_deps - old_deps:
            self.reverse[d].add(addr)
        self.forward[addr] = set(new_deps)
        self.cells[addr] = ("expr", val) if is_formula else ("val", val)
        self._propagate(addr)

    def get_cell(self, addr: str) -> float:
        return self.value.get(addr, 0.0)

    def _propagate(self, root: str) -> None:
        # Collect reachable dependents via reverse edges.
        affected: set[str] = set()
        stack = [root]
        while stack:
            u = stack.pop()
            if u in affected:
                continue
            affected.add(u)
            stack.extend(self.reverse.get(u, ()))
        # Topo sort restricted to affected, using forward edges within it.
        in_count = {u: 0 for u in affected}
        for u in affected:
            for d in self.forward.get(u, ()):
                if d in affected:
                    in_count[u] += 1
        ready = deque(u for u, c in in_count.items() if c == 0)
        order: list[str] = []
        while ready:
            u = ready.popleft()
            order.append(u)
            for v in self.reverse.get(u, ()):
                if v in affected:
                    in_count[v] -= 1
                    if in_count[v] == 0:
                        ready.append(v)
        if len(order) != len(affected):
            raise RuntimeError("Cycle in propagation (caught earlier in P3)")
        for u in order:
            self._recompute(u)

    def _recompute(self, addr: str) -> None:
        c = self.cells.get(addr)
        if c is None:
            self.value[addr] = 0.0; return
        kind, payload = c
        if kind == "val":
            self.value[addr] = payload
        else:
            self.value[addr] = evaluate_ast(
                payload, lambda a: self.value.get(a, 0.0))
```

### Tests

```python
def test_v2_basic():
    s = SpreadsheetV2()
    s.set_cell("A1", 5); s.set_cell("B1", "=A1+3"); s.set_cell("C1", "=B1*2")
    assert s.get_cell("C1") == 16.0
    s.set_cell("A1", 10)                           # eager propagation
    assert (s.get_cell("A1"), s.get_cell("B1"), s.get_cell("C1")) == (10.0, 13.0, 26.0)
    # P1 regressions:
    s.set_cell("D1", "=2+3*4");   assert s.get_cell("D1") == 14.0
    s.set_cell("E1", "=(2+3)*4"); assert s.get_cell("E1") == 20.0
    assert s.get_cell("Z99") == 0.0

def test_v2_o1_reads():
    """Build a deep chain; prove get_cell never recomputes."""
    calls = [0]
    class Counting(SpreadsheetV2):
        def _recompute(self, addr):
            calls[0] += 1; super()._recompute(addr)
    s = Counting()
    s.set_cell("A1", 1)
    for i in range(2, 51):
        s.set_cell(f"A{i}", f"=A{i-1}+1")
    before = calls[0]
    for _ in range(1000):
        s.get_cell("A50")
    assert calls[0] == before, f"{calls[0]-before} unexpected recomputes"
    assert s.get_cell("A50") == 50.0

test_v2_basic(); test_v2_o1_reads()
```

### Complexity

- `set_cell`: **O(D + |expr|)**, where D is the count of transitive dependents (cells reachable via reverse edges from `addr`).
- `get_cell`: **O(1)**.

### Commentary

Eager wins when reads dominate writes and we have a measurable read-latency budget. It loses when most writes touch cells nobody ever reads — we compute values that get thrown away. The crossover depends on the read:write ratio and the average dependent fan-out. In a UI spreadsheet a single user write triggers maybe a screen full of recomputes; in a write-heavy ETL job it could be a million dependent cells. Lazy defers cost; eager amortizes it.

A subtler tradeoff: eager makes write latency proportional to dependent count. If a hot cell has 10⁵ dependents, every write to it is a 10⁵-node recompute even if only a handful of cells are read. The escape hatch is a "dirty-flag" hybrid — propagate dirty bits eagerly, recompute lazily on read — but that gives back O(1) reads. Pure eager is the right baseline; you'd add the hybrid only when measurement says so.

---

## 5. Part 3 — cycle detection + transactional commit

### Algorithm

Three-color DFS over the **hypothetical** forward graph — the graph that would exist if we committed this `set_cell`. WHITE = unseen, GRAY = on the current DFS stack, BLACK = fully explored. A back-edge to a GRAY node is a cycle. The DFS starts at `addr`: we only need cycles reachable from the changed cell, because the graph was acyclic before this write — any new cycle must pass through `addr`.

Why DFS-coloring over alternatives:

- **Tarjan's SCC** finds *all* SCCs in O(V+E). Overkill — we need yes/no for one node and we already know the prior graph was acyclic. Same asymptotic cost, more code.
- **Incremental cycle detection** (Bender, Fineman, Gilbert et al. — maintain a topological order, update it under edge insertion) gets to amortized polylog per insert. Right answer for sustained 10⁴+ writes/sec; massive complexity overhead for a spreadsheet.

DFS is O(reachable subgraph from `addr`), runs once per `set_cell`, and is 20 lines.

### Cycle detection on a small example

```
Existing graph:                Proposed: set_cell("B1", "=A1")
  A1 = =B1+1                     A1 = =B1+1     (existing)
  B1 = 2                         B1 = =A1       (proposed; new_deps={A1})

DFS from B1 over the overlay (new_deps for B1, forward elsewhere):
  visit B1     -> color[B1] = GRAY
  succs(B1) = new_deps = {A1}
    visit A1   -> color[A1] = GRAY
    succs(A1) = forward[A1] = {B1}
      look up color[B1] = GRAY  ===>  back-edge: CYCLE
  REJECT. No state mutated.
```

### Transactional pattern

**Validate-before-mutate.** The cycle check uses the current `forward` graph plus the new edge set as a hypothetical overlay (`succs(u)` returns `new_deps` only when `u == addr`). Nothing on `self` is touched until the cycle check passes. Only then do we update `forward`, `reverse`, `cells`, and propagate. If validation fails we just `raise`; the spreadsheet state is bit-identical to before the call.

The shadow-state alternative (copy everything, mutate copies, swap atomically) is what you reach for when validation needs to *observe* the post-mutation state. Cycle detection doesn't — it observes the overlay, not real state — so shadow-state would be cost without benefit.

### Implementation

```python
# Reuses Part 1 utilities and the V2 propagation machinery.

class SpreadsheetV3(SpreadsheetV2):
    def set_cell(self, addr: str, expr) -> None:
        is_formula, val = parse_input(expr)
        new_deps = extract_refs(val) if is_formula else set()
        if self._would_cycle(addr, new_deps):
            raise ValueError(f"Cycle detected: {addr} = {expr}")
        # No mutation has happened yet -> commit is atomic.
        old_deps = self.forward.get(addr, set())
        for d in old_deps - new_deps:
            self.reverse[d].discard(addr)
        for d in new_deps - old_deps:
            self.reverse[d].add(addr)
        self.forward[addr] = set(new_deps)
        self.cells[addr] = ("expr", val) if is_formula else ("val", val)
        self._propagate(addr)

    def _would_cycle(self, addr: str, new_deps: set[str]) -> bool:
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {}

        def succs(u: str) -> set[str]:
            return new_deps if u == addr else self.forward.get(u, set())

        def dfs(u: str) -> bool:
            color[u] = GRAY
            for v in succs(u):
                c = color.get(v, WHITE)
                if c == GRAY:                # back-edge
                    return True
                if c == WHITE and dfs(v):
                    return True
            color[u] = BLACK
            return False

        return dfs(addr)
```

### Tests

```python
def test_v3():
    s = SpreadsheetV3()
    s.set_cell("A1", 5); s.set_cell("B1", "=A1+1")
    assert s.get_cell("B1") == 6.0

    # Snapshot pre-rejection state.
    snap_cells = dict(s.cells)
    snap_value = dict(s.value)
    snap_fwd = {k: set(v) for k, v in s.forward.items() if v}
    snap_rev = {k: set(v) for k, v in s.reverse.items() if v}

    # 2-cycle: existing B1 = A1+1, proposed A1 = B1+1 -> reject
    try:
        s.set_cell("A1", "=B1+1"); assert False, "should raise"
    except ValueError:
        pass
    assert dict(s.cells) == snap_cells
    assert dict(s.value) == snap_value
    assert {k: set(v) for k, v in s.forward.items() if v} == snap_fwd
    assert {k: set(v) for k, v in s.reverse.items() if v} == snap_rev

    # Self-reference rejected; cell stays unset.
    try:
        s.set_cell("C1", "=C1+1"); assert False
    except ValueError:
        pass
    assert "C1" not in s.cells

    # Update that touches the dep graph but introduces no cycle: succeeds.
    s.set_cell("C1", "=A1*2")
    assert s.get_cell("C1") == 10.0

    # 3-cycle.
    s2 = SpreadsheetV3()
    s2.set_cell("A1", "=B1"); s2.set_cell("B1", "=C1")
    try:
        s2.set_cell("C1", "=A1"); assert False
    except ValueError:
        pass

test_v3()
```

### Commentary — setup for Part 4

Validate-before-mutate is correct under single-threaded execution because nothing can sneak in between check and commit. Under concurrency that property dies: two threads can both run cycle checks against pre-state, both pass, both commit, and the resulting graph has a cycle — neither write saw the other's edges. The naive "check then mutate" is a classic TOCTOU bug. Part 4 closes this by making the check + commit atomic under a write lock.

---

## 6. Part 4 — concurrency

### Threat model

1. **Two writers, same cell.** Last-writer-wins is acceptable, but the dep-graph mutation is multi-step (`forward[addr]`, `reverse[d]` for each `d`, then `cells[addr]`). Interleaving leaves dangling reverse edges or stale `cells` entries.
2. **Two writers, same component.** Propagation reads the `value` cache. If writer B mutates `value` mid-way through writer A's propagation, A produces inconsistent downstream values.
3. **Writer + reader.** A multi-cell snapshot via two separate `get_cell` calls can see (oldA, newB) where invariants like `B = A*2` are temporarily broken.
4. **Cycle-check vs concurrent edit.** Two cycle-introducing writes can both pass independent checks against pre-state, both commit, and produce a cycle — exactly the TOCTOU from above.

### Locking strategies

| Strategy | Correctness | Throughput | Complexity | When to use |
|---|---|---|---|---|
| Single mutex | ✓ | Low (no read parallelism) | Trivial | Prototyping |
| **RWLock (chosen)** | ✓ | Parallel reads; writes serialize | Low | Read-heavy, occasional writes |
| Per-cell lock | Only with global lock-ordering — deadlock risk otherwise | High when components disjoint | High; must order locks across all touched cells | Writes truly independent across regions |
| Per-component lock | ✓ if you handle component merges | High when components stable | Components merge on edge insertion → need union-find + lock migration | Stable component partitioning |
| Optimistic (version + retry) | ✓ with retry loop | Highest under low contention | High; livelock guards, retry cost | Mostly uncontended writes |

### Justification

The problem says read-heavy, occasional bursty writes, components usually small. A **writer-preferring RWLock** fits: parallel reads are the common case and writes serialize but are rare. Per-component locking is technically optimal but components dynamically merge — `set_cell("A1", "=B1")` merges A1's component with B1's — so we'd need lock-migration machinery and cross-component lock ordering to avoid deadlock. Not worth the complexity for the stated workload. Ship the RWLock, instrument writer-wait time, and revisit if writes become the bottleneck.

Writer-preferring (vs reader-preferring) prevents writer starvation under sustained read load — important for a spreadsheet where read rate may be high enough to indefinitely block a queued writer.

### Chosen strategy

```
            +----------------------------------+
            |          SpreadsheetV4           |
            |                                  |
   parse -->|  Parser (pure, runs OUTSIDE the  |
            |          lock - no shared state) |
            |                                  |
            |             |                    |
            |             v                    |
   write -->| acquire_write -> validate (cycle)|
            |              -> commit          |
            |              -> propagate       |
            |              -> release_write   |
            |                                  |
   read --->| acquire_read  -> read value/snap |
            |              -> release_read    |
            |                                  |
            |  +----------------------------+  |
            |  | RWLock (writer-preferring) |  |
            |  +----------------------------+  |
            |  | cells, value, forward, rev |  |
            |  +----------------------------+  |
            +----------------------------------+
```

### Implementation

```python
import threading

class RWLock:
    """Writer-preferring reader-writer lock."""
    def __init__(self):
        self._cond = threading.Condition()
        self._readers = 0
        self._writers_waiting = 0
        self._writer_active = False

    def acquire_read(self) -> None:
        with self._cond:
            while self._writer_active or self._writers_waiting > 0:
                self._cond.wait()
            self._readers += 1

    def release_read(self) -> None:
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def acquire_write(self) -> None:
        with self._cond:
            self._writers_waiting += 1
            try:
                while self._writer_active or self._readers > 0:
                    self._cond.wait()
                self._writer_active = True
            finally:
                self._writers_waiting -= 1

    def release_write(self) -> None:
        with self._cond:
            self._writer_active = False
            self._cond.notify_all()


class _ReadGuard:
    def __init__(self, lock): self.lock = lock
    def __enter__(self): self.lock.acquire_read()
    def __exit__(self, *a): self.lock.release_read()

class _WriteGuard:
    def __init__(self, lock): self.lock = lock
    def __enter__(self): self.lock.acquire_write()
    def __exit__(self, *a): self.lock.release_write()


class SpreadsheetV4(SpreadsheetV3):
    def __init__(self):
        super().__init__()
        self._lock = RWLock()

    def set_cell(self, addr: str, expr) -> None:
        # Parse outside the lock - it's pure, no shared state.
        is_formula, val = parse_input(expr)
        new_deps = extract_refs(val) if is_formula else set()
        with _WriteGuard(self._lock):
            if self._would_cycle(addr, new_deps):
                raise ValueError(f"Cycle detected: {addr} = {expr}")
            old_deps = self.forward.get(addr, set())
            for d in old_deps - new_deps:
                self.reverse[d].discard(addr)
            for d in new_deps - old_deps:
                self.reverse[d].add(addr)
            self.forward[addr] = set(new_deps)
            self.cells[addr] = ("expr", val) if is_formula else ("val", val)
            self._propagate(addr)

    def get_cell(self, addr: str) -> float:
        with _ReadGuard(self._lock):
            return self.value.get(addr, 0.0)

    def get_snapshot(self, addrs: list[str]) -> dict[str, float]:
        """Atomic multi-cell read - guarantees a coherent view."""
        with _ReadGuard(self._lock):
            return {a: self.value.get(a, 0.0) for a in addrs}
```

### Deterministic concurrency tests

```python
def test_v4_writers_serialize():
    """Two writers race set_cell on the same component;
    final state must reflect exactly one of the orderings."""
    s = SpreadsheetV4()
    s.set_cell("A1", 0); s.set_cell("B1", "=A1+1"); s.set_cell("C1", "=B1*2")
    barrier = threading.Barrier(2)
    def w(v):
        barrier.wait()
        s.set_cell("A1", v)
    t1 = threading.Thread(target=w, args=(10,))
    t2 = threading.Thread(target=w, args=(20,))
    t1.start(); t2.start(); t1.join(); t2.join()
    a = s.get_cell("A1")
    assert a in (10.0, 20.0)
    assert s.get_cell("B1") == a + 1
    assert s.get_cell("C1") == (a + 1) * 2

def test_v4_consistent_reader():
    """Reader's snapshot must always satisfy B = A*2,
    even while a writer hammers A."""
    s = SpreadsheetV4()
    s.set_cell("A1", 0); s.set_cell("B1", "=A1*2")
    stop = threading.Event(); bad = []
    def reader():
        while not stop.is_set():
            snap = s.get_snapshot(["A1", "B1"])
            if snap["B1"] != snap["A1"] * 2:
                bad.append(snap); return
    def writer():
        for i in range(500):
            s.set_cell("A1", i)
    rt = threading.Thread(target=reader); wt = threading.Thread(target=writer)
    rt.start(); wt.start(); wt.join(); stop.set(); rt.join()
    assert not bad, f"inconsistent snapshot: {bad[0]}"

def test_v4_cycle_race():
    """Two writes that are individually fine but together form a cycle.
    Exactly one must succeed, regardless of scheduling."""
    s = SpreadsheetV4()
    s.set_cell("A1", 1); s.set_cell("B1", 2)
    barrier = threading.Barrier(2); successes = []
    lock = threading.Lock()
    def write(addr, expr):
        barrier.wait()
        try:
            s.set_cell(addr, expr)
            with lock: successes.append(addr)
        except ValueError:
            pass
    t1 = threading.Thread(target=write, args=("A1", "=B1+1"))
    t2 = threading.Thread(target=write, args=("B1", "=A1+1"))
    t1.start(); t2.start(); t1.join(); t2.join()
    assert len(successes) == 1, f"expected 1 success, got {successes}"

test_v4_writers_serialize()
test_v4_consistent_reader()
for _ in range(50):                   # flush scheduling-dependent bugs
    test_v4_cycle_race()
```

I ran the cycle-race test 50× during development; it passes deterministically because the write lock makes "validate + commit + propagate" atomic. The first writer to acquire the lock commits its (cycle-free) edge; the second writer's cycle check now sees the first's edge and rejects.

### What this implementation still doesn't address

- **Multi-process / distributed.** Single-process only. A real product would need a shared store (Redis, Spanner) and consensus or last-writer-wins per cell with versioning. The dep graph itself becomes the hard part — graph mutations need to be serializable across nodes.
- **Persistence and recovery.** No write-ahead log, no snapshots. Crash loses everything.
- **Undo/redo.** Would need a write log capturing prior `cells[addr]` and `forward[addr]` per mutation.
- **Range references.** `SUM(A1:A10)` requires the dep graph to track ranges (or expand them lazily and re-resolve on row insert/delete).
- **Error cells.** Currently we raise on `#DIV/0!`. A real spreadsheet has error values that propagate through formulas and render as `#DIV/0!`, `#REF!`, `#VALUE!`. That's a value-domain change (floats become a sum type), not a structural one.
- **Incremental cycle detection.** Every `set_cell` does a fresh DFS from `addr`. For sustained 10⁴+ writes/sec, switch to Bender et al.'s ordered-list maintenance algorithm.

---

## 7. Final architecture

```
                    set_cell(addr, expr)
                          |
                          v
                  +----------------+
                  |   Tokenizer    |   pure
                  |   Parser       |   (no lock, no shared state)
                  +----------------+
                          |
                          v   AST + extracted refs
                  +----------------+
                  | Lock Manager   |
                  | (RWLock,       |
                  |  writer-pref)  |
                  +----------------+
                       /        \
       acquire_write  /          \  acquire_read
                     v            v
        +-----------------+   +-----------------+
        | Cycle Detector  |   |   Cell Store    |
        | 3-color DFS     |   |   value cache   |--> get_cell
        | over overlay    |   |   (O(1) lookup) |
        +-----------------+   +-----------------+
                |                     ^
                | (validate ok)       |
                v                     |
        +---------------------+       |
        | Mutator             |       |
        |  - update forward   |       |
        |  - update reverse   |       |
        |  - update cells     |       |
        +---------------------+       |
                |                     |
                v                     |
        +---------------------+       |
        | Propagator          |       |
        |  reverse-DFS to     |       |
        |  collect affected,  |-------+
        |  topo-recompute     |
        +---------------------+

        +---------------+  +---------------+
        | forward graph |  | reverse graph |
        | addr->{deps}  |  | addr->{users} |
        +---------------+  +---------------+
        (both protected by the same RWLock as cell/value)
```

---

## 8. What I'd do with more time

- **Range references** (`SUM(A1:A10)`, `AVERAGE`). Adds aggregate-function nodes to the AST and range-edges to the dep graph; the interesting design question is whether a range edge expands to N cell edges (simple, expensive on insert) or stays compact (clever, requires range-overlap queries on every `set_cell`).
- **Error propagation** (`#REF!`, `#DIV/0!`, `#VALUE!`). Promote `value` from `float` to a sum type `float | ErrorCell`; arithmetic short-circuits errors. Cleaner than raising mid-propagation.
- **Persistence and recovery.** Append-only write log keyed by logical timestamp, periodic snapshots, replay on startup. Same shape as a database WAL.
- **Undo/redo.** Snapshot prior `cells[addr]`/`forward[addr]` per `set_cell` into a bounded ring buffer; redo is symmetric. Falls naturally out of the WAL if you have one.
- **Distributed / sharded cells.** Hash addr to a shard, cross-shard formulas use a 2PC-like commit for the dep-graph update. Cycle detection becomes a distributed reachability problem — at that point you're building a real DAG database.