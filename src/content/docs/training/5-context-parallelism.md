---
title: Context Parallelism (CP)
description: Context Parallelism (CP)
---

Where MLA, SWA+sinks, and dynamic sparse all attack *per-device* attention cost, **CP** shards the sequence dimension itself across `N` devices. At `s ≥ 128K`, even FlashAttention on H100 runs out of HBM for activations and wallclock becomes prohibitive. CP gives near-linear memory and time scaling. The catch: attention is non-local in `s`, so devices must exchange K/V to compute the full mixing matrix.

## What CP shards vs what TP/DP don't

```
Tensor (s × d):

DP:   each device holds full seq, different batches    — useless for one long seq
TP:   each device holds full seq, sharded heads/d_h    — activation mem still O(s)
CP:   each device holds s/N tokens, full d             — activation mem O(s/N) ◄
```

For long-context training, activation memory binds first. Only CP attacks it directly.

## Ring Attention — the production answer

Pass K/V chunks around a ring while computing partial attention with online softmax — FlashAttention's trick, distributed.

```
   ┌──► Device 0: Q_0, K_0, V_0 ──┐
   │                              │
   │    Device 1: Q_1, K_1, V_1   │       Q stays put.
   │                              │       K/V chunks rotate
   │    Device 2: Q_2, K_2, V_2   │       through N steps.
   │                              │
   └────────  Device N-1  ◄───────┘
```

Per-device loop (local rank `i`, `Q_i` fixed, K/V buffers rotate):

```
for t in 0..N-1:
    K_t, V_t = recv_from_prev()            # async, overlaps below
    S = Q_i @ K_t.T / √d
    m_new = max(m, S.max(-1))
    O   = O · exp(m - m_new) + exp(S - m_new) @ V_t
    ℓ   = ℓ · exp(m - m_new) + exp(S - m_new).sum(-1)
    m   = m_new
    send_to_next(K_t, V_t)
O = O / ℓ
```

After `N` steps, `O_i` is correct for device `i`'s `s/N` queries. The rest of the layer (FFN, output projection) is local — no further CP comm.

(*All-gather CP* — gather full K,V then run local attention — is simpler but defeats the memory purpose. Used only when comm is free and the problem is pure compute.)

## Causal mask load balance

Naïve contiguous sharding wrecks balance under causal masks: device 0's early tokens attend to ~1/N of keys; device `N-1`'s attend to all. Up to `N×` imbalance.

**Zigzag sharding** (Llama 3, Megatron-CP): device `i` gets tokens `{i, 2N−1−i, 2N+i, 4N−1−i, ...}`. Every device gets a mixed early/late distribution, equalizing work to within ~5%.

```
N=4, s=16, contiguous:    [0 1 2 3] [4 5 6 7] [8 9 10 11] [12 13 14 15]
                          ▲ light                              ▲ heavy

N=4, s=16, zigzag:        [0 7 8 15] [1 6 9 14] [2 5 10 13] [3 4 11 12]
                          ▲ each rank gets a mix ▲
```

## Memory & bandwidth

Per device, `N`-way CP, seq `s`, hidden `d`:

| Quantity            | No CP        | Ring CP                              |
|---|---|---|
| Activation memory   | `O(s·d)`     | `O(s/N · d)`                         |
| Attention FLOPs     | `O(s²·d)`    | `O(s²·d / N)`                        |
| P2P comm per layer  | 0            | `2·s·d` (independent of N)           |
| Comm / compute      | —            | `O(1/d_h)` per step — overlap-friendly |

Comm volume *per device* is independent of `N` — each device forwards the same total K/V volume regardless of ring size — so CP scales near-linearly until the overlap budget runs out. Llama 3 405B used `CP=16` for the 128K phase at near-ideal weak scaling.

## CP vs Ulysses (DeepSpeed)

**Ulysses** does two all-to-alls per attention, reshuffling so each device holds full `s` for a head subset. Lower latency but caps at `N ≤ N_heads`. Ring scales further; Ulysses wins when `N` is small and `N_heads` is large. Production trend is Ring for >8-way.

## Trade-offs

CP composes with TP/PP/DP in the standard 4D parallelism mesh — typical layout is `(DP, PP, TP, CP)` with CP innermost over NVLink/IB. Non-obvious gotcha: K/V projection weight gradients aggregate across CP ranks (they were logically shared), so backward needs an all-reduce on those grads — usually folded into the DP sync.

CP barely helps short-context inference (comm overhead dominates) but is increasingly standard for long-context serving: prefill parallelizes cleanly, decode KV cache shards naturally across CP ranks. Combined with MLA (less per-token cache) or dynamic sparse attention (fewer tokens touched per query), CP is what makes million-token contexts actually trainable at current parameter counts.