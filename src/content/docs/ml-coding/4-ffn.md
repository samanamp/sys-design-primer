---
title: silu FFN/MLP
description: silu FFN/MLP
---

full FFN is usually:

[
\text{FFN}(x) =
\left(\text{SiLU}(xW_{\text{gate}}) \odot xW_{\text{up}}\right) W_{\text{down}}
]

In code shape terms:
```
gate = x @ W_gate   # [B, hidden]
up   = x @ W_up     # [B, hidden]

h = silu(gate) * up
out = h @ W_down
```
For LLaMA/Gemma-style MLPs, this last version is the common one.
