---
title: "count min sketch"
description: "count min sketch"
---

```py
import random


class CountMinSketch:
    def __init__(self, width=10_000, depth=5):
        self.w = width
        self.d = depth
        self.table = [[0] * width for _ in range(depth)]
        self.seeds = [random.randint(1, 10**9) for _ in range(depth)]

    def _hash(self, item, i):
        return hash((item, self.seeds[i])) % self.w

    def add(self, item, count=1):
        for i in range(self.d):
            j = self._hash(item, i)
            self.table[i][j] += count

    def estimate(self, item):
        return min(
            self.table[i][self._hash(item, i)]
            for i in range(self.d)
        )
```