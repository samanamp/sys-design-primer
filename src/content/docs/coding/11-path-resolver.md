---
title: Path resolver
description: Path resolver
---

**Prompt:** Given a current working directory and a path (absolute or relative, with `.`, `..`, possibly symlinks), return the resolved absolute path.

**What's tested:** string handling, edge cases, and whether you handle symlinks correctly.

**The mechanical part:** split path on `/`. If absolute, start from `/`; else start from `cwd`. For each component: skip empty and `.`, pop for `..`, push otherwise. Join the result with `/`. Easy.

**The actual interview, which is symlinks.** A symlink in the middle of the path *replaces the resolution stack* at that point, and the replacement itself may contain `..` or further symlinks. You cannot just resolve the final string and then deref the final symlink — that's wrong. Real `realpath()` resolves *component by component*.

```
def resolve(path, cwd, fs):
    parts = (path if path.startswith("/") else cwd + "/" + path).split("/")
    stack = []
    seen = 0
    for part in parts:
        if part in ("", "."): continue
        if part == "..":
            if stack: stack.pop()
            continue
        stack.append(part)
        node = fs.lookup("/".join(stack))
        if node.is_symlink:
            seen += 1
            if seen > 40: raise TooManySymlinks    # Linux limit
            target = node.link_target
            # symlink replaces last component; resolve recursively
            stack.pop()
            stack = resolve(target, "/" + "/".join(stack), fs).split("/")[1:]
    return "/" + "/".join(stack)
```

**Staff signal moves:**
- Volunteer the symlink-loop protection (Linux caps at 40). Without it, `a → b, b → a` blows the stack.
- Mention `..` after a symlink is genuinely weird — POSIX says `..` is the parent of the *resolved* path, but shells (bash) often track it lexically. Different answers are defensible; state which you're implementing and why.
- Trailing slash semantics: `foo/` only resolves if `foo` is a directory. Real `realpath` enforces this.
- Don't reach for `os.path.realpath` even as a joke. They're testing whether you can *be* `realpath`.