"""Generate fault-injected torch.profiler traces (CPU) for trace-reading labs.
Each fault is real runtime behavior, not synthetic JSON — ground truth is known
because we injected it."""
import torch, time, gc, gzip, shutil, os, json

OUT = "traces_out"; os.makedirs(OUT, exist_ok=True)
D = 512

def model_step(x, w1, w2, fused=True):
    if fused:
        return torch.relu(x @ w1) @ w2
    # op-storm variant: same math shredded into many tiny ops
    h = x @ w1
    for _ in range(60):
        h = h + 0.0  # tiny elementwise ops
        h = h * 1.0
    h = torch.relu(h)
    o = h @ w2
    for _ in range(60):
        o = o - 0.0
    return o

def run(name, steps=8, loader_delay=0.0, fused=True, gc_churn=False, periodic_stall=0.0):
    torch.manual_seed(0)
    w1 = torch.randn(D, D); w2 = torch.randn(D, D)
    garbage = []
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=steps),
        on_trace_ready=lambda p: p.export_chrome_trace(f"{OUT}/{name}.json"),
    ) as prof:
        for step in range(steps + 2):
            with torch.profiler.record_function("data_loading"):
                if loader_delay: time.sleep(loader_delay)   # starved input pipeline
                x = torch.randn(64, D)
            with torch.profiler.record_function("forward"):
                y = model_step(x, w1, w2, fused=fused)
                loss = y.square().mean()
            with torch.profiler.record_function("metrics"):
                if periodic_stall and step % 3 == 2:
                    time.sleep(periodic_stall)              # periodic host stall (bad logging/lock)
                _ = loss.item()
            if gc_churn:
                garbage.append([list(range(3000)) for _ in range(40)])
                if step % 2 == 1:
                    with torch.profiler.record_function("mystery"):
                        gc.collect()                        # forced GC pause
            prof.step()
    with open(f"{OUT}/{name}.json", "rb") as fi, gzip.open(f"{OUT}/{name}.json.gz", "wb") as fo:
        shutil.copyfileobj(fi, fo)
    os.remove(f"{OUT}/{name}.json")
    print(name, "done")

run("lab-healthy")
run("lab-starved", loader_delay=0.030)
run("lab-opstorm", fused=False)
run("lab-hoststall", periodic_stall=0.060, gc_churn=True)
