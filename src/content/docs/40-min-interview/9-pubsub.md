---
title: "Real-time pub/sub for collaborative apps"
description: "Real-time pub/sub for collaborative apps"
---

“Design the real-time messaging layer for a collaborative app — think Figma-style multiplayer presence and document edits. Sub-100ms propagation, 100K concurrent rooms, rooms ranging from 2 to 1000 participants.”

---

## 1. Reframing

The framing in the prompt collapses three distinct problems into one wire. I'm pulling them apart before any boxes go on the page.

**Presence and document edits are not the same problem.** Presence is high-frequency (30–60 Hz/active user), ephemeral, lossy-tolerant, ordering-irrelevant — last-writer-wins on a cursor is fine. Edits are low-frequency (1–10 Hz/active editor), durable, causally-ordered, and convergent — losing one is data loss and a support ticket. They share the WebSocket but should share almost nothing else: separate queues, separate SLOs, separate backpressure rules, separate persistence semantics. **[STAFF SIGNAL: presence-vs-edit separation]**

**Long-lived stateful connections are the central architectural pressure**, not the WebSocket itself. A WebSocket pins a client to one gateway for minutes-to-hours. That single fact rules out "stateless load balance everything" thinking and forces decisions about: where a client lands on connect; what happens to 50K connections when that gateway dies; how an edit from client A on G1 reaches client B on G2 in <20ms; how a hot 1000-person room avoids hot-spotting whichever gateway hosts its members. Every other choice cascades from this. **[STAFF SIGNAL: long-lived-stateful-connections]**

**Room size variance from 2 to 1000 is not a parameter — it's two architectures.** A 2-person room can run on the dumbest possible direct-pipe design. A 1000-person room with 100 active users hits 3M outbound presence messages/sec on its own and breaks the dumb design at the seams. The system needs an explicit mode switch around ~50 participants where fan-out shifts from per-recipient to hierarchical/batched and presence sampling rate degrades by viewer count. **[STAFF SIGNAL: room-size-spectrum]**

These three reframings drive the rest of the design. Pub/sub choice, sharding, reconnect — all downstream.

## 2. Scoping

- **Collaboration type:** Figma-style. High-frequency cursor + viewport presence; structured operations on a CRDT document (Yjs-flavored). Not Discord chat (append-only is a much easier problem). Not Google-Docs OT (rejected in §9).
- **Persistence boundary:** the messaging layer is **not** the source of truth. A separate Document Service owns the durable CRDT log + periodic snapshots. The messaging layer propagates ops in real time and async-feeds the Document Service. **[STAFF SIGNAL: persistence-boundary discipline]**
- **Regional scope:** same-region collaboration is the SLO target. Sub-100ms p99 cross-region is mostly fiction at TCP+TLS+app-layer hops; pushed back on in §15. Cross-region is a degraded mode, not a primary requirement.
- **Room-size distribution:** Pareto. Median ~3–5 participants; p95 ~30; p99.9 ~1000. Mean ~8. Total connections ~800K.
- **Out of scope:** voice/video, the CRDT engine itself, billing, the editor UI. Auth assumed; touched only where it affects the messaging path. **[STAFF SIGNAL: scope negotiation]**

## 3. Capacity Math

```
Quantity                       Estimate              Driver
-----------------------------  --------------------  ----------------------------
Concurrent rooms               100K                  given
Avg participants/room          ~8                    Pareto, mean
Total connections              ~800K                 100K × 8
Connections/gateway (tuned)    50K–100K              epoll, tuned kernel, ~30KB/conn
Gateway count (steady)         12–16                 800K / 60K + headroom
Gateway count (with redundancy) 24–32                2× for AZ failure absorption
Connection memory              ~24 GB total          800K × 30KB
Room state memory              ~100 GB total         100K × ~1MB (membership+backlog)
Presence msg rate (ingress)    ~5M/sec               dominated by big rooms
Edit msg rate (ingress)        ~50K/sec              ~0.5 Hz/connection active
Egress msg rate (peak)         ~200M/sec             fan-out amplification ~40×
Egress bandwidth               ~10–20 GB/sec         ~80 bytes/msg post-batching
```

**Latency budget (sub-100ms p99 same-region):**

```
Hop                            Budget    Notes
-----------------------------  --------  ------------------------------
Client → gateway (TLS+RTT)     20–30ms   p99 same-region wifi/cell
Gateway → room server          2–5ms     intra-DC
Room server processing         ~5ms      ordering + fan-out plan
Pub/sub backbone hop           2–5ms     when crossing gateways
Room server → other gateway    2–5ms
Other gateway → client         20–30ms
TOTAL p99                      ~70–80ms  20–30ms headroom for tail
```

Cross-region adds 60–150ms RTT alone. Sub-100ms cross-region is not a real budget. **[STAFF SIGNAL: latency-budget decomposition] [STAFF SIGNAL: capacity math]**

## 4. High-Level Architecture

Three tiers + pub/sub backbone + persistence. Connection state at the edge; room state in the middle; document truth at the back.

```
                    ┌──────────────────────────────────────────┐
                    │           CLIENTS (WebSocket)            │
                    │  cursor, edits, ack, ping, viewport      │
                    └──────────────┬───────────────────────────┘
                                   │  TLS, sticky to gateway
                  ┌────────────────┴────────────────┐
                  │                                 │
            ┌─────▼──────┐                  ┌───────▼─────┐
            │ GATEWAY 1  │ ... 24–32 ...    │  GATEWAY N  │  <- connection tier
            │ 50K conns  │                  │  50K conns  │     stateful, sticky
            │ per-conn   │                  │  per-conn   │     ws frames in/out
            │ ratelimit  │                  │  ratelimit  │     presence batching
            └─────┬──────┘                  └──────┬──────┘
                  │       gRPC bidi streams        │
                  └──────────────┬─────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  PUB/SUB BACKBONE       │  <- routing tier
                    │  (NATS JetStream)       │     room.{id}.edit (durable)
                    │  per-room edit stream   │     room.{id}.presence (lossy)
                    │  per-room presence sub  │     R=3 replication for edits
                    └────────────┬────────────┘
                                 │
                  ┌──────────────┴──────────────┐
                  │                             │
            ┌─────▼─────┐                 ┌─────▼─────┐
            │ ROOM SVR  │  ...sharded...  │ ROOM SVR  │  <- room state tier
            │ owns 5K   │                 │ owns 5K   │     sequencer per room
            │ rooms     │                 │ rooms     │     5-min backlog
            │ + replica │                 │ + replica │     fan-out planner
            └─────┬─────┘                 └─────┬─────┘
                  │   async edit log persistence │
                  └──────────────┬───────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   DOCUMENT SERVICE      │  <- truth tier
                    │   CRDT log + snapshots  │     Kafka log + S3
                    │   source of truth       │     async write path
                    └─────────────────────────┘
```

**Tier responsibilities.** *Gateway:* terminates TLS + WebSocket. Per-connection rate limit, presence batching, ack tracking. Holds *no* room state beyond "which rooms does this connection care about." Killable without data loss because reconnect handles it (§8). *Pub/sub backbone:* routes between gateways and room servers. Edits use durable JetStream; presence uses non-durable core NATS. **[STAFF SIGNAL: rejected alternative]** Rejected: Redis pub/sub (no durability, slow consumer = silent drop), Kafka (latency floor too high for presence), direct gateway-to-gateway mesh (every gateway needs full membership map = O(N²) state explosion). *Room server:* owns ordering and fan-out for assigned rooms. Per-room monotonic sequencer, 5-minute backlog, fan-out planner. The room server is where "the room" exists as a coherent thing. *Document service:* durable CRDT log. Async write path; room server acks edits to clients only after the durable log write returns.

## 5. Presence vs Edit Channel Design

Two logical channels on one WebSocket, with different rules end-to-end.

```
   PRESENCE CHANNEL (best-effort)         EDIT CHANNEL (durable, ordered)
   ─────────────────────────────────      ─────────────────────────────────
   Frame type 0x01                        Frame type 0x02
   No app-level ack                       App-level ack required
   Server queue: ring buffer,             Server queue: bounded FIFO,
     drops tail under pressure              blocks/NACKs on full
   Coalescing: yes (50ms tick,            Coalescing: no
     last cursor wins per user)
   Ordering: none required                Ordering: per-room causal (seq#)
   Backlog: not retained                  Backlog: 5 min in-mem + log
   Persistence: never                     Persistence: yes, before ack
   SLO: p99 100ms, drops OK               SLO: p99 100ms, drops never
   Backbone: core NATS                    Backbone: NATS JetStream
   Ratelimit: 60/sec/user                 Ratelimit: 20/sec/user
```

**Server-side presence pipeline** runs on a 50ms tick per room:

```
incoming presence msgs ─► per-user latest-state map ─► tick (50ms) ─►
  batch all updated users into one frame ─► per-recipient
  filter by viewport overlap ─► fan out
```

Dominant load path. At a 1000-person room with 100 active editors at 30 Hz, raw rate is 3000 msg/sec ingress → 3M egress without batching. With 50ms ticks: 100 active editors × 20 ticks/sec = 2000 batched ingress events coalesced into ~20 batches/sec/recipient × 1000 recipients = **20K egress batches/sec**, each carrying ~5 cursor updates. Two orders of magnitude reduction. **[STAFF SIGNAL: fan-out optimization]**

**Edit pipeline** is the opposite of optimized — it's *constrained*:

1. Client sends `{op_id, parent_seq, op_payload}`.
2. Gateway forwards to room server (NATS topic `room.{id}.edit`).
3. Room server assigns next monotonic `seq`, writes to backlog, async-writes to Document Service log.
4. Room server fans out `{seq, op_id, op_payload}` to all room gateways.
5. Original gateway returns `ack{op_id, seq}` to client *only after* Document Service confirms log write.

**[STAFF SIGNAL: invariant-based thinking]** Invariants enforced: presence is best-effort; edits are durable-before-ack; per-room causal order via sequencer; convergence delegated to the CRDT engine.

The two channels also have different priority under pressure (§12): edits never starve presence completely, but presence drops first.

## 6. Room-to-Server Placement and Routing

**Decision: control-plane-assigned sticky placement with consistent-hash fallback.** Not pure consistent hashing.

Pure consistent hashing fails on hot rooms — if room R hashes to server S and R becomes a 1000-person room, S is hot and you can't move R without re-hashing chaos. Pure CP-assigned adds a control-plane dependency on the routing path. The hybrid: a **Placement Service** (small Raft-replicated KV) holds `{room_id → room_server}`. On miss, gateways consistent-hash as a default. Placement service can override for hot rooms (migrate them to dedicated servers). **[STAFF SIGNAL: rejected alternative]**

```
                          ┌─────────────────────┐
                          │ PLACEMENT SERVICE   │  Raft KV
                          │ room_id → room_svr  │  + load metrics
                          │ overrides for hot   │  + hot-room migration
                          │ rooms               │  control loop (10s tick)
                          └──────────┬──────────┘
                                     │ watch / poll (5s)
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
        ┌─────▼─────┐          ┌─────▼─────┐          ┌─────▼─────┐
        │ GATEWAY 1 │          │ GATEWAY 2 │          │ GATEWAY N │
        └─────┬─────┘          └─────┬─────┘          └─────┬─────┘
              │                      │                      │
              │   publish NATS subject room.{id}.edit
              │   (gateway subscribes to subjects for rooms it has members in)
              │                      │                      │
              └──────────┬───────────┴──────────┬───────────┘
                         │                      │
                  ┌──────▼──────┐        ┌──────▼──────┐
                  │ ROOM SVR A  │        │ ROOM SVR B  │
                  │ rooms 1..5K │        │ rooms 5K..10K│
                  │ + replica   │        │ + replica   │
                  └─────────────┘        └─────────────┘

  ROUTING WALK (client A on G1, client B on G2, same room R):
    1. A.edit → G1 (WS)
    2. G1 publishes NATS room.R.edit
    3. Room Svr (owner of R) consumes, sequences, writes log
    4. Room Svr publishes NATS room.R.edit.fanout
    5. G1 and G2 (subscribed because they host members of R) consume
    6. G1 → A: ack{seq}; G2 → B: edit{seq, op}
```

**Why pub/sub backbone for cross-gateway routing?** Three alternatives considered:

- **Direct gateway-to-gateway forwarding.** Rejected: every gateway needs the membership map for every room with any local member. With 100K rooms × 32 gateways, the map churns constantly on connect/disconnect. Memory + churn cost not worth saving one hop.
- **Route everything through the room server.** Adopted partially — edits go through the room server (necessary for sequencing), but fan-out from a single room server to 32 gateways is sequential; pub/sub parallelizes it.
- **Pub/sub backbone (NATS).** Chosen. Costs one extra hop (~2–5ms) but absorbs all fan-out parallelism.

**Room server failover:** each room has an active replica on a different host. Coordinated via Placement Service (heartbeat + lease). On primary failure, replica promotes within 1–2s. The 5-minute in-memory backlog is replicated synchronously between primary and replica via lightweight log-shipping; presence state is *not* replicated (it's lossy by design and clients re-publish on reconnect).

**Placement control loop:** every 10s, ingest load metrics per room server. If a single room exceeds 30% of any server's budget, flag for migration. Migration is online: new room server bootstraps from log + current backlog, placement flips, old server drains in-flight messages. Sequence numbers continuous; clients see no disruption.

## 7. Fan-Out at Scale: The 1000-Person Room

Mode switch at >50 participants. ≤50: flat fan-out from room server through pub/sub. >50: hierarchical fan-out with explicit fan-out worker tier.

```
   FLAT MODE (≤50 participants)              HIERARCHICAL MODE (>50)
   ─────────────────────────────             ─────────────────────────────────

   ROOM SVR ──► NATS ──► gateways            ROOM SVR ──► NATS ──► fan-out
                 │         (each handles                              workers (4–16)
                 │         5–10 conns)                                  │
                 │                                                      │
                 ▼                                                      ▼
           1 hop fan-out                                          NATS subjects
           per recipient                                       room.{id}.shard.{k}
                                                                      │
                                                                      ▼
                                                               gateways subscribe
                                                               to shard subjects
                                                               their members
                                                               are assigned to
```

**Why hierarchical?** A single room server fanning out 3M presence msg/sec to 1000 recipients across 32 gateways saturates outbound bandwidth (~1 Gbps NIC of presence alone). Sharding fan-out across 8 workers each handling 125 recipients reduces per-worker load by 8×.

**Optimizations stacked in hierarchical mode:**

1. **Tick-based batching (50ms):** §5. Cuts presence egress ~30×.
2. **Viewport filtering:** each connection reports viewport rect. Server only sends cursor updates for users whose cursor is inside (or near) the recipient's viewport. In a 1000-person room with users spread across a large canvas, this filters 70–90% of presence per recipient. Maintained as an interval tree per room, updated lazily on viewport-change events (rate-limited to 5 Hz).
3. **Presence rate degradation by viewer count:** sender sampling rate is `min(30, 30000/N)` Hz for room of N. At N=1000, server publishes each sender's cursor at 30 Hz max; recipients see at most that, filtered by viewport. Per-sender, not per-recipient.
4. **Spectator vs editor classes:** users who haven't sent input in 60s are demoted to spectator. Spectators get presence at 5 Hz max and lower priority on the egress queue. Removes the ~80% of room participants who are read-only viewers from the hot path.

**Edit fan-out** does not need hierarchical mode at any room size — edit rates are bounded (~10 Hz/active editor × 100 editors = 1000 msg/sec, fanned out to 1000 = 1M/sec, manageable in flat mode). Edits go through the room server's normal sequencing path regardless of room size.

The mode switch is observable: rooms log a `mode_transition` event. Operations sees this as a normal lifecycle event, not an alert. **[STAFF SIGNAL: room-size-spectrum] [STAFF SIGNAL: fan-out optimization]**

## 8. Disconnect / Reconnect Protocol

The reconnect protocol is the difference between "user noticed a 200ms hiccup" and "user lost their work." First-class architectural concern.

```
   STEADY STATE                    DISCONNECT                RECONNECT
   ─────────────                   ──────────                ─────────

   Client tracks last_seq          Socket closes (TCP RST,   Client reconnects with
   per room.                       idle, server shutdown,    {room_id, last_seq, jitter_ms}
                                   app suspend).
   Server backlog stores           Client kicks off          Gateway routes to placement;
   {seq → msg} for last 5min       exponential backoff       room server checks last_seq:
   (in mem, replicated to          reconnect with
   replica).                       jitter (0–5s).            last_seq within backlog?
                                                              ├── YES → replay
                                   App layer caches           │       [last_seq+1..current]
                                   pending edits with         │       ack each, resume.
                                   op_id.                    └── NO  → trigger full resync:
                                                                       client fetches snapshot
                                                                       from Doc Service,
                                                                       reapplies pending
                                                                       edits via CRDT merge.
```

**Three reconnect cases, distinct handling:**

1. **Brief (<5 min, gap fits in backlog):** server replays missed sequence range. Bounded by backlog (~5 min of edits + presence snapshot). Per-room backlog memory: ~1MB typical, ~10MB for hot rooms. Replay is single TCP burst. Latency: 100–500ms to fully resync.
2. **Long (gap exceeds backlog):** server returns `RESYNC_REQUIRED`. Client fetches snapshot from Document Service (HTTP), reapplies to local CRDT. Pending un-acked edits merged into new state via CRDT engine — convergence is the CRDT's job, not messaging. Latency: 1–5s depending on document size.
3. **Server-initiated (gateway shutdown, deploy):** server sends `GOING_AWAY` frame with `reconnect_after_ms: 50`. Client reconnects to a different gateway with normal protocol. Pre-warning lets clients reconnect proactively.

**Thundering herd handling.** A gateway dies with 50K connections. Mitigations stack:

```
   Client jitter:        0–5s uniform random delay      ─► spreads load over 5s
   LB conn rate limit:   500 new conns/sec/gateway     ─► caps inrush
   Gateway accept queue: bounded; rejected → retry      ─► prevents OOM
                         with longer jitter
   Room server warm
   replay:               backlog already in memory      ─► no DB hammer
   Doc Service
   protection:           token bucket + cached          ─► full-resync path
                         snapshots                          doesn't melt DB
```

50K reconnects across 5s jitter = 10K conns/sec total, distributed across remaining 31 gateways = 320 conns/sec/gateway. Below the 500/sec cap. Absorbed without escalation. **[STAFF SIGNAL: failure mode precision] [STAFF SIGNAL: blast radius reasoning] [STAFF SIGNAL: disconnect-reconnect discipline]**

**Heartbeats:** ping every 30s; server treats no-ping for 75s as disconnect (allows 2 missed pings + slack). On detected disconnect: clean up presence state immediately (publish `user_left`), retain edit ack tracking for 5 min in case client reconnects.

## 9. Ordering and Document Convergence

The messaging layer's job is **per-room causal ordering** with a monotonic sequence per room. The CRDT engine handles convergence. The boundary between them is an explicit invariant.

**Choice: CRDT, not OT, not raw central sequencer.**

- **OT** rejected: requires the messaging layer to participate in transformation, coupling messaging to document semantics. OT also has a known fragility class (complex transformation matrices for non-trivial ops). Google Docs makes it work but pays for it in complexity.
- **Raw central sequencer with last-writer-wins** rejected: converges but loses concurrent edits silently. Unacceptable for a Figma-style design tool where two users dragging the same shape simultaneously must produce a coherent merge.
- **CRDT (Yjs-flavored)** chosen: ops are commutative-by-construction. Messaging layer guarantees per-room causal delivery via sequence numbers; the CRDT engine on each client converges given that. Higher metadata overhead per op (~30–100 bytes); acceptable. **[STAFF SIGNAL: ordering-and-convergence] [STAFF SIGNAL: rejected alternative]**

**Boundary invariants:**
- *Messaging layer guarantees:* (a) every edit gets a monotonic per-room `seq`; (b) edits delivered in `seq` order to all subscribed gateways; (c) on reconnect, missing `seq` ranges are replayed.
- *Messaging layer does not guarantee:* exactly-once delivery (it's at-least-once; CRDT ops are idempotent by op_id, so this is fine), causal cross-room ordering (irrelevant), commutativity (CRDT's job).
- *CRDT engine guarantees:* given per-room causal delivery, all clients converge.

**Sequence assignment:** the room server's sequencer is single-writer per room (the active replica). Failover preserves the sequence (replica has the same `seq` state). Per-room, 64-bit, never reused. Gaps in a client's received stream → client requests replay.

This boundary is what allows the messaging layer to be *built* and *operated* without the team needing to be CRDT experts.

## 10. Server Failure and Recovery

Three failure scenarios. **[STAFF SIGNAL: failure mode precision]**

**A: Gateway dies (50K connections).**
- Detection: LB health check at 2s interval, dead after 3 failures = 6s.
- Client experience: socket closes within 1–2s. Client begins jittered reconnect.
- Recovery: 50K clients redistribute across remaining 31 gateways within 5–10s (per §8 math).
- State loss: nothing durable. Presence state implicitly cleaned up; clients re-publish on reconnect. Edit ack state held by room server, not gateway, so in-flight edits survive.
- **RTO: 5–10s** for full recovery; first reconnect typically <2s.

**B: Room server dies (5000 rooms).**
- Detection: replica heartbeat at 1s interval, 3-miss promotion = 3s.
- Client experience: gateway subscriptions fail over to new room server (NATS handles re-bind transparently); briefly, edits queue at gateways for ~3s.
- Recovery: replica promotes, takes over sequencing. Backlog already replicated. New replica provisioned in background.
- State loss: presence lost; clients re-publish on next presence tick (~50ms). Edit state intact via replicated backlog + Document Service log.
- **RTO: 3s** for resumed edit flow; ~100ms for resumed presence.

**C: Pub/sub backbone partial outage (one NATS node).**
- JetStream replicated R=3. One node failure: stream remains available, latency may bump 5–10ms during failover. Automatic recovery. No app-level action.

**D: Document Service write degraded.**
- Detection: ack latency from log writes exceeds 100ms p99.
- Response: room servers stop sending edit acks to clients (clients see "saving…" indicator). Continue fan-out within room (other clients see edits in real time, durability unconfirmed). If degradation persists >10s, return error to clients on new edits.
- Rationale: better to fail visibly than silently lose edits. **Edits are not acked-before-persistence, ever.**

## 11. Multi-Tenancy and Isolation

**Per-tenant limits enforced at gateway:**

```
Limit                          Default     Driver
────────────────────────────   ──────────  ────────────────────────
Max connections / customer     50K         protect against conn flood
Max rooms / customer           10K         protect placement service
Max participants / room        1000        hard cap, hierarchical limit
Max edit msg / sec / customer  5K          tenant-level edit budget
Max conn / IP                  100         basic abuse protection
```

**Per-room budgets:** each room has a CPU/bandwidth budget at its room server. Exceeding triggers throttling: presence rate reduced first, then edit ack latency allowed to grow, then new connections rejected with `ROOM_FULL_TEMPORARILY`. Placement service may migrate a budget-saturating room to dedicated.

**Per-connection rate limits:** token bucket at gateway. Presence: 60/sec burst, 30/sec sustained. Edits: 30/sec burst, 10/sec sustained. Exceed → connection terminated with `RATE_LIMIT`. Hard rule: a malicious client cannot consume more than 1× their fair share at the gateway. **[STAFF SIGNAL: multi-tenant isolation]**

**Noisy neighbor isolation:** a hot room on a shared room server cannot starve other rooms. Per-room scheduling on a fair-share queue (room servers run a coroutine pool with a `rooms × budgets` weighted scheduler).

**Abusive-client detection:** per-connection counters fed to a control plane. Patterns: high message rate from one user across many rooms (spam), high reconnect rate (DoS), abnormally large payloads. Action: progressive — rate limit → temp ban → hard ban via auth.

## 12. Failure Modes and Graceful Degradation

Priority ordering of message types under pressure (first-dropped to last-dropped):

```
1. (drop first)  Presence updates >5Hz from spectators
2.               Presence from spectators entirely
3.               Viewport-update broadcasts
4.               Active-user presence above 10Hz
5.               Active-user presence above 5Hz
6.               All presence except joins/leaves
7.               Edit acks (delay, not drop)
8. (drop never)  Durable edit propagation
```

**Specific scenarios:**

- **Pub/sub backbone overloaded:** drop rate monitored per subject. Presence subjects shed first (NATS core, lossy by design). Edit subjects (JetStream) backpressure into room servers — room servers slow acceptance from gateways, gateways NACK new edits with `BACKPRESSURE` (client retries).
- **Single hot room (bot 100× normal traffic):** detected within 30s by per-room metric anomaly. Placement migrates to dedicated server; per-tenant rate limit kicks in; if attack continues, participants notified and room throttled to tenant budget.
- **Slow gateway:** detected by p99 outbound latency per gateway. LB shifts new connections away. Existing connections drain naturally (clients reconnect). Severe degradation → force-restart.
- **Persistence layer slow:** edit ack latency rises. Room servers buffer up to 1000 unacked edits per room; beyond that, return `WRITE_BACKPRESSURE` to clients. Clients show "saving…" but edits continue to fan out (best-effort).
- **Network partition gateway↔backbone:** gateway can't route. Sends `CONNECTION_DEGRADED`; clients reconnect to a healthy gateway via LB.
- **One user's messages 30s delayed:** subtle bug, usually a slow consumer. Per-recipient outbound queue depth exported as metric; alarm when any recipient queue exceeds 1s lag. Standard remediation: kick the slow connection. **[STAFF SIGNAL: blast radius reasoning]**

## 13. Operational Reality

**Metrics that drive decisions** (not "we'll add monitoring"):

- *End-to-end propagation* (synthetic): bot pairs in rooms across regions, measure publish-to-receive p50/p99/p99.9. Bucketed by room size class. **This is the user-visible SLO.**
- *Per-recipient queue depth:* catches slow consumers before they cascade.
- *Per-room mode:* which rooms are flat vs hierarchical, which are migrated. Watched for patterns (frequent migration = placement instability).
- *Connection lifecycle:* connect rate, disconnect rate split by reason (client/server/network/timeout), reconnect rate, reconnect-success rate, time-to-first-message. Reconnect-success-rate is the leading indicator for cascading failure.
- *Backlog sizes per room server:* drives backlog-window tuning and memory provisioning.
- *Per-tenant rates:* edits/sec, presence/sec, connections, room counts. Drives billing + abuse detection.
- *Sequence gaps observed by clients* (reported back via metric): if non-zero, something's wrong with delivery guarantees.

**Synthetic monitoring:** continuous bots in test rooms across all regions, simulating real traffic patterns (small rooms, large rooms, edit-heavy, presence-heavy). Measure real propagation, not just `ping`. Alerts gate on synthetic SLO, not infra metrics.

**Deployment:** rolling, one gateway at a time. Each gateway sends `GOING_AWAY` to its connections 30s before shutdown; clients reconnect gracefully. Full deploy of 32 gateways: ~16 minutes with 30s/gateway. User-visible impact: one reconnect per user per deploy (~500ms hiccup). Room servers deploy via active/replica swap — promote replica, deploy primary, demote — zero user-visible impact for edits. **[STAFF SIGNAL: observability discipline]**

## 14. Tradeoffs Taken and What Would Change Them

- **NATS JetStream over Kafka.** Lower latency floor (sub-ms vs 5–10ms), good enough durability. Switch to Kafka if edit log retention requirements grew to days (Document Service log already plays that role).
- **CRDT over OT.** Higher per-op metadata overhead. Switch to OT if document-state size became dominant cost (large structured documents with millions of small ops; CRDT metadata bloats).
- **Same-region default.** Invest in cross-region active-active if geographically-distributed teams became the dominant use case — but the latency budget would relax to 200–300ms p99.
- **Mode switch at 50 participants.** Threshold tuned empirically; lower if per-server bandwidth bottlenecked before fan-out CPU.

## 15. What I Would Push Back On

The prompt embeds three assumptions I'd contest before agreeing to the SLO:

1. **Sub-100ms cross-region p99 is not achievable.** US-East to EU-West is ~80ms RTT alone. Adding TLS handshake, app-layer serialization, fan-out, and tail effects, p99 is 200ms+ realistically. Commit to sub-100ms *same-region*; offer a degraded cross-region SLO (300ms p99). If the requirement is hard, the architecture changes substantially (regional CRDT roots, eventual consistency across regions, conflict reconciliation on the boundary). **[STAFF SIGNAL: saying no]**

2. **"100K rooms with 2–1000 participants" treated as one system.** It's two systems sharing infrastructure: a long tail of small rooms and a small set of giants. Pretending one architecture covers both leads to over-engineering small rooms or under-provisioning large ones. The mode switch at 50 is the explicit acknowledgment.

3. **The messaging layer as document source of truth.** Tempting (data flows through it) but wrong. Messaging is for propagation; durability and convergence belong to the Document Service and the CRDT engine respectively. I'd reject any pressure to collapse this boundary for "simplicity" — the simplicity is illusory and the failure modes (data loss on server crash, ambiguous recovery) are catastrophic.

---

The compressing principle throughout: **presence is not edits, gateways are not room servers, propagation is not persistence**. The system is simpler — not more complex — when those distinctions are honored at the architecture level rather than papered over with general-purpose abstractions.