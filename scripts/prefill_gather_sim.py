#!/usr/bin/env python3
# Copyright (c) 2026 Hunter L. Allen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A tick-level model of the TP=4 scheduler's short-prompt prefill, for SPARKFLASH_PREFILL_GATHER_TICKS.

It replays the scheduler's tick (super::batch::Batch::tick): placements, at most one prefill
tick (the fewest rows left first; `together` joins the next whole-chunk rests while a bulk rows
state is left and the rows fit the busy chunk), then one decode step over every decoding
sequence. A prefill tick stalls every decoder for its whole length. The gather rule
(super::gather::prefill_turn) holds a short lead back while decoders run, unless the short rests
it can take already fill a tick, any prefilling prompt has waited N ticks of its own, or a long
prompt (or one split over ticks) is prefilling; the tick after a hold takes the prompts that have
waited N first. The DecodeShare spacing of long chunks is not modelled (it only defers them more).

The costs are measured, not assumed (rank0.log, 2026-11-09 19:40-20:43 PT, 3,261 bulk prompts):
  * a bulk prefill tick of R rows costs F + A*R ms whether it carries one prompt or several
    (prefill_ms lower envelope: singles 300 rows 390 ms, 500 478, 900 722; grouped pairs of the
    same total rows 600 610 ms, 800 699, 1000 805);
  * a decode step over n sequences costs D0 + D1*n ms (187 ms at 7-11 sequences with no prefill
    in the window, 65 ms at one);
  * curation: uncached rest and output length histograms below, 2.38 tokens a step, ~1.8 s
    outside the engine between a slot's requests (8 slots, 0.82 requests/s, 7.98 s in the engine).
Only stdlib; deterministic for a seed.
"""
import argparse
import random
import statistics

F_MS, A_MS = 200.0, 0.5
D0_MS, D1_MS = 45.0, 18.0
EXACT_ROW_MS = 3.6
BULK_MIN = 96
R_MAX = 64
DECODER_WARM = 127 * 19 + 128
BUSY_CHUNK = 2048
WAYS = 3

# Curation, 2026-11-09 19:40-20:43 PT: (bucket start, count), 50-row buckets of the uncached rest.
U_HIST = [(0, 9), (50, 15), (100, 12), (150, 27), (200, 66), (250, 186), (300, 343), (350, 533),
          (400, 494), (450, 404), (500, 298), (550, 250), (600, 180), (650, 83), (700, 66),
          (750, 48), (800, 30), (850, 19), (900, 11), (950, 9), (1000, 4), (1050, 3), (1100, 1),
          (1150, 1)]
G_HIST = [(20, 2), (30, 218), (40, 1368), (50, 1020), (60, 215), (70, 37), (80, 2), (100, 1),
          (110, 3), (120, 6), (130, 5), (140, 12), (150, 20), (160, 20), (170, 18), (180, 25),
          (190, 26), (200, 20), (210, 21), (220, 13), (230, 12), (240, 13), (250, 4), (260, 5),
          (270, 5), (280, 1)]


def draw(rng, hist, width):
    total = sum(c for _, c in hist)
    x = rng.random() * total
    for start, count in hist:
        if x < count:
            return start + int(rng.random() * width)
        x -= count
    return hist[-1][0]


class Client:
    """A closed-loop client: one request in flight, `think` seconds between requests."""

    def __init__(self, kind, rng, think_ms):
        self.kind, self.rng, self.think_ms = kind, rng, think_ms

    def request(self):
        rng = self.rng
        if self.kind == "curation":
            return max(1, draw(rng, U_HIST, 50)), max(2, draw(rng, G_HIST, 10)), 2.38
        if self.kind == "long":
            # A cold 32k-token prompt (a long document or a fresh agent transcript), a short answer.
            return 32768, 200, 2.38
        if self.kind == "agent":
            # A library-building / agent turn: a cached transcript plus a short new rest, a long answer.
            return int(rng.lognormvariate(6.3, 0.8)), int(rng.lognormvariate(6.2, 0.6)), 3.0
        raise ValueError(self.kind)


def simulate(clients, gather_ticks, seconds, seed, busy_chunk=BUSY_CHUNK, ways=WAYS,
             f_ms=F_MS, a_ms=A_MS, d0_ms=D0_MS, d1_ms=D1_MS):
    rng = random.Random(seed)
    for c in clients:
        c.rng = random.Random(rng.random())
    t = 0.0
    tick = 0
    arrivals = [(rng.random() * 2000.0, i) for i in range(len(clients))]  # (time ms, client)
    seqs = []  # dicts
    done = []
    group_sizes = []
    held_ticks = 0
    prefill_ms_total = 0.0
    decode_ms_total = 0.0
    gen_tokens = 0
    horizon = seconds * 1000.0
    while t < horizon:
        tick += 1
        # 1. placements
        still = []
        for when, ci in arrivals:
            if when <= t:
                u, g, tps = clients[ci].request()
                seqs.append(dict(client=ci, kind=clients[ci].kind, arr=when, left=u, u=u, g=g,
                                 tps=tps, emitted=0.0, phase="prefill", placed=tick, ttft=None,
                                 wait=0, split=False))
            else:
                still.append((when, ci))
        arrivals = still
        decoding = [s for s in seqs if s["phase"] == "decode"]
        pending = sorted((s for s in seqs if s["phase"] == "prefill"), key=lambda s: (s["left"], s["arr"]))
        cost = 0.0
        if pending:
            budget = busy_chunk if decoding else 4096
            most_rows = budget if decoding else budget * ways
            whole_most = min(budget, DECODER_WARM + R_MAX)

            def whole(s):
                return R_MAX < s["left"] <= whole_most and s["left"] >= BULK_MIN

            if gather_ticks > 0:
                # The prompts that have waited their ticks go first (gather::lead_order).
                pending.sort(key=lambda s: not (whole(s) and not s["split"] and s["wait"] >= gather_ticks))
            lead = pending[0]
            # together: the lead and the next whole rests while a rows state and the rows last.
            group, rows, full = [], 0, False
            if whole(lead):
                group, rows = [lead], lead["left"]
                for s in pending[1:]:
                    if len(group) == ways:
                        break
                    if not whole(s):
                        continue
                    if rows + s["left"] > most_rows:
                        full = True
                        break
                    group.append(s)
                    rows += s["left"]
            room = bool(group) and not full and len(group) < ways and rows + R_MAX < most_rows
            hold = (gather_ticks > 0 and bool(decoding) and room
                    and all(whole(s) and not s["split"] and s["wait"] < gather_ticks for s in pending))
            if hold:
                held_ticks += 1
            else:
                if lead["left"] < BULK_MIN:
                    ran = [lead]
                    cost = EXACT_ROW_MS * lead["left"]
                elif group:
                    ran = group
                    cost = f_ms + a_ms * rows
                else:
                    n = min(lead["left"], budget)
                    lead["left"] -= n
                    cost = f_ms + a_ms * n
                    ran = [] if lead["left"] > 0 else [lead]
                    lead["split"] = lead["left"] > 0
                for s in ran:
                    s["left"] = 0
                    s["phase"] = "decode"
                    s["ttft"] = t + cost - s["arr"]
                    s["emitted"] = 1.0
                if ran or cost:
                    group_sizes.append(len(ran) if len(ran) > 0 else 1)
            for s in pending:
                if s["phase"] == "prefill":
                    s["wait"] += 1
        prefill_ms_total += cost
        decoding = [s for s in seqs if s["phase"] == "decode"]
        step = d0_ms + d1_ms * len(decoding) if decoding else 0.0
        if not decoding and cost == 0.0:
            # idle: jump to the next arrival
            t = min(a for a, _ in arrivals) if arrivals else horizon
            continue
        decode_ms_total += step
        t += cost + step
        for s in decoding:
            before = s["emitted"]
            s["emitted"] = min(s["g"], s["emitted"] + s["tps"])
            gen_tokens += s["emitted"] - before
            if s["emitted"] >= s["g"]:
                s["phase"] = "done"
                s["end"] = t
                done.append(s)
                c = clients[s["client"]]
                arrivals.append((t + c.think_ms * c.rng.lognormvariate(-0.08, 0.4), s["client"]))
        seqs = [s for s in seqs if s["phase"] != "done"]
    warm = 60_000.0
    out = {}
    for kind in sorted({c.kind for c in clients}):
        rows = [s for s in done if s["kind"] == kind and s["arr"] > warm]
        if not rows:
            continue
        span = (t - warm) / 1000.0
        ttft = sorted(s["ttft"] for s in rows)
        out[kind] = dict(
            req_s=len(rows) / span,
            gen_s=sum(s["g"] for s in rows) / span,
            ttft_p50=ttft[len(ttft) // 2],
            ttft_p95=ttft[int(len(ttft) * 0.95)],
            latency=statistics.mean(s["end"] - s["arr"] for s in rows) / 1000.0,
            per_req_tok_s=statistics.mean((s["g"] - 1) / max(1e-9, (s["end"] - s["arr"] - s["ttft"]) / 1000.0) for s in rows),
        )
    out["all"] = dict(gen_s=gen_tokens / (t / 1000.0), prefill_share=prefill_ms_total / t,
                      groups={k: group_sizes.count(k) for k in sorted(set(group_sizes))},
                      held=held_ticks / tick)
    return out


SCENARIOS = {
    # 2026-11-09 tonight: 8 curation slots plus one long-answer research job.
    "tonight: curation8+agent4": lambda: [Client("curation", None, 1800.0) for _ in range(8)] + [Client("agent", None, 2000.0) for _ in range(4)],
    "curation8+agent6": lambda: [Client("curation", None, 1800.0) for _ in range(8)] + [Client("agent", None, 2000.0) for _ in range(6)],
    "agents12 (library-building)": lambda: [Client("agent", None, 2000.0) for _ in range(12)],
    "agents3 (light)": lambda: [Client("agent", None, 5000.0) for _ in range(3)],
    # Tonight's mix plus a cold 32k-token prompt about every 70 s (one client, 60 s think time).
    "tonight + 32k prompts": lambda: [Client("curation", None, 1800.0) for _ in range(8)] + [Client("agent", None, 2000.0) for _ in range(4)] + [Client("long", None, 60000.0)],
    "curation16": lambda: [Client("curation", None, 1800.0) for _ in range(16)],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=3600)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ticks", default="0,1,2,3,4,6,8")
    ap.add_argument("--f-ms", type=float, default=F_MS)
    ap.add_argument("--a-ms", type=float, default=A_MS)
    ap.add_argument("--scenario", action="append")
    args = ap.parse_args()
    ticks = [int(x) for x in args.ticks.split(",")]
    for name in args.scenario or SCENARIOS:
        print(f"== {name}  (F={args.f_ms} ms, A={args.a_ms} ms/row, {args.seeds} seeds x {args.seconds:.0f} s)")
        base = None
        for g in ticks:
            runs = [simulate(SCENARIOS[name](), g, args.seconds, seed, f_ms=args.f_ms, a_ms=args.a_ms) for seed in range(args.seeds)]
            def avg(kind, key):
                vals = [r[kind][key] for r in runs if kind in r]
                return statistics.mean(vals) if vals else float("nan")
            groups = {}
            for r in runs:
                for k, v in r["all"]["groups"].items():
                    groups[k] = groups.get(k, 0) + v
            tot = sum(groups.values())
            gen = avg("all", "gen_s")
            if base is None:
                base = gen
            line = f"  gather={g:<2} gen {gen:6.1f} tok/s ({100*(gen/base-1):+5.1f}%)  prefill share {100*avg('all','prefill_share'):4.1f}%  ticks by size " + \
                   " ".join(f"{k}:{100*v/tot:.0f}%" for k, v in sorted(groups.items()))
            print(line)
            for kind in ("curation", "agent", "long"):
                if any(kind in r for r in runs):
                    print(f"      {kind:8} {avg(kind,'req_s'):5.3f} req/s  ttft p50 {avg(kind,'ttft_p50'):5.0f} p95 {avg(kind,'ttft_p95'):5.0f} ms  latency {avg(kind,'latency'):5.2f} s  per-request {avg(kind,'per_req_tok_s'):5.1f} tok/s")


if __name__ == "__main__":
    main()
