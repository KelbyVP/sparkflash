// Copyright (c) 2026 Hunter L. Allen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Host-side lane budget: which idle lane is freed, which running sequence is preempted,
//! and which of those sequences count as running.
//!
//! The device still maps the chunks. These loops only decide who gives memory back, in
//! the same order `batch` does, so a test can run them with a small budget and no GPU.
//! A sequence with no lane is waiting, including one that was preempted; `/metrics`
//! reads the gauges these functions move.

use sparkflash_api::Metrics;
#[cfg(test)]
use std::sync::Arc;

/// What one scheduler tick did to lanes, in the order it happened.
///
/// Rank 0 turns these into metrics. The choice of who lost a lane is already made;
/// dropping a note does not put that lane back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LaneNote {
    /// An idle lane was freed so another lane could grow.
    Evicted,
    /// Sequence `id` was given a lane.
    Granted { id: u64 },
    /// Sequence `id` was preempted off its lane.
    Preempted { id: u64 },
}

/// One log line for a preemption: the sequence, where it had got to, and the lane
/// pool's held and budget memory in MiB.
#[must_use]
pub(crate) fn preemption_line(
    id: u64,
    position: usize,
    held_bytes: usize,
    budget_bytes: usize,
) -> String {
    format!(
        "batch: preempted sequence {id} at position {position}; lane memory {} MiB held of {} MiB budget",
        held_bytes >> 20,
        budget_bytes >> 20
    )
}

/// Least recently used of `(index, last_used)`, lower index on a tie.
pub(crate) fn pick_lru(lanes: impl Iterator<Item = (usize, u64)>) -> Option<usize> {
    lanes
        .min_by_key(|&(index, used)| (used, index))
        .map(|(index, _)| index)
}

/// Least recently used free lane, then the higher index. That is the order `Batch::place`
/// uses once the shared prefix does not decide. The test pool has no prefix cache, so this
/// is the whole choice; `place` itself stays on the device.
#[cfg(test)]
fn pick_free_lane(free: &[(usize, u64)]) -> Option<usize> {
    free.iter()
        .max_by(|left, right| right.1.cmp(&left.1).then_with(|| right.0.cmp(&left.0)))
        .map(|(lane, _)| *lane)
}

/// A sequence joined the scheduler. It has no lane yet, so it stays on the waiting gauge.
pub(crate) fn count_join(_metrics: &Metrics, _on_lane: &mut bool) {
    // Admission is not a lane. The waiting gauge already counted the request when it was
    // enqueued; moving it to running here is what made lane-less sequences look busy.
}

/// A sequence was given a lane: it stops waiting and starts running.
pub(crate) fn count_grant(metrics: &Metrics, on_lane: &mut bool) {
    if !*on_lane {
        *on_lane = true;
        metrics.request_admitted();
    }
}

/// An idle lane was freed so another lane could grow.
pub(crate) fn count_eviction(metrics: &Metrics) {
    metrics.record_lane_eviction();
}

/// A running sequence was preempted and is waiting for a lane again.
pub(crate) fn count_preemption(metrics: &Metrics, on_lane: &mut bool) {
    metrics.record_preemption();
    if *on_lane {
        *on_lane = false;
        metrics.request_preempted();
    }
}

/// A sequence left. `on_lane` is whether it was running.
pub(crate) fn count_finish(metrics: &Metrics, on_lane: bool) {
    if on_lane {
        metrics.request_finished();
    } else {
        // It was waiting (never granted a lane, or preempted). The running gauge does not
        // know about it.
        metrics.request_left_waiting();
    }
}

/// What [`grow_for_budget`] asks of the pool. One owner, so the borrows do not overlap.
pub(crate) trait GrowOps {
    /// Back `lane` for `rows`. False when the budget cannot, and nothing was mapped.
    ///
    /// # Errors
    /// The device could not map a chunk.
    fn try_back(&mut self, lane: usize, rows: usize) -> Result<bool, String>;

    /// Least recently used idle lane that still holds memory, other than `except`.
    fn lru_idle(&self, except: usize) -> Option<usize>;

    /// Return `lane`'s memory to the pool. Its cached prefix is gone.
    ///
    /// # Errors
    /// The device could not unmap a chunk.
    fn release_idle(&mut self, lane: usize) -> Result<(), String>;

    /// `lane` was freed by [`grow_for_budget`].
    fn note_eviction(&mut self) {}
}

/// Frees idle lanes, least recently used first, until `lane` can grow to `rows`.
///
/// False when the budget still cannot and no idle lane holds memory. Who is freed is
/// the same choice `Batch::grow` makes.
///
/// # Errors
/// Backing or releasing a lane failed.
pub(crate) fn grow_for_budget<M: GrowOps>(
    mem: &mut M,
    lane: usize,
    rows: usize,
) -> Result<bool, String> {
    loop {
        if mem.try_back(lane, rows)? {
            return Ok(true);
        }
        let Some(idle) = mem.lru_idle(lane) else {
            return Ok(false);
        };
        mem.release_idle(idle)?;
        mem.note_eviction();
    }
}

/// What [`make_room_for`] asks of the scheduler.
pub(crate) trait RoomOps {
    /// Back sequence `index`'s lane for `rows`, evicting idle lanes first.
    ///
    /// # Errors
    /// Backing or releasing a lane failed.
    fn grow_for(&mut self, index: usize, rows: usize) -> Result<bool, String>;

    /// Newest sequence that is running (holds a lane and is not finished).
    fn newest_running(&self) -> Option<usize>;

    /// Send sequence `index` back to wait and release its lane memory.
    ///
    /// # Errors
    /// Releasing the lane failed.
    fn preempt_at(&mut self, index: usize) -> Result<(), String>;

    /// Sequence `index` was preempted by [`make_room_for`].
    fn note_preemption(&mut self, _index: usize) {}
}

/// Backs sequence `index` for `rows`, preempting the newest running sequence while
/// idle lanes are not enough. False when `index` itself was preempted, or nothing
/// running can be preempted.
///
/// The caller has already checked that `index` holds a lane. Preempting that sequence
/// returns false without growing it again, matching `Batch::make_room`.
///
/// # Errors
/// A grow or a preemption failed.
pub(crate) fn make_room_for<S: RoomOps>(
    sched: &mut S,
    index: usize,
    rows: usize,
) -> Result<bool, String> {
    loop {
        if sched.grow_for(index, rows)? {
            return Ok(true);
        }
        let Some(victim) = sched.newest_running() else {
            return Ok(false);
        };
        sched.preempt_at(victim)?;
        sched.note_preemption(victim);
        if victim == index {
            return Ok(false);
        }
    }
}

/// One lane's seat in a test pool: who is on it, when it was last given up, how much
/// it holds. `backed` is an abstract chunk count, not device bytes.
#[cfg(test)]
struct Seat {
    seq: Option<u64>,
    used: u64,
    backed: usize,
}

/// One admitted sequence. `on_lane` is the gauge flag, not the seat: it moves only
/// when the metrics do.
#[cfg(test)]
struct SchedSeq {
    id: u64,
    lane: Option<usize>,
    done: bool,
    on_lane: bool,
}

/// A lane pool with a chunk budget, the scheduler's eviction and preemption, and the
/// running/waiting gauges.
#[cfg(test)]
struct LaneSched {
    lanes: Vec<Seat>,
    seqs: Vec<SchedSeq>,
    budget: usize,
    metrics: Arc<Metrics>,
}

#[cfg(test)]
impl LaneSched {
    fn new(lane_count: usize, budget: usize, metrics: Arc<Metrics>) -> Self {
        Self {
            lanes: (0..lane_count)
                .map(|_| Seat {
                    seq: None,
                    used: 0,
                    backed: 0,
                })
                .collect(),
            seqs: Vec::new(),
            budget,
            metrics,
        }
    }

    fn metrics(&self) -> &Metrics {
        &self.metrics
    }

    /// The request was accepted (`request_enqueued`) and joined the scheduler.
    fn admit(&mut self, id: u64) {
        self.metrics.request_enqueued();
        self.seqs.push(SchedSeq {
            id,
            lane: None,
            done: false,
            on_lane: false,
        });
        let index = self.seqs.len() - 1;
        let metrics = Arc::clone(&self.metrics);
        count_join(&metrics, &mut self.seqs[index].on_lane);
        self.place();
    }

    /// Gives free lanes to waiting sequences, oldest first.
    fn place(&mut self) {
        loop {
            let Some(index) = self
                .seqs
                .iter()
                .position(|seq| !seq.done && seq.lane.is_none())
            else {
                return;
            };
            let free: Vec<(usize, u64)> = self
                .lanes
                .iter()
                .enumerate()
                .filter(|(_, seat)| seat.seq.is_none())
                .map(|(lane, seat)| (lane, seat.used))
                .collect();
            let Some(lane) = pick_free_lane(&free) else {
                return;
            };
            let id = self.seqs[index].id;
            self.lanes[lane].seq = Some(id);
            self.seqs[index].lane = Some(lane);
            let metrics = Arc::clone(&self.metrics);
            count_grant(&metrics, &mut self.seqs[index].on_lane);
        }
    }

    /// The sequence ended. Its lane keeps the chunks it held (a later request may
    /// reuse the prefix); the seat is idle.
    fn finish(&mut self, id: u64) {
        let index = self.index_of(id);
        let on_lane = self.seqs[index].on_lane;
        if let Some(lane) = self.seqs[index].lane.take() {
            self.lanes[lane].seq = None;
            self.lanes[lane].used = self.lanes[lane].used.saturating_add(1);
        }
        self.seqs[index].done = true;
        count_finish(&self.metrics, on_lane);
        self.seqs[index].on_lane = false;
    }

    fn grow_id(&mut self, id: u64, rows: usize) -> Result<bool, String> {
        let index = self.index_of(id);
        let lane = self.seqs[index].lane.ok_or("the sequence has no lane")?;
        grow_for_budget(self, lane, rows)
    }

    fn make_room_id(&mut self, id: u64, rows: usize) -> Result<bool, String> {
        let index = self.index_of(id);
        if self.seqs[index].lane.is_none() {
            return Ok(false);
        }
        make_room_for(self, index, rows)
    }

    fn index_of(&self, id: u64) -> usize {
        self.seqs
            .iter()
            .position(|seq| seq.id == id)
            .unwrap_or_else(|| panic!("sequence {id} is not on the scheduler"))
    }

    fn backed(&self, lane: usize) -> usize {
        self.lanes[lane].backed
    }

    /// Running and waiting sequences still on the scheduler.
    fn counts(&self) -> (i64, i64) {
        let running = self
            .seqs
            .iter()
            .filter(|seq| !seq.done && seq.lane.is_some())
            .count() as i64;
        let waiting = self
            .seqs
            .iter()
            .filter(|seq| !seq.done && seq.lane.is_none())
            .count() as i64;
        (running, waiting)
    }
}

#[cfg(test)]
impl GrowOps for LaneSched {
    fn try_back(&mut self, lane: usize, rows: usize) -> Result<bool, String> {
        if rows <= self.lanes[lane].backed {
            return Ok(true);
        }
        let extra = rows - self.lanes[lane].backed;
        let held: usize = self.lanes.iter().map(|seat| seat.backed).sum();
        if held + extra > self.budget {
            return Ok(false);
        }
        self.lanes[lane].backed = rows;
        Ok(true)
    }

    fn lru_idle(&self, except: usize) -> Option<usize> {
        pick_lru((0..self.lanes.len()).filter_map(|other| {
            let seat = &self.lanes[other];
            if other == except || seat.seq.is_some() || seat.backed == 0 {
                None
            } else {
                Some((other, seat.used))
            }
        }))
    }

    fn release_idle(&mut self, lane: usize) -> Result<(), String> {
        self.lanes[lane].backed = 0;
        Ok(())
    }

    fn note_eviction(&mut self) {
        let metrics = Arc::clone(&self.metrics);
        count_eviction(&metrics);
    }
}

#[cfg(test)]
impl RoomOps for LaneSched {
    fn grow_for(&mut self, index: usize, rows: usize) -> Result<bool, String> {
        let Some(lane) = self.seqs[index].lane else {
            return Ok(false);
        };
        grow_for_budget(self, lane, rows)
    }

    fn newest_running(&self) -> Option<usize> {
        self.seqs
            .iter()
            .rposition(|seq| !seq.done && seq.lane.is_some())
    }

    fn preempt_at(&mut self, index: usize) -> Result<(), String> {
        let Some(lane) = self.seqs[index].lane.take() else {
            return Ok(());
        };
        self.lanes[lane].seq = None;
        self.lanes[lane].backed = 0;
        self.lanes[lane].used = self.lanes[lane].used.saturating_add(1);
        Ok(())
    }

    fn note_preemption(&mut self, index: usize) {
        let metrics = Arc::clone(&self.metrics);
        count_preemption(&metrics, &mut self.seqs[index].on_lane);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(text: &str, name: &str) -> i64 {
        text.lines()
            .find_map(|line| {
                let (key, value) = line.split_once(' ')?;
                if key == name {
                    value.parse().ok()
                } else {
                    None
                }
            })
            .unwrap_or(-1)
    }

    fn pool(lanes: usize, budget: usize) -> LaneSched {
        LaneSched::new(lanes, budget, Arc::new(Metrics::new()))
    }

    #[test]
    fn a_preemption_line_names_the_sequence_its_position_and_the_lane_budget() {
        let line = preemption_line(42, 1000, 6 << 20, 8 << 20);
        assert!(
            line.contains("sequence 42") && line.contains("position 1000"),
            "{line}"
        );
        assert!(
            line.contains("6 MiB held") && line.contains("8 MiB budget"),
            "{line}"
        );
    }

    #[test]
    fn a_full_budget_evicts_an_idle_lane_then_preempts_a_running_one() {
        let mut sched = pool(2, 8);
        sched.admit(1);
        sched.admit(2);
        sched.grow_id(1, 4).expect("lane 0 fits");
        sched.grow_id(2, 4).expect("the budget is now full");
        // The finished sequence keeps its chunks: the lane is idle and still costs.
        sched.finish(2);
        assert!(sched.grow_id(1, 6).expect("evict the idle lane"));
        assert_eq!(sched.backed(0), 6, "the growing lane took the rows");
        assert_eq!(sched.backed(1), 0, "the idle lane gave its chunks back");
        let text = sched.metrics().render();
        assert_eq!(
            sample(&text, "sparkflash_lane_evictions_total"),
            1,
            "an idle lane was freed to make room\n{text}"
        );
        assert_eq!(sample(&text, "sparkflash_preemptions_total"), 0, "{text}");

        sched.admit(3);
        sched.grow_id(3, 2).expect("fills the budget again");
        let before = sched.metrics().render();
        let sum = sample(&before, "sparkflash_num_requests_running")
            + sample(&before, "sparkflash_num_requests_waiting");
        assert!(sched.make_room_id(1, 8).expect("preempt the newest"));
        let text = sched.metrics().render();
        assert_eq!(
            sample(&text, "sparkflash_preemptions_total"),
            1,
            "no idle lane held memory, so a running sequence was preempted\n{text}"
        );
        assert_eq!(
            sample(&text, "sparkflash_lane_evictions_total"),
            1,
            "{text}"
        );
        let (running, waiting) = sched.counts();
        assert_eq!(
            (
                sample(&text, "sparkflash_num_requests_running"),
                sample(&text, "sparkflash_num_requests_waiting")
            ),
            (running, waiting),
            "gauges follow who holds a lane\n{text}"
        );
        assert_eq!(running, 1, "the newest sequence went back to waiting");
        assert_eq!(waiting, 1);
        assert_eq!(
            sample(&text, "sparkflash_num_requests_running")
                + sample(&text, "sparkflash_num_requests_waiting"),
            sum,
            "a preemption does not change the number of sequences\n{text}"
        );
        assert!(sched.seqs[2].lane.is_none(), "sequence 3 was the victim");
        assert!(
            sched.seqs[0].lane.is_some(),
            "the older sequence kept its lane"
        );
    }

    #[test]
    fn three_sequences_on_two_lanes_keep_one_waiting() {
        let mut sched = pool(2, 64);
        sched.admit(1);
        sched.admit(2);
        sched.admit(3);
        let text = sched.metrics().render();
        assert_eq!(
            sample(&text, "sparkflash_num_requests_running"),
            2,
            "{text}"
        );
        assert_eq!(
            sample(&text, "sparkflash_num_requests_waiting"),
            1,
            "{text}"
        );
        assert_eq!(sched.counts(), (2, 1));

        // The next tick gives the freed lane to the sequence that was waiting.
        sched.finish(1);
        sched.place();
        let text = sched.metrics().render();
        assert_eq!(
            sample(&text, "sparkflash_num_requests_running"),
            2,
            "{text}"
        );
        assert_eq!(
            sample(&text, "sparkflash_num_requests_waiting"),
            0,
            "{text}"
        );
        assert_eq!(sched.counts(), (2, 0));
        assert_eq!(
            sample(&text, "sparkflash_num_requests_running")
                + sample(&text, "sparkflash_num_requests_waiting"),
            2,
            "{text}"
        );
    }
}
