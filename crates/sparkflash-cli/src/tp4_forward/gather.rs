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

//! A tick's prefill turn ([`prefill_turn`]) and the short-prefill gather rule in it
//! (SPARKFLASH_PREFILL_GATHER_TICKS).
//!
//! A bulk tick costs a fixed part (every expert's weights read, about a quarter second) plus its
//! rows, whether it carries one prompt or three, and it stalls every decoder for its whole length:
//! two short prompts that share a tick ([`super::batch`]'s `together`) give the decoders back one
//! fixed part. With N gather ticks a short prompt may wait beside the decoders for more to join
//! it, but:
//!
//! * each sequence counts its own ticks in prefill ([`Wait`]), and no tick holds once any
//!   prefilling sequence has waited N, so no short prompt waits more than N ticks for its first
//!   chunk while the queued rests fit one tick (the ones that have waited N go first);
//! * no tick holds while any prefill is not one short chunk (a long prompt, one split over ticks,
//!   one with images or a rest of R_MAX rows or fewer), so the rule never delays a long prompt's
//!   chunk;
//! * the room it waits in is what `together` actually joined ([`Together`]): only rests it took
//!   count, so it never waits for company that cannot join.
//!
//! Every decision reads only the scheduler's state (rows left, phases, tick counts, the rows
//! states), which every rank holds alike from the link's admissions and cancellations, so the
//! ranks hold alike with no clock of their own. The turn runs over [`PrefillOps`]: the scheduler
//! on the device, a small fake in the tests, as [`super::lane_account`] does for lanes.

use super::rows::R_MAX;

/// The variable that sets the gather ticks.
const VAR: &str = "SPARKFLASH_PREFILL_GATHER_TICKS";

/// The most ticks a short prompt waits beside the decoders for more to share its bulk tick
/// (SPARKFLASH_PREFILL_GATHER_TICKS, default 0: never). Read once a process: a value that is not a
/// tick count is logged once and leaves the rule off, and a set count is logged once. Every rank
/// must use the same value, or the ranks hold differently and their collectives desynchronise:
/// each rank's control Hello carries this value, and rank 0 refuses to start a group that differs.
pub(super) fn ticks() -> u64 {
    static TICKS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *TICKS.get_or_init(|| match ticks_from(std::env::var(VAR).ok().as_deref()) {
        Ok(0) => 0,
        Ok(ticks) => {
            eprintln!(
                "batch: {VAR}={ticks}: a short prefill waits up to {ticks} ticks for company (every rank must use the same value)"
            );
            ticks
        }
        Err(error) => {
            eprintln!("batch: {error}");
            0
        }
    })
}

/// [`ticks`] from the variable's value: unset or blank is 0, a whole number is that many ticks,
/// anything else (a sign, a fraction, a word, past u64) is an error naming the value.
pub(super) fn ticks_from(value: Option<&str>) -> Result<u64, String> {
    let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(0);
    };
    if !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(not_ticks(value));
    }
    value.parse().map_err(|_| not_ticks(value))
}

fn not_ticks(value: &str) -> String {
    format!(
        "{VAR}={value:?} is not a tick count (a whole number, 0 or more); the gather rule is off"
    )
}

/// What the gather rule keeps of one sequence: scheduler state, so every rank keeps it alike.
/// Cleared when the sequence leaves its prefill (its first token, a preemption); a cancelled
/// sequence leaves the scheduler with it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Wait {
    /// The ticks it has been prefilling without its prefill ending, for any reason: a hold, no
    /// room in the tick, or a chunk of a longer rest.
    pub(super) ticks: u64,
    /// A tick ran part of its rest: it is a long prompt, which the rule never holds or waits for.
    pub(super) split: bool,
}

/// The sequences a tick's bulk prefill runs together (`together`), the lead first, and whether
/// that tick would still have room for another short rest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Together {
    pub(super) picked: Vec<usize>,
    /// A bulk rows state left and more than R_MAX of its `most` rows, and no short rest that was
    /// turned away for want of rows.
    pub(super) room: bool,
}

impl Together {
    /// Sequence `lead` alone, with no room: it cannot share a tick (one rows state, no bulk comm,
    /// reads staged, its rest not one chunk, or its lane not backed).
    pub(super) fn alone(lead: usize) -> Self {
        Self {
            picked: vec![lead],
            room: false,
        }
    }

    /// `picked`, `rows` rows in all, of a tick of `ways` rows states and `most` rows; `full` when
    /// a short rest that came next did not fit the rows.
    pub(super) fn new(
        picked: Vec<usize>,
        rows: usize,
        ways: usize,
        most: usize,
        full: bool,
    ) -> Self {
        // Another rest takes more than R_MAX rows (`whole_chunk`).
        let room = !full && picked.len() < ways && rows + R_MAX < most;
        Self { picked, room }
    }
}

/// One prefilling sequence as the rule sees it: whether its rest is one short chunk, and its wait.
#[derive(Clone, Copy, Debug)]
pub(super) struct Rest {
    pub(super) whole: bool,
    pub(super) wait: Wait,
}

impl Rest {
    /// It has waited its `gather_ticks` and is one short chunk: it goes first.
    fn due(self, gather_ticks: u64) -> bool {
        self.whole && !self.wait.split && self.wait.ticks >= gather_ticks
    }
}

/// Whether a tick holds its short prefill back for more short prompts: only with `gather_ticks`
/// set, while sequences decode (else nothing stalls for a prefill), while `together` left room
/// for another rest, while every prefilling sequence is one short chunk (never while a long
/// prompt waits behind it) and none has waited `gather_ticks` yet.
pub(super) fn holds(gather_ticks: u64, decoding: bool, room: bool, rests: &[Rest]) -> bool {
    gather_ticks > 0
        && decoding
        && room
        && rests
            .iter()
            .all(|rest| rest.whole && !rest.wait.split && rest.wait.ticks < gather_ticks)
}

/// The order a tick's prefill takes `order` ([`PrefillOps::order`], the fewest rows first): with
/// gather ticks set, the short rests that have waited them first (in that order among themselves),
/// so the tick that ends a hold runs every prompt it held even if shorter ones came since. With 0,
/// `order` itself.
pub(super) fn lead_order(
    gather_ticks: u64,
    order: Vec<(usize, usize)>,
    rests: &[Rest],
) -> Vec<(usize, usize)> {
    if gather_ticks == 0 || !rests.iter().any(|rest| rest.due(gather_ticks)) {
        return order;
    }
    let (mut due, rest): (Vec<_>, Vec<_>) = order
        .into_iter()
        .zip(rests)
        .partition(|(_, rest)| rest.due(gather_ticks));
    due.extend(rest);
    due.into_iter().map(|(entry, _)| entry).collect()
}

/// The BATCH_TRACE field for the rule: the longest any prefilling sequence has waited, after the
/// tick; nothing with the rule off, so the line is the one it was before.
pub(super) fn trace_field(gather_ticks: u64, longest: u64) -> String {
    if gather_ticks == 0 {
        String::new()
    } else {
        format!(" gathered {longest}")
    }
}

/// What [`prefill_turn`] asks of the scheduler. Sequences are indices into its admission order,
/// which a turn does not change.
pub(super) trait PrefillOps {
    /// What the prefills emit into.
    type Out;
    /// The prefilling sequences as (rows left, index), the fewest rows first (ties to the oldest).
    fn order(&self) -> Vec<(usize, usize)>;
    /// Sequence `index`'s id.
    fn id(&self, index: usize) -> u64;
    /// Sequence `index`'s position.
    fn pos(&self, index: usize) -> usize;
    /// Sequence `index`'s rows left to prefill (0 unless prefilling).
    fn left(&self, index: usize) -> usize;
    /// Whether sequence `index` is still prefilling.
    fn prefilling(&self, index: usize) -> bool;
    /// Whether sequence `index`'s rest is one short bulk chunk (`whole_chunk` at the tick's
    /// budget).
    fn whole(&self, index: usize, decoding: bool) -> bool;
    /// Sequence `index`'s wait.
    fn wait(&mut self, index: usize) -> &mut Wait;
    /// Copies the prefix another lane holds for sequence `index` (`adopt`).
    ///
    /// # Errors
    /// A device failure.
    fn adopt(&mut self, index: usize) -> Result<(), String>;
    /// The short rests that run with sequence `index` (`together`), the others taken in `order`.
    ///
    /// # Errors
    /// A device failure.
    fn together(
        &mut self,
        index: usize,
        decoding: bool,
        order: &[(usize, usize)],
    ) -> Result<Together, String>;
    /// Runs sequences `picked`' rests as one bulk tick; the rows run.
    ///
    /// # Errors
    /// A device failure.
    fn run_together(&mut self, picked: &[usize], out: &mut Self::Out) -> Result<usize, String>;
    /// Runs sequence `index`'s next chunk.
    ///
    /// # Errors
    /// A device failure.
    fn run_one(&mut self, index: usize, decoding: bool, out: &mut Self::Out) -> Result<(), String>;
}

/// What a tick's prefill did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Turn {
    /// Nothing was prefilling.
    Idle,
    /// The decoders' share (rank 0's `hold`) kept a long chunk back.
    Share,
    /// The gather rule held the short prefill back.
    Held,
    /// Sequence `id`'s prefill ran from position `from`, `ran` rows (with any that joined it);
    /// `long` when that was a chunk of a long prompt (the decoders' share clocks it).
    Ran {
        id: u64,
        from: usize,
        ran: usize,
        long: bool,
    },
}

/// One tick's prefill: the lead (the fewest rows left, or with gather ticks set a short rest that
/// has waited them) takes another lane's prefix, then runs, alone or with the short rests that
/// join it, unless `hold` (rank 0's decoders' share, for a lead of more than `long_rows` rows) or
/// the gather rule ([`holds`]) keeps it back. Then every sequence still prefilling has waited one
/// more tick, and one that left its prefill waits no more. Every input is the scheduler's state or
/// `hold`, so every rank takes the same turn.
///
/// # Errors
/// A device failure: the ranks may have desynchronised.
pub(super) fn prefill_turn<O: PrefillOps>(
    ops: &mut O,
    gather_ticks: u64,
    hold: bool,
    decoding: bool,
    long_rows: usize,
    out: &mut O::Out,
) -> Result<Turn, String> {
    let order = ops.order();
    let rests = |ops: &mut O, order: &[(usize, usize)]| -> Vec<Rest> {
        order
            .iter()
            .map(|&(_, index)| Rest {
                whole: ops.whole(index, decoding),
                wait: *ops.wait(index),
            })
            .collect()
    };
    let order = if gather_ticks == 0 {
        order
    } else {
        let before = rests(ops, &order);
        lead_order(gather_ticks, order, &before)
    };
    let Some(&(_, index)) = order.first() else {
        return Ok(Turn::Idle);
    };
    // The rows another lane holds are copied, not run (the prefill's own adopt then finds nothing
    // more): the decoders' share clocks only the rows the chunk computes.
    ops.adopt(index)?;
    let left = ops.left(index);
    let turn = if hold && decoding && left > long_rows {
        Turn::Share
    } else {
        let from = ops.pos(index);
        let together = ops.together(index, decoding, &order)?;
        if gather_ticks > 0 && holds(gather_ticks, decoding, together.room, &rests(ops, &order)) {
            Turn::Held
        } else if together.picked.len() > 1 {
            // Short rests only: no long chunk for the decoders' share.
            let ran = ops.run_together(&together.picked, out)?;
            Turn::Ran {
                id: ops.id(index),
                from,
                ran,
                long: false,
            }
        } else {
            ops.run_one(index, decoding, out)?;
            // A preempted sequence went back to position 0.
            let ran = ops.pos(index).saturating_sub(from);
            if ran > 0 && ops.prefilling(index) {
                ops.wait(index).split = true;
            }
            Turn::Ran {
                id: ops.id(index),
                from,
                ran,
                long: left > long_rows && ran > 0,
            }
        }
    };
    for &(_, index) in &order {
        let prefilling = ops.prefilling(index);
        let wait = ops.wait(index);
        if prefilling {
            wait.ticks += 1;
        } else {
            *wait = Wait::default();
        }
    }
    Ok(turn)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    const BUDGET: usize = 2048;
    const LONG_ROWS: usize = 16384;

    #[test]
    fn the_rule_is_off_unless_a_tick_count_is_set() {
        assert_eq!(ticks_from(None), Ok(0));
        assert_eq!(ticks_from(Some("")), Ok(0));
        assert_eq!(ticks_from(Some("  ")), Ok(0));
        assert_eq!(ticks_from(Some("0")), Ok(0));
        assert_eq!(ticks_from(Some("2")), Ok(2));
        assert_eq!(ticks_from(Some(" 3 ")), Ok(3));
        for bad in ["two", "-1", "+2", "2t", "1.5", "99999999999999999999"] {
            let error = ticks_from(Some(bad)).expect_err(bad);
            assert!(error.contains(VAR) && error.contains(bad), "{error}");
        }
    }

    #[test]
    fn the_trace_line_is_unchanged_with_the_rule_off() {
        assert_eq!(trace_field(0, 0), "");
        assert_eq!(trace_field(0, 5), "");
        assert_eq!(trace_field(2, 1), " gathered 1");
    }

    fn short(ticks: u64) -> Rest {
        Rest {
            whole: true,
            wait: Wait {
                ticks,
                split: false,
            },
        }
    }

    #[test]
    fn a_short_prefill_holds_only_with_room_decoders_and_ticks_left() {
        assert!(holds(2, true, true, &[short(0)]));
        assert!(holds(2, true, true, &[short(1), short(0)]));
        assert!(!holds(0, true, true, &[short(0)]), "off");
        assert!(!holds(2, false, true, &[short(0)]), "nothing decodes");
        assert!(!holds(2, true, false, &[short(0)]), "no room");
        assert!(!holds(2, true, true, &[short(2)]), "waited its ticks");
        assert!(
            !holds(2, true, true, &[short(0), short(2)]),
            "any sequence that has waited its ticks ends the hold"
        );
    }

    #[test]
    fn nothing_holds_while_a_long_prompt_waits() {
        let long = Rest {
            whole: false,
            wait: Wait::default(),
        };
        let tail = Rest {
            whole: true,
            wait: Wait {
                ticks: 0,
                split: true,
            },
        };
        assert!(!holds(8, true, true, &[short(0), long]));
        assert!(!holds(8, true, true, &[tail]), "a long prompt's last chunk");
    }

    #[test]
    fn the_tick_that_ends_a_hold_takes_the_prompts_it_held_first() {
        let order = vec![(300, 4), (500, 1), (900, 2)];
        let rests = [short(0), short(2), short(2)];
        assert_eq!(
            lead_order(2, order.clone(), &rests),
            vec![(500, 1), (900, 2), (300, 4)]
        );
        assert_eq!(lead_order(0, order.clone(), &rests), order, "off");
        assert_eq!(
            lead_order(2, order.clone(), &[short(0), short(1), short(1)]),
            order,
            "none due"
        );
        let long = Rest {
            whole: false,
            wait: Wait {
                ticks: 9,
                split: true,
            },
        };
        assert_eq!(
            lead_order(2, vec![(300, 4), (30_000, 1)], &[short(0), long]),
            vec![(300, 4), (30_000, 1)],
            "a long prompt keeps its place"
        );
    }

    #[test]
    fn room_counts_what_together_took() {
        assert!(Together::new(vec![0], 450, 3, BUDGET, false).room);
        assert!(Together::new(vec![0, 1], 920, 3, BUDGET, false).room);
        assert!(
            !Together::new(vec![0, 1, 2], 1200, 3, BUDGET, false).room,
            "every state"
        );
        assert!(
            !Together::new(vec![0], 1200, 3, BUDGET, true).room,
            "the next did not fit"
        );
        assert!(!Together::new(vec![0], BUDGET - R_MAX, 3, BUDGET, false).room);
        assert!(Together::new(vec![0], BUDGET - R_MAX - 1, 3, BUDGET, false).room);
        assert!(!Together::alone(0).room);
    }

    /// A sequence on [`Fake`].
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct FakeSeq {
        id: u64,
        arrived: u64,
        pos: usize,
        target: usize,
        prefilling: bool,
        wait: Wait,
        /// Whether `together` may back its lane (false: lane memory is short).
        grows: bool,
        /// The tick its prefill ended.
        first: Option<u64>,
    }

    /// What a turn asked of [`Fake`], in order.
    #[derive(Clone, Debug, PartialEq, Eq)]
    enum Op {
        Adopt(u64),
        Together(u64, Vec<u64>),
        Shared(Vec<u64>),
        One(u64),
    }

    /// The scheduler's prefill as [`PrefillOps`] sees it, with no device: every bulk chunk takes
    /// `BUDGET` rows, a short rest (more than R_MAX rows, at most `BUDGET`) runs whole, and
    /// `together` joins short rests in `order` as `Batch::together` does.
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Fake {
        seqs: Vec<FakeSeq>,
        ways: usize,
        tick: u64,
        log: Vec<Op>,
        /// Preempt this sequence when it next runs alone.
        preempt: Option<u64>,
    }

    impl Fake {
        fn new(ways: usize) -> Self {
            Self {
                seqs: Vec::new(),
                ways,
                tick: 0,
                log: Vec::new(),
                preempt: None,
            }
        }

        fn admit(&mut self, id: u64, rows: usize) {
            self.seqs.push(FakeSeq {
                id,
                arrived: self.tick,
                pos: 0,
                target: rows,
                prefilling: true,
                wait: Wait::default(),
                grows: true,
                first: None,
            });
        }

        fn rest(&self, index: usize) -> Option<usize> {
            let seq = &self.seqs[index];
            let left = seq.target - seq.pos;
            (seq.prefilling && left > R_MAX && left <= BUDGET).then_some(left)
        }

        fn finish(&mut self, index: usize) {
            let tick = self.tick;
            let seq = &mut self.seqs[index];
            seq.pos = seq.target;
            seq.prefilling = false;
            seq.first = Some(tick);
        }

        fn seq(&self, id: u64) -> &FakeSeq {
            self.seqs.iter().find(|seq| seq.id == id).expect("seq")
        }
    }

    impl PrefillOps for Fake {
        type Out = ();

        fn order(&self) -> Vec<(usize, usize)> {
            let mut order: Vec<(usize, usize)> = self
                .seqs
                .iter()
                .enumerate()
                .filter(|(_, seq)| seq.prefilling)
                .map(|(index, seq)| (seq.target - seq.pos, index))
                .collect();
            order.sort_unstable();
            order
        }

        fn id(&self, index: usize) -> u64 {
            self.seqs[index].id
        }

        fn pos(&self, index: usize) -> usize {
            self.seqs[index].pos
        }

        fn left(&self, index: usize) -> usize {
            let seq = &self.seqs[index];
            if seq.prefilling {
                seq.target - seq.pos
            } else {
                0
            }
        }

        fn prefilling(&self, index: usize) -> bool {
            self.seqs[index].prefilling
        }

        fn whole(&self, index: usize, _decoding: bool) -> bool {
            self.rest(index).is_some()
        }

        fn wait(&mut self, index: usize) -> &mut Wait {
            &mut self.seqs[index].wait
        }

        fn adopt(&mut self, index: usize) -> Result<(), String> {
            self.log.push(Op::Adopt(self.seqs[index].id));
            Ok(())
        }

        fn together(
            &mut self,
            index: usize,
            decoding: bool,
            order: &[(usize, usize)],
        ) -> Result<Together, String> {
            let ways = self.ways;
            let most = if decoding { BUDGET } else { BUDGET * ways };
            let together = match self.rest(index) {
                Some(lead) if ways > 1 && self.seqs[index].grows => {
                    let (mut picked, mut rows, mut full) = (vec![index], lead, false);
                    for &(_, other) in order {
                        if picked.len() == ways {
                            break;
                        }
                        if other == index {
                            continue;
                        }
                        let Some(left) = self.rest(other) else {
                            continue;
                        };
                        if rows + left > most {
                            full = true;
                            break;
                        }
                        if self.seqs[other].grows {
                            rows += left;
                            picked.push(other);
                        }
                    }
                    Together::new(picked, rows, ways, most, full)
                }
                _ => Together::alone(index),
            };
            let ids = together.picked.iter().map(|&i| self.seqs[i].id).collect();
            self.log.push(Op::Together(self.seqs[index].id, ids));
            Ok(together)
        }

        fn run_together(&mut self, picked: &[usize], _out: &mut ()) -> Result<usize, String> {
            let rows = picked.iter().map(|&i| self.left(i)).sum();
            self.log.push(Op::Shared(
                picked.iter().map(|&i| self.seqs[i].id).collect(),
            ));
            for &index in picked {
                self.finish(index);
            }
            Ok(rows)
        }

        fn run_one(&mut self, index: usize, _decoding: bool, _out: &mut ()) -> Result<(), String> {
            self.log.push(Op::One(self.seqs[index].id));
            if self.preempt == Some(self.seqs[index].id) {
                // Batch::preempt: back to waiting at position 0, its wait cleared.
                self.preempt = None;
                let seq = &mut self.seqs[index];
                seq.prefilling = false;
                seq.pos = 0;
                seq.wait = Wait::default();
                return Ok(());
            }
            let left = self.left(index);
            if left > BUDGET {
                self.seqs[index].pos += BUDGET;
            } else {
                self.finish(index);
            }
            Ok(())
        }
    }

    /// The tick's prefill before SPARKFLASH_PREFILL_GATHER_TICKS (live-engine 22f8f8b1's
    /// `Batch::tick`): the fewest rows first, adopt, then unless the decoders' share holds it,
    /// together or alone.
    fn reference_turn(ops: &mut Fake, hold: bool, decoding: bool) -> Result<Turn, String> {
        let Some(&(_, index)) = ops.order().first() else {
            return Ok(Turn::Idle);
        };
        ops.adopt(index)?;
        let left = ops.left(index);
        if hold && decoding && left > LONG_ROWS {
            return Ok(Turn::Share);
        }
        let from = ops.pos(index);
        let order = ops.order();
        let together = ops.together(index, decoding, &order)?;
        if together.picked.len() > 1 {
            let ran = ops.run_together(&together.picked, &mut ())?;
            return Ok(Turn::Ran {
                id: ops.id(index),
                from,
                ran,
                long: false,
            });
        }
        ops.run_one(index, decoding, &mut ())?;
        let ran = ops.pos(index).saturating_sub(from);
        Ok(Turn::Ran {
            id: ops.id(index),
            from,
            ran,
            long: left > LONG_ROWS && ran > 0,
        })
    }

    /// A small deterministic generator for the load tests.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        /// True with chance `percent` in 100.
        fn chance(&mut self, percent: u64) -> bool {
            self.next() % 100 < percent
        }

        fn rows(&mut self, low: usize, high: usize) -> usize {
            low + (self.next() % (high - low + 1) as u64) as usize
        }
    }

    /// Arrivals by tick: (id, rows).
    type Events = BTreeMap<u64, Vec<(u64, usize)>>;

    /// Short prompts (300-900 rows) arriving with chance `percent` a tick, and a `long`-row
    /// prompt every `long_every` ticks.
    fn load(seed: u64, ticks: u64, percent: u64, long_every: Option<(u64, usize)>) -> Events {
        let mut rng = Rng(seed);
        let mut events = Events::new();
        let mut id = 0;
        for tick in 0..ticks {
            if rng.chance(percent) {
                events
                    .entry(tick)
                    .or_default()
                    .push((id, rng.rows(300, 900)));
                id += 1;
            }
            if let Some((every, rows)) = long_every
                && tick % every == 0
            {
                events.entry(tick).or_default().push((id, rows));
                id += 1;
            }
        }
        events
    }

    /// Runs `ticks` ticks of `events` through [`prefill_turn`] with something always decoding;
    /// the scheduler and each tick's turn.
    fn drive(gather_ticks: u64, events: &Events, ticks: u64, ways: usize) -> (Fake, Vec<Turn>) {
        let mut fake = Fake::new(ways);
        let mut turns = Vec::new();
        for tick in 0..ticks {
            fake.tick = tick;
            for &(id, rows) in events.get(&tick).into_iter().flatten() {
                fake.admit(id, rows);
            }
            turns.push(
                prefill_turn(&mut fake, gather_ticks, false, true, LONG_ROWS, &mut ()).unwrap(),
            );
        }
        (fake, turns)
    }

    /// The ticks each short prompt (at most `BUDGET` rows) waited for its first token.
    fn short_waits(fake: &Fake) -> Vec<u64> {
        fake.seqs
            .iter()
            .filter(|seq| seq.target <= BUDGET)
            .filter_map(|seq| seq.first.map(|first| first - seq.arrived))
            .collect()
    }

    #[test]
    fn a_prompt_left_over_from_a_full_tick_does_not_get_a_fresh_hold() {
        // CV's trace: A (900 rows) at tick 0, B (1,200) at tick 1, two gather ticks. B cannot
        // join A (2,100 rows), so A runs alone at tick 1; B has then waited a tick, and may wait
        // one more, not two.
        let events = Events::from([(0, vec![(0, 900)]), (1, vec![(1, 1200)])]);
        let (fake, turns) = drive(2, &events, 8, 3);
        assert_eq!(fake.seq(0).first, Some(1));
        assert_eq!(fake.seq(1).first, Some(3), "{turns:?}");
        // The rule off: each runs on arrival.
        let (fake, _) = drive(0, &events, 8, 3);
        assert_eq!((fake.seq(0).first, fake.seq(1).first), (Some(0), Some(1)));
    }

    #[test]
    fn no_short_prompt_waits_more_than_the_gather_ticks() {
        for percent in [17, 30, 45] {
            let events = load(0x9e37_79b9_7f4a_7c15 ^ percent, 20_000, percent, None);
            let (fake, _) = drive(0, &events, 20_100, 3);
            assert_eq!(
                short_waits(&fake).into_iter().max(),
                Some(0),
                "the rule off"
            );
            for gather_ticks in [1, 2, 4, 8] {
                let (fake, turns) = drive(gather_ticks, &events, 20_100, 3);
                let waits = short_waits(&fake);
                assert_eq!(waits.len(), fake.seqs.len(), "every prompt ran");
                assert_eq!(
                    waits.iter().max(),
                    Some(&gather_ticks),
                    "{percent}% a tick, {gather_ticks} gather ticks"
                );
                assert!(turns.contains(&Turn::Held));
            }
        }
    }

    /// The tick each long prompt's prefill ended, less the tick it arrived.
    fn long_finish(fake: &Fake) -> Vec<u64> {
        fake.seqs
            .iter()
            .filter(|seq| seq.target > BUDGET)
            .map(|seq| seq.first.expect("finished") - seq.arrived)
            .collect()
    }

    #[test]
    fn a_held_short_prompt_never_holds_a_long_prompt_back() {
        for percent in [17, 30] {
            let events = load(7 + percent, 20_000, percent, Some((400, 32_768)));
            let (fake, _) = drive(0, &events, 20_400, 3);
            let base = long_finish(&fake);
            for gather_ticks in [1, 2, 4, 8] {
                let (fake, turns) = drive(gather_ticks, &events, 20_400, 3);
                // No tick held while a long prompt was prefilling.
                let mut replay = Fake::new(3);
                for (tick, turn) in turns.iter().enumerate() {
                    let tick = tick as u64;
                    replay.tick = tick;
                    for &(id, rows) in events.get(&tick).into_iter().flatten() {
                        replay.admit(id, rows);
                    }
                    let long = replay
                        .seqs
                        .iter()
                        .any(|seq| seq.prefilling && seq.target > BUDGET);
                    let again =
                        prefill_turn(&mut replay, gather_ticks, false, true, LONG_ROWS, &mut ())
                            .unwrap();
                    assert_eq!(&again, turn);
                    assert!(
                        !(long && again == Turn::Held),
                        "held at tick {tick} beside a long prompt"
                    );
                }
                // A long prompt that arrives during a hold first runs the short prompts held (at
                // the rule off they ran before it came): one tick. After that no tick holds, but
                // the tick it lost can move its short last chunk behind later short prompts (the
                // fewest rows go first, as ever), so single prompts can end a few ticks later or
                // sooner. On average it ends less than a tick later.
                let finish = long_finish(&fake);
                assert_eq!(finish.len(), base.len());
                let mean = |ticks: &[u64]| ticks.iter().sum::<u64>() as f64 / ticks.len() as f64;
                let (with, without) = (mean(&finish), mean(&base));
                assert!(
                    with <= without + 1.0,
                    "{with:.2} vs {without:.2} ticks at {gather_ticks} gather ticks"
                );
            }
        }
    }

    #[test]
    fn with_the_rule_off_every_tick_decides_as_before() {
        for seed in 1..=4 {
            let events = load(seed, 5_000, 25 + 5 * seed, Some((300, 20_000)));
            let mut rng = Rng(seed * 31);
            let (mut now, mut before) = (Fake::new(3), Fake::new(3));
            for tick in 0..5_200 {
                for fake in [&mut now, &mut before] {
                    fake.tick = tick;
                    for &(id, rows) in events.get(&tick).into_iter().flatten() {
                        fake.admit(id, rows);
                    }
                }
                // Rank 0's share hold, and an idle decoder now and then.
                let (hold, decoding) = (rng.chance(30), !rng.chance(10));
                let turn = prefill_turn(&mut now, 0, hold, decoding, LONG_ROWS, &mut ()).unwrap();
                let reference = reference_turn(&mut before, hold, decoding).unwrap();
                assert_eq!(turn, reference, "tick {tick}");
                assert_eq!(now.log, before.log, "tick {tick}");
            }
        }
    }

    #[test]
    fn two_ranks_fed_the_same_ticks_take_the_same_turns() {
        let events = load(42, 5_000, 30, Some((500, 32_768)));
        let (rank0, turns0) = drive(3, &events, 5_200, 3);
        let (rank1, turns1) = drive(3, &events, 5_200, 3);
        assert_eq!(turns0, turns1);
        assert_eq!(rank0, rank1);
        assert!(turns0.contains(&Turn::Held));
    }

    #[test]
    fn a_hold_waits_only_for_company_together_can_take() {
        // A 450-row lead and a 470-row rest whose lane cannot be backed: the lead holds for
        // later company (the tick has room), but the rest does not count as company...
        let mut fake = Fake::new(2);
        fake.admit(0, 450);
        fake.admit(1, 470);
        fake.seqs[1].grows = false;
        let turn = prefill_turn(&mut fake, 4, false, true, LONG_ROWS, &mut ()).unwrap();
        assert_eq!(
            turn,
            Turn::Held,
            "with two states, the refused rest leaves one for company"
        );
        // ...and a lead whose own lane cannot be backed never holds.
        let mut fake = Fake::new(3);
        fake.admit(0, 450);
        fake.seqs[0].grows = false;
        let turn = prefill_turn(&mut fake, 4, false, true, LONG_ROWS, &mut ()).unwrap();
        assert!(matches!(turn, Turn::Ran { id: 0, .. }), "{turn:?}");
        // One rows state: nothing can join.
        let mut fake = Fake::new(1);
        fake.admit(0, 450);
        let turn = prefill_turn(&mut fake, 4, false, true, LONG_ROWS, &mut ()).unwrap();
        assert!(matches!(turn, Turn::Ran { id: 0, .. }), "{turn:?}");
    }

    #[test]
    fn a_sequence_that_leaves_its_prefill_waits_no_more() {
        let mut fake = Fake::new(3);
        fake.admit(0, 450);
        for _ in 0..2 {
            let turn = prefill_turn(&mut fake, 2, false, true, LONG_ROWS, &mut ()).unwrap();
            assert_eq!(turn, Turn::Held);
        }
        assert_eq!(fake.seqs[0].wait.ticks, 2);
        // Its turn comes, and it is preempted back to waiting: its wait is cleared.
        fake.preempt = Some(0);
        let turn = prefill_turn(&mut fake, 2, false, true, LONG_ROWS, &mut ()).unwrap();
        assert!(matches!(turn, Turn::Ran { id: 0, ran: 0, .. }), "{turn:?}");
        assert_eq!(fake.seqs[0].wait, Wait::default());
        // Placed again, it may wait its ticks once more, then runs.
        fake.seqs[0].prefilling = true;
        assert_eq!(
            prefill_turn(&mut fake, 2, false, true, LONG_ROWS, &mut ()).unwrap(),
            Turn::Held
        );
        assert_eq!(
            prefill_turn(&mut fake, 2, false, true, LONG_ROWS, &mut ()).unwrap(),
            Turn::Held
        );
        let turn = prefill_turn(&mut fake, 2, false, true, LONG_ROWS, &mut ()).unwrap();
        assert!(
            matches!(
                turn,
                Turn::Ran {
                    id: 0,
                    ran: 450,
                    ..
                }
            ),
            "{turn:?}"
        );
        assert_eq!(fake.seqs[0].wait, Wait::default());
    }

    #[test]
    fn a_hold_needs_decoders() {
        let mut fake = Fake::new(3);
        fake.admit(0, 450);
        let turn = prefill_turn(&mut fake, 4, false, false, LONG_ROWS, &mut ()).unwrap();
        assert!(matches!(turn, Turn::Ran { id: 0, .. }), "{turn:?}");
    }

    #[test]
    fn a_long_prompt_s_last_chunk_is_never_held() {
        // 2,500 rows: a chunk of 2,048, then a 452-row tail that is one short chunk.
        let mut fake = Fake::new(3);
        fake.admit(0, 2500);
        let turn = prefill_turn(&mut fake, 4, false, true, LONG_ROWS, &mut ()).unwrap();
        assert!(
            matches!(
                turn,
                Turn::Ran {
                    id: 0,
                    ran: 2048,
                    ..
                }
            ),
            "{turn:?}"
        );
        assert!(fake.seqs[0].wait.split);
        let turn = prefill_turn(&mut fake, 4, false, true, LONG_ROWS, &mut ()).unwrap();
        assert!(
            matches!(
                turn,
                Turn::Ran {
                    id: 0,
                    ran: 452,
                    ..
                }
            ),
            "{turn:?}"
        );
    }
}
