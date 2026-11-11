#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    Pinned {
        rank: usize,
    },
    AtSlot {
        slot: usize,
    },
    EveryNItems {
        first_slot: usize,
        interval: usize,
        rotating: bool,
    },
}

impl Placement {
    pub fn occurrences(self, feed_len: usize) -> usize {
        match self {
            Placement::Pinned { .. } => 1,
            Placement::AtSlot { .. } | Placement::EveryNItems { .. } => self.slots(feed_len).len(),
        }
    }

    pub fn slots(self, feed_len: usize) -> Vec<usize> {
        match self {
            Placement::Pinned { .. } => Vec::new(),
            Placement::AtSlot { slot } => (slot <= feed_len).then_some(slot).into_iter().collect(),
            Placement::EveryNItems {
                first_slot,
                interval,
                ..
            } => {
                if interval == 0 {
                    return Vec::new();
                }
                (first_slot..=feed_len).step_by(interval).collect()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn occurrences_tell_the_source_how_many_payloads_to_fetch() {
        assert_eq!(Placement::Pinned { rank: 3 }.occurrences(30), 1);
        assert_eq!(Placement::AtSlot { slot: 3 }.occurrences(30), 1);
        assert_eq!(
            Placement::EveryNItems {
                first_slot: 5,
                interval: 5,
                rotating: false
            }
            .occurrences(30),
            6
        );
    }

    #[test]
    fn at_slot_yields_its_slot_only_when_the_page_reaches_it() {
        assert_eq!(Placement::AtSlot { slot: 3 }.slots(30), vec![3]);
        assert!(Placement::AtSlot { slot: 3 }.slots(2).is_empty());
    }

    #[test]
    fn every_n_items_fills_the_page() {
        let placement = Placement::EveryNItems {
            first_slot: 5,
            interval: 5,
            rotating: false,
        };
        assert_eq!(placement.slots(17), vec![5, 10, 15]);
    }

    #[test]
    fn every_n_items_yields_nothing_when_the_page_is_too_short() {
        let placement = Placement::EveryNItems {
            first_slot: 25,
            interval: 25,
            rotating: false,
        };
        assert!(placement.slots(10).is_empty());
    }

    #[test]
    fn every_n_items_with_zero_interval_yields_nothing() {
        let placement = Placement::EveryNItems {
            first_slot: 1,
            interval: 0,
            rotating: false,
        };
        assert!(placement.slots(50).is_empty());
    }
}
