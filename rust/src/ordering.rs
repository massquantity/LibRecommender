use std::cmp::Ordering;

/// 0: similarity, 1: label
#[derive(Debug)]
pub(crate) struct SimOrd(pub f32, pub f32);

impl Ord for SimOrd {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for SimOrd {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for SimOrd {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for SimOrd {}

#[cfg(test)]
mod tests {
    use std::collections::BinaryHeap;

    use super::*;

    #[test]
    fn test_sim_max_heap() {
        let mut heap = BinaryHeap::new();
        heap.push(SimOrd(1.1, 1.1));
        heap.push(SimOrd(0.0, 1.1));
        heap.push(SimOrd(-2.0, 0.0));
        heap.push(SimOrd(-0.2, 3.3));
        heap.push(SimOrd(8.8, 8.8));
        assert_eq!(heap.pop(), Some(SimOrd(8.8, 8.8)));
        assert_eq!(heap.pop(), Some(SimOrd(1.1, 1.1)));
        assert_eq!(heap.pop(), Some(SimOrd(0.0, 1.1)));
        assert_eq!(heap.pop(), Some(SimOrd(-0.2, 3.3)));
        assert_eq!(heap.pop(), Some(SimOrd(-2.0, 0.0)));
        assert_eq!(heap.pop(), None);
    }
}
