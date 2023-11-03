use std::cmp::Ordering;
use std::time::Instant;

use fxhash::FxHashMap;

#[derive(Debug)]
pub(crate) struct LabelSim(pub f32, pub f32);

impl Ord for LabelSim {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for LabelSim {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for LabelSim {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for LabelSim {}

pub(crate) fn invert_cosine(
    indices: &[i32],
    indptr: &[usize],
    data: &[f32],
    sum_squares: &[f32],
    prods: &mut FxHashMap<(i32, i32), f32>,
    counts: &mut FxHashMap<(i32, i32), usize>,
    n_y: usize,
    min_common: usize,
) -> FxHashMap<(i32, i32), f32> {
    let mut start = Instant::now();
    for p in 0..n_y {
        let x_start = indptr[p];
        let x_end = indptr[p + 1];
        for i in x_start..x_end {
            for j in (i + 1)..x_end {
                // let key = indices[i] * n_x + indices[j];
                // key_mapping.entry(key).or_insert((indices[i], indices[j]));
                let key = (indices[i], indices[j]);
                let value = data[i] * data[j];
                prods
                    .entry(key)
                    .and_modify(|v| *v += value)
                    .or_insert(value);
                counts
                    .entry(key)
                    .and_modify(|c| *c += 1)
                    .or_insert(1);
            }
        }
        if p % 1000 == 0 {
            let duration = start.elapsed();
            println!("num {} elapsed: {:?}", p, duration);
            start = Instant::now();
        }
    }

    let mut cosine_sims: FxHashMap<(i32, i32), f32> = FxHashMap::default();
    for (&(x1, x2), count) in counts {
        if *count >= min_common {
            let prod = *prods.get(&(x1, x2)).unwrap();
            let sq1 = sum_squares[x1 as usize];
            let sq2 = sum_squares[x2 as usize];
            if prod == 0.0 || sq1 == 0.0 || sq2 == 0.0 {
                cosine_sims.insert((x1, x2), 0.0);
            } else {
                let norm = sq1.sqrt() * sq2.sqrt();
                cosine_sims.insert((x1, x2), prod / norm);
            }
        }
    }
    cosine_sims
}

pub(crate) fn sort_by_sims(
    cosine_sims: FxHashMap<(i32, i32), f32>,
    sim_mapping: &mut FxHashMap<i32, (Vec<i32>, Vec<f32>)>,
) {
    let mut agg_sims: FxHashMap<i32, Vec<(i32, f32)>> = FxHashMap::default();
    for ((x1, x2), sim) in cosine_sims {
        agg_sims
            .entry(x1)
            .and_modify(|v| (*v).push((x2, sim)))
            .or_insert_with(|| vec![(x2, sim)]);
        agg_sims
            .entry(x2)
            .and_modify(|v| (*v).push((x1, sim)))
            .or_insert_with(|| vec![(x1, sim)]);
    }
    for (i, mut neighbor_sims) in agg_sims {
        neighbor_sims.sort_unstable_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap().reverse());
        let neighbors = neighbor_sims.iter().map(|(n, _)| *n).collect();
        let sims = neighbor_sims.iter().map(|(_, s)| *s).collect();
        sim_mapping.insert(i, (neighbors, sims));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BinaryHeap;

    #[test]
    fn test_sim_max_heap() {
        let mut heap = BinaryHeap::new();
        heap.push(LabelSim(1.1, 1.1));
        heap.push(LabelSim(0.0, 1.1));
        heap.push(LabelSim(-2.0, 0.0));
        heap.push(LabelSim(-0.2, 3.3));
        heap.push(LabelSim(8.8, 8.8));
        assert_eq!(heap.pop(), Some(LabelSim(8.8, 8.8)));
        assert_eq!(heap.pop(), Some(LabelSim(1.1, 1.1)));
        assert_eq!(heap.pop(), Some(LabelSim(0.0, 1.1)));
        assert_eq!(heap.pop(), Some(LabelSim(-0.2, 3.3)));
        assert_eq!(heap.pop(), Some(LabelSim(-2.0, 0.0)));
        assert_eq!(heap.pop(), None);
    }
}
