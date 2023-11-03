use std::cmp::Ordering;
use std::time::Instant;

use fxhash::FxHashMap;
use pyo3::PyResult;

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
    cum_values: &mut FxHashMap<i32, (i32, i32, f32, usize)>,
    n_x: usize,
    n_y: usize,
    min_common: usize,
) -> PyResult<Vec<(i32, i32, f32)>> {
    let mut start = Instant::now();
    let n_x = i32::try_from(n_x)?;
    for p in 0..n_y {
        let x_start = indptr[p];
        let x_end = indptr[p + 1];
        for i in x_start..x_end {
            for j in (i + 1)..x_end {
                let x1 = indices[i];
                let x2 = indices[j];
                let key = x1 * n_x + x2;
                let value = data[i] * data[j];
                cum_values
                    .entry(key)
                    .and_modify(|(.., v, c)| {
                        *v += value;
                        *c += 1;
                    })
                    .or_insert((x1, x2, value, 1));
            }
        }
        if p % 1000 == 0 {
            let duration = start.elapsed();
            println!("num {} elapsed: {:?}", p, duration);
            start = Instant::now();
        }
    }

    start = Instant::now();
    let mut cosine_sims: Vec<(i32, i32, f32)> = Vec::new();
    for &(x1, x2, prod, count) in cum_values.values() {
        if count >= min_common {
            let sq1 = sum_squares[usize::try_from(x1)?];
            let sq2 = sum_squares[usize::try_from(x2)?];
            if prod == 0.0 || sq1 == 0.0 || sq2 == 0.0 {
                cosine_sims.push((x1, x2, 0.0));
            } else {
                let norm = sq1.sqrt() * sq2.sqrt();
                let cosine = prod / norm;
                cosine_sims.push((x1, x2, cosine));
            }
        }
    }
    let duration = start.elapsed();
    println!(
        "cosine sim: {}, elapsed: {:?}",
        cosine_sims.len(),
        duration
    );
    Ok(cosine_sims)
}

pub(crate) fn sort_by_sims(
    n_x: usize,
    cosine_sims: Vec<(i32, i32, f32)>,
    sim_mapping: &mut FxHashMap<i32, (Vec<i32>, Vec<f32>)>,
) -> PyResult<()> {
    let mut start = Instant::now();
    let mut agg_sims: Vec<Vec<(i32, f32)>> = vec![Vec::new(); n_x];
    for (x1, x2, sim) in cosine_sims {
        agg_sims[usize::try_from(x1)?].push((x2, sim));
        agg_sims[usize::try_from(x2)?].push((x1, sim));
    }
    let duration = start.elapsed();
    println!("agg elapsed: {:?}", duration);
    start = Instant::now();
    for (i, neighbor_sims) in agg_sims.iter_mut().enumerate() {
        if neighbor_sims.is_empty() {
            continue;
        }
        neighbor_sims.sort_unstable_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap().reverse());
        let (neighbors, sims) = neighbor_sims
            .iter()
            .map(|(n, s)| (*n, *s))
            .unzip();
        sim_mapping.insert(i32::try_from(i)?, (neighbors, sims));
    }
    let duration = start.elapsed();
    println!("sort elapsed: {:?}", duration);
    Ok(())
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
