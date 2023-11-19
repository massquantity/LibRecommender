use std::cmp::Ordering;
use std::time::Instant;

use fxhash::FxHashMap;
use pyo3::PyResult;

use crate::sparse::CsrMatrix;

const MAX_BLOCK_SIZE: i64 = 200_000_000;

pub(crate) fn compute_sum_squares(interactions: &CsrMatrix<i32, f32>, num: usize) -> Vec<f32> {
    let mut sum_squares = vec![0.0; num];
    for (i, ss) in sum_squares.iter_mut().enumerate() {
        if let Some(row) = interactions.get_row(i) {
            *ss = row.map(|(_, &d)| d * d).sum()
        }
    }
    sum_squares
}

/// Divide `n_x` into several blocks to avoid huge memory consumption.
pub(crate) fn invert_cosine(
    interactions: &CsrMatrix<i32, f32>,
    sum_squares: &[f32],
    cum_values: &mut FxHashMap<i32, (i32, i32, f32, usize)>,
    n_x: usize,
    n_y: usize,
    min_common: usize,
) -> PyResult<Vec<(i32, i32, f32)>> {
    let (indices, indptr, data) = interactions.values();
    let start = Instant::now();
    let mut cosine_sims: Vec<(i32, i32, f32)> = Vec::new();
    let step = (MAX_BLOCK_SIZE as f64 / n_x as f64).ceil() as usize;
    for block_start in (0..n_x).step_by(step) {
        let block_end = std::cmp::min(block_start + step, n_x);
        let block_size = block_end - block_start;
        let mut prods = vec![0.0; block_size * n_x];
        let mut counts = vec![0; block_size * n_x];
        for p in 0..n_y {
            let x_start = indptr[p];
            let x_end = indptr[p + 1];
            for i in x_start..x_end {
                let x1 = usize::try_from(indices[i])?;
                if x1 >= block_start && x1 < block_end {
                    for j in (i + 1)..x_end {
                        let x2 = usize::try_from(indices[j])?;
                        let index = (x1 - block_start) * n_x + x2;
                        let value = data[i] * data[j];
                        prods[index] += value;
                        counts[index] += 1;
                    }
                }
            }
        }

        for x1 in block_start..block_end {
            for x2 in (x1 + 1)..n_x {
                let index = (x1 - block_start) * n_x + x2;
                let prod = prods[index];
                let sq1 = sum_squares[x1];
                let sq2 = sum_squares[x2];
                let key = i32::try_from(x1 * n_x + x2)?;
                let x1 = i32::try_from(x1)?;
                let x2 = i32::try_from(x2)?;
                let count = counts[index];
                if count >= min_common {
                    let cosine = if prod == 0.0 || sq1 == 0.0 || sq2 == 0.0 {
                        0.0
                    } else {
                        let norm = sq1.sqrt() * sq2.sqrt();
                        prod / norm
                    };
                    cosine_sims.push((x1, x2, cosine));
                }
                if count > 0 {
                    cum_values.insert(key, (x1, x2, prod, count));
                }
            }
        }
    }
    let duration = start.elapsed();
    println!(
        "cosine sim: {} elapsed: {:.4?}",
        cosine_sims.len(),
        duration
    );
    Ok(cosine_sims)
}

pub(crate) fn sort_by_sims(
    n_x: usize,
    cosine_sims: &[(i32, i32, f32)],
    sim_mapping: &mut FxHashMap<i32, (Vec<i32>, Vec<f32>)>,
) -> PyResult<()> {
    let start = Instant::now();
    let mut agg_sims: Vec<Vec<(i32, f32)>> = vec![Vec::new(); n_x];
    for &(x1, x2, sim) in cosine_sims {
        agg_sims[usize::try_from(x1)?].push((x2, sim));
        agg_sims[usize::try_from(x2)?].push((x1, sim));
    }

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
    println!("sort elapsed: {duration:.4?}");
    Ok(())
}

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
    use super::*;
    use std::collections::BinaryHeap;

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
