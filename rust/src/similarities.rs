use std::cmp::Ordering;
use std::time::Instant;

use fxhash::FxHashMap;
use pyo3::PyResult;
use rayon::prelude::*;

use crate::sparse::{get_row, CsrMatrix};
use crate::utils::CumValues;

const MAX_BLOCK_SIZE: i64 = 200_000_000;

pub(crate) fn compute_sum_squares(interactions: &CsrMatrix<i32, f32>, num: usize) -> Vec<f32> {
    (0..num)
        .map(|i| {
            get_row(interactions, i, false)
                .map_or(0.0, |row| row.fold(0.0, |ss, (_, d)| ss + d * d))
        })
        .collect()
}

fn compute_cosine(prod: f32, sum_squ1: f32, sum_squ2: f32) -> f32 {
    if prod == 0.0 || sum_squ1 == 0.0 || sum_squ2 == 0.0 {
        0.0
    } else {
        let norm = sum_squ1.sqrt() * sum_squ2.sqrt();
        prod / norm
    }
}

#[derive(Debug)]
struct SimVals {
    x1: i32,
    x2: i32,
    prod: f32,
    count: usize,
    cosine: f32,
}

fn compute_row_sims(
    interactions: &CsrMatrix<i32, f32>,
    sum_squares: &[f32],
    n_x: usize,
    x1: usize,
) -> Vec<SimVals> {
    let (indices, indptr, data) = interactions.values();
    let mut res = Vec::new();
    for x2 in (x1 + 1)..n_x {
        let mut i = indptr[x1];
        let mut j = indptr[x2];
        let end1 = indptr[x1 + 1];
        let end2 = indptr[x2 + 1];
        let mut prod = 0.0;
        let mut count = 0;
        while i < end1 && j < end2 {
            let y1 = indices[i];
            let y2 = indices[j];
            match y1.cmp(&y2) {
                Ordering::Less => i += 1,
                Ordering::Greater => j += 1,
                Ordering::Equal => {
                    prod += data[i] * data[j];
                    count += 1;
                    i += 1;
                    j += 1;
                }
            }
        }
        res.push(SimVals {
            x1: x1 as i32,
            x2: x2 as i32,
            prod,
            count,
            cosine: compute_cosine(prod, sum_squares[x1], sum_squares[x2]),
        });
    }
    res
}

pub(crate) fn forward_cosine(
    interactions: &CsrMatrix<i32, f32>,
    sum_squares: &[f32],
    cum_values: &mut FxHashMap<i32, CumValues>,
    n_x: usize,
    min_common: usize,
) -> PyResult<Vec<(i32, i32, f32)>> {
    let start = Instant::now();
    let sim_vals: Vec<SimVals> = (0..n_x)
        .into_par_iter()
        .flat_map(|x| compute_row_sims(interactions, sum_squares, n_x, x))
        .collect();
    let n_x = i32::try_from(n_x)?;
    let mut cosine_sims: Vec<(i32, i32, f32)> = Vec::new();
    for SimVals {
        x1,
        x2,
        prod,
        count,
        cosine,
    } in sim_vals
    {
        if count >= min_common {
            cosine_sims.push((x1, x2, cosine));
        }
        if count > 0 {
            let key = x1 * n_x + x2;
            cum_values.insert(key, (x1, x2, prod, count));
        }
    }
    let duration = start.elapsed();
    println!(
        "forward cosine sim: {} elapsed: {:.4?}",
        cosine_sims.len(),
        duration
    );
    Ok(cosine_sims)
}

/// Divide `n_x` into several blocks to avoid huge memory consumption.
pub(crate) fn invert_cosine(
    interactions: &CsrMatrix<i32, f32>,
    sum_squares: &[f32],
    cum_values: &mut FxHashMap<i32, CumValues>,
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
                let count = counts[index];
                let sq1 = sum_squares[x1];
                let sq2 = sum_squares[x2];
                let key = i32::try_from(x1 * n_x + x2)?;
                let x1 = i32::try_from(x1)?;
                let x2 = i32::try_from(x2)?;
                if count >= min_common {
                    cosine_sims.push((x1, x2, compute_cosine(prod, sq1, sq2)));
                }
                if count > 0 {
                    cum_values.insert(key, (x1, x2, prod, count));
                }
            }
        }
    }
    let duration = start.elapsed();
    println!(
        "invert cosine sim: {} elapsed: {:.4?}",
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
