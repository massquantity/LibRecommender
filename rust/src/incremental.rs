use std::time::Instant;

use fxhash::FxHashMap;
use pyo3::PyResult;

use crate::sparse::{get_row, CsrMatrix};
use crate::utils::CumValues;

pub(crate) fn update_sum_squares(
    sum_squares: &mut Vec<f32>,
    interactions: &CsrMatrix<i32, f32>,
    num: usize,
) {
    if num > sum_squares.len() {
        sum_squares.resize(num, 0.0);
    }
    for (i, ss) in sum_squares.iter_mut().enumerate() {
        if let Some(row) = get_row(interactions, i, false) {
            *ss += row.map(|(_, d)| d * d).sum::<f32>()
        }
    }
}

pub(crate) fn update_cosine(
    interactions: &CsrMatrix<i32, f32>,
    sum_squares: &[f32],
    cum_values: &mut FxHashMap<i32, (i32, i32, f32, usize)>,
    n_x: usize,
    min_common: usize,
) -> PyResult<Vec<(i32, i32, f32)>> {
    let start = Instant::now();
    let (indices, indptr, data) = interactions.values();
    let mut cosine_sims: Vec<(i32, i32, f32)> = Vec::new();
    for index in 0..n_x {
        let mut prods = vec![0.0; n_x];
        let mut counts = vec![0; n_x];
        let n_y = indptr.len() - 1;
        for p in 0..n_y {
            let x_start = indptr[p];
            let x_end = indptr[p + 1];
            for i in x_start..x_end {
                let x1 = usize::try_from(indices[i])?;
                if x1 == index {
                    for j in (i + 1)..x_end {
                        let x2 = usize::try_from(indices[j])?;
                        prods[x2] += data[i] * data[j];
                        counts[x2] += 1;
                    }
                }
            }
        }

        let new_cosines = accumulate_cosine(
            index,
            n_x,
            &prods,
            &counts,
            sum_squares,
            cum_values,
            min_common,
        )?;
        if !new_cosines.is_empty() {
            cosine_sims.extend(new_cosines);
        }
    }
    let duration = start.elapsed();
    println!(
        "incremental cosine sim: {} elapsed: {:.4?}",
        cosine_sims.len(),
        duration
    );
    Ok(cosine_sims)
}

fn accumulate_cosine(
    x1: usize,
    n_x: usize,
    prods: &[f32],
    counts: &[usize],
    sum_squares: &[f32],
    cum_values: &mut FxHashMap<i32, CumValues>,
    min_common: usize,
) -> PyResult<Vec<(i32, i32, f32)>> {
    let mut cosines = Vec::new();
    for x2 in (x1 + 1)..n_x {
        let prod = prods[x2];
        let count = counts[x2];
        if count == 0 {
            continue;
        }
        let sq1 = sum_squares[x1];
        let sq2 = sum_squares[x2];
        let key = i32::try_from(x1 * n_x + x2)?;
        let x1 = i32::try_from(x1)?;
        let x2 = i32::try_from(x2)?;
        let (.., cum_prod, cum_count) = cum_values
            .entry(key)
            .and_modify(|(.., v, c)| {
                *v += prod;
                *c += count;
            })
            .or_insert((x1, x2, prod, count));

        if *cum_count >= min_common {
            let cosine = if *cum_prod == 0.0 || sq1 == 0.0 || sq2 == 0.0 {
                0.0
            } else {
                let norm = sq1.sqrt() * sq2.sqrt();
                *cum_prod / norm
            };
            cosines.push((x1, x2, cosine));
        }
    }
    Ok(cosines)
}

pub(crate) fn update_by_sims(
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

    for (i, new_neighbor_sims) in agg_sims.into_iter().enumerate() {
        if new_neighbor_sims.is_empty() {
            continue;
        }
        let key = i32::try_from(i)?;
        let mut combined_sims: Vec<(i32, f32)> = if let Some((n, s)) = sim_mapping.get(&key) {
            let pairs = n.iter().zip(s.iter()).map(|(a, b)| (*a, *b));
            let mut original_sims: FxHashMap<i32, f32> = FxHashMap::from_iter(pairs);
            original_sims.extend(new_neighbor_sims); // replace old sims with new ones in map
            original_sims.into_iter().collect()
        } else {
            new_neighbor_sims
        };
        combined_sims.sort_unstable_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap().reverse());
        sim_mapping.insert(key, combined_sims.into_iter().unzip());
    }
    let duration = start.elapsed();
    println!("incremental sort elapsed: {duration:.4?}");
    Ok(())
}
