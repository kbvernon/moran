use extendr_api::prelude::*;
use faer::prelude::*;
use faer::stats::*;

/// @export
#[extendr]
fn mem(x: Mat<f64>) -> Mat<f64> {
    let n: usize = x.nrows();

    // double center x
    let mut row_mean = Row::zeros(x.ncols());
    stats::row_mean(
        row_mean.as_mut(), 
        x.as_ref(), 
        NanHandling::Propagate
    );

    let mut col_mean = Col::zeros(x.nrows());
    stats::col_mean(
        col_mean.as_mut(), 
        x.as_ref(), 
        NanHandling::Propagate
    );
    
    let mut sum: f64 = 0.0;
    
    for j in 0..n {
        for i in 0..n {
            sum += x[(i,j)] 
        }
    }

    let g_mean = sum/((n^2) as f64);

    let mut omega = Mat::<f64>::zeros(n, n);

    for j in 0..n {
        for i in 0..n {
            omega[(i,j)] = x[(i,j)] - row_mean[i] - col_mean[j] + g_mean;
        }
    }

    // eigendecomposition
    let evd = omega.selfadjoint_eigendecomposition(faer::Side::Lower);

    // moran eigenvector maps
    evd.u().to_owned()
}

// Macro to generate exports.
// This ensures exported functions are registered with R.
// See corresponding C code in `entrypoint.c`.
extendr_module! {
    mod moran;
    fn mem;
}
