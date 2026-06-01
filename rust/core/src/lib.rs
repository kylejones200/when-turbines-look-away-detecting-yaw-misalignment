//! Yaw misalignment statistics (degrees).

pub fn yaw_misalignment_stats(yaw: &[f64], wind_dir: &[f64]) -> (f64, f64) {
    assert_eq!(yaw.len(), wind_dir.len());
    let mut errs = Vec::with_capacity(yaw.len());
    for (&y, &w) in yaw.iter().zip(wind_dir) {
        let mut e = (y - w).rem_euclid(360.0);
        if e > 180.0 {
            e -= 360.0;
        }
        if e < -180.0 {
            e += 360.0;
        }
        errs.push(e.abs());
    }
    let mean = errs.iter().sum::<f64>() / errs.len().max(1) as f64;
    let max = errs.iter().cloned().fold(0.0, f64::max);
    (mean, max)
}
