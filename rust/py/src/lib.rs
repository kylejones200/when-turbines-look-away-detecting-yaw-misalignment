use when_turbines_look_away_detecting_yaw_misalignment_core::yaw_misalignment_stats;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

#[pyfunction]
fn yaw_misalignment_stats_py(
    yaw: PyReadonlyArray1<f64>,
    wind_dir: PyReadonlyArray1<f64>,
) -> PyResult<(f64, f64)> {
    Ok(yaw_misalignment_stats(yaw.as_slice()?, wind_dir.as_slice()?))
}

#[pyfunction]
#[pyo3(signature = (yaw, wind_dir, iterations=10_000))]
fn bench_kernel_py(
    yaw: PyReadonlyArray1<f64>,
    wind_dir: PyReadonlyArray1<f64>,
    iterations: usize,
) -> PyResult<f64> {
    let y = yaw.as_slice()?.to_vec();
    let w = wind_dir.as_slice()?.to_vec();
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = yaw_misalignment_stats(&y, &w);
    }
    Ok(start.elapsed().as_secs_f64())
}

#[pymodule]
fn when_turbines_look_away_detecting_yaw_misalignment_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(yaw_misalignment_stats_py, m)?)?;
    m.add_function(wrap_pyfunction!(bench_kernel_py, m)?)?;
    Ok(())
}
