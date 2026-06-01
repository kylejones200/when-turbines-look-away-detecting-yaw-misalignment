use when_turbines_look_away_detecting_yaw_misalignment_core::yaw_misalignment_stats;

fn main() {
    let n = 5000usize;
    let yaw: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7) % 360.0).collect();
    let wind: Vec<f64> = (0..n).map(|i| (i as f64 * 0.5 + 10.0) % 360.0).collect();
    for _ in 0..10000 {
        let _ = yaw_misalignment_stats(&yaw, &wind);
    }
}
