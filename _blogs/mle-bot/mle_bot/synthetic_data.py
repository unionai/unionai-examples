"""Synthetic predictive maintenance dataset generator.

Generates realistic sensor data for industrial pumps with embedded failure patterns.
The dataset is designed to be a genuine ML challenge:
  - Class imbalance (~3-5% failure rate)
  - Signal-to-noise: failures are predictable but not trivially so
  - Rolling window features matter (degradation builds over time)
  - Multiple machines with different baseline behaviors

This is intentionally NOT private data — safe to demo publicly.
"""

import numpy as np
import pandas as pd


def generate_predictive_maintenance(
    n_machines: int = 20,
    n_days: int = 365,
    readings_per_hour: int = 1,
    failure_rate: float = 0.04,
    seed: int = 42,
    output_path: str = "data/predictive_maintenance.csv",
) -> pd.DataFrame:
    """Generate a synthetic predictive maintenance dataset.

    Simulates sensor readings from industrial pumps. Each pump has baseline
    sensor behavior plus random drift and pre-failure degradation patterns.

    Sensors:
        - vibration_mms: Vibration in mm/s. Increases sharply before failure.
        - temperature_c: Operating temperature. Gradual rise before failure.
        - pressure_bar: Pressure. Drops or spikes before failure.
        - rpm: Rotational speed. Becomes erratic before failure.
        - power_kw: Power consumption. Rises before failure.

    Target:
        - failure_24h: 1 if the machine failed within the next 24 hours, 0 otherwise.

    Args:
        n_machines: Number of distinct machines to simulate.
        n_days: Number of days to simulate per machine.
        readings_per_hour: Sensor readings per hour (1 = hourly, 4 = every 15 min).
        failure_rate: Approximate fraction of time steps labeled as failure=1.
        seed: Random seed for reproducibility.
        output_path: Where to save the CSV file (creates parent dirs).

    Returns:
        DataFrame with columns:
            machine_id, timestamp, vibration_mms, temperature_c, pressure_bar,
            rpm, power_kw, hours_since_maintenance, failure_24h
    """
    rng = np.random.default_rng(seed)
    records = []
    n_hours = n_days * 24
    n_steps = n_hours * readings_per_hour
    step_hours = 1.0 / readings_per_hour

    for machine_id in range(n_machines):
        # Each machine has its own baseline characteristics
        base_vibration = rng.uniform(1.5, 4.0)
        base_temp = rng.uniform(60.0, 85.0)
        base_pressure = rng.uniform(4.0, 7.0)
        base_rpm = rng.uniform(1400, 1600)
        base_power = rng.uniform(15.0, 25.0)

        # Simulate several failure events spaced randomly across the timeline
        n_failures = max(1, int(n_steps * failure_rate / 24))
        failure_times = sorted(rng.integers(n_steps // 4, n_steps - 48, size=n_failures))
        # Ensure failures are spaced at least 7 days apart
        spaced_failures = [failure_times[0]]
        for ft in failure_times[1:]:
            if ft - spaced_failures[-1] > 7 * 24 * readings_per_hour:
                spaced_failures.append(ft)
        failure_set = set(spaced_failures)

        # Build a set of steps labeled as failure_24h=1 (24h window before each failure)
        positive_steps = set()
        for ft in spaced_failures:
            window = 24 * readings_per_hour
            for s in range(max(0, ft - window), ft):
                positive_steps.add(s)

        # Maintenance events (reset degradation)
        maintenance_interval = rng.integers(14 * 24, 30 * 24) * readings_per_hour
        last_maintenance = 0

        # Machine-specific quirks: some machines naturally run hotter/louder
        # This makes raw sensor values overlap between healthy and degrading machines,
        # but the *trend* (rolling mean rising over time) still signals failure.
        quirk_vibration = rng.uniform(0.0, 1.5)
        quirk_temp = rng.uniform(0.0, 10.0)
        quirk_noise_scale = rng.uniform(0.8, 1.5)

        # A few confounder events per machine (load spikes that aren't failures)
        n_confounders = rng.integers(3, 10)
        confounder_times = set(rng.integers(0, n_steps, size=n_confounders))
        confounder_duration = 4 * readings_per_hour  # short spike, ~4h

        for step in range(n_steps):
            hours_elapsed = step * step_hours
            timestamp = pd.Timestamp("2023-01-01") + pd.Timedelta(hours=hours_elapsed)

            hours_since_maintenance = (step - last_maintenance) * step_hours
            if hours_since_maintenance >= maintenance_interval / readings_per_hour:
                last_maintenance = step
                hours_since_maintenance = 0.0

            # Degradation ramps up across the full 24h window but non-linearly —
            # weak early signal that builds. This means a single raw reading is
            # ambiguous, but a rolling window over 6-12h reveals the trend.
            degradation = 0.0
            for ft in spaced_failures:
                window = 24 * readings_per_hour
                steps_to_failure = ft - step
                if 0 < steps_to_failure <= window:
                    # Quadratic ramp: slow start, accelerates near failure
                    progress = 1.0 - steps_to_failure / window   # 0 → 1 as failure approaches
                    degradation = max(degradation, progress ** 2)

            # Confounder spikes (brief, don't accumulate in rolling windows)
            confounder_strength = 0.0
            for ct in confounder_times:
                if 0 <= step - ct < confounder_duration:
                    confounder_strength = max(confounder_strength, rng.uniform(0.2, 0.5))

            slow_drift = np.sin(2 * np.pi * step / (n_steps / 3)) * 0.05
            noise = quirk_noise_scale

            vibration = (
                base_vibration
                + quirk_vibration
                + slow_drift * base_vibration
                + degradation * rng.uniform(2.0, 4.0)          # meaningful but noisy
                + confounder_strength * rng.uniform(1.0, 2.5)
                + rng.normal(0, 0.5 * noise)
            )
            temperature = (
                base_temp
                + quirk_temp
                + slow_drift * 2
                + degradation * rng.uniform(5.0, 12.0)
                + confounder_strength * rng.uniform(2.0, 5.0)
                + rng.normal(0, 2.5 * noise)
            )
            pressure = (
                base_pressure
                + slow_drift * 0.2
                - degradation * rng.uniform(0.3, 0.8)
                + confounder_strength * rng.uniform(-0.2, 0.2)
                + rng.normal(0, 0.15 * noise)
            )
            rpm = (
                base_rpm
                + slow_drift * 10
                + degradation * rng.uniform(-35, 35) * rng.choice([-1, 1])
                + confounder_strength * rng.uniform(-20, 20)
                + rng.normal(0, 20 * noise)
            )
            power = (
                base_power
                + slow_drift * 0.5
                + degradation * rng.uniform(1.5, 3.5)
                + confounder_strength * rng.uniform(0.5, 2.0)
                + rng.normal(0, 0.8 * noise)
            )

            failure_24h = 1 if step in positive_steps else 0

            records.append({
                "machine_id": f"PUMP-{machine_id:03d}",
                "timestamp": timestamp.isoformat(),
                "vibration_mms": round(max(0.1, vibration), 3),
                "temperature_c": round(max(20.0, temperature), 2),
                "pressure_bar": round(max(0.5, pressure), 3),
                "rpm": round(max(500.0, rpm), 1),
                "power_kw": round(max(1.0, power), 2),
                "hours_since_maintenance": round(hours_since_maintenance, 1),
                "failure_24h": failure_24h,
            })

    df = pd.DataFrame(records)

    actual_rate = df["failure_24h"].mean()
    print(f"Generated {len(df):,} rows across {n_machines} machines")
    print(f"Failure rate: {actual_rate:.1%} ({df['failure_24h'].sum():,} positive samples)")

    import os
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")
    return df


if __name__ == "__main__":
    generate_predictive_maintenance()
