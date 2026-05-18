# Repository

Companion code for a Medium article.

## Business context

Wind turbines must face the wind to capture energy efficiently. The yaw system rotates the nacelle atop the tower to keep the rotor perpendicular to the wind direction. A wind vane mounted on the nacelle measures wind direction, and the controller commands yaw motors to adjust the turbine's heading. In ideal conditions, the rotor plane aligns perfectly with the wind, maximizing the swept area that intercepts the flow.

Reality is messier. Wind vanes drift, requiring recalibration every few months but often going years without service. Yaw drives wear, creating backlash that prevents precise positioning. Wind direction changes rapidly due to turbulence and gusts, and the yaw system cannot track perfectly—it responds with lag and dead bands to avoid excessive wear from constant small adjustments. Wake effects from upstream turbines create local wind direction changes that differ from free-stream direction. The result is that turbines operate misaligned more often than wind farm operators realize.

The cost of misalignment is significant. At ten-degree misalignment, power output drops by approximately five percent due to reduced effective swept area—cosine losses where the rotor intercepts only part of the wind. At fifteen degrees, losses reach eight to ten percent. At twenty degrees, losses exceed fifteen percent. Additionally, misalignment creates asymmetric blade loading that accelerates fatigue damage. Yaw bearings experience increased wear, and drivetrain components see higher stress from periodic loading as each blade passes through asymmetric flow. Over twenty years, chronic misalignment can reduce turbine availability by multiple percentage points and shorten component life substantially.

## Setup

1. Copy `.env.example` to `.env` and set `NREL_API_KEY` (free at [developer.nrel.gov/signup](https://developer.nrel.gov/signup/)). Optionally set `NREL_EMAIL` for large downloads.
2. Adjust non-secret NREL settings in `config.yaml` (`nrel.lat`, `nrel.lon`, `nrel.years`, etc.).
3. Install dependencies: `uv sync` (or `pip install -e .`).

Runnable scripts load `config.yaml` and read secrets from `.env` via `python-dotenv` (see `nrel_wtk.py`).

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).