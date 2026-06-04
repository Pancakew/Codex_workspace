"""Low-level FW-mini command converter.

This module is a small, standalone controller used by verification/demo scripts.
It translates high-level chassis commands into the eight Isaac joint targets:

- 4 steering hinge position targets: [LF, LR, RR, RF]
- 4 wheel velocity targets: [LF, LR, RR, RF]

The obstacle/static DirectRLEnv implementations currently duplicate the same
math internally for vectorized Torch execution. Keep this file as the readable
single-robot reference implementation.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class FWMiniCmdVelConfig:
    """Geometry and safety limits for the FW-mini chassis controller."""

    wheel_base: float = 0.35
    track_width: float = 0.25
    wheel_radius: float = 0.07

    max_speed_mps: float = 0.6
    max_steer_rad: float = 0.30
    steer_rate_limit: float = 0.5

    spin_steer_rad: float = 0.60
    spin_wheel_linear_speed: float = 0.25


class FWMiniCmdVelController:
    """Convert `v_cmd` and `steer_cmd` to FW-mini steering/wheel targets.

    Gear modes:
    - 6: normal four-wheel steering/four-wheel drive.
    - 7: parallel steering placeholder.
    - 8: spin-in-place mode.
    """

    def __init__(self, cfg: FWMiniCmdVelConfig | None = None):
        self.cfg = cfg or FWMiniCmdVelConfig()
        self.last_steer = [0.0, 0.0, 0.0, 0.0]

    def reset(self):
        """Forget steering history so the next command starts from zero steer."""
        self.last_steer = [0.0, 0.0, 0.0, 0.0]

    @staticmethod
    def _clip(x: float, low: float, high: float) -> float:
        return max(low, min(high, x))

    def step(
        self,
        v_cmd: float,
        steer_cmd: float,
        dt: float,
        gear: int = 6,
    ) -> Tuple[List[float], List[float]]:
        """Return `(steer_angles, wheel_omegas)` for one controller step."""

        cfg = self.cfg
        v_cmd = self._clip(v_cmd, -cfg.max_speed_mps, cfg.max_speed_mps)
        steer_cmd = self._clip(steer_cmd, -cfg.max_steer_rad, cfg.max_steer_rad)

        if gear == 8:
            if abs(steer_cmd) < 1e-6:
                steer_angles = [0.0, 0.0, 0.0, 0.0]
                wheel_omegas = [0.0, 0.0, 0.0, 0.0]
                return self._apply_steer_rate_limit(steer_angles, dt), wheel_omegas

            spin_angle = cfg.spin_steer_rad
            steer_angles = [spin_angle, -spin_angle, -spin_angle, spin_angle]

            spin_speed = cfg.spin_wheel_linear_speed
            if steer_cmd > 0.0:
                wheel_linear = [-spin_speed, -spin_speed, spin_speed, spin_speed]
            else:
                wheel_linear = [spin_speed, spin_speed, -spin_speed, -spin_speed]

            wheel_omegas = [v / cfg.wheel_radius for v in wheel_linear]
            return self._apply_steer_rate_limit(steer_angles, dt), wheel_omegas

        if gear == 7:
            steer_angles = [steer_cmd, steer_cmd, steer_cmd, steer_cmd]
            if abs(v_cmd) < 1e-6:
                wheel_omegas = [0.0, 0.0, 0.0, 0.0]
            else:
                omega = v_cmd / cfg.wheel_radius
                wheel_omegas = [omega, omega, omega, omega]
            return self._apply_steer_rate_limit(steer_angles, dt), wheel_omegas

        if abs(steer_cmd) < 1e-6:
            steer_angles = [0.0, 0.0, 0.0, 0.0]
            if abs(v_cmd) < 1e-6:
                wheel_omegas = [0.0, 0.0, 0.0, 0.0]
            else:
                omega = v_cmd / cfg.wheel_radius
                wheel_omegas = [omega, omega, omega, omega]
            return self._apply_steer_rate_limit(steer_angles, dt), wheel_omegas

        sign = 1.0 if steer_cmd > 0.0 else -1.0
        abs_steer = max(abs(steer_cmd), 1e-4)
        radius = cfg.wheel_base / math.tan(abs_steer)

        fl = math.atan(cfg.wheel_base / (radius - cfg.track_width / 2.0)) * sign
        fr = math.atan(cfg.wheel_base / (radius + cfg.track_width / 2.0)) * sign
        rl = -math.atan(cfg.wheel_base / (radius - cfg.track_width / 2.0)) * sign
        rr = -math.atan(cfg.wheel_base / (radius + cfg.track_width / 2.0)) * sign

        steer_angles = [
            self._clip(fl, -cfg.max_steer_rad, cfg.max_steer_rad),
            self._clip(rl, -cfg.max_steer_rad, cfg.max_steer_rad),
            self._clip(rr, -cfg.max_steer_rad, cfg.max_steer_rad),
            self._clip(fr, -cfg.max_steer_rad, cfg.max_steer_rad),
        ]

        # v_cmd == 0 means "align steering only"; do not implicitly spin.
        if abs(v_cmd) < 1e-6:
            wheel_omegas = [0.0, 0.0, 0.0, 0.0]
        else:
            fl_v = v_cmd * (radius - cfg.track_width / 2.0 * sign) / radius
            fr_v = v_cmd * (radius + cfg.track_width / 2.0 * sign) / radius
            rl_v = v_cmd * (radius - cfg.track_width / 2.0 * sign) / radius
            rr_v = v_cmd * (radius + cfg.track_width / 2.0 * sign) / radius
            wheel_omegas = [v / cfg.wheel_radius for v in [fl_v, rl_v, rr_v, fr_v]]

        return self._apply_steer_rate_limit(steer_angles, dt), wheel_omegas

    def _apply_steer_rate_limit(self, target_angles: List[float], dt: float) -> List[float]:
        """Limit steering angle change per step to avoid instant joint jumps."""

        if dt <= 0.0:
            return target_angles

        max_delta = self.cfg.steer_rate_limit * dt
        limited = []

        for i in range(4):
            delta = target_angles[i] - self.last_steer[i]
            delta = self._clip(delta, -max_delta, max_delta)
            limited.append(self.last_steer[i] + delta)

        self.last_steer = limited.copy()
        return limited
