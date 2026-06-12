"""FW-mini observation contracts.

This module is intentionally independent from IsaacLab and ROS. The same
constants and pure tensor functions can be used by Isaac training code now and
by a future ROS1 `policy_node` later.
"""

from __future__ import annotations

import torch


PROFILE_MAIN_SCAN44 = "main_scan44"
OBS_DIM_MAIN_SCAN44 = 44
MAIN_SCAN_DIM = 36
MAIN_SCAN_ANGLE_MIN = -1.57
MAIN_SCAN_ANGLE_MAX = 1.57
MAIN_SCAN_RANGE_MIN = 0.2
MAIN_SCAN_RANGE_MAX = 5.0
MAIN_SCAN_SENSOR_X = 0.17
MAIN_SCAN_SENSOR_Y = 0.0

MAIN_SCAN_SLICE = slice(0, 36)
WAYPOINT_X_IDX = 36
WAYPOINT_Y_IDX = 37
WAYPOINT_DIST_IDX = 38
WAYPOINT_HEADING_IDX = 39
CURRENT_V_IDX = 40
CURRENT_W_IDX = 41
LAST_CMD_0_IDX = 42
LAST_CMD_1_IDX = 43

# Neutral names are deliberate:
# - Current Isaac profile: LAST_CMD_0 = last_v_cmd, LAST_CMD_1 = last_steer_cmd.
# - Future ROS [v, w] profile: LAST_CMD_0 = last_v_cmd, LAST_CMD_1 = last_w_cmd.
# Keeping neutral names prevents training/checkpoint metadata from baking in
# the temporary Isaac steering-command interpretation.

PROFILE_MULTI_SCAN152 = "multi_scan152"
OBS_DIM_MULTI_SCAN152 = 152

MAIN_SCAN_SLICE_V2 = slice(0, 36)
FRONT_DEPTH_SCAN_SLICE_V2 = slice(36, 72)
REAR_DEPTH_SCAN_SLICE_V2 = slice(72, 108)
ULTRASONIC_SCAN_SLICE_V2 = slice(108, 144)
WAYPOINT_X_IDX_V2 = 144
WAYPOINT_Y_IDX_V2 = 145
WAYPOINT_DIST_IDX_V2 = 146
WAYPOINT_HEADING_IDX_V2 = 147
CURRENT_V_IDX_V2 = 148
CURRENT_W_IDX_V2 = 149
LAST_CMD_0_IDX_V2 = 150
LAST_CMD_1_IDX_V2 = 151

MAIN_SCAN44_LABELS = (
    [f"main_scan_{i:02d}" for i in range(MAIN_SCAN_DIM)]
    + [
        "waypoint_x",
        "waypoint_y",
        "waypoint_dist",
        "waypoint_heading",
        "current_v",
        "current_w",
        "last_cmd_0",
        "last_cmd_1",
    ]
)


def normalize_ranges(ranges, range_min, range_max):
    """Clip raw ranges and normalize them to [0, 1].

    A normalized value of 1.0 means no hit up to `range_max`. A normalized value
    of 0.0 means the hit is at or below `range_min`.
    """

    ranges_t = ranges if isinstance(ranges, torch.Tensor) else torch.as_tensor(ranges)
    clipped = torch.clamp(ranges_t, min=float(range_min), max=float(range_max))
    return (clipped - float(range_min)) / (float(range_max) - float(range_min))


def _as_feature(value, batch_shape, device, dtype, name: str) -> torch.Tensor:
    value_t = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=device, dtype=dtype)
    value_t = value_t.to(device=device, dtype=dtype)
    feature_shape = tuple(batch_shape) + (1,)

    if value_t.ndim == 0:
        return value_t.reshape((1,) * len(feature_shape)).expand(feature_shape)

    if tuple(value_t.shape) == tuple(batch_shape):
        return value_t.unsqueeze(-1)

    if tuple(value_t.shape) == feature_shape:
        return value_t

    if value_t.numel() == 1:
        return value_t.reshape((1,) * len(feature_shape)).expand(feature_shape)

    raise ValueError(f"{name} shape {tuple(value_t.shape)} is not broadcastable to {feature_shape}")


def build_main_scan44_obs(
    main_scan_norm,
    waypoint_x,
    waypoint_y,
    waypoint_dist,
    waypoint_heading,
    current_v,
    current_w,
    last_cmd_0,
    last_cmd_1,
):
    """Build a `main_scan44` observation tensor.

    The last dimension of `main_scan_norm` must be 36. The output last dimension
    is always 44. This function does not depend on IsaacLab or ROS.
    """

    scan = main_scan_norm if isinstance(main_scan_norm, torch.Tensor) else torch.as_tensor(main_scan_norm)
    if scan.shape[-1] != MAIN_SCAN_DIM:
        raise ValueError(f"main_scan_norm last dim must be {MAIN_SCAN_DIM}, got {scan.shape[-1]}")

    scan = scan.to(dtype=torch.float32)
    batch_shape = scan.shape[:-1]
    device = scan.device
    dtype = scan.dtype

    obs = torch.cat(
        [
            scan,
            _as_feature(waypoint_x, batch_shape, device, dtype, "waypoint_x"),
            _as_feature(waypoint_y, batch_shape, device, dtype, "waypoint_y"),
            _as_feature(waypoint_dist, batch_shape, device, dtype, "waypoint_dist"),
            _as_feature(waypoint_heading, batch_shape, device, dtype, "waypoint_heading"),
            _as_feature(current_v, batch_shape, device, dtype, "current_v"),
            _as_feature(current_w, batch_shape, device, dtype, "current_w"),
            _as_feature(last_cmd_0, batch_shape, device, dtype, "last_cmd_0"),
            _as_feature(last_cmd_1, batch_shape, device, dtype, "last_cmd_1"),
        ],
        dim=-1,
    )

    if obs.shape[-1] != OBS_DIM_MAIN_SCAN44:
        raise RuntimeError(f"main_scan44 obs last dim must be {OBS_DIM_MAIN_SCAN44}, got {obs.shape[-1]}")

    return obs
