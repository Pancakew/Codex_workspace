import argparse
import math


def obstacle_clearance(
    x: float,
    y: float,
    obstacles,
    robot_radius: float,
    obstacle_collision_margin: float,
) -> float:
    best = 1e9

    for cx, cy, sx, sy, _sz in obstacles:
        hx = sx / 2.0 + robot_radius + obstacle_collision_margin
        hy = sy / 2.0 + robot_radius + obstacle_collision_margin

        dx = abs(x - cx) - hx
        dy = abs(y - cy) - hy

        outside_dx = max(dx, 0.0)
        outside_dy = max(dy, 0.0)
        outside_dist = math.sqrt(outside_dx * outside_dx + outside_dy * outside_dy)

        inside_dist = min(max(dx, dy), 0.0)
        clearance = outside_dist + inside_dist

        best = min(best, clearance)

    return best


def probe_line(y, obstacles, robot_radius, collision_margin, x_min=0.0, x_max=6.0, x_step=0.05):
    x = x_min
    min_clearance = 1e9

    while x <= x_max + 1e-9:
        c = obstacle_clearance(
            x=x,
            y=y,
            obstacles=obstacles,
            robot_radius=robot_radius,
            obstacle_collision_margin=collision_margin,
        )
        min_clearance = min(min_clearance, c)
        x += x_step

    blocked = min_clearance < 0.0
    return min_clearance, blocked


def main():
    parser = argparse.ArgumentParser(description="Standalone obstacle-map clearance probe.")

    parser.add_argument("--robot-radius", type=float, default=0.22)
    parser.add_argument("--collision-margin", type=float, default=0.02)

    args = parser.parse_args()

    # Must match fwmini_nav_obstacle_vec_env_cfg.py
    obstacles = (
        (2.10, 0.0, 0.30, 0.45, 0.40),
    )

    candidate_lanes = [
        -1.80, -1.60, -1.40, -1.20, -1.00, -0.80,
        -0.60, -0.40, 0.00, 0.40, 0.60,
        0.80, 1.00, 1.20, 1.40, 1.60, 1.80,
    ]

    waypoint_candidates = [
        (2.60, 0.90),
        (2.60, -0.90),
        (2.80, 1.00),
        (2.80, -1.00),
        (3.00, 1.10),
        (3.00, -1.10),
    ]

    print("\n================ OBSTACLE MAP CLEARANCE PROBE ================")
    print("This script does NOT run Isaac Sim.")
    print("It only checks analytical 2D clearance.\n")

    print("obstacles:")
    for obs in obstacles:
        print(f"  {obs}")

    print(f"\nrobot_radius              : {args.robot_radius}")
    print(f"obstacle_collision_margin : {args.collision_margin}")

    print("\nLine clearance test:")
    passable_lanes = []

    for y in candidate_lanes:
        min_clearance, blocked = probe_line(
            y=y,
            obstacles=obstacles,
            robot_radius=args.robot_radius,
            collision_margin=args.collision_margin,
        )

        status = "BLOCKED" if blocked else "FREE"
        print(f"  y={y:+.2f} | min_clearance={min_clearance:+.4f} | {status}")

        if not blocked:
            passable_lanes.append(y)

    print("\nWaypoint clearance test:")
    safe_waypoints = []

    for x, y in waypoint_candidates:
        c = obstacle_clearance(
            x=x,
            y=y,
            obstacles=obstacles,
            robot_radius=args.robot_radius,
            obstacle_collision_margin=args.collision_margin,
        )

        status = "SAFE" if c > 0.15 else "RISKY_OR_COLLISION"
        print(f"  waypoint=({x:.2f}, {y:+.2f}) | clearance={c:+.4f} | {status}")

        if c > 0.15:
            safe_waypoints.append((x, y, c))

    print("\nConclusion:")
    if len(passable_lanes) == 0:
        print("  [FAIL] No passable lane found. Do not run simulation.")
    elif len(safe_waypoints) == 0:
        print("  [WARN] Passable lanes exist, but no safe waypoint was found.")
    else:
        print(f"  [PASS] Passable lanes found: {passable_lanes}")
        print("  [PASS] Safe waypoint candidates:")
        for x, y, c in safe_waypoints:
            print(f"      ({x:.2f}, {y:+.2f}) clearance={c:+.4f}")
        print("  You can run waypoint smoke test.")

    print("===============================================================\n")


if __name__ == "__main__":
    main()