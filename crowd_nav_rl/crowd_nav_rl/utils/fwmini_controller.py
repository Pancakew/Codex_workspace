# 输入
# speed: float
# steer: float
# 输出到关节
# left_steering_hinge_joint -> steer
# right_steering_hinge_joint -> steer
# rear_left_steering_hinge_joint -> -steer
# rear_right_steering_hinge_joint -> -steer

# 同时：

# 4 个 wheel joint -> speed

# 注意，这里我沿用了你刚刚测试通过的“前后反向 steering”模式。
# 这不是凭空猜的，是基于你已经验证“转向 + 行驶能够形成平滑曲线”。

from omni.isaac.dynamic_control import _dynamic_control


class FWMiniController:
    def __init__(self, articulation_path: str = "/World/fw_mini/base_link"):
        self.dc = _dynamic_control.acquire_dynamic_control_interface()
        self.articulation_path = articulation_path
        self.art = None

        self.steer_names = [
            "left_steering_hinge_joint",
            "right_steering_hinge_joint",
            "rear_left_steering_hinge_joint",
            "rear_right_steering_hinge_joint",
        ]

        self.wheel_names = [
            "left_front_wheel_joint",
            "right_front_wheel_joint",
            "left_rear_wheel_joint",
            "right_rear_wheel_joint",
        ]

    def initialize(self):
        self.art = self.dc.get_articulation(self.articulation_path)
        if self.art == _dynamic_control.INVALID_HANDLE:
            raise RuntimeError(f"Failed to get articulation at {self.articulation_path}")
        self.dc.wake_up_articulation(self.art)

    def _get_dof(self, name: str):
        dof = self.dc.find_articulation_dof(self.art, name)
        if dof == _dynamic_control.INVALID_HANDLE:
            raise RuntimeError(f"DOF not found: {name}")
        return dof

    def zero_action(self):
        self.apply_action(speed=0.0, steer=0.0)

    def apply_action(self, speed: float, steer: float):
        if self.art is None:
            raise RuntimeError("Controller not initialized. Call initialize() first.")

        self.dc.wake_up_articulation(self.art)

        steer_targets = {
            "left_steering_hinge_joint": steer,
            "right_steering_hinge_joint": steer,
            "rear_left_steering_hinge_joint": -steer,
            "rear_right_steering_hinge_joint": -steer,
        }

        for name, val in steer_targets.items():
            dof = self._get_dof(name)
            self.dc.set_dof_position_target(dof, float(val))

        for name in self.wheel_names:
            dof = self._get_dof(name)
            self.dc.set_dof_velocity_target(dof, float(speed))