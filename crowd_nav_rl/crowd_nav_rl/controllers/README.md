# controllers

这里放底盘控制器。

主要文件：

- `fwmini_cmdvel_controller.py`

它做的事情是：

`v_cmd / steer_cmd -> 4 个 steering position target + 4 个 wheel velocity target`

注意：当前 DirectRLEnv 为了并行训练，在 env 内部用 Torch 重写了一份相似的控制转换逻辑。这个 controller 更像单机器人验证和 demo 的可读参考实现。

