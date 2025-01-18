
import time
import datetime
import numpy as np
import math
import torch
import itertools
import h5py
import argparse
import os
import multiprocessing as mp
from timer_fd import Timer

from setproctitle import setproctitle
from scipy.spatial.transform import Rotation as R
from go2deploy import ONNXModule, init_channel
from go2deploy.utils import lerp, normalize
from go2deploy.robot import Go2Iface
from go2deploy.commands import VelocityControl

from torch.utils._pytree import tree_map
from dataclasses import dataclass

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

@dataclass
class BasicObsCfg:


    jpos_steps: int = 3
    jvel_steps: int = 3
    gyro_steps: int = 3
    gravity_steps: int = 3


class ImpedanceControl:

    oscillator_history: bool = False
    command_dim: int = 10 + 3 * 4

    def __init__(
        self,
        robot: Go2Iface,
        obs_cfg: BasicObsCfg,
    ):
        self.robot = robot
        self.obs_cfg = obs_cfg

        self.phi = np.zeros(4, dtype=np.float32)
        self.phi[0] = np.pi
        self.phi[3] = np.pi
        self.phi_history = np.zeros((4, 4))
        self.phi_dot = np.zeros(4)
        self.omega = math.pi * 4
        
        self.robot._robot.set_kp(25.)
        self.robot._robot.set_kd(0.5)

        self.command = np.zeros(self.command_dim, dtype=np.float32)

        self.jpos_multistep = np.zeros((self.obs_cfg.jpos_steps, 12))
        self.jvel_multistep = np.zeros((self.obs_cfg.jvel_steps, 12))
        self.gyro_multistep = np.zeros((self.obs_cfg.gyro_steps, 3))
        self.gravity_multistep = np.zeros((self.obs_cfg.gravity_steps, 3))
        self.target_yaw = self.robot.robot_state.rpy[2]
    
    def update_command(self):
        mass = 3 
        lin_kp = 24 + 16 * self.robot.robot_state.rxy[1]
        lin_kd = 2. * math.sqrt(lin_kp)

        ang_kp = 24 # lin_kp
        ang_kd = 2. * math.sqrt(ang_kp)

        # self.command[0] = self.robot.robot_state.lxy[1] * lin_kd / lin_kp
        # self.command[1] = - self.robot.robot_state.lxy[0] * lin_kd / lin_kp
        # self.target_yaw = self.target_yaw - 0.02 * self.robot.robot_state.rxy[0]
        # self.target_yaw = self.target_yaw * 0.98 + self.robot.robot_state.rpy[2] * 0.02
        print(self.target_yaw)

        yaw_diff = wrap_to_pi(self.target_yaw - self.robot.robot_state.rpy[2])

        self.command[0] = 2.2 * self.robot.robot_state.lxy[1]
        self.command[1] = - self.robot.robot_state.lxy[0]
        self.command[2] = yaw_diff
        self.command[3:5] = self.command[:2] * lin_kp
        self.command[5:8] = lin_kd
        self.command[8] = ang_kp * yaw_diff # yaw
        self.command[9:10] = mass

        dt = 0.02
        move = True # np.abs(self.command[:3]).sum() > 0.1
        if move:
            dphi = self.omega # + self.trot(self.phi)
        else:
            dphi = self.stand(self.phi)
        self.phi_dot[:] = dphi
        self.phi = (self.phi + self.phi_dot * dt) % (2 * np.pi)
        self.phi_history = np.roll(self.phi_history, 1, axis=0)
        self.phi_history[0] = self.phi

        if self.oscillator_history:
            phi_sin = np.sin(self.phi_history)
            phi_cos = np.cos(self.phi_history)
        else:
            phi_sin = np.sin(self.phi)
            phi_cos = np.cos(self.phi)
        
        osc = np.concatenate([phi_sin, phi_cos, self.phi_dot], axis=-1)
        self.command[10:] = osc.reshape(-1)
        return self.command

    def compute_obs(self):
        # common
        self.jpos_multistep = np.roll(self.jpos_multistep, shift=1, axis=0)
        self.jpos_multistep[0] = self.robot.robot_state.jpos
        self.jvel_multistep = np.roll(self.jvel_multistep, shift=1, axis=0)
        self.jvel_multistep[0] = self.robot.robot_state.jvel

        self.gyro_multistep = np.roll(self.gyro_multistep, shift=1, axis=0)
        self.gyro_multistep[0] = self.robot.robot_state.angvel

        self.gravity_multistep = np.roll(self.gravity_multistep, shift=1, axis=0)
        self.gravity_multistep[0] = self.robot.robot_state.projected_gravity

        jpos_multistep = self.jpos_multistep.copy()
        jvel_multistep = self.jvel_multistep.copy()

        obs = [
            self.gravity_multistep.reshape(-1),
            jpos_multistep.reshape(-1),
            jvel_multistep.reshape(-1),
            self.robot.action_buf[:, :3].reshape(-1),
        ]
        obs = np.concatenate(obs, dtype=np.float32)
        return obs
    
    def trot(self,phi: torch.Tensor):
        dphi = np.zeros(4)
        dphi[0] = (phi[3] - phi[0]) # + ((phi[1] + math.pi - phi[0]) % (2 * math.pi))
        dphi[1] = (phi[2] - phi[1]) + ((phi[0] + math.pi - phi[1]) % (2 * math.pi))
        dphi[2] = (phi[1] - phi[2]) + ((phi[0] + math.pi - phi[2]) % (2 * math.pi))
        dphi[3] = (phi[0] - phi[3]) # + ((phi[1] + math.pi - phi[3]) % (2 * math.pi))
        return dphi

    def stand(self, phi: torch.Tensor, target=math.pi * 3 / 2):
        dphi = 2.0 * ((target - phi) % (2 * math.pi))
        return dphi


def loop_rate(loop_cnt: mp.Value, policy_cnt: mp.Value):
    timer = Timer(1.0)
    while True:
        print(f"Loop freq: {loop_cnt.value}, Policy freq: {policy_cnt.value}")
        loop_cnt.value = 0
        policy_cnt.value = 0
        timer.sleep()


def wrap_to_pi(angles):
    r"""Wraps input angles (in radians) to the range :math:`[-\pi, \pi]`.

    This function wraps angles in radians to the range :math:`[-\pi, \pi]`, such that
    :math:`\pi` maps to :math:`\pi`, and :math:`-\pi` maps to :math:`-\pi`. In general,
    odd positive multiples of :math:`\pi` are mapped to :math:`\pi`, and odd negative
    multiples of :math:`\pi` are mapped to :math:`-\pi`.

    The function behaves similar to MATLAB's `wrapToPi <https://www.mathworks.com/help/map/ref/wraptopi.html>`_
    function.

    Args:
        angles: Input angles of any shape.

    Returns:
        Angles in the range :math:`[-\pi, \pi]`.
    """
    # wrap to [0, 2*pi)
    wrapped_angle = (angles + np.pi) % (2 * np.pi)
    # map to [-pi, pi]
    # we check for zero in wrapped angle to make it go to pi when input angle is odd multiple of pi
    return np.where((wrapped_angle == 0) & (angles > 0), np.pi, wrapped_angle - np.pi)


@torch.inference_mode()
# @set_exploration_type(ExplorationType.MODE)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default=None)
    parser.add_argument("-l", "--log", action="store_true", default=False)
    args = parser.parse_args()

    timestr = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
    setproctitle("play_go2")

    init_channel("eth0")

    init_pos = np.array([
        0.0, 0.9, -1.8, 
        0.0, 0.9, -1.8, 
        0.0, 0.9, -1.8, 
        0.0, 0.9, -1.8
    ])

    if args.log:
        os.makedirs("logs", exist_ok=True)
        log_file = h5py.File(f"logs/{timestr}.h5py", "a")
    else:
        log_file = None

    robot = Go2Iface({})
    # client = ImpedanceControl(robot, BasicObsCfg())    
    client = VelocityControl(robot, BasicObsCfg())

    path = args.path
    if path is None:
        policy = None
    elif path.endswith(".onnx"):
        backend = "onnx"
        policy_module = ONNXModule(path)
        def policy(inp):
            out = policy_module(inp)
            action = out["action"].reshape(-1)
            carry = {k[1]: v for k, v in out.items() if k[0] == "next"}
            return action, carry
    else:
        raise NotImplementedError
        backend = "torch"
        policy_module = torch.load(path)
        policy_module.module[0].set_missing_tolerance(True)
        def policy(inp):
            inp = TensorDict(tree_map(torch.as_tensor, inp), [1])
            out = policy_module(inp)
            action = out["action"].numpy().reshape(-1)
            carry = dict(out["next"])
            return action, carry

    cmd = client.update_command()
    obs = client.compute_obs()
    
    loop_cnt = mp.Value("i", 0)
    policy_cnt = mp.Value("i", 0)
    
    mp.Process(target=loop_rate, args=(loop_cnt, policy_cnt)).start()

    try:
        inp = {
            "is_init": np.array([True]),
            "adapt_hx": np.zeros((1, 128), dtype=np.float32),
            "context_adapt_hx": np.zeros((1, 128), dtype=np.float32),
        }
        timer = Timer(0.005)
        for i in itertools.count():
            
            iter_start = time.perf_counter()
            robot.update_state()

            if i % 4 == 0:
                cmd = client.update_command()
                obs = client.compute_obs()
                inp["command_"]  = cmd[None, ...]
                inp["policy"]   = obs[None, ...]
                inp["is_init"]  = np.array([False], dtype=bool)
                
                action, carry = policy(inp)
                robot.apply_action(action, alpha=0.7)
                inp = carry
                policy_cnt.value += 1
            
            if i % 200 == 0:
                print("gravity:", robot.robot_state.projected_gravity)
                print("action:", action)
            loop_cnt.value += 1
            timer.sleep()

    except KeyboardInterrupt:
        print("End")
        
if __name__ == "__main__":
    main()

    
