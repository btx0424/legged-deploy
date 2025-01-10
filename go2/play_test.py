
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

from setproctitle import setproctitle
from scipy.spatial.transform import Rotation as R
from go2deploy import ONNXModule, init_channel, RobotIface, SecondOrderLowPassFilter
from go2deploy.utils import lerp, normalize
from torch.utils._pytree import tree_map
from timer_fd import Timer

try:
    from tensordict import TensorDict
    from torchrl.envs.utils import set_exploration_type, ExplorationType
except ModuleNotFoundError:
    pass

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

ORBIT_JOINT_ORDER = [
    'FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint', 
    'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint', 
    'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint'
]

SDK_JOINT_ORDER = [
    'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
    'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
    'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
    'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint'
]

sdk2isaac = [SDK_JOINT_ORDER.index(name) for name in ORBIT_JOINT_ORDER]
isaac2sdk = [ORBIT_JOINT_ORDER.index(name) for name in SDK_JOINT_ORDER]

class Go2Iface:

    smoothing_length: int = 5
    smoothing_ratio: float = 0.4
    command_dim: int

    def __init__(self, cfg, log_file: h5py.File=None):
        self.cfg = cfg
        self.log_file = log_file

        self.acc_bias = np.array([0.851, 0.310, 9.580])

        self._robot = RobotIface()
        self._robot.start_control(interval=2000)
        self.default_joint_pos = np.array(
            [
                0.1, -0.1,  0.1, -0.1,  
                0.7,  0.7,  0.8,  0.8, 
                -1.5, -1.5, -1.5, -1.5
            ], 
        )
        
        self.dt = 0.02
        self.latency = 0.0
        self.rpy = np.zeros(3)
        self.angvel_history = np.zeros((3, self.smoothing_length))
        self.projected_gravity_history = np.zeros((3, self.smoothing_length))
        self.angvel = np.zeros(3)
        self.action_buf = np.zeros((12, 4))
        self.command = np.zeros(self.command_dim, dtype=np.float32)
        self.lxy = 0.
        self.rxy = 0.
        self.action_buf_steps = 1
        self.last_action = np.zeros(12)
        self.start_t = time.perf_counter()
        self.timestamp = time.perf_counter()
        self.step_count = 0

        self.jpos_sdk = np.zeros(12)
        self.jvel_sdk = np.zeros(12)
        self.jpos_sdk_substep = np.zeros((6, 12))
        self.jvel_sdk_substep = np.zeros((6, 12))

        self.rpy = np.zeros(3)
        self.gyro = np.zeros(3)
        self.rpy_prev = np.zeros(3)
        self.rpy_substep = np.zeros((2, 3))
        self.angvel_substep = np.zeros((2, 3))
        self.smoothing_substeps = 4
        self.gyro_substep = np.zeros((self.smoothing_substeps, 3))
        self.acc_substep = np.zeros((self.smoothing_substeps, 3))

        self.acc = np.zeros(3)
        self.rot = R.from_quat([1, 0, 0, 0], scalar_first=True)
        
        self.robot_state = self._robot.get_robot_state()
        self.update_state()
        self.update_command()
        _obs = self._compute_obs()
        self.obs_dim = _obs.shape[0]
        self.obs_buf = np.zeros((self.obs_dim * 6))
        self.filter = SecondOrderLowPassFilter(50, 400)
        self.filter_jvel = SecondOrderLowPassFilter(50, 400)
        self.filter_rpy = SecondOrderLowPassFilter(50, 200)
        self.filter_acc = SecondOrderLowPassFilter(50, 200)
        self.jpos_target_sdk = self.default_joint_pos[isaac2sdk]
        
        if self.log_file is not None:
            default_len = 50 * 60
            self.log_file.attrs["cursor"] = 0
            log_file.create_dataset("control_mode", (default_len, 1), maxshape=(None, 1))
            log_file.create_dataset("command", (default_len, self.command_dim), maxshape=(None, self.command_dim))
            log_file.create_dataset("observation", (default_len, self.obs_dim), maxshape=(None, self.obs_dim))
            log_file.create_dataset("action", (default_len, 12), maxshape=(None, 12))

            # imu readings
            log_file.create_dataset("rpy", (default_len, 3), maxshape=(None, 3))
            log_file.create_dataset("quat", (default_len, 4), maxshape=(None, 4))
            log_file.create_dataset("gravity", (default_len, 3), maxshape=(None, 3))
            log_file.create_dataset("angvel", (default_len, 3), maxshape=(None, 3))
            log_file.create_dataset("acc", (default_len, 3), maxshape=(None, 3))
            log_file.create_dataset("linvel", (default_len, 3), maxshape=(None, 3))

            # joint readings
            log_file.create_dataset("jpos", (default_len, 12), maxshape=(None, 12))
            log_file.create_dataset("jvel", (default_len, 12), maxshape=(None, 12))
            log_file.create_dataset("jpos_des", (default_len, 12), maxshape=(None, 12))
            log_file.create_dataset("tau_est", (default_len, 12), maxshape=(None, 12))

            # others
            log_file.create_dataset("foot_force", (default_len, 4), maxshape=(None, 4))

    def reset(self):
        self.start_t = time.perf_counter()
        self.update_state()
        self.update_command()
        return self._compute_obs()
    
    def _update_state(self, i):
        self.robot_state = self._robot.get_robot_state()
        self.jpos_sdk = np.asarray(self.robot_state.jpos)
        self.jvel_sdk = np.asarray(self.robot_state.jvel)
        self.jpos_sdk_substep = np.roll(self.jpos_sdk_substep, shift=-1, axis=0)
        self.jpos_sdk_substep[-1] = self.jpos_sdk
        self.jvel_sdk_substep = np.roll(self.jvel_sdk_substep, shift=-1, axis=0)
        self.jvel_sdk_substep[-1] = self.jvel_sdk
        
        self.rpy = self.filter_rpy.update(np.asarray(self.robot_state.rpy))
        self.rot = R.from_quat(self.robot_state.quat, scalar_first=True)
        # self.gyro = np.asarray(self.robot_state.gyro)
        self.gyro_substep[i % self.smoothing_substeps] = np.asarray(self.robot_state.gyro)
        self.acc_substep[i % self.smoothing_substeps] = self.filter_acc.update(np.asarray(self.robot_state.acc))
        
    def update_state(self):
        # self.jvel_sdk = self.jvel_sdk_substep.mean(0)
        # self.jvel_sdk = np.diff(self.jpos_sdk_substep, axis=0).mean(0) / 0.005
        self.jpos_sdk = self.jpos_sdk_substep.mean(0)
        self.jvel_sdk = self.jvel_sdk_substep.mean(0)
        self.jpos_isaac = self.jpos_sdk[sdk2isaac]
        self.jvel_isaac = self.jvel_sdk[sdk2isaac]
        self.rpy_prev = self.rpy
        # self.rpy = self.rpy_substep.mean(0)
        self.angvel = (self.rpy - self.rpy_prev) / 0.02
        self.gyro = self.gyro_substep.mean(0)
        self.acc = self.acc_substep.mean(0)
        self.acc = np.where(self.acc < 0.1, 0., self.acc)

        self.gravity = self.rot.inv().apply([0., 0., -1.])

        self.lxy = lerp(self.lxy, self._robot.lxy(), 0.5)
        self.rxy = lerp(self.rxy, self._robot.rxy(), 0.5)
    
    def apply_action(self, action: np.ndarray):
        self.action_buf[:, 1:] = self.action_buf[:, :-1]
        self.action_buf[:, 0] = action.clip(-6, 6)
        self.last_action = self.last_action * 0.5 + self.action_buf[:, 0] * 0.5

        jpos_target = self.last_action * 0.5 + self.default_joint_pos
        self.jpos_target_sdk = self.orbit_to_sdk(jpos_target)

        self._maybe_log()
        self.step_count += 1
    
    def _write_cmd(self):
        t0 = time.perf_counter()
        for i in itertools.count():
            jpos_target = self.filter.update(self.jpos_target_sdk)
            # jpos_target = self.jpos_target_sdk
            self._robot.set_command(jpos_target)
            time.sleep(0.005)
            if i % 200 == 0:
                print("Cmd write freq: ", i / (time.perf_counter() - t0))

    def _send_command(self):
        jpos_target = self.filter.update(self.jpos_target_sdk)
        # jpos_target = self.jpos_target_sdk
        self._robot.set_command(jpos_target)

    def get_obs(self):
        self.update_state()
        self.update_command()
        self.obs = self._compute_obs()
        # self.obs_buf = np.roll(self.obs_buf, -self.obs_dim)
        # self.obs_buf[-self.obs_dim:] = obs
        return self.obs
        
    def update_command(self):
        pass

    def _compute_obs(self):
        raise NotImplementedError
    
    def _maybe_log(self):
        if self.log_file is None:
            return
        self.log_file["control_mode"][self.step_count] = self.robot_state.control_mode
        self.log_file["command"][self.step_count] = self.command
        self.log_file["observation"][self.step_count] = self.obs
        self.log_file["action"][self.step_count] = self.action_buf[:, 0]
        
        # imu readings
        self.log_file["rpy"][self.step_count] = self.robot_state.rpy
        self.log_file["quat"][self.step_count] = self.robot_state.quat
        self.log_file["angvel"][self.step_count] = self.robot_state.gyro
        self.log_file["acc"][self.step_count] = self.robot_state.acc
        
        # joint readings
        self.log_file["jpos"][self.step_count] = self.jpos_isaac
        self.log_file["jvel"][self.step_count] = self.jvel_isaac
        self.log_file["jpos_des"][self.step_count] = self.robot_state.jpos_des
        
        # others
        self.log_file["foot_force"][self.step_count] = self.robot_state.foot_force

        # self.log_file["tau_est"][self.step_count] = self.tau_sim
        self.log_file.attrs["cursor"] = self.step_count

        if self.step_count == self.log_file["jpos"].len() - 1:
            new_len = self.step_count + 1 + 3000
            print(f"Extend log size to {new_len}.")
            for key, value in self.log_file.items():
                value.resize((new_len, value.shape[1]))

    @staticmethod
    def orbit_to_sdk(joints: np.ndarray):
        return joints[isaac2sdk]
        return np.flip(joints.reshape(3, 2, 2), axis=2).transpose(1, 2, 0).reshape(-1)
    
    @staticmethod
    def sdk_to_orbit(joints: np.ndarray):
        return joints[sdk2isaac]
        return np.flip(joints.reshape(2, 2, 3), axis=1).transpose(2, 0, 1).reshape(-1)

    def process_action(self, action: np.ndarray):
        return self.orbit_to_sdk(action * 0.5 + self.default_joint_pos)
    
    def process_action_inv(self, jpos_sdk: np.ndarray):
        return (self.sdk_to_orbit(jpos_sdk) - self.default_joint_pos) / 0.5



class Go2Vel(Go2Iface):
    
    command_dim: int = 4 # linvel_xy, angvel_z, base_height

    def update_command(self):
        t = time.perf_counter() - self.start_t
        vx = np.sin(t * 0.75)
        self.command[0] = lerp(self.command[0] * 0.95, self.lxy[1] * 1.0, 0.2)
        self.command[1] = lerp(self.command[1], -self.lxy[0], 0.2)

        self.command[2] = -self.rxy[0] * 1.2
        self.command[3] = 0.75 #

    def _compute_obs(self):
        # self.rot = R.from_euler("xyz", self.rpy)
        # angvel = self.rot.inv().apply(self.angvel)
        angvel = self.angvel
        
        obs = [
            self.command,
            # angvel,
            self.projected_gravity,
            self.jpos_sim,
            self.jvel_sim,
            self.action_buf[:, :3].reshape(-1),
        ]
        obs = np.concatenate(obs, dtype=np.float32)
        return obs


class Go2Impd(Go2Iface):

    oscillator_history: bool = False
    command_dim: int = 10
    # + 3 * 4
    command_dim: int = 10 + 3 * 4

    def __init__(self, cfg, log_file = None):
        self.phi = np.zeros(4)
        self.phi[0] = np.pi
        self.phi[3] = np.pi
        self.phi_history = np.zeros((4, 4))
        self.phi_dot = np.zeros(4)

        self.jpos_multistep = np.zeros((3, 12))
        self.jvel_multistep = np.zeros((3, 12))
        self.gyro_multistep = np.zeros((3, 3))
        self.gravity_multistep = np.zeros((3, 3))


        super().__init__(cfg, log_file)
    
    def update_command(self):
        mass = 3 
        lin_kp = 24 + 16 * self.rxy[1]
        lin_kd = 2. * math.sqrt(lin_kp)

        ang_kp = 16.
        ang_kd = 2. * math.sqrt(ang_kp)
        # self.command[0] = max(self.lxy[1], 0.)
        # self.command[1] = - self.lxy[0]
        # self.command[2] = - 2.0 * self.rxy[0]
        # self.command[3:5] = self.command[:2] * kp
        # self.command[5:6] = kd
        # self.command[6:7] = kp
        # self.command[7:8] = kd
        # self.command[8:9] = 3.0
        # rpy = self.rot.as_euler("xyz")


        # OLD Command, fixed sampled setpoint
        self.command[0] = self.lxy[1] * 2.0 # * lin_kd / lin_kp
        self.command[1] = - self.lxy[0] * 2.0 # lin_kd / lin_kp
        self.command[2] = 0.0 # 0.2 * self.rxy[1] - rpy[1] # pitch
        self.command[3:5] = self.command[:2] * lin_kp
        self.command[5:8] = lin_kd
        self.command[8] = - 1.0 * self.rxy[0] # yaw
        self.command[9:10] = mass

        # New command with pitch
        # self.command[0] = self.lxy[1] * lin_kd / lin_kp
        # self.command[1] = - self.lxy[0] * lin_kd / lin_kp
        # self.command[2:4] = 0.0 # 0.2 * self.rxy[1] - rpy[1] # pitch
        # self.command[4:6] = self.command[:2] * lin_kp
        # self.command[6:7] = lin_kd
        # self.command[7:9] = 0.0
        # self.command[9:10] = lin_kd
        # self.command[10:11] = mass
        # self.command[11] = 1.
        # self.command[12] = 0.

        omega = math.pi * 4
        dt = 0.02
        move = True # np.abs(self.command[:3]).sum() > 0.1
        if move:
            dphi = omega + self.trot(self.phi)
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

    def update_state(self):
        super().update_state()
        # common
        self.jpos_multistep = np.roll(self.jpos_multistep, shift=1, axis=0)
        self.jpos_multistep[0] = self.jpos_sdk[sdk2isaac]
        self.jvel_multistep = np.roll(self.jvel_multistep, shift=1, axis=0)
        self.jvel_multistep[0] = self.jvel_sdk[sdk2isaac]
        self.gyro_multistep = np.roll(self.gyro_multistep, shift=1, axis=0)
        self.gyro_multistep[0] = self.gyro
        self.gravity_multistep = np.roll(self.gravity_multistep, shift=1, axis=0)
        self.gravity_multistep[0] = self.gravity

    def _compute_obs(self):
        jpos_multistep = self.jpos_multistep.copy()
        # jpos_multistep[1:] = self.jpos_multistep[1:] - self.jpos_multistep[:-1]
        jvel_multistep = self.jvel_multistep.copy()
        # jvel_multistep[1:] = self.jvel_multistep[1:] - self.jvel_multistep[:-1]
        obs = [
            # self.gyro_multistep.reshape(-1),
            # self.gravity,
            self.gravity_multistep.reshape(-1),
            jpos_multistep.reshape(-1),
            jvel_multistep.reshape(-1),
            self.action_buf[:, :3].reshape(-1),
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


class Go2Loco(Go2Iface):

    oscillator_history: bool = False
    command_dim: int = 4 + 4 * 2 + 4 # linvel_xy, angvel_z, base_height

    def __init__(self, cfg, log_file = None):
        # oscillators
        self.phi = np.zeros(4)
        # self.phi[0] = np.pi
        # self.phi[3] = np.pi
        self.phi_history = np.zeros((4, 4)) # [t, legs]
        self.phi_dot = np.zeros(4)

        self.jpos_multistep = np.zeros((4, 12))
        self.jvel_multistep = np.zeros((4, 12))
        self.gyro_multistep = np.zeros((4, 3))

        super().__init__(cfg, log_file)

    def update_state(self):
        super().update_state()
        # common
        self.jpos_multistep = np.roll(self.jpos_multistep, shift=1, axis=0)
        self.jpos_multistep[0] = self.jpos_sdk[sdk2isaac]
        self.jvel_multistep = np.roll(self.jvel_multistep, shift=1, axis=0)
        self.jvel_multistep[0] = self.jvel_sdk[sdk2isaac]
        self.gyro_multistep = np.roll(self.gyro_multistep, shift=1, axis=0)
        # self.gyro_multistep[0] = self.robot_state.gyro
        self.gyro_multistep[0] = self.gyro

    def _compute_obs(self):
        jpos_multistep = self.jpos_multistep.copy()
        jpos_multistep[1:] = self.jpos_multistep[1:] - self.jpos_multistep[:-1]
        jvel_multistep = self.jvel_multistep.copy()
        jvel_multistep[1:] = self.jvel_multistep[1:] - self.jvel_multistep[:-1]
        obs = [
            # self.gyro_multistep.reshape(-1),
            self.gravity,
            jpos_multistep.reshape(-1),
            jvel_multistep.reshape(-1),
            self.action_buf[:, :3].reshape(-1),
        ]
        obs = np.concatenate(obs, dtype=np.float32)
        return obs

    def update_command(self):
        t = time.perf_counter() - self.start_t
        vx = np.sin(t * 0.75)
        self.command[0] = lerp(self.command[0] * 0.95, self.lxy[1] * 1.0, 0.2)
        self.command[1] = lerp(self.command[1], -self.lxy[0], 0.2)
        
        dt = 0.02
        self.command[2] = -self.rxy[0] * 2.0
        self.command[3] = 0.75 #
        
        omega = math.pi * 4
        move = True # np.abs(self.command[:3]).sum() > 0.1
        if move:
            dphi = omega + self.trot(self.phi)
        else:
            dphi = self.stand(self.phi)
        # self.phi_dot[:] = dphi
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
        self.command[4:] = osc.reshape(-1)
        return self.command

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


@torch.inference_mode()
# @set_exploration_type(ExplorationType.MODE)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str)
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

    robot = Go2Impd({}, log_file)
    
    robot._robot.set_kp(25.)
    robot._robot.set_kd(0.5)
    robot._robot.lerp_command = False

    gyro_pred = np.zeros(3)
    path = args.path
    if path.endswith(".onnx"):
        backend = "onnx"
        policy_module = ONNXModule(path)
        def policy(inp):
            out = policy_module(inp)
            action = out["action"].reshape(-1)
            carry = {k[1]: v for k, v in out.items() if k[0] == "next"}
            # gyro_pred[:] = out[("info", "ext_rec")]
            # print(out[("info", "ext_rec")])
            return action, carry
    else:
        backend = "torch"
        policy_module = torch.load(path)
        policy_module.module[0].set_missing_tolerance(True)
        def policy(inp):
            inp = TensorDict(tree_map(torch.as_tensor, inp), [1])
            out = policy_module(inp)
            action = out["action"].numpy().reshape(-1)
            carry = dict(out["next"])
            return action, carry

    robot._robot.set_command(init_pos)
    obs = robot.reset()
    obs = robot.get_obs()
    print(obs.shape)

    import zmq
    import threading
    def pub():
        context = zmq.Context()
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        socket.bind("tcp://*:5555")
        while True:
            robot_state = robot._robot.get_robot_state()
            socket.send_pyobj({
                "jpos": robot.jpos_isaac[isaac2sdk],
                "jpos_des": robot_state.jpos_des,
                # "jpos": robot_state.jvel,
                # "jpos_des": robot_state.jvel_raw,
                "rpy": robot_state.gyro,
                "gyro": robot.gyro,
                "acc": robot_state.acc,
                "foot_force": np.asarray(robot_state.foot_force, dtype=float),
            })
            time.sleep(0.005)

    # threading.Thread(target=pub).start()
    threading.Thread(target=robot._write_cmd).start()
    
    loop_cnt = mp.Value("i", 0)
    policy_cnt = mp.Value("i", 0)
    
    mp.Process(target=loop_rate, args=(loop_cnt, policy_cnt)).start()

    try:
        inp = {
            "command": robot.command[None, ...],
            "policy": obs[None, ...],
            "is_init": np.array([True]),
            "adapt_hx": np.zeros((1, 128), dtype=np.float32),
            "context_adapt_hx": np.zeros((1, 128), dtype=np.float32),
        }
        
        timer = Timer(0.005)
        for i in itertools.count():
            iter_start = time.perf_counter()
            robot._update_state(i)
            if i % 4 == 0:
                robot.update_state()
                obs = robot.get_obs()
                inp["command_"]  = robot.command[None, ...]
                inp["policy"]   = obs[None, ...]
                inp["is_init"]  = np.array([False], dtype=bool)
                
                action, carry = policy(inp)
                robot.apply_action(action)
                inp = carry
                policy_cnt.value += 1

            if i % 200 == 0:
                print(robot.command)
            # robot._send_command()
            loop_cnt.value += 1
            timer.sleep()

    except KeyboardInterrupt:
        print("End")
        
if __name__ == "__main__":
    main()

    
