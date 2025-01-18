import numpy as np
import h5py
import time
import itertools
import threading

from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass

from go2deploy.go2py import RobotIface
from go2deploy.utils import lerp
from go2deploy.filters import SecondOrderLowPassFilter

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

@dataclass
class RobotState:
    jpos: np.ndarray = np.zeros(12)
    jvel: np.ndarray = np.zeros(12)

    rpy: np.ndarray = np.zeros(3)
    quat: np.ndarray = np.zeros(4)
    projected_gravity: np.ndarray = np.zeros(3)

    angvel: np.ndarray = np.zeros(3)
    acc: np.ndarray = np.zeros(3)

    lxy: np.ndarray = np.zeros(2)
    rxy: np.ndarray = np.zeros(2)

class Go2Iface:

    smoothing_length: int = 5
    smoothing_ratio: float = 0.4
    command_dim: int

    def __init__(self, cfg, log_file: h5py.File=None):
        self.cfg = cfg
        self.log_file = log_file
        
        self._robot = RobotIface()
        self._robot.start_control(interval=2000)
        
        self.default_joint_pos = np.array(
            [
                0.1, -0.1,  0.1, -0.1,  
                0.7,  0.7,  0.8,  0.8, 
                -1.5, -1.5, -1.5, -1.5
            ], 
        )
        self.action_scaling = 0.5
        
        self.robot_state = RobotState()
        self.action_buf = np.zeros((12, 4), dtype=np.float32)
        self.applied_action = np.zeros(12, dtype=np.float32)

        self.step_count = 0
        
        self.update_state()
        self.filter_command = SecondOrderLowPassFilter(50, 400)
        self.jpos_target_sdk = self.default_joint_pos[isaac2sdk]
        
        if self.log_file is not None:
            default_len = 50 * 60
            self.log_file.attrs["cursor"] = 0
            log_file.create_dataset("control_mode", (default_len, 1), maxshape=(None, 1))
            log_file.create_dataset("command", (default_len, self.command_dim), maxshape=(None, self.command_dim))
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
        
        self.command_thread = threading.Thread(target=self._write_cmd)
        self.command_thread.start()

    def update_state(self):
        self._robot_state = self._robot.get_robot_state()

        self.robot_state.jpos = np.asarray(self._robot_state.jpos)[sdk2isaac]
        self.robot_state.jvel = np.asarray(self._robot_state.jvel)[sdk2isaac]
        
        self.robot_state.rpy = np.asarray(self._robot_state.rpy)
        self.robot_state.angvel = np.asarray(self._robot_state.gyro)
        self.rot = R.from_quat(self._robot_state.quat, scalar_first=True)
        self.robot_state.projected_gravity = self.rot.inv().apply(np.array([0., 0., -1.]))

        self.robot_state.lxy = lerp(self.robot_state.lxy, self._robot.lxy(), 0.5)
        self.robot_state.rxy = lerp(self.robot_state.rxy, self._robot.rxy(), 0.5)
    
    def apply_action(self, action: np.ndarray, alpha: float = 0.7):
        action = action.clip(-6, 6)
        self.action_buf[:, 1:] = self.action_buf[:, :-1]
        self.action_buf[:, 0] = action
        self.applied_action = self.applied_action * (1-alpha) + action * alpha
        jpos_target = self.applied_action * self.action_scaling + self.default_joint_pos
        self.jpos_target_sdk = jpos_target[isaac2sdk]
        self.step_count += 1
    
    def _write_cmd(self):
        t0 = time.perf_counter()
        for i in itertools.count():
            jpos_target = self.filter_command.update(self.jpos_target_sdk)
            # jpos_target = self.jpos_target_sdk
            self._robot.set_command(jpos_target)
            time.sleep(0.005)
            if i % 200 == 0:
                print("Cmd write freq: ", i / (time.perf_counter() - t0))

    def _send_command(self):
        jpos_target = self.filter.update(self.jpos_target_sdk)
        jpos_target = self.jpos_target_sdk
        self._robot.set_command(jpos_target)
    
    def _maybe_log(self):
        if self.log_file is None:
            return
        self.log_file["control_mode"][self.step_count] = self.robot_state.control_mode
        self.log_file["command"][self.step_count] = self.command
        self.log_file["observation"][self.step_count] = self.obs
        
        # imu readings
        self.log_file["rpy"][self.step_count] = self.robot_state.rpy
        self.log_file["quat"][self.step_count] = self.robot_state.quat
        self.log_file["angvel"][self.step_count] = self.robot_state.angvel
        self.log_file["acc"][self.step_count] = self.robot_state.acc
        
        # joint readings
        self.log_file["jpos"][self.step_count] = self.robot_state.jpos
        self.log_file["jvel"][self.step_count] = self.robot_state.jvel
        self.log_file["jpos_des"][self.step_count] = self.robot_state.jpos_des
        
        # others
        # self.log_file["foot_force"][self.step_count] = self.robot_state.foot_force

        # self.log_file["tau_est"][self.step_count] = self.tau_sim
        self.log_file.attrs["cursor"] = self.step_count

        if self.step_count == self.log_file["jpos"].len() - 1:
            new_len = self.step_count + 1 + 3000
            print(f"Extend log size to {new_len}.")
            for key, value in self.log_file.items():
                value.resize((new_len, value.shape[1]))

