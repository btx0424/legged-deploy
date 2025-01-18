import numpy as np
import math

from go2deploy.robot import Go2Iface
from dataclasses import dataclass

@dataclass
class BasicObsCfg:


    jpos_steps: int = 3
    jvel_steps: int = 3
    gyro_steps: int = 3
    gravity_steps: int = 3


class VelocityControl:

    oscillator_history: bool = False
    command_dim: int = 4 + 12

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
    
    def update_command(self):
        self.command[0] = self.robot.robot_state.lxy[1]
        self.command[1] = - self.robot.robot_state.lxy[0]
        self.command[2] = - self.robot.robot_state.rxy[0]
        self.command[3] = 0.75

        dt = 0.02
        move = np.abs(self.command[:3]).sum() > 0.08
        
        if move:
            self.phi_dot = self.omega + self.trot(self.phi, self.phi_dot)
        else:
            self.phi_dot = self.stand(self.phi, self.phi_dot)
        self.phi = (self.phi + self.phi_dot * dt)
        if (self.phi > np.pi * 2).all():
            self.phi -= np.pi * 2
        
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
            self.gravity_multistep[0],
            jpos_multistep[0],
            jvel_multistep[0],
            self.robot.action_buf[:, :3].reshape(-1),
        ]
        obs = np.concatenate(obs, dtype=np.float32)
        return obs
    
    def trot(self, phi, phi_dot):
        phi_dot = np.zeros((1, 4))
        phi = phi[None, :]
        phi_dot[:, 0] = (phi[:, 3] - phi[:, 0]) + (phi[:, 1] + np.pi - phi[:, 0]) 
        phi_dot[:, 1] = (phi[:, 2] - phi[:, 1]) + (phi[:, 0] - np.pi - phi[:, 1]) 
        phi_dot[:, 2] = (phi[:, 1] - phi[:, 2]) + (phi[:, 0] - np.pi - phi[:, 2])
        phi_dot[:, 3] = (phi[:, 0] - phi[:, 3]) + (phi[:, 1] + np.pi - phi[:, 3])
        return phi_dot[0]
    
    def stand(self, phi, phi_dot):
        dt = 0.02
        two_pi = np.pi * 2
        target = np.pi * 3 / 2
        a = ((phi % two_pi) < target - 1e-4) & (((phi + phi_dot * dt) % two_pi) > target + 1e-4)
        b = np.abs((phi % two_pi) - target) < 1e-4
        phi_dot = np.where(a, (((target - phi) % two_pi) / dt), phi_dot)
        return phi_dot * (~b)
