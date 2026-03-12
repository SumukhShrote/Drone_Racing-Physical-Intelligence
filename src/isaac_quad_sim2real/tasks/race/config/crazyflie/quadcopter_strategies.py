# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Modular strategy classes for quadcopter environment rewards, observations, and resets."""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from isaaclab.utils.math import subtract_frame_transforms, quat_from_euler_xyz, euler_xyz_from_quat, wrap_to_pi, matrix_from_quat

if TYPE_CHECKING:
    from .quadcopter_env import QuadcopterEnv

D2R = np.pi / 180.0
R2D = 180.0 / np.pi


class DefaultQuadcopterStrategy:
    """Strategy implementation for quadcopter racing environment."""

    def __init__(self, env: QuadcopterEnv):
        """Initialize the default strategy.

        Args:
            env: The quadcopter environment instance.
        """
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.cfg = env.cfg

        # Reward defaults. Anything in self.env.rew overrides these.
        self.reward_defaults = {
            "gate_pass_reward_scale": 15.0,
            "lap_reward_scale": 40.0,
            "progress_reward_scale": 3.0,
            "center_reward_scale": 0.5,
            "align_reward_scale": 0.25,
            "speed_reward_scale": 0.10,
            "action_mag_reward_scale": -0.01,
            "ang_vel_reward_scale": -0.005,
            "crash_reward_scale": -2.0,
            "time_penalty_reward_scale": -0.01,
        }
        self.death_cost_default = -10.0

        self.reward_terms = [
            "gate_pass",
            "lap",
            "progress",
            "center",
            "align",
            "speed",
            "action_mag",
            "ang_vel",
            "crash",
            "time_penalty",
        ]

        # Initialize episode sums for logging if in training mode
        if self.cfg.is_train:
            self._episode_sums = {
                key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for key in self.reward_terms
            }

        # Initialize fixed parameters once (no domain randomization)
        # These parameters remain constant throughout the simulation
        # Aerodynamic drag coefficients
        self.env._K_aero[:, :2] = self.env._k_aero_xy_value
        self.env._K_aero[:, 2] = self.env._k_aero_z_value

        # PID controller gains for angular rate control
        # Roll and pitch use the same gains
        self.env._kp_omega[:, :2] = self.env._kp_omega_rp_value
        self.env._ki_omega[:, :2] = self.env._ki_omega_rp_value
        self.env._kd_omega[:, :2] = self.env._kd_omega_rp_value

        # Yaw has different gains
        self.env._kp_omega[:, 2] = self.env._kp_omega_y_value
        self.env._ki_omega[:, 2] = self.env._ki_omega_y_value
        self.env._kd_omega[:, 2] = self.env._kd_omega_y_value

        # Motor time constants (same for all 4 motors)
        self.env._tau_m[:] = self.env._tau_m_value

        # Thrust to weight ratio
        self.env._thrust_to_weight[:] = self.env._twr_value

        # Performance-based curriculum state
        self._curriculum_alpha_value = 0.0
        self._curriculum_score_ema = 0.0
        self._curriculum_updates = 0
    
    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _reward_scale(self, key: str) -> float:
        if hasattr(self.env, "rew") and self.env.rew is not None and key in self.env.rew:
            return float(self.env.rew[key])
        return float(self.reward_defaults[key])

    def _death_cost(self) -> float:
        if hasattr(self.env, "rew") and self.env.rew is not None and "death_cost" in self.env.rew:
            return float(self.env.rew["death_cost"])
        return float(self.death_cost_default)

    def _curriculum_alpha(self) -> float:
        if not self.cfg.is_train:
            return 1.0
        return float(self._curriculum_alpha_value)

    def _sample_factor(self, nominal: float, min_factor: float, max_factor: float, alpha: float, n: int) -> torch.Tensor:
        low = nominal * (1.0 + alpha * (min_factor - 1.0))
        high = nominal * (1.0 + alpha * (max_factor - 1.0))
        lo = min(low, high)
        hi = max(low, high)
        return torch.empty(n, device=self.device).uniform_(lo, hi)
    
    def _update_curriculum_from_performance(self, env_ids: torch.Tensor):
        if not self.cfg.is_train or len(env_ids) == 0:
            return

        num_gates = float(self.env._waypoints.shape[0])
        gates_passed = self.env._n_gates_passed[env_ids].float()

        # Fraction of one lap completed, clipped to [0, 1]
        progress_frac = torch.clamp(gates_passed / max(num_gates, 1.0), 0.0, 1.0)

        # Whether the policy passed at least one gate in the episode
        passed_one_gate = (gates_passed > 0).float()

        # Simple blended score
        batch_score = 0.7 * progress_frac.mean().item() + 0.3 * passed_one_gate.mean().item()

        # EMA for stability
        if self._curriculum_updates == 0:
            self._curriculum_score_ema = batch_score
        else:
            self._curriculum_score_ema = 0.95 * self._curriculum_score_ema + 0.05 * batch_score

        self._curriculum_updates += 1

        # Map score to curriculum alpha
        self._curriculum_alpha_value = float(np.clip((self._curriculum_score_ema - 0.10) / 0.60, 0.0, 1.0))

    def _apply_domain_randomization(self, env_ids: torch.Tensor):
        n = len(env_ids)
        if not self.cfg.is_train:
            self.env._thrust_to_weight[env_ids] = self.env._twr_value
            self.env._K_aero[env_ids, :2] = self.env._k_aero_xy_value
            self.env._K_aero[env_ids, 2] = self.env._k_aero_z_value
            self.env._kp_omega[env_ids, :2] = self.env._kp_omega_rp_value
            self.env._ki_omega[env_ids, :2] = self.env._ki_omega_rp_value
            self.env._kd_omega[env_ids, :2] = self.env._kd_omega_rp_value
            self.env._kp_omega[env_ids, 2] = self.env._kp_omega_y_value
            self.env._ki_omega[env_ids, 2] = self.env._ki_omega_y_value
            self.env._kd_omega[env_ids, 2] = self.env._kd_omega_y_value
            self.env._tau_m[env_ids] = self.env._tau_m_value
            return

        alpha = self._curriculum_alpha()

        # Aligns with the project handout ranges, ramped from nominal to full range.
        self.env._thrust_to_weight[env_ids] = self._sample_factor(
            self.env._twr_value, 0.95, 1.05, alpha, n
        )

        kxy = self._sample_factor(self.env._k_aero_xy_value, 0.5, 2.0, alpha, n)
        kz = self._sample_factor(self.env._k_aero_z_value, 0.5, 2.0, alpha, n)
        self.env._K_aero[env_ids, :2] = kxy.unsqueeze(1)
        self.env._K_aero[env_ids, 2] = kz

        kp_rp = self._sample_factor(self.env._kp_omega_rp_value, 0.85, 1.15, alpha, n)
        ki_rp = self._sample_factor(self.env._ki_omega_rp_value, 0.85, 1.15, alpha, n)
        kd_rp = self._sample_factor(self.env._kd_omega_rp_value, 0.7, 1.3, alpha, n)

        kp_y = self._sample_factor(self.env._kp_omega_y_value, 0.85, 1.15, alpha, n)
        ki_y = self._sample_factor(self.env._ki_omega_y_value, 0.85, 1.15, alpha, n)
        kd_y = self._sample_factor(self.env._kd_omega_y_value, 0.7, 1.3, alpha, n)

        self.env._kp_omega[env_ids, :2] = kp_rp.unsqueeze(1)
        self.env._ki_omega[env_ids, :2] = ki_rp.unsqueeze(1)
        self.env._kd_omega[env_ids, :2] = kd_rp.unsqueeze(1)

        self.env._kp_omega[env_ids, 2] = kp_y
        self.env._ki_omega[env_ids, 2] = ki_y
        self.env._kd_omega[env_ids, 2] = kd_y

        self.env._tau_m[env_ids] = self.env._tau_m_value
    
    
    
    # ---------------------------------------------------------------------
    # Rewards
    # ---------------------------------------------------------------------
    def get_rewards(self) -> torch.Tensor:
        """get_rewards() is called per timestep. This is where you define your reward structure and compute them
        according to the reward scales you tune in train_race.py. The following is an example reward structure that
        causes the drone to hover near the zeroth gate. It will not produce a racing policy, but simply serves as proof
        if your PPO implementation works. You should delete it or heavily modify it once you begin the racing task."""

        # TODO ----- START ----- Define the tensors required for your custom reward structure
        # Current gate pose in gate frame, already refreshed by env before reward call.
        pose_gate = self.env._pose_drone_wrt_gate
        curr_x = pose_gate[:, 0]
        curr_y = pose_gate[:, 1]
        curr_z = pose_gate[:, 2]
        prev_x = self.env._prev_x_drone_wrt_gate.clone()

        num_gates = self.env._waypoints.shape[0]
        gate_half = 0.5 * float(self.cfg.gate_model.gate_side)
        gate_margin = 0.90 * gate_half

        yz_error = torch.sqrt(curr_y.pow(2) + curr_z.pow(2))
        inside_gate = (curr_y.abs() <= gate_margin) & (curr_z.abs() <= gate_margin)
        crossed_plane = (prev_x > 0.0) & (curr_x <= 0.0)
        gate_passed = crossed_plane & inside_gate

        old_idx = self.env._idx_wp.clone()
        passed_ids = torch.where(gate_passed)[0]

        if len(passed_ids) > 0:
            self.env._idx_wp[passed_ids] = (self.env._idx_wp[passed_ids] + 1) % num_gates
            self.env._n_gates_passed[passed_ids] += 1

        lap_complete = gate_passed & (old_idx == (num_gates - 1))

        # Desired position always tracks current target gate.
        self.env._desired_pos_w[:] = self.env._waypoints[self.env._idx_wp, :3]

        # Dense signed progress through the gate plane.
        progress = torch.clamp(prev_x - curr_x, min=-0.25, max=0.25)

        # Keep center-of-gate traversal attractive but secondary.
        center_reward = 1.0 - torch.tanh(yz_error / 0.5)

        # Body-frame gate observations for alignment/speed shaping.
        drone_pos_w = self.env._robot.data.root_link_pos_w
        drone_quat_w = self.env._robot.data.root_quat_w
        drone_lin_vel_b = self.env._robot.data.root_com_lin_vel_b
        drone_ang_vel_b = self.env._robot.data.root_ang_vel_b

        curr_gate_pos_w = self.env._waypoints[old_idx, :3]
        gate_pos_b, _ = subtract_frame_transforms(drone_pos_w, drone_quat_w, curr_gate_pos_w)
        gate_dist_b = torch.linalg.norm(gate_pos_b, dim=1, keepdim=True).clamp_min(1e-6)
        gate_dir_b = gate_pos_b / gate_dist_b

        # If gate is ahead in body x, this is positive.
        align_reward = gate_dir_b[:, 0].clamp(-1.0, 1.0)

        # Reward forward speed in the direction of the gate.
        speed_toward_gate = torch.sum(drone_lin_vel_b * gate_dir_b, dim=1)
        speed_reward = torch.clamp(speed_toward_gate / 5.0, min=-1.0, max=1.0)

        # Contact-based crash accumulation.
        contact_forces = self.env._contact_sensor.data.net_forces_w
        contact_mag = torch.linalg.norm(contact_forces, dim=-1)
        if contact_mag.dim() > 1:
            contact_mag = torch.amax(contact_mag, dim=1)
        crashed_now = (contact_mag > 1e-3) & (self.env.episode_length_buf > 25)

        self.env._crashed = torch.where(
            crashed_now,
            self.env._crashed + 1,
            torch.clamp(self.env._crashed - 1, min=0),
        )

        crash_reward = crashed_now.float()
        gate_reward = gate_passed.float()
        lap_reward = lap_complete.float()
        time_penalty = torch.ones(self.num_envs, device=self.device)
        action_mag = torch.linalg.norm(self.env._actions, dim=1)
        ang_vel_mag = torch.linalg.norm(drone_ang_vel_b, dim=1)

        rewards = {
            "gate_pass": gate_reward * self._reward_scale("gate_pass_reward_scale"),
            "lap": lap_reward * self._reward_scale("lap_reward_scale"),
            "progress": progress * self._reward_scale("progress_reward_scale"),
            "center": center_reward * self._reward_scale("center_reward_scale"),
            "align": align_reward * self._reward_scale("align_reward_scale"),
            "speed": speed_reward * self._reward_scale("speed_reward_scale"),
            "action_mag": action_mag * self._reward_scale("action_mag_reward_scale"),
            "ang_vel": ang_vel_mag * self._reward_scale("ang_vel_reward_scale"),
            "crash": crash_reward * self._reward_scale("crash_reward_scale"),
            "time_penalty": time_penalty * self._reward_scale("time_penalty_reward_scale"),
        }

        reward = torch.zeros(self.num_envs, device=self.device)
        for value in rewards.values():
            reward = reward + value

        reward = reward + self.env.reset_terminated.float() * self._death_cost()


        if self.cfg.is_train:
            for key, value in rewards.items():
                self._episode_sums[key] += value

        # Update previous signed gate-plane distance for next step.
        self.env._prev_x_drone_wrt_gate = curr_x.clone()

        return reward
    
    # ---------------------------------------------------------------------
    # Observations
    # ---------------------------------------------------------------------
    def get_observations(self) -> Dict[str, torch.Tensor]:
        """Get observations. Read reset_idx() and quadcopter_env.py to see which drone info is extracted from the sim.
        The following code is an example. You should delete it or heavily modify it once you begin the racing task."""

        # TODO ----- START ----- Define tensors for your observation space. Be careful with frame transformations
        drone_pos_w = self.env._robot.data.root_link_pos_w
        drone_quat_w = self.env._robot.data.root_quat_w
        drone_lin_vel_b = self.env._robot.data.root_com_lin_vel_b
        drone_ang_vel_b = self.env._robot.data.root_ang_vel_b

        curr_idx = self.env._idx_wp
        next_idx = (curr_idx + 1) % self.env._waypoints.shape[0]

        curr_gate_pos_w = self.env._waypoints[curr_idx, :3]
        next_gate_pos_w = self.env._waypoints[next_idx, :3]

        curr_gate_pos_b, _ = subtract_frame_transforms(drone_pos_w, drone_quat_w, curr_gate_pos_w)
        next_gate_pos_b, _ = subtract_frame_transforms(drone_pos_w, drone_quat_w, next_gate_pos_w)

        curr_gate_tip_w = curr_gate_pos_w + self.env._normal_vectors[curr_idx]
        next_gate_tip_w = next_gate_pos_w + self.env._normal_vectors[next_idx]

        curr_gate_tip_b, _ = subtract_frame_transforms(drone_pos_w, drone_quat_w, curr_gate_tip_w)
        next_gate_tip_b, _ = subtract_frame_transforms(drone_pos_w, drone_quat_w, next_gate_tip_w)

        curr_gate_normal_b = curr_gate_tip_b - curr_gate_pos_b
        next_gate_normal_b = next_gate_tip_b - next_gate_pos_b

        obs = torch.cat(
            [
                drone_lin_vel_b / 5.0,
                drone_ang_vel_b / 10.0,
                curr_gate_pos_b / 5.0,
                next_gate_pos_b / 5.0,
                curr_gate_normal_b,
                next_gate_normal_b,
                self.env._pose_drone_wrt_gate / 5.0,
                self.env._previous_actions,
            ],
            dim=-1,
        )

        return {"policy": obs}
    
    # ---------------------------------------------------------------------
    # Reset / curriculum / domain randomization
    # ---------------------------------------------------------------------


    def reset_idx(self, env_ids: Optional[torch.Tensor]):
        """Reset specific environments to initial states."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.env._robot._ALL_INDICES
        
        if self.cfg.is_train:
            self._update_curriculum_from_performance(env_ids)

        # Logging for training mode
        if self.cfg.is_train and hasattr(self, '_episode_sums'):
            extras = dict()
            for key in self._episode_sums.keys():
                episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.env.max_episode_length_s
                self._episode_sums[key][env_ids] = 0.0

            extras["Episode_Track/gates_passed_mean"] = torch.mean(self.env._n_gates_passed[env_ids].float()).item()
            extras["Episode_Track/passed_one_gate_rate"] = torch.mean((self.env._n_gates_passed[env_ids] > 0).float()).item()
            extras["Episode_Track/crash_count_mean"] = torch.mean(self.env._crashed[env_ids].float()).item()
            self.env.extras["log"] = dict()
            self.env.extras["log"].update(extras)
            extras = dict()
            extras["Episode_Termination/died"] = torch.count_nonzero(self.env.reset_terminated[env_ids]).item()
            extras["Episode_Termination/time_out"] = torch.count_nonzero(self.env.reset_time_outs[env_ids]).item()
            extras["Curriculum/alpha"] = self._curriculum_alpha_value
            extras["Curriculum/score_ema"] = self._curriculum_score_ema

            self.env.extras["log"].update(extras)


        # Call robot reset first
        self.env._robot.reset(env_ids)

        # Initialize model paths if needed
        if not self.env._models_paths_initialized:
            num_models_per_env = self.env._waypoints.size(0)
            model_prim_names_in_env = [f"{self.env.target_models_prim_base_name}_{i}" for i in range(num_models_per_env)]

            self.env._all_target_models_paths = []
            for env_path in self.env.scene.env_prim_paths:
                paths_for_this_env = [f"{env_path}/{name}" for name in model_prim_names_in_env]
                self.env._all_target_models_paths.append(paths_for_this_env)

            self.env._models_paths_initialized = True

        n_reset = len(env_ids)
        if n_reset == self.num_envs and self.num_envs > 1:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        # Reset action buffers
        self.env._actions[env_ids] = 0.0
        self.env._previous_actions[env_ids] = 0.0
        self.env._previous_yaw[env_ids] = 0.0
        self.env._motor_speeds[env_ids] = 0.0
        self.env._previous_omega_meas[env_ids] = 0.0
        self.env._previous_omega_err[env_ids] = 0.0
        self.env._omega_err_integral[env_ids] = 0.0

        # Reset joints state
        joint_pos = self.env._robot.data.default_joint_pos[env_ids]
        joint_vel = self.env._robot.data.default_joint_vel[env_ids]
        self.env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        default_root_state = self.env._robot.data.default_root_state[env_ids].clone()

        # TODO ----- START ----- Define the initial state during training after resetting an environment.
        # This example code initializes the drone 2m behind the first gate. You should delete it or heavily
        # modify it once you begin the racing task.

        # Apply training-time dynamics randomization.
        self._apply_domain_randomization(env_ids)

        if self.cfg.is_train:
            alpha = self._curriculum_alpha()
            num_gates = self.env._waypoints.shape[0]

            # Curriculum over which part of the track is sampled.
            max_gate_index = max(1, int(round(1 + alpha * (num_gates - 1))))
            waypoint_indices = torch.randint(
                low=0, high=max_gate_index, size=(n_reset,), device=self.device, dtype=self.env._idx_wp.dtype
            )

            # Easy early resets: close to gate, low lateral/yaw/speed noise.
            # Hard later resets: farther back, larger offsets, nonzero initial velocity.
            dist_back = torch.empty(n_reset, device=self.device).uniform_(
                0.75, 1.25 + 2.0 * alpha
            )
            y_local = torch.empty(n_reset, device=self.device).uniform_(
                -(0.10 + 0.80 * alpha), (0.10 + 0.80 * alpha)
            )
            z_local = torch.empty(n_reset, device=self.device).uniform_(
                -(0.05 + 0.40 * alpha), (0.05 + 0.40 * alpha)
            )
            yaw_noise = torch.empty(n_reset, device=self.device).uniform_(
                -(0.05 + 0.45 * alpha), (0.05 + 0.45 * alpha)
            )
            vz_noise = torch.empty(n_reset, device=self.device).uniform_(
                -(0.05 + 0.30 * alpha), (0.05 + 0.30 * alpha)
            )
            init_speed = torch.empty(n_reset, device=self.device).uniform_(
                0.0, 0.5 + 2.0 * alpha
            )

            x0_wp = self.env._waypoints[waypoint_indices, 0]
            y0_wp = self.env._waypoints[waypoint_indices, 1]
            z_wp = self.env._waypoints[waypoint_indices, 2]
            theta = self.env._waypoints[waypoint_indices, -1]

            # Positive gate-frame x means before the gate, so local x is negative.
            x_local = -dist_back

            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
            x_rot = cos_theta * x_local - sin_theta * y_local
            y_rot = sin_theta * x_local + cos_theta * y_local

            initial_x = x0_wp - x_rot
            initial_y = y0_wp - y_rot
            initial_z = z_wp + z_local

            default_root_state[:, 0] = initial_x
            default_root_state[:, 1] = initial_y
            default_root_state[:, 2] = initial_z

            initial_yaw = torch.atan2(y0_wp - initial_y, x0_wp - initial_x) + yaw_noise
            zeros = torch.zeros(n_reset, device=self.device)
            quat = quat_from_euler_xyz(zeros, zeros, initial_yaw)
            default_root_state[:, 3:7] = quat

            # Small initial forward velocity toward the target gate.
            default_root_state[:, 7] = init_speed * torch.cos(initial_yaw)
            default_root_state[:, 8] = init_speed * torch.sin(initial_yaw)
            default_root_state[:, 9] = vz_noise
            default_root_state[:, 10:13] = 0.0

        else:
            x_local = torch.empty(1, device=self.device).uniform_(-3.0, -0.5)
            y_local = torch.empty(1, device=self.device).uniform_(-1.0, 1.0)

            x0_wp = self.env._waypoints[self.env._initial_wp, 0]
            y0_wp = self.env._waypoints[self.env._initial_wp, 1]
            z0_wp = self.env._waypoints[self.env._initial_wp, 2]
            theta = self.env._waypoints[self.env._initial_wp, -1]

            cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
            x_rot = cos_theta * x_local - sin_theta * y_local
            y_rot = sin_theta * x_local + cos_theta * y_local

            x0 = x0_wp - x_rot
            y0 = y0_wp - y_rot
            z0 = z0_wp

            yaw0 = torch.atan2(y0_wp - y0, x0_wp - x0)

            default_root_state = self.env._robot.data.default_root_state[0].unsqueeze(0).clone()
            default_root_state[:, 0] = x0
            default_root_state[:, 1] = y0
            default_root_state[:, 2] = z0

            quat = quat_from_euler_xyz(
                torch.zeros(1, device=self.device),
                torch.zeros(1, device=self.device),
                yaw0,
            )
            default_root_state[:, 3:7] = quat
            default_root_state[:, 7:13] = 0.0
            waypoint_indices = torch.full(
                (1,), int(self.env._initial_wp), device=self.device, dtype=self.env._idx_wp.dtype
            )

        self.env._idx_wp[env_ids] = waypoint_indices
        self.env._desired_pos_w[env_ids] = self.env._waypoints[waypoint_indices, :3].clone()
        self.env._n_gates_passed[env_ids] = 0
        self.env._yaw_n_laps[env_ids] = 0
        self.env._crashed[env_ids] = 0

        self.env._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self.env._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)

        self.env._pose_drone_wrt_gate[env_ids], _ = subtract_frame_transforms(
            self.env._waypoints[self.env._idx_wp[env_ids], :3],
            self.env._waypoints_quat[self.env._idx_wp[env_ids], :],
            default_root_state[:, :3],
        )

        self.env._prev_x_drone_wrt_gate[env_ids] = self.env._pose_drone_wrt_gate[env_ids, 0].clone()
        self.env._last_distance_to_goal[env_ids] = torch.linalg.norm(
            self.env._desired_pos_w[env_ids] - default_root_state[:, :3], dim=1
        )