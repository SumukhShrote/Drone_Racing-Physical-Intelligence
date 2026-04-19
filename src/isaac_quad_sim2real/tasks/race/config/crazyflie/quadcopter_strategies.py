# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Modular strategy classes for quadcopter environment rewards, observations, and resets."""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING, Dict, Optional

from isaaclab.utils.math import subtract_frame_transforms, quat_from_euler_xyz

if TYPE_CHECKING:
    from .quadcopter_env import QuadcopterEnv

class DefaultQuadcopterStrategy:
    """Strategy implementation for quadcopter racing environment."""

    STRATEGY_DEFAULTS = {
        "center_reward_width": 1.5,
        "gate_pass_center_width": 0.10,
        "gate_pass_center_power": 2.0,
        "gate_pass_min_fraction": 0.05,
        "progress_clip": 0.28,
        "distance_progress_clip": 0.35,
        "speed_target": 2.5,
        "speed_tolerance": 0.55,
        "max_safe_speed": 3.50,             # <--- POLICY B: Slightly reduced to account for blind flying
        "stall_distance_threshold": 1.20,
        "stall_speed_threshold": 0.30,
        "trajectory_history_length": 4.0,
        "dr_twr_min_factor": 0.85,
        "dr_twr_max_factor": 1.15,
        "dr_aero_min_factor": 0.80,         # <--- POLICY B: Normal aerodynamic bounds
        "dr_aero_max_factor": 1.20,         # <--- POLICY B: Normal aerodynamic bounds
        "dr_kp_ki_min_factor": 0.85,
        "dr_kp_ki_max_factor": 1.15,
        "dr_kd_min_factor": 0.70,
        "dr_kd_max_factor": 1.30,
    }

    def __init__(self, env: QuadcopterEnv):
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.cfg = env.cfg

        if not hasattr(self.env, "_lap_steps"):
            self.env._lap_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        if not hasattr(self.env, "_last_lap_time"):
            self.env._last_lap_time = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        if not hasattr(self.env, "_ever_crashed"):
            self.env._ever_crashed = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        if not hasattr(self.env, "_last_lap_steps"):
            self.env._last_lap_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)

        self.reward_defaults = {
            "gate_pass_reward_scale": 120.0,
            "lap_reward_scale": 300.0,
            "progress_reward_scale": 10.0,
            "center_reward_scale": 2.0,
            "align_reward_scale": 0.8,
            "speed_reward_scale": 0.8,
            "speed_excess_reward_scale": -0.20,
            "lateral_motion_reward_scale": -0.20,
            "stall_reward_scale": -2.0,
            "action_mag_reward_scale": -0.02,
            "ang_vel_reward_scale": -0.002,
            "crash_reward_scale": -80.0,
            "time_penalty_reward_scale": -0.02,
            "miss_reward_scale": 10.0,
            "wrong_direction_penalty_reward_scale": 80.0,
            "action_smoothness_reward_scale": -0.005, # <--- POLICY B: Removed speed limit so it can snap-correct lag
        }
        self.strategy_defaults = dict(self.STRATEGY_DEFAULTS)
        self.death_cost_default = -25.0

        self.reward_terms = [
            "gate_pass",
            "lap",
            "progress",
            "center",
            "align",
            "speed",
            "speed_excess",
            "lateral_motion",
            "stall",
            "action_mag",
            "ang_vel",
            "crash",
            "time_penalty",
            "miss",
            "wrong_direction",
            "action_smoothness",
        ]

        if self.cfg.is_train:
            self._episode_sums = {
                key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for key in self.reward_terms
            }

        self.env._K_aero[:, :2] = self.env._k_aero_xy_value
        self.env._K_aero[:, 2] = self.env._k_aero_z_value

        self.env._kp_omega[:, :2] = self.env._kp_omega_rp_value
        self.env._ki_omega[:, :2] = self.env._ki_omega_rp_value
        self.env._kd_omega[:, :2] = self.env._kd_omega_rp_value

        self.env._kp_omega[:, 2] = self.env._kp_omega_y_value
        self.env._ki_omega[:, 2] = self.env._ki_omega_y_value
        self.env._kd_omega[:, 2] = self.env._kd_omega_y_value

        self.env._twr_value = 3.30
        self.env._tau_m_value = 0.035 # <--- POLICY B: Extreme latency simulation (35ms)

        self.env._thrust_to_weight[:] = self.env._twr_value
        self.env._tau_m[:] = self.env._tau_m_value

        self._last_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

        self._trajectory_history_length = max(int(round(self._strategy_value("trajectory_history_length"))), 1)
        self._trajectory_feature_dim = 13
        self._trajectory_history = torch.zeros(
            self.num_envs,
            self._trajectory_history_length,
            self._trajectory_feature_dim,
            device=self.device,
        )
        self._trajectory_history_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )

        force_final = getattr(self.cfg, "final_phase", False)
        self._curriculum_alpha_value = 1.0 if force_final else 0.0
        self._curriculum_score_ema = 0.0
        self._curriculum_updates = 0

    def _reward_scale(self, key: str) -> float:
        if hasattr(self.env, "rew") and self.env.rew is not None and key in self.env.rew:
            return float(self.env.rew[key])
        return float(self.reward_defaults[key])

    def _death_cost(self) -> float:
        if hasattr(self.env, "rew") and self.env.rew is not None and "death_cost" in self.env.rew:
            return float(self.env.rew["death_cost"])
        return float(self.death_cost_default)

    def _strategy_value(self, key: str) -> float:
        overrides = getattr(self.env, "strategy_overrides", None)
        if overrides is not None and key in overrides:
            return float(overrides[key])
        return float(self.strategy_defaults[key])

    def _curriculum_alpha(self) -> float:
        if not self.cfg.is_train or getattr(self.cfg, "final_phase", False):
            return 1.0
        return float(self._curriculum_alpha_value)

    def _sample_factor(self, nominal: float, min_factor: float, max_factor: float, alpha: float, n: int) -> torch.Tensor:
        low = nominal * (1.0 + alpha * (min_factor - 1.0))
        high = nominal * (1.0 + alpha * (max_factor - 1.0))
        lo = min(low, high)
        hi = max(low, high)
        return torch.empty(n, device=self.device).uniform_(lo, hi)

    def _trajectory_history_feature(
        self,
        drone_lin_vel_b: torch.Tensor,
        drone_ang_vel_b: torch.Tensor,
        target_pose_gate: torch.Tensor,
        env_ids: Optional[torch.Tensor] = None, # <--- Added env_ids to fix tensor mismatch
    ) -> torch.Tensor:
        if env_ids is not None:
            prev_actions = self.env._previous_actions[env_ids]
        else:
            prev_actions = self.env._previous_actions
            
        return torch.cat(
            [
                target_pose_gate / 5.0,
                drone_lin_vel_b / 5.0,
                drone_ang_vel_b / 10.0,
                prev_actions, 
            ],
            dim=-1,
        )

    def _reset_trajectory_history(self, env_ids: torch.Tensor):
        self._trajectory_history[env_ids] = 0.0
        self._trajectory_history_step[env_ids] = -1

    def _update_trajectory_history(self, trajectory_feature: torch.Tensor):
        current_steps = self.env.episode_length_buf.to(dtype=torch.long)
        needs_update = self._trajectory_history_step != current_steps
        if not torch.any(needs_update):
            return

        updated_history = torch.roll(self._trajectory_history[needs_update], shifts=-1, dims=1)
        updated_history[:, -1, :] = trajectory_feature[needs_update]
        self._trajectory_history[needs_update] = updated_history
        self._trajectory_history_step[needs_update] = current_steps[needs_update]

    def _update_curriculum_from_performance(self, env_ids: torch.Tensor):
        if not self.cfg.is_train or len(env_ids) == 0:
            return

        if getattr(self.cfg, "final_phase", False):
            self._curriculum_alpha_value = 1.0
            return

        num_gates = float(self.env._waypoints.shape[0])
        gates_passed = self.env._n_gates_passed[env_ids].float()
        progress_frac = torch.clamp(gates_passed / max(num_gates, 1.0), 0.0, 1.0)
        passed_one_gate = (gates_passed > 0).float()
        batch_score = 0.7 * progress_frac.mean().item() + 0.3 * passed_one_gate.mean().item()

        if self._curriculum_updates == 0:
            self._curriculum_score_ema = batch_score
        else:
            self._curriculum_score_ema = 0.95 * self._curriculum_score_ema + 0.05 * batch_score

        self._curriculum_updates += 1
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
            if hasattr(self.env, "_control_latency_steps"):
                self.env._control_latency_steps[env_ids] = int(self.env.cfg.control_latency_steps)
            return

        alpha = self._curriculum_alpha()
        effective_alpha = max(0.4, alpha)

        self.env._thrust_to_weight[env_ids] = self._sample_factor(
            self.env._twr_value,
            self._strategy_value("dr_twr_min_factor"),
            self._strategy_value("dr_twr_max_factor"),
            effective_alpha,
            n,
        )

        kxy = self._sample_factor(
            self.env._k_aero_xy_value,
            self._strategy_value("dr_aero_min_factor"),
            self._strategy_value("dr_aero_max_factor"),
            effective_alpha,
            n,
        )
        kz = self._sample_factor(
            self.env._k_aero_z_value,
            self._strategy_value("dr_aero_min_factor"),
            self._strategy_value("dr_aero_max_factor"),
            effective_alpha,
            n,
        )
        self.env._K_aero[env_ids, :2] = kxy.unsqueeze(1)
        self.env._K_aero[env_ids, 2] = kz

        pid_min = self._strategy_value("dr_kp_ki_min_factor")
        pid_max = self._strategy_value("dr_kp_ki_max_factor")
        kd_min = self._strategy_value("dr_kd_min_factor")
        kd_max = self._strategy_value("dr_kd_max_factor")

        kp_rp = self._sample_factor(self.env._kp_omega_rp_value, pid_min, pid_max, alpha, n)
        ki_rp = self._sample_factor(self.env._ki_omega_rp_value, pid_min, pid_max, alpha, n)
        kd_rp = self._sample_factor(self.env._kd_omega_rp_value, kd_min, kd_max, alpha, n)

        kp_y = self._sample_factor(self.env._kp_omega_y_value, pid_min, pid_max, alpha, n)
        ki_y = self._sample_factor(self.env._ki_omega_y_value, pid_min, pid_max, alpha, n)
        kd_y = self._sample_factor(self.env._kd_omega_y_value, kd_min, kd_max, alpha, n)

        self.env._kp_omega[env_ids, :2] = kp_rp.unsqueeze(1)
        self.env._ki_omega[env_ids, :2] = ki_rp.unsqueeze(1)
        self.env._kd_omega[env_ids, :2] = kd_rp.unsqueeze(1)

        self.env._kp_omega[env_ids, 2] = kp_y
        self.env._ki_omega[env_ids, 2] = ki_y
        self.env._kd_omega[env_ids, 2] = kd_y

        self.env._tau_m[env_ids] = self.env._tau_m_value

        if hasattr(self.env, "_control_latency_steps"):
            max_delay = int(self.env.cfg.control_latency_steps) + 1
            min_delay = max(0, int(self.env.cfg.control_latency_steps) - 1)
            if alpha > 0.5:
                self.env._control_latency_steps[env_ids] = torch.randint(
                    min_delay, max_delay + 1, (n,), device=self.device
                )
            else:
                self.env._control_latency_steps[env_ids] = int(self.env.cfg.control_latency_steps)

    def get_rewards(self) -> torch.Tensor:
        pose_gate = self.env._pose_drone_wrt_gate
        curr_x = pose_gate[:, 0]
        curr_y = pose_gate[:, 1]
        curr_z = pose_gate[:, 2]
        prev_x = self.env._prev_x_drone_wrt_gate.clone()

        num_gates = self.env._waypoints.shape[0]
        gate_half = 0.5 * float(self.cfg.gate_model.gate_side)
        gate_margin = 0.40 * gate_half

        inside_gate = (curr_y.abs() <= gate_margin) & (curr_z.abs() <= gate_margin)
        crossed_plane = (prev_x > 0.0) & (curr_x <= 0.0)
        gate_passed = crossed_plane & inside_gate
        missed_gate = crossed_plane & (~inside_gate)
        wrong_direction_crossed = (prev_x <= 0.0) & (curr_x > 0.0)

        drone_pos_w = self.env._robot.data.root_link_pos_w
        drone_quat_w = self.env._robot.data.root_quat_w
        drone_lin_vel_b = self.env._robot.data.root_com_lin_vel_b
        drone_lin_vel_w = self.env._robot.data.root_com_lin_vel_w
        drone_ang_vel_b = self.env._robot.data.root_ang_vel_b

        old_idx = self.env._idx_wp.clone()
        curr_target_pos_w = self.env._waypoints[old_idx, :3].clone()
        gate_pos_b, _ = subtract_frame_transforms(drone_pos_w, drone_quat_w, curr_target_pos_w)
        gate_dist_b = torch.linalg.norm(gate_pos_b, dim=1, keepdim=True).clamp_min(1e-6)
        gate_dir_b = gate_pos_b / gate_dist_b

        distance_to_goal = torch.linalg.norm(curr_target_pos_w - drone_pos_w, dim=1)
        prev_distance_to_goal = self.env._last_distance_to_goal.clone()
        speed_toward_gate = torch.sum(drone_lin_vel_b * gate_dir_b, dim=1)
        speed_target = self._strategy_value("speed_target")
        speed_tolerance = max(self._strategy_value("speed_tolerance"), 1e-3)

        plane_progress = torch.where(
            curr_x > 0.0,
            torch.clamp(prev_x - curr_x, min=0.0, max=self._strategy_value("progress_clip")),
            torch.zeros_like(curr_x),
        )
        distance_progress = torch.clamp(
            prev_distance_to_goal - distance_to_goal,
            min=0.0,
            max=self._strategy_value("distance_progress_clip"),
        )

        yz_error = torch.sqrt(curr_y.pow(2) + curr_z.pow(2))
        center_potential = torch.where(
            curr_x > 0.0,
            1.0 - torch.tanh(yz_error / self._strategy_value("center_reward_width")),
            torch.zeros_like(yz_error),
        )
        align_potential = torch.clamp(gate_dir_b[:, 0], min=0.0, max=1.0)
        forward_motion_gate = torch.clamp(speed_toward_gate / max(speed_target, 1e-3), min=0.0, max=1.0)
        progress_gate = torch.clamp(
            distance_progress / max(self._strategy_value("distance_progress_clip"), 1e-6),
            min=0.0,
            max=1.0,
        )
        shaping_gate = forward_motion_gate * progress_gate

        passed_ids = torch.where(gate_passed)[0]
        if len(passed_ids) > 0:
            self.env._idx_wp[passed_ids] = (self.env._idx_wp[passed_ids] + 1) % num_gates
            self.env._n_gates_passed[passed_ids] += 1

        lap_complete = gate_passed & (old_idx == (num_gates - 1))

        target_pos_w = self.env._waypoints[self.env._idx_wp, :3].clone()
        self.env._desired_pos_w[:] = target_pos_w

        progress = 0.5 * plane_progress + 0.5 * distance_progress
        pass_center_quality = torch.exp(
            -0.5 * (yz_error / max(self._strategy_value("gate_pass_center_width"), 1e-3)).pow(2)
        )
        pass_center_weight = pass_center_quality.pow(self._strategy_value("gate_pass_center_power"))
        gate_reward = gate_passed.float() * (
            self._strategy_value("gate_pass_min_fraction")
            + (1.0 - self._strategy_value("gate_pass_min_fraction")) * pass_center_weight
        )

        center_reward = center_potential * shaping_gate
        align_reward = align_potential * shaping_gate
        tracking_quality = align_potential * shaping_gate
        speed_reward = torch.exp(
            -0.5 * ((speed_toward_gate - speed_target) / speed_tolerance).pow(2)
        ) * tracking_quality

        total_speed = torch.linalg.norm(drone_lin_vel_w, dim=1)
        speed_excess = torch.clamp(total_speed - self._strategy_value("max_safe_speed"), min=0.0)
        lateral_motion = torch.linalg.norm(drone_lin_vel_b[:, 1:], dim=1)
        stall_penalty = (
            (distance_to_goal < self._strategy_value("stall_distance_threshold"))
            & (curr_x > 0.0)
            & (~gate_passed)
        ).float()
        stall_penalty = stall_penalty * torch.clamp(
            self._strategy_value("stall_speed_threshold") - speed_toward_gate,
            min=0.0,
        ) / max(self._strategy_value("stall_speed_threshold"), 1e-3)
        stall_penalty = stall_penalty * (0.25 + 0.75 * center_potential)

        self.env._lap_steps += 1
        completed_ids = torch.where(lap_complete)[0]
        if len(completed_ids) > 0:
            dt_policy = self.env.cfg.sim.dt * self.env.cfg.decimation
            self.env._last_lap_time[completed_ids] = self.env._lap_steps[completed_ids].float() * dt_policy
            self.env._last_lap_steps[completed_ids] = self.env._lap_steps[completed_ids].clone()
            self.env._lap_steps[completed_ids] = 0

        contact_forces = self.env._contact_sensor.data.net_forces_w
        contact_mag = torch.linalg.norm(contact_forces, dim=-1)
        if contact_mag.dim() > 1:
            contact_mag = torch.amax(contact_mag, dim=1)
        crashed_now = (contact_mag > 1e-3) & (self.env.episode_length_buf > 25)

        self.env.reset_terminated = self.env.reset_terminated | crashed_now
        self.env._crashed = torch.where(
            crashed_now,
            self.env._crashed + 1,
            torch.clamp(self.env._crashed - 1, min=0),
        )
        self.env._ever_crashed = torch.where(
            crashed_now,
            torch.ones_like(self.env._ever_crashed),
            self.env._ever_crashed,
        )

        miss_penalty = missed_gate.float()
        wrong_direction_penalty = wrong_direction_crossed.float()
        time_penalty = torch.ones(self.num_envs, device=self.device)
        action_mag = torch.linalg.norm(self.env._actions, dim=1)
        ang_vel_mag = torch.linalg.norm(drone_ang_vel_b, dim=1)
        action_smoothness = torch.linalg.norm(self.env._actions - self._last_actions, dim=1)

        rewards = {
            "gate_pass": gate_reward * self._reward_scale("gate_pass_reward_scale"),
            "lap": lap_complete.float() * self._reward_scale("lap_reward_scale"),
            "progress": progress * self._reward_scale("progress_reward_scale"),
            "center": center_reward * self._reward_scale("center_reward_scale"),
            "align": align_reward * self._reward_scale("align_reward_scale"),
            "speed": speed_reward * self._reward_scale("speed_reward_scale"),
            "speed_excess": speed_excess * self._reward_scale("speed_excess_reward_scale"),
            "lateral_motion": lateral_motion * self._reward_scale("lateral_motion_reward_scale"),
            "stall": stall_penalty * self._reward_scale("stall_reward_scale"),
            "action_mag": action_mag * self._reward_scale("action_mag_reward_scale"),
            "ang_vel": ang_vel_mag * self._reward_scale("ang_vel_reward_scale"),
            "crash": crashed_now.float() * self._reward_scale("crash_reward_scale"),
            "time_penalty": time_penalty * self._reward_scale("time_penalty_reward_scale"),
            "miss": -miss_penalty * self._reward_scale("miss_reward_scale"),
            "wrong_direction": -wrong_direction_penalty * self._reward_scale("wrong_direction_penalty_reward_scale"),
            "action_smoothness": action_smoothness * self._reward_scale("action_smoothness_reward_scale"),
        }

        reward = torch.zeros(self.num_envs, device=self.device)
        for value in rewards.values():
            reward = reward + value
        reward = reward + self.env.reset_terminated.float() * self._death_cost()

        if self.cfg.is_train:
            for key, value in rewards.items():
                self._episode_sums[key] += value

        next_prev_x = curr_x.clone()
        if len(passed_ids) > 0:
            next_pose_gate, _ = subtract_frame_transforms(
                self.env._waypoints[self.env._idx_wp[passed_ids], :3],
                self.env._waypoints_quat[self.env._idx_wp[passed_ids], :],
                drone_pos_w[passed_ids],
            )
            next_prev_x[passed_ids] = next_pose_gate[:, 0]

        self.env._prev_x_drone_wrt_gate = next_prev_x
        self.env._last_distance_to_goal = torch.linalg.norm(self.env._desired_pos_w - drone_pos_w, dim=1)
        self._last_actions = self.env._actions.clone()
        return reward

    def get_observations(self) -> Dict[str, torch.Tensor]:
        drone_pos_w = self.env._robot.data.root_link_pos_w
        drone_quat_w = self.env._robot.data.root_quat_w
        drone_lin_vel_b = self.env._robot.data.root_com_lin_vel_b
        drone_ang_vel_b = self.env._robot.data.root_ang_vel_b

        # <--- POLICY B: Double the Vicon Noise (20%)
        drone_lin_vel_obs = drone_lin_vel_b + (torch.randn_like(drone_lin_vel_b) * 0.2)
        drone_ang_vel_obs = drone_ang_vel_b + (torch.randn_like(drone_ang_vel_b) * 0.2)

        curr_idx = self.env._idx_wp
        next_idx = (curr_idx + 1) % self.env._waypoints.shape[0]
        next_next_idx = (curr_idx + 2) % self.env._waypoints.shape[0]

        curr_gate_pos_w = self.env._waypoints[curr_idx, :3].clone()
        curr_gate_quat_w = self.env._waypoints_quat[curr_idx].clone()
        curr_gate_normal_w = self.env._normal_vectors[curr_idx].clone()
        next_gate_pos_w = self.env._waypoints[next_idx, :3].clone()
        next_next_gate_pos_w = self.env._waypoints[next_next_idx, :3].clone()
        next_gate_normal_w = self.env._normal_vectors[next_idx].clone()

        curr_gate_pos_b, _ = subtract_frame_transforms(drone_pos_w, drone_quat_w, curr_gate_pos_w)
        next_gate_pos_b, _ = subtract_frame_transforms(drone_pos_w, drone_quat_w, next_gate_pos_w)
        next_next_gate_pos_b, _ = subtract_frame_transforms(drone_pos_w, drone_quat_w, next_next_gate_pos_w)

        curr_gate_tip_w = curr_gate_pos_w + curr_gate_normal_w
        next_gate_tip_w = next_gate_pos_w + next_gate_normal_w

        curr_gate_tip_b, _ = subtract_frame_transforms(drone_pos_w, drone_quat_w, curr_gate_tip_w)
        next_gate_tip_b, _ = subtract_frame_transforms(drone_pos_w, drone_quat_w, next_gate_tip_w)

        curr_gate_normal_b = curr_gate_tip_b - curr_gate_pos_b
        next_gate_normal_b = next_gate_tip_b - next_gate_pos_b

        target_pose_gate, _ = subtract_frame_transforms(curr_gate_pos_w, curr_gate_quat_w, drone_pos_w)

        num_gates = self.env._waypoints.shape[0]
        gate_onehot = torch.zeros(self.num_envs, num_gates, device=self.device)
        gate_onehot.scatter_(1, curr_idx.unsqueeze(1).long(), 1.0)

        trajectory_feature = self._trajectory_history_feature(
            drone_lin_vel_obs,
            drone_ang_vel_obs,
            target_pose_gate,
        )
        trajectory_history = torch.flatten(self._trajectory_history, start_dim=1)

        obs = torch.cat(
            [
                drone_lin_vel_obs / 5.0,
                drone_ang_vel_obs / 10.0,
                curr_gate_pos_b / 5.0,
                next_gate_pos_b / 5.0,
                next_next_gate_pos_b / 5.0,
                curr_gate_normal_b,
                next_gate_normal_b,
                target_pose_gate / 5.0,
                self.env._previous_actions,
                gate_onehot,
                trajectory_history,
            ],
            dim=-1,
        )

        self._update_trajectory_history(trajectory_feature)
        return {"policy": obs}

    def reset_idx(self, env_ids: Optional[torch.Tensor]):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.env._robot._ALL_INDICES

        if self.cfg.is_train and torch.any(self.env.episode_length_buf[env_ids] > 0):
            self._update_curriculum_from_performance(env_ids)

        if self.cfg.is_train and hasattr(self, "_episode_sums"):
            extras = dict()
            for key in self._episode_sums.keys():
                episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.env.max_episode_length_s
                self._episode_sums[key][env_ids] = 0.0

            extras["Episode_Track/gates_passed_mean"] = torch.mean(self.env._n_gates_passed[env_ids].float()).item()
            extras["Episode_Track/passed_one_gate_rate"] = torch.mean((self.env._n_gates_passed[env_ids] > 0).float()).item()
            extras["Episode_Track/crash_count_mean"] = torch.mean(self.env._crashed[env_ids].float()).item()

            valid_laps_mask = self.env._last_lap_time[env_ids] > 0
            if valid_laps_mask.any():
                valid_lap_times = self.env._last_lap_time[env_ids][valid_laps_mask]
                valid_lap_steps = self.env._last_lap_steps[env_ids][valid_laps_mask]
                extras["Episode_Track/lap_time_seconds"] = torch.mean(valid_lap_times).item()
                extras["Episode_Track/lap_steps_mean"] = torch.mean(valid_lap_steps.float()).item()
                extras["Episode_Track/fastest_lap_time_s"] = torch.min(valid_lap_times).item()

            drone_vel = self.env._robot.data.root_com_lin_vel_b[env_ids]
            speed_mag = torch.linalg.norm(drone_vel, dim=1)
            extras["Episode_Track/flight_speed_mean"] = torch.mean(speed_mag).item()
            extras["Episode_Track/flight_speed_max"] = torch.max(speed_mag).item()

            self.env.extras["log"] = dict()
            self.env.extras["log"].update(extras)

            extras = dict()
            extras["Episode_Termination/died"] = torch.count_nonzero(self.env.reset_terminated[env_ids]).item()
            extras["Episode_Termination/time_out"] = torch.count_nonzero(self.env.reset_time_outs[env_ids]).item()
            extras["Curriculum/alpha"] = self._curriculum_alpha_value
            extras["Curriculum/score_ema"] = self._curriculum_score_ema
            extras["Curriculum/safety_alpha"] = max(0.4, self._curriculum_alpha_value) if self.cfg.is_train else 1.0
            self.env.extras["log"].update(extras)

        self.env._robot.reset(env_ids)

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
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        self.env._actions[env_ids] = 0.0
        self.env._previous_actions[env_ids] = 0.0
        if hasattr(self.env, "_action_queue"):
            self.env._action_queue[:, env_ids, :] = 0.0
        if hasattr(self.env, "_control_latency_steps"):
            self.env._control_latency_steps[env_ids] = int(self.env.cfg.control_latency_steps)
        self.env._previous_yaw[env_ids] = 0.0
        self.env._motor_speeds[env_ids] = 0.0
        self.env._previous_omega_meas[env_ids] = 0.0
        self.env._previous_omega_err[env_ids] = 0.0
        self.env._omega_err_integral[env_ids] = 0.0
        self._last_actions[env_ids] = 0.0

        joint_pos = self.env._robot.data.default_joint_pos[env_ids]
        joint_vel = self.env._robot.data.default_joint_vel[env_ids]
        self.env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        default_root_state = self.env._robot.data.default_root_state[env_ids].clone()

        self._apply_domain_randomization(env_ids)

        if self.cfg.is_train:
            alpha = self._curriculum_alpha()
            num_gates = self.env._waypoints.shape[0]
            waypoint_indices = torch.randint(
                low=0, high=num_gates, size=(n_reset,), device=self.device, dtype=self.env._idx_wp.dtype
            )

            dist_back = torch.empty(n_reset, device=self.device).uniform_(0.75, 1.25 + 2.0 * alpha)
            y_local = torch.empty(n_reset, device=self.device).uniform_(
                -(0.10 + 0.80 * alpha), (0.10 + 0.80 * alpha)
            )
            yaw_noise = torch.empty(n_reset, device=self.device).uniform_(
                -(0.05 + 0.45 * alpha), (0.05 + 0.45 * alpha)
            )
            vz_noise = torch.empty(n_reset, device=self.device).uniform_(
                -(0.05 + 0.30 * alpha), (0.05 + 0.30 * alpha)
            )
            init_speed = torch.empty(n_reset, device=self.device).uniform_(0.0, 0.5 + 2.0 * alpha)

            x0_wp = self.env._waypoints[waypoint_indices, 0]
            y0_wp = self.env._waypoints[waypoint_indices, 1]
            theta = self.env._waypoints[waypoint_indices, -1]

            x_local = -dist_back

            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
            x_rot = cos_theta * x_local - sin_theta * y_local
            y_rot = sin_theta * x_local + cos_theta * y_local

            initial_x = x0_wp - x_rot
            initial_y = y0_wp - y_rot
            initial_z = torch.empty(n_reset, device=self.device).uniform_(0.01, max(0.02, 1.0 - alpha))

            default_root_state[:, 0] = initial_x
            default_root_state[:, 1] = initial_y
            default_root_state[:, 2] = initial_z

            initial_yaw = torch.atan2(y0_wp - initial_y, x0_wp - initial_x) + yaw_noise
            zeros = torch.zeros(n_reset, device=self.device)
            quat = quat_from_euler_xyz(zeros, zeros, initial_yaw)
            default_root_state[:, 3:7] = quat

            default_root_state[:, 7] = init_speed * torch.cos(initial_yaw)
            default_root_state[:, 8] = init_speed * torch.sin(initial_yaw)
            default_root_state[:, 9] = vz_noise
            default_root_state[:, 10:13] = 0.0

        else:
            x0_wp = self.env._waypoints[self.env._initial_wp, 0]
            y0_wp = self.env._waypoints[self.env._initial_wp, 1]
            theta = self.env._waypoints[self.env._initial_wp, -1]

            x_local = torch.tensor([-1.5], device=self.device)
            y_local = torch.tensor([0.0], device=self.device)

            cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
            x_rot = cos_theta * x_local - sin_theta * y_local
            y_rot = sin_theta * x_local + cos_theta * y_local

            x0 = x0_wp - x_rot
            y0 = y0_wp - y_rot
            z0 = 0.01

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
        self.env._ever_crashed[env_ids] = 0
        self.env._lap_steps[env_ids] = 0
        self.env._last_lap_time[env_ids] = 0.0
        self.env._last_lap_steps[env_ids] = 0

        self.env._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self.env._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)

        self.env._pose_drone_wrt_gate[env_ids], _ = subtract_frame_transforms(
            self.env._waypoints[self.env._idx_wp[env_ids], :3],
            self.env._waypoints_quat[self.env._idx_wp[env_ids], :],
            default_root_state[:, :3],
        )

        self.env._prev_x_drone_wrt_gate[env_ids] = self.env._pose_drone_wrt_gate[env_ids, 0].clone()
        
        self._trajectory_history_step[env_ids] = -1
        
        initial_target_pose_gate, _ = subtract_frame_transforms(
            self.env._waypoints[self.env._idx_wp[env_ids], :3],
            self.env._waypoints_quat[self.env._idx_wp[env_ids], :],
            default_root_state[:, :3],
        )

        initial_feature = self._trajectory_history_feature(
            torch.zeros_like(default_root_state[:, 7:10]), 
            torch.zeros_like(default_root_state[:, 10:13]), 
            initial_target_pose_gate,
            env_ids=env_ids, 
        )

        self._trajectory_history[env_ids] = initial_feature.unsqueeze(1).repeat(
            1, self._trajectory_history_length, 1
        )

        self.env._last_distance_to_goal[env_ids] = torch.linalg.norm(
            self.env._desired_pos_w[env_ids] - default_root_state[:, :3], dim=1
        )