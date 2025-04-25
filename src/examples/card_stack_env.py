import sapien
import torch
import numpy as np
from typing import Dict, Any, Union, List
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.sapien_utils import look_at
from mani_skill.utils.structs import Pose
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.sensors.camera import CameraConfig

CARD_SIZE = [0.05, 0.08, 0.001]  # half-sizes for x, y, z (mm)


@register_env("CardStack-v1", max_episode_steps=100)
class CardStackEnv(BaseEnv):
    """
    Task: Stack 3 cards on top of each other at a specified location.
    """

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]

    def __init__(
        self,
        *args,
        robot_uids="panda",
        num_envs=1,
        reconfiguration_freq=None,
        n_cards=3,
        **kwargs,
    ):

        # Set reconfiguration frequency - for single env, reconfigure every time
        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if num_envs == 1 else 0
        self.n_cards = n_cards

        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq,
            num_envs=num_envs,
            **kwargs,
        )

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(env=self)
        self.table_scene.build()

        self.cards = []
        for i in range(self.n_cards):
            builder = self.scene.create_actor_builder()
            builder.add_box_collision(half_size=CARD_SIZE)
            color = [(i + 1) / self.n_cards, 0.2, 0.8, 1.0]
            builder.add_box_visual(
                half_size=CARD_SIZE,
                material=sapien.render.RenderMaterial(base_color=color),
            )

            # Place cards initially above the table, to be randomized later
            builder.initial_pose = sapien.Pose(
                p=[0, 0, CARD_SIZE[2] + 0.5], q=[1, 0, 0, 0]
            )
            card = builder.build(name=f"card_{i}")
            self.cards.append(card)

        # Create goal marker (visual only)
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(
            half_size=[CARD_SIZE[0], CARD_SIZE[1], 0.001],
            material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.3]),
        )
        builder.initial_pose = sapien.Pose(p=[0, 0, 0.001], q=[1, 0, 0, 0])
        self.goal_marker = builder.build_static(name="goal_marker")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.table_scene.initialize(env_idx)

        with torch.device(self.device):
            b = len(env_idx)
            goal_pos = torch.zeros((b, 3))
            goal_pos[..., 0] = torch.rand((b,)) * 0.3 - 0.15
            goal_pos[..., 1] = torch.rand((b,)) * 0.3 - 0.15
            goal_pos[..., 2] = 0.001
            goal_pose = Pose.create_from_pq(p=goal_pos, q=[1, 0, 0, 0])
            self.goal_marker.set_pose(goal_pose)
            self.goal_position = goal_pos

            for i, card in enumerate(self.cards):
                card_pos = torch.zeros((b, 3))
                card_pos[..., 0] = torch.rand((b,)) * 0.4 - 0.2
                card_pos[..., 1] = torch.rand((b,)) * 0.4 - 0.2
                card_pos[..., 2] = CARD_SIZE[2]

                card_pose = Pose.create_from_pq(p=card_pos, q=[1, 0, 0, 0])
                card.set_pose(card_pose)

                card.set_linear_velocity(torch.zeros((b, 3), device=self.device))
                card.set_angular_velocity(torch.zeros((b, 3), device=self.device))

    def evaluate(self):
        """Determine success/failure of the task"""
        with torch.device(self.device):
            card_positions = torch.stack(
                [card.pose.p for card in self.cards]
            )  # Shape: [n_cards, b, 3]

            # Check for cards at goal location (xy within threshold)
            at_goal = (
                torch.linalg.norm(
                    card_positions[..., :2] - self.goal_position[:, :2], dim=-1
                )
                < (CARD_SIZE[0] + CARD_SIZE[1]) / 2
            )

            # Check heights of cards to see if they're stacked
            card_heights = card_positions[..., 2]
            first_layer = torch.logical_and(
                card_heights[0] > CARD_SIZE[2] * 0.8,
                card_heights[0] < CARD_SIZE[2] * 1.5,
            )
            stacked = [first_layer]
            for i in range(1, self.n_cards):
                expected_height = (i * 2 + 1) * CARD_SIZE[2]
                current_layer = torch.logical_and(
                    card_heights[i] > expected_height * 0.8,
                    card_heights[i] < expected_height * 1.5,
                )
                stacked.append(current_layer)
            all_stacked = torch.all(
                torch.stack([at_goal[i] & stacked[i] for i in range(self.n_cards)]),
                dim=0,
            )

            return {
                "success": all_stacked,
                "at_goal": at_goal,
                "stacked": torch.stack(stacked),
            }

    def _get_obs_extra(self, info: Dict):
        """Additional observations for solving the task"""
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            goal_pos=self.goal_position,
        )

        if self.obs_mode_struct.use_state:
            # Add ground truth information if using state observations
            for i, card in enumerate(self.cards):
                obs[f"card_{i}_pose"] = card.pose.raw_pose
                obs[f"card_{i}_vel"] = card.linear_velocity

        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """Compute a dense reward signal to guide learning"""
        with torch.device(self.device):
            reward = torch.zeros(self.num_envs, device=self.device)

            # Success reward
            success = info["success"]
            stacked = info["stacked"]
            reward = torch.where(success, reward + 10.0, reward)

            # Reward for having cards close to goal
            tcp_pose = self.agent.tcp.pose
            tcp_pos = tcp_pose.p
            card_positions = torch.stack([card.pose.p for card in self.cards])
            goal_distances = torch.linalg.norm(
                card_positions[..., :2] - self.goal_position[:, :2], dim=-1
            )
            tcp_to_cards = torch.linalg.norm(
                card_positions - tcp_pos.unsqueeze(0), dim=-1
            )
            closest_card_to_tcp = torch.min(tcp_to_cards, dim=0)[0]

            for i in range(self.n_cards):
                # Reward for getting cards to goal
                reward += torch.exp(-5.0 * goal_distances[i]) * 0.2
                # Reward for stacking correctly
                if i > 0:
                    reward = torch.where(stacked[i], reward + 0.5, reward)

            # Reward for getting the TCP close to any card (to encourage picking)
            reward += torch.exp(-10.0 * closest_card_to_tcp) * 0.1
            return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        """Normalize the dense reward"""
        max_reward = (
            13.0  # Maximum possible reward (success + all intermediate rewards)
        )
        return self.compute_dense_reward(obs, action, info) / max_reward

    @property
    def _default_sensor_configs(self):
        """Configure camera sensors for the environment"""
        # Top-down camera view
        top_camera = CameraConfig(
            "top_camera",
            pose=look_at(eye=[0, 0, 0.8], target=[0, 0, 0]),
            width=128,
            height=128,
            fov=np.pi / 3,
            near=0.01,
            far=100,
        )

        # Side view camera
        side_camera = CameraConfig(
            "side_camera",
            pose=look_at(eye=[0.5, 0, 0.5], target=[0, 0, 0.2]),
            width=128,
            height=128,
            fov=np.pi / 3,
            near=0.01,
            far=100,
        )
        return [top_camera, side_camera]

    @property
    def _default_human_render_camera_configs(self):
        """Configure camera for human viewing"""
        return CameraConfig(
            "render_camera",
            pose=look_at(eye=[0.6, 0.6, 0.6], target=[0, 0, 0.1]),
            width=512,
            height=512,
            fov=np.pi / 3,
            near=0.01,
            far=100,
        )
