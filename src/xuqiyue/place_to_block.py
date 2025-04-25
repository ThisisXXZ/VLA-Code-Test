from typing_extensions import *
import sapien
import torch
import numpy as np
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.sapien_utils import look_at
from mani_skill.utils.structs import Pose
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.structs.actor import Actor


@register_env("Customv1", max_episode_steps=100)
class CustomEnv(BaseEnv):
    """
    Task: Put a thing into a container.
    """

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]

    def __init__(
        self,
        *args,
        robot_uids="panda",
        num_envs=1,
        reconfiguration_freq=None,
        num_things=2,
        **kwargs,
    ):
        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if num_envs == 1 else 0

        self.num_things = num_things

        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq,
            num_envs=num_envs,
            **kwargs,
        )

    def _load_agent(self, options, initial_agent_poses=sapien.Pose(p=[-0.615, 0, 0])):
        return super()._load_agent(options, initial_agent_poses)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(env=self)
        self.table_scene.build()

        import random
        from .drawer import get_drawer

        self.thickness = 0.02
        self.center = (0, -0.5, 0.01)
        self.size = (0.2, 0.3, 0.1)
        self.container = get_drawer(
            scene=self.scene,
            size=self.size,
            thickness=self.thickness,
            name="container",
            center=self.center,
        )

        #! Things to put into container
        self.box_size = 0.03
        self.things = []
        for i in range(self.num_things):
            builder = self.scene.create_actor_builder()
            builder.add_box_collision(half_size=[self.box_size / 2] * 3)
            builder.add_box_visual(
                half_size=[self.box_size / 2] * 3,
                material=sapien.render.RenderMaterial(base_color=[0, 0, 0.7, 1.0]),
            )
            builder.initial_pose = sapien.Pose(
                p=[random.random() / 2, random.random() / 2, self.box_size / 2]
            )
            thing = builder.build(name=f"thing_{i}")
            self.things.append(thing)

    def _initialize_episode(self, env_idx: torch.Tensor, options):
        self.table_scene.initialize(env_idx)
        with torch.device(self.device):
            b = len(env_idx)

            #! container position
            center_pos = torch.ones(b, 3)
            center_pos[:, 0] = torch.rand((b,)) / 4
            center_pos[:, 1] = torch.rand((b,)) / 4
            center_pos[:, 2] = self.thickness / 2
            self.container.set_pose(Pose.create_from_pq(p=center_pos, q=[1, 0, 0, 0]))
            self.container.set_linear_velocity(torch.zeros((b, 3), device=self.device))
            self.container.set_angular_velocity(torch.zeros((b, 3), device=self.device))

            #! things position
            for i, thing in enumerate(self.things):
                thing_pos = torch.ones(b, 3)
                thing_pos[:, 0] = torch.rand((b,)) / 2
                thing_pos[:, 1] = torch.rand((b,)) / 2
                thing_pos[:, 2] = self.box_size / 2
                thing.set_pose(Pose.create_from_pq(p=thing_pos, q=[1, 0, 0, 0]))

                thing.set_linear_velocity(torch.zeros((b, 3), device=self.device))
                thing.set_angular_velocity(torch.zeros((b, 3), device=self.device))

    def evaluate(self):
        with torch.device(self.device):
            pose = torch.zeros((self.num_things, 3))
            success = torch.zeros(self.num_things, dtype=torch.bool)
            distance = torch.zeros(self.num_things)

            for i, thing in enumerate(self.things):
                pose[i, :] = thing.pose.p
                distance[i] = torch.linalg.norm(self.container.pose.p - thing.pose.p)
                success[i] = (
                    (
                        self.center[0] - self.size[0] / 2
                        <= pose[i, 0]
                        <= self.center[0] + self.size[0] / 2
                    )
                    and (
                        self.center[1] - self.size[1] / 2
                        <= pose[i, 1]
                        <= self.center[1] + self.size[1] / 2
                    )
                    and (self.center[2] <= pose[i, 2] <= self.center[2] + self.size[2])
                )
                
            success = torch.tensor([success.all()])

            return {
                "success": success,
                "pos": pose,
                "distance": distance,
            }

    def _get_obs_extra(self, info: Dict):
        """Additional observations for solving the task"""
        # obs = dict(
        #     tcp_pose=self.agent.tcp.pose.raw_pose,
        #     goal_pos=self.goal_position,
        # )

        # if self.obs_mode_struct.use_state:
        #     # Add ground truth information if using state observations
        #     for i, card in enumerate(self.cards):
        #         obs[f"card_{i}_pose"] = card.pose.raw_pose
        #         obs[f"card_{i}_vel"] = card.linear_velocity

        return {}

    def compute_dense_reward(self, obs, action, info: Dict):
        with torch.device(self.device):
            reward = torch.zeros(self.num_envs, device=self.device)
            success = info["success"]
            distance = info["distance"]
            reward = torch.where(success, reward + 10, reward)

            for i in range(self.num_things):
                reward += torch.exp(-10.8 * distance[i]) * 0.2

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
