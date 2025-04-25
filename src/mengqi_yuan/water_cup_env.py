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
from scipy.spatial.transform import Rotation as R
import random

@register_env("WaterCup-v1", max_episode_steps=100)
class WaterCupEnv(BaseEnv):

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]

    def __init__(self, *args, robot_uids="panda", num_envs=1, reconfiguration_freq=None, **kwargs):
        
        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if num_envs == 1 else 0

        super().__init__(*args, robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq, num_envs=num_envs, **kwargs)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(env=self)
        self.table_scene.build()
        
        self.cups = []
        self.sizes = [0.1, 0.05, 0.1, 0.05, 0.075]
        colors = [[0.5, 0.2, 0.8, 1.0], [0.2, 0.5, 0.8, 1.0], [0.8, 0.2, 0.5, 1.0], [0.2, 0.8, 0.5, 1.0], [0.8, 0.5, 0.2, 1.0]]

        # create 5 cups
        for i, size in enumerate(self.sizes):
            builder = self.scene.create_actor_builder()
            builder.add_cylinder_collision(radius=0.05, half_length=size)
            builder.add_cylinder_visual(radius=0.05, half_length=size, material=sapien.render.RenderMaterial(base_color=colors[i])) #作用：
            place = 0.2 * i
            builder.initial_pose = sapien.Pose(p=[0.5, place, size + 0.5], q=[0.7071, 0, 0.7071, 0]) 
            cup = builder.build(name=f"cup_{i}")
            self.cups.append(cup)

        # create 5 target place
        self.marker_positions = []
        y_start, y_end = -0.4, 0.4  
        marker_positions = [[0, y, 0.01] for y in np.linspace(y_start, y_end, 5)]
        marker_color = [0.9, 0.1, 0.1, 0.3]  

        for i, pos in enumerate(marker_positions):
            marker_builder = self.scene.create_actor_builder()
            marker_builder.add_cylinder_visual(radius=0.05, half_length=0.001, material=sapien.render.RenderMaterial(base_color=marker_color))
            marker_builder.initial_pose = sapien.Pose(p=pos, q=[0.7071, 0, 0.7071, 0]) # switch to xy plane
            marker_position = marker_builder.build_static(name=f"marker_{i}")
            self.marker_positions.append(marker_position)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.table_scene.initialize(env_idx)
        
        with torch.device(self.device):
            b = len(env_idx)
            # set marker positions
            marker_pos = torch.tensor([
                [0, -0.4, 0.01],
                [0, -0.2, 0.01],
                [0, 0.0, 0.01],
                [0, 0.2, 0.01],
                [0, 0.4, 0.01]
            ])
            marker_pose = Pose.create_from_pq(p=marker_pos, q=[0.7071, 0, 0.7071, 0])
            for i in range(len(self.marker_positions)):
                #print(f"Setting marker {i}: {self.marker_positions[i]} to pose {marker_pose[i]}")
                self.marker_positions[i].set_pose(marker_pose[i])

            # sort the cups by size, and store the order. set the goal positions accordingly
            self.order = []
            for i in range(len(self.sizes)):
                self.order.append(i)
            self.order.sort(key=lambda x: self.sizes[x], reverse=True)
            #print(self.order)
            goal_pos = torch.tensor([
                [0, -0.4, self.sizes[self.order[0]]],
                [0, -0.2, self.sizes[self.order[1]]],
                [0, 0.0, self.sizes[self.order[2]]],
                [0, 0.2, self.sizes[self.order[3]]],
                [0, 0.4, self.sizes[self.order[4]]]
            ])
            goal_pose = Pose.create_from_pq(p=goal_pos, q=[0.7071, 0, 0.7071, 0])
            self.goal_positions = goal_pose
            
            for i, cup in enumerate(self.cups):
                valid_pos = False
                # randomly decide if cup is upright or sideways
                orientations = [
                    [1, 0, 0, 0],  # Parallel to xz plane
                    [0.7071, 0, 0.7071, 0]  # Parallel to xy plane
                ]
                cup_orientation = random.choice(orientations)

                # set the cup pose, make sure one cups won't embed into another
                while not valid_pos:
                    cup_pos = torch.zeros((b, 3))
                    cup_pos[..., 0] = torch.rand((b,)) * 0.6 - 0.3
                    cup_pos[..., 1] = torch.rand((b,)) * 0.8 - 0.4
                    valid_pos = True
                    # find a valid position that is not occupied by another existing cup
                    for j in range(i):
                        prev_pos = self.cups[j].pose.p
                        q = self.cups[j].pose.q
                        if q == [0.7071, 0, 0.7071, 0] and cup_orientation == [0.7071, 0, 0.7071, 0]:
                            distances = torch.norm(cup_pos[..., :2] - prev_pos[..., :2], dim=-1)
                            max_dis = 0.1
                        elif q == [1, 0, 0, 0] and cup_orientation == [1, 0, 0, 0]:
                            distances = torch.norm(cup_pos[..., :2] - prev_pos[..., :2], dim=-1)
                            max_dis = torch.sqrt(0.1 ** 2 + (cup_pos[..., 2] + prev_pos[..., 2]) ** 2)
                        elif q == [0.7071, 0, 0.7071, 0] and cup_orientation == [1, 0, 0, 0]:
                            distances = torch.norm(cup_pos[..., :2] - prev_pos[..., :2], dim=-1)
                            max_dis = torch.sqrt(0.1 ** 2 + (0.05 + prev_pos[..., 2]) ** 2)
                        else:
                            distances = torch.norm(cup_pos[..., :2] - prev_pos[..., :2], dim=-1)
                            max_dis = torch.sqrt(0.1 ** 2 + (0.05 + cup_pos[..., 2]) ** 2)
                        if torch.any(distances < max_dis):
                            valid_pos = False
                            break      
                # decide the height based on the pose
                cup_pos[..., 2] = self.sizes[i] if cup_orientation == [0.7071, 0, 0.7071, 0] else 0.05
                cup_pose = Pose.create_from_pq(p=cup_pos, q=cup_orientation)
                cup.set_pose(cup_pose)
                cup.set_linear_velocity(torch.zeros((b, 3), device=self.device))
                cup.set_angular_velocity(torch.zeros((b, 3), device=self.device))

    def evaluate(self):
        with torch.device(self.device):
            at_goal = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            upright_all = []
            distance_all = []

            for i, cup in enumerate(self.cups):
                # find the index of the cup in the target order
                for j in range(len(self.order)):
                    if self.order[j] == i:
                        break
                cup_position = cup.pose.p  # Shape: [b, 3]
                goal_position = self.goal_positions[j]  # Shape: [b, 3]
                #print(f"cup_position: {cup_position}, goal_position: {goal_position}")
                at_position = torch.linalg.norm(cup_position[..., :2] - goal_position.raw_pose[:, :2], dim=-1) < 0.005
                target_q = torch.tensor([0.7071, 0, 0.7071, 0], device=cup.pose.q.device)
                upright = torch.norm(cup.pose.q - target_q, dim=-1) < 0.005
                # success if the cup is at the right position (in right order & right place) and upright
                at_goal &= at_position & upright
                upright_all.append(upright)
                distance_all.append(torch.linalg.norm(cup_position[..., :2] - goal_position.raw_pose[:, :2], dim=-1))
            
            return {
                "success": at_goal,
                "at_goal": at_goal,
                "upright": torch.stack(upright_all),
                "distance": torch.stack(distance_all)
            }

    def _get_obs_extra(self, info: Dict):
        """Additional observations for solving the task"""
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            goal_positions=self.goal_positions.raw_pose,
        )
        
        if self.obs_mode_struct.use_state:
            # Add ground truth information if using state observations
            for i, cup in enumerate(self.cups):
                obs[f"cup_{i}_pose"] = cup.pose.raw_pose
                obs[f"cup_{i}_vel"] = cup.linear_velocity
        
        return obs
        
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """Compute a dense reward signal to guide learning"""
        with torch.device(self.device):
            reward = torch.zeros(self.num_envs, device=self.device)
            
            # Success reward
            success = info["success"]
            reward = torch.where(success, 10.0, reward)
            
            # reward for each cup: upright & right place
            for i, cup in enumerate(self.cups):
                upright = info["upright"][i]
                distance = info["distance"][i]
                reward += torch.where(upright, torch.exp(-5.0 * distance) * 0.5, reward)
            return reward
    
    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """Normalize the dense reward"""
        # manually set the max reward
        max_reward = 12.5
        return self.compute_dense_reward(obs, action, info) / max_reward

    @property
    def _default_sensor_configs(self):
        """Configure camera sensors for the environment"""
        # Top-down camera view
        top_camera = CameraConfig("top_camera", pose=look_at(eye=[0, 0, 1.0], target=[0, 0, 0]),
            width=128, height=128, fov=np.pi/3, near=0.01, far=100)
        
        # Side view camera
        side_camera = CameraConfig("side_camera", pose=look_at(eye=[1.0, 0, 0.5], target=[0, 0, 0.2]),
            width=128, height=128, fov=np.pi/3, near=0.01, far=100)
        return [top_camera, side_camera]
    
    @property
    def _default_human_render_camera_configs(self):
        """Configure camera for human viewing"""
        return CameraConfig("render_camera", pose=look_at(eye=[1.0, 0, 0.7], target=[0, 0, 0.2]),
            width=512, height=512, fov=np.pi/3, near=0.01, far=100)