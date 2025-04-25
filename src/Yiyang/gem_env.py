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

# Define gem and container sizes
GEM_SIZE = [0.02, 0.02, 0.02]  # Half-sizes for x, y, z (m)
CONTAINER_SIZE = [0.05, 0.05, 0.02]  # Half-sizes for x, y, z (m)

# Define colors for gems and containers
COLORS = [
    [1.0, 0.2, 0.2, 1.0],  # Red
    [0.2, 0.6, 1.0, 1.0],  # Blue
    [0.2, 0.8, 0.2, 1.0],  # Green
    [0.8, 0.8, 0.2, 1.0],  # Yellow
]

@register_env("GemSorting-v1", max_episode_steps=150)
class GemSortingEnv(BaseEnv):
    """
    Task: Sort colored gems into their matching color containers.
    The robot needs to pick up gems and place them in containers of the same color.
    """
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]

    def __init__(self, *args, robot_uids="panda", num_envs=1,
                 reconfiguration_freq=None, num_gems=3, num_colors=3, **kwargs):
        # Set reconfiguration frequency - for single env, reconfigure every time
        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if num_envs == 1 else 0
        
        # Use 3 gems and 3 colors by default to ensure each gem has a unique color
        self.num_gems = num_gems
        self.num_colors = min(num_colors, len(COLORS))
        # Ensure the number of gems does not exceed the number of colors
        if self.num_gems > self.num_colors:
            self.num_gems = self.num_colors
        
        super().__init__(*args, robot_uids=robot_uids,
                         reconfiguration_freq=reconfiguration_freq, num_envs=num_envs, **kwargs)

    def _load_scene(self, options: dict):
        # Create the table scene
        self.table_scene = TableSceneBuilder(env=self)
        self.table_scene.build()
        
        self.gems = []
        self.containers = []
        
        # Create gems with different colors
        for i in range(self.num_gems):
            color_id = i % self.num_colors
            color = COLORS[color_id]
            
            builder = self.scene.create_actor_builder()
            builder.add_sphere_collision(radius=GEM_SIZE[0])
            builder.add_sphere_visual(
                radius=GEM_SIZE[0],
                material=sapien.render.RenderMaterial(base_color=color)
            )
            
            # Place gems initially above the table
            builder.initial_pose = sapien.Pose(p=[0, 0, GEM_SIZE[2] + 0.5], q=[1, 0, 0, 0])
            gem = builder.build(name=f"gem_{i}")
            
            self.gems.append({
                "actor": gem,
                "color": color,
                "color_id": color_id
            })
        
        # Create containers for each color
        for i in range(self.num_colors):
            color = COLORS[i]
            container_color = [color[0], color[1], color[2], 0.5]  # Semi-transparent
            
            builder = self.scene.create_actor_builder()
            
            # Create container with hollow center (using multiple boxes)
            # Base (bottom)
            builder.add_box_collision(
                half_size=[CONTAINER_SIZE[0], CONTAINER_SIZE[1], 0.005],
                pose=sapien.Pose(p=[0, 0, 0.005])
            )
            builder.add_box_visual(
                half_size=[CONTAINER_SIZE[0], CONTAINER_SIZE[1], 0.005],
                pose=sapien.Pose(p=[0, 0, 0.005]),
                material=sapien.render.RenderMaterial(base_color=container_color)
            )
            
            # Walls (4 sides)
            wall_height = CONTAINER_SIZE[2] - 0.005
            wall_half_height = wall_height / 2
            wall_thickness = 0.005
            
            # Front wall
            builder.add_box_collision(
                half_size=[CONTAINER_SIZE[0], wall_thickness, wall_half_height],
                pose=sapien.Pose(p=[0, CONTAINER_SIZE[1] - wall_thickness, 0.005 + wall_half_height])
            )
            builder.add_box_visual(
                half_size=[CONTAINER_SIZE[0], wall_thickness, wall_half_height],
                pose=sapien.Pose(p=[0, CONTAINER_SIZE[1] - wall_thickness, 0.005 + wall_half_height]),
                material=sapien.render.RenderMaterial(base_color=container_color)
            )
            
            # Back wall
            builder.add_box_collision(
                half_size=[CONTAINER_SIZE[0], wall_thickness, wall_half_height],
                pose=sapien.Pose(p=[0, -CONTAINER_SIZE[1] + wall_thickness, 0.005 + wall_half_height])
            )
            builder.add_box_visual(
                half_size=[CONTAINER_SIZE[0], wall_thickness, wall_half_height],
                pose=sapien.Pose(p=[0, -CONTAINER_SIZE[1] + wall_thickness, 0.005 + wall_half_height]),
                material=sapien.render.RenderMaterial(base_color=container_color)
            )
            
            # Left wall
            builder.add_box_collision(
                half_size=[wall_thickness, CONTAINER_SIZE[1], wall_half_height],
                pose=sapien.Pose(p=[-CONTAINER_SIZE[0] + wall_thickness, 0, 0.005 + wall_half_height])
            )
            builder.add_box_visual(
                half_size=[wall_thickness, CONTAINER_SIZE[1], wall_half_height],
                pose=sapien.Pose(p=[-CONTAINER_SIZE[0] + wall_thickness, 0, 0.005 + wall_half_height]),
                material=sapien.render.RenderMaterial(base_color=container_color)
            )
            
            # Right wall
            builder.add_box_collision(
                half_size=[wall_thickness, CONTAINER_SIZE[1], wall_half_height],
                pose=sapien.Pose(p=[CONTAINER_SIZE[0] - wall_thickness, 0, 0.005 + wall_half_height])
            )
            builder.add_box_visual(
                half_size=[wall_thickness, CONTAINER_SIZE[1], wall_half_height],
                pose=sapien.Pose(p=[CONTAINER_SIZE[0] - wall_thickness, 0, 0.005 + wall_half_height]),
                material=sapien.render.RenderMaterial(base_color=container_color)
            )
            
            # Place container on the table
            builder.initial_pose = sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0])
            container = builder.build_static(name=f"container_{i}")
            
            self.containers.append({
                "actor": container,
                "color": color,
                "color_id": i
            })

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # Initialize the table scene
        self.table_scene.initialize(env_idx)
        
        with torch.device(self.device):
            b = len(env_idx)
            
            # Define workspace boundaries to ensure all objects are on the table
            table_bounds = {
                "min_x": -0.3,
                "max_x": 0.3,
                "min_y": -0.3,
                "max_y": 0.3
            }
            
            # Randomize container positions (in a semi-circle arrangement)
            container_radius = 0.25  # Distance from center
            container_angle_step = np.pi / (self.num_colors + 1)
            
            for i, container_info in enumerate(self.containers):
                container = container_info["actor"]
                
                # Calculate position in semi-circle
                angle = np.pi/2 + container_angle_step * (i + 1)
                container_pos = torch.zeros((b, 3), device=self.device)
                container_pos[..., 0] = torch.tensor(container_radius * np.cos(angle), device=self.device)
                container_pos[..., 1] = torch.tensor(container_radius * np.sin(angle), device=self.device)
                container_pos[..., 2] = 0.001  # Slightly above table
                
                # Ensure container is within table boundaries
                container_pos[..., 0] = torch.clamp(
                    container_pos[..., 0],
                    min=table_bounds["min_x"] + CONTAINER_SIZE[0],
                    max=table_bounds["max_x"] - CONTAINER_SIZE[0]
                )
                container_pos[..., 1] = torch.clamp(
                    container_pos[..., 1],
                    min=table_bounds["min_y"] + CONTAINER_SIZE[1],
                    max=table_bounds["max_y"] - CONTAINER_SIZE[1]
                )
                
                container_pose = Pose.create_from_pq(p=container_pos, q=[1, 0, 0, 0])
                container.set_pose(container_pose)
            
            # Refine gem randomization
            max_attempts = 50
            center_area_size = 0.15
            
            # Generate a grid of possible positions and shuffle
            grid_size = 5  # 5x5 grid
            positions_x = torch.linspace(-center_area_size/2, center_area_size/2, grid_size)
            positions_y = torch.linspace(-center_area_size/2, center_area_size/2, grid_size)
            
            grid_positions = []
            for x in positions_x:
                for y in positions_y:
                    grid_positions.append((x.item(), y.item()))
            
            # Shuffle positions (shared across all environments)
            np.random.shuffle(grid_positions)
            
            # Assign unique colors to each gem in each environment
            color_indices = torch.arange(self.num_colors, device=self.device)
            for i, gem_info in enumerate(self.gems):
                if i < self.num_gems:
                    color_ids = torch.zeros(b, device=self.device, dtype=torch.long)
                    for env_i in range(b):
                        color_indices_shuffled = color_indices[torch.randperm(self.num_colors)]
                        color_ids[env_i] = color_indices_shuffled[i]
                    gem_info["color_id"] = color_ids
            
            # Place gems
            for i, gem_info in enumerate(self.gems):
                gem = gem_info["actor"]
                
                gem_pos = torch.zeros((b, 3), device=self.device)
                
                for env_i in range(b):
                    pos_idx = i % len(grid_positions)
                    gem_pos[env_i, 0] = torch.tensor(grid_positions[pos_idx][0], device=self.device)
                    gem_pos[env_i, 1] = torch.tensor(grid_positions[pos_idx][1], device=self.device)
                
                gem_pos[..., 2] = GEM_SIZE[2]  # Height from table
                
                # Ensure gem is within table boundaries
                gem_pos[..., 0] = torch.clamp(
                    gem_pos[..., 0],
                    min=table_bounds["min_x"] + GEM_SIZE[0] + 0.02,
                    max=table_bounds["max_x"] - GEM_SIZE[0] - 0.02
                )
                gem_pos[..., 1] = torch.clamp(
                    gem_pos[..., 1],
                    min=table_bounds["min_y"] + GEM_SIZE[1] + 0.02,
                    max=table_bounds["max_y"] - GEM_SIZE[1] - 0.02
                )
                
                gem_pose = Pose.create_from_pq(p=gem_pos, q=[1, 0, 0, 0])
                gem.set_pose(gem_pose)
                
                # Reset velocities
                gem.set_linear_velocity(torch.zeros((b, 3), device=self.device))
                gem.set_angular_velocity(torch.zeros((b, 3), device=self.device))
            
            # Record initial positions of gems for reward calculation
            self.gem_initial_positions = torch.stack([gem_info["actor"].pose.p for gem_info in self.gems])

    def evaluate(self):
        """Determine success/failure of the task"""
        with torch.device(self.device):
            # Get gem positions
            gem_positions = torch.stack([gem_info["actor"].pose.p for gem_info in self.gems])
            # Get container positions
            container_positions = torch.stack([container_info["actor"].pose.p for container_info in self.containers])
            
            # Define container height range for success condition
            container_bottom_height = 0.005  # Container bottom height
            container_top_height = container_bottom_height + 0.015  # Container top height (bottom + wall)
            
            # Check if each gem is in its matching color container
            gems_in_correct_container = torch.zeros((self.num_gems, self.num_envs), 
                                                    dtype=torch.bool, device=self.device)
            gems_in_wrong_container = torch.zeros((self.num_gems, self.num_envs), 
                                                  dtype=torch.bool, device=self.device)
            
            for i, gem_info in enumerate(self.gems):
                gem_color_id = gem_info["color_id"]
                
                for j, container_info in enumerate(self.containers):
                    container_color_id = container_info["color_id"]
                    
                    # Check if gem is above container
                    dist_xy = torch.linalg.norm(
                        gem_positions[i, :, :2] - container_positions[j, :, :2], dim=1
                    )
                    
                    # Check if gem is inside container boundaries
                    inside_container = dist_xy < (CONTAINER_SIZE[0] - 0.01)
                    
                    # Check if gem is at proper height (inside container)
                    proper_height = (gem_positions[i, :, 2] > container_bottom_height) & \
                                    (gem_positions[i, :, 2] < container_top_height)
                    
                    # Check if gem is in this container
                    in_this_container = inside_container & proper_height
                    
                    # Check if color matches
                    color_match = gem_color_id == container_color_id
                    
                    # Gem is correctly placed if it's in the matching color container
                    gems_in_correct_container[i] |= in_this_container & color_match
                    
                    # Detect wrong placement: gem in container but color does not match
                    gems_in_wrong_container[i] |= in_this_container & ~color_match
            
            # Count correctly placed gems for each environment
            correct_count = torch.sum(gems_in_correct_container, dim=0)
            wrong_count = torch.sum(gems_in_wrong_container, dim=0)
            
            # Success if all gems are in correct containers
            success = correct_count == self.num_gems
            
            return {
                "success": success,
                "gems_correct": gems_in_correct_container,
                "gems_wrong": gems_in_wrong_container,
                "correct_count": correct_count,
                "wrong_count": wrong_count,
            }

    def _get_obs_extra(self, info: Dict):
        """Additional observations for solving the task"""
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        
        if self.obs_mode_struct.use_state:
            # Add ground truth information if using state observations
            for i, gem_info in enumerate(self.gems):
                gem = gem_info["actor"]
                obs[f"gem_{i}_pose"] = gem.pose.raw_pose
                obs[f"gem_{i}_vel"] = gem.linear_velocity
                obs[f"gem_{i}_color_id"] = gem_info["color_id"]
            
            for i, container_info in enumerate(self.containers):
                container = container_info["actor"]
                obs[f"container_{i}_pose"] = container.pose.raw_pose
                obs[f"container_{i}_color_id"] = torch.tensor(
                    container_info["color_id"], device=self.device
                ).expand(self.num_envs)
        
        return obs
        
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """Compute a dense reward signal to guide learning"""
        with torch.device(self.device):
            reward = torch.zeros(self.num_envs, device=self.device)
            
            # Success reward for each correctly placed gem
            gems_correct = info["gems_correct"]  # Shape: [n_gems, b]
            for i in range(self.num_gems):
                reward = torch.where(gems_correct[i], reward + 2.0, reward)
            
            # Penalty for incorrectly placed gems
            gems_wrong = info["gems_wrong"]
            for i in range(self.num_gems):
                reward = torch.where(gems_wrong[i], reward - 1.0, reward)
            
            # Large success reward for completing the task
            success = info["success"]
            reward = torch.where(success, reward + 5.0, reward)
            
            # Get gem and TCP positions for distance-based rewards
            gem_positions = torch.stack([gem_info["actor"].pose.p for gem_info in self.gems])
            container_positions = torch.stack([container_info["actor"].pose.p for container_info in self.containers])
            tcp_pos = self.agent.tcp.pose.p  # Shape: [b, 3]
            
            # Prioritize unfinished gems
            for i, gem_info in enumerate(self.gems):
                gem_color_id = gem_info["color_id"]  # [num_envs]
                gem_correctly_placed = gems_correct[i]  # [num_envs]
                
                # TCP to gem proximity reward for unfinished gems
                tcp_to_gem_dist = torch.linalg.norm(gem_positions[i] - tcp_pos, dim=1)
                proximity_reward = torch.exp(-5.0 * tcp_to_gem_dist) * 0.2
                reward = torch.where(~gem_correctly_placed, reward + proximity_reward, reward)
                
                for j in range(self.num_colors):
                    matching_color_mask = (gem_color_id == j)
                    
                    if torch.any(matching_color_mask):
                        combined_mask = matching_color_mask & ~gem_correctly_placed
                        
                        if torch.any(combined_mask):
                            container_pos = container_positions[j]
                            gem_to_container_dist = torch.linalg.norm(
                                gem_positions[i] - container_pos, dim=1
                            )
                            
                            distance_reward = torch.exp(-3.0 * gem_to_container_dist) * 0.3
                            reward = torch.where(combined_mask, reward + distance_reward, reward)
            
            # Penalty for gems falling off the table
            table_height = 0.0  # Table height
            for i in range(self.num_gems):
                gem_below_table = gem_positions[i, :, 2] < table_height - 0.05
                reward = torch.where(gem_below_table, reward - 2.0, reward)
            
            return reward
    
    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """Normalize the dense reward"""
        # Maximum possible reward: success + all correct placements + proximity rewards
        max_reward = 5.0 + (2.0 * self.num_gems) + 1.0
        # Minimum possible reward: all gems wrong + all gems fallen
        min_reward = -(1.0 * self.num_gems) - (2.0 * self.num_gems)
        
        raw_reward = self.compute_dense_reward(obs, action, info)
        return (raw_reward - min_reward) / (max_reward - min_reward)

    @property
    def _default_sensor_configs(self):
        """Configure camera sensors for the environment"""
        # Top-down camera view
        top_camera = CameraConfig("top_camera", pose=look_at(eye=[0, 0, 0.6], target=[0, 0, 0]),
                                  width=128, height=128, fov=np.pi/3, near=0.01, far=100)
        
        # Front view camera - looks toward the containers
        front_camera = CameraConfig("front_camera", pose=look_at(eye=[0, -0.5, 0.4], target=[0, 0.2, 0.1]),
                                    width=128, height=128, fov=np.pi/3, near=0.01, far=100)
            
        return [top_camera, front_camera]
    
    @property
    def _default_human_render_camera_configs(self):
        """Configure camera for human viewing"""
        return CameraConfig("render_camera", pose=look_at(eye=[0.5, -0.5, 0.6], target=[0, 0, 0.1]),
                            width=512, height=512, fov=np.pi/3, near=0.01, far=100)
