import os
import gymnasium as gym
import time
from mani_skill.utils.wrappers.record import RecordEpisode
from examples.card_stack_env import CardStackEnv
from xxz.collect_capsules_env import CollectCapsulesEnv
from xxz.five_in_a_row import FiveInARowEnv

def generate_videos(n_episodes=10, max_steps_per_episode=100, video_dir="card_stack_videos"):
    """
    Generate and save videos of random agent interactions in the CardStack environment.
    """
    env = gym.make("FiveInARowEnv-v0", obs_mode="state", render_mode="human")
    obs, _ = env.reset(options=dict(reconfigure=True))
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

    # env = gym.make("DrawSVG-v1", obs_mode="state", render_mode="rgb_array")
    # video_dir = os.path.join(video_dir, time.strftime("%Y%m%d-%H%M%S"))
    # os.makedirs(video_dir, exist_ok=True)

    # env = RecordEpisode(env, output_dir=video_dir, save_video=True, 
    #                     trajectory_name="random_actions", max_steps_per_video=max_steps_per_episode)
    # for _ in range(n_episodes):
    #     obs, info = env.reset()
    #     for _ in range(max_steps_per_episode):
    #         action = env.action_space.sample()  # Take random action
    #         obs, reward, terminated, truncated, info = env.step(action)
    #         if terminated or truncated:
    #             break
    # env.close()

if __name__ == "__main__":
    generate_videos(n_episodes=10)
