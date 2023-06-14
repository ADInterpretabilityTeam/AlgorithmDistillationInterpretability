import cv2
import numpy as np
import torch
from minigrid.core.constants import IDX_TO_OBJECT


def find_agent(observation):
    height = observation.shape[0]
    width = observation.shape[1]
    for i in range(width):
        for j in range(height):
            object = IDX_TO_OBJECT[int(observation[i, j][0])]
            if object == "agent":
                return i, j

    return -1, -1


def render_minigrid_observation(env, observation):
    if isinstance(observation, np.ndarray):
        observation = (
            observation.copy()
        )  # so we don't edit the original object
    elif isinstance(observation, torch.Tensor):
        observation = observation.numpy().copy()

    agent_pos = find_agent(observation)
    agent_dir = observation[agent_pos[0], agent_pos[1]][2]

    # print(agent_pos, agent_dir)
    # observation[agent_pos[0], agent_pos[1]] = [0, 0, 0]
    # import streamlit as st
    # st.write(env.spec.id)
    # st.write(env.observation_space)
    grid, _ = env.grid.decode(observation.astype(np.uint8))

    i = agent_pos[0]
    j = agent_pos[1]

    return grid.render(32, (i, j), agent_dir=agent_dir)


def render_minigrid_observations(env, observations):
    return np.array(
        [
            render_minigrid_observation(env, observation)
            for observation in observations
        ]
    )


def render_video_from_policy(model, video_file_name):
    env = model.get_env()

    obs = env.reset()
    n_episodes, n_obs = obs.shape
    lstm_states = None
    episode_starts = np.ones((n_episodes,), dtype=bool)

    fourc = cv2.VideoWriter_fourcc(*'avc1')
    x, y, _ = env.render().shape
    video = cv2.VideoWriter(f'{video_file_name}.mp4', fourc, 2.0, (x, y))
    done = np.zeros((n_episodes,), dtype=bool)

    while not np.all(done):
        # Write to video
        render = env.render()
        render = cv2.cvtColor(render, cv2.COLOR_BGR2RGB)
        video.write(render)
        # Step    
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=False)
        obs, reward, done, info = env.step(action)
        episode_starts = done

    video.release()


def visualize_learning_history(file_name, every_n_eps=1):
    # Load in data
    data = np.load(file_name, allow_pickle=True)
    env = data['env'].item()
    _ = env.reset()
    ep_len = env.max_steps
    states = data["observations"]
    actions = data["actions"]
    total_eps = data["observations"].shape[0] // ep_len
    # Create video
    frames = []
    # Iterate over episodes
    for ep in range(0, total_eps, every_n_eps):
        start_idx = ep * ep_len
        end_idx = start_idx + ep_len
        # Initialize counters
        _ = env.reset()
        prev_state = -1
        prev_action = -1
        current_state = np.argmax(states[start_idx, :])
        for i in range(start_idx + 1, end_idx):
            # Write to video
            env.current_state = current_state
            env.prev_state = prev_state
            env.prev_action = prev_action
            env.update_flag()
            render = env.render()
            frames.append(render)
            # Step
            prev_state = current_state
            prev_action = actions[i - 1, 0]
            current_state = np.argmax(states[i, :])
    return frames


def learning_history_to_video(file_name, video_file_name, every_n_eps=1):
    # Load in data
    frames = visualize_learning_history(file_name, every_n_eps)
    # Create video
    fourc = cv2.VideoWriter_fourcc(*'avc1')
    x, y, _ = frames[0].shape
    video = cv2.VideoWriter(f'{video_file_name}.mp4', fourc, 1.0, (x, y))
    # Iterate over frames
    for frame in frames:
        render = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # cv2 reads in images as BGR by default
        video.write(render)
    video.release()