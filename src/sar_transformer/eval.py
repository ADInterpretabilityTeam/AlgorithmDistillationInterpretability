import os

import gymnasium as gym
import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm
from copy import deepcopy

import wandb
from src.models.trajectory_transformer import (
    DecisionTransformer,
    AlgorithmDistillationTransformer
)
from src.sar_transformer.utils import get_max_len_from_model_type

from src.config import EnvironmentConfig
from src.generation import value_iteration


def evaluate_random_agent(env, n_its=1_000):
    # Simulate random walks
    scores = []
    for i in range(n_its):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
        scores.append(total_reward)
    # Average episodic rewards
    avg_reward = sum(scores) / len(scores)
    return avg_reward


def evaluate_optimal_agent(env, n_rollouts=1_000):
    values, policy = value_iteration(env, 0.99, 1e-5)
    policy = policy.astype(np.int32)
    # Do a bunch of rollouts
    scores = []
    for _ in range(n_rollouts):
        obs, _  = env.reset()
        done = False
        total_reward = 0
        while not done:
            curr_state = np.argmax(obs)
            action = policy[curr_state]
            obs, rew, done, _, _ = env.step(action)
            total_reward += rew
        scores.append(total_reward)
    return sum(scores) / len(scores)


def evaluate_ad_agent(
    model: AlgorithmDistillationTransformer,
    env_config: EnvironmentConfig,
    n_episodes: int,
    temp: float = 1.,
    device: str = "cuda",
    track: bool = False,
    return_cache: bool = False
):
    # Set up model for evaluation
    model = model.to(device)
    model.eval()
    max_len = get_max_len_from_model_type(
        model.model_type,
        model.transformer_config.n_ctx,
    )
    
    # Set up enivornment statistics
    ep_rewards = [0]
    episode_length = env_config.max_steps
    assert max_len % episode_length == 0  # This should be the case, or else the model and env are mismatched
    n_prev_episodes = max_len // episode_length
    env = env_config.env
    env.generate()
    n_obs = env.observation_space.shape[0]
    
    # Measure baseline scores
    random_score = evaluate_random_agent(env)
    optimal_score = evaluate_optimal_agent(env)
    high_score = -1
    
    # Set up buffers
    total_steps = 0
    tokens = None
    embeddings = None
    state_buffer = np.zeros((1, max_len, n_obs))
    action_buffer = np.zeros((1, max_len, 1))
    reward_buffer = np.zeros((1, max_len, 1))
    time_buffer = np.zeros((1, max_len, 1))
    
    # Progress bar
    pbar = tqdm(total=n_episodes)
    current_episode = 0
    
    obs, _ = env.reset()
    current_timestep = 0
    state_buffer[0, 0] = obs
        
    while not (current_episode == n_episodes):
                
        # Convert buffers to torch tensors
        states = torch.as_tensor(state_buffer, dtype=torch.double, device=device)
        actions = torch.as_tensor(action_buffer, dtype=torch.long, device=device)[:, :-1]
        rewards = torch.as_tensor(reward_buffer, dtype=torch.double, device=device)[:, :-1]
        time = torch.as_tensor(time_buffer, dtype=torch.long, device=device)
               
        # Get action prediction
        _, action_preds, _ = model.forward(
            states=states,
            actions=actions,
            rewards=rewards,
            timesteps=time,
        )
        idx = min(max_len - 1, total_steps)
        act_probs = torch.softmax(action_preds[0, idx] / temp, dim=-1).detach().cpu().numpy()
        if temp == 0:
            act = act_probs.argmax(-1)  # Greedy sampling
        else:
            act = np.random.choice(np.arange(act_probs.shape[0]), p=act_probs)
         
        # Environment step
        obs, reward, done, _, info = env.step(act)
        total_steps += 1
        current_timestep += 1
        
        # Check for done and reset appropriately
        if done:
            # Update tokens and embeddings
            embeddings = model.to_tokens(
                states=states,
                actions=actions,
                rewards=rewards,
                timesteps=time,
            )
            # Reset environment
            obs, _ = env.reset()
            current_timestep = 0
            current_episode += 1
            # Update pbar
            pbar.update(1)
            ad_score = sum(ep_rewards[-n_prev_episodes:]) / len(ep_rewards[-n_prev_episodes:])
            if high_score < ad_score:
                high_score = ad_score
            pbar.set_description(f"EVAL  - Random walk score: {random_score:.4f}, Optimal score: {optimal_score:.4f}, AD high score: {high_score:.4f}, AD final score: {ad_score:.4f}")
            ep_rewards.append(0)

        if not (current_episode == n_episodes):
            # Update buffers
            ep_rewards[-1] = ep_rewards[-1] + reward
            if total_steps >= max_len:
                # Token deletion, left shift all the elements
                state_buffer = np.roll(state_buffer, -1, axis=1)
                action_buffer = np.roll(action_buffer, -1, axis=1)
                reward_buffer = np.roll(reward_buffer, -1, axis=1)
                time_buffer = np.roll(time_buffer, -1, axis=1)
                
            idx = min(max_len - 1, total_steps)
            state_buffer[:, idx] = obs
            action_buffer[:, idx - 1, 0] = act
            reward_buffer[:, idx - 1, 0] = reward
            time_buffer[:, idx, 0] = current_timestep       

    env.close()
    pbar.close()
    
    if track:
        # log statistics at batch number but prefix with eval
                wandb.log(
                    {
                        f"eval/ad_score": ad_score
                        
                    },
                    #step=batch_number,
                )
                wandb.log(
                    {
                        f"eval/high_score":
                            high_score
                        
                    },
                    #step=batch_number,
                )

    if return_cache:
        _, cache = model.transformer.run_with_cache(embeddings)
        return state_buffer, action_buffer, reward_buffer, embeddings, cache

    return random_score, optimal_score, ep_rewards[:-1]
