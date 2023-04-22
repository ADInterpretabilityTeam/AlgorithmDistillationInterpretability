import os
from argparse import Namespace

import gymnasium.vector
import numpy as np
import pytest
import torch as t
import torch.nn as nn
from einops import rearrange
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm

import wandb
from src.config import EnvironmentConfig
from src.models.trajectory_transformer import (
    AlgorithmDistillationTransformer,
    TrajectoryTransformer,
)

from .offline_dataset import TrajectoryDataset
from .utils import get_max_len_from_model_type


def train(
    model: TrajectoryTransformer,
    trajectory_data_set: TrajectoryDataset,
    env,
    make_env,
    batch_size=128,
    lr=0.0001,
    weight_decay=0.0,
    device="cpu",
    track=False,
    train_epochs=100,
    test_epochs=10,
    test_frequency=10,
    eval_frequency=10,
    eval_episodes=10,
    eval_max_time_steps=100,
    eval_num_envs=8,
):
    loss_fn = nn.CrossEntropyLoss()
    model = model.to(device)
    optimizer = t.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    train_dataset, test_dataset = random_split(
        trajectory_data_set, [0.90, 0.10]
    )

    # Create the train DataLoader
    train_sampler = WeightedRandomSampler(
        weights=trajectory_data_set.sampling_probabilities[
            train_dataset.indices
        ],
        num_samples=len(train_dataset),
        replacement=True,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler
    )

    # Create the test DataLoader
    test_sampler = WeightedRandomSampler(
        weights=trajectory_data_set.sampling_probabilities[
            test_dataset.indices
        ],
        num_samples=len(test_dataset),
        replacement=True,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, sampler=test_sampler
    )

    train_batches_per_epoch = len(train_dataloader)
    pbar = tqdm(range(train_epochs))
    for epoch in pbar:
        for batch, (s, a, r, d, rtg, ti, m) in enumerate(train_dataloader):
            total_batches = epoch * train_batches_per_epoch + batch

            model.train()

            if model.transformer_config.time_embedding_type == "linear":
                ti = ti.to(t.float32)

            a[a == -10] = env.action_space.n  # dummy action for padding

            optimizer.zero_grad()

            if isinstance(model, AlgorithmDistillationTransformer):
                action = a[:, :-1].unsqueeze(-1) if a.shape[1] > 1 else None #TODO understand and maybe change this
                _, action_preds, _ = model.forward(
                    states=s,
                    # remove last action
                    actions=action,
                    rewards=r[:, :-1],  #TODO figure this out, remove last reward?  
                    timesteps=ti.unsqueeze(-1),
                )

            action_preds = rearrange(action_preds, "b t a -> (b t) a")
            a_exp = rearrange(a, "b t -> (b t)").to(t.int64)

            # ignore dummy action
            loss = loss_fn(
                action_preds[a_exp != env.action_space.n],
                a_exp[a_exp != env.action_space.n],
            )

            loss.backward()
            optimizer.step()

            pbar.set_description(f"Training DT: {loss.item():.4f}")

            if track:
                wandb.log({"train/loss": loss.item()}, step=total_batches)
                tokens_seen = (
                    (total_batches + 1)
                    * batch_size
                    * (model.transformer_config.n_ctx // 3)
                )
                wandb.log(
                    {"metrics/tokens_seen": tokens_seen}, step=total_batches
                )

        # # at test frequency
        if epoch % test_frequency == 0:
            test(
                model=model,
                dataloader=test_dataloader,
                env=env,
                epochs=test_epochs,
                track=track,
                batch_number=total_batches,
            )

        eval_env_config = EnvironmentConfig(
            env_id=env.spec.id,
            capture_video=True,
            max_steps=min(
                model.environment_config.max_steps, eval_max_time_steps
            ),
            fully_observed=False,
            one_hot_obs=(trajectory_data_set.observation_type == "one_hot"),
            view_size=env.observation_space["image"].shape[0]
            if "image" in list(env.observation_space.keys())
            else 7,
        )

        eval_env_func = make_env(
            config=eval_env_config,
            seed=batch,
            idx=0,
            run_name=f"dt_eval_videos_{batch}",
        )

        if epoch % eval_frequency == 0:
                evaluate_dt_agent(
                    env_id=env.spec.id,
                    model=model,
                    env_func=eval_env_func,
                    trajectories=eval_episodes,
                    track=track,
                    batch_number=total_batches,
                    device=device,
                    num_envs=eval_num_envs,
                )

    return model


@pytest.mark.skip(reason="This is not a test")
def test(
    model: TrajectoryTransformer,
    dataloader: DataLoader,
    env,
    epochs=10,
    track=False,
    batch_number=0,
):
    model.eval()

    loss_fn = nn.CrossEntropyLoss()

    loss = 0
    n_correct = 0
    n_actions = 0

    pbar = tqdm(range(epochs))
    test_batches_per_epoch = len(dataloader)

    for epoch in pbar:
        for batch, (s, a, r, d, reward, ti, m) in enumerate(dataloader):
            if model.transformer_config.time_embedding_type == "linear":
                ti = ti.to(t.float32)

            a[a == -10] = env.action_space.n
            
            if isinstance(model, AlgorithmDistillationTransformer):#TODO change to AD transformer and do something on else maybe?
                _, action_preds, _ = model.forward(
                    states=s,
                    actions=a[:, :-1].unsqueeze(-1)
                    if a.shape[1] > 1
                    else None,
                    rewards=reward[:, :-2],
                    timesteps=ti.unsqueeze(-1),
                )
          

            action_preds = rearrange(action_preds, "b t a -> (b t) a")
            a_exp = rearrange(a, "b t -> (b t)").to(t.int64)

            a_hat = t.argmax(action_preds, dim=-1)
            a_exp = rearrange(a, "b t -> (b t)").to(t.int64)

            action_preds = action_preds[a_exp != env.action_space.n]
            a_hat = a_hat[a_exp != env.action_space.n]
            a_exp = a_exp[a_exp != env.action_space.n]

            n_actions += a_exp.shape[0]
            n_correct += (a_hat == a_exp).sum()
            loss += loss_fn(action_preds, a_exp)

            accuracy = n_correct.item() / n_actions
            pbar.set_description(f"Testing DT: Accuracy so far {accuracy:.4f}")

    mean_loss = loss.item() / epochs * test_batches_per_epoch

    if track:
        wandb.log({"test/loss": mean_loss}, step=batch_number)
        wandb.log({"test/accuracy": accuracy}, step=batch_number)

    return mean_loss, accuracy


def evaluate_dt_agent(
    env_id: str,
    model: TrajectoryTransformer,
    env_func,
    trajectories=300,
    track=False,
    batch_number=0,
    use_tqdm=True,
    device="cpu",
    num_envs=8,
):
    model.eval()

    env = gymnasium.vector.SyncVectorEnv([env_func for _ in range(num_envs)])
    video_path = os.path.join("videos", env.envs[0].run_name)

    if not hasattr(model, "transformer_config"):
        model.transformer_config = Namespace(
            n_ctx=model.n_ctx,
            time_embedding_type=model.time_embedding_type,
        )
 
    max_len = get_max_len_from_model_type(#TODO change this
        model_type="algorithm_distillation"
        if isinstance(model, AlgorithmDistillationTransformer)
        else "clone_transformer",
        n_ctx=model.transformer_config.n_ctx,
    )

    traj_lengths = []
    rewards = []
    n_terminated = 0
    n_truncated = 0
    reward_total = 0
    n_positive = 0

    if not os.path.exists(video_path):
        os.makedirs(video_path)

    videos = [i for i in os.listdir(video_path) if i.endswith(".mp4")]
    for video in videos:
        os.remove(os.path.join(video_path, video))
    videos = [i for i in os.listdir(video_path) if i.endswith(".mp4")]

    if use_tqdm:
        pbar = tqdm(range(trajectories), desc="Evaluating DT")
        pbar_it = iter(pbar)
    else:
        pbar = range(trajectories)

    # each env will get its own seed by incrementing on the given seed
    obs, _ = env.reset(seed=0)
    obs = t.tensor(obs["image"]).unsqueeze(1)
    #rewards = rearrange(t.ones(num_envs, dtype=t.int) * initial_rtg, "e -> e 1 1") #TODO theres no initial reward in AD, figure out what to do about that
   
    a = rearrange(t.zeros(num_envs, dtype=t.int), "e -> e 1 1")
    timesteps = rearrange(t.zeros(num_envs, dtype=t.int), "e -> e 1 1")

    obs = obs.to(device)
    rewards = rewards.to(device)
    a = a.to(device)
    timesteps = timesteps.to(device)

    if model.transformer_config.time_embedding_type == "linear":
        timesteps = timesteps.to(t.float32)

    # get first action

    if isinstance(model, AlgorithmDistillationTransformer):
        state_preds, action_preds, reward_preds = model.forward(
            states=obs, actions=None, rewards=rewards, timesteps=timesteps
        )
        
    new_action = t.argmax(action_preds, dim=-1).squeeze(-1)
    new_obs, new_reward, terminated, truncated, info = env.step(new_action)

    current_trajectory_length = t.ones(num_envs, dtype=t.int)
    while n_terminated + n_truncated < trajectories:
        # concat init obs to new obs
        obs = t.cat(
            [obs, t.tensor(new_obs["image"]).unsqueeze(1).to(device)], dim=1
        )

        # add new reward
        rewards.append(new_reward)

        # add new timesteps
        timesteps = t.cat(
            [
                timesteps,
                rearrange(current_trajectory_length.to(device), "e -> e 1 1"),
            ],
            dim=1,
        )

        if model.transformer_config.time_embedding_type == "linear":
            timesteps = timesteps.to(t.float32)

        a = t.cat([a, rearrange(new_action, "e -> e 1 1")], dim=1)

        # truncations:
        obs = obs[:, -max_len:] if obs.shape[1] > max_len else obs
        actions = (
            a[:, -(obs.shape[1] - 1) :]
            if (a.shape[1] > 1 and max_len > 1)
            else None
        )
        timesteps = (
            timesteps[:, -max_len:]
            if timesteps.shape[1] > max_len
            else timesteps
        )
        rewards = rewards[:, -max_len:] if rewards.shape[1] > max_len else rewards

        if isinstance(model, AlgorithmDistillationTransformer):#TODO change this to Algorithm distilation model 
            state_preds, action_preds, reward_preds = model.forward(
                states=obs, actions=actions, rewards=rewards, timesteps=timesteps
            )

        new_action = t.argmax(action_preds, dim=-1).squeeze(-1)
        if new_action.dim() > 1:
            new_action = new_action[:, -1]
        # convert to numpy
        new_obs, new_reward, terminated, truncated, info = env.step(new_action)
        # print(f"took action  {action} at timestep {i} for reward {new_reward}")

        n_positive = n_positive + sum(new_reward > 0)
        reward_total += sum(new_reward)
        n_terminated += sum(terminated)
        n_truncated += sum(truncated)

        if use_tqdm:
            pbar.set_description(
                f"Evaluating DT: Finished running {n_terminated + n_truncated} episodes."
                f"Current episodes are at timestep {current_trajectory_length.tolist()} for reward {new_reward}"
            )

        dones = np.logical_or(terminated, truncated)
        current_trajectory_length += np.invert(dones)
        traj_lengths.extend(current_trajectory_length[dones].tolist())
        rewards.extend(new_reward[dones])
        current_trajectory_length[dones] = 0

        if np.any(dones):
            if use_tqdm:
                [next(pbar_it, None) for _ in range(sum(dones))]

            current_videos = [
                i for i in os.listdir(video_path) if i.endswith(".mp4")
            ]
            if track and (
                len(current_videos) > len(videos)
            ):  # we have a new video
                new_videos = [i for i in current_videos if i not in videos]
                for new_video in new_videos:
                    path_to_video = os.path.join(video_path, new_video)
                    wandb.log(
                        {
                            f"media/video/": wandb.Video(
                                path_to_video,
                                fps=4,
                                format="mp4",
                                caption=f"{env_id}, after {n_terminated + n_truncated} episodes, reward {new_reward}",
                            )
                        },
                        step=batch_number,
                    )
            videos = current_videos  # update videos

    collected_trajectories = n_terminated + n_truncated

    statistics = {
        "prop_completed": n_terminated / collected_trajectories,
        "prop_truncated": n_truncated / collected_trajectories,
        "mean_reward": reward_total / collected_trajectories,
        "prop_positive_reward": n_positive / collected_trajectories,
        "mean_traj_length": sum(traj_lengths) / collected_trajectories,
        "traj_lengths": traj_lengths,
        "rewards": rewards,
    }

    env.close()
    if track:
        # log statistics at batch number but prefix with eval
        for key, value in statistics.items():
            if key == "traj_lengths":
                wandb.log(
                    {
                        f"eval/traj_lengths": wandb.Histogram(
                            value
                        )
                    },
                    step=batch_number,
                )
            elif key == "rewards":
                wandb.log(
                    {
                        f"eval/rewards": wandb.Histogram(
                            value
                        )
                    },
                    step=batch_number,
                )
            wandb.log(
                {f"eval/" + key: value}, step=batch_number
            )

    return statistics
