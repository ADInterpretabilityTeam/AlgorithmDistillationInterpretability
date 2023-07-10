import math

import json
import gymnasium as gym
import minigrid
import streamlit as st
import torch as t

from src.config import *
from src.models.trajectory_transformer import (
    DecisionTransformer,
    CloneTransformer,
)
import torch.nn.functional as F
from src.sar_transformer.utils import (
    load_algorithm_distillation_transformer,
    load_concat_transformer,
    get_max_len_from_model_type,
)
from src.environments.environments import make_env
from src.utils import pad_tensor
from src.generation import *
from src.utils import create_environment_from_id


@st.cache(allow_output_mutation=True)
def get_env_and_dt(model_path,seed=None):
    # we need to one if the env was one hot encoded. Some tech debt here.
    state_dict = t.load(model_path)
    env_config = EnvironmentConfig(**json.loads(state_dict["environment_config"]))
    offline_config = OfflineTrainConfig(**json.loads(state_dict["offline_config"]))
    env = env_config.env

    env.set_seed(selected_seed=seed)
    env.generate()

    if offline_config.model_type == "algorithm_distillation":
        dt = load_algorithm_distillation_transformer(model_path, env)
    elif offline_config.model_type == "concat_transformer":
        dt = load_concat_transformer(model_path, env)

    dt = dt.to(
        dt.transformer_config.device
    )  # TODO make this unecesary by moving into the load
    if not hasattr(dt, "n_ctx"):
        dt.n_ctx = dt.transformer_config.n_ctx
    if not hasattr(dt, "time_embedding_type"):
        dt.time_embedding_type = dt.transformer_config.time_embedding_type
    return env, dt


def get_action_preds(dt,use_modified=None):
    # so we can ignore older models when making updates

    max_len = get_max_len_from_model_type(
        dt.model_type,
        dt.transformer_config.n_ctx,
    )
    if use_modified and "modified_a" in st.session_state:
        timesteps = st.session_state.timesteps[:, -max_len:]
        obs_original =st.session_state.obs
        obs =  torch.cat((st.session_state.modified_obs, obs_original[:,:obs_original.shape[1] - st.session_state.modified_obs.shape[1]]), dim=1)
        action_original =st.session_state.a
        actions =  torch.cat((st.session_state.modified_a, action_original[:,:action_original.shape[1] -  st.session_state.modified_a.shape[1]]), dim=1)
        reward_original =st.session_state.reward
        reward =  torch.cat((st.session_state.modified_reward, reward_original[:,:reward_original.shape[1] - st.session_state.modified_reward.shape[1]]), dim=1)
    else:
        timesteps = st.session_state.timesteps[:, -max_len:]
        obs = st.session_state.obs
        actions = st.session_state.a
        reward = st.session_state.reward

    # truncations:
    obs = obs[:, -max_len:] if obs.shape[1] > max_len else obs
    if actions is not None:
        actions = (
            actions[:, -(obs.shape[1] - 1) :]
            if (actions.shape[1] > 1 and max_len > 1)
            else None #torch.empty(0)
        )
        reward = (
        reward[:, -(obs.shape[1] - 1) :]
        if (reward.shape[1] > 1 and max_len > 1)
        else None # torch.empty(0)
    )
    timesteps = timesteps[:, -max_len:] if timesteps.shape[1] > max_len else timesteps
  
    obs = obs.to(dt.transformer_config.device)
    if reward is not None:
        reward = reward.to(dt.transformer_config.device)
    if actions is not None:
        actions = actions.to(dt.transformer_config.device)   
    
    timesteps = timesteps.to(dt.transformer_config.device)

    if dt.time_embedding_type == "linear":
        timesteps = timesteps.to(dtype=t.float32)
    else:
        timesteps = timesteps.to(dtype=t.long)
    tokens = dt.to_tokens(obs, actions, reward, timesteps)
    x, cache = dt.transformer.run_with_cache(tokens, remove_batch_dim=False)

    state_preds, action_preds, reward_preds = dt.get_logits(
        x, batch_size=1, seq_length=obs.shape[1], no_actions=actions is None
    )
    return action_preds, x, cache, tokens


def respond_to_action(env, action):
    new_obs, reward, done, trunc, info = env.step(action)
    if done:
        reset_env(env)

    # append to session state
    st.session_state.obs = t.cat(
        [
            st.session_state.obs,
            t.tensor(new_obs).unsqueeze(0).unsqueeze(0),
        ],
        dim=1,
    )

    # store the rendered image
    st.session_state.rendered_obs = t.cat(
        [
            st.session_state.rendered_obs,
            t.from_numpy(env.render()).unsqueeze(0),
        ],
        dim=0,
    )

    if st.session_state.a is None:
        st.session_state.a = t.tensor([action]).unsqueeze(0).unsqueeze(0)
    else:
        st.session_state.a = t.cat(
        [st.session_state.a, t.tensor([action]).unsqueeze(0).unsqueeze(0)],
        dim=1,
    )

    if st.session_state.reward is None: 
        st.session_state.reward = t.tensor([reward]).unsqueeze(0).unsqueeze(0)
    else:
        st.session_state.reward = t.cat(
        [
            st.session_state.reward,
            t.tensor([reward]).unsqueeze(0).unsqueeze(0),
        ],
        dim=1,)


    if not done:
        time = st.session_state.timesteps[-1][-1] + 1
        st.session_state.timesteps = t.cat(
            [
                st.session_state.timesteps,
                time.clone().detach().unsqueeze(0).unsqueeze(0),
            ],
            dim=1,
        )


def reset_env(env):
    env.reset()  # TODO add some clearer way to indicate a episode ended
    st.session_state.n_episode = st.session_state.n_episode + 1
    time = (
        st.session_state.timesteps[-1][-1] * 0
    )  # TODO do this i n a less hacky way, -1 cause timesteps gets added +1
    st.session_state.timesteps = t.cat(
        [
            st.session_state.timesteps,
            time.clone().detach().unsqueeze(0).unsqueeze(0),
        ],
        dim=1,
    )


def get_action_from_user(env,dt):
    # create a series of buttons for each action
    num_actions = int(env.action_space.n)
    button_labels = [f"Action {i}" for i in range(num_actions)]
    # Create a list to store the button objects
    buttons = []
    # Create the buttons and add them to the list
    button_columns = st.columns(num_actions)
    for i in range(num_actions):
        with button_columns[i]:
            button = st.button(button_labels[i], key=f"button_{i}")
            buttons.append(button)
    # Check if any button is pressed and take the corresponding action
    for i, button in enumerate(buttons):
        if button:
            action = i
            respond_to_action(env, action)
            break
    sample_button = st.button("Sample", key=f"sample_button")
    if(sample_button):
        action_preds, x, cache, tokens=get_action_preds(dt)
        action_probabilities=F.softmax(action_preds.cpu().detach()[0][-1],dtype=t.double,dim=0)
        action= np.random.choice(len(action_probabilities), p=action_probabilities)
        respond_to_action(env,action)
        

    
