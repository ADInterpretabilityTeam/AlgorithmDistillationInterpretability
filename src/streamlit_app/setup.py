import streamlit as st
import torch as t
import numpy as np

from src.sar_transformer.utils import (
    get_max_len_from_model_type,
)
from .environment import get_env_and_dt, get_action_from_user


def initialize_playground(model_path):
    if "env" not in st.session_state or "dt" not in st.session_state:
        
        env, dt = get_env_and_dt(model_path,seed=st.session_state.seed)
        max_len = get_max_len_from_model_type(
            dt.model_type, dt.transformer_config.n_ctx
        )
        
        mask = t.concat(
        (
            t.zeros((1, max_len - 1), dtype=t.bool),
            t.ones((1, 1), dtype=t.bool),
        ),
        dim=1,
    )
        
        
        n_obs = env.observation_space.shape[0]
        obs = np.zeros((1, 1, n_obs))
        initial_obs, _ = env.reset()
        obs[0, 0] = initial_obs
        obs=t.tensor(obs)
        #actions = t.zeros((1, 1, 1), dtype=t.long)
        #reward = t.zeros((1, 1, 1))
        actions=None
        reward=None
        timesteps = t.zeros((1, 1, 1))
        current_episode = 0

        rendered_obs = t.from_numpy(env.render()).unsqueeze(0)

        st.session_state.max_len = max_len
        st.session_state.obs = obs
        st.session_state.rendered_obs = rendered_obs
        st.session_state.reward = reward
        st.session_state.a = actions
        st.session_state.timesteps = timesteps
        st.session_state.n_episode=0
        st.session_state.dt = dt
        st.session_state.mask = mask

    else:
        env = st.session_state.env
        dt = st.session_state.dt
    

    if "action" in st.session_state:
        action_options = [f"Action {i}" for i in range(1, env.action_space.n + 1)]#TODO maybe needs a done option and be a function since its needed multiple times
        action_string_to_id = {element: index for index, element in enumerate(action_options)}
        action_id_to_string = {v: k for k, v in action_string_to_id.items()}
        action = st.session_state.action
        if isinstance(action, str):
            action = action_string_to_id[action]
        st.write(
            f"just took action '{action_id_to_string[st.session_state.action]}'"
        )
        del action
        del st.session_state.action
    else:
        get_action_from_user(env,dt)



    return env, dt
