import torch
import streamlit as st
import streamlit.components.v1 as components
import uuid

from .environment import get_action_preds
from .utils import read_index_html
from .visualizations import plot_action_preds, render_env


def render_game_screen(dt, env):
    columns = st.columns(2)
    with columns[0]:
        action_preds, x, cache, tokens = get_action_preds(dt)
        plot_action_preds(action_preds)
    with columns[1]:
        current_time = st.session_state.timesteps 
        
        
        st.write(f"Current Time: {int(current_time[0][-1].item())} Current episode: {st.session_state.n_episode}" )
        fig = render_env(env)
        st.pyplot(fig)

    return x, cache, tokens


def hyperpar_side_bar():
    with st.sidebar:
        st.subheader("Hyperparameters")
        seed = st.number_input(
            "Enviroment Seed",
            min_value=0.0,
            value=0.0,
            step=1.0,
        )
        st.session_state.seed = int(seed)

def render_trajectory_details():
    with st.expander("Trajectory Details"):
        # write out actions, rtgs, rewards, and timesteps
        st.write(f"max timeteps: {st.session_state.max_len}")
        st.write(f"trajectory length: {len(st.session_state.obs[0])}")
        if st.session_state.a is not None:
            st.write(f"actions: {st.session_state.a[0].squeeze(-1).tolist()}")
        if st.session_state.reward is not None:
            st.write(f"rewards: {st.session_state.reward[0].squeeze(-1).tolist()}")
        st.write(
            f"timesteps: {st.session_state.timesteps[0].squeeze(-1).tolist()}"
        )


def reset_button():
    if st.button("reset"):
        reset_env_dt()
        st.experimental_rerun()


def record_keypresses():
    components.html(
        read_index_html(),
        height=0,
        width=0,
    )


def reset_env_dt():
    if "env" in st.session_state:
        del st.session_state.env
    if "dt" in st.session_state:
        del st.session_state.dt
