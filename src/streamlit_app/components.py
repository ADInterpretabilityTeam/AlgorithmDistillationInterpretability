import torch
import streamlit as st
import streamlit.components.v1 as components
import uuid
import numpy as np

from .environment import get_action_preds,respond_to_action
from .utils import read_index_html
from .visualizations import plot_action_preds, render_env
import torch.nn.functional as F

def render_game_screen(dt, env):
    columns = st.columns(2)
    with columns[0]:
        action_preds, x, cache, tokens = get_action_preds(dt)
        _, _, cache_modified, _ = get_action_preds(dt,use_modified=True)
        plot_action_preds(action_preds)
    with columns[1]:
        current_time = st.session_state.timesteps 
        
        
        st.write(f"Current Time: {int(current_time[0][-1].item())} Current episode: {st.session_state.n_episode}" )
        fig = render_env(env)
        st.pyplot(fig)
        

    return action_preds,x, cache,cache_modified, tokens


def hyperpar_side_bar():
    with st.sidebar:
        st.subheader("Parameters")
        seed = st.number_input(
            "Enviroment Seed",
            min_value=0.0,
            value=0.0,
            step=1.0,
        )
        st.session_state.seed = int(seed)
        
def utils_side_bar():
    with st.sidebar:
        st.subheader("Utils")
        multiple_sample_button()
        next_episode_sample_button()

def multiple_sample_button():
    sample_amount = int(st.number_input("Sample Multiple", key=f"sample_steps_amount",min_value=0,step=1))
    sample_button = st.button("Sample Multiple", key=f"sample_multi_button")
    if(sample_button):
        for i in range(sample_amount):
            action_preds, x, cache, tokens=get_action_preds(st.session_state.dt)
            action_probabilities=F.softmax(action_preds.cpu().detach()[0][-1],dtype=torch.double,dim=0)
            action= np.random.choice(len(action_probabilities), p=action_probabilities)
            respond_to_action(st.session_state.env,action)

def next_episode_sample_button():
    sample_button = st.button("Sample Untill Next Episode", key=f"next_episode_sample_button")
    if(sample_button):
        sample_amount= int(st.session_state.env.max_steps - st.session_state.timesteps[0][-1].item())
        for i in range(sample_amount):
                action_preds, x, cache, tokens=get_action_preds(st.session_state.dt)
                action_probabilities=F.softmax(action_preds.cpu().detach()[0][-1],dtype=torch.double,dim=0)
                action= np.random.choice(len(action_probabilities), p=action_probabilities)
                respond_to_action(st.session_state.env,action)


    

def modify_buffer(dt):
    with st.expander("Modify Buffer"):#TODO deal with longer than context len buffers
        if (st.session_state.timesteps.shape[1]>1):
            current_state =np.argmax(st.session_state.obs[0].squeeze(-1).tolist(),axis=1).tolist() 
            state_modified_string=st.text_area("Modified States", value=current_state) 
            if "modified_obs" in st.session_state:
                st.text(np.argmax(st.session_state.modified_obs[0].squeeze(-1).tolist(),axis=1).tolist())
            state_modified = [int(float(num)) for num in state_modified_string.strip('[]').split(',')]
            current_actions = st.session_state.a[0].squeeze(-1).tolist()
            actions_modified_string=st.text_area("Modified Actions", value=current_actions)
            if "modified_a" in st.session_state:
                st.text(st.session_state.modified_a[0].squeeze(-1).tolist())

            actions_modified = [int(float(num)) for num in actions_modified_string.strip('[]').split(',')]
            current_reward = st.session_state.reward[0].squeeze(-1).tolist()
            reward_modified_string=st.text_area("Modified Reward", value=current_reward)  
            if "modified_reward" in st.session_state:
                st.text(st.session_state.modified_reward[0].squeeze(-1).tolist())
            reward_modified = [int(float(num)) for num in reward_modified_string.strip('[]').split(',')]
            if st.button("Modify"): 
                st.session_state.modified_obs=torch.nn.functional.one_hot(torch.tensor(state_modified),dt.environment_config.n_states).unsqueeze(0)#TODO check if env uses onehot obs instead
                st.session_state.modified_a= torch.tensor(actions_modified).unsqueeze(0).unsqueeze(2)
                st.session_state.modified_reward= torch.tensor(reward_modified).unsqueeze(0).unsqueeze(2)
        else:
            st.warning("Cant modify whithout buffer please take a step")#TODO should maybe let you modify obs 

        

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
            f"obs: {np.argmax(st.session_state.obs[0].squeeze(-1).tolist(),axis=1)}"
        )
        st.write(
            f"timesteps: {st.session_state.timesteps[0].squeeze(-1).tolist()}"
        )
        


def reset_button():
    if st.button("Reset"):
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
