import time

import streamlit as st
import plotly.express as px
from src.visualization import visualize_learning_history
from src.streamlit_app.causal_analysis_components import show_ablation,show_ablation_all
from src.streamlit_app.components import (
    hyperpar_side_bar,
    utils_side_bar,
    record_keypresses,
    render_game_screen,
    render_trajectory_details,
    reset_button,
    reset_env_dt,
    modify_buffer,
)
from src.streamlit_app.content import (
    analysis_help,
    help_page,

)
from src.streamlit_app.dynamic_analysis_components import (
    render_observation_view,
    show_attention_pattern,
    show_residual_stream_contributions_single,
)
from src.streamlit_app.setup import initialize_playground
from src.streamlit_app.static_analysis_components import (
    show_ov_circuit,
    show_qk_circuit,
    show_rtg_embeddings,
    show_time_embeddings,
)
from src.streamlit_app.visualizations import action_string_to_id

from src.streamlit_app.model_index import model_index

start = time.time()

st.set_page_config(
    page_title="Algorithm Distillation Interpretability",
    page_icon="assets/logofiles/Logo_black.ico",
)


with st.sidebar:
    
    st.image(
        "assets/logofiles/Logo_transparent.png", use_column_width="always"
    )
    st.title("Decision Transformer Interpretability")

    model_directory = "models"

    with st.form("model_selector"):
        selected_model_path = st.selectbox(
            label="Select Model",
            options=model_index.keys(),
            format_func=lambda x: model_index[x],
            key="model_selector",
        )
        submitted = st.form_submit_button("Load Model")
        if submitted:
            reset_env_dt()

hyperpar_side_bar()
utils_side_bar()




# st.session_state.max_len = 1
env, dt = initialize_playground(selected_model_path)
action_preds,x, cache,cache_modified, tokens = render_game_screen(dt, env)

action_options = [f"Action {i}" for i in range(1, env.action_space.n + 1)]#TODO maybe needs a done option
action_string_to_id = {element: index for index, element in enumerate(action_options)}
action_id_to_string = {v: k for k, v in action_string_to_id.items()}
record_keypresses()


with st.sidebar:
    st.subheader("Directional Analysis")
    comparing = st.checkbox("comparing directions", value=True)
    if comparing:
        positive_action_direction = st.selectbox(
            "Positive Action Direction",
            action_options,
            index=0,
        )
        negative_action_direction = st.selectbox(
            "Negative Action Direction",
            action_options,
            index=1,
        )
        positive_action_direction = action_string_to_id[
            positive_action_direction
        ]
        negative_action_direction = action_string_to_id[
            negative_action_direction
        ]

        logit_dir = (
            dt.action_predictor.weight[positive_action_direction]
            - dt.action_predictor.weight[negative_action_direction]
        )
    else:
        st.warning("Single Logit Analysis may be misleading.")
        selected_action_direction = st.selectbox(
            "Selected Action Direction",
            action_options,
            index=2,
        )
        selected_action_direction = action_string_to_id[
            selected_action_direction
        ]
        logit_dir = dt.action_predictor.weight[selected_action_direction]
    
    select_utils = st.multiselect("Select Utils", ["Modify Buffer"])

    st.subheader("Analysis Selection")
    static_analyses = st.multiselect(
        "Select Static Analyses",
        ["Reward Embeddings", "Time Embeddings", "OV Circuit", "QK Circuit"],
    )
    dynamic_analyses = st.multiselect(
        "Select Dynamic Analyses",
        [
            "Residual Stream Contributions",
            "Attention Pattern",
            "Observation View",
        ],
    )
    causal_analyses = st.multiselect("Select Causal Analyses", ["Ablation","Ablation Effects"])

analyses = dynamic_analyses + static_analyses + causal_analyses
utils= select_utils

with st.sidebar:
    render_trajectory_details()
    reset_button()

if len(analyses) == 0:
    st.warning("Please select at least one analysis.")
#Utils
if "Modify Buffer" in utils:
    modify_buffer(dt)
#Analyses
if "reward Embeddings" in analyses:
    show_rtg_embeddings(dt, logit_dir)
if "Time Embeddings" in analyses:
    show_time_embeddings(dt, logit_dir)
if "QK Circuit" in analyses:
    show_qk_circuit(dt)
if "OV Circuit" in analyses:
    show_ov_circuit(dt)

if "Ablation" in analyses:
    show_ablation(dt, logit_dir=logit_dir, original_cache=cache)
if "Ablation Effects" in analyses:
    show_ablation_all(dt,positive_action_direction,negative_action_direction,action_preds)



if "Residual Stream Contributions" in analyses:
    show_residual_stream_contributions_single(dt, cache, logit_dir=logit_dir)
if "Attention Pattern" in analyses:
    show_attention_pattern(dt, cache,cache_modified)
if "Observation View" in analyses:
    render_observation_view(dt, tokens, logit_dir)




st.markdown("""---""")

with st.expander("Show history"):
    rendered_obss = st.session_state.rendered_obs
    trajectory_length = rendered_obss.shape[0]

    if trajectory_length > 1:
        historic_actions = st.session_state.a[0, -trajectory_length:].flatten()
        historic_rewards=st.session_state.reward[0, -trajectory_length:].flatten()
        right_adjustment = 0
        if(trajectory_length > 2):
            state_number = st.slider(
                "Step Number",
                min_value=0,
                max_value=trajectory_length - 2,
                step=1,
                format="Step Number: %d",
            )
        else:
            state_number=0

        i = state_number
        action_name_func = (
            lambda a: "None" if a == 7 else action_id_to_string[a]
        )
        visualize_learning_history
        st.write(f"A{i}:", action_name_func(historic_actions[i].item()))
        if(state_number< trajectory_length -2):
            st.write(f"A{i+1}:", action_name_func(historic_actions[i + 1].item()))
        st.write(f"R{i}:", historic_rewards[i].item())
        
        st.plotly_chart(px.imshow(rendered_obss[i, :, :, :]))
    else:
        st.warning("No history to show")

st.markdown("""---""")

with st.expander("Show Dataset"):
    filename=st.text_input("File name:")

    if st.button("Visualize Data"):
        st.session_state.dataset_frames = visualize_learning_history(filename)#"histories/train_dark_room/5.npz")
    if "dataset_frames" in st.session_state:
        trajectory_length = len(st.session_state.dataset_frames)

        if trajectory_length > 1:
            
            state_number = st.slider("Step Number",
                    min_value=0,
                    max_value=trajectory_length,
                    step=1,
                    format="Step Number: %d")


            i = state_number

            st.plotly_chart(px.imshow(st.session_state.dataset_frames[i]))
    else:
        st.warning("No history to show")

st.markdown("""---""")
with st.expander("Show Rules"):
    #Rules are of the form (old_state, new_state, action, probability, value, flag)
    st.write(f'Old State: {env.reward_rules[0][0]}')
    st.write(f'New State: {env.reward_rules[0][1]}')
    st.write(f'Action: {env.reward_rules[0][2]}')
    st.write(f'Probability: {env.reward_rules[0][3]}')
    st.write(f'Value:{env.reward_rules[0][4]}')
    st.write(f'Flag:{env.reward_rules[0][5]}')

st.session_state.env = env
st.session_state.dt = dt

with st.sidebar:
    end = time.time()
    st.write(f"Time taken: {end - start}")

record_keypresses()

help_page()
analysis_help()
