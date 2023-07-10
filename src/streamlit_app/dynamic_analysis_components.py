import pandas as pd
import plotly.express as px
import streamlit as st
import torch as t
from einops import rearrange
from fancy_einsum import einsum
from minigrid.core.constants import IDX_TO_COLOR, IDX_TO_OBJECT

from src.sar_transformer.utils import get_max_len_from_model_type

from .analysis import get_residual_decomp
from .constants import (
    IDX_TO_ACTION,
    IDX_TO_STATE,
    three_channel_schema,
    twenty_idx_format_func,
)
from .utils import fancy_histogram, fancy_imshow
from .visualizations import (
    plot_attention_pattern_single,
    plot_attention_pattern_difference,
    plot_single_residual_stream_contributions,
)


def show_attention_pattern(dt, cache,cache_modified):
    with st.expander("Attention Pattern at at current Reward-to-Go"):
        st.latex(
            r"""
            h(x)=\left(A \otimes W_O W_V\right) \cdot x \newline
            """
        )

        st.latex(
            r"""
            A=\operatorname{softmax}\left(x^T W_Q^T W_K x\right)
            """
        )
        n_heads = dt.transformer_config.n_heads
        n_layers = dt.transformer_config.n_layers
        softmax = st.checkbox("softmax", value=True)
        heads = st.multiselect(
            "Select Heads",
            options=list(range(n_heads)),
            default=list(range(n_heads)),
            key="heads attention",
        )

        if n_layers == 1:
            plot_attention_pattern_single(
                cache, 0, softmax=softmax, specific_heads=heads
            )
            layer=0
        else:
            layer = st.slider(
                "Layer",
                min_value=0,
                max_value=n_layers - 1,
                value=0,
                step=1,
            )
            plot_attention_pattern_single(
                cache, layer, softmax=softmax, specific_heads=heads
            )
        if st.checkbox("Display_modified"):

            if st.checkbox("Display_diference"):
                plot_attention_pattern_difference(
                        cache,cache_modified, layer, softmax=softmax, specific_heads=heads
                    )
            else:
                plot_attention_pattern_single(
                        cache_modified, layer, softmax=softmax, specific_heads=heads
                    )



def show_residual_stream_contributions_single(dt, cache, logit_dir):
    with st.expander(
        "Show Residual Stream Contributions"
    ):
        residual_decomp = get_residual_decomp(dt, cache, logit_dir)

        # this plot assumes you only have a single dim
        plot_single_residual_stream_contributions(residual_decomp)

    return

def render_observation_view(dt, tokens, logit_dir):
    last_obs = st.session_state.obs[0][-1]

    last_obs_reshaped = rearrange(last_obs, "h w c -> c h w")

    height, width, n_channels = dt.environment_config.observation_space[
        "image"
    ].shape

    weights = dt.state_embedding.weight.detach().cpu()

    weights_reshaped = rearrange(
        weights, "d (c h w) -> c d h w", c=n_channels, h=height, w=width
    )

    embeddings = einsum(
        "c d h w, c h w -> c d",
        weights_reshaped,
        last_obs_reshaped.to(t.float32),
    )

    weight_projections = einsum(
        "d, c d h w -> c h w", logit_dir, weights_reshaped
    )

    activation_projection = weight_projections * last_obs_reshaped

    timesteps = st.session_state.timesteps[0][-1]
    if dt.time_embedding_type == "linear":
        timesteps = timesteps.to(t.float32)
    else:
        timesteps = timesteps.to(t.long)

    time_embedding = dt.time_embedding(timesteps)

    with st.expander("Show observation view"):
        st.subheader("Observation View")
        if n_channels == 3:

            def format_func(x):
                return three_channel_schema[x]

        else:
            format_func = twenty_idx_format_func

        selected_channels = st.multiselect(
            "Select Observation Channels",
            options=list(range(n_channels)),
            format_func=format_func,
            key="channels obs",
            default=[0, 1, 2],
        )
        n_selected_channels = len(selected_channels)

        check_columns = st.columns(4)
        with check_columns[0]:
            contributions_check = st.checkbox("Show contributions", value=True)
        with check_columns[1]:
            input_channel_check = st.checkbox(
                "Show input channels", value=True
            )
        with check_columns[2]:
            weight_proj_check = st.checkbox(
                "Show channel weight proj onto logit dir", value=True
            )
        with check_columns[3]:
            activ_proj_check = st.checkbox(
                "Show channel activation proj onto logit dir", value=True
            )

        if contributions_check:
            contributions = {
                format_func(i): (embeddings[i] @ logit_dir).item()
                for i in selected_channels
            }

            if dt.time_embedding_type == "linear":
                time_contribution = (time_embedding @ logit_dir).item()
            else:
                time_contribution = (time_embedding[0] @ logit_dir).item()

            token_contribution = (tokens[0][-1] @ logit_dir).item()

            contributions = {
                **contributions,
                "time": time_contribution,
                "token": token_contribution,
            }

            fig = px.bar(
                contributions.items(),
                x=0,
                y=1,
                labels={"0": "Channel", "1": "Contribution"},
                text=1,
            )

            # add the value to the bar
            fig.update_traces(texttemplate="%{text:.3f}", textposition="auto")
            fig.update_layout(uniformtext_minsize=8, uniformtext_mode="hide")
            fig.update_yaxes(range=[-8, 8])
            st.plotly_chart(fig, use_container_width=True)

        if input_channel_check:
            columns = st.columns(n_selected_channels)
            for i, channel in enumerate(selected_channels):
                with columns[i]:
                    st.write(format_func(channel))
                    fancy_imshow(last_obs_reshaped[channel].detach().numpy().T)

                    if n_channels == 3:
                        if i == 0:
                            st.write(IDX_TO_OBJECT)
                        elif i == 1:
                            st.write(IDX_TO_COLOR)
                        else:
                            st.write(IDX_TO_STATE)

        if weight_proj_check:
            columns = st.columns(n_selected_channels)
            for i, channel in enumerate(selected_channels):
                with columns[i]:
                    st.write(format_func(channel))
                    fancy_imshow(
                        weight_projections[channel].detach().numpy().T
                    )
                    fancy_histogram(
                        weight_projections[channel].detach().numpy().flatten()
                    )

        if activ_proj_check:
            columns = st.columns(n_selected_channels)
            for i, channel in enumerate(selected_channels):
                with columns[i]:
                    st.write(format_func(channel))
                    fancy_imshow(
                        activation_projection[channel].detach().numpy().T
                    )
                    fancy_histogram(
                        activation_projection[channel]
                        .detach()
                        .numpy()
                        .flatten()
                    )


def project_weights_onto_dir(weights, dir):
    return t.einsum(
        "d, d h w -> h w", dir, weights.reshape(128, 7, 7)
    ).detach()
