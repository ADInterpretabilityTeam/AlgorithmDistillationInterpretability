import streamlit as st
from torchtyping import TensorType as TT
from transformer_lens.hook_points import HookPoint
import numpy as np

from .analysis import get_residual_decomp
from .environment import get_action_preds
from .visualizations import (
    plot_action_preds,
    plot_single_residual_stream_contributions,
    plot_single_residual_stream_contributions_comparison,
    plot_action_diferences,
)
def ablate_all_hooks(dt,ablate_to_mean,positive_dir,negative_dir,original_preds=None):
    n_layers = dt.transformer_config.n_layers
    n_heads = dt.transformer_config.n_heads
    action_preds={}
    if original_preds is not None:
        original_preds = np.exp(original_preds) / np.sum(np.exp(original_preds))
        original_diference= original_preds[positive_dir] - original_preds[negative_dir]
    else:
        original_diference= None
    
    for layer in  range(n_layers):
        for head in  range(n_heads):
            ablation_func = get_ablation_function(ablate_to_mean, head)
            dt.transformer.blocks[layer].attn.hook_z.add_hook(ablation_func)
            action_preds[f'layer{layer} Head {head}']=get_ablation_preds(dt,positive_dir,negative_dir,original_diference=original_diference)
        ablation_func = get_ablation_function(
                ablate_to_mean, layer, component="MLP"
            )
        dt.transformer.blocks[layer].hook_mlp_out.add_hook(ablation_func)
        action_preds[f'layer{layer} MLP']=get_ablation_preds(dt,positive_dir,negative_dir,original_diference=original_diference)
    return action_preds
    

def get_ablation_preds(dt,positive_dir,negative_dir,original_diference=None):
    action_preds, x, cache, tokens = get_action_preds(dt)
    action_preds=action_preds.detach().cpu().numpy()[0][-1]
    action_preds = np.exp(action_preds) / np.sum(np.exp(action_preds))
    ablated_diference=action_preds[positive_dir]-action_preds[negative_dir]
    if original_diference is not None:
        ablated_diference=ablated_diference-original_diference
    dt.transformer.reset_hooks()

    return ablated_diference

def show_ablation_all(dt,positive_dir, negative_dir, original_predictions):
    with st.expander("Ablation Experiment"):
        # make a streamlit form for choosing a component to ablate

        ablate_to_mean = st.checkbox("Ablate all to mean", value=True)
        if st.checkbox("Compare to original", value=True):
            org_preds=original_predictions.detach().cpu().numpy()[0][-1]
        else:
            org_preds= None
        action_preds=ablate_all_hooks(dt,ablate_to_mean,positive_dir,negative_dir,original_preds=org_preds)
        plot_action_diferences(action_preds)

        



def show_ablation(dt, logit_dir, original_cache):
    with st.expander("Ablate all"):
        # make a streamlit form for choosing a component to ablate
        n_layers = dt.transformer_config.n_layers
        n_heads = dt.transformer_config.n_heads

        columns = st.columns(4)
        with columns[0]:
            layer = st.selectbox("Layer", list(range(n_layers)))
        with columns[1]:
            component = st.selectbox("Component", ["MLP", "HEAD"], index=1)
        with columns[2]:
            if component == "HEAD":
                head = st.selectbox("Head", list(range(n_heads)))
        with columns[3]:
            ablate_to_mean = st.checkbox("Ablate to mean", value=True)
        

        if component == "HEAD":
            ablation_func = get_ablation_function(ablate_to_mean, head)
            dt.transformer.blocks[layer].attn.hook_z.add_hook(ablation_func)
        elif component == "MLP":
            ablation_func = get_ablation_function(
                ablate_to_mean, layer, component="MLP"
            )
            dt.transformer.blocks[layer].hook_mlp_out.add_hook(ablation_func)

        action_preds, x, cache, tokens = get_action_preds(dt)
        dt.transformer.reset_hooks()
        if st.checkbox("show action predictions"):
            plot_action_preds(action_preds)
        if st.checkbox("show counterfactual residual contributions"):
            original_residual_decomp = get_residual_decomp(
                dt, original_cache, logit_dir
            )
            ablation_residual_decomp = get_residual_decomp(
                dt, cache, logit_dir
            )
            plot_single_residual_stream_contributions_comparison(
                original_residual_decomp, ablation_residual_decomp
            )
        

    # then, render a single residual stream contribution with the ablation


def get_ablation_function(ablate_to_mean, head_to_ablate, component="HEAD"):
    def head_ablation_hook(
        value: TT["batch", "pos", "head_index", "d_head"],  # noqa: F821
        hook: HookPoint,
    ) -> TT["batch", "pos", "head_index", "d_head"]:  # noqa: F821

        if ablate_to_mean:
            value[:, :, head_to_ablate, :] = value[
                :, :, head_to_ablate, :
            ].mean(dim=2, keepdim=True)
        else:
            value[:, :, head_to_ablate, :] = 0.0
        return value

    def mlp_ablation_hook(
        value: TT["batch", "pos", "d_model"], hook: HookPoint  # noqa: F821
    ) -> TT["batch", "pos", "d_model"]:  # noqa: F821

        if ablate_to_mean:
            value[:, :, :] = value[:, :, :].mean(dim=2, keepdim=True)
        else:
            value[:, :, :] = 0  # ablate all but the last token
        return value

    if component == "HEAD":
        return head_ablation_hook
    elif component == "MLP":
        return mlp_ablation_hook
