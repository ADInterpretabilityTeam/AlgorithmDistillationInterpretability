
import argparse
import json
import numpy as np
import torch as t
from src.config import (
    EnvironmentConfig,
    TransformerModelConfig,
    ConfigJsonEncoder
)
from src.models.trajectory_transformer import (
    AlgorithmDistillationTransformer,
    CloneTransformer,
    ConcatTransformer
)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Algorithm distilation",
        description="Train a algorithm distillation transformer on training histories",
        epilog="The last enemy that shall be defeated is death.",
    )
    parser.add_argument("--exp_name", type=str, default="Dev")
    parser.add_argument("--env_id", type=str, required=True, choices=[
        "DarkRoom",
        "DarkKeyDoor",
        "ArmedBandit",
        "SimpleDarkRoom",
        "SimpleDarkKeyDoor"])
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--trajectory_path", type=str)
    parser.add_argument("--test_trajectory_path", type=str)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_mlp", type=int, default=1024)
    parser.add_argument("--d_head", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_episodes_per_seq", type=int, default=10)
    parser.add_argument("--layer_norm", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--test_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument(
        "--linear_time_embedding",
        
        default=False,
        action="store_true",
    )
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--track",
        
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="DecisionTransformerInterpretability",
    )
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--test_frequency", type=int, default=100)
    parser.add_argument("--eval_temp", type=int, default=1)
    parser.add_argument("--eval_frequency", type=int, default=10)
    parser.add_argument("--eval_episodes", type=int, default=1000)
    parser.add_argument("--eval_num_envs", type=int, default=8)
    parser.add_argument(
        "--initial_rtg",
        action="append",
        help="<Required> Set flag",
        required=False,
        default=[0, 1],
    )
    parser.add_argument("--prob_go_from_end", type=float, default=0.1)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument(
        "--model_type", type=str, default="algorithm_distillation", choices=[
            "algorithm_distillation",
            "concat_transformer"
        ]
    )
    parser.add_argument(
        "--convert_to_one_hot",
        
        default=False,
        action="store_true",
    )
    args = parser.parse_args()
    return args


def load_algorithm_distillation_transformer(model_path, env=None) -> AlgorithmDistillationTransformer:
    """ """

    model_info = t.load(model_path)
    state_dict = model_info["model_state_dict"]
    transformer_config = TransformerModelConfig(
        **json.loads(model_info["model_config"])
    )

    environment_config = EnvironmentConfig(
        **json.loads(model_info["environment_config"])
    )

    model = AlgorithmDistillationTransformer(
        environment_config=environment_config,
        transformer_config=transformer_config,
    )

    model.load_state_dict(state_dict)
    return model


def load_concat_transformer(model_path, env=None) -> AlgorithmDistillationTransformer:
    """ """

    model_info = t.load(model_path)
    state_dict = model_info["model_state_dict"]
    transformer_config = TransformerModelConfig(
        **json.loads(model_info["model_config"])
    )

    environment_config = EnvironmentConfig(
        **json.loads(model_info["environment_config"])
    )

    model = ConcatTransformer(
        environment_config=environment_config,
        transformer_config=transformer_config,
    )

    model.load_state_dict(state_dict)
    return model


def get_max_len_from_model_type(model_type: str, n_ctx: int):
    """
    Ihe max len in timesteps is 3 for decision transformers
    and 2 for clone transformers since decision transformers
    have 3 tokens per timestep and clone transformers have 2.

    This is a map between timestep and tokens. We start with one
    for the most recent state/action and then add another
    timestep for every 3 tokens for decision transformers and
    every 2 tokens for clone transformers.
    """
    assert model_type in ["algorithm_distillation", "concat_transformer", "clone_transformer"]
    if model_type == "algorithm_distillation":
        return 1 + n_ctx // 3
    elif model_type == "concat_transformer":
        return n_ctx
    else:
        return 1 + n_ctx // 2

    
def store_transformer_model(path, model, offline_config):
    t.save(
        {
            "model_state_dict": model.state_dict(),
            "offline_config": json.dumps(
                offline_config, cls=ConfigJsonEncoder
            ),
            "environment_config": json.dumps(
                model.environment_config, cls=ConfigJsonEncoder
            ),
            "model_config": json.dumps(
                model.transformer_config, cls=ConfigJsonEncoder
            ),
        },
        path,
    )
