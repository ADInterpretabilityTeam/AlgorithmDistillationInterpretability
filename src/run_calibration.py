from environments import make_env
from decision_transformer.model import DecisionTransformer
from decision_transformer.calibration import calibration_statistics, plot_calibration_statistics
import argparse 
import warnings
import torch as t
import numpy as np
import os

# import a  base python logger
import logging
logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(
        prog = "Get Calibration of Decision Transformer",
        description="Assess the RTG calibration of a decision transformer")

    parser.add_argument("--env_id", type=str, default="MiniGrid-Dynamic-Obstacles-8x8-v0", help="Environment ID")
    parser.add_argument("--model_path", type=str, default="models/dt.pt", help="Path to model")
    parser.add_argument("--n_trajectories", type=int, default=100, help="Number of trajectories to evaluate")
    parser.add_argument("--initial_rtg_min", type=float, default=-1, help="Minimum initial RTG")
    parser.add_argument("--initial_rtg_max", type=float, default=1, help="Maximum initial RTG")
    parser.add_argument("--initial_rtg_step", type=float, default=0.1, help="Step size for initial RTG")
    args = parser.parse_args()

    logger.info(f"Loading model from {args.model_path}")
    logger.info(f"Using environment {args.env_id}")
    print(args)
    env = make_env(args.env_id, seed = 1, idx = 0, capture_video=False, run_name = "dev", fully_observed=False, max_steps=300)
    env = env()

    n_ctx = 3
    max_len = n_ctx // 3
    dt = DecisionTransformer(
        env = env, 
        d_model = 128,
        n_heads = 4,
        d_mlp = 256,
        n_layers = 1,
        n_ctx=n_ctx,
        layer_norm=True,
        state_embedding_type="grid", # hard-coded for now to minigrid.
        max_timestep=300
    ) 

    dt.load_state_dict(t.load(args.model_path))

    warnings.filterwarnings("ignore", category=UserWarning)
    statistics = calibration_statistics(
        dt, 
        args.env_id, 
        make_env,
        initial_rtg_range=np.linspace(args.initial_rtg_min, args.initial_rtg_max, int((args.initial_rtg_max - args.initial_rtg_min) / args.initial_rtg_step)),
        trajectories=args.n_trajectories
    )
    
    fig = plot_calibration_statistics(statistics)

    if not os.path.exists("figures"):
        os.mkdir("figures")

    # format the output path according to the input path.
    args.output_path = f"figures/{args.model_path.split('/')[-1].split('.')[0]}_calibration.png"
    fig.write_image(args.output_path)