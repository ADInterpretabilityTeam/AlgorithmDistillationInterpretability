import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from tqdm import tqdm

import wandb
from src.config import EnvironmentConfig
from src.models.trajectory_transformer import (
    AlgorithmDistillationTransformer,
    TrajectoryTransformer,
)
from .utils import store_transformer_model
from .dataset import create_history_dataloader
from .eval import evaluate_ad_agent


def train(
    model,
    train_dataloader,
    test_dataloader,
    env_config,
    lr=3e-4,
    clip=1.0,
    device="cuda",
    track=False,
    train_epochs=500,
    test_frequency=1,
    eval_frequency=5,
    num_evals=8,
    eval_length=1_000,
    eval_temp=1.,
    run_name=None,
    checkpoint_path=None,
    offline_config=None
):
    # Create loss function and model optimizer
    loss_fn = nn.CrossEntropyLoss()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_epochs, 2e-6)
    train_batches_per_epoch = len(train_dataloader)

    # Start training
    for epoch in range(train_epochs):

        pbar = tqdm(total=train_batches_per_epoch)
        model.train()
        losses = []
        accs = []

        for batch, (s, a, r, ti) in enumerate(train_dataloader):
            total_batches = epoch * train_batches_per_epoch + batch

            # Random augmentations
            state_perm = torch.randperm(s.shape[-1])
            s = s.index_select(-1, state_perm)
            
            if model.transformer_config.time_embedding_type == "linear":
                ti = ti.to(torch.float32)

            optimizer.zero_grad()

            states = s.to(device)
            action_labels = a.to(device)
            actions = action_labels[:, :-1]
            rewards = r.to(device)[:, :-1]
            time = ti.to(device)

            _, action_preds, _ = model.forward(
                states=states,
                actions=actions,
                rewards=rewards,
                timesteps=time,
            )

            action_preds = rearrange(action_preds, "b t a -> (b t) a")
            a_exp = rearrange(action_labels, "b t i -> (b t i)")
            
            acc = torch.mean((action_preds.argmax(-1) == a_exp).float()).item() * 100
            accs.append(acc)
            loss = loss_fn(action_preds, a_exp.to(torch.int64))
            losses.append(loss.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            pbar.set_description(f"TRAIN - Epoch: {epoch+1}, Loss: {sum(losses)/len(losses):.4f}, Acc: {sum(accs)/len(accs):.4f}%")
            pbar.update(1)

        if track:
            batch_size = s.shape[0]
            wandb.log({"train/loss": sum(losses)/len(losses)}, step=total_batches)
            wandb.log({"train/acc": sum(accs)/len(accs)}, step=total_batches)
            tokens_seen = (
                (total_batches + 1)
                * batch_size
                * (model.transformer_config.n_ctx // 3)
            )
            wandb.log({"metrics/tokens_seen": tokens_seen}, step=total_batches)

        scheduler.step()
        pbar.close()
        model.eval()

        test(
            model=model,
            test_dataloader=test_dataloader,
            env_config=env_config,
            current_epoch=epoch+1,
            device=device,
            track=track,
            logging_step=total_batches
        )

        if (epoch + 1) % eval_frequency == 0:
            if track:
                    artifact = wandb.Artifact(run_name, type="model")
                    current_checkpoint_path=f"{checkpoint_path}/checkpoint_epoch_{epoch + 1}.pt"
                    store_transformer_model(path=current_checkpoint_path,model=model,offline_config=offline_config)
                    artifact.add_file(current_checkpoint_path)
                    wandb.log_artifact(artifact)
                    
                    
            for _ in range(num_evals):
                # Evaluate the performance of the model on the new env
                evaluate_ad_agent(
                    model=model,
                    env_config=env_config,
                    n_episodes=eval_length,
                    temp=eval_temp,
                    device=device,
                    track=track,
                )
            

    return model


def test(
    model,
    test_dataloader,
    env_config,
    current_epoch,
    logging_step,
    device="cuda",
    track=False,
):
    # Create loss function
    loss_fn = nn.CrossEntropyLoss()
    model = model.to(device)
    test_batches_per_epoch = len(test_dataloader)

    # Start training
    with torch.no_grad():

        pbar = tqdm(total=test_batches_per_epoch)
        losses = []
        accs = []

        for batch, (s, a, r, ti) in enumerate(test_dataloader):
            if model.transformer_config.time_embedding_type == "linear":
                ti = ti.to(torch.float32)

            states = s.to(device)
            action_labels = a.to(device)
            actions = action_labels[:, :-1]
            rewards = r.to(device)[:, :-1]
            time = ti.to(device)

            _, action_preds, _ = model.forward(
                states=states,
                actions=actions,
                rewards=rewards,
                timesteps=time,
            )

            action_preds = rearrange(action_preds, "b t a -> (b t) a")
            a_exp = rearrange(action_labels, "b t i -> (b t i)")
            
            acc = torch.mean((action_preds.argmax(-1) == a_exp).float()).item() * 100
            accs.append(acc)
            loss = loss_fn(action_preds, a_exp.to(torch.int64))
            losses.append(loss.item())


            pbar.set_description(f"TEST  - Epoch: {current_epoch}, Loss: {sum(losses)/len(losses):.4f}, Acc: {sum(accs)/len(accs):.4f}%")
            pbar.update(1)

        if track:
            batch_size = s.shape[0]
            wandb.log({"test/loss": sum(losses)/len(losses)}, step=logging_step)
            wandb.log({"test/acc": sum(accs)/len(accs)}, step=logging_step)

        pbar.close()
