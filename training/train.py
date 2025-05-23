import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from training.torus import Torus
from model.score_model import TensorProductScoreModel


def train(
    model: TensorProductScoreModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    torus: Torus,
    device: torch.device,
    ) -> tuple[float, float]:
    model.train()
    loss_tot = 0
    base_tot = 0

    for data in tqdm(loader, total=len(loader)):
        data = data.to(device)
        optimizer.zero_grad()

        data = model(data)
        pred = data.edge_pred

        score = torus.score(
            data.edge_rotate.cpu().numpy(),
            data.edge_sigma.cpu().numpy())
        score = torch.tensor(score, device=pred.device)
        score_norm = torus.score_norm(data.edge_sigma.cpu().numpy())
        score_norm = torch.tensor(score_norm, device=pred.device)
        loss = ((score - pred) ** 2 / score_norm).mean()

        loss.backward()
        optimizer.step()
        loss_tot += loss.item()
        base_tot += (score ** 2 / score_norm).mean().item()

    loss_avg = loss_tot / len(loader)
    base_avg = base_tot / len(loader)
    return loss_avg, base_avg


@torch.no_grad()
def validate(
    model: TensorProductScoreModel,
    loader: DataLoader,
    torus: Torus,
    device: torch.device,
    ) -> tuple[float, float]:
    model.eval()
    loss_tot = 0
    base_tot = 0

    for data in tqdm(loader, total=len(loader)):

        data = data.to(device)
        data = model(data)
        pred = data.edge_pred.cpu()

        score = torus.score(
            data.edge_rotate.cpu().numpy(),
            data.edge_sigma.cpu().numpy())
        score = torch.tensor(score)
        score_norm = torus.score_norm(data.edge_sigma.cpu().numpy())
        score_norm = torch.tensor(score_norm)
        loss = ((score - pred) ** 2 / score_norm).mean()

        loss_tot += loss.item()
        base_tot += (score ** 2 / score_norm).mean().item()

    loss_avg = loss_tot / len(loader)
    base_avg = base_tot / len(loader)
    return loss_avg, base_avg

