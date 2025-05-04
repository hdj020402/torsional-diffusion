import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model.score_model import TensorProductScoreModel
from training.torus import Torus

def gen_model(param: dict, torus: Torus) -> TensorProductScoreModel:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = TensorProductScoreModel(
        in_node_features=len(param['atom_type']) + 39,
        in_edge_features=4,
        sigma_embed_dim=param['sigma_embed_dim'],
        sigma_min=param['sigma_min'],
        sigma_max=param['sigma_max'],
        sh_lmax=param['sh_lmax'],
        ns=param['ns'],
        nv=param['nv'],
        num_conv_layers=param['num_conv_layers'],
        max_radius=param['max_radius'],
        radius_embed_dim=param['radius_embed_dim'],
        scale_by_sigma=param['scale_by_sigma'],
        torus=torus,
        use_second_order_repr=param['use_second_order_repr'],
        batch_norm=param['batch_norm'],
        residual=param['residual'],
        )
    model = net.to(device)
    return model

def gen_optimizer(param: dict, model: TensorProductScoreModel) -> torch.optim.Optimizer:
    optimizer = getattr(torch.optim, param['optimizer'])(model.parameters(), lr = param['lr'])
    return optimizer

def gen_scheduler(param: dict, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler | None:
    if param['scheduler']['type'] == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(
            optimizer, mode = 'min',
            factor = param['scheduler']['factor'], patience = param['scheduler']['patience'],
            min_lr = param['scheduler']['min_lr']
            )
    else:
        print('No scheduler')
        scheduler = None
    return scheduler
