import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model.score_model import TensorProductScoreModel

def gen_model(param: dict, dataset, ) -> TensorProductScoreModel:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = TensorProductScoreModel(
        ...   # TODO: Complete the arguments
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
