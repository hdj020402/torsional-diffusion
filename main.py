import torch, optuna
import os, time, yaml

from dataset.data_processing import DataProcessing
from model.model_utils import gen_model, gen_optimizer, gen_scheduler
from training.torus import Torus
from training.train import train, validate
from utils.setup_seed import setup_seed
from utils.file_processing import FileProcessing
from utils.gpu_monitor import GPUMonitor
from utils.reprocess import reprocess

def training(param: dict, ht_param: dict | None = None, trial: optuna.Trial | None = None):
    fp = FileProcessing(param, ht_param, trial)
    fp.pre_make()
    plot_dir, model_dir, ckpt_dir = fp.plot_dir, fp.model_dir, fp.ckpt_dir
    log_file = fp.log_file
    training_logger = fp.training_logger
    gpu_logger = fp.gpu_logger
    gpu_monitor = GPUMonitor(gpu_logger)
    gpu_monitor.start()
    error_dict = fp.error_dict

    epoch_num = param['epoch_num']

    dp_start_time = time.perf_counter()
    DATA = DataProcessing(param, reprocess = reprocess(param))
    dataset = DATA.dataset
    train_loader = DATA.train_loader
    val_loader = DATA.val_loader
    test_loader = DATA.test_loader
    dp_end_time = time.perf_counter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = gen_model(param, dataset)
    optimizer = gen_optimizer(param, model)
    scheduler = gen_scheduler(param, optimizer)

    torus = Torus(os.path.dirname(log_file), param['seed'])
    for epoch in range(start_epoch, epoch_num+1):
        train()
        validate()

def main():
    TIME = time.strftime('%b_%d_%Y_%H%M%S', time.localtime())
    with open('model_parameters.yml', 'r', encoding='utf-8') as mp:
        param: dict = yaml.full_load(mp)
    param['time'] = TIME

    seed = param['seed']
    setup_seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)

if __name__ == '__main__':
    main()
