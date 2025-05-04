import torch, optuna
import os, time, yaml, json

from dataset.data_processing import DataProcessing
from model.model_utils import gen_model, gen_optimizer, gen_scheduler
from training.torus import Torus
from training.train import train, validate
from utils.setup_seed import setup_seed
from utils.file_processing import FileProcessing
from utils.gpu_monitor import GPUMonitor
from utils.reprocess import reprocess
from utils.save_model import SaveModel

def training(param: dict, ht_param: dict | None = None, trial: optuna.Trial | None = None):
    fp = FileProcessing(param, ht_param, trial)
    fp.pre_make()
    plot_dir, model_dir, ckpt_dir = fp.plot_dir, fp.model_dir, fp.ckpt_dir
    log_file = fp.log_file
    training_logger = fp.training_logger
    gpu_logger = fp.gpu_logger
    gpu_monitor = GPUMonitor(gpu_logger)
    gpu_monitor.start()

    epoch_num = param['epoch_num']

    dp_start_time = time.perf_counter()
    DATA = DataProcessing(param, reprocess = reprocess(param))
    dataset = DATA.dataset
    train_loader = DATA.train_loader
    val_loader = DATA.val_loader
    test_loader = DATA.test_loader
    dp_end_time = time.perf_counter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = gen_model(param, torus)
    optimizer = gen_optimizer(param, model)
    scheduler = gen_scheduler(param, optimizer)

    fp.basic_info_log(
        dataset, train_loader, val_loader, test_loader, None,
        model, dp_end_time, dp_start_time
        )

    model_saving = SaveModel(param, model_dir, ckpt_dir, training_logger.info)
    start_epoch = fp.pre_train(model, optimizer, device, model_saving)
    torus = Torus(os.path.dirname(log_file), param['seed'])
    start_time = time.perf_counter()
    for epoch in range(start_epoch, epoch_num+1):
        try:
            lr = scheduler.optimizer.param_groups[0]['lr']
            train_loss, base_train_loss = train(model, train_loader, optimizer, torus, device)
            val_loss, base_val_loss = validate(model, val_loader, torus, device)
            info = json.dumps({
                'Epoch': epoch, 'LR': round(lr, 7),
                'train': {'loss': train_loss, 'base_loss': base_train_loss},
                'val': {'loss': val_loss, 'base_loss': base_val_loss}
                })
            if scheduler:
                scheduler.step(val_loss)
            model_saving.best_model(model, optimizer, epoch, val_loss)
            model_saving.regular_model(model, optimizer, epoch)
            fp.training_log(epoch, info, model_saving.best_val_loss, model_saving.best_epoch)
            torch.cuda.empty_cache()

            if model_saving.check_early_stopping():
                break

        except torch.cuda.OutOfMemoryError as e:
            training_logger.error(e)
            break
    end_time = time.perf_counter()

    fp.ending_log(end_time, start_time, epoch)
    gpu_monitor.stop()

def main():
    TIME = time.strftime('%b_%d_%Y_%H%M%S', time.localtime())
    with open('model_parameters.yml', 'r', encoding='utf-8') as mp:
        param: dict = yaml.full_load(mp)
    param['time'] = TIME

    seed = param['seed']
    setup_seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)

    if param['mode'] in ['training', 'fine-tuning']:
        training(param)

if __name__ == '__main__':
    main()
