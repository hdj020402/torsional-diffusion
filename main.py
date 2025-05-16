import torch, optuna
import os, time, yaml, json, pickle
from multiprocessing import Pool
from functools import partial

from dataset.data_processing import DataProcessing
from model.model_utils import gen_model, gen_optimizer, gen_scheduler
from training.torus import Torus
from training.train import train, validate
from generation.sampling import sample_confs
from evaluation.eval_utils import (EvaluationMetrics,
                                   preparation,
                                   calculate_rmsds_for_true_conformer,
                                   aggregate_evaluation_results,
                                   RMSD_THRESHOLDS)
from utils.setup_seed import setup_seed
from utils.file_processing import FileProcessing
from utils.gpu_monitor import GPUMonitor
from utils.reprocess import reprocess
from utils.save_model import SaveModel
from utils.utils import Timer

def training(param: dict, ht_param: dict | None = None, trial: optuna.Trial | None = None):
    fp = FileProcessing(param, ht_param, trial)
    fp.pre_make()
    plot_dir, model_dir, ckpt_dir = fp.plot_dir, fp.model_dir, fp.ckpt_dir
    log_file = fp.log_file
    torus = Torus(param['path'], param['seed'])
    training_logger = fp.training_logger
    gpu_logger = fp.gpu_logger
    gpu_monitor = GPUMonitor(gpu_logger)
    gpu_monitor.start()

    epoch_num = param['epoch_num']

    dp_timer = Timer()
    dp_timer.start()
    DATA = DataProcessing(param, reprocess = reprocess(param))
    train_loader = DATA.train_loader
    val_loader = DATA.val_loader
    test_loader = DATA.test_loader
    dp_timer.end()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = gen_model(param, torus)
    optimizer = gen_optimizer(param, model)
    scheduler = gen_scheduler(param, optimizer)

    fp.basic_info_log(train_loader, val_loader, test_loader, model, dp_timer)

    model_saving = SaveModel(param, model_dir, ckpt_dir, training_logger.info)
    start_epoch = fp.pre_train(model, optimizer, device, model_saving)
    timer = Timer()
    timer.start()
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
    timer.end()

    fp.ending_log(timer, epoch)
    gpu_monitor.stop()

def generation(param: dict):
    fp = FileProcessing(param)
    fp.pre_make()
    pdb_dir, model_dir, data_dir = fp.pdb_dir, fp.model_dir, fp.data_dir
    log_file = fp.log_file
    torus = Torus(param['path'], param['seed'])
    generation_logger = fp.generation_logger
    gpu_logger = fp.gpu_logger
    gpu_monitor = GPUMonitor(gpu_logger)
    gpu_monitor.start()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = gen_model(param, torus)
    optimizer = gen_optimizer(param, model)
    fp.pre_train(model, optimizer, device)

    import pandas as pd
    test_data = pd.read_csv(param['test_csv'])

    conformer_dict = {}
    timer = Timer()
    timer.start()
    for smi_idx, (n_confs, smi) in enumerate(zip(test_data['n_confs'], test_data['smiles'])):
        mols = sample_confs(2 * n_confs, smi, param, model, pdb_dir)
        fp.generation_log(smi_idx, smi, mols)
        conformer_dict[smi] = mols
    timer.end()

    fp.ending_log(timer, conformer_dict=conformer_dict)
    with open(f'{data_dir}/output.pkl', 'wb') as f:
        pickle.dump(conformer_dict, f)

    gpu_monitor.stop()

def evaluation(param: dict):
    fp = FileProcessing(param)
    fp.pre_make()
    log_file = fp.log_file
    evaluation_logger = fp.evaluation_logger

    timer = Timer()
    timer.start()
    evaluation_logger.info('Preparing RMSD Calculation Jobs...')
    num_model_failures, evaluation_results, rmsds_calculation_jobs = preparation(param, evaluation_logger.warning)

    evaluation_logger.info(f'Performing Parallel RMSD Calculation with {param["num_workers"]} workers...')
    
    if param['num_workers'] > 1:
        with Pool(param['num_workers']) as p:
            map_function = p.imap_unordered
            results_iterator = map_function(
                partial(calculate_rmsds_for_true_conformer, trace_func=evaluation_logger.warning),
                rmsds_calculation_jobs
                )
            for result in results_iterator:
                smi, true_conf_idx, rmsds = result
                evaluation_results[smi]['rmsd'][true_conf_idx] = rmsds
    else:
        results_iterator = map(
            partial(calculate_rmsds_for_true_conformer, trace_func=evaluation_logger.warning),
            rmsds_calculation_jobs
            )
        for result in results_iterator:
            smi, true_conf_idx, rmsds = result
            evaluation_results[smi]['rmsd'][true_conf_idx] = rmsds

    evaluation_logger.info('Aggregating Performance Stats...')
    (all_recall_coverages, all_amr_recalls, all_precision_coverages, all_amr_precisions,
     num_additional_failures) = aggregate_evaluation_results(evaluation_results)

    eval_stats = EvaluationMetrics(
        RMSD_THRESHOLDS=RMSD_THRESHOLDS,
        num_model_failures=num_model_failures,
        num_additional_failures=num_additional_failures,
        all_recall_coverages=all_recall_coverages,
        all_amr_recalls=all_amr_recalls,
        all_precision_coverages=all_precision_coverages,
        all_amr_precisions=all_amr_precisions,
        evaluation_results=evaluation_results
    )
    timer.end()
    fp.ending_log(timer, eval_stats=eval_stats)

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
    elif param['mode'] == 'generation':
        generation(param)
    elif param['mode'] == 'evaluation':
        evaluation(param)

if __name__ == '__main__':
    main()
