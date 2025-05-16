import os, json, yaml, pickle, shutil, logging
import torch, optuna
import numpy as np
from torch_geometric.loader import DataLoader
from typing import Literal
from copy import deepcopy

from model.score_model import TensorProductScoreModel
from evaluation.eval_utils import EvaluationMetrics
from utils.save_model import SaveModel
from utils.post_processing import read_log
from utils.utils import Timer

def setup_logger(logger_name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    file_handler = logging.FileHandler(log_file, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger

def get_exp_dir(model_path: str) -> str | None:
    current_dir = os.path.dirname(model_path)

    while True:
        model_subdir_path = os.path.join(current_dir, 'Model')
        params_file_path = os.path.join(current_dir, 'model_parameters.yml')

        model_subdir_exists = os.path.isdir(model_subdir_path)
        params_file_exists = os.path.isfile(params_file_path)

        if model_subdir_exists and params_file_exists:
            return current_dir

        parent_dir = os.path.dirname(current_dir)

        if parent_dir == current_dir:
            return None

        current_dir = parent_dir

class FileProcessing:
    def __init__(
        self,
        param: dict,
        ht_param: dict | None = None,
        trial: optuna.Trial | None = None
        ) -> None:
        self.param = param
        self.TIME = self.param['time']
        self.jobtype = self.param['jobtype']
        self.ht_param = ht_param
        self.trial = trial

    def pre_make(self) -> None:
        if self.param['mode'] == 'hparam_tuning':
            os.makedirs(f'HPTuning_Recording/{self.jobtype}/{self.TIME}', exist_ok=True)
            self.optuna_log = f'HPTuning_Recording/{self.jobtype}/{self.TIME}/hptuning_{self.TIME}.log'
            self.optuna_db = f'sqlite:///HPTuning_Recording/{self.jobtype}/{self.TIME}/hptuning_{self.TIME}.db'
            self.hptuning_logger = setup_logger(f'hptuning_{self.TIME}_logger', self.optuna_log)

            if self.trial is None:
                return

            n_trials = self.ht_param['optuna']['n_trials']
            trial_name = f'Trial_{self.trial.number:0{len(str(n_trials))}d}'
            os.makedirs(f'HPTuning_Recording/{self.jobtype}/{self.TIME}/{trial_name}')
            with open(f'HPTuning_Recording/{self.jobtype}/{self.TIME}/{trial_name}/model_parameters.yml', 'w', encoding = 'utf-8') as mp:
                yaml.dump(self.param, mp, allow_unicode = True, sort_keys = False)
            if not os.path.exists(f'HPTuning_Recording/{self.jobtype}/{self.TIME}/hparam_tuning.yml'):
                with open(f'HPTuning_Recording/{self.jobtype}/{self.TIME}/hparam_tuning.yml', 'w', encoding = 'utf-8') as mp:
                    yaml.dump(self.ht_param, mp, allow_unicode = True, sort_keys = False)
            self.plot_dir =  f'HPTuning_Recording/{self.jobtype}/{self.TIME}/{trial_name}/Plot'
            os.makedirs(self.plot_dir)
            self.model_dir = f'HPTuning_Recording/{self.jobtype}/{self.TIME}/{trial_name}/Model'
            os.makedirs(self.model_dir)
            self.ckpt_dir = f'HPTuning_Recording/{self.jobtype}/{self.TIME}/{trial_name}/Model/checkpoint'
            os.makedirs(self.ckpt_dir)
            self.log_file = f'HPTuning_Recording/{self.jobtype}/{self.TIME}/{trial_name}/training_{trial_name}.log'
            self.training_logger = setup_logger(f'training_{trial_name}_logger', self.log_file)

        elif self.param['mode'] == 'generation':
            os.makedirs(f'Generation_Recording/{self.jobtype}/{self.TIME}')
            if self.param['dump_pymol']:
                self.pdb_dir = f'Generation_Recording/{self.jobtype}/{self.TIME}/PDB'
                os.makedirs(self.pdb_dir)
            else:
                self.pdb_dir = None
            self.data_dir = f'Generation_Recording/{self.jobtype}/{self.TIME}/Data'
            os.makedirs(self.data_dir)
            self.model_dir = f'Generation_Recording/{self.jobtype}/{self.TIME}/Model'
            os.makedirs(self.model_dir)
            shutil.copy(self.param['pretrained_model'], self.model_dir)
            self.log_file = f'Generation_Recording/{self.jobtype}/{self.TIME}/generation_{self.TIME}.log'
            self.generation_logger = setup_logger(f'generation_{self.TIME}_logger', self.log_file)

        elif self.param['mode'] == 'evaluation':
            os.makedirs(f'Evaluation_Recording/{self.jobtype}/{self.TIME}')
            self.data_dir = f'Evaluation_Recording/{self.jobtype}/{self.TIME}/Data'
            os.makedirs(self.data_dir)
            shutil.copy(self.param['true_confs'], self.data_dir)
            shutil.copy(self.param['gen_confs'], self.data_dir)
            self.log_file = f'Evaluation_Recording/{self.jobtype}/{self.TIME}/evaluation_{self.TIME}.log'
            self.evaluation_logger = setup_logger(f'evaluation_{self.TIME}_logger', self.log_file)

        else:
            os.makedirs(f'Training_Recording/{self.jobtype}/{self.TIME}')
            with open(f'Training_Recording/{self.jobtype}/{self.TIME}/model_parameters.yml', 'w', encoding = 'utf-8') as mp:
                yaml.dump(self.param, mp, allow_unicode = True, sort_keys = False)
            self.plot_dir = f'Training_Recording/{self.jobtype}/{self.TIME}/Plot'
            os.makedirs(self.plot_dir)
            self.model_dir = f'Training_Recording/{self.jobtype}/{self.TIME}/Model'
            os.makedirs(self.model_dir)
            self.ckpt_dir = f'Training_Recording/{self.jobtype}/{self.TIME}/Model/checkpoint'
            os.makedirs(self.ckpt_dir)
            if not os.path.isdir(f'Training_Recording/{self.jobtype}/recording'):
                os.makedirs(f'Training_Recording/{self.jobtype}/recording')
            self.log_file = f'Training_Recording/{self.jobtype}/{self.TIME}/training_{self.TIME}.log'
            self.training_logger = setup_logger(f'training_{self.TIME}_logger', self.log_file)

        if not self.param['mode'] == 'evaluation':
            self.gpu_logger = setup_logger(f'gpu_{self.TIME}_logger', f'{os.path.dirname(self.log_file)}/gpu_monitor.log')

    def basic_info_log(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        model: TensorProductScoreModel,
        timer: Timer
        ) -> None:
        days, hours, minutes, seconds = timer.get_tot_time()
        # if self.param['mode'] == 'prediction':
        #     self.prediction_logger.info(f"data_path: {os.path.abspath(self.param['path'])}")
        #     self.prediction_logger.info(json.dumps(self.param))
        #     self.prediction_logger.info(f"dataset: {str(dataset.data)}")
        #     self.prediction_logger.info(f"size of pred set: {len(pred_loader.dataset)}")
        #     self.prediction_logger.info(f"batch size: {pred_loader.batch_size}")
        #     self.prediction_logger.info(f"mean: {mean}, std: {std}")
        #     self.prediction_logger.info(f"Model:\n{model}")
        #     self.prediction_logger.info(f"Data processing time: {hours} h {minutes} m {seconds} s")
        #     self.prediction_logger.info("Begin predicting...")
        # else:
        self.training_logger.info(f"data_path: {os.path.abspath(self.param['path'])}")
        self.training_logger.info(json.dumps(self.param))
        self.training_logger.info(f"size of test set: {len(test_loader.dataset)}")
        self.training_logger.info(f"size of val set: {len(val_loader.dataset)}")
        self.training_logger.info(f"size of training set: {len(train_loader.dataset)}")
        self.training_logger.info(f"batch size: {train_loader.batch_size}")
        self.training_logger.info(f"Model:\n{model}")
        self.training_logger.info(f"Data processing time: {days} d {hours} h {minutes} m {seconds} s")
        self.training_logger.info("Begin training...")

    def load_model(
        self,
        state_dict: dict,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        mode: Literal['training', 'prediction', 'fine-tuning']
        ) -> None:
        model.load_state_dict(state_dict['model'])
        if mode == 'training':
            optimizer.load_state_dict(state_dict['optimizer'])

    def pre_train(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        model_saving: SaveModel | None=None
        ) -> int:
        start_epoch = 1
        pretrained_model = self.param['pretrained_model']
        mode = self.param['mode']
        if mode in ['generation', 'fine-tuning']:
            state_dict: dict = torch.load(pretrained_model, map_location=device)
            self.load_model(state_dict, model, optimizer, mode)
            start_epoch = 1
            pre_dir = get_exp_dir(pretrained_model)
            with open(f'{pre_dir}/model_parameters.yml', 'r', encoding='utf-8') as mp:
                pre_param: dict = yaml.full_load(mp)
            for p in [
                'atom_type', 'sigma_embed_dim', 'sigma_min', 'sigma_max', 'sh_lmax', 'ns',
                'nv', 'num_conv_layers', 'max_radius', 'radius_embed_dim', 'scale_by_sigma',
                'use_second_order_repr', 'batch_norm', 'residual'
                ]:
                self.param[p] = pre_param[p]
        # resume training
        elif mode == 'training':
            if pretrained_model:
                state_dict: dict = torch.load(pretrained_model, map_location=device)
                self.load_model(state_dict, model, optimizer, mode)
                start_epoch = state_dict['epoch'] + 1
                pre_dir = get_exp_dir(pretrained_model)
                pre_TIME = os.path.basename(pre_dir)
                pre_log_file = os.path.join(pre_dir, f'training_{pre_TIME}.log')
                shutil.copy(pre_log_file, f'Training_Recording/{self.jobtype}/{self.TIME}/pre.log')
                pre_log_info = read_log(pre_log_file, self.param)
                pre_log_text = pre_log_info.restart(start_epoch)
                with open(self.log_file, 'a') as lf:
                    lf.writelines(pre_log_text)

                best_model, best_optimizer = deepcopy(model), deepcopy(optimizer)
                pre_best_model: dict = torch.load(f'{pre_dir}/Model/best_model_{pre_TIME}.pth')
                self.load_model(pre_best_model, best_model, best_optimizer, mode)
                model_saving.best_model(best_model, best_optimizer, pre_best_model['epoch'], pre_best_model['val_loss'])
        self.start_epoch = start_epoch
        return start_epoch

    def training_log(
        self,
        epoch: int,
        info: dict,
        best_val_loss: float,
        best_epoch: int,
        ) -> None:
        self.best_val_loss = best_val_loss
        self.best_epoch = best_epoch
        if epoch % self.param['output_step'] == 0:
            self.training_logger.info(
                f'{info} '
                f'Best is epoch {best_epoch} with value: {best_val_loss}.'
                )

    def generation_log(self, idx: int, smi: str, mols: list | None):
        if mols:
            info = json.dumps({
                'idx': idx, 'smiles': smi, 'rotable_bonds': mols[0].n_rotable_bonds, 'n_confs': len(mols)
                })
            self.generation_logger.info(info)
        else:
            self.generation_logger.error(f'Failed to generate conformers for {smi}')
    
    def hptuning_log(self, study: optuna.Study) -> None:
        self.hptuning_logger.info(f'best value: {study.best_value}')
        self.hptuning_logger.info(f'best params: {study.best_params}')

    def ending_log(
        self,
        timer: Timer,
        epoch: int | None=None,
        conformer_dict: dict | None=None,
        eval_stats: EvaluationMetrics | None=None
        ) -> None:
        if self.param['mode'] == 'training':
            self.training_logger.info('Ending...')
            self.training_logger.info(f"Best val loss: {self.best_val_loss}")
            self.training_logger.info(f"Best epoch: {self.best_epoch}")
            days, hours, minutes, seconds = timer.get_tot_time()
            self.training_logger.info(f'Total time: {days} d {hours} h {minutes} m {seconds} s')
            days, hours, minutes, seconds = timer.get_average_time(epoch - self.start_epoch + 1)
            self.training_logger.info(f'Time per epoch: {days} d {hours} h {minutes} m {seconds} s')

        elif self.param['mode'] == 'generation':
            num_fail = sum(1 for value in conformer_dict.values() if value is None)
            num_success = len(conformer_dict) - num_fail
            self.generation_logger.info('Ending...')
            self.generation_logger.info(
                f'Successfully generated conformers for {num_success} out of '
                f'{len(conformer_dict)} molecules; {num_fail} failed.'
                )
            days, hours, minutes, seconds = timer.get_tot_time()
            self.generation_logger.info(f'Total time: {days} d {hours} h {minutes} m {seconds} s')
            days, hours, minutes, seconds = timer.get_average_time(num_success)
            self.generation_logger.info(f'Time per molecule: {days} d {hours} h {minutes} m {seconds} s')

        elif self.param['mode'] == 'evaluation':
            with open(f'{self.data_dir}/evaluation_results.pkl', 'wb') as f:
                pickle.dump(eval_stats.evaluation_results, f)
            for i, current_threshold in enumerate(eval_stats.RMSD_THRESHOLDS):
                self.evaluation_logger.info(f'\n--- RMSD Threshold: {current_threshold:.3f} ---')

                # Account for molecules with complete model failures (no predicted conformers)
                # These failures contribute 0% to coverage
                current_recall_coverage_vals = np.concatenate((
                    eval_stats.all_recall_coverages[:, i],
                    np.zeros(eval_stats.num_model_failures)
                    ))
                current_precision_coverage_vals = np.concatenate((
                    eval_stats.all_precision_coverages[:, i],
                    np.zeros(eval_stats.num_model_failures)
                    ))

                self.evaluation_logger.info(f'Recall Coverage:')
                self.evaluation_logger.info(f'  Mean = {np.nanmean(current_recall_coverage_vals) * 100:.2f}%')
                self.evaluation_logger.info(f'  Median = {np.nanmedian(current_recall_coverage_vals) * 100:.2f}%')

                self.evaluation_logger.info(f'Precision Coverage:')
                self.evaluation_logger.info(f'  Mean = {np.nanmean(current_precision_coverage_vals) * 100:.2f}%')
                self.evaluation_logger.info(f'  Median = {np.nanmedian(current_precision_coverage_vals) * 100:.2f}%')

            self.evaluation_logger.info(f'\n--- Overall Average Minimum RMSD ---')
            self.evaluation_logger.info(f'Average Minimum RMSD Recall (AMR Recall):')
            self.evaluation_logger.info(f'  Mean = {np.nanmean(eval_stats.all_amr_recalls):.4f}')
            self.evaluation_logger.info(f'  Median = {np.nanmedian(eval_stats.all_amr_recalls):.4f}')

            self.evaluation_logger.info(f'Average Minimum RMSD Precision (AMR Precision):')
            self.evaluation_logger.info(f'  Mean = {np.nanmean(eval_stats.all_amr_precisions):.4f}')
            self.evaluation_logger.info(f'  Median = {np.nanmedian(eval_stats.all_amr_precisions):.4f}')

            self.evaluation_logger.info(f'\n--- Summary ---')
            self.evaluation_logger.info(f'Compared {len(eval_stats.evaluation_results)} conformer sets.')
            self.evaluation_logger.info(f'Model failed to generate conformers for {eval_stats.num_model_failures} molecules.')
            self.evaluation_logger.info(f'Encountered RDKit errors during RMSD calculation for {eval_stats.num_additional_failures} molecules.')
