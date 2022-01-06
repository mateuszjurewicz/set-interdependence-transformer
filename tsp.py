"""
Basic script for running a full experiment on the Travelling Salesman Problem.
Should include a unique id for each run, a choice of model, full logging
(including training progress and evaluation metrics in e.g. Sacred), model
persistance and possibly generation of plots.
"""
import json
import logging
import os

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from datetime import timezone

# torch data
from torch.utils.data import DataLoader

# sacred experiment
from sacred import Experiment
from sacred.observers import MongoObserver

# custom imports
from utils import get_run_id, count_params, get_logger, \
    get_run_full_name, get_model_full_name, get_config_readable, \
    add_needed_folders
from utils_data import TSPDataset, get_example, show_tsp, get_tour_length, \
    reorder, get_permuted_batch
from models import unpack_model_config, get_model, train_epochs
from metrics import run_tsp_tests, plot_losses

if __name__ == "__main__":

    # configuration
    cfg = dict()

    # cfg experiment
    cfg['experiment_name'] = 'tsp'
    cfg['model_configs_path'] = os.path.join('model_configs', 'tsp.json')
    cfg['generate_new_data'] = False  # keep this False for most experiments

    # cfg seed
    cfg['seed'] = random.randint(0, 10000)

    # reporting and model persisting
    cfg['report_every_n_batches'] = 50
    cfg['validate_every_n_epochs'] = 10
    cfg['save_model_every_n_epochs'] = 10

    # cfg metrics tracking
    cfg['log_level'] = logging.INFO
    cfg['track_metrics'] = True
    cfg['db_name'] = 'sacred'
    cfg['db_url'] = 'localhost:27017'

    # cfg data
    cfg['train_size'] = 20000
    cfg['test_size'] = 5000
    cfg['cv_size'] = 3000 # test and cv need different sizes
    cfg['train_set_size'] = 11  # 20 takes a long, long time
    cfg['cv_set_size'] = 11
    cfg['test_set_size'] = 11
    cfg['train_batch_size'] = 64
    cfg['cv_batch_size'] = 64
    cfg['test_batch_size'] = 64
    cfg['dataset_train'] = '_'.join(
        [str(cfg['train_set_size']), str(cfg['train_size'])])
    cfg['dataset_cv'] = '_'.join(
        [str(cfg['cv_set_size']), str(cfg['cv_size'])])
    cfg['dataset_test'] = '_'.join(
        [str(cfg['test_set_size']), str(cfg['test_size'])])
    cfg['train_path'] = 'data/{}_{}.pth'.format(cfg['experiment_name'],
                                                cfg['dataset_train'])
    cfg['cv_path'] = 'data/{}_{}.pth'.format(cfg['experiment_name'],
                                             cfg['dataset_cv'])
    cfg['test_path'] = 'data/{}_{}.pth'.format(cfg['experiment_name'],
                                               cfg['dataset_test'])
    cfg['x_name'] = 'X'
    cfg['y_name'] = 'Y'

    # add dirs if necessary
    add_needed_folders()

    # read model configs
    with open(cfg['model_configs_path'], 'r', encoding='UTF-8') as f:
        model_configs = json.load(f)

    for model_cfg in model_configs['models']:

        # unpack the model params
        cfg = unpack_model_config(model_cfg, cfg)

        # handle gpu/cpu
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # track duration
        time_start = datetime.now(timezone.utc)

        # unique run ID
        run_id = get_run_id()
        cfg['run_id'] = run_id

        # cfg run full name and model path
        cfg['run_full_name'] = get_run_full_name(cfg)
        cfg['model_path'] = 'models/{}'.format(get_model_full_name(cfg))
        cfg['logs_path'] = 'logs/{}'.format(cfg['run_full_name'])

        # logs
        log = get_logger(cfg)

        # log start time
        log.info('EXPERIMENT run started, id: {}'.format(cfg['run_id']))
        log.info('Time started (UTC): {}'.format(str(time_start)))

        # reproducibility
        log.info('Seeding for reproducibility with: {}'.format(cfg['seed']))
        torch.manual_seed(cfg['seed'])
        random.seed(cfg['seed'])
        np.random.seed(cfg['seed'])

        # log config
        cfg_pretty = get_config_readable(cfg)
        log.info('Config for {}'.format(cfg['run_full_name']))
        log.info('\n' + cfg_pretty)

        # data generation
        if cfg['generate_new_data']:
            # training data
            log.info('Data generation, train set')
            train_set = TSPDataset(cfg['train_size'],
                                   cfg['train_set_size'])
            train_dataloader = DataLoader(train_set,
                                          batch_size=cfg['train_batch_size'],
                                          shuffle=True, num_workers=4)

            # cv data
            log.info('Data generation, cv set')
            test_set = TSPDataset(cfg['cv_size'], cfg['cv_set_size'])
            cv_dataloader = DataLoader(test_set,
                                       batch_size=cfg['cv_batch_size'],
                                       shuffle=True, num_workers=4)

            # test data
            log.info('Data generation, test set')
            test_set = TSPDataset(cfg['test_size'], cfg['test_set_size'])
            test_dataloader = DataLoader(test_set,
                                         batch_size=cfg['test_batch_size'],
                                         shuffle=True, num_workers=4)

            # data persistance
            log.info('Data persistance (saving dataloaders)')
            torch.save(train_dataloader, cfg['train_path'])
            torch.save(cv_dataloader, cfg['cv_path'])
            torch.save(test_dataloader, cfg['test_path'])

        # data loading
        log.info('Data (re-)loading')
        train_dataloader = torch.load(cfg['train_path'])
        cv_dataloader = torch.load(cfg['cv_path'])
        test_dataloader = torch.load(cfg['test_path'])

        # model
        model = get_model(cfg)
        model.to(device)
        log.info('Model architecture')
        log.info(model)

        # optimizer
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg['learning_rate'])

        # log params
        num_params, params_string = count_params(model, return_string=True)
        log.info(params_string)

        # permutation invariance check
        # test perm-invar on 2-dim elems
        batch_check_perm = get_permuted_batch(cfg['train_set_size'],
                                              cfg['elem_dims'],
                                              is_random=False)
        _, perm_check_preds = model(batch_check_perm)
        log.info('Permutation invariance of output check (2-dim elems):')
        log.info(perm_check_preds)
        log.info(batch_check_perm[0])
        log.info(batch_check_perm[1])

        # sample prediction prior to training
        log.info('Single prediction (before training) ... ')
        single_x, single_y = get_example(test_dataloader, as_batch=True)
        log.info('Sample x: {} \n{}'.format(single_x.size(), single_x))
        log.info('Sample y: {} \n{}'.format(single_y.size(), single_y))

        # predict (pre training)
        model.eval()
        _, single_pred = model(single_x)
        model.train()
        log.info('(before training) Prediction: \n{}'.format(single_pred))

        # move batches back to cpu for some metric functions
        single_pred = single_pred.to('cpu')
        single_x = single_x.to('cpu')
        single_y = single_y.to('cpu')

        # restore predicted order (permute)
        predicted_order = reorder(single_x, single_pred)
        log.info(
            '(before training) Predicted order: \n{}'.format(predicted_order))
        log.info('Correct order: \n{}'.format(reorder(single_x, single_y)))

        # include tour length info
        target_tour_length = get_tour_length(single_x[0], single_y[0])
        predicted_tour_length = get_tour_length(single_x[0], single_pred[0])
        log.info(
            '(before training) Sample tour lengths: target: {:.5f}, predicted: {:.5f}'.format(
                target_tour_length, predicted_tour_length))

        # select the model-appropriate training function and loss
        if cfg['permute_module_type'] == 'futurehistory':
            criterion = nn.NLLLoss(reduction='none')
        else:
            criterion = torch.nn.CrossEntropyLoss()

        # model training
        log.info('Model training started')
        model.train()
        last_loss, losses = train_epochs(
            model,
            cfg['model_path'],
            optimizer,
            criterion,
            train_dataloader,
            cv_dataloader,
            cfg,
            cfg['num_epochs'],
            cfg['x_name'],
            cfg['y_name'],
            logger=log,
            tqdm_file=
            logging.root.handlers[
                0].stream,
            run_tests_function=run_tsp_tests,
            allow_gpu=True,
            is_futurehistory=cfg['permute_module_type'] == 'futurehistory',  # check if it's future-history
            report_every_n_batches=cfg['report_every_n_batches'],
            validate_every_n_epochs=cfg['validate_every_n_epochs'],
            save_model_every_n_epochs=cfg['save_model_every_n_epochs']
        )

        # sample prediction after training
        log.info('Single prediction (after training) ... ')
        single_x, single_y = get_example(test_dataloader, as_batch=True)
        log.info('Sample x: {} \n{}'.format(single_x.size(), single_x))
        log.info('Sample y: {} \n{}'.format(single_y.size(), single_y))

        # predict (post training)
        model.eval()
        _, single_pred = model(single_x)
        log.info('(after training) Prediction: \n{}'.format(single_pred))

        # move batches back to cpu for some metric functions
        single_pred = single_pred.to('cpu')
        single_x = single_x.to('cpu')
        single_y = single_y.to('cpu')

        # restore predicted order (permute)
        predicted_order = reorder(single_x, single_pred)
        log.info(
            '(after training) Predicted order: \n{}'.format(predicted_order))
        log.info('Correct order: \n{}'.format(reorder(single_x, single_y)))

        # include tour length info
        target_tour_length = get_tour_length(single_x[0], single_y[0])
        predicted_tour_length = get_tour_length(single_x[0], single_pred[0])
        log.info(
            '(after training) Sample tour lengths: target: {:.5f}, predicted: {:.5f}'.format(
                target_tour_length, predicted_tour_length))

        # plot
        show_tsp(single_x[0], single_pred[0], cfg['run_id'],
                 predicted_tour_length,
                 target_tour_length)

        # model testing
        # track metrics via Sacred
        log.info('Model testing')
        if cfg['track_metrics']:

            # initiate the experiment (interactive False for reproducibility)
            log.info('Model testing - initiating sacred experiment')
            ex = Experiment(name=cfg['experiment_name'], interactive=False)

            # add an observer, storing experiment info
            log.info('Model testing - adding observers')
            ex.observers.append(
                MongoObserver(url=cfg['db_url'], db_name=cfg['db_name']))


            # experiment config
            @ex.config
            def ex_config():
                run_id_own = None
                run_full_name = None
                learning_rate = None
                dataset_train = None
                dataset_test = None
                epochs = None
                model = None
                model_id = None
                masking = None
                final_training_loss = None
                optimizer = None
                db_name = None
                db_url = None
                num_params = None
                cfg = cfg  # needed


            # experiment run function
            @ex.main
            def run_model_tests():
                # run all tests, get all results
                perc_legal, avg_tl_target, avg_tl_predicted, result_tau, \
                result_spearman, rank_valid_perc = run_tsp_tests(
                    model, test_dataloader, log, cfg, batched=True)

                # log params and results
                log_experiment_results(perc_legal, avg_tl_target,
                                       avg_tl_predicted, result_tau,
                                       result_spearman,
                                       rank_valid_perc)

                return round(avg_tl_predicted, 5)


            def log_experiment_results(perc_legal, avg_tl_target,
                                       avg_tl_predicted, r_tau, r_spearman,
                                       rank_valid_perc):
                # round if not none
                if r_tau:
                    r_tau = round(r_tau, 5)
                if r_spearman:
                    r_spearman = round(r_spearman, 5)

                # log the results
                ex.log_scalar('test.rank_correlation_valid_perc',
                              rank_valid_perc)
                ex.log_scalar('test.tsp_avg_tl_predicted', avg_tl_predicted, 5)
                ex.log_scalar('test.tsp_avg_tl_target', avg_tl_target, 5)
                ex.log_scalar('test.tsp_perc_legal_tours', perc_legal)
                ex.log_scalar('test.tau', r_tau, 5)
                ex.log_scalar('test.rho', r_spearman, 5)


            # experiment config
            current_tested_model = model
            current_optimizer = optimizer
            current_last_loss = last_loss

            config_update = {
                'run_id_own': cfg['run_id'],
                'run_full_name': cfg['run_full_name'],
                'learning_rate': cfg['learning_rate'],
                'dataset_train': cfg['dataset_train'],
                'dataset_test': cfg['dataset_test'],
                'epochs': cfg['num_epochs'],
                'model': cfg['model_type'],
                'model_id': cfg['run_id'],
                'masking': cfg['permute_module_masking'],
                'final_training_loss': round(current_last_loss.item(), 10),
                'optimizer': current_optimizer.__class__.__name__,
                'db_name': cfg['db_name'],
                'db_url': cfg['db_url'],
                'num_params': num_params,
                'cfg': cfg,
            }

            # run the experiment, with updated config
            log.info('Model testing - running sacred experiment')
            run_info = ex.run(config_updates=config_update)

        # model persisting
        log.info('Model saving ... ')
        torch.save(model, cfg['model_path'] + '.pth')

        # loss plotting
        log.info('Plotting losses ... ')
        plot_losses(losses, cfg)

        # duration
        time_end = datetime.now(timezone.utc)
        log.info('Time ended (UTC): {}'.format(str(time_end)))

        duration = time_end - time_start
        log.info('Time taken: {}'.format(str(duration)))

        log.info('Run {} finished.'.format(cfg['run_full_name']))
