"""
Basic script for running a full experiment on grammars (context sensitive and
context free, i.e. Dyck Language) framed as permutation learning.
"""
import copy
import json
import logging
import os
import random
from datetime import datetime
from datetime import timezone

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# sacred experiment
from sacred import Experiment
from sacred.observers import MongoObserver

from metrics import run_grammar_tests, plot_losses, restore_grammar_sequence, \
    examine_single_prediction
from models import unpack_model_config, get_model, train_epochs
# custom imports
from utils import get_run_id, count_params, get_logger, \
    get_run_full_name, get_model_full_name, get_config_readable, \
    add_needed_folders
from utils_data import get_example, get_grammar_dataloader

# torch data

if __name__ == "__main__":

    # confguration
    cfg = dict()

    # cfg experiment
    cfg['grammar_type'] = 'harder'  # [canonical, brackets, harder]
    cfg['model_configs_path'] = os.path.join('model_configs', 'grammar.json')
    cfg['experiment_name'] = 'grammar_{}'.format(cfg['grammar_type'])
    cfg['generate_new_data'] = True

    # cfg seed
    cfg['seed'] = random.randint(0, 10000)

    # reporting and model persisting
    cfg['report_every_n_batches'] = 50
    cfg['validate_every_n_epochs'] = 10
    cfg['save_model_every_n_epochs'] = 10

    # cfg grammar
    cfg['padding_symbol'] = 'P'  # don't change this, hardcoded
    if cfg['grammar_type'] == 'canonical' or cfg['grammar_type'] == 'harder':
        cfg['word2idx'] = {'a': 0, 'b': 1, 'c': 2, cfg['padding_symbol']: 3}
    elif cfg['grammar_type'] == 'brackets':
        cfg['word2idx'] = {'(': 0, ')': 1, '{': 2, '}': 3,
                           cfg['padding_symbol']: 4}
    else:
        error_msg = 'Unknown grammar type chosen, try canonical, ' \
                    'brackets or harder'
        logging.exception(error_msg)
        raise Exception(error_msg)

    cfg['idx2word'] = {v: k for k, v in cfg['word2idx'].items()}

    # cfg metrics tracking
    cfg['log_level'] = logging.INFO
    cfg['track_metrics'] = True
    cfg['db_name'] = 'sacred'
    cfg['db_url'] = 'localhost:27017'

    # cfg data
    cfg['train_size'] = 5000
    cfg['test_size'] = 1000
    cfg['cv_size'] = 1000
    cfg['train_set_size'] = 675
    cfg['test_set_size'] = 675
    cfg['cv_set_size'] = 675
    cfg['batch_size'] = 32
    cfg['dataset_train'] = '_'.join([str(cfg['train_set_size']), str(cfg['train_size'])])
    cfg['dataset_test'] = '_'.join([str(cfg['test_set_size']), str(cfg['test_size'])])
    cfg['dataset_cv'] = '_'.join([str(cfg['cv_set_size']), str(cfg['cv_size'])])
    cfg['train_path'] = 'data/{}_{}.pth'.format(cfg['experiment_name'], cfg['dataset_train'])
    cfg['test_path'] = 'data/{}_{}.pth'.format(cfg['experiment_name'], cfg['dataset_test'])
    cfg['cv_path'] = 'data/{}_{}.pth'.format(cfg['experiment_name'], cfg['dataset_cv'])
    cfg['x_name'] = 0
    cfg['y_name'] = 1

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

        # cfg run full name
        cfg['run_full_name'] = get_run_full_name(cfg)
        cfg['model_path'] = 'models/{}.pth'.format(get_model_full_name(cfg))
        cfg['logs_path'] = 'logs/{}'.format(cfg['run_full_name'])

        # logs
        log = get_logger(cfg)

        # log start time
        log.info('EXPERIMENT run started, name: {} id: {}'.format(
            cfg['experiment_name'], cfg['run_id']))
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
            train_dataloader = get_grammar_dataloader(
                grammar_type=cfg['grammar_type'],
                dataset_type='train',
                logger=log,
                config=cfg)

            # test data
            test_dataloader = get_grammar_dataloader(
                grammar_type=cfg['grammar_type'],
                dataset_type='test',
                logger=log,
                config=cfg)

            # test data
            cv_dataloader = get_grammar_dataloader(
                grammar_type=cfg['grammar_type'],
                dataset_type='cv',
                logger=log,
                config=cfg)

            # data persistance
            log.info('Data persistance (saving dataloaders)')
            torch.save(train_dataloader, cfg['train_path'])
            torch.save(test_dataloader, cfg['test_path'])
            torch.save(cv_dataloader, cfg['cv_path'])

        # data loading
        log.info('Data (re-)loading')
        train_dataloader = torch.load(cfg['train_path'])
        test_dataloader = torch.load(cfg['test_path'])
        cv_dataloader = torch.load(cfg['cv_path'])

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

        # sample prediction prior to training
        log.info('Single prediction (before training) ... ')
        model.eval()
        single_x, single_y = get_example(test_dataloader, as_batch=True,
                                         x_name=cfg['x_name'],
                                         y_name=cfg['y_name'])
        examine_single_prediction(model, single_x, single_y, log,
                                  restore_grammar_sequence, idx2word=cfg['idx2word'])

        # select the model-appropriate training function and loss
        criterion = nn.NLLLoss(reduction='none')

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
            run_tests_function=run_grammar_tests,
            allow_gpu=True,
            is_futurehistory=cfg['permute_module_type'] == 'futurehistory',  # check if it's future-history
            report_every_n_batches=cfg['report_every_n_batches'],
            validate_every_n_epochs=cfg['validate_every_n_epochs'],
            save_model_every_n_epochs=cfg['save_model_every_n_epochs']
        )

        # predict on sample (post training)
        log.info('Single prediction (after training) ... ')
        examine_single_prediction(model, single_x, single_y, log,
                                  restore_grammar_sequence, idx2word=cfg['idx2word'])

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
                result_general, result_tau, result_spearman, rank_valid_perc = run_grammar_tests(
                    model, test_dataloader, log, cfg)

                # log params and results
                log_experiment_results(result_general, result_tau,
                                       result_spearman,
                                       rank_valid_perc)

                return round(result_general, 5)


            def log_experiment_results(r_general, r_tau, r_spearman,
                                       rank_valid_perc):
                # round if not none
                if r_tau:
                    r_tau = round(r_tau, 5)
                if r_spearman:
                    r_spearman = round(r_spearman, 5)

                # log the results
                ex.log_scalar('test.rank_correlation_valid_perc',
                              rank_valid_perc)
                ex.log_scalar('test.general', r_general)
                ex.log_scalar('test.tau', r_tau, 5)
                ex.log_scalar('test.rho', r_spearman, 5)


            # experiment config
            current_tested_model = model
            current_optimizer = optimizer
            current_last_loss = last_loss

            # make all cfg values be strings (Sacred doesn't like dicts as values)
            cfg_update = copy.deepcopy(cfg)
            for k, v in cfg_update.items():
                if type(v) == dict:
                    cfg_update[k] = str(v)

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
                'cfg': cfg_update, }

            # run the experiment, with updated config
            log.info('Model testing - running sacred experiment')
            run_info = ex.run(config_updates=config_update)

        # model persisting
        log.info('Model saving ... ')
        torch.save(model, cfg['model_path'])

        # loss plotting
        log.info('Plotting losses ... ')
        plot_losses(losses, cfg)

        # duration
        time_end = datetime.now(timezone.utc)
        log.info('Time ended (UTC): {}'.format(str(time_end)))

        duration = time_end - time_start
        log.info('Time taken: {}'.format(str(duration)))

        log.info('Run {} finished.'.format(cfg['run_full_name']))
