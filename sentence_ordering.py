"""
Basic script for running a full experiment on the ROCStory dataset.
"""
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

from metrics import run_sentence_ordering_tests, plot_losses, \
    examine_single_sentence_order_prediction
from models import get_model, unpack_model_config, train_epochs
# custom imports
from utils import get_run_id, count_params, get_logger, get_run_full_name, get_model_full_name, \
    get_config_readable, add_needed_folders
from utils_data import get_sentence_ordering_dataloader, \
    get_sentence_ordering_tokenizer, decode_sentence_batch

# torch data

if __name__ == "__main__":

    # confguration
    cfg = dict()

    # cfg experiment
    cfg['dataset_name'] = 'rocstory'  # ['rocstory']
    cfg['experiment_name'] = 'sentence_ordering_{}'.format(cfg['dataset_name'])
    cfg['language_model'] = 'bert-base-cased'  # ['bert-base-cased']
    cfg['generate_new_data'] = False
    cfg['debug_dataset_sizes'] = False
    cfg['model_configs_path'] = os.path.join('model_configs', 'sentence_ordering.json')

    # cfg seed
    cfg['seed'] = random.randint(0, 10000)

    # reporting and model persisting
    cfg['report_every_n_batches'] = 50
    cfg['validate_every_n_epochs'] = 25
    cfg['save_model_every_n_epochs'] = 10

    # rocstory
    cfg['max_sentence_length'] = 42  # checked, 42 is max after BERT tokenization for entire ROCStory set
    cfg['num_sentences'] = 5

    # cfg metrics tracking
    cfg['log_level'] = logging.INFO
    cfg['track_metrics'] = True
    cfg['db_name'] = 'sacred'
    cfg['db_url'] = 'localhost:27017'

    # cfg data
    cfg['dataset_train'] = 'rocstory'  # for backwards compatibility
    cfg['dataset_test'] = 'rocstory'  # for backwards compatibility
    cfg['batch_size'] = 32
    cfg['train_dataset_path'] = os.path.join(
        os.getcwd(), 'data', 'rocstory',
        '{}_setlen_{}_batch_{}_train.pb'.format(
            cfg['language_model'], cfg['max_sentence_length'],
            cfg['batch_size']))
    cfg['test_dataset_path'] = os.path.join(
        os.getcwd(), 'data', 'rocstory', '{}_setlen_{}_batch_{}_test.pb'.format(
            cfg['language_model'], cfg['max_sentence_length'],
            cfg['batch_size']))
    cfg['cv_dataset_path'] = os.path.join(
        os.getcwd(), 'data', 'rocstory', '{}_setlen_{}_batch_{}_cv.pb'.format(
            cfg['language_model'], cfg['max_sentence_length'],
            cfg['batch_size']))

    # add dirs if necessary
    add_needed_folders()

    # read model configs
    with open(cfg['model_configs_path'], 'r', encoding='UTF-8') as f:
        model_configs = json.load(f)

    for model_cfg in model_configs['models']:

        # track duration
        time_start = datetime.now(timezone.utc)

        # unpack the model params
        cfg = unpack_model_config(model_cfg, cfg)

        # handle gpu/cpu
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        log.info('EXPERIMENT run started, name: {} id: {}'.format(
            cfg['run_full_name'], cfg['run_id']))
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

        # prepare tokenizer
        log.info('Loading the proper tokenizer for language model: {}...'.
                 format(cfg['language_model']))
        tokenizer = get_sentence_ordering_tokenizer(cfg['language_model'])

        # data
        if cfg['generate_new_data']:
            # new dataloaders
            log.info('Obtaining test data ...')
            test_dataloader = get_sentence_ordering_dataloader('test', log, cfg,
                                                               tokenizer)
            # check batch
            log.info('Checking test dataloader ...')
            a_batch = next(iter(test_dataloader))
            decode_sentence_batch(a_batch, tokenizer, cfg['dataset_name'], log,
                                  10)

            log.info('Obtaining validation data ...')
            cv_dataloader = get_sentence_ordering_dataloader('validation',
                                                                log,
                                                                cfg,
                                                                tokenizer)

            log.info('Obtaining train data ...')
            train_dataloader = get_sentence_ordering_dataloader('train', log,
                                                                cfg,
                                                                tokenizer)
            # save dataloaders
            log.info('Data persistance (saving dataloaders)')
            torch.save(train_dataloader, cfg['train_dataset_path'])
            torch.save(test_dataloader, cfg['test_dataset_path'])
            torch.save(cv_dataloader, cfg['cv_dataset_path'])

        # reload dataloaders
        log.info('Data (re-)loading')
        train_dataloader = torch.load(cfg['train_dataset_path'])
        test_dataloader = torch.load(cfg['test_dataset_path'])
        cv_dataloader = torch.load(cfg['cv_dataset_path'])

        # model
        model = get_model(cfg)
        model.to(device)
        log.info('Model architecture')
        log.info(model)

        # freeze the BERT layers
        for param in model.sentence_embedder.parameters():
            param.requires_grad = False

        # loss
        CCE = torch.nn.CrossEntropyLoss()

        # optimizer
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg['learning_rate'])

        # log params
        num_params, params_string = count_params(model, return_string=True)
        log.info(params_string)

        # sample prediction prior to training
        log.info('Single prediction (before training) ...')
        examine_single_sentence_order_prediction(
            model, test_dataloader, tokenizer, cfg['dataset_name'], log,
            display_func=decode_sentence_batch)

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
            None,
            None,
            logger=log,
            tqdm_file=
            logging.root.handlers[
                0].stream,
            run_tests_function=run_sentence_ordering_tests,
            allow_gpu=True,
            is_futurehistory=cfg['permute_module_type'] == 'futurehistory',  # check if it's future-history
            is_sentence_ordering=True,
            report_every_n_batches=cfg['report_every_n_batches'],
            validate_every_n_epochs=cfg['validate_every_n_epochs'],
            save_model_every_n_epochs=cfg['save_model_every_n_epochs']
        )

        # sample prediction after training
        model.eval()
        log.info('Single prediction (after training) ...')
        examine_single_sentence_order_prediction(
            model, test_dataloader, tokenizer, cfg['dataset_name'], log,
            display_func=decode_sentence_batch)

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
                dataset_train = cfg['experiment_name']
                dataset_test = cfg['experiment_name']
                epochs = None
                model = None
                model_id = None
                masking = None
                final_training_loss = None
                optimizer = None
                db_name = None
                db_url = None
                num_params = None
                language_model = cfg['language_model']
                cfg = cfg  # needed


            # experiment run function
            @ex.main
            def run_model_tests():
                # run all tests, get all results
                result_general, result_tau, result_spearman, rank_valid_perc = run_sentence_ordering_tests(
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

            config_update = {
                'run_id_own': cfg['run_id'],
                'run_full_name': cfg['run_full_name'],
                'learning_rate': cfg['learning_rate'],
                'epochs': cfg['num_epochs'],
                'model': current_tested_model.__class__.__name__,
                'model_id': cfg['run_id'],
                'masking': cfg['permute_module_masking'],
                'final_training_loss': round(current_last_loss.item(), 10),
                'optimizer': current_optimizer.__class__.__name__,
                'db_name': cfg['db_name'],
                'db_url': cfg['db_url'],
                'num_params': num_params,
                'cfg': cfg, }

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
