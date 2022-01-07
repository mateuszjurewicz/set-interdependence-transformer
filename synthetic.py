"""
Basic script for running a full experiment on the synthetic structures.
Should include a unique id for each run, a choice of model, full logging
(including training progress and evaluation metrics in e.g. Sacred), model
persistance and possibly generation of plots.
"""
import json
import logging
import os
import random
from datetime import datetime
from datetime import timezone

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
# sacred experiment
from sacred import Experiment
from sacred.observers import MongoObserver

from metrics import run_synthetic_tests, plot_losses, examine_single_prediction
from models import unpack_model_config, get_model, train_epochs
# custom imports
from utils import get_run_id, count_params, get_logger, \
    get_synthetic_run_full_name, get_synthetic_model_full_name, get_config_readable, \
    add_needed_folders
from utils_data import get_synthetic_dataloader, get_example, restore_synthetic_catalog

# torch data

if __name__ == "__main__":

    # configuration
    cfg = dict()

    # cfg experiment
    cfg['seed'] = random.randint(0, 10000)
    cfg['experiment_name'] = 'synthetic'
    cfg['generate_new_data'] = True

    # cfg model path
    cfg['model_configs_path'] = os.path.join('model_configs', 'synthetic.json')

    # reporting and model persisting
    cfg['report_every_n_batches'] = 50
    cfg['validate_every_n_epochs'] = 10
    cfg['save_model_every_n_epochs'] = 25

    # cfg metrics tracking
    cfg['log_level'] = logging.INFO
    cfg['track_metrics'] = True
    cfg['db_name'] = 'sacred'
    cfg['db_url'] = 'localhost:27017'

    # cfg data
    cfg['train_n_catalogs'] = 10000
    cfg['test_n_catalogs'] = 5000
    cfg['cv_n_catalogs'] = 3000  # test and cv need different sizes
    cfg['train_n_tokens'] = 30
    cfg['train_n_offers'] = 25
    cfg['max_pages'] = 20
    cfg['min_pages'] = 5
    cfg['batch_size'] = 32
    cfg['n_mset_shuffles'] = 1
    cfg['train_offer_distributions'] = {
        'b': 100,
        'r': 100,
        'y': 100,
        'g': 5,
        'p': 3}
    cfg['word2idx'] = {'b': 0, 'r': 1, 'y': 2, 'g': 3, 'p': 4,
                       'catalog_start': 5, 'page_break': 6, 'catalog_end': 7}
    cfg['idx2word'] = {v: k for k, v in cfg['word2idx'].items()}

    # same for test
    cfg['test_n_tokens'] = cfg['train_n_tokens']
    cfg['test_n_offers'] = cfg['train_n_offers']
    cfg['test_offer_distributions'] = cfg['train_offer_distributions']

    # same for cv
    cfg['cv_n_tokens'] = cfg['train_n_tokens']
    cfg['cv_n_offers'] = cfg['train_n_offers']
    cfg['cv_offer_distributions'] = cfg['train_offer_distributions']

    # names for paths
    cfg['train_dataset_name'] = 'c{}_ntok{}_noff{}_maxp{}_minp{}'.format(
        cfg['train_n_catalogs'],
        cfg['train_n_tokens'],
        cfg['train_n_offers'],
        cfg['max_pages'],
        cfg['min_pages'])
    cfg['test_dataset_name'] = 'c{}_ntok{}_noff{}_maxp{}_minp{}'.format(
        cfg['test_n_catalogs'],
        cfg['test_n_tokens'],
        cfg['test_n_offers'],
        cfg['max_pages'],
        cfg['min_pages'])
    cfg['cv_dataset_name'] = 'c{}_ntok{}_noff{}_maxp{}_minp{}'.format(
        cfg['cv_n_catalogs'],
        cfg['cv_n_tokens'],
        cfg['cv_n_offers'],
        cfg['max_pages'],
        cfg['min_pages'])

    # actual paths
    cfg['train_dataset_path'] = os.path.join(os.getcwd(), 'data',
                                             '{}_c{}'.format(
                                                 cfg['experiment_name'],
                                                 cfg['train_dataset_name']))
    cfg['test_dataset_path'] = os.path.join(os.getcwd(), 'data',
                                            '{}_c{}'.format(
                                                cfg['experiment_name'],
                                                cfg['test_dataset_name']))
    cfg['cv_dataset_path'] = os.path.join(os.getcwd(), 'data',
                                          '{}_c{}'.format(
                                              cfg['experiment_name'],
                                              cfg['test_dataset_name']))

    cfg['x_name'] = 0
    cfg['y_name'] = 1

    # catalog-level rules
    basic_catalog_rules = [
        {'name': 'first_page',
         'spec': 'all_red'
         },
        {'name': 'last_page',
         'spec': 'all_blue'
         },
        {'name': 'max_pages',
         'spec': cfg['max_pages']},
        {'name': 'min_pages',
         'spec': cfg['min_pages']}
    ]

    green_only_catalog_rules = [
        {'name': 'first_page',
         'spec': 'all_red'
         },
        {'name': 'last_page',
         'spec': 'all_blue'
         },
        {'name': 'max_pages',
         'spec': cfg['max_pages']},
        {'name': 'min_pages',
         'spec': cfg['min_pages']}
    ]

    purple_only_catalog_rules = [
        {'name': 'first_page',
         'spec': 'all_blue'
         },
        {'name': 'last_page',
         'spec': 'all_blue'
         },
        {'name': 'max_pages',
         'spec': cfg['max_pages']},
        {'name': 'min_pages',
         'spec': cfg['min_pages']}
    ]

    purple_and_green_catalog_rules = [
        {'name': 'first_page',
         'spec': 'all_red'
         },
        {'name': 'last_page',
         'spec': 'all_purple'
         },
        {'name': 'max_pages',
         'spec': cfg['max_pages']},
        {'name': 'min_pages',
         'spec': cfg['min_pages']}
    ]

    # valid section rules
    basic_valid_sections = [
        {'name': 'all_red',
         'mixed': False,
         'allowed_offer_types': ['r'],
         'offers_per_section': [4, 3, 2, 1],
         },
        {'name': 'all_blue',
         'mixed': False,
         'allowed_offer_types': ['b'],
         'offers_per_section': [4, 3, 2, 1]
         },
        {'name': 'all_yellow',
         'mixed': False,
         'allowed_offer_types': ['y'],
         'offers_per_section': [4, 3, 2, 1],
         'min_offers': 1
         },
        {'name': 'mix_red_yellow',
         'mixed': True,
         'allowed_offer_types': ['r', 'y'],
         'offers_per_section': [4, 2],
         'mix_ratio': [0.5, 0.5]
         }
    ]

    green_only_valid_sections = [
        {'name': 'all_red',
         'mixed': False,
         'allowed_offer_types': ['r'],
         'offers_per_section': [4, 3, 2, 1],
         },
        {'name': 'all_blue',
         'mixed': False,
         'allowed_offer_types': ['b'],
         'offers_per_section': [4, 3, 2, 1]
         },
        {'name': 'all_yellow',
         'mixed': False,
         'allowed_offer_types': ['y'],
         'offers_per_section': [4, 3, 2, 1],
         'min_offers': 1
         },
        {'name': 'mix_green_red',
         'mixed': True,
         'allowed_offer_types': ['r', 'g'],
         'offers_per_section': [4, 2],
         'mix_ratio': [0.5, 0.5]
         }
    ]

    purple_only_valid_sections = [
        {'name': 'all_red',
         'mixed': False,
         'allowed_offer_types': ['r'],
         'offers_per_section': [4, 3, 2, 1],
         },
        {'name': 'all_blue',
         'mixed': False,
         'allowed_offer_types': ['b'],
         'offers_per_section': [4, 3, 2, 1]
         },
        {'name': 'all_yellow',
         'mixed': False,
         'allowed_offer_types': ['y'],
         'offers_per_section': [4, 3, 2, 1],
         'min_offers': 1
         },
        {'name': 'mix_red_yellow',
         'mixed': True,
         'allowed_offer_types': ['r', 'y'],
         'offers_per_section': [4, 2],
         'mix_ratio': [0.5, 0.5]
         },
        {'name': 'all_purple',
         'mixed': False,
         'allowed_offer_types': ['p'],
         'offers_per_section': [4, 3, 2, 1],
         }
    ]

    purple_and_green_valid_sections = [
        {'name': 'all_red',
         'mixed': False,
         'allowed_offer_types': ['r'],
         'offers_per_section': [4, 3, 2, 1],
         },
        {'name': 'all_blue',
         'mixed': False,
         'allowed_offer_types': ['b'],
         'offers_per_section': [4, 3, 2, 1]
         },
        {'name': 'all_yellow',
         'mixed': False,
         'allowed_offer_types': ['y'],
         'offers_per_section': [4, 3, 2, 1],
         'min_offers': 1
         },
        {'name': 'mix_red_yellow',
         'mixed': True,
         'allowed_offer_types': ['r', 'y'],
         'offers_per_section': [4, 2],
         'mix_ratio': [0.5, 0.5]
         },
        {'name': 'all_purple',
         'mixed': False,
         'allowed_offer_types': ['p'],
         'offers_per_section': [4, 3, 2, 1],
         },
        {'name': 'mix_green_red',
         'mixed': True,
         'allowed_offer_types': ['r', 'g'],
         'offers_per_section': [4, 2],
         'mix_ratio': [0.5, 0.5]
         }
    ]

    # full rules, jointly
    rules = {'basic':
                 {'catalog_rules': basic_catalog_rules,
                  'valid_sections': basic_valid_sections},
             'green_only':
                 {'catalog_rules': green_only_catalog_rules,
                  'valid_sections': green_only_valid_sections},
             'purple_only':
                 {'catalog_rules': purple_only_catalog_rules,
                  'valid_sections': purple_only_valid_sections},
             'purple_and_green':
                 {'catalog_rules': purple_and_green_catalog_rules,
                  'valid_sections': purple_and_green_valid_sections}
             }

    cfg['rules'] = rules

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
        cfg['run_full_name'] = get_synthetic_run_full_name(cfg)
        cfg['model_path'] = 'models/{}'.format(get_synthetic_model_full_name(cfg))
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
            log.info('Generating data ...')
            train_dataloader = get_synthetic_dataloader('train', log, cfg)
            test_dataloader = get_synthetic_dataloader('test', log, cfg)
            cv_dataloader = get_synthetic_dataloader('cv', log, cfg)

            # data persistance
            log.info('Data persistance (saving dataloaders)')
            torch.save(train_dataloader, cfg['train_dataset_path'])
            torch.save(test_dataloader, cfg['test_dataset_path'])
            torch.save(test_dataloader, cfg['cv_dataset_path'])

        # data loading
        log.info('Data (re-)loading')
        train_dataloader = torch.load(cfg['train_dataset_path'])
        test_dataloader = torch.load(cfg['test_dataset_path'])
        cv_dataloader = torch.load(cfg['cv_dataset_path'])

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
        single_x, single_y = get_example(test_dataloader, as_batch=True,
                                         x_name=cfg['x_name'],
                                         y_name=cfg['y_name'])
        examine_single_prediction(model, single_x, single_y, log,
                                  restore_synthetic_catalog)

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
            run_tests_function=run_synthetic_tests,
            allow_gpu=True,
            is_futurehistory=cfg['permute_module_type'] == 'futurehistory',  # check if it's future-history
            report_every_n_batches=cfg['report_every_n_batches'],
            validate_every_n_epochs=cfg['validate_every_n_epochs'],
            save_model_every_n_epochs=cfg['save_model_every_n_epochs']
        )

        # predict on sample (post training)
        log.info('Single prediction (after training) ... ')
        single_x, single_y = get_example(test_dataloader, as_batch=True,
                                         x_name=cfg['x_name'],
                                         y_name=cfg['y_name'])
        examine_single_prediction(model, single_x, single_y, log,
                                  restore_synthetic_catalog)

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
                dataset_train = cfg['train_dataset_name']
                dataset_test = cfg['test_dataset_name']
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
                result_general, result_tau, result_spearman, rank_valid_perc, catalogs_testable, catalogs_corrent_n_tokens, \
                catalogs_num_offers_match, valid_sections_avg, \
                valid_sections_avg_basic, valid_sections_avg_green_only, \
                valid_sections_avg_purple_only, valid_sections_avg_purple_and_green, \
                valid_struture_avg, valid_struture_avg_basic, \
                valid_struture_avg_green_only, valid_struture_avg_purple_only, \
                valid_struture_avg_purple_and_green = run_synthetic_tests(
                    current_tested_model, test_dataloader, log, cfg)

                # log params and results
                log_experiment_results(result_general, result_tau,
                                       result_spearman, rank_valid_perc,
                                       catalogs_testable,
                                       catalogs_corrent_n_tokens,
                                       catalogs_num_offers_match,
                                       valid_sections_avg,
                                       valid_sections_avg_basic,
                                       valid_sections_avg_green_only,
                                       valid_sections_avg_purple_only,
                                       valid_sections_avg_purple_and_green,
                                       valid_struture_avg,
                                       valid_struture_avg_basic,
                                       valid_struture_avg_green_only,
                                       valid_struture_avg_purple_only,
                                       valid_struture_avg_purple_and_green)

                return round(result_general, 5)


            def log_experiment_results(r_general, r_tau, r_spearman,
                                       rank_valid_perc, catalogs_testable,
                                       catalogs_corrent_n_tokens,
                                       catalogs_num_offers_match,
                                       valid_sections_avg,
                                       valid_sections_avg_basic,
                                       valid_sections_avg_green_only,
                                       valid_sections_avg_purple_only,
                                       valid_sections_avg_purple_and_green,
                                       valid_struture_avg,
                                       valid_struture_avg_basic,
                                       valid_struture_avg_green_only,
                                       valid_struture_avg_purple_only,
                                       valid_struture_avg_purple_and_green):

                # round if not none
                if r_tau:
                    r_tau = round(r_tau, 5)
                if r_spearman:
                    r_spearman = round(r_spearman, 5)

                # log the results
                ex.log_scalar('test.general', r_general)
                ex.log_scalar('test.rank_correlation_valid_perc',
                              rank_valid_perc)
                ex.log_scalar('test.tau', r_tau, 5)
                ex.log_scalar('test.rho', r_spearman, 5)

                # general
                ex.log_scalar('test.gen_c_testable', catalogs_testable)
                ex.log_scalar('test.gen_c_corr_n_tokens',
                              catalogs_corrent_n_tokens)
                ex.log_scalar('test.gen_c_num_off_match',
                              catalogs_num_offers_match)

                # sections
                ex.log_scalar('test.sec_valid_s_avg', valid_sections_avg)
                ex.log_scalar('test.sec_valid_basic', valid_sections_avg_basic)
                ex.log_scalar('test.sec_valid_green',
                              valid_sections_avg_green_only)
                ex.log_scalar('test.sec_valid_purple',
                              valid_sections_avg_purple_only)
                ex.log_scalar('test.sec_valid_p_and_g',
                              valid_sections_avg_purple_and_green)

                # structural
                ex.log_scalar('test.struct_valid_avg', valid_struture_avg)
                ex.log_scalar('test.struct_avg_basic', valid_struture_avg_basic)
                ex.log_scalar('test.struct_avg_green',
                              valid_struture_avg_green_only)
                ex.log_scalar('test.struct_avg_purple',
                              valid_struture_avg_purple_only)
                ex.log_scalar('test.struct_avg_p_and_g',
                              valid_struture_avg_purple_and_green)


            # experiment config
            current_tested_model = model
            current_optimizer = optimizer
            current_last_loss = last_loss

            # find nested dicts, show then, turn their values to strings, preserve them for later re-insertion
            config_safe_copy = dict()
            for k, v in cfg.items():
                if type(v) == dict:
                    print(k)
                    config_safe_copy[k] = str(v)
                else:
                    config_safe_copy[k] = v

            config_update = {
                'run_id_own': cfg['run_id'],
                'run_full_name': cfg['run_full_name'],
                'learning_rate': cfg['learning_rate'],
                'epochs': cfg['num_epochs'],
                'model': cfg['model_type'],
                'model_id': cfg['run_id'],
                'masking': cfg['permute_module_masking'],
                'final_training_loss': round(current_last_loss.item(), 10),
                'optimizer': current_optimizer.__class__.__name__,
                'db_name': cfg['db_name'],
                'db_url': cfg['db_url'],
                'num_params': num_params,
                'cfg': config_safe_copy}

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
