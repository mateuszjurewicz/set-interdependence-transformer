"""
Any and all useful abstractions, not directly fitting in other helper categories,
such as models or data generation.
"""
import tqdm
import uuid
import logging
import sys
import pprint
import os


class TqdmStream(object):
    file = None

    def __init__(self, file):
        self.file = file

    def write(self, x):
        if len(x.rstrip()) > 0:
            tqdm.tqdm.write(x, file=self.file, end='')

    def flush(self):
        return getattr(self.file, 'flush', lambda: None)()


def get_run_id(max_length=10):
    """
    Get a unique string id for an experiment run.
    :param max_length: id length (in characters), default 10
    :return: the string id
    """
    run_id = str(uuid.uuid4().fields[-1])[:max_length]
    return run_id


def count_params(model, return_string=False):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params = '{:,}'.format(params)
    if return_string:
        return params, 'The model has {} trainable parameters'.format(params)
    else:
        print('The model has {} trainable parameters'.format(params))


def get_logger(config):
    """Get a basic logger that can handle tqdm"""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=config['log_level'],
        format='\n%(asctime)s | %(levelname)s | %(message)s\n',
        handlers=[
            logging.FileHandler(filename=config['logs_path'], encoding='utf-8'),
            # log file
            logging.StreamHandler(sys.stdout)  # stdout
        ])
    log = logging.getLogger(__name__)
    logging.root.handlers[0].stream = TqdmStream(
        logging.root.handlers[0].stream)
    return log


def get_run_full_name(config):
    """Get a single string representing full experiment run"""
    r = '_'.join([config['experiment_name'], config['model_type'], 'mask',
                  str(config['permute_module_masking']), 'train', config['dataset_train'],
                  'test', config['dataset_test'], 'id', config['run_id']])
    return r


def get_procat_run_full_name(config):
    """Get a single string representing full PROCAT experiment run"""
    r = '_'.join([config['experiment_name'], config['model_type'], 'mask',
                  str(config['masking']), 'id', config['run_id']])
    return r


def get_synthetic_run_full_name(config):
    """Get a single string representing full synthetic experiment run"""
    r = '_'.join([config['experiment_name'], config['model_type'], 'mask',
                  str(config['permute_module_masking']), 'train', config['train_dataset_name'],
                  'test', config['test_dataset_name'], 'id', config['run_id']])
    return r


def get_sentence_ordering_run_full_name(config):
    """Get a single string representing full sentence ordering experiment run"""
    r = '_'.join([config['experiment_name'], config['language_model'],
                  config['model_type'], 'mask', str(config['masking']),
                  'id', config['run_id']])
    return r


def get_model_full_name(config):
    """Get a model's full name, based on training"""
    r = '_'.join([config['experiment_name'], config['model_type'], 'mask',
                  str(config['permute_module_masking']), 'train', config['dataset_train'],
                  'id', config['run_id']])
    return r


def get_procat_model_full_name(config):
    """Get a model's full name, based on training"""
    r = '_'.join([config['experiment_name'], config['model_type'], 'mask',
                  str(config['masking']), 'id', config['run_id']])
    return r


def get_synthetic_model_full_name(config):
    """Get a synthetic model's full name, based on training"""
    r = '_'.join([config['experiment_name'], config['model_type'], 'mask',
                  str(config['permute_module_masking']), 'train', config['train_dataset_name'],
                  'id', config['run_id']])
    return r


def get_sentence_ordering_model_full_name(config):
    """Get a sentence ordering model's full name, based on training"""
    r = '_'.join([config['experiment_name'], config['language_model'],
                  config['model_type'], 'mask', str(config['masking']),
                  'id', config['run_id']])
    return r


def get_test_full_name(config):
    """Get a single string representing full experiment run"""
    r = '_'.join(['testlog', 'model_id', config['model_id'],
                  'test', config['dataset_test'], 'id', config['test_id']])
    return r


def get_procat_test_full_name(config):
    """Get a single string representing full experiment run"""
    r = '_'.join(['testlog', 'model_id', config['model_id'],
                  'test', config['test_name'], 'id', config['test_id']])
    return r


def get_sentence_ordering_test_full_name(config):
    """Get a single string representing full experiment run"""
    r = '_'.join(['testlog', 'model_id', config['model_id'],
                  'test', config['test_name'], 'id', config['test_id']])
    return r


def get_synthetic_test_full_name(config):
    """Get a single string representing full experiment run"""
    r = '_'.join(['testlog', 'model_id', config['model_id'],
                  'test', config['test_dataset_name'], 'id', config['test_id']])
    return r


def get_config_readable(config):
    """Get a human-readable version of config for logs"""
    pp = pprint.PrettyPrinter(indent=4)
    config_readable = pp.pformat(config)
    return config_readable


def add_needed_folders():
    """
    Check if the folders for storing logs, plots and models are present.
    If not, create them.
    """
    needed_dirs = ['logs', 'models', 'plots']
    for path in needed_dirs:
        if not os.path.isdir(path):
            os.mkdir(path)

