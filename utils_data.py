"""
Script for all utilities regarding data & its generation.
"""
import ast
import copy
import csv
import datasets
import itertools
import numpy as np
import random
import statistics
import torch
import logging

from scipy.spatial import distance
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, Dataset
from transformers import BertTokenizer, AutoTokenizer
import matplotlib.pyplot as plt


class Brackets:

    def __init__(self, type):
        self.status = 'unopened'
        self.type = type

        if type == 'parenthesis':
            self.left = '('
            self.right = ')'
        elif type == 'curly':
            self.left = '{'
            self.right = '}'

        self.symbols = [self.left, self.right]

    def open(self):
        self.symbols.remove(self.left)
        self.status = 'opened'
        return self.left

    def close(self):
        self.symbols.remove(self.right)
        self.status = 'closed'
        return self.right

    def get_symbol(self):
        if self.status == 'unopened':
            return self.open()
        elif self.status == 'opened':
            return self.close()
        elif self.status == 'closed':
            return None
        else:
            raise Exception('Run out of states!')

    def get_status(self):
        return self.status

    def __repr__(self):
        return str(self.type + '_' + self.status)


class Offer:
    # get unique id, increment counter
    of_id_counter = 1

    def __init__(self, color):
        self.text = color
        self.image = color

        self.id = 'of_{}_{}{}'.format(self.text[0],
                                      '0' * (6 - len(str(self.of_id_counter))),
                                      str(self.of_id_counter))
        self.of_id_counter += 1

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text

    @classmethod
    def _multiple(cls, *colors):
        # Create multiple instances of object `Offer`
        return [Offer(c) for c in colors]


class Page:
    # get unique id, increment counter
    pa_id_counter = 1

    def __init__(self, offers, page_num):
        self.offers = offers
        self.number = page_num
        self.num_offers = len(offers)

        # get unique id, increment counter
        self.id = 'pa_{}{}'.format('0' * (6 - len(str(self.pa_id_counter))),
                                   str(self.pa_id_counter))
        self.pa_id_counter += 1

    def __str__(self):
        s = 'Page nr {}, {} offers: {}'.format(self.number, self.num_offers,
                                               self.offers)
        return s


class Catalog:
    # get unique id, increment counter
    ca_id_counter = 1

    def __init__(self, word2idx, pages):
        self.word2idx = word2idx
        self.pages = pages
        self.num_pages = len(pages)
        self.sequence_readable = self.as_sequence()
        self.sequence_one_hot = self.one_hot_encode()
        self.sequence_indices = self.as_indices()

        # get unique id, increment counter
        self.id = 'ca_{}{}'.format('0' * (6 - len(str(self.ca_id_counter))),
                                   str(self.ca_id_counter))
        self.ca_id_counter += 1

    def __str__(self):
        s = 'Catalog id: {}, {} pages.'.format(self.id, self.num_pages)
        return s

    def show_all_pages(self):
        s = ''
        for page in self.pages:
            print(page)

    def as_sequence(self):
        """
        For the generative task I want the catalog to be in linear form of:
        [catalog_start, (R), (R), (Y), (Y), page_break , ... , page_break, (B)x4, catalog_end]
        This will differ based on the chosen synthetic data features.
        I will need 3 special tokens: catalog_start, page_break, catalog_end.
        """
        catalog_as_sequence = []

        # first append the one-hot encoded catalog_start
        catalog_as_sequence.append('catalog_start')

        # process all pages
        for page in self.pages:

            # every offer
            for offer in page.offers:
                catalog_as_sequence.append(str(offer))

            # page ends in page break
            catalog_as_sequence.append('page_break')

        # catalog ends with a marker (replacing the preceeding page break)
        catalog_as_sequence[-1] = 'catalog_end'

        return catalog_as_sequence

    def one_hot_encode(self):
        """
        I wanted both the human-readable sequence and the one-hot encoded version for the model input.
        """
        catalog_as_sequence_one_hot = []

        for token in self.sequence_readable:
            catalog_as_sequence_one_hot.append(
                to_categorical(self.word2idx[token], len(self.word2idx)))

        return catalog_as_sequence_one_hot

    def as_indices(self):
        """
        I also need a catalog as a sequence of indices from the word2idx.
        """
        catalog_as_sequence_of_indices = []

        for token in self.sequence_readable:
            catalog_as_sequence_of_indices.append(self.word2idx[token])

        return catalog_as_sequence_of_indices


def tsp_opt(points):
    """
    Dynamic programing solution for TSP - O(2^n*n^2)
    https://gist.github.com/mlalevic/6222750
    :param points: List of (x, y) points
    :return: Optimal solution
    """

    def length(x_coord, y_coord):
        return np.linalg.norm(np.asarray(x_coord) - np.asarray(y_coord))

    # Calculate all lengths
    all_distances = [[length(x, y) for y in points] for x in points]
    # Initial value - just distance from 0 to every other point + keep the track of edges
    A = {(frozenset([0, idx+1]), idx+1): (dist, [0, idx+1]) for idx, dist in enumerate(all_distances[0][1:])}
    cnt = len(points)
    for m in range(2, cnt):
        B = {}
        for S in [frozenset(C) | {0} for C in itertools.combinations(range(1, cnt), m)]:
            for j in S - {0}:
                # This will use 0th index of tuple for ordering, the same as if key=itemgetter(0) used
                B[(S, j)] = min([(A[(S-{j}, k)][0] + all_distances[k][j], A[(S-{j}, k)][1] + [j])
                                 for k in S if k != 0 and k != j])
        A = B
    res = min([(A[d][0] + all_distances[0][d[1]], A[d][1]) for d in iter(A)])
    return np.asarray(res[1])


def connectpoints(x,y,p1,p2):
    x1, x2 = x[p1], x[p2]
    y1, y2 = y[p1], y[p2]
    plt.plot([x1,x2],[y1,y2],'k-')


def show_tsp(points, solution, run_id=None, tour_length_predicted=None, tour_length_target=None):
    """
    Function for presenting points and viewing solutions to the TSP problem.
    Ideally it should show the connections for illegal tours too.
    """
    # set figsize
    plt.rcParams['figure.figsize'] = [7, 7]

    # keep x and y scale same
    plt.axis('scaled')

    # add first point for a full return journey
    points = torch.cat((points, points[0].unsqueeze(0)), 0)
    solution = torch.cat((solution, solution[0].unsqueeze(0)), 0)

    # convert to numpy
    points = points.numpy()
    solution = solution.numpy()

    # order them according to solutions
    # (using numpy ordering)
    ordered_points = points[solution]

    # extract x and y from ordered points and points separately
    # to plot even illegal orders that ommit points
    xs = points[:, 0]
    ys = points[:, 1]
    con_xs = ordered_points[:, 0]
    con_ys = ordered_points[:, 1]

    # plot points first
    plt.scatter(xs, ys)

    # plot connecting lines
    for i in range(0,len(con_xs)-1):
        connectpoints(con_xs,con_ys,i,i+1)

    # add tour length calculation(s)
    if tour_length_predicted and tour_length_target:
        plt.xlabel('Tour Lengths\nPredicted: {:.4f}\nTarget: {:.4f}'.format(tour_length_predicted,
                                                                            tour_length_target), fontsize=10)
    elif tour_length_predicted:
        plt.xlabel('Tour Length Predicted: {:.4f}'.format(tour_length_predicted), fontsize=10)

    # save it
    plt.title('Predicted tour')
    plt.savefig('plots/tsp_tour_{}.png'.format(run_id))

    # or show it
    # plt.show


def get_tour_length(points, solution):
    """
    Calculate the length of the tour between points, given their order provided in the solution.
    """
    tour_length = 0

    # add first point for a full return journey
    points = torch.cat((points, points[0].unsqueeze(0)), 0)
    solution = torch.cat((solution, solution[0].unsqueeze(0)), 0)

    # convert to numpy
    points = points.numpy()
    solution = solution.numpy()

    # order them according to solutions
    # (using numpy ordering)
    ordered_points = points[solution]

    # calculate addiitive distance
    for i in range(len(ordered_points)):

        # end condition
        if i + 1 == len(ordered_points):
            break
        else:
            tour_length += distance.euclidean(ordered_points[i], ordered_points[i+1])

    return tour_length


def held_karp(dists):
    """
    Implementation of Held-Karp, an algorithm that solves the Traveling
    Salesman Problem using dynamic programming with memoization.
    Parameters:
        dists: distance matrix
    Returns:
        A tuple, (cost, path).
    """
    n = len(dists)

    # Maps each subset of the nodes to the cost to reach that subset, as well
    # as what node it passed before reaching this subset.
    # Node subsets are represented as set bits.
    C = {}

    # Set transition cost from initial state
    for k in range(1, n):
        C[(1 << k, k)] = (dists[0][k], 0)

    # Iterate subsets of increasing length and store intermediate results
    # in classic dynamic programming manner
    for subset_size in range(2, n):
        for subset in itertools.combinations(range(1, n), subset_size):
            # Set bits for all nodes in this subset
            bits = 0
            for bit in subset:
                bits |= 1 << bit

            # Find the lowest cost to get to this subset
            for k in subset:
                prev = bits & ~(1 << k)

                res = []
                for m in subset:
                    if m == 0 or m == k:
                        continue
                    res.append((C[(prev, m)][0] + dists[m][k], m))
                C[(bits, k)] = min(res)

    # We're interested in all bits but the least significant (the start state)
    bits = (2**n - 1) - 1

    # Calculate optimal cost
    res = []
    for k in range(1, n):
        res.append((C[(bits, k)][0] + dists[k][0], k))
    opt, parent = min(res)

    # Backtrack to find full path
    path = []
    for i in range(n - 1):
        path.append(parent)
        new_bits = bits & ~(1 << parent)
        _, parent = C[(bits, parent)]
        bits = new_bits

    # Add implicit start state
    path.append(0)

    return opt, list(reversed(path))


def is_path_legal(solution):
    """
    Take a solution (order of town indices to visit) and return Boolean legality:
      - no repeated visits
      - no town skipped
    These are equivalent, as the model can only skip a town by pointing to an already visited one.
    """

    # check no duplicates
    solution = solution.numpy()
    set_solution = set(solution)

    if len(solution) != len(set_solution):
        return False
    else:
        return True

    # to test:
    # legal_s = torch.tensor([0, 1, 2, 3, 4, 5])
    # is_path_legal(legal_s)
    # illegaL_s = torch.tensor([0, 1, 2, 3, 3, 5])
    # is_path_legal(illegaL_s)


class TSPDataset(Dataset):
    """
    Random TSP dataset
    """

    def __init__(self, data_size, seq_len, solver=tsp_opt, solve=True):
        self.data_size = data_size
        self.seq_len = seq_len
        self.solve = solve
        self.solver = solver
        self.data = self._generate_data()

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        tensor = torch.from_numpy(self.data['X'][idx]).float()
        solution = torch.from_numpy(self.data['Y'][idx]).long() if self.solve else None

        sample = {'X': tensor, 'Y': solution}

        return sample

    def _generate_data(self):
        """
        :return: Set of points_list and their One-Hot vector solutions
        """
        points_list = []
        solutions = []
        data_iter = tqdm(range(self.data_size), unit='data')
        for i, _ in enumerate(data_iter):
            data_iter.set_description('Data points %i/%i' % (i+1, self.data_size))
            points_list.append(np.random.random((self.seq_len, 2)))
        solutions_iter = tqdm(points_list, unit='solve')
        if self.solve:
            for i, x in enumerate(solutions_iter):
                solutions_iter.set_description('Solved %i/%i' % (i+1, len(points_list)))
                y = self.solver(x)
                idx_of_second = y[1]
                x_prim = np.delete(x, idx_of_second, 0)
                second_element = x[idx_of_second]
                x_prim = np.insert(x_prim, 1, second_element, axis=0)
                y_final = self.solver(x_prim)
                solutions.append(y_final)
                points_list[i] = x_prim
        else:
            solutions = None

        return {'X': points_list, 'Y': solutions}

    def _to1hotvec(self, points):
        """
        :param points: List of integers representing the points indexes
        :return: Matrix of One-Hot vectors
        """
        vec = np.zeros((len(points), self.seq_len))
        for i, v in enumerate(vec):
            v[points[i]] = 1

        return vec


def to_categorical(y, num_classes=None, dtype='float32'):
    """
    Copy of keras.to_categorical().
    """

    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def get_example(a_dataloader, as_batch=True, x_name='X', y_name='Y'):
    """
    Take a torch dataloader and get a single example.
    """
    # handle gpu/cpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    a_batch = next(iter(a_dataloader))

    example_points = a_batch[x_name][0]
    example_solution = a_batch[y_name][0]

    if as_batch:
        example_points = example_points.unsqueeze(0)
        example_solution = example_solution.unsqueeze(0)

        # handle gpu/cpu
        example_points = example_points.to(device)
        example_solution = example_solution.to(device)

    return example_points, example_solution


def get_batch(a_dataloader, x_name, y_name):
    """
    Get a batch of points and solutions from a torch dataloader.
    """
    a_batch = next(iter(a_dataloader))
    batch_points = a_batch[x_name]
    batch_solutions = a_batch[y_name]

    return batch_points, batch_solutions


def reorder(x, y, as_batch=True):
    """
    Take a 1-elem batch x and reorder according to y
    (either target or prediction).
    """
    r = []
    if torch.is_tensor(x):
        x = x.numpy()
    if torch.is_tensor(y):
        y = y.numpy()

    if as_batch:
        for i in y[0]:
            r.append(x[0][i])
    else:
        for i in y:
            r.append(x[i])
    return r


def sort_solver(floats):
    """
    Take a list of floats and sort them. Return the sorted indices as np array.
    """
    return np.argsort(floats)


class FloatSortDataset(Dataset):
    """
    Random (seedable) integer sorting dataset, with length specified durining initialization.
    We only generate floats between 0 and 1. we don't care about actual 0 ocurring (should be rare)
    nor about duplicates (equally rare).
    """

    def __init__(self, data_size, seq_len, solver=sort_solver, solve=True):
        self.data_size = data_size
        self.seq_len = seq_len
        self.solve = solve
        self.solver = solver
        self.data = self._generate_data()

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        tensor = torch.from_numpy(self.data['X'][idx]).float()
        solution = torch.from_numpy(self.data['Y'][idx]).long()

        sample = {'X': tensor, 'Y': solution}

        return sample

    def _generate_data(self):
        """
        :return: Set of float_list and their vector solutions
        """
        floats_list = []
        solutions = []
        data_iter = tqdm(range(self.data_size), unit='data')
        for i, _ in enumerate(data_iter):
            data_iter.set_description(
                'Data points %i/%i' % (i + 1, self.data_size))
            floats_list.append(np.random.random(self.seq_len))

        solutions_iter = tqdm(floats_list, unit='solve')
        if self.solve:
            # get solutions
            for i, floats in enumerate(solutions_iter):
                solutions_iter.set_description(
                    'Solved %i/%i' % (i + 1, len(floats_list)))
                solution = self.solver(floats)
                solutions.append(solution)
        else:
            solutions = None

        return {'X': floats_list, 'Y': solutions}


def generate_sequences_canonical(n_sequences, seq_length=30, padding='P'):
    """
    Generate lists of symbols forming syntactically correct sentences in the canonical grammar (context-sensitive).
    """
    sequences = []

    for i in range(n_sequences):
        # randomly choose the number of symbols per type (same in this grammar)
        n_symbols = random.randint(1, seq_length // 3)

        # create the start of the sentence
        seq = []
        seq.extend(['a'] * n_symbols)
        seq.extend(['b'] * n_symbols)
        seq.extend(['c'] * n_symbols)

        # pad
        current_length = len(seq)
        n_padding = seq_length - len(seq)
        seq.extend([padding] * n_padding)

        # append
        sequences.append(seq)

    return sequences


def generate_sequences_brackets(n_sequences, seq_length=30, padding='P'):
    """
    Generate lists of symbols forming syntactically correct sentences in the canonical grammar (context-sensitive).
    """
    sequences = []

    for i in range(n_sequences):

        sequence = []

        # randomly choose the number of each bracket types in general
        # two of each have to be in there
        n_paren = 1
        n_curly = 1

        # subtract this many symbols (brackets * 2)
        remaining_length = seq_length - n_paren * 2 - n_curly * 2

        n_paren += random.randint(0, remaining_length // 2)
        remaining_length = remaining_length - n_paren * 2 + 2  # 1 for original
        n_curly += random.randint(0, remaining_length // 2)

        # get a list of total pairs to be used, tracking status
        brackets_to_use = []
        for j in range(n_paren):
            brackets_to_use.append(Brackets('parenthesis'))
        for k in range(n_curly):
            brackets_to_use.append(Brackets('curly'))

        # put all brackets in the unopened list
        brackets_unopened = brackets_to_use[:]
        brackets_opened = []
        brackets_closed = []

        # begin constructing the sequence
        all_closed = False
        while not all_closed:

            # randomly choose between opening and closing a bracket
            open_new_one = random.choice([True, False])

            # open a new bracket, if there is such one
            if open_new_one and len(brackets_unopened) != 0:

                # grab an unopened bracket
                current_idx = random.choice(range(len(brackets_unopened)))
                current_bracket = brackets_unopened[current_idx]

                # get its symbol
                symbol = current_bracket.get_symbol()

                # add it to the sequence
                sequence.append(symbol)

                # move the bracket to the opened ones
                brackets_opened.append(copy.deepcopy(current_bracket))

                # remove it from unopened by index
                del brackets_unopened[current_idx]

            # close an already opened bracket, if there are open ones
            elif len(brackets_opened) != 0:

                # grab the LAST opened one
                current_bracket = brackets_opened[-1]

                # get its symbol
                symbol = current_bracket.get_symbol()

                # add it to the sequence
                sequence.append(symbol)

                # move the bracket to the closed ones
                brackets_closed.append(copy.deepcopy(current_bracket))

                # remove it from opened ones
                brackets_opened.pop()

            # if there's no unopened and opened brackets, we're done
            if len(brackets_unopened) == 0 and len(brackets_opened) == 0:
                all_closed = True

        # pad
        current_length = len(sequence)
        n_padding = seq_length - current_length
        sequence.extend([padding] * n_padding)

        # append
        sequences.append(sequence)

    return sequences


def generate_sequences_harder_context_sensitive(n_sequences, seq_length=30,
                                                padding='P'):
    """
    Generate lists of symbols forming syntactically correct sentences
    in the harder context-sensitive grammar: a^n b^n c^{n*m}
    """
    sequences = []

    for i in range(n_sequences):
        # randomly choose the number of symbols per type
        n_symbols_1 = random.randint(1, (seq_length // 2) - 1)
        n_symbols_2 = random.randint(1, ((seq_length - n_symbols_1)
                                         // (1 + n_symbols_1)))
        n_symbols_c = n_symbols_1 * n_symbols_2

        # randomly decide whether sumbol 1 is a or b
        if random.random() >= 0.5:
            n_symbols_a = n_symbols_1
            n_symbols_b = n_symbols_2
        else:
            n_symbols_b = n_symbols_1
            n_symbols_a = n_symbols_2

        # create the start of the sentence
        seq = []
        seq.extend(['a'] * n_symbols_a)
        seq.extend(['b'] * n_symbols_b)
        seq.extend(['c'] * n_symbols_c)

        # pad
        current_length = len(seq)
        n_padding = seq_length - len(seq)
        seq.extend([padding] * n_padding)

        # append
        sequences.append(seq)

    return sequences


def get_x_y_grammar(sequences, vocabulary):
    """
    Turn a list of sequences into x and y, according to a vocabulary.
    """
    # create shuffled sets from sequences)
    X = []
    Y = []

    for seq in sequences:

        # turn to numbers
        seq_as_numbers = [vocabulary[symbol] for symbol in seq]

        # x and y start as a placeholders
        x = ['n/a' for l in range(len(seq_as_numbers))]
        y = [0 + j for j in range(len(seq_as_numbers))]
        random.shuffle(y)

        for i, e in enumerate(y):
            x[i] = seq_as_numbers[e]

        # y's indices and elements need to be flipped to represent what the model outputs
        final_y = copy.deepcopy(y)
        for i, e in enumerate(y):
            final_y[e] = i

        # append
        X.append(np.asarray(x))
        Y.append(np.asarray(final_y))

    return X, Y


def get_grammar_dataloader(grammar_type, dataset_type, logger, config):
    """Get dataloaders for chosen grammar and dataset type"""

    logger.info('Data generation, confirming correct parameters')

    # error checking
    if grammar_type not in ['canonical', 'brackets', 'harder']:
        raise Exception('Invalid grammar type supplied, try e.g. canonical')
    if dataset_type not in ['train', 'test', 'cv']:
        raise Exception('Incorrect dataset_type, try train or test.')

    # identify the right sequence generation function
    if grammar_type == 'canonical':
        generate_func = generate_sequences_canonical
    elif grammar_type == 'brackets':
        generate_func = generate_sequences_brackets
    elif grammar_type == 'harder':
        generate_func = generate_sequences_harder_context_sensitive
    else:
        generate_func = None

    # handle train vs test sizes
    if dataset_type == 'train':
        logger.info('Data generation, traing set ...')
        sequences = generate_func(
            config['train_size'], config['train_set_size'],
            config['padding_symbol'])
    elif dataset_type == 'test':
        logger.info('Data generation, test set ...')
        sequences = generate_func(
            config['test_size'], config['test_set_size'],
            config['padding_symbol'])
    elif dataset_type == 'cv':
        logger.info('Data generation, cv set ...')
        sequences = generate_func(
            config['cv_size'], config['cv_set_size'],
            config['padding_symbol'])

    # turn to x & y
    logger.info('Data generation, getting X and Y ...')
    X, Y = get_x_y_grammar(sequences, config['word2idx'])
    X = torch.from_numpy(np.asarray(X)).float()
    Y = torch.from_numpy(np.asarray(Y)).long()

    # dataloader
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'],
                            shuffle=True, num_workers=4)

    return dataloader


def get_procat_dataloader(dataset_type, logger, config, tokenizer):
    """
    Create the dataloaders for the procat datasets,
    per language model.
    """
    available_dataset_types = ['train', 'test', 'validation']
    if dataset_type not in available_dataset_types:
        error_msg = 'Incorrect dataset type specified, ' \
                    'try {}'.format(available_dataset_types)
        logging.exception(error_msg)
        raise Exception(error_msg)

    # handle each dataset separately (move if to function at some point)
    if config['dataset_name'] == 'PROCAT':
        # load the csv
        logger.info('Handling {} csv files'.format(config['dataset_name']))

        # load datasets
        dset = ProcatData(dataset_type, logger, config, tokenizer)
        dloader = dset.get_dataloader()

        return dloader


def get_procat_dataloader_old(dataset_type, logger, config):
    """Create the dataloaders for the PROCAT or PROCAT mini train or test set"""
    if dataset_type not in ['train', 'test']:
        raise Exception('Incorrect dataset type specified, try train or test.')

    logger.info('Data loading {} ...'.format(dataset_type))
    X = torch.load(config['{}_X_path'.format(dataset_type)])
    Y = torch.load(config['{}_Y_path'.format(dataset_type)])
    Y = Y.long()

    # datasets
    logger.info('Creating TensorDataset {} ...'.format(dataset_type))
    dataset = TensorDataset(X, Y)

    # dataloaders
    logger.info('Creating DataLoader {} ...'.format(dataset_type))
    dataloader = DataLoader(dataset, batch_size=config['batch_size'],
                            shuffle=True, num_workers=4)

    # confirm sizes
    logger.info('Checking DataLoader {} dimensions ...'.format(dataset_type))
    single_x, single_y = get_example(dataloader, as_batch=True,
                                     x_name=config['x_name'],
                                     y_name=config['y_name'])
    logger.info('x: {}, y: {}'.format(single_x.size(), single_y.size()))

    return dataloader


def get_synthetic_dataloader(dataset_type, logger, config):
    """Create the dataloaders for the synthetic/functional train or test set"""

    # generate the raw catalogs
    catalogs_raw = generate_from_rules(dataset_type, config['rules'],
                                       config, logger, )

    # optionally throw out catalogs that didn't match target n tokens from config
    catalogs_raw = discard_catalogs_with_wrong_n_tokens(catalogs_raw, config,
                                                        dataset_type)

    # instantiate class objects
    catalogs_instantiated = instantiate_synthetic_catalogs(catalogs_raw, config)

    # turn to indices
    catalogs_as_indices = [c.as_indices() for c in catalogs_instantiated]
    logger.info('Sample synthetic catalogue as indices: {}'.format(
        catalogs_as_indices[0]))

    # show metrics
    metrics = get_rule_metrics(catalogs_as_indices, config['rules'], config,
                               dataset_type)
    log_synthetic_metrics(metrics, logger)

    # vectorize
    X, Y = get_xy_synthetic(catalogs_instantiated, config['n_mset_shuffles'])

    # need them as torch tensors first
    X = torch.from_numpy(np.asarray(X))
    Y = torch.from_numpy(np.asarray(Y))

    # dataset
    dataset = TensorDataset(X, Y)

    # loader
    dataloader = DataLoader(dataset, batch_size=config['batch_size'],
                            shuffle=True, num_workers=4)

    # check
    x_batch, y_batch = next(iter(dataloader))
    logger.info(
        'Checking synth dataloader (x and y batch sizes) ... {}, {}'.format(
            str(x_batch.size()), str(y_batch.size())))

    return dataloader


class ProcatData:
    """
    Object capable of turning a PROCAT dataset into a dataloader.
    """

    def __init__(self, dataset_type, logger, config, tokenizer):
        self.dataset_type = dataset_type
        self.logger = logger
        self.config = config
        self.tokenizer = tokenizer

    def get_dataloader(self):
        # load as a datasets object
        dataset = datasets.load_dataset(split=self.dataset_type,
                                        download_mode="reuse_dataset_if_exists",  # "reuse_dataset_if_exists" or "force_redownload" for debug
                                        **{'path': 'data/PROCAT/procat_dataset_loader.py'})

        # make smaller for debugging, if necessary
        if self.config['debug_dataset_sizes']:
            dataset = dataset.select(range(100))

        # map convert_to_features batch wise
        dataset = dataset.map(self.convert_to_features, batched=True,
                              batch_size=100)

        columns = ['s{}_input_ids'.format(i + 1) for i in range(200)] + \
                  ['s{}_attention_mask'.format(i + 1) for i in range(200)] + \
                  ['s{}_token_type_ids'.format(i + 1) for i in range(200)] + ['label']
        dataset.set_format(type="torch", columns=columns)

        # dataloader
        dataloader = DataLoader(dataset, shuffle=True, batch_size=self.config['batch_size'])

        return dataloader

    def convert_to_features(self, batch):
        """A version that works with batching and dataloaders"""
        batch_y = batch['label']

        batch_x = [
            self.tokenizer(s, max_length=self.config['max_sentence_length'],
                           padding="max_length",
                           truncation=True,
                           return_tensors="pt", ) for s in
            batch['shuffled_sentences']]

        # placeholder
        b = {}

        # extract
        b_input_ids = {'s{}_input_ids'.format(i + 1): [e['input_ids'][i].tolist() for e in batch_x] for i in range(200)}
        b_attention_masks = {'s{}_attention_mask'.format(i + 1): [e['attention_mask'][i].tolist() for e in batch_x] for i in range(200)}
        b_token_type_ids = {'s{}_token_type_ids'.format(i + 1): [e['token_type_ids'][i].tolist() for e in batch_x] for i in range(200)}

        # update
        b.update(b_input_ids)
        b.update(b_attention_masks)
        b.update(b_token_type_ids)
        b['labels'] = batch_y

        return b

    def convert_to_features_single(self, batch):
        """A version that works with 1-elem batches"""
        batch_y = batch['label']

        batch_x = [
            self.tokenizer(s, max_length=self.config['max_sentence_length'],
                           padding="max_length",
                           truncation=True,
                           return_tensors="pt", ) for s in
            batch['shuffled_sentences']]

        b = {
            "input_ids": [e["input_ids"].tolist() for e in batch_x],
            "attention_mask": [e["attention_mask"].tolist() for e in batch_x],
            "token_type_ids": [e["token_type_ids"].tolist() for e in batch_x],
            "labels": batch_y,
        }
        return b


class ROCStoryData:
    """
    Object capable of turning a dataset into a dataloader.
    """

    def __init__(self, dataset_type, logger, config, tokenizer):
        self.dataset_type = dataset_type
        self.logger = logger
        self.config = config
        self.tokenizer = tokenizer

    def get_dataloader(self):
        # load as a datasets object
        dataset = datasets.load_dataset(split=self.dataset_type, **{
            'path': 'data/rocstory/rocstory_ordering.py'})

        # make smaller for debugging, if necessary
        if self.config['debug_dataset_sizes']:
            dataset = dataset.select(range(100))

        # map convert_to_features batch wise
        dataset = dataset.map(self.convert_to_features, batched=True,
                              batch_size=100)

        columns = ['s1_input_ids',
                   's1_attention_mask',
                   's1_token_type_ids',
                   's2_input_ids',
                   's2_attention_mask',
                   's2_token_type_ids',
                   's3_input_ids',
                   's3_attention_mask',
                   's3_token_type_ids',
                   's4_input_ids',
                   's4_attention_mask',
                   's4_token_type_ids',
                   's5_input_ids',
                   's5_attention_mask',
                   's5_token_type_ids',
                   'label']
        dataset.set_format(type="torch", columns=columns)

        # dataloader
        dataloader = DataLoader(dataset, shuffle=True, batch_size=self.config['batch_size'])

        return dataloader

    def convert_to_features(self, batch):
        """A version that works with batching and dataloaders"""
        batch_y = batch['label']

        batch_x = [
            self.tokenizer(s, max_length=self.config['max_sentence_length'],
                           padding="max_length",
                           truncation=True,
                           return_tensors="pt", ) for s in
            batch['shuffled_sentences']]

        b = {
            's1_input_ids': [e['input_ids'][0].tolist() for e in batch_x],
            's1_attention_mask': [e['attention_mask'][0].tolist() for e in batch_x],
            's1_token_type_ids': [e['token_type_ids'][0].tolist() for e in batch_x],
            's2_input_ids': [e['input_ids'][1].tolist() for e in batch_x],
            's2_attention_mask': [e['attention_mask'][1].tolist() for e in batch_x],
            's2_token_type_ids': [e['token_type_ids'][1].tolist() for e in batch_x],
            's3_input_ids': [e['input_ids'][2].tolist() for e in batch_x],
            's3_attention_mask': [e['attention_mask'][2].tolist() for e in batch_x],
            's3_token_type_ids': [e['token_type_ids'][2].tolist() for e in batch_x],
            's4_input_ids': [e['input_ids'][3].tolist() for e in batch_x],
            's4_attention_mask': [e['attention_mask'][3].tolist() for e in batch_x],
            's4_token_type_ids': [e['token_type_ids'][3].tolist() for e in batch_x],
            's5_input_ids': [e['input_ids'][4].tolist() for e in batch_x],
            's5_attention_mask': [e['attention_mask'][4].tolist() for e in batch_x],
            's5_token_type_ids': [e['token_type_ids'][4].tolist() for e in batch_x],
            'labels': batch_y,
        }
        return b

    def convert_to_features_single(self, batch):
        """A version that works with 1-elem batches"""
        batch_y = batch['label']

        batch_x = [
            self.tokenizer(s, max_length=self.config['max_sentence_length'],
                           padding="max_length",
                           truncation=True,
                           return_tensors="pt", ) for s in
            batch['shuffled_sentences']]

        b = {
            "input_ids": [e["input_ids"].tolist() for e in batch_x],
            "attention_mask": [e["attention_mask"].tolist() for e in batch_x],
            "token_type_ids": [e["token_type_ids"].tolist() for e in batch_x],
            "labels": batch_y,
        }
        return b


def decode_sentence_batch_single(batch, tokenizer, dataset_name):
    """
    Take a batch from the dataloader and print full ordered
    sentences. Pairs with convert_to_features_single().
    :param batch:
    :return:
    """
    if dataset_name == 'rocstory':
        sentences = []

        # get tokens
        for i in range(5):
            s = batch['input_ids'][i][0]
            s = s.numpy().tolist()
            sentences.append(s)

        # decode
        decoded = []
        for s in sentences:
            ds = tokenizer.decode(s)

            # clean
            ds = ds.replace('[PAD]', '')
            ds = ds.rstrip()

            decoded.append(ds)

        # reorder
        reordered = [decoded[i] for i in batch['label'][0]]
        return reordered


def decode_sentence_batch(batch, tokenizer, dataset_name, logger, n_to_print=1):
    """
    Take a batch from the dataloader and print full ordered
    sentences.
    """
    if dataset_name == 'rocstory':
        for bi in range(n_to_print):
            sentences = []

            # get tokens
            for i in range(5):
                s = batch['s{}_input_ids'.format(i+1)][bi]
                s = s.numpy().tolist()
                sentences.append(s)

            # decode
            decoded = []
            for s in sentences:
                ds = tokenizer.decode(s)

                # clean
                ds = ds.replace('[PAD]', '')
                ds = ds.rstrip()

                decoded.append(ds)

            # reorder
            reordered = [decoded[i] for i in batch['label'][bi]]
            logger.info('Decoded, reordered sentences {}: {}'.format(bi+1, reordered))


def decode_procat_batch(a_batch, a_tokenizer, a_logger):
    """Use the tokenizer to turn an offer vector back into readable text"""

    # get single offer's input ids
    single_input_ids = a_batch['s1_input_ids'][0]
    a_logger.info('Original input ids [:20]: {}'.format(single_input_ids[:20]))

    # detokenize
    single_readable = a_tokenizer.decode(single_input_ids)
    a_logger.info(single_readable)


def get_sentence_ordering_dataloader(dataset_type, logger, config, tokenizer):
    """
    Create the dataloaders for sentence ordering datasets,
    per language model.
    """
    available_dataset_types = ['train', 'test', 'validation']
    if dataset_type not in available_dataset_types:
        error_msg = 'Incorrect dataset type specified, ' \
                    'try {}'.format(available_dataset_types)
        logging.exception(error_msg)
        raise Exception(error_msg)

    # handle each dataset separately (move if to function at some point)
    if config['dataset_name'] == 'rocstory':
        # load the csv
        logger.info('Handling {} csv files'.format(config['dataset_name']))

        # load datasets
        dset = ROCStoryData(dataset_type, logger, config, tokenizer)
        dloader = dset.get_dataloader()

        return dloader


def restore_grammar_sequence(x, y):
    """
    Take an x & y pair, where x is shuffled elements and y is the proper order
    indices, restore original sequence
    """
    # turn all inputs to lists, if they aren't ones already
    # (sometimes we get tensors)
    if type(x) == torch.Tensor:
        x = x.to(dtype=int)
        x = x.tolist()
    if type(y) == torch.Tensor:
        y = y.to(dtype=int)
        y = y.tolist()

    rec = ['n/a' for t in range(len(x))]

    for i, e in enumerate(y):
        rec[i] = x[e]

    return rec


def get_offers(offer_distributions, n_offers):
    """
    Get n offer instances, based on distribution rules
    :return: a list of offers following the distribution rules
    """
    offers = random.choices(
        list(offer_distributions.keys()),
        weights=list(offer_distributions.values()),
        k=n_offers)

    return offers


def get_ruleset(offers, general_rules, return_name=False):
    """
    Take a list of offer types, get the appropriate catalog
    rules and valid sections based on which scenario of composition
    we're dealing with (e.g. whether special types of offers are present,
    and in what combination).
    :param offers: list of offer instances by type
    :param general_rules: dictionary of rules for different catalog compositions
    :return: ruleset for this composition (catalog rules and valid sections)
    """
    ruleset_name = None

    if 'g' in offers and 'p' in offers:
        ruleset = general_rules['purple_and_green']
        ruleset_name = 'purple_and_green'
    elif 'g' in offers:
        ruleset = general_rules['green_only']
        ruleset_name = 'green_only'
    elif 'p' in offers:
        ruleset = general_rules['purple_only']
        ruleset_name = 'purple_only'
    else:
        ruleset = general_rules['basic']
        ruleset_name = 'basic'

    if return_name:
        return ruleset, ruleset_name
    else:
        return ruleset


def get_rarest_offer(offers, offer_distro, config):
    """
    Take a list of offers and a config, return the rarest present
    offer type.
    :param offers: a list of offers for current catalog
    :param config: a config
    :return: the rarest available offer type, based on config
    """
    offers_sorted_with_rarity = sorted(offer_distro.items(),
                                       key=lambda item: item[1])
    offer_types_from_rarest = [o[0] for o in offers_sorted_with_rarity]
    rarest_found_offer_type = None
    rarest_offer_found = False
    while not rarest_offer_found:
        for offer_type in offer_types_from_rarest:
            if offer_type in offers:
                rarest_found_offer_type = offer_type
                rarest_offer_found = True
                break

    return rarest_found_offer_type


def get_valid_sections(an_offer, a_ruleset):
    """
    Take an offer and a ruleset, return a list of valid sections.
    """
    all_valid_sections = a_ruleset['valid_sections']

    # find all valid sections where the given offer is present
    relevant_sections = []
    for section in all_valid_sections:
        allowed_offers = section['allowed_offer_types']
        if an_offer in allowed_offers:
            relevant_sections.append(section)

    return relevant_sections


def get_section(an_offer, available_offers, available_sections, log):
    """
    Take the current offer, other available offers and valid sections,
    return a composed section and the remaining available offers.
    """
    composition_found = False
    composed_section = None
    remaining_offers = available_offers

    # shuffle available sections (in place)
    random.shuffle(available_sections)

    for section in available_sections:

        needed_offers = []

        # shuffle the list of possible offer amounts in this section
        possible_num_offers = section['offers_per_section']
        random.shuffle(possible_num_offers)

        for num_offers in possible_num_offers:

            # get how many offers of each type we need (mixed and full)
            if section['mixed']:
                proportion_per_type = dict(zip(section['allowed_offer_types'],
                                               section['mix_ratio']))
                needed_per_type = {k: v * num_offers for k, v in
                                   proportion_per_type.items()}
            # sections made of a single type
            else:
                needed_per_type = {t: num_offers for t in
                                   section['allowed_offer_types']}

            # try to find this many offers in the available ones
            is_doable = check_if_composable(an_offer,
                                            available_offers,
                                            needed_per_type)

            # if it's not composable, we can try another num_offers
            if not is_doable:
                continue
            # otherwise compose this one
            else:
                composed_section, remaining_offers = compose_section(
                    an_offer, available_offers, needed_per_type)
                composition_found = True

                # if this composition worked, don't try other num offers
                break

        # if a composition is found, don't try other sections
        if composition_found:
            break

    # if we couldn't find a composition, report it
    if not composed_section:
        log.debug(
            'WARNING: Could not compose a legal section from remaining offers')

    return composed_section, remaining_offers


def check_if_composable(an_offer, available_offers, needed_per_type):
    """
    Check if a section can be composed from given offer and other available
    ones. Return True/False.
    """
    # first, remove the current offer from needed ones
    need_per_type = copy.deepcopy(needed_per_type)
    need_per_type[an_offer] -= 1

    # by default, a composition is possible until proven otherwise
    is_composable = True
    for offer_type, needed_amount in need_per_type.items():
        if available_offers.count(offer_type) < needed_amount:
            is_composable = False

    return is_composable


def compose_section(an_offer, available_offers, needed_per_type):
    """
    Once a section is confirmed to be composable, compose it.
    Return the composed section and the remaining list of available
    offers.
    """
    # first, remove the current offer from needed ones
    need_per_type = copy.deepcopy(needed_per_type)
    need_per_type[an_offer] -= 1

    # then begin composing the section
    composed_section = [an_offer]

    # and remove the used offers from the list of available ones
    for offer_type, needed_amount in need_per_type.items():
        for i in range(int(needed_amount)):
            available_offers.remove(offer_type)
            composed_section.append(offer_type)

    # shuffle the composed section, as the internal order of a section is noise
    random.shuffle(composed_section)

    return composed_section, available_offers


def create_section_from_rules(section_rules, possible_num_offers=None):
    """
    Take a valid section rule and return an actual section, as in
    a list of offer types
    """
    # shuffle the list of possible offer amounts in this section
    if not possible_num_offers:
        possible_num_offers = section_rules['offers_per_section']
    random.shuffle(possible_num_offers)

    for num_offers in possible_num_offers:

        # get how many offers of each type we need (mixed and full)
        if section_rules['mixed']:
            proportion_per_type = dict(zip(section_rules['allowed_offer_types'],
                                           section_rules['mix_ratio']))
            needed_per_type = {k: v * num_offers for k, v in
                               proportion_per_type.items()}
        # sections made of a single type
        else:
            needed_per_type = {t: num_offers for t in
                               section_rules['allowed_offer_types']}

    # then begin composing the section
    # composed_section = [section_rules['allowed_offer_types'][0]]
    composed_section = []

    # and remove the used offers from the list of available ones
    for offer_type, needed_amount in needed_per_type.items():
        for i in range(int(needed_amount)):
            composed_section.append(offer_type)

    # shuffle the composed section, as the internal order of a section is noise
    random.shuffle(composed_section)

    return composed_section


def apply_first_page_rule(a_catalog, first_page_rule, valid_sections):
    """
    Try to find a valid first section in the catalog and put it at the start
    of the catalog. If no such section can be found, add it from the
    basic ruleset.
    """
    valid_section_found = False
    valid_section = None

    # get the valid section for this rule
    first_section_name = first_page_rule['spec']
    first_section_rules = [s for s in valid_sections
                           if s['name'] == first_section_name][0]

    # try to find a section conforming to the rule from the catalog
    for section in a_catalog:

        # we start assuming the section is valid and then disprove it
        section_is_valid = True

        # check if only the right offer types are present
        for offer in section:
            if offer not in first_section_rules['allowed_offer_types']:
                section_is_valid = False
                break

        if not section_is_valid:
            continue

        # check if the number of offers is correct
        if len(section) not in first_section_rules['offers_per_section']:
            section_is_valid = False

        if not section_is_valid:
            continue

        # mixed section checks
        if first_section_rules['mixed']:

            # count offers in section, get proportion per type
            counts = {offer_type: 0 for offer_type in section}
            for offer in section:
                counts[offer] += 1
            total_offers = sum(counts.values())
            proportions = {offer_type: count / total_offers
                           for offer_type, count in counts.items()}

            # if proportion is not valid, neither is the section
            for i, type in enumerate(
                    first_section_rules['allowed_offer_types']):
                target_proportion = first_section_rules['mix_ratio'][i]
                if target_proportion != proportions[type]:
                    section_is_valid = False

        if not section_is_valid:
            continue
        else:
            valid_section_found = True
            valid_section = section
            break

    # at this point we tried all sections to find a matching first section
    # if we haven't, me must create and add it based on the first page rule
    if valid_section_found:
        # remove the found valid section
        a_catalog.remove(valid_section)

        # re-insert the section at the start of the catalog
        a_catalog.insert(0, valid_section)

    else:
        # create a new first page and add it to the catalog at the start
        valid_section = create_section_from_rules(first_section_rules)
        a_catalog.insert(0, valid_section)

    return a_catalog


def apply_last_page_rule(a_catalog, last_page_rule, valid_sections):
    """
    Try to find a valid last section in the catalog and put it at the end
    of the catalog. If no such section can be found, add it from the
    basic ruleset.
    """
    valid_section_found = False
    valid_section = None

    # get the valid section for this rule
    last_section_name = last_page_rule['spec']
    last_section_rules = [s for s in valid_sections
                          if s['name'] == last_section_name][0]

    # try to find a section conforming to the rule from the catalog
    # excluding the first one (which might be meaningful)
    for i, section in enumerate(a_catalog[1:]):

        # we start assuming the section is valid and then disprove it
        section_is_valid = True

        # check if only the right offer types are present
        for offer in section:
            if offer not in last_section_rules['allowed_offer_types']:
                section_is_valid = False
                break

        if not section_is_valid:
            continue

        # check if the number of offers is correct
        if len(section) not in last_section_rules['offers_per_section']:
            section_is_valid = False

        if not section_is_valid:
            continue

        # mixed section checks
        if last_section_rules['mixed']:

            # count offers in section, get proportion per type
            counts = {offer_type: 0 for offer_type in section}
            for offer in section:
                counts[offer] += 1
            total_offers = sum(counts.values())
            proportions = {offer_type: count / total_offers
                           for offer_type, count in counts.items()}

            # if proportion is not valid, neither is the section
            for i, type in enumerate(last_section_rules['allowed_offer_types']):
                target_proportion = last_section_rules['mix_ratio'][i]
                if target_proportion != proportions[type]:
                    section_is_valid = False

        if not section_is_valid:
            continue
        else:
            valid_section_found = True
            valid_section_idx = copy.deepcopy(i)
            valid_section = section
            break

    # at this point we tried all sections to find a matching last section
    # if we haven't, me must create and add it based on the last page rule
    if valid_section_found:
        # remove the found valid section by index (otherwise an earlier
        # identical section might be removed, breaking rules)
        del a_catalog[valid_section_idx + 1]  # because we skip first page

        # re-insert the section at the end of the catalog
        a_catalog.append(valid_section)

    else:
        # create a new last page and add it to the catalog at the end
        valid_section = create_section_from_rules(last_section_rules)
        a_catalog.append(valid_section)

    return a_catalog


def apply_max_pages_rule(a_catalog, a_rule, valid_sections):
    """
    Take a catalog (list of offer types), a specific rule of the 'max_pages'
    type and a list of valid sections, and check whether the catalog
    exceeds max pages, removing a basic page if it does.
    """
    page_limit = a_rule['spec']
    page_limit_exceeded = False

    if len(a_catalog) > page_limit:
        num_pages_to_remove = len(a_catalog) - page_limit
        page_limit_exceeded = True

    if page_limit_exceeded:

        for i in range(num_pages_to_remove):
            # find a section that is basic (only full red, blue or yellow)
            # should have some concept of basic pages for all catalogs in the cfg
            for i, section in enumerate(a_catalog):

                # we don't remove first and last pages
                if i != 0 and i + 1 != len(a_catalog) \
                        and 'g' not in section \
                        and 'p' not in section:
                    # remove it and exit the loop
                    del a_catalog[i]
                    break

    return a_catalog


def apply_min_pages_rule(a_catalog, a_rule, valid_sections):
    """
    Take a catalog (list of offer types), a specific rule of the 'min_pages'
    type and a list of valid sections, and check whether the catalog
    is shorter than min pages, addiing as many basic pages in the middle
    as needed.
    """
    page_min_limit = a_rule['spec']
    too_few_pages = False

    if len(a_catalog) < page_min_limit:
        num_pages_to_add = page_min_limit - len(a_catalog)
        too_few_pages = True

    if too_few_pages:

        for i in range(num_pages_to_add):
            # randomly choose a valid section scenario
            a_valid_section = random.choice(valid_sections)

            # create that section
            created_section = create_section_from_rules(a_valid_section)

            # and add it in the middle (not start, not finish) of the catalog
            valid_indices = [n for n in range(len(a_catalog)) + 1]
            del valid_indices[0]
            valid_indices.pop()
            chosen_index = random.choice(valid_indices)
            a_catalog.insert(chosen_index, created_section)

    return a_catalog


def apply_catalog_rules(a_catalog, a_ruleset, log):
    """
    Take a list of sections, made up of offer types (representing a catalog)
    and make them conform to the specified structural, catalog-level rules
    from the ruleset.
    :param a_catalog: a list of sections, with offers, making up a catalog
    :param a_ruleset: a list of rules, including structural ones
    :return:
    """
    catalog_rules = a_ruleset['catalog_rules']
    valid_sections = a_ruleset['valid_sections']

    for rule in catalog_rules:

        # handle the current types of rules
        if rule['name'] == 'first_page':
            a_catalog = apply_first_page_rule(a_catalog, rule, valid_sections)

            log.debug('Catalog after apply_first_page_rule()')
            log.debug(a_catalog)

        if rule['name'] == 'last_page':
            a_catalog = apply_last_page_rule(a_catalog, rule, valid_sections)

            log.debug('Catalog after apply_last_page_rule()')
            log.debug(a_catalog)

        if rule['name'] == 'max_pages':
            a_catalog = apply_max_pages_rule(a_catalog, rule, valid_sections)

            log.debug('Catalog after apply_max_pages_rule()')
            log.debug(a_catalog)

        if rule['name'] == 'min_pages':
            a_catalog = apply_min_pages_rule(a_catalog, rule, valid_sections)

            log.debug('Catalog after apply_min_pages_rule()')
            log.debug(a_catalog)

    return a_catalog


def count_tokens(a_catalog):
    """
    Count tokens in a catalog, so how many offers there are plus
    number of pages - 1 for page breaks.
    """
    current_n_tokens = 0

    # calculate current n tokens in the catalog
    for section in a_catalog:
        current_n_tokens += len(section)

    # also need to add 1 token for page breaks (minus 1, first page has none)
    current_n_tokens += len(a_catalog)
    current_n_tokens -= 1

    return current_n_tokens


def make_catalog_longer(a_catalog, a_ruleset, num_to_add):
    """
    Add offers to catalog pages to reach expected number of tokens.
    If that's impossible, at basic page.
    """
    # go through every page, but in random order
    section_indices = [n for n in range(len(a_catalog))]

    # remove first and last page, if needed
    # del section_indices[0]
    # section_indices.pop()

    # shuffle
    random.shuffle(section_indices)

    for section_index in section_indices:

        # locate section
        section = a_catalog[section_index]

        # check if it's a basic one (no p or g, not mixed)
        # and special offers explicit.
        if 'p' in section or 'g' in section or len(set(section)) > 1:
            continue

        # find the scenario of this page among valid ones
        current_section_scenario = identify_section_scenario(section, a_ruleset)

        # check how many offers you can add
        can_be_added = max(current_section_scenario['offers_per_section']) - \
                       len(section)

        # calculate how many we actually should add
        if num_to_add > can_be_added:
            should_add = can_be_added
        else:
            should_add = num_to_add
        a_catalog[section_index].extend(
            should_add * current_section_scenario['allowed_offer_types'])

        num_to_add -= should_add

        if num_to_add == 0:
            break

    # by this point, we either managed to squeeze extra offers into
    # existing pages, or we need to add a new page, possibly breaking
    # the max page limit
    if num_to_add == 0:
        return a_catalog
    elif num_to_add == 1:
        section_indices = [n for n in range(len(a_catalog))]
        random.shuffle(section_indices)

        # we need to remove one offer from an existing basic page
        for section_index in (section_indices):

            # locate section
            section = a_catalog[section_index]

            # skip special sections
            if 'p' in section or 'g' in section or len(set(section)) > 1:
                continue
            current_section_scenario = identify_section_scenario(section,
                                                                 a_ruleset)
            # check how many offers I can subtract
            can_be_subtracted = len(section) - \
                                min(current_section_scenario[
                                        'offers_per_section'])

            # if we can subtract 1, do so
            # this still says num_to_add, but it's to remove from 1 section
            # and then add later, through adding a new section
            if can_be_subtracted > num_to_add:
                a_catalog[section_index].remove(
                    current_section_scenario['allowed_offer_types'][0])
                num_to_add += 1  # we need to add 1 more token now
                break

        # and add a 1-offer page to the catalog, somewhere in the middle
        # (but not first or last)
        random_section_index = random.randint(1, len(a_catalog) - 1)

        # randomly choose a valid section scenario
        # but we need to make sure it's not mixed!
        non_mixed_found = False
        while not non_mixed_found:
            a_valid_section = random.choice(a_ruleset['valid_sections'])
            if not a_valid_section['mixed']:
                non_mixed_found = True

        # create that section (with a single offer)
        created_section = create_section_from_rules(a_valid_section,
                                                    possible_num_offers=[1, ])
        a_catalog.insert(random_section_index, created_section)

    else:
        while num_to_add != 0:
            # we have more than 1 token to add
            # so create a new page, from the basic types
            random_section_index = random.randint(1, len(a_catalog) - 1)

            # randomly choose a valid section scenario
            # but we need to make sure it's not mixed!
            # cause those we may not be able to produce with the right
            # number of offers (e.g. 50/50 with 3 offers)
            non_mixed_found = False
            while not non_mixed_found:
                a_valid_section = random.choice(a_ruleset['valid_sections'])
                if not a_valid_section['mixed']:
                    non_mixed_found = True

            # check how many offer tokens we can have in that section
            can_be_added = max(a_valid_section['offers_per_section'])
            if num_to_add > can_be_added:
                should_add = can_be_added
                num_to_add -= should_add
                num_to_add -= 1  # we're creatig a page, so extra page break
            else:
                should_add = num_to_add - 1  # we're creatig a page (+ 1 break)
                num_to_add = 0

            # create that section
            created_section = create_section_from_rules(a_valid_section,
                                                        possible_num_offers=[
                                                            should_add])
            a_catalog.insert(random_section_index, created_section)

    return a_catalog


def identify_section_scenario(a_section, a_ruleset):
    """
    Take a section (a list of offer types) and a ruleset, return the
    valid section scenario that this section belongs to.
    """
    valid_sections = a_ruleset['valid_sections']
    found_matching_section = None

    # go through all valid section, identify which one of them
    # the current section belongs to
    for valid_section in valid_sections:
        is_match = True

        # check length
        if len(a_section) not in valid_section['offers_per_section']:
            is_match = False

        if not is_match:
            continue

        # check offer types
        for offer in a_section:
            if offer not in valid_section['allowed_offer_types']:
                is_match = False

        if not is_match:
            continue

        # check if mixed
        if valid_section['mixed']:

            # count types per offer
            proportion_per_type = dict(zip(valid_section['allowed_offer_types'],
                                           valid_section['mix_ratio']))
            needed_per_type = {k: v * len(a_section) for k, v in
                               proportion_per_type.items()}

            # check if we have the right proportion in the current section
            for offer in a_section:
                needed_per_type[offer] -= 1

            if sum(needed_per_type.values()) != 0:
                is_match = False

        if is_match:
            found_matching_section = valid_section
            break

    # if we found a matching section, return it
    return found_matching_section


def make_catalog_shorter(a_catalog, a_ruleset, num_to_subtract):
    """
    Subtract offers from catalog pages to reach expected number of tokens.
    If that's impossible, remove an entire page.
    """
    is_right_num_tokens = False

    # go through every page, but in random order
    section_indices = [n for n in range(len(a_catalog))]
    random.shuffle(section_indices)

    for section_index in section_indices:

        # locate section
        section = a_catalog[section_index]

        # check if it's a basic one (no p or g, not mixed)
        # and special offers explicit.
        if 'p' in section or 'g' in section or len(set(section)) > 1:
            continue

        current_section_scenario = identify_section_scenario(section, a_ruleset)

        # check how many offers you can subtract
        can_be_subtracted = len(section) - min(
            current_section_scenario['offers_per_section'])

        # calculate how many we actually should add
        if num_to_subtract > can_be_subtracted:
            should_subtract = can_be_subtracted
        else:
            should_subtract = num_to_subtract

        # remove them
        for i in range(should_subtract):
            a_catalog[section_index].remove(
                current_section_scenario['allowed_offer_types'][0])

        num_to_subtract -= should_subtract

        if num_to_subtract == 0:
            is_right_num_tokens = True
            break

    # by this point, we either managed to remove enough or we need to remove
    # an entire page, possibly breaking the min page limit
    if not is_right_num_tokens:

        # we can try to find a page with (num_to_subtract - 1) offers
        # (extra one for the page break)
        # go through every page, but in random order
        # excluding first and last, which are special
        section_indices = [n for n in range(len(a_catalog))]
        section_indices.remove(0)
        section_indices.remove(len(a_catalog) - 1)
        random.shuffle(section_indices)

        for s_i in section_indices:

            # locate section
            section = a_catalog[s_i]

            # avoid special sections
            if 'p' in section or 'g' in section or len(set(section)) > 1:
                continue

            # remove it, if it matches our target num_to_subtract - 1
            if len(section) == num_to_subtract - 1:
                del a_catalog[s_i]
                is_right_num_tokens = True
                break

    # for now, break the valid_section rule if we still
    # can't make it fit and just take away num_to_subtract offer from the sections
    if not is_right_num_tokens:
        section_indices = [n for n in range(len(a_catalog))]
        section_indices.remove(0)
        section_indices.remove(len(a_catalog) - 1)
        random.shuffle(section_indices)

        # we stil try to avoid special and first-last sections
        for s_i in section_indices:

            # locate section
            section = a_catalog[s_i]

            # avoid special sections, but not mixed ones
            if 'p' in section or 'g' in section:
                continue

            # get scenario, to know what to remove
            current_section_scenario = identify_section_scenario(section,
                                                                 a_ruleset)

            # check how many offers you can subtract (to not hit an
            # empty page)
            can_be_subtracted = len(section) - 1

            # calculate how many we actually can remove
            if num_to_subtract > can_be_subtracted:
                will_subtract = can_be_subtracted
            else:
                will_subtract = num_to_subtract

            # remove them
            for i in range(will_subtract):
                a_catalog[s_i].pop()

            num_to_subtract -= will_subtract

            if num_to_subtract == 0:
                is_right_num_tokens = True
                break

    return a_catalog


def instantiate_synthetic_catalogs(raw_catalogs, cfg):
    """
    Turn a list of raw catalogs into instances of Offer(), Page() and
    Catalog() class. Return that list.
    :param raw_catalogs: list of raw catalogs (e.g. [['r'], ['b', 'b']])
    :param cfg: config with a word2idx defined
    :return: list of Catalog() objects
    """
    instantiated_catalogs = []

    for c in raw_catalogs:
        current_pages = []
        page_num = 0
        for p in c:
            page_num += 1
            current_offers = []
            for o in p:
                current_offer = Offer(o)
                current_offers.append(current_offer)
            current_page = Page(current_offers, page_num)
            current_pages.append(current_page)
        current_catalog = Catalog(cfg['word2idx'], current_pages)
        instantiated_catalogs.append(current_catalog)

    return instantiated_catalogs


def enforce_token_number(a_catalog, a_ruleset, n_tokens, log):
    """
    Take a catalog (a list of section, consisting of offer types)
    and force it to consist of a set number of tokens (from the config), by
    adding and removing offers from pages, within valid section rules.
    """

    target_n_tokens = n_tokens
    current_n_tokens = count_tokens(a_catalog)

    # a_catalog = [['r', 'r', 'r', 'r'], ['y', 'y', 'r', 'r'], ['y', 'r', 'y', 'r'],
    # ['y', 'y', 'r', 'r'], ['y', 'r'], ['y', 'r', 'r', 'y'], ['b', 'b', 'b'], ['b', 'b'], ['b', 'b']]

    log.debug(
        'N tokens before enforce_token_number(): {}'.format(current_n_tokens))

    is_too_short = False
    is_too_long = False
    to_add = 0
    to_subtract = 0

    # calculate how many we must add / subtract
    if target_n_tokens > current_n_tokens:
        is_too_short = True
        to_add = target_n_tokens - current_n_tokens
    elif target_n_tokens < current_n_tokens:
        is_too_long = True
        to_subtract = current_n_tokens - target_n_tokens

    # handle both scenarios
    if is_too_short:
        a_catalog = make_catalog_longer(a_catalog, a_ruleset, to_add)

        log.debug('Catalog after make_catalog_longer()')
        log.debug(a_catalog)
        log.debug('N tokens: {}'.format(str(count_tokens(a_catalog))))
        if count_tokens(a_catalog) != target_n_tokens:
            log.debug('**** ERROR **** WRONG NUMBER OF TOKENS STILL')

    elif is_too_long:
        a_catalog = make_catalog_shorter(a_catalog, a_ruleset, to_subtract)

        log.debug('Catalog after make_catalog_shorter()')
        log.debug(str(a_catalog))
        log.debug('N tokens: {}'.format(str(count_tokens(a_catalog))))
        if count_tokens(a_catalog) != target_n_tokens:
            log.debug('**** ERROR **** WRONG NUMBER OF TOKENS STILL')

    return a_catalog


def generate_from_rules(dataset_type, r, c, logger):
    """
    Take a set of rules and a config, return catalog data following those
    rules. Config contains rules for all catalogs, rules contain rules
    for catalogs consisting of specific sets of offers (composition matters).
    :param dataset_type: test or train, used to find the right values in cfg
    :param r: rule set
    :param c: config
    :param logger: logger
    :return: catalogs
    """
    catalogs = []
    if dataset_type == 'train':
        target_n_tokens = c['train_n_tokens']
        n_catalogs = c['train_n_catalogs']
        offer_distributions = c['train_offer_distributions']
        n_offers = c['train_n_offers']
    elif dataset_type == 'test':
        target_n_tokens = c['test_n_tokens']
        n_catalogs = c['test_n_catalogs']
        offer_distributions = c['test_offer_distributions']
        n_offers = c['test_n_offers']
    elif dataset_type == 'cv':
        target_n_tokens = c['cv_n_tokens']
        n_catalogs = c['cv_n_catalogs']
        offer_distributions = c['cv_offer_distributions']
        n_offers = c['cv_n_offers']

    for i in range(n_catalogs):

        logger.debug('Processing catalog {}'.format(i + 1))

        # track our sections, in the current catalog
        current_catalog = []

        # get n offer instances, based on distribution rules
        current_offers = get_offers(offer_distributions, n_offers)

        # get the rules for current offer set composition
        current_ruleset, _ = get_ruleset(current_offers, r, return_name=True)

        logger.debug('Current offers {}'.format(current_offers))
        logger.debug('Current ruleset {}'.format(current_ruleset))

        # begin trying to compose the catalog
        catalog_composed = False
        while not catalog_composed:

            # pick the rarest offers first
            current_offer = get_rarest_offer(current_offers,
                                             offer_distributions, c)

            logger.debug(
                'Current offer (rarest first) {}'.format(current_offer))

            # remove it from current offers
            current_offers.remove(current_offer)

            # identify valid sections for the current offer
            valid_sections = get_valid_sections(current_offer, current_ruleset)

            logger.debug('Valid sections {}'.format(valid_sections))

            # try the valid sections until one can be composed
            composed_section, current_offers = get_section(current_offer,
                                                           current_offers,
                                                           valid_sections,
                                                           log=logger)

            logger.debug('Composed section: {}'.format(str(composed_section)))

            # add the composed section to current ones
            if composed_section:
                current_catalog.append(composed_section)

                logger.debug('Catalog after new composed section: {}'.format(
                    current_catalog))

            # if there are no more offers available, the catalog is done
            if len(current_offers) == 0:
                catalog_composed = True

                logger.debug('No more offers to turn into pages')

        # shuffle the pages to avoid first-rarest shift
        random.shuffle(current_catalog)

        # once we have a preliminary version of the catalog, composed
        # of valid sections, we need to apply the catalog rules to it

        current_catalog = apply_catalog_rules(current_catalog, current_ruleset,
                                              log=logger)

        logger.debug('Catalog after apply_catalog_rules()')
        logger.debug(current_catalog)

        # enforce an exact, set number of tokens (in the config)
        current_catalog = enforce_token_number(current_catalog,
                                               current_ruleset,
                                               target_n_tokens,
                                               log=logger)

        logger.debug('Catalog after enforce_token_number()')
        logger.debug(current_catalog)

        # append catalog to the others
        catalogs.append(current_catalog)

    return catalogs


def discard_catalogs_with_wrong_n_tokens(raw_catalogs, cfg, dset_type):
    """
    Take raw catalogs and throw out the ones that have the wrong
    number of tokens, after all the adjustment functions. This is
    done because we sometimes generate 99% corrent n tokens catalogs,
    with rare exceptions, and that messes up the tensor shapes.
    """

    if dset_type == 'train':
        target_n_tokens = cfg['train_n_tokens']
    elif dset_type == 'test':
        target_n_tokens = cfg['test_n_tokens']
    elif dset_type == 'cv':
        target_n_tokens = cfg['test_n_tokens']
    else:
        raise Exception('Incorrect dataset type specified, try train or test.')

    cleaned_catalogs = []

    for c_raw in raw_catalogs:

        actual_n_tokens = count_tokens(c_raw)

        if actual_n_tokens != target_n_tokens:
            pass
        else:
            cleaned_catalogs.append(c_raw)

    return cleaned_catalogs


def report_catalog_stats(generated_catalogs, config, log):
    """
    Print basic statistics about the generated catalogs.
    """
    stats = []

    # check how many offers of each type there are
    total_offer_counts = {o: 0 for o in config['offer_distributions'].keys()}
    for c in generated_catalogs:
        for o in total_offer_counts.keys():
            for s in c:
                if o in s:
                    total_offer_counts[o] += 1
    log.info('Total offer counts: ', total_offer_counts)
    stats.append({'Total offer counts': total_offer_counts})

    # turn to %
    total_offer_percs = {k: round((v * 100) / sum(total_offer_counts.values()),
                                  2) for k, v in total_offer_counts.items()}

    log.info('Total offer %: ', total_offer_percs)
    stats.append({'Total offer %': total_offer_percs})

    # check how many offers on average there are in a catalog
    num_offers_in_catalogs = []
    for c in generated_catalogs:
        num_offers_in_catalog = 0
        for s in c:
            num_offers_in_section = len(s)
            num_offers_in_catalog += num_offers_in_section
        num_offers_in_catalogs.append(num_offers_in_catalog)

    mean = statistics.mean(num_offers_in_catalogs)
    stdev = statistics.stdev(num_offers_in_catalogs)

    avg_num_offers_per_catalog = dict()
    stdev_num_offers_per_catalog = dict()

    avg_num_offers_per_catalog['Average offers per catalog'] = round(
        mean, 2)
    stdev_num_offers_per_catalog['Standard Dev offers per catalog'] = round(
        stdev, 2)

    log.info('Average offers per catalog:',
             avg_num_offers_per_catalog['Average offers per catalog'])
    log.info('Standard Dev offers per catalog:',
             stdev_num_offers_per_catalog['Standard Dev offers per catalog'])
    stats.append(avg_num_offers_per_catalog)
    stats.append(stdev_num_offers_per_catalog)

    # check num catalogs per num offers in it
    catalogs_per_num_offers = dict()
    for c in generated_catalogs:
        offers_in_c = 0
        for s in c:
            offers_in_c += len(s)
        if offers_in_c not in catalogs_per_num_offers.keys():
            catalogs_per_num_offers[offers_in_c] = 1
        else:
            catalogs_per_num_offers[offers_in_c] += 1

    log.info('Catalogs per num offers: ', catalogs_per_num_offers)
    stats.append({'Catalogs per num offers:': catalogs_per_num_offers})

    # check num catalogs with each offer type
    per_catalog_offer_counts = {o: 0 for o in
                                config['offer_distributions'].keys()}
    for c in generated_catalogs:
        for o in per_catalog_offer_counts.keys():
            for s in c:
                if o in s:
                    per_catalog_offer_counts[o] += 1
                    break

    log.info('Catalogs per offer type: ', per_catalog_offer_counts)
    stats.append({'Catalogs per offer type': per_catalog_offer_counts})

    # get % from all catalogs
    per_catalog_offer_percs = {k: round((v * 100) / config['n_catalogs'],
                                        2) for k, v in
                               per_catalog_offer_counts.items()}

    # calculate how common the special mix is
    # check num catalogs with each offer type
    purple_and_green_count = 0

    for c in generated_catalogs:
        is_purple = False
        is_green = False
        for s in c:
            if 'p' in s:
                is_purple = True
            if 'g' in s:
                is_green = True
        if is_green and is_purple:
            purple_and_green_count += 1

    per_catalog_offer_percs['p_and_g'] = (purple_and_green_count * 100) / \
                                         config[
                                             'n_catalogs']

    log.info('Catalogs per offer type %: ', per_catalog_offer_percs)
    stats.append({'Catalogs per offer type %': per_catalog_offer_percs})

    # catalogs per number of sections
    section_lengths = dict()
    for c in generated_catalogs:
        num_sections = len(c)
        if num_sections not in section_lengths.keys():
            section_lengths[num_sections] = 1
        else:
            section_lengths[num_sections] += 1

    log.info('Catalogs per number of sections:',
             dict(sorted(section_lengths.items())))
    stats.append({'Catalogs per number of sections':
                      dict(sorted(section_lengths.items()))})

    section_lengths_perc = {k: round((v * 100) / config['n_catalogs'],
                                     2) for k, v in
                            section_lengths.items()}
    log.info('Catalogs per number of sections %:',
             dict(sorted(section_lengths_perc.items())))
    stats.append({'Catalogs per number of sections %': dict(
        sorted(section_lengths_perc.items()))})

    # offers per page
    offers_per_page = dict()
    for c in generated_catalogs:
        for s in c:
            num_offers_per_section = len(s)
            if num_offers_per_section not in offers_per_page.keys():
                offers_per_page[num_offers_per_section] = 1
            else:
                offers_per_page[num_offers_per_section] += 1

    log.info('Section count by number of offers:',
             dict(sorted(offers_per_page.items())))
    stats.append({'Section count by number of offers': dict(
        sorted(offers_per_page.items()))})

    # %
    offers_per_page_percs = {k: round((v * 100) / sum(offers_per_page.values()),
                                      2) for k, v in offers_per_page.items()}

    log.info('Section % by number of offers:',
             dict(sorted(offers_per_page_percs.items())))
    stats.append({'Section % by number of offers': dict(
        sorted(offers_per_page_percs.items()))})

    # section counts by scenario

    return stats


def get_rule_metrics(catalogs_as_indices, rules, cfg, dataset_type):
    """
    Take a list of catalogs as indices in the form of [5, 1, 1, 6, 0, 0 ...],
    as well as an object representing structural rules to which these
    catalogs are meant to adhere to, and a general config dict.
    Measure how well these catalogs adhere to each rule and return
    a report.
    :param catalogs_as_indices: a list of catalogs as indices
    :param rules: a dictionary mapping set composition to structural rules
    :param cfg: a dictionary with general, high level config
    :return: a metrics report
    """
    # metrics object storing the results of tests
    metrics = {}

    # Metric 0: Confirm correct number of tokens
    # without this, things break unexpectedly due to tensor size mismatch
    correct_n_tokens = get_token_metric(catalogs_as_indices, cfg, dataset_type)
    metrics['correct_n_tokens_%'] = correct_n_tokens

    # Metric 1: Percentage of valid catalogs
    # check how many are fundamentally valid catalogs
    valid_catalogs_perc = get_validity_metric(catalogs_as_indices, cfg)
    metrics['valid_catalogs_%'] = valid_catalogs_perc

    # Metric 2: Num offers adheres to config (by how much does it deviate)
    # Remember that this might not be useful, the original data
    # won't match the config 100% either
    num_offers_correct_perc = get_num_offers_metric(catalogs_as_indices, cfg,
                                                    dataset_type)
    metrics['num_offers_match_config_%'] = num_offers_correct_perc

    # Metric 3: Valid sections percentage, averaged over all catalogs
    valid_sections_percentages_avg, ruleset_valid_sections_perc = \
        get_valid_sections_perc_metric(catalogs_as_indices, rules, cfg)
    metrics['valid_sections_%_avg'] = valid_sections_percentages_avg
    metrics['valid_sections_%_per_ruleset'] = ruleset_valid_sections_perc

    # Metric 3: Percentage of catalogs following catalog rules, per ruleset
    # and in total
    valid_catalog_total_perc, valid_structure_perc_per_ruleset, \
    valid_structure_perc_per_sub_rule = \
        get_valid_structure_perc_metric(catalogs_as_indices, rules, cfg)
    metrics['valid_structure_%_total'] = valid_catalog_total_perc
    metrics['valid_structure_%_per_ruleset'] = valid_structure_perc_per_ruleset
    metrics[
        'valid_structure_%_per_sub_rule'] = valid_structure_perc_per_sub_rule

    return metrics


def get_token_metric(catalogs_as_indices, cfg, dataset_type):
    n_catalogs = len(catalogs_as_indices)
    n_valid = 0
    for c in catalogs_as_indices:

        # check if it is a fundamentally valid catalog
        is_valid = check_token_n_correct(c, cfg, dataset_type)
        if is_valid:
            n_valid += 1

    # update the appropriate metrics
    return round(100 * n_valid / n_catalogs, 2)


def count_offers(readable_catalog, cfg, offer_distro):
    count = 0
    offer_types = list(offer_distro.keys())
    for elem in readable_catalog:
        if elem in offer_types:
            count += 1
    return count


def get_num_offers_metric(catalogs_as_indices, cfg, dset_type):
    if dset_type == 'train':
        target_number_of_offers = cfg['train_n_offers']
        offer_distro = cfg['train_offer_distributions']
    elif dset_type == 'test':
        target_number_of_offers = cfg['test_n_offers']
        offer_distro = cfg['test_offer_distributions']
    elif dset_type == 'cv':
        target_number_of_offers = cfg['cv_n_offers']
        offer_distro = cfg['cv_offer_distributions']
    else:
        raise Exception('Wrong dataset type, try train or test.')

    n_catalogs = len(catalogs_as_indices)
    n_correct = 0
    word2idx = cfg['word2idx']
    idx2word = {v: k for k, v in word2idx.items()}

    for c in catalogs_as_indices:

        # first, turn indices to readable strings
        catalog_readable = [idx2word[e] for e in c]

        # count offers
        num_offers = count_offers(catalog_readable, cfg, offer_distro)
        if target_number_of_offers == num_offers:
            n_correct += 1

    # update the appropriate metrics
    return round(100 * n_correct / n_catalogs, 2)


def check_token_n_correct(catalog_as_indices, cfg, dataset_type):
    """
    Take a catalog as indices in the form of [5, 1, 1, 6, 0, 0 ...] and a config
    object containing the word2idx mapping, check if the number of tokens
    is correct.
    """
    if dataset_type == 'train':
        target_n_tokens = cfg['train_n_tokens']
    elif dataset_type == 'test':
        target_n_tokens = cfg['test_n_tokens']
    elif dataset_type == 'cv':
        target_n_tokens = cfg['cv_n_tokens']
    else:
        raise Exception('Wrong dataset type, try train or test.')

    is_valid = True
    current_n_tokens = len(catalog_as_indices) - 2  # ignore start and end

    if current_n_tokens != target_n_tokens:
        is_valid = False

    return is_valid


def log_synthetic_metrics(m, log):
    """
    Take a metrics dictionary and display it in easy to read format.
    """
    log.info('*** METRICS ***')

    # overall
    log.info('{:>40}'.
             format('GENERAL:'))
    log.info('{:>40} {:>10}%'.
             format('Total testable catalogs %:', m['valid_catalogs_%']))
    log.info('{:>40} {:>10}%'.
             format('Catalogs with corrent n tokens %:',
                    m['correct_n_tokens_%']))
    log.info('{:>40} {:>10}%'.
             format('Catalogs with target num offers %:',
                    m['num_offers_match_config_%']))
    log.info('')

    # sections
    log.info('{:>40}'.
             format('SECTIONS:'))
    log.info('{:>40} {:>10}%'.
             format('Valid section average %:',
                    m['valid_sections_%_avg']))

    log.info('{:>40}'.
             format('Valid section per ruleset % :'))
    for k, v in m['valid_sections_%_per_ruleset'].items():
        log.info('{:>45} {:>10}%'.
                 format(k, v))

    log.info('')

    # catalog structure
    log.info('{:>40}'.
             format('STRUCTURAL:'))
    log.info('{:>40} {:>10}%'.
             format('Valid catalog structure %:',
                    m['valid_structure_%_total']))
    log.info('{:>40}'.
             format('Valid catalog % per ruleset:'))
    for k, v in m['valid_structure_%_per_ruleset'].items():
        log.info('{:>45} {:>10}%'.
                 format(k, v))
    log.info('{:>40}'.
             format('Valid catalog % per sub rule:'))
    for rset, v in m['valid_structure_%_per_sub_rule'].items():
        log.info('{:>45}'.
                 format(rset))
        for sr, e in v.items():
            log.info('{:>50} {:>10}%'.
                     format(sr, e))


def get_validity_metric(catalogs_as_indices, cfg):
    n_catalogs = len(catalogs_as_indices)
    n_valid = 0
    for c in catalogs_as_indices:

        # check if it is a fundamentally valid catalog
        is_valid = check_basic_validity(c, cfg)
        if is_valid:
            n_valid += 1

    # update the appropriate metrics
    return round(100 * n_valid / n_catalogs, 2)


def check_basic_validity(catalog_as_indices, cfg):
    """
    Take a catalog as indices in the form of [5, 1, 1, 6, 0, 0 ...] and a config
    object containing the word2idx mapping, check if the basic, fundamental
    rules are followed. I.e. starts with catalog start, ends with catalog
    end, page breaks separate offers, and don't come immediately before or
    after catalog start tokens etc.
    """
    is_valid = True
    word2idx = cfg['word2idx']
    idx2word = {v: k for k, v in word2idx.items()}

    # first, turn indices to readable strings
    catalog_readable = [idx2word[e] for e in catalog_as_indices]

    # series of checks is what follows
    # starts with catalog start
    if catalog_readable[0] != 'catalog_start':
        is_valid = False
        return is_valid

    # doesn't end with catalog end
    if catalog_readable[-1] != 'catalog_end':
        is_valid = False
        return is_valid

    # only one catalog start and catalog end present
    if catalog_readable.count('catalog_start') != 1:
        is_valid = False
        return is_valid

    if catalog_readable.count('catalog_end') != 1:
        is_valid = False
        return is_valid

    # page break doesn't occur immediately after some offer
    for i, token in enumerate(catalog_readable):
        if token == 'page_break':
            preceding_token = catalog_readable[i - 1]
            following_token = catalog_readable[i + 1]
            if preceding_token not in ['b', 'g', 'p', 'r', 'y']:
                is_valid = False
                return is_valid
            # page break before catalog_end
            if following_token not in ['b', 'g', 'p', 'r', 'y',
                                       'catalog_end']:
                is_valid = False
                return is_valid

    return is_valid


def get_xy_synthetic(some_catalogs, n_mset_shuffles):
    """
    Turn a list of Catalog() instances into X and Y.
    """
    # create shuffled sets from sequences)
    X = []
    Y = []

    for c in some_catalogs:

        # target is the original sequence order
        o = c.as_indices()

        # create n permutations
        for i in range(n_mset_shuffles):

            # x and y start as a placeholders
            x = ['n/a' for l in range(len(o))]
            y = [0 + j for j in range(len(o))]
            random.shuffle(y)

            for i, e in enumerate(y):
                x[i] = o[e]

            # y's indices and elements need to be flipped to represent what the model outputs
            final_y = copy.deepcopy(y)
            for i, e in enumerate(y):
                final_y[e] = i

            # recreate original catalog from shuffled indices and x
            rec = restore_synthetic_catalog(x, final_y)

            # append
            X.append(np.asarray(x))
            Y.append(np.asarray(final_y))

    return X, Y


def restore_synthetic_catalog(x, y):
    """
    Take an x & y pair, where x is shuffled elements and y is the proper order indices, restore original catalog.
    """
    # turn all inputs to lists, if they aren't ones already
    # (sometimes we get tensors)
    if type(x) == torch.Tensor:
        x = x.to(dtype=int)
        x = x.tolist()
    if type(y) == torch.Tensor:
        y = y.to(dtype=int)
        y = y.tolist()

    rec = ['n/a' for t in range(len(x))]

    for i, e in enumerate(y):
        rec[i] = x[e]

    return rec


def restore_synthetic_catalogs(X, Y):
    """
    Restore a batch of catalogs.
    """
    # turn all inputs to lists, if they aren't ones already
    # (sometimes we get tensors)
    r = []

    for i, x in enumerate(X):
        restored_c = restore_synthetic_catalog(x, Y[i])
        r.append(restored_c)

    return r


def get_valid_sections_perc_metric(catalogs_as_indxs, rules, cfg):
    """
    Metric for what percentage of sections in each catalog is valid,
    based on the rules, and returns the average % for all catalogs.
    """
    catalogs_as_indices = copy.deepcopy(catalogs_as_indxs)

    # for per-ruleset metrics
    ruleset_sections = {r: 0 for r in rules}
    ruleset_valid_sections = {r: 0 for r in rules}

    valid_sections_percentages_avg = None
    valid_sections_percentages = []

    # turn catalog into raw form
    catalogs_as_raw = synthetic_from_indices_to_raw(catalogs_as_indices, cfg)

    # flatten to identify which ruleset is applicable to each catalog
    catalogs_as_raw_flattened = synthetic_from_raw_to_flat(catalogs_as_raw)

    # for every catalog, check if sections are valid
    for i, c_raw in enumerate(catalogs_as_raw):

        # identify ruleset by set composition
        relevant_ruleset, ruleset_name = get_ruleset(
            catalogs_as_raw_flattened[i], rules, return_name=True)

        num_total_sections = len(c_raw)
        num_valid_sections = 0
        for s in c_raw:

            # increment number of sections per ruleset
            ruleset_sections[ruleset_name] += 1

            found_scenario = identify_section_scenario(s, relevant_ruleset)

            if found_scenario:
                num_valid_sections += 1

                # increment valid sections counter
                ruleset_valid_sections[ruleset_name] += 1

        # if there are no sections, valid_sect_per = 0
        if num_total_sections == 0:
            valid_sections_perc = 0.0
        else:
            valid_sections_perc = num_valid_sections * 100 / num_total_sections

        # track for all catalogs
        valid_sections_percentages.append(valid_sections_perc)

    # turn into an average per catalog
    valid_sections_percentages_avg = round(sum(valid_sections_percentages)
                                           / len(valid_sections_percentages), 2)

    # calculate % valid sections per ruleset
    unseen_rulesets = []
    for r, n in ruleset_sections.items():
        if n == 0:
            unseen_rulesets.append(r)
            # replace the zero, we'll remove the stat later
            ruleset_sections[r] = 1

    ruleset_valid_sections_perc = {r: round(v * 100 / ruleset_sections[r], 2)
                                   for r, v in ruleset_valid_sections.items()}

    # now remove the invalid stat
    for r in unseen_rulesets:
        ruleset_valid_sections_perc[r] = 'div by zero'

    return valid_sections_percentages_avg, ruleset_valid_sections_perc


def synthetic_from_indices_to_raw(catalogs_as_indices, cfg):
    """
    Take a list of catalogs in the form of [5, 0, 0, 6, 1, 2, 1, 2, 6 ... ]
    and use the config to turn them into valid [['r', 'r'], ['b'] ... ].
    """
    catalogs_as_ind = copy.deepcopy(catalogs_as_indices)
    catalogs_as_raw = []
    word2idx = cfg['word2idx']
    idx2word = {v: k for k, v in word2idx.items()}

    for c_indices in catalogs_as_ind:
        # c_indices = [1, 1, 6, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1]

        # we are assuming we have a fundamentally valid catalog here
        # with only 1 catalog start and end token

        # remove all catalog start tokens
        # since some badly predicted catalogs without masking
        # never include a start/end or have multiple:
        if word2idx['catalog_start'] in c_indices:
            c_indices[:] = [e for e in c_indices if
                            e != word2idx['catalog_start']]

        if word2idx['catalog_end'] in c_indices:
            c_indices[:] = [e for e in c_indices if
                            e != word2idx['catalog_end']]

        # split into sub-lists at section breaks
        c_raw = []
        current_section = []
        for j, i in enumerate(c_indices):
            current_token = idx2word[i]
            # not page break, not last token
            if current_token != 'page_break' and j != len(c_indices) - 1:
                current_section.append(current_token)
            # page break is last token
            elif current_token == 'page_break' and j == len(c_indices) - 1:
                c_raw.append(current_section)
                current_section = []
            # page break, but not last token
            elif current_token == 'page_break':
                c_raw.append(current_section)
                current_section = []
            # last token isn't page break
            elif current_token != 'page_break' and j == len(c_indices) - 1:
                current_section.append(current_token)
                c_raw.append(current_section)

        catalogs_as_raw.append(c_raw)

    return catalogs_as_raw


def synthetic_from_raw_to_flat(catalogs_as_raw):
    catalogs_flat = []
    # unnest them
    for c_raw in catalogs_as_raw:
        c_flat = list(itertools.chain.from_iterable(c_raw))
        catalogs_flat.append(c_flat)

    return catalogs_flat


def get_valid_structure_perc_metric(catalogs_as_indxs, rules, cfg):
    """
    Take a list of catalogs as indices and a rules object. Go over
    each catalog and each rule, measuring the percentage of catalogs
    that are correct based on each ruleset (set composition) and each
    structural rule.
    """
    # work on copy, not original
    catalogs_as_indices = copy.deepcopy(catalogs_as_indxs)

    # for per-ruleset metrics
    catalogs_per_ruleset = {r: 0 for r in rules}
    good_catalogs_per_ruleset = {r: 0 for r in rules}
    catalogs_per_subrule_per_ruleset = get_sub_rule_dict(rules)
    good_catalogs_per_subrule_per_ruleset = get_sub_rule_dict(rules)

    # turn catalog into raw form
    catalogs_as_raw = synthetic_from_indices_to_raw(catalogs_as_indices, cfg)

    # flatten to identify which ruleset is applicable to each catalog
    catalogs_as_raw_flattened = synthetic_from_raw_to_flat(catalogs_as_raw)

    # for every catalog, check which ruleset it belongs to
    # and for every structural rule, check if it is valid
    for i, c_raw in enumerate(catalogs_as_raw):

        # identify ruleset by set composition
        relevant_ruleset, ruleset_name = get_ruleset(
            catalogs_as_raw_flattened[i], rules, return_name=True)

        # increment catalog counts per ruleset
        catalogs_per_ruleset[ruleset_name] += 1

        # go through every subrule in the ruleset and check if it's followed
        all_subrules_followed = True

        for subrule in relevant_ruleset['catalog_rules']:
            subrule_name = subrule['name'] + '_' + str(subrule['spec'])

            is_subrule_followed = check_catalog_rule(c_raw, subrule,
                                                     relevant_ruleset)

            # update number of catalogs per subrule
            catalogs_per_subrule_per_ruleset[ruleset_name][
                subrule_name] += 1

            # and number of good ones, if it was followed
            if is_subrule_followed:
                good_catalogs_per_subrule_per_ruleset[ruleset_name][
                    subrule_name] += 1

            # mark if at least 1 subrule wasn't followed for this catalog
            if not is_subrule_followed:
                all_subrules_followed = False

        if all_subrules_followed:
            good_catalogs_per_ruleset[ruleset_name] += 1

    # turn to percentages

    # total
    n_total_catalogs = sum(catalogs_per_ruleset.values())
    n_good_catalogs = sum(good_catalogs_per_ruleset.values())
    good_catalog_total_perc = round(n_good_catalogs * 100 / n_total_catalogs, 2)

    # per ruleset
    good_catalogs_per_ruleset_perc = {k: None for k in
                                      good_catalogs_per_ruleset}
    for k in good_catalogs_per_ruleset_perc:

        # sometimes, there are 0 valid catalogs for a ruleset:
        if catalogs_per_ruleset[k] == 0:
            good_catalogs_per_ruleset_perc[k] = 0.0
        else:
            good_catalogs_per_ruleset_perc[k] = round(
                good_catalogs_per_ruleset[k] * 100 / catalogs_per_ruleset[k], 2)

    # per ruleset, per sub rule
    good_catalogs_per_subrule_per_ruleset_perc = copy.deepcopy(
        good_catalogs_per_subrule_per_ruleset)
    for k, v in good_catalogs_per_subrule_per_ruleset.items():
        for s in v:
            # sometimes, there are 0 valid catalogs for a ruleset:
            if catalogs_per_subrule_per_ruleset[k][s] == 0:
                good_catalogs_per_subrule_per_ruleset_perc[k][s] = 0.0
            else:
                good_catalogs_per_subrule_per_ruleset_perc[k][s] = round(
                    good_catalogs_per_subrule_per_ruleset[k][s] * 100 /
                    catalogs_per_subrule_per_ruleset[k][s], 2)

    return good_catalog_total_perc, good_catalogs_per_ruleset_perc, \
           good_catalogs_per_subrule_per_ruleset_perc


def get_sub_rule_dict(rules):
    """
    Return a properly structured dict for counting well-structured catalogs
    per rulest, per rule.
    """
    x1 = {r: rs['catalog_rules'] for r, rs in rules.items()}
    x2 = {r: {s['name'] + '_' + str(s['spec']): 0 for s in rs} for r, rs in
          x1.items()}

    return x2


def check_catalog_rule(catalog_raw, subrule, ruleset):
    """
    Take a structural subrule and a raw catalog, return True/False
    if followed properly.
    """

    if subrule['name'] == 'first_page':
        is_followed = check_first_page_rule(catalog_raw, subrule, ruleset)
    elif subrule['name'] == 'last_page':
        is_followed = check_last_page_rule(catalog_raw, subrule, ruleset)
    elif subrule['name'] == 'max_pages':
        is_followed = check_max_pages_rule(catalog_raw, subrule, ruleset)
    elif subrule['name'] == 'min_pages':
        is_followed = check_min_pages_rule(catalog_raw, subrule, ruleset)

    return is_followed


def check_first_page_rule(catalog_raw, subrule, ruleset):
    is_followed = False

    # sometimes, there's not even a first section (empty catalog predicted)
    if len(catalog_raw) == 0:
        is_followed = False
        return is_followed

    # but normally
    first_section = catalog_raw[0]
    target_section_name = subrule['spec']

    actual_section_scenario = identify_section_scenario(first_section, ruleset)

    # scenario not found
    if not actual_section_scenario:
        return is_followed

    if actual_section_scenario['name'] == target_section_name:
        is_followed = True

    return is_followed


def check_last_page_rule(catalog_raw, subrule, ruleset):
    is_followed = False

    # sometimes, there's not even any sections (empty catalog predicted)
    if len(catalog_raw) == 0:
        is_followed = False
        return is_followed

    last_section = catalog_raw[-1]
    target_section_name = subrule['spec']

    actual_section_scenario = identify_section_scenario(last_section, ruleset)

    # scenario not found
    if not actual_section_scenario:
        return is_followed

    if actual_section_scenario['name'] == target_section_name:
        is_followed = True

    return is_followed


def check_max_pages_rule(catalog_raw, subrule, ruleset):
    is_followed = False
    actual_num_sections = len(catalog_raw)
    max_num_sections = subrule['spec']

    if actual_num_sections <= max_num_sections:
        is_followed = True

    return is_followed


def check_min_pages_rule(catalog_raw, subrule, ruleset):
    is_followed = False
    actual_num_sections = len(catalog_raw)
    min_num_sections = subrule['spec']

    if actual_num_sections >= min_num_sections:
        is_followed = True

    return is_followed


def predict_synthetic_catalogs_as_indices(a_dataloader, a_model):
    """
    Take a model, predict on every batch in the dataloader
    and return predicted catalogs as indices (a list of lists).
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    predicted_cs_as_indices = []

    for batch in a_dataloader:
        # get x batch from batch of x & y
        batch_x = batch[0]

        # handle cpu/gpu
        batch_x = batch_x.to(device)

        # predict on batch
        _, batch_preds = a_model(batch_x)

        # restore catalog from preds (as indices)
        catalogs_as_ind = restore_synthetic_catalogs(batch_x, batch_preds)

        predicted_cs_as_indices.extend(catalogs_as_ind)

    return predicted_cs_as_indices


def get_permuted_floats_batch(set_length, elem_dims=None, is_random=True):
    """
    Simple function for getting a 2-elem batch of a set and its random
    permutation.
    Can return the exact opposite order instead of a fully pseudo-random one.
    """
    if elem_dims:
        in1 = np.random.rand(1, set_length, elem_dims)
    else:
        in1 = np.random.rand(1, set_length)
    in2 = copy.deepcopy(in1)

    if is_random:
        for set in in2:
            np.random.shuffle(set)
    else:
        in2 = np.flip(in2, 1).astype(float)
    x1 = torch.from_numpy(in1).float()
    x2 = torch.from_numpy(in2).float()
    b = torch.stack([x1, x2])
    b = b.squeeze(1)
    return b


def get_sentence_ordering_tokenizer(language_model):
    """Instantiate the right tokenizer for a given language model"""
    known_language_models = ['bert-base-cased']

    # catch unknown models
    if language_model not in known_language_models:
        error_msg = 'Unknown language model provided: {}, try {}'.format(
            language_model, known_language_models)
        logging.exception(error_msg)
        raise Exception(error_msg)
    # BERT cased
    elif language_model == 'bert-base-cased':
        tokenizer = BertTokenizer.from_pretrained(language_model)
    else:
        error_msg = 'Language model known but not handled ' \
                    'in get_sentence_ordering_tokenizer().'
        logging.exception(error_msg)
        raise Exception(error_msg)

    return tokenizer


def get_procat_tokenizer(language_model):
    """Instantiate the right tokenizer for a given language model"""
    known_language_models = ['danish-bert-botxo']

    # catch unknown models
    if language_model not in known_language_models:
        error_msg = 'Unknown language model provided: {}, try {}'.format(
            language_model, known_language_models)
        logging.exception(error_msg)
        raise Exception(error_msg)
    # BERT cased
    elif language_model == 'danish-bert-botxo':
        tokenizer = AutoTokenizer.from_pretrained(language_model, from_tf=True)
    else:
        error_msg = 'Language model known but not handled ' \
                    'in get_sentence_ordering_tokenizer().'
        logging.exception(error_msg)
        raise Exception(error_msg)

    return tokenizer


def get_permuted_batch(set_length, elem_dims, is_random=True):
    """
    Simple function for getting a 2-elem batch of a set and its random
    permutation.
    Can return the exact opposite order instead of a fully pseudo-random one.
    """
    in1 = np.random.rand(1, set_length, elem_dims)
    in2 = copy.deepcopy(in1)
    if is_random:
        for set in in2:
            np.random.shuffle(set)
    else:
        in2 = np.flip(in2, 1).astype(float)
    x1 = torch.from_numpy(in1).float()
    x2 = torch.from_numpy(in2).float()
    b = torch.stack([x1, x2])
    b = b.squeeze(1)

    # handle gpu/cpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    b = b.to(device)
    return b

