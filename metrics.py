"""
One place for all testing functions and metrics.
"""
import ast
import copy

import matplotlib.pyplot as plt
import numpy as np
# torch
import torch
from scipy.stats import spearmanr, kendalltau
from torch.autograd import Variable

from utils_data import restore_grammar_sequence, Brackets, reorder, \
    predict_synthetic_catalogs_as_indices, synthetic_from_indices_to_raw, \
    instantiate_synthetic_catalogs, get_rule_metrics, is_path_legal, get_tour_length


def test_model_custom(a_model, a_dataloader, logger, config,
                      comparison_func=None, print_every=500):
    """
    I would now like a function that takes a dataloader of test data and a model package,
    making it predict on each and then outputs the average loss.
    """
    # handle gpu/cpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # metrics, placeholders
    num_examples = len(a_dataloader.dataset)
    individual_scores = []
    counter = 0

    # iterate over dataset to predict and track
    # no grad here
    with torch.no_grad():
        for single_batch in a_dataloader:

            train_batch = Variable(single_batch[config['x_name']])
            target_batch = Variable(single_batch[config['y_name']])

            # handle gpu/cpu
            train_batch = train_batch.to(device)
            target_batch = target_batch.to(device)

            # predict
            o, batched_predictions = a_model(train_batch)

            # move back to cpu for metric functions
            train_batch = train_batch.to('cpu')
            target_batch = target_batch.to('cpu')
            batched_predictions = batched_predictions.to('cpu')

            # track
            for idx, model_solution in enumerate(batched_predictions):

                # compare solutions (might need a custom function per dataset)
                current_score = comparison_func(prediction=model_solution,
                                                y=target_batch[idx],
                                                x=train_batch[idx],
                                                config=config)
                individual_scores.append(current_score)

                # update counter & report
                counter += 1
                if counter % print_every == 0:
                    logger.info('Calculating example {} / {} ...'.format(
                        counter, num_examples
                    ))

        final_score = sum(individual_scores) / num_examples

    return final_score, individual_scores


def test_sentence_ordering_model_custom(a_model, a_dataloader,
                                        logger, config,
                                        comparison_func=None, print_every=500):
    """
    I would now like a function that takes a dataloader of test data and a model package,
    making it predict on each and then outputs the average loss.
    """
    # handle gpu/cpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    a_model.eval().to(device)

    # metrics, placeholders
    num_examples = len(a_dataloader.dataset)
    individual_scores = []
    counter = 0

    # iterate over dataset to predict and track
    # no grad here
    with torch.no_grad():
        for single_batch in a_dataloader:

            # move batch to right device
            single_batch = {k: v.to(device) for k, v in single_batch.items()}

            # handle huggingface batches
            train_batch = single_batch
            target_batch = single_batch['label']

            # predict
            o, batched_predictions = a_model(train_batch)

            # track
            for idx, model_solution in enumerate(batched_predictions):

                # compare solutions (might need a custom function per dataset)
                current_score = comparison_func(prediction=model_solution,
                                                y=target_batch[idx],
                                                x=None,
                                                config=config)
                individual_scores.append(current_score)

                # update counter & report
                counter += 1
                if counter % print_every == 0:
                    logger.info('Calculating example {} / {} ...'.format(
                        counter, num_examples
                    ))

        final_score = sum(individual_scores) / num_examples

    return final_score, individual_scores


def test_procat_model_custom(a_model, a_dataloader,
                             logger, config,
                             comparison_func=None, print_every=500):
    """
    I would now like a function that takes a dataloader of test data and a model package,
    making it predict on each and then outputs the average loss.
    """
    # handle gpu/cpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    a_model.eval().to(device)

    # metrics, placeholders
    num_examples = len(a_dataloader.dataset)
    individual_scores = []
    counter = 0

    # iterate over dataset to predict and track
    # no grad here
    with torch.no_grad():
        for single_batch in a_dataloader:

            # move batch to right device
            single_batch = {k: v.to(device) for k, v in single_batch.items()}

            # handle huggingface batches
            train_batch = single_batch
            target_batch = single_batch['label']

            # predict
            o, batched_predictions = a_model(train_batch)

            # track
            for idx, model_solution in enumerate(batched_predictions):

                # compare solutions (might need a custom function per dataset)
                current_score = comparison_func(prediction=model_solution,
                                                y=target_batch[idx],
                                                x=None,
                                                config=config)
                individual_scores.append(current_score)

                # update counter & report
                counter += 1
                if counter % print_every == 0:
                    logger.info('Calculating example {} / {} ...'.format(
                        counter, num_examples
                    ))

        final_score = sum(individual_scores) / num_examples

    return final_score, individual_scores


def compare_solved_float_sorting(prediction, y, x, config):
    """
    Take a single model prediction and a single actual soltuion,
    compare them in a way that return a single number marking the quality
    of the prediction.
    """
    correct_preds = 0
    total_preds = len(prediction)
    for i, p in enumerate(prediction):
        if p == y[i]:
            correct_preds += 1

    return correct_preds / total_preds


def compare_solved_grammar_canonical(prediction, y, x, config):
    """
    Check if prediction is grammatical, according to the canonical grammar.
    We don't actually need the solution.
    """
    # first, restore the predicted order from permutation prediction
    predicted_symbol_indices = restore_grammar_sequence(x, prediction)

    # reverse the wright ord2idx
    idx2word = {v: k for k, v in config['word2idx'].items()}

    # turn to actual symbols
    predicted_symbols = [idx2word[i] for i in predicted_symbol_indices]

    # also need original symbols, for non-masked models
    original_symbols = [idx2word[i.item()] for i in x]

    # check correctness individually
    n_correct, _ = check_sequences_canonical([predicted_symbols],
                                             [original_symbols])

    return n_correct


def compare_solved_grammar_brackets(prediction, y, x, config):
    """
    Check if prediction is grammatical, according to the brackets grammar.
    We don't actually need the solution.
    """
    # first, restore the predicted order from permutation prediction
    predicted_symbol_indices = restore_grammar_sequence(x, prediction)

    # reverse the wright ord2idx
    idx2word = {v: k for k, v in config['word2idx'].items()}

    # turn to actual symbols
    predicted_symbols = [idx2word[i] for i in predicted_symbol_indices]

    # also need original symbols, for non-masked models
    original_symbols = [idx2word[i.item()] for i in x]

    # check correctness individually
    n_correct, _ = check_sequences_brackets([predicted_symbols],
                                            [original_symbols])

    return n_correct


def compare_solved_grammar_harder_context_sensitive(prediction, y, x, config):
    """
    Check if prediction is grammatical, according to the canonical grammar.
    We don't actually need the solution.
    """
    # first, restore the predicted order from permutation prediction
    predicted_symbol_indices = restore_grammar_sequence(x, prediction)

    # reverse the right ord2idx
    idx2word = {v: k for k, v in config['word2idx'].items()}

    # turn to actual symbols
    predicted_symbols = [idx2word[i] for i in predicted_symbol_indices]

    # also need original symbols, for non-masked models
    original_symbols = [idx2word[i.item()] for i in x]

    # check correctness individually
    correct_ratio, _ = check_sequences_harder_context_sensitive([predicted_symbols],
                                                                [original_symbols])

    return correct_ratio


def compare_solved_procat(prediction, y, x, config):
    """
    Take a single model prediction and a single actual solution,
    compare them in a way that return a single number marking the quality
    of the prediction.
    """
    correct_preds = 0
    total_preds = len(prediction)
    for i, p in enumerate(prediction):
        if p == y[i]:
            correct_preds += 1

    return correct_preds / total_preds


def compare_solved_sentence_ordering(prediction, y, x, config):
    """
    Take a single model prediction and a single actual solution,
    compare them in a way that return a single number marking the quality
    of the prediction.
    """
    correct_preds = 0
    total_preds = len(prediction)
    for i, p in enumerate(prediction):
        if p == y[i]:
            correct_preds += 1

    return correct_preds / total_preds


def compare_solved_sort_unique(prediction, solution):
    """
    Take a single model prediction and a single actual soltuion,
    compare them in a way that return a single number marking the quality
    of the prediction.
    """
    correct_preds = 0
    total_preds = len(prediction)
    for i, p in enumerate(prediction):
        if p == solution[i]:
            correct_preds += 1

    return correct_preds / total_preds


def check_sequences_canonical(sequences_predicted, sequences_original=None,
                              debug=False):
    """
    Take a list of symbol sequences, check if they're grammatical according
    to the canonical grammar (context sensitive).
    """
    correct_counter = 0
    incorrect_sequences = []

    # get target counts from original sequence (should all be the same n)
    if sequences_original:
        # get target counts from original sequence
        for seq_original in sequences_original:
            target_count_a = seq_original.count('a')
            target_count_b = seq_original.count('b')
            target_count_c = seq_original.count('c')

            if target_count_a != target_count_b or target_count_a != target_count_c or \
                    target_count_b != target_count_c:
                raise Exception(
                    'Original sequence is wrong! Different numbers of a, b and c')

    for ind, seq in enumerate(sequences_predicted):

        # if it doesn't start with 'a', it's wrong, go to next sequence
        if seq[0] != 'a':
            incorrect_sequences.append(seq)
            if debug:
                print('FALSE! ', seq)
            continue
        # else, start counting 'a's, and if the next one isn't a 'b',
        # it's wrong
        else:
            counter_a = 0
            counter_b = 0
            counter_c = 0

            # 'a' loop (count 'a's)
            for idx_a, symbol in enumerate(seq):
                if symbol == 'a':
                    counter_a += 1
                else:
                    break

            # 'b' loop
            seq_b = seq[:][idx_a:]  # copy

            # if the next one isn't a 'b' it's wrong
            if seq_b[0] != 'b':
                incorrect_sequences.append(seq)
                if debug:
                    print('FALSE! ', seq)
                continue

            for idx_b, symbol in enumerate(seq_b):
                if symbol == 'b':
                    counter_b += 1
                else:
                    break

            # 'c' loop
            seq_c = seq_b[:][idx_b:]  # copy

            # if the next one isn't a 'c' it's wrong
            if seq_c[0] != 'c':
                incorrect_sequences.append(seq)
                if debug:
                    print('FALSE! ', seq)
                continue

            for idx_c, symbol in enumerate(seq_c):
                if symbol == 'c':
                    counter_c += 1
                else:
                    break

            # finally, if there is anything but padding afterwards,
            # it's also wrong.
            seq_remainder = seq_c[:][idx_c:]  # copy
            if 'a' in seq_remainder or 'b' in seq_remainder \
                    or 'c' in seq_remainder:
                incorrect_sequences.append(seq)
                if debug:
                    print('FALSE! ', seq)
                continue

            # check counts
            if counter_a != counter_b or counter_a != counter_c \
                    or counter_b != counter_c:
                incorrect_sequences.append(seq)
                if debug:
                    print('FALSE! ', seq)
                continue

            # ultimate check is for non-masked models,
            # which can repeat symbols, here we need the
            # sequence to be grammatical and have the same number of
            # a's b's and c's as the original
            if sequences_original:
                target_count_a = sequences_original[ind].count('a')
                target_count_b = sequences_original[ind].count('b')
                target_count_c = sequences_original[ind].count('c')
                if counter_a != target_count_a or counter_b != target_count_b \
                        or counter_c != target_count_c:
                    incorrect_sequences.append(seq)
                    if debug:
                        print('FALSE UNUSUALLY! ', seq)
                    continue

            # if we haven't broken out of the loop yet,
            # we know it's correct
            if debug:
                print('TRUE! ', seq)
            correct_counter += 1

    correct_ratio = correct_counter / len(sequences_predicted)

    return correct_ratio, incorrect_sequences


def check_sequences_brackets(sequences_predicted, sequences_original=None,
                             debug=False):
    """
    Take a list of symbol sequences, check if they're grammatical according
    to the brackets grammar (context free).
    """
    correct_counter = 0
    incorrect_sequences = []

    for idx, seq in enumerate(sequences_predicted):

        # get target counts from original sequence
        if sequences_original:
            target_count_lp = sequences_original[idx].count('(')
            target_count_rp = sequences_original[idx].count(')')
            target_count_lc = sequences_original[idx].count('{')
            target_count_rc = sequences_original[idx].count('}')

            if target_count_lp != target_count_rp \
                    or target_count_lc != target_count_rc:
                raise Exception(
                    'Original sequence is wrong! Left and right'
                    ' brackets numbers do not match')

        # if it doesn't start with left bracket, it's wrong, go to next sequence
        if seq[0] not in ['(', '{']:
            incorrect_sequences.append(seq)
            if debug:
                print('FALSE! ', seq)
            continue

        # else, start tracing opened and closed brackets
        else:
            opened_brackets = []
            closed_brackets = []
            sentence_is_valid = True

            # main loop
            for idx, symbol in enumerate(seq):

                # 1. Current symbol is an open paren
                if symbol == '(':
                    a_bracket = Brackets(type='parenthesis')
                    a_bracket.open()
                    opened_brackets.append(a_bracket)

                # 2. Current symbol is an open curly
                elif symbol == '{':
                    a_bracket = Brackets(type='curly')
                    a_bracket.open()
                    opened_brackets.append(a_bracket)

                # 3. Current symbol is a closed paren
                elif symbol == ')':
                    # check if we have an opened bracket there
                    if len(opened_brackets) == 0:
                        sentence_is_valid = False
                        break
                    # check if last opened bracket matches style
                    last_opened_bracket = opened_brackets[-1]
                    if last_opened_bracket.type == 'parenthesis':
                        last_opened_bracket.close()
                        # remove it from opened
                        opened_brackets.pop()
                        # add it to closed
                        closed_brackets.append(last_opened_bracket)
                    else:
                        sentence_is_valid = False
                        # break out of the symbol loop
                        break

                # 4. Current symbol is a closed curly
                elif symbol == '}':
                    if len(opened_brackets) == 0:
                        sentence_is_valid = False
                        break
                    # check if last opened bracket matches style
                    last_opened_bracket = opened_brackets[-1]
                    if last_opened_bracket.type == 'curly':
                        # remove it from opened
                        last_opened_bracket.close()
                        opened_brackets.pop()
                        # add it to closed
                        closed_brackets.append(last_opened_bracket)
                    else:
                        sentence_is_valid = False
                        # break out of the symbol loop
                        break

                # 5. Current symbol is padding
                elif symbol == 'P':
                    # break out of the symbol loop and check remainder of seq
                    break

            # check if there are any remaining opened brackets
            if len(opened_brackets) > 0:
                sentence_is_valid = False

            # check if remainder of sequence contains anything but padding
            # this might be an empty sequence
            if '(' in seq[idx + 1:] or ')' in seq[idx + 1:] or '{' in seq[
                                                                      idx + 1:] \
                    or '}' in seq[idx + 1:]:
                sentence_is_valid = False

            # check only if original sequences are provided
            if sequences_original:
                # ultimate checkfor non-masked models, check if there's the
                # same amount of each bracket symbols.
                actual_count_lp = seq.count('(')
                actual_count_rp = seq.count(')')
                actual_count_lc = seq.count('{')
                actual_count_rc = seq.count('}')

                if actual_count_lp != target_count_lp or \
                        actual_count_rp != target_count_rp or \
                        actual_count_lc != target_count_lc or \
                        actual_count_rc != target_count_rc:
                    sentence_is_valid = False

            # handle sentence not being valid
            if not sentence_is_valid:
                incorrect_sequences.append(seq)
                if debug:
                    print('FALSE! ', seq)
            # otherwise, the sentence is correct
            else:
                if debug:
                    print('TRUE! ', seq)
                correct_counter += 1

    correct_ratio = correct_counter / len(sequences_predicted)

    return correct_ratio, incorrect_sequences


def check_sequences_harder_context_sensitive(sequences_predicted,
                                             sequences_original=None,
                                             debug=False):
    """
    Take a list of symbol sequences, check if they're grammatical according
    to the harder context sensitive grammar: a^n b^m c^{n*m}
    """
    correct_counter = 0
    incorrect_sequences = []

    # get target counts from original sequence (should all be the same n)
    if sequences_original:
        # get target counts from original sequence
        for seq_original in sequences_original:
            target_count_a = seq_original.count('a')
            target_count_b = seq_original.count('b')
            target_count_c = seq_original.count('c')

            if target_count_c != target_count_a * target_count_b:
                raise Exception(
                    'Original sequence is wrong! Different numbers n(c) != n(a) * n(b)')

    for ind, seq in enumerate(sequences_predicted):

        # if it doesn't start with 'a', it's wrong, go to next sequence
        if seq[0] != 'a':
            incorrect_sequences.append(seq)
            if debug:
                print('FALSE! ', seq)
            continue
        # else, start counting 'a's, and if the next one isn't a 'b',
        # it's wrong
        else:
            counter_a = 0
            counter_b = 0
            counter_c = 0

            # 'a' loop (count 'a's)
            for idx_a, symbol in enumerate(seq):
                if symbol == 'a':
                    counter_a += 1
                else:
                    break

            # 'b' loop
            seq_b = seq[:][idx_a:]  # copy

            # if the next one isn't a 'b' it's wrong
            if seq_b[0] != 'b':
                incorrect_sequences.append(seq)
                if debug:
                    print('FALSE! ', seq)
                continue

            for idx_b, symbol in enumerate(seq_b):
                if symbol == 'b':
                    counter_b += 1
                else:
                    break

            # 'c' loop
            seq_c = seq_b[:][idx_b:]  # copy

            # if the next one isn't a 'c' it's wrong
            if seq_c[0] != 'c':
                incorrect_sequences.append(seq)
                if debug:
                    print('FALSE! ', seq)
                continue

            for idx_c, symbol in enumerate(seq_c):
                if symbol == 'c':
                    counter_c += 1
                else:
                    break

            # finally, if there is anything but padding afterwards,
            # it's also wrong.
            seq_remainder = seq_c[:][idx_c:]  # copy
            if 'a' in seq_remainder or 'b' in seq_remainder \
                    or 'c' in seq_remainder:
                incorrect_sequences.append(seq)
                if debug:
                    print('FALSE! ', seq)
                continue

            # check counts
            if counter_a * counter_b != counter_c:
                incorrect_sequences.append(seq)
                if debug:
                    print('FALSE! ', seq)
                continue

            # ultimate check is for non-masked models,
            # which can repeat symbols, here we need the
            # sequence to be grammatical and have the same number of
            # a's b's and c's as the original
            if sequences_original:
                target_count_a = sequences_original[ind].count('a')
                target_count_b = sequences_original[ind].count('b')
                target_count_c = sequences_original[ind].count('c')
                if counter_a != target_count_a or counter_b != target_count_b \
                        or counter_c != target_count_c:
                    incorrect_sequences.append(seq)
                    if debug:
                        print('FALSE UNUSUALLY! ', seq)
                    continue

            # if we haven't broken out of the loop yet,
            # we know it's correct
            if debug:
                print('TRUE! ', seq)
            correct_counter += 1

    correct_ratio = correct_counter / len(sequences_predicted)

    return correct_ratio, incorrect_sequences


def get_single_kendall_tau(single_y, single_pred):
    """Returns the Kendall Tau score for a single prediction"""

    def get_x_idx_and_rank_dict(a_y):
        x_idx_and_rank = dict()
        for i, e in enumerate(a_y):
            x_idx_and_rank[int(e)] = int(i)
        return x_idx_and_rank

    # map ranks to idx in x
    y = get_x_idx_and_rank_dict(single_y)
    p = get_x_idx_and_rank_dict(single_pred)

    # sort
    ranks_y = [y[key] for key in sorted(y.keys(), reverse=True)]
    ranks_p = [p[key] for key in sorted(p.keys(), reverse=True)]

    # compare
    tau, _ = kendalltau(ranks_y, ranks_p)

    return tau


def get_single_spearman_rho(single_y, single_pred):
    """Returns the Spearman Rho score for a single prediction"""

    def get_x_idx_and_rank_dict(a_y):
        x_idx_and_rank = dict()
        for i, e in enumerate(a_y):
            x_idx_and_rank[int(e)] = int(i)
        return x_idx_and_rank

    # map ranks to idx in x
    y = get_x_idx_and_rank_dict(single_y)
    p = get_x_idx_and_rank_dict(single_pred)

    # sort
    ranks_y = [y[key] for key in sorted(y.keys(), reverse=True)]
    ranks_p = [p[key] for key in sorted(p.keys(), reverse=True)]

    # compare
    tau, _ = spearmanr(ranks_y, ranks_p)

    return tau


def check_is_prediction_invalid(single_pred_as_batch):
    """
    Take tensors of shape [n], return bool if no repetition of elements (as if it was masked).
    """
    is_repeated = True

    n_target = len(single_pred_as_batch)
    n_ranked = len(single_pred_as_batch.unique())

    if n_target == n_ranked:
        is_repeated = False

    return is_repeated


def get_batch_rank_correlation_and_perc_valid(y_dataloader, a_model,
                                              rank_correlation_func,
                                              logger, config,
                                              print_every=1000):
    """
    Take a dataloader and a model, predict, return average of chosen rank
    correlation metric and what % of predictions could even be tested.
    """
    # handle gpu/cpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ranks = []
    valid_predictions = 0
    c = 0

    # go through every batch
    for batch in y_dataloader:

        # predict
        batch_x = batch[config['x_name']]
        batch_y = batch[config['y_name']]

        # move batch to the right device
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        _, batch_pred = a_model(batch_x)

        # move back to cpu for validation metrics
        batch_x = batch_x.to('cpu')
        batch_y = batch_y.to('cpu')
        batch_pred = batch_pred.to('cpu')

        # get individual correlation ranks
        for i, y in enumerate(batch_y):
            c += 1
            pred = batch_pred[i]

            logger.debug('x: {}'.format(batch_x[i]))
            logger.debug('target y: {}'.format(y))
            logger.debug('predicted: {}'.format(pred))

            # figure out if prediction has no repetition (predicts a rank for each element)
            prediction_is_repeated = check_is_prediction_invalid(pred)

            if not prediction_is_repeated:
                valid_predictions += 1
                rank = rank_correlation_func(y, pred)
                ranks.append(rank)

            if c % print_every == 0:
                print("{} / {}".format(c, len(y_dataloader.dataset)))

    # aggregate (avg)
    if len(ranks) > 0:
        r = sum(ranks) / len(ranks)
    else:
        r = 0

    # % of valid
    perc_valid = round(valid_predictions * 100 / c, 2)
    return r, perc_valid


def get_batch_rank_correlation_and_perc_valid_sentence_ordering(
        y_dataloader, a_model, rank_correlation_func, logger,
        config, print_every=1000):
    """
    Take a dataloader and a model, predict, return average of chosen rank
    correlation metric and what % of predictions could even be tested.
    """
    # model mode
    a_model.eval()

    # device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ranks = []
    valid_predictions = 0
    c = 0

    # go through every batch
    for batch in y_dataloader:

        # move batch to right device
        batch = {k: v.to(device) for k, v in batch.items()}

        # predict
        batch_x = batch
        batch_y = batch['label']
        _, batch_pred = a_model(batch_x)

        # get individual correlation ranks
        for i, y in enumerate(batch_y):
            c += 1
            pred = batch_pred[i]
            logger.debug('target y: {}'.format(y))
            logger.debug('predicted: {}'.format(pred))

            # figure out if prediction has no repetition (predicts a rank for each element)
            prediction_is_repeated = check_is_prediction_invalid(pred)

            if not prediction_is_repeated:
                valid_predictions += 1
                rank = rank_correlation_func(y, pred)
                ranks.append(rank)

            if c % print_every == 0:
                print("{} / {}".format(c, len(y_dataloader.dataset)))

    # aggregate (avg)
    if len(ranks) > 0:
        r = sum(ranks) / len(ranks)
    else:
        r = 0

    # % of valid
    perc_valid = round(valid_predictions * 100 / c, 2)
    return r, perc_valid


def get_batch_rank_correlation_and_perc_valid_procat(
        y_dataloader, a_model, rank_correlation_func, logger,
        config, print_every=1000):
    """
    Take a dataloader and a model, predict, return average of chosen rank
    correlation metric and what % of predictions could even be tested.
    """
    # model mode
    a_model.eval()

    # device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ranks = []
    valid_predictions = 0
    c = 0

    # go through every batch
    for batch in y_dataloader:

        # move batch to right device
        batch = {k: v.to(device) for k, v in batch.items()}

        # predict
        batch_x = batch
        batch_y = batch['label']
        _, batch_pred = a_model(batch_x)

        # get individual correlation ranks
        for i, y in enumerate(batch_y):
            c += 1
            pred = batch_pred[i]
            logger.debug('target y: {}'.format(y))
            logger.debug('predicted: {}'.format(pred))

            # figure out if prediction has no repetition (predicts a rank for each element)
            prediction_is_repeated = check_is_prediction_invalid(pred)

            if not prediction_is_repeated:
                valid_predictions += 1
                rank = rank_correlation_func(y, pred)
                ranks.append(rank)

            if c % print_every == 0:
                print("{} / {}".format(c, len(y_dataloader.dataset)))

    # aggregate (avg)
    if len(ranks) > 0:
        r = sum(ranks) / len(ranks)
    else:
        r = 0

    # % of valid
    perc_valid = round(valid_predictions * 100 / c, 2)
    return r, perc_valid


def run_float_sorting_tests(a_model, a_dataloader, logger, config):
    a_model.eval()

    result_general, _ = test_model_custom(a_model, a_dataloader, logger, config,
                                          comparison_func=compare_solved_float_sorting,
                                          print_every=999999)
    logger.info('Result (general): {:.4f}'.format(result_general))

    result_tau, rank_valid_perc = get_batch_rank_correlation_and_perc_valid(
        a_dataloader, a_model, get_single_kendall_tau, logger, config,
        print_every=999999)
    logger.info('K-Tau: {:.4f}, perc_valid: {}'.format(result_tau,
                                                       rank_valid_perc))

    result_spearman, rank_valid_perc = get_batch_rank_correlation_and_perc_valid(
        a_dataloader, a_model, get_single_spearman_rho, logger, config,
        print_every=999999)
    logger.info('S-Rho: {:.4f}, perc_valid: {}'.format(result_spearman,
                                                       rank_valid_perc))

    a_model.train()

    return result_general, result_tau, result_spearman, rank_valid_perc


def run_grammar_tests(a_model, a_dataloader, logger, config, batched=None):
    a_model.eval()

    # choose the right comparison function
    logger.info('Choosing the appropriate comparison function ...')
    if config['grammar_type'] == 'canonical':
        comp_func = compare_solved_grammar_canonical
    elif config['grammar_type'] == 'brackets':
        comp_func = compare_solved_grammar_brackets
    elif config['grammar_type'] == 'harder':
        comp_func = compare_solved_grammar_harder_context_sensitive
    else:
        raise Exception('Unknown grammar type found in run_grammar_tests() !')

    logger.info('Running test_model_custom() ...')
    result_general, _ = test_model_custom(a_model, a_dataloader, logger, config,
                                          comparison_func=comp_func,
                                          print_every=999999)
    logger.info('Result (general): {:.4f}'.format(result_general))

    result_tau, rank_valid_perc = get_batch_rank_correlation_and_perc_valid(
        a_dataloader, a_model, get_single_kendall_tau, logger, config,
        print_every=999999)
    logger.info('K-Tau: {:.4f}, perc_valid: {}'.format(result_tau,
                                                       rank_valid_perc))

    result_spearman, rank_valid_perc = get_batch_rank_correlation_and_perc_valid(
        a_dataloader, a_model, get_single_spearman_rho, logger, config,
        print_every=999999)
    logger.info('S-Rho: {:.4f}, perc_valid: {}'.format(result_spearman,
                                                       rank_valid_perc))

    a_model.train()

    return result_general, result_tau, result_spearman, rank_valid_perc


def run_procat_tests(a_model, a_dataloader, logger, config):
    a_model.eval()

    result_general, _ = test_procat_model_custom(a_model, a_dataloader, logger, config,
                                                 comparison_func=compare_solved_procat,
                                                 print_every=999999)
    logger.info('Result (general): {:.4f}'.format(result_general))

    result_tau, rank_valid_perc = get_batch_rank_correlation_and_perc_valid_procat(
        a_dataloader, a_model, get_single_kendall_tau, logger, config,
        print_every=999999)
    logger.info('K-Tau: {:.4f}, perc_valid: {}'.format(result_tau,
                                                       rank_valid_perc))

    result_spearman, rank_valid_perc = get_batch_rank_correlation_and_perc_valid_procat(
        a_dataloader, a_model, get_single_spearman_rho, logger, config,
        print_every=999999)
    logger.info('S-Rho: {:.4f}, perc_valid: {}'.format(result_spearman,
                                                       rank_valid_perc))

    a_model.train()

    return result_general, result_tau, result_spearman, rank_valid_perc


def run_sentence_ordering_tests(a_model, a_dataloader, logger, config, batched=None):
    # handle mode
    a_model.eval()

    result_general, _ = test_sentence_ordering_model_custom(
        a_model, a_dataloader, logger, config,
        comparison_func=compare_solved_sentence_ordering, print_every=999999)
    logger.info('Result (general): {:.4f}'.format(result_general))

    result_tau, rank_valid_perc = get_batch_rank_correlation_and_perc_valid_sentence_ordering(
        a_dataloader, a_model, get_single_kendall_tau, logger, config,
        print_every=999999)
    logger.info('K-Tau: {:.4f}, perc_valid: {}'.format(result_tau,
                                                       rank_valid_perc))

    result_spearman, rank_valid_perc = get_batch_rank_correlation_and_perc_valid_sentence_ordering(
        a_dataloader, a_model, get_single_spearman_rho, logger, config,
        print_every=999999)
    logger.info('S-Rho: {:.4f}, perc_valid: {}'.format(result_spearman,
                                                       rank_valid_perc))

    a_model.train()

    return result_general, result_tau, result_spearman, rank_valid_perc


def plot_losses(losses, config):
    # Data for plotting
    t = np.array([i + 1 for i in range(len(losses))])

    fig, ax = plt.subplots()
    ax.plot(t, losses)

    ax.set(xlabel='batch', ylabel='loss',
           title='Losses during training')
    ax.grid()

    fig.savefig('plots/losses_{}.png'.format(config['run_full_name']))
    # plt.show()


def examine_single_prediction(a_model, x, y, logger, restore_func, idx2word=None):
    logger.info('Sample x: {} \n{}'.format(x.size(), x))
    logger.info('Sample y: {} \n{}'.format(y.size(), y))

    # check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # move the data to the appropriate device
    x = x.to(device)
    y = y.to(device)

    # predict (pre training)
    _, single_pred = a_model(x)
    logger.info('Prediction: {} \n{}'.format(
        single_pred.size(), single_pred))

    # move back to cpu for the metrics function
    x = x.to('cpu')
    single_pred.to('cpu')

    # restore predicted order (permute)
    predicted_order = restore_func(x.squeeze(0), single_pred.squeeze(0))
    correct_order = restore_func(x.squeeze(0), y.squeeze(0))

    # for grammars, show decoded
    if idx2word:
        logger.info('Predicted order: \n{}'.format(predicted_order))
        logger.info('Prediction decoded: \n{}'.format([idx2word[e] for e in predicted_order]))

        logger.info('Correct order: \n{}'.format(correct_order))
        logger.info('Correct decoded: \n{}'.format([idx2word[e] for e in correct_order]))
    else:
        logger.info('Predicted order: \n{}'.format(predicted_order))
        logger.info('Correct order: \n{}'.format(correct_order))


def display_catalog(a_catalog, offers_df, display_fun):
    """
    Nicely display the (predicted) catalog, based on text features of offers.
    """
    # display knobs
    h_idx = 40
    d_idx = 40

    catalog_as_one_string = ['\n\n']

    # iterate
    for offer in a_catalog:
        # handle special tokens
        if offer == '?NOT_REAL_OFFER?':
            catalog_as_one_string.append(('\t{}'.format(offer)))
        elif offer == '?PAGE_BREAK?':
            catalog_as_one_string.append('------------------------')
        else:
            # get header and description
            header = \
                offers_df.loc[offers_df['offer_id'] == offer][
                    'text'].values[0]
            description = \
                offers_df.loc[offers_df['offer_id'] == offer][
                    'description'].values[0]

            # clear NaN
            if header is np.NaN:
                header = ''
            if description is np.NaN:
                description = ''
            catalog_as_one_string.append(('\t{} h: {}, d: {}'.format(
                offer, header[:h_idx],
                description[:d_idx])))

    # display the catalog
    display_fun('\n'.join(catalog_as_one_string))


def get_x_y_from_dataframe(catalog_id, catalogs_df):
    """Get a 1-elem torch batch containing catalog x and y,
    from a pandas dataframe."""
    x = catalogs_df.loc[catalogs_df['catalog_id'] == catalog_id]['x'].values
    x = ast.literal_eval(x[0])
    x = torch.from_numpy(np.asarray(x, dtype=np.int32))
    x = x.unsqueeze(0)

    y = catalogs_df.loc[catalogs_df['catalog_id'] == catalog_id]['y'].values
    y = ast.literal_eval(y[0])
    y = torch.from_numpy(np.asarray(y, dtype=np.int32))
    y = y.unsqueeze(0)

    return x, y


def get_offer_ids_from_dataframe(catalog_id, catalogs_df):
    """Get the offer tokens for chosen catalog, from a pandas dataframe"""
    offer_ids = ast.literal_eval(
        catalogs_df.loc[catalogs_df['catalog_id'] == catalog_id][
            'offer_ids_with_pb'].values[0])
    return offer_ids


def examine_single_procat_prediction(a_model, catalog_id, catalogs_df,
                                     offers_df, display_func):
    """
    Take a model and a catalogue id, find its features in the dataframes,
    make the model predict and then show / log both the correct catalog
    and the predicted one.
    """

    # get x and y
    x, y = get_x_y_from_dataframe(catalog_id, catalogs_df)
    display_func('x: {}'.format(x.size()))
    display_func('y: {}'.format(y.size()))

    # display correct catalog
    catalog_as_offer_ids = get_offer_ids_from_dataframe(catalog_id, catalogs_df)
    display_func('Original catalog:')
    display_catalog(catalog_as_offer_ids, offers_df, display_func)

    # find correct order of x
    correct_order = reorder(x, y)

    # map offer vectors (as bytes) to offer ids
    vectors_to_ids = {v.tobytes(): catalog_as_offer_ids[i] for i, v in
                      enumerate(correct_order)}

    # predict
    _, predicted_y = a_model(x)
    display_func('predicted y: {}'.format(predicted_y.size()))

    # restore order
    predicted_order = reorder(x, predicted_y)
    display_func('predicted order: {}'.format(len(predicted_order)))

    # restore predicted order as offer ids
    predicted_catalog = [vectors_to_ids[v.tobytes()] for v in predicted_order]
    display_func('Predicted catalog:')
    display_catalog(predicted_catalog, offers_df, display_func)


def examine_single_sentence_order_prediction(a_model, a_dataloader, a_tokenizer,
                                             a_dataset_name, a_logger, display_func):
    """
    Take a model and a catalogue id, find its features in the dataframes,
    make the model predict and then show / log both the correct catalog
    and the predicted one.
    """
    # handle gpu/cpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # set to eval
    a_model.eval()

    # get batch
    a_batch = next(iter(a_dataloader))

    # handle gpu
    if device == 'cuda':
        a_batch = {k: v.to(device) for k, v in a_batch.items()}

    # # get a single y
    # target_y = a_batch['label'][0]  # BERT specific?

    # predict
    _, predicted_y = a_model(a_batch)

    # easier to use existing function if we turn it into batches
    target_batch = copy.deepcopy(a_batch)
    pred_batch = copy.deepcopy(a_batch)
    pred_batch['label'] = predicted_y

    # move back to cpu
    if device == 'cuda':
        target_batch = {k: v.to('cpu') for k, v in target_batch.items()}
        pred_batch = {k: v.to('cpu') for k, v in pred_batch.items()}

    # target
    a_logger.info('Target sentence order:')
    display_func(target_batch, a_tokenizer, a_dataset_name, a_logger)

    a_logger.info('Predicted sentence order:')
    display_func(pred_batch, a_tokenizer, a_dataset_name, a_logger)


def run_synthetic_tests(a_model, a_dataloader, a_logger, a_config, batched=None):
    a_logger.info('Model: {}'.format(a_model.__class__))

    # handle gpu/cpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    a_model.eval().to(device)

    # general accuracy
    result_general, _ = test_model_synthetic_custom(a_model, a_dataloader,
                                                    compare_solved_sort_unique,
                                                    print_every=999999, x_name=0,
                                                    y_name=1)
    a_logger.info('Result: {:.4f}'.format(result_general))

    # functional metrics
    catalogs_testable, catalogs_corrent_n_tokens, \
    catalogs_num_offers_match, valid_sections_avg, \
    valid_sections_avg_basic, valid_sections_avg_green_only, \
    valid_sections_avg_purple_only, valid_sections_avg_purple_and_green, \
    valid_struture_avg, valid_struture_avg_basic, \
    valid_struture_avg_green_only, valid_struture_avg_purple_only, \
    valid_struture_avg_purple_and_green = test_model_functional(a_model,
                                                                a_dataloader,
                                                                a_config)

    result_tau, rank_valid_perc = get_batch_rank_correlation_and_perc_valid(
        a_dataloader, a_model, get_single_kendall_tau, a_logger, a_config, print_every=999999)
    a_logger.info('K-Tau: {:.4f}, perc_valid: {}'.format(result_tau, rank_valid_perc))

    result_spearman, rank_valid_perc = get_batch_rank_correlation_and_perc_valid(
        a_dataloader, a_model, get_single_spearman_rho, a_logger, a_config, print_every=999999)
    a_logger.info('S-Rho: {:.4f}, perc_valid: {}'.format(result_spearman,
                                                         rank_valid_perc))

    a_model.train()

    return result_general, result_tau, result_spearman, rank_valid_perc, catalogs_testable, catalogs_corrent_n_tokens, \
           catalogs_num_offers_match, valid_sections_avg, \
           valid_sections_avg_basic, valid_sections_avg_green_only, \
           valid_sections_avg_purple_only, valid_sections_avg_purple_and_green, \
           valid_struture_avg, valid_struture_avg_basic, \
           valid_struture_avg_green_only, valid_struture_avg_purple_only, \
           valid_struture_avg_purple_and_green


def test_model_functional(model, dataloader, a_config):
    # predict all catalogs from dataloader
    predicted_catalogs_as_indices = predict_synthetic_catalogs_as_indices(
        dataloader,
        model)
    predicted_catalogs_as_raw = synthetic_from_indices_to_raw(
        predicted_catalogs_as_indices, a_config)
    predicted_catalogs_as_instances = instantiate_synthetic_catalogs(
        predicted_catalogs_as_raw, a_config)

    # get structural / functional metrics
    metrics = get_rule_metrics(predicted_catalogs_as_indices, a_config['rules'],
                               a_config, dataset_type='test')

    # return unpacked
    # this has to be hardcoded for now

    # general
    catalogs_testable = metrics['valid_catalogs_%']
    catalogs_corrent_n_tokens = metrics['correct_n_tokens_%']
    catalogs_num_offers_match = metrics['num_offers_match_config_%']

    # sections
    valid_sections_avg = metrics['valid_sections_%_avg']
    valid_sections_avg_basic = metrics['valid_sections_%_per_ruleset']['basic']
    valid_sections_avg_green_only = metrics['valid_sections_%_per_ruleset'][
        'green_only']
    valid_sections_avg_purple_only = metrics['valid_sections_%_per_ruleset'][
        'purple_only']
    valid_sections_avg_purple_and_green = \
        metrics['valid_sections_%_per_ruleset']['purple_and_green']

    # structure
    valid_struture_avg = metrics['valid_structure_%_total']
    valid_struture_avg_basic = metrics['valid_structure_%_per_ruleset']['basic']
    valid_struture_avg_green_only = metrics['valid_structure_%_per_ruleset'][
        'green_only']
    valid_struture_avg_purple_only = metrics['valid_structure_%_per_ruleset'][
        'purple_only']
    valid_struture_avg_purple_and_green = \
        metrics['valid_structure_%_per_ruleset']['purple_and_green']

    return catalogs_testable, catalogs_corrent_n_tokens, \
           catalogs_num_offers_match, valid_sections_avg, \
           valid_sections_avg_basic, valid_sections_avg_green_only, \
           valid_sections_avg_purple_only, valid_sections_avg_purple_and_green, \
           valid_struture_avg, valid_struture_avg_basic, \
           valid_struture_avg_green_only, valid_struture_avg_purple_only, \
           valid_struture_avg_purple_and_green


def test_model_synthetic_custom(a_model, a_dataloader,
                                comparison_func=None, print_every=500,
                                x_name='X', y_name='Y'):
    """
    I would now like a function that takes a dataloader of test data and a model package,
    making it predict on each and then outputs the average loss.
    """
    # handle gpu/cpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # metrics, placeholders
    num_examples = len(a_dataloader.dataset)
    individual_scores = []
    counter = 0

    # iterate over dataset to predict and track
    # no grad here
    with torch.no_grad():
        for single_batch in a_dataloader:

            train_batch = Variable(single_batch[x_name])
            target_batch = Variable(single_batch[y_name])

            # move batch to the right device
            train_batch = train_batch.to(device)
            target_batch = target_batch.to(device)

            # predict
            o, batched_predictions = a_model(train_batch)

            # track
            for idx, model_solution in enumerate(batched_predictions):

                # compare solutions (might need a custom function per dataset)
                current_score = comparison_func(prediction=model_solution,
                                                solution=target_batch[idx])
                individual_scores.append(current_score)

                # update counter & report
                counter += 1
                if counter % print_every == 0:
                    print('... Calculating example {} / {} ...'.format(
                        counter, num_examples
                    ))

        final_score = sum(individual_scores) / num_examples

    return final_score, individual_scores


def run_tsp_tests(a_model, a_dataset, logger, config, batched=False):
    """
    I would now like a function that takes a dataloader of test data and a model package,
    making it predict on each and then outputs the average tour length and legal solutions percentage.
    Ideally it can also take the actual solver to see the target tour length and a 100% legal ratio.
    """
    # handle gpu/cpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    a_model.eval().to(device)

    # metrics, placeholders
    num_examples = a_dataset.dataset.data_size
    num_legal = 0
    tour_lengths_predicted = []
    tour_lengths_target = []

    logger.info('Begining to run_tsp_tests() ... ')

    # iterate over dataset to predict and track
    # no grad here
    if batched:
        with torch.no_grad():
            for single_batch in a_dataset:

                train_batch = Variable(single_batch[config['x_name']])
                target_batch = Variable(single_batch[config['y_name']])

                # move batch to the right device
                train_batch = train_batch.to(device)
                target_batch = target_batch.to(device)

                # predict
                o, batched_predictions = a_model(train_batch)

                # move batch back to cpu for metrics
                batched_predictions = batched_predictions.to('cpu')
                train_batch = train_batch.to('cpu')
                target_batch = target_batch.to('cpu')

                # track
                for idx, model_solution in enumerate(batched_predictions):
                    if is_path_legal(model_solution.int()):
                        num_legal += 1
                    tour_length_predicted = get_tour_length(train_batch[idx], model_solution.int())
                    tour_length_target = get_tour_length(train_batch[idx], target_batch[idx])
                    tour_lengths_predicted.append(tour_length_predicted)
                    tour_lengths_target.append(tour_length_target)

            # post-process
            perc_legal = num_legal * 100 / num_examples
            avg_tl_target = sum(tour_lengths_target) / len(tour_lengths_target)
            avg_tl_predicted = sum(tour_lengths_predicted) / len(tour_lengths_predicted)

    # not batched
    else:
        with torch.no_grad():
            for single_batch in a_dataset:

                # get a single batch
                train_batch = Variable(single_batch[config['x_name']])
                target_batch = Variable(single_batch[config['y_name']])

                # move batch to the right device
                train_batch = train_batch.to(device)
                target_batch = target_batch.to(device)

                # get a single element
                for i, train_example in enumerate(train_batch):
                    train_example = train_batch[i]
                    target_example = target_batch[i]

                    # predict
                    pointer_attentions, model_solution = a_model(train_example)

                    # move batch back to cpu for metrics
                    model_solution = model_solution.to('cpu')
                    train_batch = train_batch.to('cpu')
                    target_batch = target_batch.to('cpu')

                    # track
                    if is_path_legal(model_solution.int()):
                        num_legal += 1
                    tour_length_predicted = get_tour_length(train_example, model_solution.int())
                    tour_length_target = get_tour_length(train_example, target_example)
                    tour_lengths_predicted.append(tour_length_predicted)
                    tour_lengths_target.append(tour_length_target)

            # post-process
            perc_legal = num_legal * 100 / num_examples
            avg_tl_target = sum(tour_lengths_target) / len(tour_lengths_target)
            avg_tl_predicted = sum(tour_lengths_predicted) / len(tour_lengths_predicted)

    # let's also get rank correlation coefficients
    result_tau, rank_valid_perc = get_batch_rank_correlation_and_perc_valid(
        a_dataset, a_model, get_single_kendall_tau, logger, config,
        print_every=999999)
    logger.info('K-Tau: {:.4f}, perc_valid: {}'.format(result_tau,
                                                       rank_valid_perc))

    result_spearman, rank_valid_perc = get_batch_rank_correlation_and_perc_valid(
        a_dataset, a_model, get_single_spearman_rho, logger, config,
        print_every=999999)
    logger.info('S-Rho: {:.4f}, perc_valid: {}'.format(result_spearman,
                                                       rank_valid_perc))

    logger.info('TSP test results:')
    logger.info('Percentage valid tours: {:.2f}% of {}'.format(perc_legal, a_dataset.dataset.data_size))
    logger.info('Average predicted tour length: {:.5f}'.format(avg_tl_predicted))
    logger.info('Average target tour length: {:.5f}'.format(avg_tl_target))

    return perc_legal, round(avg_tl_target, 5), round(avg_tl_predicted, 5), result_tau, \
           result_spearman, rank_valid_perc
