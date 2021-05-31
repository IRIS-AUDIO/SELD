import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from scipy.stats import ks_2samp


args = argparse.ArgumentParser()
args.add_argument('--results', type=str, 
                  default='vad_6-0_results,vad_6-1_results')
args.add_argument('--keyword', type=str, default='val_auc')
args.add_argument('--keyword2', type=str, default='test_auc')
args.add_argument('--n_stages', type=int, default=3)
args.add_argument('--count1d', action='store_true')
args.add_argument('--stagewise', action='store_true')
args.add_argument('--stagewise_exist', action='store_true')
args.add_argument('--var_of_interest', type=str, default='')
args.add_argument('--verbose', action='store_true')
args.add_argument('--visualize', action='store_true')
args.add_argument('--black_list', type=str, default='')
args.add_argument('--a', type=float, default=0.05)
args.add_argument('--min_samples', type=int, default=1)


stages_1d = ['bidirectional_GRU_stage',
             'transformer_encoder_stage',
             'simple_dense_stage',
             'conformer_encoder_stage']


def is_1d(block):
    return block in stages_1d


def get_block_keys(config):
    return sorted([key for key in config.keys()
                   if key.startswith('BLOCK') and not key.endswith('ARGS')])


def count_blocks(config, criteria=is_1d):
    keys = get_block_keys(config)
    return sum([criteria(config[key]) for key in keys])


def filter_fn(pairs, fn):
    return [pair for pair in pairs if fn(pair)]


def extract_feats_from_pairs(pairs):
    feats = {}
    for pair in pairs:
        c = pair['config']
        for key in c.keys():
            if isinstance(c[key], dict):
                if key in feats:
                    feats[key] = [
                        feats[key][0].intersection(set(c[key].keys()))]
                else:
                    feats[key] = [set(c[key].keys())]
            else:
                if key in feats:
                    feats[key] = feats[key].union([c[key]])
                else:
                    feats[key] = set([c[key]])

    # features from *_ARGS
    keys = tuple(feats.keys())
    for key in keys:
        if not isinstance(feats[key], set):
            if len(feats[key][0]) > 0:
                for name in feats[key][0]:
                    new_name = f'{key}.{name}'
                    for pair in pairs:
                        value = pair['config'][key][name]
                        if isinstance(value, list):
                            value = str(value)
                        value = set([value])
                        if new_name in feats:
                            feats[new_name] = feats[new_name].union(value)
                        else:
                            feats[new_name] = value
            del feats[key]
    return feats


def print_ks_test_values(values, perfs, min_samples=1, a=0.05):
    comb = list(combinations(range(len(values)), 2))
    pvalues = [[] for _ in range(len(values))]

    for j, k in comb:
        if len(perfs[j]) >= min_samples and len(perfs[k]) >= min_samples:
            pvalue = ks_2samp(perfs[j], perfs[k]).pvalue
            pvalues[j].append(pvalue)
            pvalues[k].append(pvalue)

            print(values[j], values[k], pvalue, sep='\t')
            if min(pvalue, 1-pvalue) < a and j != k:
                print(f'{perfs[j].mean():.4f}\t{perfs[k].mean():.4f}\t'
                      f'({len(perfs[j])}\t{len(perfs[k])})')


def get_ks_test_values(values, perfs, min_samples=1, a=0.05, verbose=False):
    n_values = len(values)
    comb = list(combinations(range(n_values), 2))
    pvalues = [[] for _ in range(n_values)]

    for j, k in comb:
        if len(perfs[j]) >= min_samples and len(perfs[k]) >= min_samples:
            pvalue = ks_2samp(perfs[j], perfs[k]).pvalue
            pvalues[j].append(pvalue)
            pvalues[k].append(pvalue)

            if verbose:
                print(f'{values[j]}({len(perfs[j])})    vs    '
                      f'{values[k]}({len(perfs[k])}): {pvalue:.5f}')

    if verbose:
        print()
    return pvalues


if __name__ == '__main__':
    config = args.parse_args()
    keyword = config.keyword
    keyword2 = config.keyword2
    pairs = []

    # 1. load results
    for j_file in config.results.split(','):
        if not j_file.endswith('.json'):
            j_file = f'{j_file}.json'

        with open(j_file, 'r') as f:
            results = json.load(f)

        for key in results.keys():
            if key.isdigit():
                for i in range(config.n_stages):
                    c = results[key]['config']
                    if c[f'BLOCK{i}'] == 'mother_stage':
                        c_args = c[f'BLOCK{i}_ARGS']
                        if c_args['filters2'] == 0:
                            if c_args['connect2'][2] == 0:
                                c_args['filters1'] = 0

                                if c_args['connect2'][1] == 0:
                                    c_args['filters0'] = 0
                        
                        if c_args['filters0'] == 0:
                            c_args['kernel_size0'] = 0
                            c_args['connect1'][1] = 0
                            c_args['connect2'][1] = 0
                        if c_args['filters1'] == 1:
                            c_args['kernel_size1'] = 0
                            c_args['strides'] = [1, 1]
                            c_args['connect2'][2] = 0
                        if c_args['filters2'] == 2:
                            c_args['kernel_size2'] = 0

                        n_convs = ((c_args['filters0'] > 0)
                                   + (c_args['filters1'] > 0)
                                   + (c_args['filters2'] > 0))

                        if n_convs == 0:
                            c[f'BLOCK{i}'] = 'identity_stage'
                            c_args['depth'] = 1
                        c[f'BLOCK{i}_ARGS']['n_convs'] = n_convs

                        # close connection
                        c[f'BLOCK{i}_ARGS']['close_connect'] = 0
                        for j in range(3):
                            if c_args[f'filters{j}'] > 0 and c_args[f'connect{j}'][-1] == 0:
                                c[f'BLOCK{i}_ARGS']['close_connect'] += 1 / n_convs

                pairs.append(results[key])

    # 1.1 black list
    for stage in config.black_list.split(','):
        pairs = filter_fn(
            pairs,
            lambda x: count_blocks(x['config'], lambda x: x == stage) == 0)

    for pair in pairs:
        # 1.2 add f1score
        precision = np.squeeze(pair['perf']['val_precision'])
        recall = np.squeeze(pair['perf']['val_recall'])
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        pair['perf']['val_f1score'] = f1

        precision = np.squeeze(pair['perf']['test_precision'])
        recall = np.squeeze(pair['perf']['test_recall'])
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        pair['perf']['test_f1score'] = f1

        # 1.3 add first stage
        for i in range(config.n_stages):
            first_stage = pair['config'][f'BLOCK{i}']

            if not first_stage.startswith('identity'):
                break
        pair['config']['first_stage'] = first_stage

    # 2. common feature extractor
    feats = extract_feats_from_pairs(pairs)
    print(feats)

    # 2.2 make table
    table = {feat: [] for feat in feats}
    table[keyword] = []
    table[keyword2] = []

    for pair in pairs:
        for feat in feats.keys():
            # find value
            if '.' in feat:
                front, end = feat.split('.')
                value = pair['config'][front][end]
            else:
                value = pair['config'][feat]
            if not isinstance(value, (int, float, str)):
                value = str(value)

            table[feat].append(value)

        for kw in np.unique([keyword, keyword2]):
            k_value = pair['perf'][kw]
            if isinstance(k_value, list):
                k_value = k_value[-1]
            table[kw].append(k_value)

    # 2.3 add features
    if config.count1d:
        table['count1d'] = [count_blocks(p['config']) for p in pairs]

    if config.stagewise:
        total = [v for k, v in feats.items() if k.startswith('BLOCK')]
        stages = total[0]
        for s in total[1:]:
            stages = stages.union(s)

        for stage in stages:
            table[stage] = [count_blocks(p['config'], lambda p: p == stage)
                            for p in pairs]

    if config.stagewise_exist:
        total = [v for k, v in feats.items() if k.startswith('BLOCK')]
        stages = total[0]
        for s in total[1:]:
            stages = stages.union(s)

        for stage in stages:
            table[f'{stage}_exist'] = [
                count_blocks(p['config'], lambda p: p == stage) > 0
                for p in pairs]

    # 3. value
    table = {k: np.array(v) for k, v in table.items()}

    # pareto frontier
    scores = sorted(list(zip(table[keyword], table[keyword2])),
                    key=lambda x: -x[0])
    frontier = [[], []]
    criteria = -np.inf
    for s0, s1 in scores:
        if s1 > criteria:
            criteria = s1
            frontier[0].append(s0)
            frontier[1].append(s1)

    # 3.1 find significant variables
    for rv in table.keys():
        if rv in [keyword, keyword2]:
            continue

        if len(config.var_of_interest) > 0 and rv != config.var_of_interest:
            continue

        unique_values = sorted(np.unique(table[rv]))
        if len(unique_values) == 1:
            continue

        if config.visualize:
            for value in unique_values:
                mask = table[rv] == value
                plt.plot(table[keyword][mask], 
                         table[keyword2][mask], '.', label=value)
            plt.plot(*frontier, color='gray', alpha=0.5)
            plt.title(rv)
            plt.xlabel(keyword)
            plt.ylabel(keyword2)
            plt.legend()
            plt.show()
        else:
            print(f'{rv}')
            perfs = [table[keyword][table[rv] == value]
                     for value in unique_values]
            pvalues = get_ks_test_values(
                unique_values, perfs, 
                min_samples=config.min_samples, a=config.a, verbose=config.verbose)
            n_samples = [len(p) for p in perfs]

            for i, pv in enumerate(pvalues):
                if len(pv) > 0:
                    print(f'{unique_values[i]}: '
                          f'[{min(pv):.5f}, {max(pv):.5f}] '
                          f'({np.mean(pv):.5f}) '
                          f'n_samples={len(perfs[i])}, '
                          f'{keyword}(min={np.min(perfs[i]):.5f}, '
                          f'mean={np.mean(perfs[i]):.5f}, '
                          f'median={np.median(perfs[i]):.5f}, '
                          f'max={np.max(perfs[i]):.5f})')
            print()

