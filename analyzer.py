import argparse
import json
import numpy as np
from itertools import combinations
from scipy.stats import ks_2samp


args = argparse.ArgumentParser()
args.add_argument('--results', type=str, default='vad_6-0_results.json')
args.add_argument('--keyword', type=str, default='val_auc')
args.add_argument('--count1d', action='store_true')
args.add_argument('--stagewise', action='store_true')
args.add_argument('--filters', action='store_true')
args.add_argument('--var_of_interest', type=str, default=None)
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
    return pvalues


if __name__ == '__main__':
    config = args.parse_args()
    keyword = config.keyword
    pairs = []

    # 1. load results
    for j_file in config.results.split(','):
        if not j_file.endswith('.json'):
            j_file = f'{j_file}.json'

        with open(j_file, 'r') as f:
            results = json.load(f)

        for key in results.keys():
            if key.isdigit():
                pairs.append(results[key])

    # 1.1 black list
    for stage in config.black_list.split(','):
        pairs = filter_fn(
            pairs,
            lambda x: count_blocks(x['config'], lambda x: x == stage) == 0)

    # 1.2 add f1score
    for pair in pairs:
        precision = pair['perf']['val_precision'][0]
        recall = pair['perf']['val_recall'][0]
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        pair['perf']['val_f1score'] = f1

    # 2. common feature extractor
    feats = extract_feats_from_pairs(pairs)
    print(feats)

    # 2.2 make table
    table = {feat: [] for feat in feats}
    table[keyword] = []

    for pair in pairs:
        for feat in feats.keys():
            # find value
            if '.' in feat:
                front, end = feat.split('.')
                value = pair['config'][front][end]
            else:
                value = pair['config'][feat]
            if isinstance(value, list):
                value = str(value)

            table[feat].append(value)

        k_value = pair['perf'][keyword]
        if isinstance(k_value, list):
            k_value = k_value[-1]
        table[keyword].append(k_value)

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

    if config.filters:
        table['filters'] = np.sign(np.array(table['BLOCK1_ARGS.filters']) 
                                   -np.array(table['BLOCK0_ARGS.filters']))

    # 3. value
    table = {k: np.array(v) for k, v in table.items()}

    # 3.1 find significant variables
    # [(var, values, perfs, pvalues, n_samples)]
    same_candidates = []
    diff_candidates = []

    for rv in table.keys():
        if rv == keyword:
            continue

        unique_values = sorted(np.unique(table[rv]))
        if len(unique_values) == 1:
            continue

        perfs = [table[keyword][table[rv] == value]
                 for value in unique_values]
        pvalues = get_ks_test_values(
            unique_values, perfs, 
            min_samples=config.min_samples, a=config.a)
        n_samples = [len(p) for p in perfs]

        min_pvalues = [min(pv) for pv in pvalues if len(pv) > 0]
        max_pvalues = [max(pv) for pv in pvalues if len(pv) > 0]

        if len(min_pvalues) == 0:
            continue

        if max(min_pvalues) > 1-config.a:
            same_candidates.append(
                (rv, unique_values, perfs, pvalues, n_samples))
        if min(max_pvalues) < config.a:
            diff_candidates.append(
                (rv, unique_values, perfs, pvalues, n_samples))

    # 3.2 similar groups 
    print([rv for rv, *_ in same_candidates])

    # 3.3 different groups
    print([rv for rv, *_ in diff_candidates])
    for rv, unique_values, perfs, pvalues, *_ in diff_candidates:
        controlled_pvalues = [[] for _ in range(len(unique_values))]

        # collect conditional pvalues
        for control in table.keys():
            if control == keyword or control == rv:
                continue

            unique_c_values = np.unique(table[control])
            if len(unique_c_values) == 1:
                continue # var with single category

            for c_value in unique_c_values:
                c_perfs = [
                    table[keyword][(table[control] == c_value)
                                   * (table[rv] == value)]
                    for value in unique_values]
                c_pvalues = get_ks_test_values(
                    unique_values, c_perfs, 
                    min_samples=config.min_samples, a=config.a)
                
                for i, pv in enumerate(c_pvalues):
                    if len(pv) > 0 and max(pv) > 0.5:
                        print(rv, unique_values[i], control, c_value)
                    controlled_pvalues[i].extend(pv)
        
        print(f'Var of interest: {rv}')
        for i, v in enumerate(unique_values):
            if len(controlled_pvalues[i]) > 0:
                print(f'{v}: [{min(controlled_pvalues[i]):.5f},'
                      f'{max(controlled_pvalues[i]):.5f}] '
                      f'({np.mean(controlled_pvalues[i]):.5f})')
                print(f'prev pvalues: [{min(pvalues[i]):.5f}, '
                      f'{max(pvalues[i]):.5f}]')
                print(f'perfs: {np.mean(perfs[i]):.5f}')
        print()

