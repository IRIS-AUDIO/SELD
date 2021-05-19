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


def print_ks_test_values(values, stats, min_samples=1, a=0.05):
    comb = list(combinations(range(len(values)), 2))

    for j, k in comb:
        if len(stats[j]) >= min_samples and len(stats[k]) >= min_samples:
            pvalue = ks_2samp(stats[j], stats[k]).pvalue
            print(values[j], values[k], pvalue, sep='\t')
            if min(pvalue, 1-pvalue) < a and j != k:
                print(f'{stats[j].mean():.4f}\t{stats[k].mean():.4f}\t'
                      f'({len(stats[j])}\t{len(stats[k])})')
    print()


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

    if config.var_of_interest is not None:
        groups = np.unique(table[config.var_of_interest])
        print(groups)

    for rv in table.keys():
        if rv == keyword:
            continue

        print(f'--------        {rv}        --------')
        unique_values = np.unique(table[rv])

        if config.var_of_interest is None:
            stats = [table[keyword][table[rv] == value]
                     for value in unique_values]

            print_ks_test_values(unique_values, stats, 
                                 min_samples=config.min_samples, a=config.a)
        else:
            if rv == config.var_of_interest:
                continue

            if len(unique_values) == 1:
                continue

            for value in unique_values:
                print(f'{rv} == {value}')
                stats = [
                    table[keyword][(table[rv] == value)
                                   * (table[config.var_of_interest] == group)]
                    for group in groups]

                print_ks_test_values(groups, stats, 
                                     min_samples=config.min_samples, a=config.a)
        print()

