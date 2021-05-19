import argparse
import json
import numpy as np
from itertools import combinations
from scipy.stats import ks_2samp


args = argparse.ArgumentParser()
args.add_argument('--results', type=str, default='vad_6-0_results.json')
args.add_argument('--keyword', type=str, default='val_auc')
args.add_argument('--stage', type=str, default=None)
args.add_argument('--count1d', action='store_true')
args.add_argument('--black_list', type=str, default='')
args.add_argument('--min_samples', type=int, default=1)


stages_1d = ['bidirectional_GRU_stage',
             'transformer_encoder_stage',
             'simple_dense_stage',
             'conformer_encoder_stage']
stages_2d = ['simple_conv_stage',
             'another_conv_stage',
             'res_basic_stage',
             'res_bottleneck_stage',
             'dense_net_stage', 
             'sepformer_stage',
             'xception_basic_stage',
             'identity_block']
stages_total = stages_1d + stages_2d


def is_1d(block):
    return block in stages_1d


def get_block_keys(config):
    return sorted([key for key in config.keys()
                   if key.startswith('BLOCK') and not key.endswith('ARGS')])


def count_blocks(config, criteria=is_1d, include_seddoa=False):
    keys = get_block_keys(config)
    if include_seddoa:
        keys.extend(['SED', 'DOA'])
    return sum([criteria(config[key]) for key in keys])


def filter_fn(pairs, fn):
    return [pair for pair in pairs if fn(pair)]


def plot_pairs(pairs, keyword='test_f', label=None, plot=None):
    if plot is None:
        plot = plt
    plot.plot([x['perf'][keyword] for x in pairs], 
             np.linspace(0, 1, len(pairs)),
             label=label)


def sort_pairs(pairs, keyword='test_f'):
    return sorted(pairs, key=lambda x: x['perf'][keyword], reverse=True)


if __name__ == '__main__':
    config = args.parse_args()
    pairs = []

    for j_file in config.results.split(','):
        if not j_file.endswith('.json'):
            j_file = f'{j_file}.json'

        with open(j_file, 'r') as f:
            results = json.load(f)

        for key in results.keys():
            if key.isdigit():
                pairs.append(results[key])

    # add f1score
    for pair in pairs:
        precision = pair['perf']['val_precision'][0]
        recall = pair['perf']['val_recall'][0]
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        pair['perf']['val_f1score'] = f1

    keyword = config.keyword

    # black list
    for stage in config.black_list.split(','):
        pairs = filter_fn(
            pairs,
            lambda x: count_blocks(x['config'], lambda x: x == stage) == 0)

    # common feature extractor
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

    print(feats)

    # make table
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

    table = {k: np.array(v) for k, v in table.items()}

    for rv in feats.keys():
        print(rv)
        stats = []
        unique_values = np.unique(table[rv])
        for value in unique_values:
            stats.append(table[keyword][table[rv] == value])

        comb = list(combinations(range(len(unique_values)), 2))

        # import pdb; pdb.set_trace()

        for j, k in comb:
            if len(stats[j]) > config.min_samples \
                and len(stats[k]) > config.min_samples:
                pvalue = ks_2samp(stats[j], stats[k]).pvalue
                # if (pvalue < 0.1 or pvalue > 0.9) and j != k:
                print(unique_values[j], unique_values[k], pvalue, sep='\t')
        print()

