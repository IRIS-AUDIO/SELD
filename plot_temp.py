import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import math


args = argparse.ArgumentParser()
args.add_argument('--keyword', type=str, default='val_auc')


if __name__ == '__main__':
    config = args.parse_args()

    records = [
    ]

    keyword = config.keyword
    perfs = []
    labels = []

    outputs = {
        'filters0': [],
        'depth0': [],
        'groups0': [],
        'filters1': [],
        'depth1': [],
        'strides1': [],
        'groups1': [],
        'bottleneck_ratio': [],
        keyword: []
    }

    pairs = []
    for j in ['vad_6-0_results', 'vad_6-1_results']:
        if not j.endswith('.json'):
            j = f'{j}.json'

        with open(j, 'r') as f:
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

    for pair in pairs:
        outputs['filters0'].append(pair['config']['BLOCK0_ARGS']['filters'])
        outputs['depth0'].append(pair['config']['BLOCK0_ARGS']['depth'])
        outputs['groups0'].append(pair['config']['BLOCK0_ARGS']['groups'])

        outputs['filters1'].append(pair['config']['BLOCK1_ARGS']['filters'])
        outputs['depth1'].append(pair['config']['BLOCK1_ARGS']['depth'])
        outputs['strides1'].append(pair['config']['BLOCK1_ARGS']['strides'][-1])
        outputs['groups1'].append(pair['config']['BLOCK1_ARGS']['groups'])
        outputs['bottleneck_ratio'].append(
            pair['config']['BLOCK1_ARGS']['bottleneck_ratio'])

        score = pair['perf'][keyword]
        if isinstance(score, list):
            score = score[-1]
        outputs[keyword].append(score)

    mapper = {0: 'filters0',
              1: 'depth0',
              2: 'groups0',
              3: 'filters1',
              4: 'depth1',
              5: 'strides1',
              6: 'groups1',
              7: 'bottleneck_ratio'}
    xs = np.array([outputs[mapper[i]] for i in range(8)],
                  dtype=np.float32)
    xs = xs.T
    ys = np.array(outputs[keyword], dtype=np.float32)

    from scipy.stats import ks_2samp
    from itertools import combinations

    for i in range(8):
        print(mapper[i])
        stats = []
        unique_values = np.unique(xs[..., i])
        for value in unique_values:
            stats.append(ys[xs[..., i] == value])

        # comb = list(combinations(range(len(unique_values)), 2))
        comb = [[j, k] for j in range(len(unique_values))
                       for k in range(len(unique_values))]

        for j, k in comb:
            pvalue = ks_2samp(stats[j], stats[k]).pvalue
            if pvalue < 0.2 and i != k:
                print(unique_values[j], unique_values[k], pvalue, sep='\t')
        print()

