import argparse
import json
import matplotlib.pyplot as plt
import numpy as np

import matplotlib
matplotlib.use('TkAgg')

args = argparse.ArgumentParser()
args.add_argument('--keyword', type=str, default='val_auc')


if __name__ == '__main__':
    config = args.parse_args()

    records = [
        ['stage1', ['vad_results', 'vad_0-0_results', 'vad_0-1_results']],
        ['stage2', ['vad_2-0_results', 'vad_2-1_results']],
        ['stage3', ['vad_3-0_results', 'vad_3-1_results']],
        ['stage4', ['vad_4-0_results', 'vad_4-1_results']],
        ['stage5', ['vad_5-0_results', 'vad_5-1_results']],
        ['stage6', ['vad_6-0_results', 'vad_6-1_results']],
        # ['stage7', ['vad_7-0_results', 'vad_7-1_results']],
        ['bdnn', ['bdnn_results']],
    ]

    keyword = config.keyword
    perfs = []
    labels = []

    for record in records:
        label, jsons = record
        pairs = []

        for j in jsons:
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

        scores = [x['perf'][keyword] for x in pairs]
        if isinstance(scores[0], list):
            scores = [s[-1] for s in scores]
        perfs.append(scores)
        labels.append(f'{label}({len(scores)})')

    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(1, len(labels)+1))
    ax.set_xticklabels(labels)
    ax.violinplot(perfs, showmeans=True, showmedians=True)

    plt.title(keyword)
    plt.show()

