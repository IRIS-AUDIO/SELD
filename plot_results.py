import argparse
import json
import matplotlib.pyplot as plt
import numpy as np

import matplotlib
matplotlib.use('TkAgg')

args = argparse.ArgumentParser()
args.add_argument('--results', type=str, default='vad_results.json')
args.add_argument('--keyword', type=str, default='val_auc')
args.add_argument('--keyword2', type=str, default='val_f1score')
args.add_argument('--stage', type=str, default=None)
args.add_argument('--n_stage', type=int, default=3)
args.add_argument('--count1d', action='store_true')
args.add_argument('--black_list', type=str, default='')
args.add_argument('--min_samples', type=int, default=1)
args.add_argument('--test', action='store_true')


stages_1d = ['bidirectional_GRU_stage',
             'transformer_encoder_stage',
             'simple_dense_stage',
             'conformer_encoder_stage']
stages_2d = ['simple_conv_stage',
             'another_conv_stage',
             'res_basic_stage',
             'res_bottleneck_stage',
             'dense_net_stage', 
             # 'sepformer_stage',
             'xception_basic_stage']
             # 'identity_block']
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

    '''
    pairs = filter_fn(
        pairs,
        lambda x: x['config']['BLOCK0'] != 'res_bottleneck_stage')
    pairs = filter_fn(
        pairs,
        lambda x: x['config']['BLOCK1'] == 'res_bottleneck_stage')
    '''

    perfs = np.array([x['perf'][keyword][0] for x in pairs])
    print('total', np.mean(perfs), np.std(perfs), len(perfs))
    # plot total
    # pairs = sort_pairs(pairs, keyword=keyword)
    # plot_pairs(pairs, keyword=keyword, label=f'total({len(pairs)})') 
    # print(pairs[:5])

    if config.test:
        import stage_complexity
        def extract_feats(config):
            config = config['config']
            keys = get_block_keys(config)
            '''
            strides = [
                config[f'{key}_ARGS'].get('strides', 
                    config[f'{key}_ARGS'].get('pool_size', [1, 1]))
                for key in keys]
            result = [0, 0] # , 0, 0]
            for i, s in enumerate(strides):
                result[i] = s[-1] # s[-1]-1] += 1
            return result
            '''
            # WIDTH
            shape = [7, 80, 1]
            feats = []
            cxs = []
            for key in keys:
                cx, shape = getattr(stage_complexity,
                                    f'{config[f"{key}"]}_complexity')(
                    config[f'{key}_ARGS'], shape)
                feats.append(shape[-1])
                cxs.append(cx)

            mapper = {
                'simple_conv_stage': 1,
                'another_conv_stage': 2,
                'res_basic_stage': 2,
                'res_bottleneck_stage': 3,
                'dense_net_stage': 2,
                'xception_basic_stage': 3,
            }

            feats = [config[f'{key}_ARGS']['depth'] * 1 # mapper[config[key]]
                     for key in keys]
            if feats[0] > feats[1]:
                result = -1
            elif feats[0] < feats[1]:
                result = 1
            else:
                result = 0
            return result

        def extract_feats3(config):
            config = config['config']
            keys = get_block_keys(config)
            feats = [config[f'{key}_ARGS']['groups'] for key in keys]
            feats = config['BLOCK1_ARGS']['bottleneck_ratio']
            result = 0

            feats = [config['BLOCK0_ARGS']['groups'],
                     config['BLOCK1_ARGS']['groups']]
            return feats

        combinations = [[j, i] for i in [0, 0.5, 1] for j in [0, 0.5, 1]]
        # combinations = [0.25, 0.35, 0.5, 0.7, 1, 1.41, 2, 2.83, 4]
        combinations = [[i, j] for i in range(1, 4) for j in range(1, 4)]
        combinations = [[i, j] for i in [0, 0.5, 1] for j in [0, 0.5, 1]]
        # combinations = [1, 2]
        combinations = [-1, 0, 1] 

        perfs = [[np.log(1-x['perf'][keyword][0]) for x in pairs]]

        labels = ['total']
        for i, s in enumerate(combinations):
            new_pairs = filter_fn(
                pairs,
                lambda x: extract_feats(x) == s)
            new_pairs = sort_pairs(new_pairs, keyword=keyword)
            labels.append(f'{s}({len(new_pairs)})')

            if len(new_pairs) > 0:
                perfs.append([np.log(1-x['perf'][keyword][0]) for x in new_pairs])
            else:
                perfs.append([np.mean(perfs[0])] * 3)

        '''
        labels = ['filter0', 'filter1', 'depth0', 'depth1',
                  'groups0', 'groups1', 'strides', 'bottleneck_ratio']
        perfs = [
            # [p['config']['BLOCK0_ARGS']['filters'] for p in pairs],
            # [p['config']['BLOCK1_ARGS']['filters'] for p in pairs],
            [p['config']['BLOCK0_ARGS']['depth'] for p in pairs],
            [p['config']['BLOCK1_ARGS']['depth'] for p in pairs],
            # [p['config']['BLOCK0_ARGS']['groups'] for p in pairs],
            # [p['config']['BLOCK1_ARGS']['groups'] for p in pairs],
            # [p['config']['BLOCK1_ARGS']['strides'][-1] for p in pairs],
            # [p['config']['BLOCK1_ARGS']['bottleneck_ratio'] for p in pairs],
        ]
        plt.plot(perfs)
        '''

        fig, ax = plt.subplots()
        ax.set_xticks(np.arange(1, len(labels)+1))
        ax.set_xticklabels(labels)
        ax.violinplot(perfs, showmeans=True, showmedians=True)
    elif config.stage:
        for i in range(config.n_stage + 1):
            new_pairs = filter_fn(
                pairs,
                lambda x: count_blocks(x['config'], 
                                       lambda x: x == config.stage) == i)

            if len(new_pairs) >= config.min_samples:
                plot_pairs(sort_pairs(new_pairs, keyword=keyword), 
                           keyword=keyword,
                           label=f'{i} {config.stage}({len(new_pairs)})')
    elif config.count1d:
        for i in range(config.n_stage + 1):
            new_pairs = filter_fn(pairs, 
                                  lambda x: count_blocks(x['config']) == i)
            if len(new_pairs) >= config.min_samples:
                plot_pairs(sort_pairs(new_pairs, keyword=keyword), 
                           keyword=keyword,
                           label=f'{i} 1d stages({len(new_pairs)})')
    else:
        for i, stage in enumerate(stages_total):
            new_pairs = filter_fn(
                pairs,
                # lambda x: x['config']['BLOCK1'] == stage)
                lambda x: count_blocks(x['config'], 
                                       lambda x: x == stage) > 0)
            if len(new_pairs) >= config.min_samples:
                plot_pairs(sort_pairs(new_pairs, keyword=keyword), 
                           keyword=keyword,
                           label=f'>0 {stage}({len(new_pairs)})')

    plt.title(keyword)
    plt.legend()
    plt.show()

    print(len(pairs))

