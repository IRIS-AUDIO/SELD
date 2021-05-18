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
    for j in ['vad_7-0_results', 'vad_7-1_results']:
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
    pairs = [pair for pair in pairs 
             if pair['config']['BLOCK1_ARGS']['strides'][-1] == 2]

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

    xs = np.array([
        outputs['filters0'],
        outputs['depth0'],
        outputs['groups0'],
        outputs['filters1'],
        outputs['depth1'],
        # outputs['strides1'],
        outputs['groups1'],
        # outputs['bottleneck_ratio']
    ], dtype=np.float32)
    xs = xs.T
    ys = np.array(outputs[keyword], dtype=np.float32)

    import tensorflow as tf

    class DistLayer(tf.keras.layers.Layer):
        def __init__(self, num_feats=8):
            super(DistLayer, self).__init__()
            self.num_feats = num_feats

        def build(self, input_shape):
            sqrt8 = math.sqrt(8)

            # for relative
            self.offset = self.add_variable(
                'offset', initializer='random_normal', 
                shape=[1, self.num_feats])
            self.relations = self.add_variable(
                'relations', initializer='random_normal', 
                shape=[1, self.num_feats])
            self.f_weights = self.add_variable(
                'f_weights', initializer='random_normal', 
                shape=[1, self.num_feats])
            self.r_weight = self.add_variable(
                'r_weight', initializer='random_uniform', shape=[])

            self.y = self.add_variable(
                'y', initializer='random_uniform', shape=[])

        def call(self, inputs):
            gumbel_noise = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(inputs)) + 1e-8)+1e-8)
            f_weights = tf.keras.activations.sigmoid((self.f_weights + gumbel_noise) * 5)
            rel_dist = tf.keras.losses.cosine_similarity(
                self.relations * f_weights,
                (inputs+self.offset) * f_weights)
            rel_dist = (rel_dist + 1) \
                     * tf.keras.activations.softplus(self.r_weight)

            return - rel_dist + self.y # 0.92 # 0.9158343

    def generate_model(num_feats=8):
        inputs = tf.keras.layers.Input(shape=(num_feats,))
        outputs = DistLayer(num_feats)(inputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    model = generate_model(xs.shape[-1])
    model.compile(optimizer=tf.keras.optimizers.SGD(momentum=0.9), loss='mse')

    model.fit(xs, ys, batch_size=100, epochs=4000, shuffle=True, verbose=False)

    print('losses(MSE)', tf.keras.losses.MSE(ys, model(xs)))

    print('offset', model.weights[0].numpy())
    print('relations:', tf.math.l2_normalize(model.weights[1]).numpy())
    print('f_weights:', 
          tf.keras.activations.sigmoid(model.weights[-3] * 5).numpy())
    print('r_weight', tf.keras.activations.softplus(model.weights[-2]).numpy())
    print('imaginary y', model.weights[-1].numpy())

