import json
from glob import glob
import argparse


args = argparse.ArgumentParser()
args.add_argument('--name', type=str, default='2021_1')


def main():
    config = args.parse_args()
    name = config.name
    paths = sorted(glob(f'./{name}*'))
    merged_json = {}
    for idx, path in enumerate(paths):
        with open(path, 'r') as f:
            tmp = json.load(f)
        if idx == 0:
            merged_json = tmp
        else:
            length = len(merged_json)
            for key, val in tmp.items():
                if key != 'train_config':
                    merged_json[f'{int(key) + length - 1:03}'] = val

    with open(f'merged_{name}.json', 'w') as f:
        json.dump(merged_json, f, indent=4)


if __name__ == '__main__':
    main()

