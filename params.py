import argparse
import json
import os
import numpy as np 

from config_manager import get_config


def get_param(known=None):
    args = argparse.ArgumentParser()
    
    args.add_argument('--name', type=str, required=True)

    args.add_argument('--gpus', type=str, default='-1')
    args.add_argument('--resume', action='store_true')    
    args.add_argument('--abspath', type=str, default='/root/datasets')
    args.add_argument('--config_mode', type=str, default='')
    args.add_argument('--doa_loss', type=str, default='MSE', 
                      choices=['MAE', 'MSE', 'MSLE', 'MMSE'])
    args.add_argument('--model', type=str, default='seldnet', 
                      choices=['seldnet', 'seldnet_v1', 'seldnet_architecture',
                               'xception_gru'])
    args.add_argument('--model_config', type=str, default='')
    
    # training
    args.add_argument('--lr', type=float, default=0.001)
    args.add_argument('--decay', type=float, default=0.9)
    args.add_argument('--batch', type=int, default=256)
    args.add_argument('--epoch', type=int, default=1000)
    args.add_argument('--loss_weight', type=str, default='1,1000')
    args.add_argument('--patience', type=int, default=100)
    args.add_argument('--freq_mask_size', type=int, default=8)
    args.add_argument('--time_mask_size', type=int, default=24)
    args.add_argument('--loop_time', type=int, default=5, 
                      help='the number of iterations per epoch (trainset)')

    # metric
    args.add_argument('--lad_doa_thresh', type=int, default=20)

    config = args.parse_known_args(known)[0]
    
    if len(config.model_config) == 0:
        config.model_config = config.model
    config.model_config = os.path.splitext(config.model_config)[0]

    # config
    model_name = config.model
    if config.model_config != model_name:
        model_name = f'{model_name}_{config.model_config}'        
    config.name = f'{model_name}_{config.doa_loss}_{config.name}'
    config = get_config(config.name, config, mode=config.config_mode)

    # model config
    model_config_path = os.path.join('./model_config', 
                                     config.model_config+'.json')
    if not os.path.exists(model_config_path):
        raise ValueError(f'"{model_config_path}" does not exist')
    model_config = argparse.Namespace(
        **json.load(open(model_config_path, 'rb')))

    return config, model_config


if __name__ == '__main__':
    import sys
    config = get_param(sys.argv[1:])
    print(config)

