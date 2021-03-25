import argparse
import json
import os
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
                      choices=['seldnet', 'seldnet_v1'])
    args.add_argument('--model_config', type=str, default='')
    
    # training
    args.add_argument('--lr', type=float, default=0.001)
    args.add_argument('--decay', type=float, default=0.9)
    args.add_argument('--batch', type=int, default=256)
    args.add_argument('--agc', type=bool, default=False)
    args.add_argument('--epoch', type=int, default=1000)
    args.add_argument('--loss_weight', type=str, default='1,1000')
    args.add_argument('--lr_patience', type=int, default=5, help='learning rate decay patience for plateau')
    args.add_argument('--patience', type=int, default=50, help='early stop patience, real using value is patience // loop_time')
    args.add_argument('--freq_mask_size', type=int, default=16)
    args.add_argument('--time_mask_size', type=int, default=24)
    args.add_argument('--loop_time', type=int, default=5, help='times of train dataset iter for an epoch')

    # metric
    args.add_argument('--lad_doa_thresh', type=int, default=20)

    config = args.parse_args()

    # model config
    if len(config.model_config) == 0:
        config.model_config = config.model
    model_config_name = config.model_config
    model_config = model_config_name + '.json'
    model_config = os.path.join('./model_config', model_config)
    
    if not os.path.exists(model_config):
        raise ValueError('Model config is not exists')
    model_config = argparse.Namespace(**json.load(open(model_config,'rb')))

    config.name = f'{config.model}_{model_config_name}_{config.doa_loss}_{config.name}'
    config = get_config(config.name, config, mode=config.config_mode)

    return config, model_config


if __name__ == '__main__':
    config = get_param()
    print(config)
