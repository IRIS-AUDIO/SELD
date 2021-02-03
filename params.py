from config_manager import get_config
import argparse
import numpy as np 

def get_param(known=[]):
    args = argparse.ArgumentParser()
    
    args.add_argument('--gpus', type=str, default='-1')
    args.add_argument('--name', type=str, default='')
    args.add_argument('--resume', action='store_true')    
    args.add_argument('--abspath', type=str, default='/root/datasets')
    args.add_argument('--config_mode', type=str, default='')
    

    # training
    args.add_argument('--lr', type=float, default=0.01)
    args.add_argument('--decay', type=float, default=1/np.sqrt(2))
    args.add_argument('--batch', type=int, default=256)
    args.add_argument('--epoch', type=int, default=50)
    args.add_argument('--loss_weight', type=str, default='1,1000')
    args.add_argument('--patience', type=int, default=10)

    # metric
    args.add_argument('--lad_doa_thresh', type=int, default=20)

    config = args.parse_known_args(known)[0]
    
    config = get_config(config.name, config, mode=config.config_mode)
    return config

if __name__ == '__main__':
    import sys
    config = getParam(sys.argv[1:])
    print(config)