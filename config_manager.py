import json, argparse, os, pdb

def save_Config(path, name, config:dict):
    jsonpath = os.path.join(path, name)
    print(f'Save config as {name}')
    with open(jsonpath, 'w') as f:
        json.dump(config, f, sort_keys=True, indent=4)

def loadConfig(path, name):
    if not (os.path.splitext(name)[-1] == '.json'):
        name += '.json'
    jsonpath = os.path.join(path, name)
    if os.path.exists(jsonpath): # name.json exists
        with open(jsonpath, 'r') as f:
            res = json.load(f)
        return res
    else:
        print(f"{jsonpath} config don't exists")
        raise ValueError()

def manageVersion(path, name):
    from glob import glob
    configs = sorted(glob(os.path.splitext(os.path.join(path, name))[0] + '*'))
    latest = os.path.splitext(os.path.basename(configs[-1]))[0]
    oldversion = int(latest.split('v.')[-1])
    newversion = str(oldversion + 1)
    name = name.replace(f'_v.{oldversion}', f'_v.{newversion}')
    return name

def findDuplicateConfig(jsonpath, newconfig, mode):
    from glob import glob
    name = getName(jsonpath)
    for i in glob(os.path.dirname(os.path.abspath(jsonpath)) + f'/{name}*.json'):
        _config = loadConfig(os.path.dirname(i), os.path.basename(i))
        _config = manageMode(_config, mode)
        _config = manageGPU(_config)
        if _config == newconfig:
            return os.path.splitext(os.path.basename(i))[0]
    return False

def getName(name):
    return os.path.basename(os.path.splitext(name)[0].split('_v.')[0])

def manageGPU(config):
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = config['gpus']
        del config['gpus']
    except:
        pass
    return config

def overWriteConfig(loaded_config, config):
    ''' overwrite config to loaded_config '''
    for key in config.keys():
        loaded_config[key] = config[key]
    return loaded_config

def manageMode(config, mode):
    for key in config.keys():
        if config[key] == mode:
            del config[key]
            break
    return config

def getConfig(name:str, 
              config:argparse.Namespace, 
              path:str='./config', 
              mode:str=''):
    '''
    name: CONFIG.json name
    config: parsed config
    use_only_saved: if True, ignore your typed configuration
    path: CONFIG.json path
    mode: (l: loading config, o: only use loaded config), default is ''
    '''
    print('---------------- config manager start ----------------')
    modes = ('l', 'o')
    if mode == 'o':
        print("WARINING: you didn't load config.")
    assert len(name) > 0, 'name must be typed'
    for i in mode:
        assert i in modes, 'mode must be l, o, lo, or ol'

    if not os.path.exists(path): # directory manage
        os.makedirs(path)
        print(f'configuration directory is made at {path}')

    config = vars(config) # gpus delete
    config = manageMode(config, mode)
    config = manageGPU(config)
    
    name = name + '.json' if not('.json' in name) else name
    while True:
        if 'l' in mode:
            if not os.path.exists(os.path.join(path, name)):
                print(f'There is no {os.path.splitext(name)[0]} config')
                raise ValueError()
            loaded_config = loadConfig(path, name)

            if 'o' in mode:
                print(f'Loaded config name: {name}')
                print(f'Only use it')
                
                final_config = overWriteConfig(loaded_config, {'gpus': config.gpus})
                break
        elif 'o' in mode:
            print("You can't use only saved config without loading")
            raise ValueError()
        else:
            final_config = config
            name = os.path.splitext(name)[0] + '_v.0.json'
            final_config['name'] = os.path.splitext(name)[0]
            if not os.path.exists(os.path.join(path, name)):
                saveConfig(path, name, final_config)
            break
        
        dup = findDuplicateConfig(os.path.join(path, name), final_config, mode)
        if dup:
            print(dup, 'is the same config with you final config and change name')
            name = os.path.splitext(os.path.basename(dup))[0]
            final_config['name'] = os.path.splitext(name)[0]
            break
        name = manageVersion(path, name)
        saveConfig(path, name, final_config)
        break
    print('----------------- config manager end -----------------')
    return argparse.Namespace(**final_config)
        


if __name__ == "__main__":
    import sys
    arg = argparse.ArgumentParser()
    arg.add_argument('--hi', type=str, default='bye')
    config = arg.parse_known_args(sys.argv[1:])[0]
    getConfig('b_v.0)', config, False)