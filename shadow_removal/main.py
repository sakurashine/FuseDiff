import os
import cv2
import random
import numpy as np
import torch
import argparse
from shutil import copyfile
from src.config import Config
from src.lifsr import LIFSR
import torch.nn as nn
import ast
import time

def main(mode=None):


    config = load_config(mode)


    # cuda visble devices
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)


    # init device
    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
    else:
        config.DEVICE = torch.device("cpu")



    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    cv2.setNumThreads(0)


    # initialize random seed
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)



    # build the model and initialize
    model = LIFSR(config)
    model.load()

    num_all = sum(p.numel() for p in model.inpaint_model.generator.parameters())

    iteration = model.inpaint_model.iteration
    if len(config.GPU) > 1:
        print('GPU:{}'.format(config.GPU))
        model.inpaint_model.generator = nn.DataParallel(model.inpaint_model.generator, config.GPU)

    model.inpaint_model.iteration = iteration



    # model training
    if config.MODE == 1:
        print('\nstart training...\n')
        model.train()

    # model test
    elif config.MODE == 2:
        print('\nstart testing...\n')
        test_start_time = time.time()
        model.test()
        test_end_time = time.time()
        print(f"Total testing time: {test_end_time - test_start_time:.2f} seconds")

    # eval mode
    else:
        print('\nstart eval...\n')
        model.eval()


def load_config(mode=None):


    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '--checkpoints', type=str, default='./shadow_removal/checkpoints', help='model checkpoints path (default: ./checkpoints)')
    parser.add_argument('--config', type=str, default='config_UAV-SC_tiles.yml', help='config file name or path (default: config.yml)')
    parser.add_argument('--override', nargs='*', default=[], help='Override config values, format: key=value')
    args = parser.parse_args()
    if os.path.isabs(args.config):
        config_path = args.config
    else:
        config_path = os.path.join(args.path, args.config)

    # create checkpoints path if does't exist
    if not os.path.exists(args.path):
        os.makedirs(args.path)

    # copy config template if does't exist
    if not os.path.exists(config_path):
        copyfile('./config.yml.example', config_path)

    # load config file
    config = Config(config_path)

    for override in args.override:
        if '=' in override:
            key, value_str = override.split('=', 1)
            
            keys = key.split('.')
            
            def parse_value(value_str):
                value_str = value_str.strip()
                if value_str.startswith('"') and value_str.endswith('"'):
                    return value_str[1:-1]
                if value_str.startswith("'") and value_str.endswith("'"):
                    return value_str[1:-1]
                
                try:
                    return ast.literal_eval(value_str)
                except (ValueError, SyntaxError):
                    return value_str
            
            value = parse_value(value_str)
            
            target = config
            for k in keys[:-1]:
                target = getattr(target, k)
            setattr(target, keys[-1], value)

    # train mode
    if mode == 1:
        config.MODE = 1

    # test mode
    elif mode == 2:
        config.MODE = 2

    # eval mode
    elif mode == 3:
        config.MODE = 3
        config.MODEL = args.model if args.model is not None else 3

    return config


if __name__ == "__main__":
    main()
