import os
import yaml

class Config(dict):
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self._yaml = f.read()
            # use safe_load to avoid requiring an explicit Loader and to mitigate yaml vulnerabilities
            self._dict = yaml.safe_load(self._yaml)
            self._dict['PATH'] = os.path.dirname(config_path)

    def __getattr__(self, name):
        if self._dict.get(name) is not None:
            return self._dict[name]

        if DEFAULT_CONFIG.get(name) is not None:
            return DEFAULT_CONFIG[name]

        return None

    def print(self):
        print('Model configurations:')
        print('---------------------------------')
        print(self._yaml)
        print('')
        print('---------------------------------')
        print('')

DEFAULT_CONFIG = {
    'MODE': 1,                      # 1: train, 2: test, 3: eval
    'MODEL': 2,                     # 2: inpaint model
    'MASK': 3,                      # 3: external
    'NMS': 1,
    'SEED': 10,                     # random seed
    'GPU': [0],                     # list of gpu ids
    'DEBUG': 0,                     # turns on debugging mode
    'VERBOSE': 0,                   # turns on verbose mode in the output console

    'LR': 0.0001,                   # learning rate
    'D2G_LR': 0.1,                  # discriminator/generator learning rate ratio
    'BETA1': 0.0,                   # adam optimizer beta1
    'BETA2': 0.9,                   # adam optimizer beta2
    'BATCH_SIZE': 8,                # input batch size for training
    'INPUT_SIZE': 256,              # input image size for training 0 for original size
    'SIGMA': 2,                     # standard deviation of the Gaussian filter used in Canny edge detector (0: random, -1: no edge)
    'MAX_ITERS': 2e6,               # maximum number of iterations to train the model

    'EDGE_THRESHOLD': 0.5,          # edge detection threshold
    'L1_LOSS_WEIGHT': 1,            # l1 loss weight
    'COARSE_L1_WEIGHT': 1.0,        # weight for coarse branch L1
    'FM_LOSS_WEIGHT': 10,           # feature-matching loss weight
    'STYLE_LOSS_WEIGHT': 1,         # style loss weight
    'CONTENT_LOSS_WEIGHT': 1,       # perceptual loss weight
    'INPAINT_ADV_LOSS_WEIGHT': 0.01,# adversarial loss weight

    # diffusion refiner
    'DIFF_ENABLE': True,
    'DIFF_FREEZE_COARSE': 1,        # 1 freeze coarse generator, 0 finetune
    'DIFF_LR': 1e-4,
    'DIFF_TRAIN_STEPS': 50,         # T for training noise schedule
    'DIFF_INFER_STEPS': 8,          # fast sampling steps
    'DIFF_BETA_START': 1e-4,
    'DIFF_BETA_END': 2e-2,
    'DIFF_DDIM_ETA': 0.0,
    'DIFF_TIME_EMB_DIM': 256,
    'DIFF_BASE_CH': 64,
    'DIFF_L2_WEIGHT': 1.0,
    'DIFF_LPIPS_WEIGHT': 0.5,

    # adaptive fusion time conditioning
    'AFN_TIME_COND_ENABLE': False,   # enable time-conditioned fusion weights
    'AFN_TIME_STEPS': 1000,          # max timestep sampled during training
    'AFN_TIME_EMB_DIM': 128,         # sinusoidal embedding dimension for fusion
    'AFN_INFER_T': 0,                # default inference timestep (0 = earliest / more texture bias)

    'GAN_LOSS': 'nsgan',            # nsgan | lsgan | hinge
    'GAN_POOL_SIZE': 0,             # fake images pool size

    'SAVE_INTERVAL': 1000,          # how many iterations to wait before saving model (0: never)
    'SAMPLE_INTERVAL': 1000,        # how many iterations to wait before sampling (0: never)
    'SAMPLE_SIZE': 12,              # number of images to sample
    'EVAL_INTERVAL': 0,             # how many iterations to wait before model evaluation (0: never)
    'LOG_INTERVAL': 10,             # how many iterations to wait before logging training status (0: never)
}
