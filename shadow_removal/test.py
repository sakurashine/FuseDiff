import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from main import main
main(mode=2)