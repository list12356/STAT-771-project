
import argparse
from models.brs_gan.brs_gan import BRSGAN


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dir', default="out")
parser.add_argument('--alpha', type=float, default=0.2)
parser.add_argument('--l', type=float, default=1.0)
parser.add_argument('--sigma', type=int, default=0)
parser.add_argument('--mode', default="gradient")
parser.add_argument('--pac_num', type=int, default=1)
parser.add_argument('--gan', default="vanila")
parser.add_argument('--dataset', default='mnist')
args = parser.parse_args()

out_dir = args.dir
save_step = 1000
data_dim = 784
Z_dim = 100
search_num = 64
alpha = args.alpha
v = 0.02
_lambda = args.l
mode = args.mode
restore = False
D_lr = 1e-4
pac_num =  args.pac_num
sigma = args.sigma
gan = args.gan
dataset = args.dataset

model = BRSGAN(out_dir=out_dir, alpha=alpha, v=v, 
    _lambda=_lambda, sigma=sigma, mode=mode, pac_num=pac_num, gan_structure = gan, dataset=dataset)
model.train()
