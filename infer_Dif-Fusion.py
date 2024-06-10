import os
import sys
from pathlib import Path
from argparse import ArgumentParser

# run Dif-Fusion for the MF dataset
# modified from Dif-Fusion/t_fusion.py

parser = ArgumentParser()
parser.add_argument('-I', '--in_dp', type=Path, default='data/MF', help='input image path')
parser.add_argument('-O', '--out_dp', type=Path, default='data/MF_Dif-Fusion', help='output image path')
args = parser.parse_args()

# NOTE: first expand paths to absolute here!!
args.in_dp = Path(args.in_dp).absolute()
args.out_dp = Path(args.out_dp).absolute()
print('>> in_dp:', args.in_dp)
print('>> out_dp:', args.out_dp)
assert args.in_dp.is_dir()

# NOTE: then we can safely chdir to repo root path!!
try:
  os.chdir(os.path.join('repo', 'Dif-Fusion'))
  print('>> cwd:', os.getcwd())
  sys.path.append(os.getcwd())
except:
  print('>> Error: run repo\init_repos.cmd first')
  exit(0)


# ↓↓↓ copy & modify things from Dif-Fusion/t_fusion.py
