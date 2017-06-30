import argparse
import sys
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# General settings
parser.add_argument("--run",               type=int,   default=0,      help="Run index. Use 0 if first run.")
parser.add_argument("--n-save",            type=int,   default=50,     help="Number of epochs before save.")
parser.add_argument("--save-dir",          type=str,   default='/scratch/users/rshu15/bcde',
                                                                       help="Save model directory.")

# Data settings
parser.add_argument("--task",              type=str,   default='q2',   help="Data task")
parser.add_argument("--shift",             type=str,   default='none', help="Whether to shift data")
parser.add_argument("--n-label",           type=int,   default=25000,  help="Number of labeled data.")
parser.add_argument("--n-total",           type=int,   default=50000,  help="Total number of data.")
parser.add_argument('--seed',              type=int,   default=0,      help='Seed for semi-sup conversion')

# Model settings
parser.add_argument('--model',             type=str,   default='hybrid', choices=['hybrid', 'hybrid_factored', 'conditional', 'pretrained'],
                                                                       help='Type of model')
parser.add_argument("--z-size",            type=int,   default=50,     help="Size of z.")
parser.add_argument("--h-size",            type=int,   default=500,    help="Size of h.")
parser.add_argument("--nonlin",            type=str,   default='elu',  help="Activation function.")
parser.add_argument("--eps",               type=float, default=1e-5,   help="Distribution epsilon.")
parser.add_argument("--l2",                type=float, default=0.01,   help="L2 weight to use (if !NA)")

# Optimization settings
parser.add_argument("--n-epochs",          type=int,   default=300,    help="Number of epochs.")
parser.add_argument("--bs",                type=int,   default=100,    help="Minibatch size.")
parser.add_argument("--lr",                type=float, default=5e-4,   help="Learning rate.")
parser.add_argument("--adamax",            type=int,   default=0,      help="Adamax v. Adam")

# Log settings
parser.add_argument("--n-checks",          type=int,   default=50,     help="Number of IW=100 checks.")
parser.add_argument("--n-models",          type=int,   default=1,      help="Max number of models to save.")
parser.add_argument("--n-pretrain-epochs", type=int,   default=150,    help="Number of pretrain epochs (if !NA).")

if 'ipykernel' in sys.argv[0]:
    parser.set_defaults(run=999, seed=999)
    args = parser.parse_args([])
else:
    args = parser.parse_args()

if args.task in {'q2', 'td'}:
    args.x_size = 392
elif args.task == 'q3':
    args.x_size = 588
elif args.task == 'q1':
    args.x_size = 196
else:
    raise Exception('Unrecognized args.task')
args.y_size = 784 - args.x_size
