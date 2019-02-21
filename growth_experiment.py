import os
from utils.reporters import get_reporter
import data.data_loader as data
from copy import deepcopy
import argparse
from argparse import Namespace
from utils.mutli_gpu_runner import MultiGPURunner
import datetime
from opt.bnn_function_caller import BNNMLPFunctionCaller
from utils.nn_utils import get_dense_nn


parser = argparse.ArgumentParser(description='Script for dispatching train runs of BNNs over larger search spaces')
parser.add_argument('-d', '--depth', required=True, type=int)
parser.add_argument('-w', '--width', required=True, type=int)
parser.add_argument('-s', '--step', required=True, type=int) # number of exta neurons to add in
parser.add_argument('-ld', '--logdir')
parser.add_argument('-dd', '--datadir')
parser.add_argument('-ds', '--dataset')
parser.add_argument('-cm', '--commonname', default=None)
parser.add_argument('--gpu', nargs='+', type=int)
parser.add_argument('-nd', '--nodatetime', action='store_true')
args = parser.parse_args()

# Set up logging directory and grab the config file
date_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

if args.commonname is not None :
    if args.nodatetime:
        folder_name = args.commonname
    else:
        folder_name = f'{args.commonname}-{date_time}'
else:
    if not args.nodatetime:
        folder_name = f'{date_time}'
    else:
        raise ValueError('Must supply a common name, or set ude datetime to True')

results_dir = f'{args.logdir}/{args.dataset}/{folder_name}'

latest_dir = f'{args.logdir}/{args.dataset}/latest'

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

if os.path.islink(latest_dir):
    os.unlink(latest_dir)

os.symlink(os.path.abspath(results_dir), latest_dir)

max_layers = args.depth
max_layer_size = args.width
hidden_jump = args.step

seed = []

def mutate_point(point, points, stack):
    ret = []

    if len(point) < 3:
        new_point = deepcopy(point)
        new_point = point + [hidden_jump]
        if not new_point in points and not new_point in stack:
            ret.append(new_point)

    for i in range(len(point)):
        if point[i] + hidden_jump <= max_layer_size:
            new_point = deepcopy(point)
            new_point[i] += hidden_jump
            if not new_point in points and not new_point in stack:
                ret.append(new_point)

    return ret

points = []
stack = []
stack.append([])

while stack != []:
    point = stack.pop()
    points.append(point)
    stack = mutate_point(point, points, stack) + stack

points = sorted(points, key=lambda x: len(x))

params = {
            'learning_rate': 0.001,
            'prior_var': 1.,
            'hyperprior': True
        }

points = [(get_dense_nn(point), params) for point in points]

proceed = input(f'This will test {len(points)} networks, proceed? [y/N]')
if not proceed in ['y', 'yes']:
    import sys
    sys.exit()

if args.dataset == 'mnist':
    data_loader = data.ClassificationDataloader(args.dataset, args.datadir, batch_size=experiment_config['batch_size'])
    metric = 'test_acc'
else:
    data_loader = data.RegressionDataloaderFixedSplits(args.dataset, args.datadir)
    metric = 'test_ll_true'

REPORTER = get_reporter(open(os.path.join(results_dir, 'log'), 'w'))
train_params = Namespace(data_set=args.dataset,
                         data_dir=args.datadir,
                         tf_params={
                             'batchSize': 1000,
                             'epochs': 50000
                         },
                         metric=metric)

func_caller = BNNMLPFunctionCaller(data_loader, args.dataset, None, train_params,
                               reporter=REPORTER,
                               tmp_dir=None)

gpu_runner = MultiGPURunner(func_caller, gpu_ids=args.gpu, log_dir=results_dir, tmp_dir=os.path.join(results_dir, 'tmp'))
gpu_runner.run_points(points)

