from config import args
import tensorflow as tf
from tensorbayes.tbutils import clip_gradients
from adamax import AdamaxOptimizer
import os

AdamOptimizer = tf.train.AdamOptimizer

def make_file_name():
    log_file_format = 'results_{:s}/task={:s}/n_label={:05d}/run={:02d}.csv'
    args_slice = lambda args: (args.model, args.task, args.n_label, args.run)
    log_file = log_file_format.format(*args_slice(args))

    if args.run >= 999:
        return log_file
    else:
        while os.path.exists(log_file):
            args.run += 1
            args.seed += 1
            log_file = log_file_format.format(*args_slice(args))
        return log_file

def build_optimizer(loss, update_ops=[], scope=None, reuse=None):
    with tf.variable_scope(scope, 'gradients', reuse=reuse):
        print "Building optimizer"
        optimizer = AdamaxOptimizer(args.lr) if args.adamax else AdamOptimizer(args.lr)
        # max clip and max norm hyperparameters from Sonderby's LVAE code
        clipped, grad_norm = clip_gradients(optimizer, loss,
                                            max_clip=0.9,
                                            max_norm=4)
        with tf.control_dependencies(update_ops):
            train_step = optimizer.apply_gradients(clipped)
    return train_step
