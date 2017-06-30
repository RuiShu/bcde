import tensorflow as tf
from config import args
from models import *
from utils import *
from data import Mnist
import tensorbayes as tb
from itertools import izip
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

def main():
    log_file = make_file_name()
    print args

    def evaluate(x, y, xu, yu, eval_tensors, iw=1):
        if iw == 1:
            xs, ys, xus, yus = [x], [y], [xu], [yu]
        else:
            batches = 2000
            xs, ys = list(tb.nputils.split(x, batches)), list(tb.nputils.split(y, batches))
            xus, yus = list(tb.nputils.split(xu, batches)), list(tb.nputils.split(yu, batches))

        values = []
        for x, y, xu, yu in zip(xs, ys, xus, yus):
            feed_dict = {T.x: x, T.xu: xu, T.y: y, T.yu: yu, T.phase: 0, T.u: u, T.iw: iw}
            v = T.sess.run(eval_tensors, feed_dict)
            values += [v]

        values = [np.mean(v).astype(v[0].dtype) for v in zip(*values)]
        return values

    def train(T_train_step, T_loss, data, iterep, n_epochs):
        for i in xrange(iterep * n_epochs):
            x, y, xu, yu = data.next_batch(args.bs)
            feed_dict = {T.x: x, T.xu: xu, T.y: y, T.yu: yu, T.phase: 1, T.u: u, T.iw: 1}
            _, loss = T.sess.run([T_train_step, T_loss], feed_dict)

            message = "loss: {:.2e}".format(loss)
            end_epoch, epoch = tb.utils.progbar(i, iterep, message, bar_length=5)

            if np.isnan(loss):
                print "NaN detected"
                quit()

            if end_epoch:
                iw = 100 if epoch % args.n_checks == 0 else 1
                tr_values = evaluate(data.x_label, data.y_label, data.x_train,
                                     data.y_train, writer.tensors, iw=1)
                va_values = evaluate(data.x_valid, data.y_valid, data.x_valid,
                                     data.y_valid, writer.tensors[:-1], iw=iw)
                te_values = evaluate(data.x_test, data.y_test, data.x_test,
                                     data.y_test, writer.tensors[:-1], iw=iw)
                values = tr_values + va_values + te_values + [epoch]
                writer.write(values=values)

    def make_writer():
        # Make log file
        writer = tb.FileWriter(log_file, args=args, pipe_to_sys=True, overwrite=args.run >= 999)
        # Train log
        writer.add_var('train_iw', '{:4d}', T.iw)
        for v in ['bcde', 'bjde_x', 'bjde_xy', 'bjde_xu', 'bjde_yu', 'loss']:
            writer.add_var('train_{:s}'.format(v), '{:8.3f}', T[v])
        writer.add_var('l2_loss', '{:9.2e}', T.l2)

        # Validation log
        writer.add_var('valid_iw', '{:4d}')
        for v in ['bcde', 'bcde_x', 'bjde_xy', 'bjde_xu', 'bjde_yu', 'loss']:
            writer.add_var('valid_{:s}'.format(v), '{:8.3f}')

        # Test log
        writer.add_var('test_iw', '{:4d}')
        for v in ['bcde', 'bcde_x', 'bjde_xy', 'bjde_xu', 'bjde_yu', 'loss']:
            writer.add_var('test_{:s}'.format(v), '{:8.3f}')

        # Extra info
        writer.add_var('epoch', '{:>8d}')
        writer.initialize()
        return writer

    ###############
    # Build model #
    ###############
    tf.reset_default_graph()
    T = tb.utils.TensorDict(dict(
        bcde=constant(0),
        bjde_x=constant(0),
        bjde_xu=constant(0),
        bjde_yu=constant(0),
        bjde_xy=constant(0),
        l2=constant(0),
        loss=constant(0)))
    T.xu = placeholder((None, args.x_size), name='xu')
    T.yu = placeholder((None, args.y_size), name='yu')
    T.x = placeholder((None, args.x_size), name='x')
    T.y = placeholder((None, args.y_size), name='y')
    T.iw = placeholder(None, 'int32', name='iw') * 1 # hack for pholder eval
    T.u = placeholder(None, name='u')
    T.phase = placeholder(None, tf.bool, name='phase')

    if args.model == 'conditional':
        conditional(T)
    elif args.model in {'hybrid', 'hybrid_factored'}:
        hybrid(T)
    elif args.model == 'pretrained':
        pretrained(T)

    T.sess = tf.Session()
    T.sess.run(tf.global_variables_initializer())

    # Push all labeled data into unlabeled data set as well if using pretraining
    mnist = Mnist(args.n_label, args.seed, args.task, shift=args.shift,
                  duplicate='pretrain' in args.model, binarize=True)

    # Define remaining optimization hyperparameters
    if args.model == 'conditional':
        iterep = args.n_label / args.bs
        u = 1
    elif args.model in {'hybrid', 'hybrid_factored'}:
        iterep = args.n_total / args.bs
        u = 1 - args.n_label / float(args.n_total)
    elif args.model == 'pretrained':
        pretrain_iterep = args.n_total / args.bs
        iterep = args.n_label / args.bs
        u = 1

    # Sanity checks and creation of logger
    print "Data/Task statistics"
    print "Task:", args.task
    print "Data shapes of (x, y) for Labeled/Train/Valid/Test sets"
    print (mnist.x_label.shape, mnist.y_label.shape)
    print (mnist.x_train.shape, mnist.y_train.shape)
    print (mnist.x_valid.shape, mnist.y_valid.shape)
    print (mnist.x_test.shape, mnist.y_test.shape)
    writer = make_writer()

    ###############
    # Train model #
    ###############
    if 'pretrained' in args.model:
        print "Pretrain epochs, iterep", args.n_pretrain_epochs, pretrain_iterep
        train(T.pre_train_step, T.pre_loss, mnist, pretrain_iterep, args.n_pretrain_epochs)

    if 'hybrid' in args.model:
        print "Hybrid weighting on x_train and x_label:", (u, 1 - u)
    print "Epochs, Iterep", args.n_epochs, iterep
    train(T.train_step, T.loss, mnist, iterep, args.n_epochs)

if __name__ == '__main__':
    import warnings
    if '1.1.0' not in tf.__version__:
        warnings.warn("Library only tested in tf=1.1.0")
    main()
