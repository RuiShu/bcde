from config import args
from extra_layers import *
from utils import *
from tensorflow.contrib.framework import arg_scope

def conditional(T):
    with arg_scope([dense], bn=True, phase=T.phase):
        with tf.variable_scope('bcde'):
            print "Building BCDE"
            T.bcde = bcde(T.x, T.y, T.iw)

        T.loss = tf.identity(T.bcde, name='loss')

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    T.train_step = build_optimizer(T.loss, update_ops)

def hybrid(T):
    def build_l2_loss():
        l2 = []
        bcde_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'bcde')
        for bcde_v in bcde_variables:
            bjde_v_name = bcde_v.name.replace('bcde/','bjde/')
            bjde_v_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, bjde_v_name)
            assert len(bjde_v_list) == 1
            print "Pairing {:s}".format(bcde_v.name)
            l2 += [tf.nn.l2_loss(bcde_v - bjde_v_list[0])]

        l2_loss = tf.add_n(l2)
        return l2_loss

    with arg_scope([dense], bn=True, phase=T.phase):
        with tf.variable_scope('bjde') as sc:
            run_marginal = args.n_label < args.n_total
            print "Building BJDE"
            T.bjde_xu = bjde_x(T.xu) if run_marginal else constant(0)
            T.bjde_yu = bjde_y(T.yu) if run_marginal else constant(0)
            T.bjde_x = bjde_x(T.x, reuse=run_marginal)
            T.bjde_xy = bjde_xy(T.x, T.y, reuse_x=True, reuse_y=run_marginal)

        with tf.variable_scope('bcde'):
            print "Building BCDE"
            T.bcde = bcde(T.x, T.y, T.iw)

        with tf.name_scope('l2_loss'):
            print "Building l2_loss"
            T.l2 = build_l2_loss()

        with tf.name_scope('loss'):
            # Eq. 13 from BCDE paper
            T.loss = (args.l2 * T.l2 +
                      T.u * (T.bjde_xu + T.bjde_yu) +
                      0.5 * (1 - T.u) *
                      (T.bjde_xy + T.bjde_x + T.bcde))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    T.train_step = build_optimizer(T.loss, update_ops)

def pretrained(T):
    with arg_scope([dense], bn=True, phase=T.phase):
        with tf.variable_scope('bcde'):
            print "Building BCDE (pretrain component)"
            T.bjde_xu = bjde_x(T.xu)
            T.bjde_yu = bjde_y(T.yu)
            T.pre_loss = tf.add(T.bjde_xu, T.bjde_yu, name='pre_loss')

    pre_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    T.pre_train_step = build_optimizer(T.pre_loss, pre_update_ops)

    with arg_scope([dense], bn=True, phase=T.phase):
        with tf.variable_scope('bcde'):
            print "Building BCDE"
            T.bcde = bcde(T.x, T.y, T.iw, reuse_x=True, reuse_y=True)
        T.loss = tf.identity(T.bcde, name='loss')

    # Remove pre_update_ops
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    update_ops = [op for op in update_ops if op not in pre_update_ops]
    T.train_step = build_optimizer(T.loss, update_ops)
