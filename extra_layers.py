from config import args
import tensorbayes as tb
from tensorbayes.layers import *
from tensorbayes.distributions import log_bernoulli_with_logits, log_normal

x_size = args.x_size
y_size = args.y_size
z_size = args.z_size

if args.nonlin == 'elu':
    activate = tf.nn.elu
elif args.nonlin == 'relu':
    activate = tf.nn.relu

# Extra tensorflow layers
def recode(x, size, scope=None, default_scope='recode', reuse=None, var=True):
    with tf.variable_scope(scope, default_scope, reuse):
        h = dense(x, args.h_size, activation=activate)
        h = dense(h, args.h_size, activation=activate)
        if var:
            z_m = dense(h, size, 'mean')
            z_v = dense(h, size, 'var', activation=tf.nn.softplus) + args.eps
            return (z_m, z_v)
        else:
            # When var=False, we're at top logit layer. Do not use BN here.
            z_m = dense(h, size, 'mean', bn=False)
            return z_m

def infer(likelihood, prior=None, scope=None, reuse=None, sample_only=False):
    with tf.variable_scope(scope, 'infer', reuse):
        if prior is None:
            posterior = likelihood
        else:
            args = likelihood + prior
            posterior = gaussian_update(*args, scope='pwn')
        z = gaussian_sample(*posterior, scope='sample')
    if sample_only:
        return z
    else:
        return (z, posterior)

# Extra chains (layer with no scope arg)
def infer_recode(likelihood, prior, size, sample_only=False, var=True):
    if sample_only:
        z = infer(likelihood, prior, sample_only=False)
        x_prior = recode(z, size, var=var)
        return x_prior
    else:
        z, z_post = infer(likelihood, prior)
        x_prior = recode(z, size, var=var)
        return (z, z_post, x_prior)

def per_sample_loss(zs, xs):
    kl = [log_normal(z, *post) - log_normal(z, *prior) for (z, post, prior) in zs]
    rc = [-log_bernoulli_with_logits(x, logits, args.eps) for (x, logits) in xs]
    return tf.add_n(kl + rc)

def bjde_x(x, reuse=None):
    z_prior = (constant(0), constant(1))

    with tf.name_scope('bjde_x'):
        with tf.variable_scope('enc/x', reuse=reuse):
            z_like = recode(x, z_size)
            z_init = None

        with tf.variable_scope('dec/x', reuse=reuse):
            z, z_post, x_logits = infer_recode(z_like, z_init, x_size, var=False)

        with tf.name_scope('loss') as sc:
            loss = tf.reduce_mean(per_sample_loss([[z, z_post, z_prior]],
                                                  [[x, x_logits]]))
    return loss

def bjde_y(y, reuse=None):
    z2_prior = (constant(0), constant(1))

    with tf.name_scope('bjde_y'):
        with tf.variable_scope('enc/y', reuse=reuse):
            z1_like = recode(y, z_size)
            z2_like = recode(z1_like[0], z_size)
            z2_init = z2_prior if 'factored' in args.model else None

        with tf.variable_scope('dec/y', reuse=reuse):
            z2, z2_post, z1_prior = infer_recode(z2_like, z2_init, z_size)
            z1, z1_post, y_logits = infer_recode(z1_like, z1_prior, y_size, var=False)

        with tf.name_scope('loss') as sc:
            loss = tf.reduce_mean(per_sample_loss([[z1, z1_post, z1_prior],
                                                   [z2, z2_post, z2_prior]],
                                                  [[y, y_logits]]))
    return loss

def bjde_xy(x, y, reuse_x=False, reuse_y=False, reuse_xy=False):
    z2_prior = (constant(0), constant(1))

    with tf.name_scope('bjde_xy'):
        if 'factored' in args.model:
            with tf.variable_scope('enc/y', reuse=reuse_y):
                z1_like = recode(y, z_size)
                z2_like = recode(z1_like[0], z_size)
            with tf.variable_scope('enc/x', reuse=reuse_x):
                z2_init = recode(x, z_size)
        else:
            with tf.variable_scope('enc/y', reuse=reuse_y):
                z1_like = recode(y, z_size)
            with tf.variable_scope('enc/xy', reuse=reuse_xy):
                z2_like = recode(tf.concat([x, y], 1), z_size)
                z2_init = None

        with tf.variable_scope('dec/y', reuse=reuse_y):
            z2, z2_post, z1_prior = infer_recode(z2_like, z2_init, z_size)
            z1, z1_post, y_logits = infer_recode(z1_like, z1_prior, y_size, var=False)
        with tf.variable_scope('dec/x', reuse=reuse_x):
            x_logits = recode(z2, x_size, var=False)

        with tf.name_scope('loss') as sc:
            loss = tf.reduce_mean(per_sample_loss([[z1, z1_post, z1_prior],
                                                   [z2, z2_post, z2_prior]],
                                                  [[x, x_logits],
                                                   [y, y_logits]]))
    return loss

def bcde(x, y, iw, reuse_x=False, reuse_y=False, reuse_xy=False):
    with tf.name_scope('bcde'):
        with tf.variable_scope('enc/x', reuse=reuse_x):
            z2_prior = recode(x, z_size)
        if 'factored' in args.model:
            with tf.variable_scope('enc/y', reuse=reuse_y):
                z1_like = recode(y, z_size)
                z2_like = recode(z1_like[0], z_size)
        else:
            with tf.variable_scope('enc/y', reuse=reuse_y):
                z1_like = recode(y, z_size)
            with tf.variable_scope('enc/xy', reuse=reuse_xy):
                z2_like = recode(tf.concat([x, y], 1), z_size)

        with tf.name_scope('iw_duplicate'):
            y = duplicate(y, iw)
            z2_prior = tuple([duplicate(v, iw) for v in z2_prior])
            z1_like = tuple([duplicate(v, iw) for v in z1_like])
            z2_like = tuple([duplicate(v, iw) for v in z2_like])
            z2_init = z2_prior if 'factored' in args.model else None

        with tf.variable_scope('dec/y', reuse=reuse_y):
            z2, z2_post, z1_prior = infer_recode(z2_like, z2_init, z_size)
            z1, z1_post, y_logits = infer_recode(z1_like, z1_prior, y_size, var=False)

        with tf.name_scope('loss') as sc:
            ps_loss = per_sample_loss([[z1, z1_post, z1_prior],
                                       [z2, z2_post, z2_prior]],
                                      [[y, y_logits]])
            ps_loss = tf.reshape(ps_loss, [iw, -1])
            # Employ IS: [log of average of (p_i / q_i)] for i = 1,...,iw
            ps_gain = tb.tbutils.log_sum_exp(-ps_loss, axis=0) - tf.log(tf.cast(iw, 'float32'))
            loss = tf.negative(tf.reduce_mean(ps_gain))
    return loss
