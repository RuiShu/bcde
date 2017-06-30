import tensorbayes as tb
from tensorbayes.nputils import convert_to_ssl
import numpy as np
import pickle as pkl
import os, urllib, gzip

class Mnist(object):
    def __init__(self, n_label, seed, quad_type, binarize=True, duplicate=True, shift=None):
        self._load_mnist()
        self.quad_type = quad_type
        if binarize:
            self.binarize()
        self.convert_to_ssl(n_label, seed, duplicate)
        if shift in {None, 'none'}:
            self.split()
        elif shift == 'sensitive':
            self.shift(n_label)
            self.split()
        elif shift == 'invariant':
            self.split()
            self.shift(n_label)
        else:
            raise Exception('Unrecognized setting for shift: {:s}'.format(shift))

    def next_batch(self, bs):
        xu_idx = np.random.choice(len(self.x_train), bs, replace=False)
        yu_idx = np.random.choice(len(self.y_train), bs, replace=False)
        l_idx = np.random.choice(len(self.x_label), bs, replace=False)
        return self.x_label[l_idx], self.y_label[l_idx], self.x_train[xu_idx], self.y_train[yu_idx]

    @staticmethod
    def _download_mnist():
        folder = os.path.join('data', 'mnist_real')
        data_loc = os.path.join(folder, 'mnist.pkl.gz')
        if not os.path.exists(folder):
            os.makedirs(folder)
        if not os.path.exists(data_loc):
            url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
            print "Downloading data from:", url
            urllib.urlretrieve(url, data_loc)
        return data_loc

    def _load_mnist(self):
        f = gzip.open(self._download_mnist(), 'rb')
        train, valid, test = pkl.load(f)
        f.close()
        self.x_train, self.y_train = train[0], train[1]
        self.x_valid, self.y_valid = valid[0], valid[1]
        self.x_test, self.y_test = test[0], test[1]

    def binarize(self, seed=42):
        state = np.random.get_state()
        np.random.seed(seed)
        self.x_train = np.random.binomial(1, self.x_train)
        self.x_valid = np.random.binomial(1, self.x_valid)
        self.x_test  = np.random.binomial(1, self.x_test)
        np.random.set_state(state)

    def convert_to_ssl(self, n_label, seed, duplicate):
        state = np.random.get_state()
        np.random.seed(seed)
        if n_label == 50000:
            # Be very careful: if x_label and x_train are binarized
            # differently, we actually accidentally increase our dataset size
            print "Using full data set. No conversion used"
            self.x_label, self.y_label = np.copy(self.x_train), np.copy(self.y_train)
        else:
            xl, yl, xu, yu = tb.nputils.convert_to_ssl(self.x_train,
                                                       self.y_train,
                                                       n_label,
                                                       n_classes=10,
                                                       complement=not duplicate)
            self.x_label, self.y_label = xl, yl
            self.x_train, self.y_train = xu, yu
        np.random.set_state(state)

    def shift(self, n_label):
        state = np.random.get_state()
        np.random.seed(42)
        def transform(x):
            H = len(x)/28
            y = np.zeros((H, 28))
            s = np.random.randint(0, 5)
            if s == 0:
                return x, 0
            if np.random.choice([0, 1]):
                y[:, :-s] = x.reshape(H, 28)[:, s:]
                return y.reshape(-1), -s
            else:
                y[:, s:] = x.reshape(H, 28)[:, :-s]
                return y.reshape(-1), s
        def batch_transform(x, s):
            for i in xrange(len(x)):
                x[i], s[i] = transform(x[i])
        self.s_train = np.zeros(len(self.y_train))
        self.s_valid = np.zeros(len(self.y_valid))
        self.s_test = np.zeros(len(self.y_test))
        batch_transform(self.x_train, self.s_train)
        batch_transform(self.x_valid, self.s_valid)
        batch_transform(self.x_test, self.s_test)
        if n_label == 50000:
            print "Using full data set. Copying shifted train -> label"
            self.s_label = np.copy(self.s_train)
            self.x_label = np.copy(self.x_train)
        else:
            self.s_label = np.zeros(len(self.y_label))
            batch_transform(self.x_label, self.s_label)
        np.random.set_state(state)

    def split(self):
        self.z_train, self.z_valid, self.z_test, self.z_label = self.y_train, self.y_valid, self.y_test, self.y_label
        self.x_train, self.y_train = self._split(self.x_train)
        self.x_label, self.y_label = self._split(self.x_label)
        self.x_valid, self.y_valid = self._split(self.x_valid)
        self.x_test, self.y_test = self._split(self.x_test)

    def _split(self, data):
        assert len(data.shape) == 2
        assert data.shape[1] == 784
        size = data.shape[0]
        spatial_idxs = np.arange(784).reshape(28, 28)
        if self.quad_type == 'q1':
            x_idx = spatial_idxs[14:, :14].reshape(-1)
            y_idx = np.delete(spatial_idxs.reshape(-1), x_idx)
        elif self.quad_type == 'q2':
            x_idx = spatial_idxs[:, :14].reshape(-1)
            y_idx = np.delete(spatial_idxs.reshape(-1), x_idx)
        elif self.quad_type == 'q3':
            y_idx = spatial_idxs[14:, 14:].reshape(-1)
            x_idx = np.delete(spatial_idxs.reshape(-1), y_idx)
        elif self.quad_type == 'td':
            x_idx = spatial_idxs[:14, :].reshape(-1)
            y_idx = np.delete(spatial_idxs.reshape(-1), x_idx)
        else:
            raise Exception('quadrant not specified')
        data_x = data[:, x_idx]
        data_y = data[:, y_idx]
        return data_x, data_y

    def stitch(self, xs, ys):
        assert len(xs.shape) == 2
        assert len(ys.shape) == 2
        assert len(xs) == len(ys)
        imgs = np.empty((len(xs), 784))
        spatial_idxs = np.arange(784).reshape(28,28)
        if self.quad_type == 'q1':
            x_idx = spatial_idxs[14:, :14].reshape(-1)
            y_idx = np.delete(spatial_idxs.reshape(-1), x_idx)
        elif self.quad_type == 'q2':
            x_idx = spatial_idxs[:, :14].reshape(-1)
            y_idx = np.delete(spatial_idxs.reshape(-1), x_idx)
        elif self.quad_type == 'q3':
            y_idx = spatial_idxs[14:, 14:].reshape(-1)
            x_idx = np.delete(spatial_idxs.reshape(-1), y_idx)
        elif self.quad_type == 'td':
            x_idx = spatial_idxs[:14, :].reshape(-1)
            y_idx = np.delete(spatial_idxs.reshape(-1), x_idx)
        else:
            raise Exception('quadrant not specified')
        imgs[:, x_idx] = xs
        imgs[:, y_idx] = ys
        return imgs
