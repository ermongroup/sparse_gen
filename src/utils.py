"""Some common utils"""
# pylint: disable = C0301, C0103, C0111

import os
import pickle
import shutil
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('agg')
# matplotlib.rc('font', family='Times New Roman')
# matplotlib.font_manager._rebuild()
# matplotlib.rcParams['text.usetex'] = False
# matplotlib.rcParams['text.latex.unicode']=True
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import mnistomni_estimators
import celebA_estimators

from sklearn.linear_model import Lasso
from l1regls import l1regls
from cvxopt import matrix
from cvxopt import solvers



class BestKeeper(object):
    """Class to keep the best stuff"""
    def __init__(self, hparams):
        self.batch_size = hparams.batch_size
        self.losses_val_best = [1e10 for _ in range(hparams.batch_size)]
        self.x_hat_batch_val_best = np.zeros((hparams.batch_size, hparams.n_input))

    def report(self, x_hat_batch_val, losses_val):
        for i in range(self.batch_size):
            if losses_val[i] < self.losses_val_best[i]:
                self.x_hat_batch_val_best[i, :] = x_hat_batch_val[i, :]
                self.losses_val_best[i] = losses_val[i]

    def get_best(self):
        return self.x_hat_batch_val_best


def get_l2_loss(image1, image2):
    """Get L2 loss between the two images"""
    assert image1.shape == image2.shape
    return np.mean((image1 - image2)**2)


def get_measurement_loss(x_hat, A, y):
    """Get measurement loss of the estimated image"""
    if A is None:
        y_hat = x_hat
    else:
        y_hat = np.matmul(x_hat, A)
    assert y_hat.shape == y.shape
    return np.mean((y - y_hat) ** 2)


def save_to_pickle(data, pkl_filepath):
    """Save the data to a pickle file"""
    with open(pkl_filepath, 'wb') as pkl_file:
        pickle.dump(data, pkl_file)


def load_if_pickled(pkl_filepath):
    """Load if the pickle file exists. Else return empty dict"""
    if os.path.isfile(pkl_filepath):
        with open(pkl_filepath, 'rb') as pkl_file:
            data = pickle.load(pkl_file)
    else:
        data = {}
    return data


def get_estimator(hparams, model_type):
    if hparams.dataset == 'omniglot' or hparams.dataset == 'mnist':
        if model_type == 'vae':
            estimator = mnistomni_estimators.vae_estimator(hparams)
        elif model_type == 'lasso':
            estimator = mnistomni_estimators.lasso_estimator(hparams)
        elif model_type == 'vae_l1':
            estimator = mnistomni_estimators.vae_l1_estimator(hparams)
        else:
            raise NotImplementedError
    elif hparams.dataset == 'celebA':
        if model_type == 'lasso-dct':
            estimator = celebA_estimators.lasso_dct_estimator(hparams)
        elif model_type == 'lasso-wavelet':
            estimator = celebA_estimators.lasso_wavelet_estimator(hparams)
        elif model_type == 'dcgan':
            estimator = celebA_estimators.dcgan_estimator(hparams)
        elif model_type == 'dcgan_l1' or model_type == 'dcgan_l1_wavelet' or model_type == 'dcgan_l1_dct':
            estimator = celebA_estimators.dcgan_l1_estimator(hparams, model_type)
        else:
            raise NotImplementedError
    return estimator


def get_estimators(hparams):
    estimators = {model_type: get_estimator(hparams, model_type) for model_type in hparams.model_types}
    return estimators


def setup_checkpointing(hparams):
    # Set up checkpoint directories
    for model_type in hparams.model_types:
        checkpoint_dir = get_checkpoint_dir(hparams, model_type)
        set_up_dir(checkpoint_dir)


def save_images(est_images, save_image, hparams):
    """Save a batch of images to png files"""
    image_array = []
    no_array = []
    for model_type in hparams.model_types:
        # print(model_type)
        for image_num, image in est_images[model_type].iteritems():
            
            image_array.append(np.reshape(np.array(image), (-1)))
            no_array.append(image_num)

            save_path = get_save_paths(hparams, image_num)[model_type]
            image = image.reshape(hparams.image_shape)            
            # save_image(image, save_path)
    return (image_array, no_array)

def checkpoint(est_images, measurement_losses, l2_losses, l1_losses, linf_losses, save_image, hparams):
    """Save images, measurement losses and L2 losses for a batch"""
    ret = ([],[])
    if hparams.save_images:
        ret = save_images(est_images, save_image, hparams)

    if hparams.save_stats:
        for model_type in hparams.model_types:
            m_losses_filepath, l2_losses_filepath, l1_losses_filepath, linf_losses_filepath  = get_pkl_filepaths(hparams, model_type)
            save_to_pickle(measurement_losses[model_type], m_losses_filepath)
            save_to_pickle(l2_losses[model_type], l2_losses_filepath)
            save_to_pickle(l1_losses[model_type], l1_losses_filepath)
            save_to_pickle(linf_losses[model_type], linf_losses_filepath)
    
    return ret


def load_checkpoints(hparams):
    measurement_losses, l2_losses, l1_losses, linf_losses = {}, {}, {}, {}
    if hparams.save_images:
        # Load pickled loss dictionaries
        for model_type in hparams.model_types:
            m_losses_filepath, l2_losses_filepath, l1_losses_filepath, linf_losses_filepath = get_pkl_filepaths(hparams, model_type)
            measurement_losses[model_type] = load_if_pickled(m_losses_filepath)
            l2_losses[model_type] = load_if_pickled(l2_losses_filepath)
            l1_losses[model_type] = load_if_pickled(l1_losses_filepath)
            linf_losses[model_type] = load_if_pickled(linf_losses_filepath)

    else:
        for model_type in hparams.model_types:
            measurement_losses[model_type] = {}
            l2_losses[model_type] = {}
            l1_losses[model_type] = {}
            linf_losses[model_type] = {}
    return measurement_losses, l2_losses, l1_losses, linf_losses


def image_matrix(images, est_images, view_image, hparams, alg_labels=True):
    """Display images"""

    
    figure_height = 1 + len(hparams.model_types)

    fig = plt.figure(figsize=[2*len(images), 2*figure_height])

    outer_counter = 0
    inner_counter = 0

    # Show original images
    outer_counter += 1
    for image in images.values():
        inner_counter += 1
        ax = fig.add_subplot(figure_height, 1, outer_counter, frameon=False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_ticks([])
        if alg_labels:
            ax.set_ylabel('Original', fontsize=14)
        _ = fig.add_subplot(figure_height, len(images), inner_counter)
        view_image(image, hparams)

    for model_type in hparams.model_types:
        outer_counter += 1
        for image in est_images[model_type].values():
            inner_counter += 1
            ax = fig.add_subplot(figure_height, 1, outer_counter, frameon=False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_ticks([])
            if alg_labels:
                ax.set_ylabel(model_type, fontsize=14)
            _ = fig.add_subplot(figure_height, len(images), inner_counter)
            view_image(image, hparams)

    if hparams.image_matrix >= 2:
        save_path = get_matrix_save_path(hparams)
        plt.savefig(save_path)

    if hparams.image_matrix in [1, 3]:
        plt.show()


def plot_image(image, cmap=None):
    """Show the image"""
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    frame = frame.imshow(image, cmap=cmap)


def get_checkpoint_dir(hparams, model_type):
    base_dir = './estimated/{0}/{1}/{2}/{3}/{4}/{5}/'.format(
        hparams.dataset,
        hparams.input_type,
        hparams.measurement_type,
        hparams.noise_std,
        hparams.num_measurements,
        model_type
    )

    if model_type in ['lasso', 'lasso-dct', 'lasso-wavelet']:
        dir_name = '{}'.format(
            hparams.lmbd,
        )
    elif model_type in ['vae']:
        dir_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
            hparams.mloss1_weight,
            hparams.mloss2_weight,
            hparams.zprior_weight,
            hparams.optimizer_type,
            hparams.learning_rate,
            hparams.momentum,
            hparams.decay_lr,
            hparams.max_update_iter,
            hparams.num_random_restarts,
        )
    elif model_type in ['vae_l1']:
        dir_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
            hparams.mloss1_weight,
            hparams.mloss2_weight,
            hparams.zprior_weight,
            hparams.sparse_gen_weight,
            hparams.optimizer_type,
            hparams.learning_rate,
            hparams.momentum,
            hparams.decay_lr,
            hparams.max_update_iter,
            hparams.num_random_restarts,
        )
    elif model_type in ['dcgan']:
        dir_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
            hparams.mloss1_weight,
            hparams.mloss2_weight,
            hparams.zprior_weight,
            hparams.dloss1_weight,
            hparams.dloss2_weight,
            hparams.optimizer_type,
            hparams.learning_rate,
            hparams.momentum,
            hparams.decay_lr,
            hparams.max_update_iter,
            hparams.num_random_restarts,
        )
    elif model_type in ['dcgan_l1', 'dcgan_l1_wavelet', 'dcgan_l1_dct']:
        dir_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
            hparams.mloss1_weight,
            hparams.mloss2_weight,
            hparams.zprior_weight,
            hparams.sparse_gen_weight,
            hparams.dloss1_weight,
            hparams.dloss2_weight,
            hparams.optimizer_type,
            hparams.learning_rate,
            hparams.momentum,
            hparams.decay_lr,
            hparams.max_update_iter,
            hparams.num_random_restarts,
        )
    else:
        raise NotImplementedError

    ckpt_dir = base_dir + dir_name + '/'

    return ckpt_dir


def get_pkl_filepaths(hparams, model_type):
    """Return paths for the pickle files"""
    checkpoint_dir = get_checkpoint_dir(hparams, model_type)
    m_losses_filepath = checkpoint_dir + 'measurement_losses.pkl'
    l2_losses_filepath = checkpoint_dir + 'l2_losses.pkl'
    l1_losses_filepath = checkpoint_dir + 'l1_losses.pkl'
    linf_losses_filepath = checkpoint_dir + 'linf_losses.pkl'

    return m_losses_filepath, l2_losses_filepath, l1_losses_filepath, linf_losses_filepath


def get_save_paths(hparams, image_num):
    save_paths = {}
    for model_type in hparams.model_types:
        checkpoint_dir = get_checkpoint_dir(hparams, model_type)
        save_paths[model_type] = checkpoint_dir + '{0}.png'.format(image_num)
    return save_paths


def get_matrix_save_path(hparams):
    save_path = './estimated/{0}/{1}/{2}/{3}/{4}/matrix_{5}.png'.format(
        hparams.dataset,
        hparams.input_type,
        hparams.measurement_type,
        hparams.noise_std,
        hparams.num_measurements,
        '_'.join(hparams.model_types)
    )
    return save_path


def set_up_dir(directory, clean=False):
    if os.path.exists(directory):
        if clean:
            shutil.rmtree(directory)
    else:
        os.makedirs(directory)


def print_hparams(hparams):
    print ''
    for temp in dir(hparams):
        if temp[:1] != '_':
            print '{0} = {1}'.format(temp, getattr(hparams, temp))
    print ''


def get_learning_rate(global_step, hparams):
    if hparams.decay_lr:
        return tf.train.exponential_decay(hparams.learning_rate,
                                          global_step,
                                          50,
                                          0.7,
                                          staircase=True)
    else:
        return tf.constant(hparams.learning_rate)


def get_optimizer(learning_rate, hparams):
    if hparams.optimizer_type == 'sgd':
        return tf.train.GradientDescentOptimizer(learning_rate)
    if hparams.optimizer_type == 'momentum':
        return tf.train.MomentumOptimizer(learning_rate, hparams.momentum)
    elif hparams.optimizer_type == 'rmsprop':
        return tf.train.RMSPropOptimizer(learning_rate)
    elif hparams.optimizer_type == 'adam':
        return tf.train.AdamOptimizer(learning_rate)
    else:
        raise Exception('Optimizer ' + hparams.optimizer_type + ' not supported')

def get_A(hparams):
    np.random.seed(0)
    if hparams.measurement_type == 'gaussian':
        A = np.random.randn(hparams.n_input, hparams.num_measurements)
    elif hparams.measurement_type == 'project':
        A = None
    else:
        raise NotImplementedError
    return A

def set_num_measurements(hparams):
    if hparams.measurement_type == 'project':
        hparams.num_measurements = hparams.n_input
    else:
        hparams.num_measurements = get_A(hparams).shape[1]


def get_checkpoint_path(ckpt_dir):
    ckpt_dir = os.path.abspath(ckpt_dir)
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_path = os.path.join(ckpt_dir,
                                 ckpt.model_checkpoint_path)
    else:
        print 'No checkpoint file found'
        ckpt_path = ''
    return ckpt_path

def save_plot(is_save, save_path):
    if is_save:
        pdf = PdfPages(save_path)
        pdf.savefig(bbox_inches='tight')
        pdf.close()


def solve_lasso(A_val, y_val, hparams):
    if hparams.lasso_solver == 'sklearn':
        lasso_est = Lasso(alpha=hparams.lmbd)
        lasso_est.fit(A_val.T, y_val.reshape(hparams.num_measurements))
        x_hat = lasso_est.coef_
        x_hat = np.reshape(x_hat, [-1])
    if hparams.lasso_solver == 'cvxopt':
        A_mat = matrix(A_val.T)
        y_mat = matrix(y_val)
        x_hat_mat = l1regls(A_mat, y_mat)
        x_hat = np.asarray(x_hat_mat)
        x_hat = np.reshape(x_hat, [-1])
    return x_hat

def get_opt_reinit_op(opt, var_list, global_step):
    opt_slots = [opt.get_slot(var, name) for name in opt.get_slot_names() for var in var_list]
    if isinstance(opt, tf.train.AdamOptimizer):
        opt_slots.extend(opt._get_beta_accumulators())  #pylint: disable = W0212
    all_opt_variables = opt_slots + var_list + [global_step] + opt.variables()
    opt_reinit_op = tf.variables_initializer(all_opt_variables)
    return opt_reinit_op
