"""Compressed sensing main script"""
# pylint: disable=C0301,C0103,C0111

from __future__ import division
import os
from argparse import ArgumentParser
import numpy as np
import utils


def main(hparams):

    # Set up some stuff accoring to hparams
    hparams.n_input = np.prod(hparams.image_shape)
    utils.set_num_measurements(hparams)
    utils.print_hparams(hparams)

    # get inputs
    xs_dict = model_input(hparams)
    

    estimators = utils.get_estimators(hparams)
    utils.setup_checkpointing(hparams)
    measurement_losses, l2_losses, l1_losses, linf_losses = utils.load_checkpoints(hparams)

    x_hats_dict = {model_type : {} for model_type in hparams.model_types}
    x_batch_dict = {}

    for key, x in xs_dict.iteritems():
        x_batch_dict[key] = x
        if len(x_batch_dict) < hparams.batch_size:
            continue

        # Reshape input
        x_batch_list = [x.reshape(1, hparams.n_input) for _, x in x_batch_dict.iteritems()]
        x_batch = np.concatenate(x_batch_list)

        # Construct noise and measurements
        A = utils.get_A(hparams)
        noise_batch = hparams.noise_std * np.random.randn(hparams.batch_size, hparams.num_measurements)
        if hparams.measurement_type == 'project':
            y_batch = x_batch + noise_batch
        else:
            y_batch = np.matmul(x_batch, A) + noise_batch

        # Construct estimates using each estimator
        for model_type in hparams.model_types:
            estimator = estimators[model_type]
            x_hat_batch = estimator(A, y_batch, hparams)

            for i, key in enumerate(x_batch_dict.keys()):
                x = xs_dict[key]
                y = y_batch[i]
                x_hat = x_hat_batch[i]

                # Save the estimate
                x_hats_dict[model_type][key] = x_hat

                # Compute and store measurement and l2 loss
                measurement_losses[model_type][key] = utils.get_measurement_loss(x_hat, A, y)
                l2_losses[model_type][key] = utils.get_l2_loss(x_hat, x)
                l1_losses[model_type][key] = np.mean(np.abs(x_hat-x))
                linf_losses[model_type][key] = np.amax(np.abs(x_hat-x))

        print 'Processed upto image {0} / {1}'.format(key+1, len(xs_dict))

        # Checkpointing
        if (hparams.save_images) and ((key+1) % hparams.checkpoint_iter == 0):
            utils.checkpoint(x_hats_dict, measurement_losses, l2_losses, l1_losses, linf_losses, save_image, hparams)
            x_hats_dict = {model_type : {} for model_type in hparams.model_types}
            print '\nProcessed and saved first ', key+1, 'images\n'

        x_batch_dict = {}

    # Final checkpoint
    if hparams.save_images:
        utils.checkpoint(x_hats_dict, measurement_losses, l2_losses, l1_losses, linf_losses, save_image, hparams)
        print '\nProcessed and saved all {0} image(s)\n'.format(len(xs_dict))


    if hparams.print_stats:
        for model_type in hparams.model_types:
            print model_type
            mean_m_loss = np.mean(measurement_losses[model_type].values())
            mean_l2_loss = np.mean(l2_losses[model_type].values())
            mean_l1_loss = np.mean(l1_losses[model_type].values())
            print 'mean measurement loss = {0}'.format(mean_m_loss)
            print 'mean l2 loss = {0}'.format(mean_l2_loss)
            print 'mean l1 loss = {0}'.format(mean_l1_loss)


    if hparams.image_matrix > 0:
        utils.image_matrix(xs_dict, x_hats_dict, view_image, hparams)

    # Warn the user that some things were not processsed
    if len(x_batch_dict) > 0:
        print '\nDid NOT process last {} images because they did not fill up the last batch.'.format(len(x_batch_dict))
        print 'Consider rerunning lazily with a smaller batch size.'


if __name__ == '__main__':
    PARSER = ArgumentParser()

    # Pretrained model
    PARSER.add_argument('--pretrained-model-dir', type=str, default='./models/celebA_64_64/', help='Directory containing pretrained model')

    # Input
    PARSER.add_argument('--dataset', type=str, default='celebA', help='Dataset to use (celebA, omniglot, mnist)')
    PARSER.add_argument('--input-type', type=str, default='full_input', help='Where to take input from')
    PARSER.add_argument('--input-path-pattern', type=str, default='./data/celebA/*.jpg', help='Pattern to match to get images')
    PARSER.add_argument('--num-input-images', type=int, default=10, help='number of input images')
    PARSER.add_argument('--batch-size', type=int, default=64, help='How many examples are processed together')

    # Problem definition
    PARSER.add_argument('--measurement-type', type=str, default='gaussian', help='measurement type')
    PARSER.add_argument('--noise-std', type=float, default=0.1, help='std dev of noise')

    # Measurement type specific hparams
    PARSER.add_argument('--num-measurements', type=int, default=200, help='number of gaussian measurements')

    # Model
    PARSER.add_argument('--model-types', type=str, nargs='+', default=None, help='model(s) used for estimation')
    PARSER.add_argument('--mloss1_weight', type=float, default=0.0, help='L1 measurement loss weight')
    PARSER.add_argument('--mloss2_weight', type=float, default=0.0, help='L2 measurement loss weight')
    PARSER.add_argument('--zprior_weight', type=float, default=0.0, help='weight on z prior')
    PARSER.add_argument('--dloss1_weight', type=float, default=0.0, help='-log(D(G(z))')
    PARSER.add_argument('--dloss2_weight', type=float, default=0.0, help='log(1-D(G(z))')
    PARSER.add_argument('--sparse_gen_weight', type=float, default=0.0, help='weight for sparse deviations')

    # NN specfic hparams
    PARSER.add_argument('--optimizer-type', type=str, default='momentum', help='Optimizer type')
    PARSER.add_argument('--learning-rate', type=float, default=0.01, help='learning rate')
    PARSER.add_argument('--momentum', type=float, default=0.9, help='momentum value')
    PARSER.add_argument('--max-update-iter', type=int, default=100, help='maximum updates to z')
    PARSER.add_argument('--num-random-restarts', type=int, default=10, help='number of random restarts')
    PARSER.add_argument('--decay-lr', action='store_true', help='whether to decay learning rate')

    # LASSO specific hparams
    PARSER.add_argument('--lmbd', type=float, default=0.1, help='lambda : regularization parameter for LASSO')
    PARSER.add_argument('--lasso-solver', type=str, default='sklearn', help='Solver for LASSO')
    PARSER.add_argument('--const_dummy', type=bool, default=False, help='dummy hack')

    # Output
    PARSER.add_argument('--save-images', action='store_true', help='whether to save estimated images')
    PARSER.add_argument('--save-stats', action='store_true', help='whether to save estimated images')
    PARSER.add_argument('--print-stats', action='store_true', help='whether to print statistics')
    PARSER.add_argument('--checkpoint-iter', type=int, default=50, help='checkpoint every x batches')
    PARSER.add_argument('--image-matrix', type=int, default=0,
                        help='''
                                0 = 00 =      no       image matrix,
                                1 = 01 =          show image matrix
                                2 = 10 = save          image matrix
                                3 = 11 = save and show image matrix
                             '''
                       )

    HPARAMS = PARSER.parse_args()

    if HPARAMS.dataset == 'mnist':
        HPARAMS.image_shape = (28, 28, 1)
        from data_utils import mnist_input as model_input
        from data_utils import view_mnistomni_image as view_image
        from data_utils import save_mnistomni_image as save_image
    elif HPARAMS.dataset == 'celebA':
        HPARAMS.image_shape = (64, 64, 3)
        from data_utils import celebA_input as model_input
        from data_utils import view_celebA_image as view_image
        from data_utils import save_celebA_image as save_image
    elif HPARAMS.dataset == 'omniglot':
        HPARAMS.image_shape = (28, 28, 1)
        from data_utils import omniglot_input as model_input
        from data_utils import view_mnistomni_image as view_image
        from data_utils import save_mnistomni_image as save_image
    else:
        raise NotImplementedError

    main(HPARAMS)
