"""Estimators for compressed sensing"""
# pylint: disable = C0301, C0103, C0111, R0914



import os
import sys
import copy
import tensorflow as tf
import numpy as np
import utils
import scipy.fftpack as fftpack

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from celebA_dcgan import model_def as celebA_dcgan_model_def


def dcgan_discrim(x_hat_batch, hparams):

    assert hparams.batch_size in [1, 64], 'batch size should be either 64 or 1'
    x_hat_image = tf.reshape(x_hat_batch, [-1, 64, 64, 3])
    all_zeros = tf.zeros([64, 64, 64, 3])
    discrim_input = all_zeros + x_hat_image

    model_hparams = celebA_dcgan_model_def.Hparams()
    prob, _ = celebA_dcgan_model_def.discriminator(model_hparams, discrim_input, train=False, reuse=False)
    prob = tf.reshape(prob, [-1])
    prob = prob[:hparams.batch_size]

    restore_vars = celebA_dcgan_model_def.gen_restore_vars()
    restore_dict = {var.op.name: var for var in tf.global_variables() if var.op.name in restore_vars}
    restore_path = tf.train.latest_checkpoint(hparams.pretrained_model_dir)

    return prob, restore_dict, restore_path



def dcgan_gen(z, hparams):

    assert hparams.batch_size in [1, 64], 'batch size should be either 64 or 1'
    z_full = tf.zeros([64, 100]) + z

    model_hparams = celebA_dcgan_model_def.Hparams()

    x_hat_full = celebA_dcgan_model_def.generator(model_hparams, z_full, train=False, reuse=False)
    x_hat_batch = tf.reshape(x_hat_full[:hparams.batch_size], [hparams.batch_size, 64*64*3])

    restore_vars = celebA_dcgan_model_def.gen_restore_vars()
    restore_dict = {var.op.name: var for var in tf.global_variables() if var.op.name in restore_vars}
    restore_path = tf.train.latest_checkpoint(hparams.pretrained_model_dir)

    return x_hat_batch, restore_dict, restore_path


def dct2(image_channel):
    return fftpack.dct(fftpack.dct(image_channel.T, norm='ortho').T, norm='ortho')


def idct2(image_channel):
    return fftpack.idct(fftpack.idct(image_channel.T, norm='ortho').T, norm='ortho')


def vec(channels):
    image = np.zeros((64, 64, 3))
    for i, channel in enumerate(channels):
        image[:, :, i] = channel
    return image.reshape([-1])


def devec(vector):
    image = np.reshape(vector, [64, 64, 3])
    channels = [image[:, :, i] for i in range(3)]
    return channels


def wavelet_basis(path='./wavelet_basis.npy'):
    W_ = np.load(path)
    # W_ initially has shape (4096,64,64), i.e. 4096 64x64 images
    # reshape this into 4096x4096, where each row is an image
    # take transpose to make columns images
    W_ = W_.reshape((4096, 4096))
    W = np.zeros((12288, 12288))
    W[0::3, 0::3] = W_
    W[1::3, 1::3] = W_
    W[2::3, 2::3] = W_
    return W


def lasso_dct_estimator(hparams):  #pylint: disable = W0613
    """LASSO with DCT"""
    def estimator(A_val, y_batch_val, hparams):
        # One can prove that taking 2D DCT of each row of A,
        # then solving usual LASSO, and finally taking 2D ICT gives the correct answer.
        A_new = copy.deepcopy(A_val)
        for i in range(A_val.shape[1]):
            A_new[:, i] = vec([dct2(channel) for channel in devec(A_new[:, i])])

        x_hat_batch = []
        for j in range(hparams.batch_size):
            y_val = y_batch_val[j]
            z_hat = utils.solve_lasso(A_new, y_val, hparams)
            x_hat = vec([idct2(channel) for channel in devec(z_hat)]).T
            x_hat = np.maximum(np.minimum(x_hat, 1), -1)
            x_hat_batch.append(x_hat)
        return x_hat_batch
    return estimator


def lasso_wavelet_estimator(hparams):  #pylint: disable = W0613
    """LASSO with Wavelet"""
    def estimator(A_val, y_batch_val, hparams):
        x_hat_batch = []
        W = wavelet_basis()
        WA = np.dot(W, A_val)
        for j in range(hparams.batch_size):
            y_val = y_batch_val[j]
            z_hat = utils.solve_lasso(WA, y_val, hparams)
            x_hat = np.dot(z_hat, W)
            x_hat_max = np.abs(x_hat).max()
            x_hat = x_hat / (1.0 * x_hat_max)
            x_hat_batch.append(x_hat)
        x_hat_batch = np.asarray(x_hat_batch)
        return x_hat_batch
    return estimator

def dcgan_estimator(hparams):
    # pylint: disable = C0326

    # Get a session
    tf.reset_default_graph()
    g1 = tf.Graph()
    with g1.as_default() as g:
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session( config=tf.ConfigProto(gpu_options=gpu_options\
          , allow_soft_placement=True))

        # Set up palceholders
        A = tf.placeholder(tf.float32, shape=(hparams.n_input, hparams.num_measurements), name='A')
        y_batch = tf.placeholder(tf.float32, shape=(hparams.batch_size, hparams.num_measurements), name='y_batch')

        # Create the generator
        z_batch = tf.Variable(tf.random_normal([hparams.batch_size, 100]), name='z_batch')
        x_hat_batch, restore_dict_gen, restore_path_gen = dcgan_gen(z_batch, hparams)

        # Create the discriminator
        prob, restore_dict_discrim, restore_path_discrim = dcgan_discrim(x_hat_batch, hparams)

        # measure the estimate
        if hparams.measurement_type == 'project':
            y_hat_batch = tf.identity(x_hat_batch, name='y2_batch')
        else:
            y_hat_batch = tf.matmul(x_hat_batch, A, name='y2_batch')

        # define all losses
        m_loss1_batch =  tf.reduce_sum(tf.abs(y_batch - y_hat_batch), 1)
        m_loss2_batch =  tf.reduce_sum((y_batch - y_hat_batch)**2, 1)
        zp_loss_batch =  tf.reduce_sum(z_batch**2, 1)
        d_loss1_batch = -tf.log(prob)
        d_loss2_batch =  tf.log(1-prob)

        # define total loss
        total_loss_batch = hparams.mloss1_weight * m_loss1_batch \
                         + hparams.mloss2_weight * m_loss2_batch \
                         + hparams.zprior_weight * zp_loss_batch \
                         + hparams.dloss1_weight * d_loss1_batch \
                         + hparams.dloss2_weight * d_loss2_batch
        total_loss = tf.reduce_mean(total_loss_batch)

        # Compute means for logging
        m_loss1 = tf.reduce_mean(m_loss1_batch)
        m_loss2 = tf.reduce_mean(m_loss2_batch)
        zp_loss = tf.reduce_mean(zp_loss_batch)
        d_loss1 = tf.reduce_mean(d_loss1_batch)
        d_loss2 = tf.reduce_mean(d_loss2_batch)

        # Set up gradient descent
        var_list = [z_batch]
        global_step = tf.Variable(0, trainable=False, name='global_step')
        learning_rate = utils.get_learning_rate(global_step, hparams)
        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            opt = utils.get_optimizer(learning_rate, hparams)
            update_op = opt.minimize(total_loss, var_list=var_list, global_step=global_step, name='update_op')
        opt_reinit_op = utils.get_opt_reinit_op(opt, var_list, global_step)

        # Intialize and restore model parameters
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        restorer_gen = tf.train.Saver(var_list=restore_dict_gen)
        restorer_discrim = tf.train.Saver(var_list=restore_dict_discrim)
        restorer_gen.restore(sess, restore_path_gen)
        restorer_discrim.restore(sess, restore_path_discrim)

        def estimator(A_val, y_batch_val, hparams):
            """Function that returns the estimated image"""
            best_keeper = utils.BestKeeper(hparams)

            if hparams.measurement_type == 'project':
                feed_dict = {y_batch: y_batch_val}
            else:
                feed_dict = {A: A_val, y_batch: y_batch_val}

            for i in range(hparams.num_random_restarts):
                sess.run(opt_reinit_op)
                for j in range(hparams.max_update_iter):
                
                    _, lr_val, total_loss_val, \
                    m_loss1_val, \
                    m_loss2_val, \
                    zp_loss_val, \
                    d_loss1_val, \
                    d_loss2_val = sess.run([update_op, learning_rate, total_loss,
                                            m_loss1,
                                            m_loss2,
                                            zp_loss,
                                            d_loss1,
                                            d_loss2], feed_dict=feed_dict)


                x_hat_batch_val, total_loss_batch_val = sess.run([x_hat_batch, total_loss_batch], feed_dict=feed_dict)
                best_keeper.report(x_hat_batch_val, total_loss_batch_val)
            return best_keeper.get_best()

        return estimator

def dcgan_l1_estimator(hparams, model_type):
    # pylint: disable = C0326

    tf.reset_default_graph()
    g1 = tf.Graph()
    with g1.as_default() as g:
        # Set up palceholders
        A = tf.placeholder(tf.float32, shape=(hparams.n_input, hparams.num_measurements), name='A')
        y_batch = tf.placeholder(tf.float32, shape=(hparams.batch_size, hparams.num_measurements), name='y_batch')

        # Create the generator
        z_batch = tf.Variable(tf.random_normal([hparams.batch_size, 100]), name='z_batch')
        x_hat_batch, restore_dict_gen, restore_path_gen = dcgan_gen(z_batch, hparams)

        # Create the discriminator
        prob, restore_dict_discrim, restore_path_discrim = dcgan_discrim(x_hat_batch, hparams)
        nu_estim = tf.get_variable("x_estim", dtype=tf.float32, shape=x_hat_batch.get_shape() ,initializer=tf.constant_initializer(0))
        x_estim = nu_estim + x_hat_batch
        
        # measure the estimate
        if hparams.measurement_type == 'project':
            y_hat_batch = tf.identity(x_estim, name='y2_batch')
        else:
            y_hat_batch = tf.matmul(x_estim, A, name='y2_batch')

        # define all losses
        m_loss1_batch =  tf.reduce_mean(tf.abs(y_batch - y_hat_batch), 1)
        m_loss2_batch =  tf.reduce_mean((y_batch - y_hat_batch)**2, 1)
        zp_loss_batch =  tf.reduce_sum(z_batch**2, 1)
        d_loss1_batch = -tf.log(prob)
        d_loss2_batch =  tf.log(1-prob)
        
        if model_type == 'dcgan_l1':
            l1_loss = tf.reduce_sum(tf.abs(nu_estim),1)
        elif model_type == 'dcgan_l1_wavelet':
            W = wavelet_basis()
            Winv = np.linalg.inv(W)   
            l1_loss = tf.reduce_sum(tf.abs(tf.matmul(nu_estim, tf.constant(Winv ,dtype=tf.float32))),1)
        elif model_type == 'dcgan_l1_dct':
            dct_proj = np.reshape(np.array([dct2(np.eye(64)) for itr in range(hparams.batch_size*3)]), [hparams.batch_size, 3, 64, 64])
            nu_re = tf.transpose(tf.reshape(nu_estim, (-1,64,64,3)),[0,3,1,2])
            l1_loss = tf.reduce_sum(tf.abs(tf.matmul(nu_re, tf.constant(dct_proj ,dtype=tf.float32))),[1,2,3])

        # define total loss
        total_loss_batch = hparams.mloss1_weight * m_loss1_batch \
                         + hparams.mloss2_weight * m_loss2_batch \
                         + hparams.zprior_weight * zp_loss_batch \
                         + hparams.dloss1_weight * d_loss1_batch \
                         + hparams.dloss2_weight * d_loss2_batch \
                         + hparams.sparse_gen_weight * l1_loss
        total_loss = tf.reduce_mean(total_loss_batch)

        # Compute means for logging
        m_loss1 = tf.reduce_mean(m_loss1_batch)
        m_loss2 = tf.reduce_mean(m_loss2_batch)
        zp_loss = tf.reduce_mean(zp_loss_batch)
        d_loss1 = tf.reduce_mean(d_loss1_batch)
        d_loss2 = tf.reduce_mean(d_loss2_batch)


        # Set up gradient descent z_batch, 
        var_list = [nu_estim, z_batch] 
        global_step = tf.Variable(0, trainable=False, name='global_step')
        learning_rate = utils.get_learning_rate(global_step, hparams)
        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            opt = utils.get_optimizer(learning_rate, hparams)
            update_op = opt.minimize(total_loss, var_list=var_list, global_step=global_step, name='update_op')
            update_init_op = opt.minimize(total_loss, var_list=[z_batch], name='update_init_op')
            nu_estim_clip = nu_estim.assign(tf.maximum(tf.minimum(1.0-x_hat_batch, nu_estim), -1.0-x_hat_batch))

        opt_reinit_op = utils.get_opt_reinit_op(opt, var_list, global_step)

        # Intialize and restore model parameters
        init_op = tf.global_variables_initializer()
        
    # Get a session
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(graph=g1, config=tf.ConfigProto(gpu_options=gpu_options\
      , allow_soft_placement=True))
    
    sess.run(init_op)
    restorer_gen = tf.train.Saver(var_list=restore_dict_gen)
    restorer_discrim = tf.train.Saver(var_list=restore_dict_discrim)
    restorer_gen.restore(sess, restore_path_gen)
    restorer_discrim.restore(sess, restore_path_discrim)

    def estimator(A_val, y_batch_val, hparams):
        """Function that returns the estimated image"""
        best_keeper = utils.BestKeeper(hparams)

        if hparams.measurement_type == 'project':
            feed_dict = {y_batch: y_batch_val}
        else:
            feed_dict = {A: A_val, y_batch: y_batch_val}

        for i in range(hparams.num_random_restarts):
            sess.run(opt_reinit_op)
            if hparams.max_update_iter > 250:
                init_itr_no = 250
            else:
                init_itr_no = 0

            for j in range(init_itr_no):
                sess.run([update_init_op],feed_dict=feed_dict)
                x_estim_val, total_loss_batch_val = sess.run([x_estim, total_loss_batch], feed_dict=feed_dict)
                best_keeper.report(x_estim_val, total_loss_batch_val)

            for j in range(int(hparams.max_update_iter - init_itr_no)):
                _, lr_val, total_loss_val, \
                m_loss1_val, \
                m_loss2_val, \
                zp_loss_val, \
                d_loss1_val, \
                d_loss2_val = sess.run([update_op, learning_rate, total_loss,
                                        m_loss1,
                                        m_loss2,
                                        zp_loss,
                                        d_loss1,
                                        d_loss2], feed_dict=feed_dict)
                sess.run(nu_estim_clip)

            x_estim_val, total_loss_batch_val = sess.run([x_estim, total_loss_batch], feed_dict=feed_dict)
            best_keeper.report(x_estim_val, total_loss_batch_val)
        return best_keeper.get_best()

    return estimator