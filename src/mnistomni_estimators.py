"""Estimators for compressed sensing"""
# pylint: disable = C0301, C0103, C0111, R0914

import os
import sys
import numpy as np
import tensorflow as tf
import utils
import copy

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from vae_models.src import model_def as vae_model_def

def construct_gen(hparams, model_def, name='gen'):

    model_hparams = model_def.Hparams()

    z = model_def.get_z_var(model_hparams, hparams.batch_size)
    x_logits, x_hat = model_def.generator(model_hparams, z, name, reuse=False)

    restore_vars = model_def.gen_restore_vars()
    restore_dict = {var.op.name: var for var in tf.global_variables() if var.op.name in restore_vars}
    restore_path = tf.train.latest_checkpoint(hparams.pretrained_model_dir)

    return z, x_hat, x_logits, restore_path, restore_dict

def lasso_estimator(hparams):  # pylint: disable = W0613
    """LASSO estimator"""
    def estimator(A_val, y_batch_val, hparams):
        x_hat_batch = []
        for i in range(hparams.batch_size):
            y_val = copy.deepcopy(y_batch_val[i])
            x_hat = utils.solve_lasso(A_val, y_val, hparams)
            x_hat = np.maximum(np.minimum(x_hat, 1), 0)
            x_hat_batch.append(x_hat)
        x_hat_batch = np.asarray(x_hat_batch)
        return x_hat_batch
    return estimator


def vae_estimator(hparams):

    # Get a session
    tf.reset_default_graph()
    g1 = tf.Graph()
    with g1.as_default() as g:
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options\
          , allow_soft_placement=True))
        
        # Set up palceholders
        A = tf.placeholder(tf.float32, shape=(hparams.n_input, hparams.num_measurements), name='A')
        y_batch = tf.placeholder(tf.float32, shape=(hparams.batch_size, hparams.num_measurements), name='y_batch')

        # Create the generator
        z_batch, x_hat_batch, _, restore_path, restore_dict = construct_gen(hparams, vae_model_def, 'gen')

        # measure the estimate
        if hparams.measurement_type == 'project':
            y_hat_batch = tf.identity(x_hat_batch, name='y_hat_batch')
        else:
            y_hat_batch = tf.matmul(x_hat_batch, A, name='y_hat_batch')

        # define all losses
        m_loss1_batch = tf.reduce_mean(tf.abs(y_batch - y_hat_batch), 1)
        m_loss2_batch = tf.reduce_mean((y_batch - y_hat_batch)**2, 1)
        zp_loss_batch = tf.reduce_sum(z_batch**2, 1)

        # define total loss
        total_loss_batch = hparams.mloss1_weight * m_loss1_batch \
                         + hparams.mloss2_weight * m_loss2_batch \
                         + hparams.zprior_weight * zp_loss_batch
        total_loss = tf.reduce_mean(total_loss_batch)

        # Compute means for logging
        m_loss1 = tf.reduce_mean(m_loss1_batch)
        m_loss2 = tf.reduce_mean(m_loss2_batch)
        zp_loss = tf.reduce_mean(zp_loss_batch)

        # Set up gradient descent
        var_list = [z_batch]
        global_step = tf.Variable(0, trainable=False, name='global_step')
        learning_rate = utils.get_learning_rate(global_step, hparams)
        opt = utils.get_optimizer(learning_rate, hparams)
        update_op = opt.minimize(total_loss, var_list=var_list, global_step=global_step, name='update_op')
        opt_reinit_op = utils.get_opt_reinit_op(opt, var_list, global_step)

        # Intialize and restore model parameters
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        restorer = tf.train.Saver(var_list=restore_dict)
        restorer.restore(sess, restore_path)

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
                    zp_loss_val = sess.run([update_op, learning_rate, total_loss,
                                            m_loss1,
                                            m_loss2,
                                            zp_loss], feed_dict=feed_dict)

                x_hat_batch_val, total_loss_batch_val = sess.run([x_hat_batch, total_loss_batch], feed_dict=feed_dict)
                best_keeper.report(x_hat_batch_val, total_loss_batch_val)
            return best_keeper.get_best()

        return estimator

def vae_l1_estimator(hparams):

    # Get a session
    tf.reset_default_graph()
    g1 = tf.Graph()
    with g1.as_default() as g:
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(graph=g1, config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

        # Set up palceholders
        A = tf.placeholder(tf.float32, shape=(hparams.n_input, hparams.num_measurements), name='A')
        y_batch = tf.placeholder(tf.float32, shape=(hparams.batch_size, hparams.num_measurements), name='y_batch')

        # Create the generator
        z_batch, x_hat_batch, _, restore_path, restore_dict = construct_gen(hparams, vae_model_def, 'gen')
        nu_estim = tf.get_variable("x_estim", dtype=tf.float32, shape=x_hat_batch.get_shape() ,initializer=tf.constant_initializer(0))
        
        x_estim = nu_estim + x_hat_batch
        # measure the estimate
        if hparams.measurement_type == 'project':
            y_hat_batch = tf.identity(x_estim, name='y_hat_batch')
        else:
            y_hat_batch = tf.matmul(x_estim, A, name='y_hat_batch')

        # define all losses
        m_loss1_batch = tf.reduce_mean(tf.abs(y_batch - y_hat_batch), 1)
        m_loss2_batch = tf.reduce_mean((y_batch - y_hat_batch)**2, 1)
        zp_loss_batch = tf.reduce_sum(z_batch**2, 1)
        l1_loss = tf.reduce_sum(tf.abs(nu_estim),1)
        
        # define total loss
        total_loss_batch = hparams.mloss1_weight * m_loss1_batch \
                         + hparams.mloss2_weight * m_loss2_batch \
                         + hparams.zprior_weight * zp_loss_batch \
                         + hparams.sparse_gen_weight * l1_loss
        total_loss = tf.reduce_mean(total_loss_batch)

        # Compute means for logging
        m_loss1 = tf.reduce_mean(m_loss1_batch)
        m_loss2 = tf.reduce_mean(m_loss2_batch)
        zp_loss = tf.reduce_mean(zp_loss_batch)

        # Set up gradient descent
        var_list = [z_batch, nu_estim]
        global_step = tf.Variable(0, trainable=False, name='global_step')
        learning_rate = utils.get_learning_rate(global_step, hparams)
        opt = utils.get_optimizer(learning_rate, hparams)
        update_op = opt.minimize(total_loss, var_list=var_list, global_step=global_step, name='update_op')
        opt_reinit_op = utils.get_opt_reinit_op(opt, var_list, global_step)
        update_init_op = opt.minimize(total_loss, var_list=[z_batch], name='update_init_op')

        # Intialize and restore model parameters
        nu_estim_clip = nu_estim.assign(tf.maximum(tf.minimum(1-x_hat_batch, nu_estim), -x_hat_batch))
        init_op = tf.global_variables_initializer()
    
    sess.run(init_op)
    restorer = tf.train.Saver(var_list=restore_dict)
    restorer.restore(sess, restore_path)

    def estimator(A_val, y_batch_val, hparams):
        """Function that returns the estimated image"""
        best_keeper = utils.BestKeeper(hparams)
        if hparams.measurement_type == 'project':
            feed_dict = {y_batch: y_batch_val}
        else:
            feed_dict = {A: A_val, y_batch: y_batch_val}

        for i in range(hparams.num_random_restarts):
            sess.run(opt_reinit_op)
            if hparams.max_update_iter > 800:
                init_itr_no = 800
            else:
                init_itr_no = 0

            for j in range(init_itr_no):
                sess.run([update_init_op],feed_dict=feed_dict)
                x_estim_val, total_loss_batch_val = sess.run([x_estim, total_loss_batch], feed_dict=feed_dict)
                best_keeper.report(x_estim_val, total_loss_batch_val)
            
            for j in range(int(hparams.max_update_iter)-init_itr_no):
                _, lr_val, total_loss_val, \
                m_loss1_val, \
                m_loss2_val, \
                zp_loss_val = sess.run([update_op, learning_rate, total_loss,
                                        m_loss1,
                                        m_loss2,
                                        zp_loss], feed_dict=feed_dict)

                sess.run(nu_estim_clip)

            x_estim_val, total_loss_batch_val = sess.run([x_estim, total_loss_batch], feed_dict=feed_dict)
            best_keeper.report(x_estim_val, total_loss_batch_val)

        return (best_keeper.get_best())

    return estimator
