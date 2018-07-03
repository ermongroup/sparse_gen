"""Utils for the handling data
Parts of file based on : https://github.com/carpedm20/DCGAN-tensorflow/blob/master/utils.py
It comes with the following license: https://github.com/carpedm20/DCGAN-tensorflow/blob/master/LICENSE
"""

from __future__ import division
import scipy.misc
import numpy as np
import png
import tensorflow as tf
import glob
from tensorflow.examples.tutorials.mnist import input_data

NUM_TEST_IMAGES = 10000

### Mnist and omniglot utils

def mnist_input(hparams):
    """Create input tensors"""

    mnist = input_data.read_data_sets('./data/mnist', one_hot=True)

    if hparams.input_type == 'full-input' or hparams.input_type == 'full-transfer':
        images = {i: image for (i, image) in enumerate(mnist.test.images[:hparams.num_input_images])}
    elif hparams.input_type == 'validation':
        images = {i: image for (i, image) in enumerate(mnist.validation.images[-hparams.num_input_images:])}
    else:
        raise NotImplementedError

    return images


def omniglot_input(hparams):
    """Create input tensors"""

    omniglot_test = np.load('./data/omniglot/test.npy')
    
    if hparams.input_type == 'full-input' or hparams.input_type == 'transfer-full':
        images = {i: image for (i, image) in enumerate(omniglot_test[:hparams.num_input_images])}
    elif hparams.input_type == 'validation':
        save_img_list = np.load('./data/omniglot/validation.npy')
        images = {i: image for (i, image) in enumerate(save_img_list[:hparams.num_input_images])} 
    else:
        raise NotImplementedError

    return images


def view_mnistomni_image(image, hparams, mask=None):
    """Process and show the image"""
    image = np.squeeze(image)
    if len(image) == hparams.n_input:
        image = image.reshape([28, 28])
        if mask is not None:
            mask = mask.reshape([28, 28])
            image = np.maximum(np.minimum(1.0, image - 1.0*(1-mask)), 0.0)
    utils.plot_image(image, 'Greys')


def save_mnistomni_image(image, path):
    """Save an image as a png file"""
    png_writer = png.Writer(28, 28, greyscale=True)
    with open(path, 'wb') as outfile:
        png_writer.write(outfile, 255*image)

### CelebA utils

def get_celebA_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def inverse_transform(images):
    return (images+1.)/2.

def save_celebA_images(images, size, image_path):
    return scipy.misc.imsave(image_path, merge(inverse_transform(images), size))

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img



def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])

def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def celebA_input(hparams):
    """Create input tensors"""

    image_paths = glob.glob(hparams.input_path_pattern)
    if hparams.input_type == 'full-input':
        image_paths.sort()
        image_paths = image_paths[:hparams.num_input_images]
    elif hparams.input_type == 'validation':
        image_paths.sort()
        image_paths = image_paths[-hparams.num_input_images:]
    else:
        raise NotImplementedError
    image_size = 108
    images = [get_celebA_image(image_path, image_size) for image_path in image_paths]
    images = {i: image.reshape([64*64*3]) for (i, image) in enumerate(images)}

    return images


def view_celebA_image(image, hparams, mask=None):
    """Process and show the image"""
    image = inverse_transform(image)
    if len(image) == hparams.n_input:
        image = image.reshape(hparams.image_shape)
        if mask is not None:
            mask = mask.reshape(hparams.image_shape)
            image = np.maximum(np.minimum(1.0, image - 1.0*image*(1-mask)), -1.0)
    utils.plot_image(image)


def save_celebA_image(image, path):
    """Save an image as a png file"""
    image = inverse_transform(image)
    image = np.maximum(np.minimum(image,1.),0.)
    png_writer = png.Writer(64, 64)
    with open(path, 'wb') as outfile:
        png_writer.write(outfile, 255*image.reshape([64,-1]))
