# Get input

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

def mnist_data_iteratior():
	mnist = input_data.read_data_sets('./data/mnist', one_hot=True)
	def iterator(hparams, num_batches):
		for _ in range(num_batches):
			yield mnist.train.next_batch(hparams.batch_size)
	return iterator

def omniglot_data_iterator():
	omniglot_train = np.load('../data/omniglot/train.npy')
	def iterator(hparams, num_batches):
		for batch_idx in range(num_batches):
			yield omniglot_train[batch_idx*hparams.batch_size: (batch_idx+1)*hparams.batch_size], None
	return iterator
