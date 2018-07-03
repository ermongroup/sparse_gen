import numpy as np
from scipy.io import loadmat

np.random.seed(0)
matfile = './data/omniglot/chardata.mat'
matdict = loadmat(matfile)
trainvaliddata = matdict['data'].T
testdata = matdict['testdata'].T

valididx = np.random.choice(trainvaliddata.shape[0], size=1200, replace=False)
trainidx = np.array(list(set(np.arange(trainvaliddata.shape[0]))))

traindata = trainvaliddata[trainidx]
validdata = trainvaliddata[valididx]
print(traindata.shape)
print(validdata.shape)
print(testdata.shape)


def binarize_data(data):

	binary_data = np.random.binomial(1, p=data)
	# print(data[0])
	print(binary_data.dtype)
	# print(np.sum(data), np.sum(binary_data))

	return binary_data.astype(np.float32)


binary_traindata = binarize_data(traindata)
binary_validdata = binarize_data(validdata)
binary_testdata = binarize_data(testdata)

savedir = './data/omniglot/'
import os
np.save(os.path.join(savedir, 'train.npy'), binary_traindata)
np.save(os.path.join(savedir, 'validation.npy'), binary_validdata)
np.save(os.path.join(savedir, 'test.npy'), binary_testdata)

print(binary_traindata.shape,  binary_testdata.shape)
print(binary_traindata.dtype,  binary_testdata.dtype)
exit()