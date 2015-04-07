
# coding: utf-8

# In[1]:
from __future__ import division
import theano.tensor as T
from theano import function
from theano import shared
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams
import os
from sys import argv
import time
from scipy import ndimage,misc
from pandas import DataFrame, Series
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
import numpy as np
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
import cPickle,pickle

# In[2]:

from MainClasses import *

allFiles = []

for root, dirs, files in os.walk(argv[1]):
    allFiles += [os.path.join(root, name) for name in files if "DS_Store" not in name]

# COMMENT ME OUT IF NOT TESTING!
##allFiles = allFiles[:5]


# In[12]:

learning_rate=0.1
n_epochs=200
nkerns=[20, 50]
batch_size=20

height = 28
width = 28
layers = 1

filterHeight = 5
filterWidth = 5
maxPoolSize = 2
l1H = int((height - filterHeight + 1)/maxPoolSize)
l1W = int((width - filterWidth + 1)/maxPoolSize)

l2H = int((l1H - filterHeight + 1)/maxPoolSize)
l2W = int((l1W - filterWidth + 1)/maxPoolSize)

rng = numpy.random.RandomState(23455)
# allocate symbolic variables for the data
index = T.lscalar()  # index to a [mini]batch

x = T.matrix('x')   # the data is presented as rasterized images
y = T.ivector('y')  # the labels are presented as 1D vector of
                    # [int] labels
    
def shared_dataset(data_xy, borrow=True):    
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    return shared_x, T.cast(shared_y, 'int32')


# In[13]:

test = np.arange(5293*784).reshape(5293,784).astype(float)

test_set_x, test_set_y = shared_dataset([test, np.arange(5293)])


# In[24]:

# Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
# to a 4D tensor, compatible with our LeNetConvPoolLayer
# (28, 28) is the size of MNIST images.
layer0_input = x.reshape((batch_size, 1, height, width))

# Construct the first convolutional pooling layer:
# filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
## filtering reduces the image size to (135-12+1 , 240-12+1) = (124,224)
# maxpooling reduces this further to (24/2, 24/2) = (12, 12)
## maxpooling reduces this further to (124/2, 224/2) = (62, 112)
# 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
## 4D output tensor is thus of shape (batch_size, nkerns[0], 62,112)
layer0 = LeNetConvPoolLayer(
    rng,
    input=layer0_input,
    image_shape=(batch_size, 1, height, width),
    filter_shape=(nkerns[0], 1, filterHeight,filterWidth),
    poolsize=(maxPoolSize, maxPoolSize)
)
# Construct the second convolutional pooling layer
# filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
# maxpooling reduces this further to (8/2, 8/2) = (4, 4)
# 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
layer1 = LeNetConvPoolLayer(
    rng,
    input=layer0.output,
    image_shape=(batch_size, nkerns[0], l1H, l1W),
    filter_shape=(nkerns[1], nkerns[0], filterHeight, filterWidth),
    poolsize=(maxPoolSize, maxPoolSize)
)
# the HiddenLayer being fully-connected, it operates on 2D matrices of
# shape (batch_size, num_pixels) (i.e matrix of rasterized images).
# This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
# or (500, 50 * 4 * 4) = (500, 800) with the default values.
layer2_input = layer1.output.flatten(2)

# construct a fully-connected sigmoidal layer
layer2 = HiddenLayer(
    rng,
    input=layer2_input,
    n_in=nkerns[1] * l2H * l2W,
    n_out=500,
    activation=T.tanh
)

# classify the values of the fully-connected sigmoidal layer
layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)

# the cost we minimize during training is the NLL of the model
cost = layer3.negative_log_likelihood(y)

# create a list of all model parameters to be fit by gradient descent
params = layer3.params + layer2.params + layer1.params + layer0.params

# create a list of gradients for all model parameters
grads = T.grad(cost, params)

# train_model is a function that updates the model parameters by
# SGD Since this model has many parameters, it would be tedious to
# manually create an update rule for each model parameter. We thus
# create the updates list by automatically looping over all
# (params[i], grads[i]) pairs.
updates = [
    (param_i, param_i - learning_rate * grad_i)
    for param_i, grad_i in zip(params, grads)
]


# In[24]:




# In[25]:

f = file('classifier.save', 'rb')
loaded_objects = []
for i in range(8):
    loaded_objects.append(cPickle.load(f))
f.close()

for i in range(len(params)):
    params[i].set_value(loaded_objects[i])


# In[26]:

def swapIn(ims,label=None):
    test = np.arange(5293*784).reshape(5293,784).astype(float)
    #print "currentTest",test[0,:][:5]
    #print "swap",item[:5]
    #test[0,:] = item
    test[:len(ims),:] = np.array(ims)
    #print "updated",test[0,:][:5],"expected label",label
    test_set_x, test_set_y = shared_dataset([test, np.arange(5293)])
    return test_set_x

def predict(test_set_x,num):
    batch_size = 20
    model_predict = theano.function([index], layer3.y_pred,
             givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size]})
    return Series(np.concatenate([model_predict(i) for i in xrange(1)]))[:num]


def openIm(im,grey=True):
    if grey:
        return misc.imread(im,flatten=1)
    return misc.imread(im)

def shrink(im,height=28,width=28):
    return misc.imresize(im,(height,width))

def splitIm(im,grey=True):
    if grey:
        return im[:,:im.shape[1]/2],im[:,(im.shape[1]/2):]
    return im[:,:im.shape[1]/2,:],im[:,(im.shape[1]/2):,:]

def processIm(im,grey=True):
    im1,im2 = splitIm(openIm(im,grey),grey)
    return shrink(im1).flatten()/255,shrink(im2).flatten()/255


# In[28]:

def runPipeline(allFiles):
    ims = []
    for f in allFiles:
        im1,im2 = processIm(f)
        ims.append(im1)
    return predict(swapIn(ims,label=None),len(ims))



results = runPipeline(allFiles)

aerial,normal = [],[]

for res in zip(results,allFiles):
    if res[0] == 1:
        aerial.append(res)
    else:
        normal.append(res)

f = open("./aerial.txt",'w')
for res in aerial:
    f.writelines("\t".join([str(i) for i in res])+"\n")
f.close()

f = open("./normal.txt",'w')
for res in normal:
    f.writelines("\t".join([str(i) for i in res])+"\n")
f.close()


# In[ ]:



