# 
#
# Author : fcbruce <fcbruce8964@gmail.com>
#
# Time : Wed 14 Jun 2017 18:46:19
#
#

from __future__ import print_function

import time
from PIL import Image
import numpy as np

from keras import backend
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave

height = width = 512

def preprocess(x):
    x[:, :, 0] -= 103.939
    x[:, :, 1] -= 116.779
    x[:, :, 2] -= 123.68
    x = x[:, :, :, ::-1]
    return x

content_img_path = 'lena.jpg'
content_img = Image.open(content_img_path)
content_img = content_img.resize((height, width))
content_arr = np.asarray(content_img, dtype='float32')
content_arr = np.expand_dims(content_arr, axis=0)
print(content_arr.shape)
content_arr = preprocess(content_arr)

style_img_path = 'wave.jpg'
style_img = Image.open(style_img_path)
style_img = style_img.resize((height, width))
style_arr = np.asarray(style_img, dtype='float32')
style_arr = np.expand_dims(style_arr, axis=0)
style_arr = preprocess(style_arr)

content_img = backend.variable(content_arr)
style_img = backend.variable(style_arr)
combination_img = backend.placeholder((1, height, width, 3))

input_tensor = backend.concatenate([content_img, style_img, combination_img], axis=0)

model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)
layers = dict([(layer.name, layer.output) for layer in model.layers])

c_w = 0.025
s_w = 5.0
w = 1.

loss = backend.variable(0.)

def content_loss(content, combination):
    return backend.sum(backend.square(combination - content))

layer_features = layers['block2_conv2']
content_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += c_w * content_loss(content_features, combination_features)

def gram_matrix(x):
    features = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
    gram = backend.dot(features, backend.transpose(features))
    return gram

def style_loss(style, combination):
    s = gram_matrix(style)
    c = gram_matrix(combination)
    channels = 3
    size = height * width
    return backend.sum(backend.square(s - c) / (4. * (channels ** 2) * (size ** 2)))

feature_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']

for layer_name in feature_layers:
    layer_features = layers[layer_name]
    style_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_features, combination_features)
    loss += (s_w / len(feature_layers)) * sl

def total_variation_loss(x):
    a = backend.square(x[:, :height - 1, :width - 1, :] - x[:, 1:, :width - 1, :])
    b = backend.square(x[:, :height - 1, :width - 1, :] - x[:, :height - 1, :1, :])
    return backend.sum(backend.pow(a + b, 1.25))

loss += w * total_variation_loss(combination_img)

grads = backend.gradients(loss, combination_img)

outputs = [loss]
outputs += grads
f_outputs = backend.function([combination_img], outputs)

def eval_loss_and_grads(x):
    x = x.reshape((1, height, width, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    grad_values = outs[1].flatten().astype('float64')
    return loss_value, grad_values

class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grad_values = None

    def loss(self, x):
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

x = np.random.uniform(0, 255, (1, height, width, 3)) - 128.0

iteration = 100

for i in range(iteration):

    print("start iteration #", i)

    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxfun=20)
    print("current loss value: ", min_val)

    y = x.reshape((height, width, 3))
    y = y[:, :, ::-1]
    y[:, :, 0] += 103.939
    y[:, :, 1] += 116.779
    y[:, :, 2] += 123.68
    y = np.clip(y, 0, 255).astype('uint8')

    img = Image.fromarray(y)
    imsave('lena_%d.jpg' % i, img)
