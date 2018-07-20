import sys
sys.path.insert(0,'tensorflow-classification')
import matplotlib.pyplot as plt
import pylab
# imports from tensorflow_classification
from nets.vgg_f import vggf
from nets.caffenet import caffenet
from nets.vgg_16 import vgg16
from nets.vgg_19 import vgg19
from nets.googlenet import googlenet
from nets.resnet_152 import resnet152
from misc.utils import *

import tensorflow as tf
import numpy as np
import argparse
import os
import time
import math
import utils.functions as func
import utils.losses as losses
from misc.layers import *
from generator import *



## Setting the configrations
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
config = tf.ConfigProto()
config.gpu_options.allow_growth = False
sess = tf.Session(config=config)

## Arguments

net_name = 'vggf'

if 'vgg' in net_name or net_name == 'caffenet':
	pre_softmax = 'fc8'
elif net_name == 'resnet152':
	pre_softmax = 'fc1000'
elif net_name == 'googlenet':
	pre_softmax = 'loss3_classifier'


batch_size = 1
learning_rate = 10
if net_name == 'caffenet':
	size = 227
else:
	size = 224
def validate_arguments(args):
	nets = ['vggf', 'caffenet', 'vgg16', 'vgg19', 'googlenet',  'resnet152']

	if not(args.network in nets):
		print ('invalid network')
		exit (-1)

def choose_net(network, input_image):
	MAP = {
			'vggf'     : vggf,
			'caffenet' : caffenet,
			'vgg16'    : vgg16,
			'vgg19'    : vgg19, 
			'googlenet': googlenet, 
			'resnet152': resnet152
			}
	if network == 'caffenet':
		size = 227
	else:
		size = 224
	# placeholder to pass image
	
	with tf.variable_scope('target_classifier'):
		t_net = MAP[network](input_image)
	return t_net

def show_img_rgb(x):
	x_squeezed = np.squeeze(x)
	temp = x_squeezed - np.min(x_squeezed)
	temp = temp/np.max(temp)
	temp1 = np.copy(temp)
	for i in range(3):
		temp1[:,:,i] = temp[:,:,2-i]
	plt.imshow(temp1)
	plt.show()

def save_img_rgb_2(x, cls_no, iter_):
	x_squeezed = np.squeeze(x)
	temp = x_squeezed - np.min(x_squeezed)
	temp = temp/np.max(temp)
	temp1 = np.copy(temp)
	for i in range(3):
		temp1[:,:,i] = temp[:,:,2-i]
	image_name  = 'proxy_data/ci/noise_image_' + str(cls_no) + '_' + str(iter_)+ '.png'
	plt.imshow(temp1)
	plt.savefig(image_name)

noise_l = tf.placeholder(tf.float32, shape=[batch_size,1])
noise_l_strip = tf.cast(tf.squeeze(noise_l, axis = 1), tf.int32)
jitter_16 = tf.tile(tf.random_uniform([1,1,1,3],minval=-16.0,maxval=16.0 ), [1,size +2*16 ,size + 2*16 , 1])
scale_f  = tf.placeholder(tf.float32, shape = [batch_size])
paded_size_16 = size + 2*16

scaled_size = tf.cast(paded_size_16*scale_f , tf.int32)
scale_tensor = tf.tile(scaled_size,[2])
rotate_f = tf.placeholder(tf.float32, shape = [batch_size])
padding_16 = tf.constant([[0,0],[16,16], [16, 16], [0,0]])
jitter_8 = tf.tile(tf.random_uniform([1,1,1,3],minval=-8.0,maxval=8.0 ), [1,scaled_size[0],scaled_size[0], 1])
crop_size = tf.constant([1, size, size, 3])
aug_noise = tf.random_uniform([1,size,size,3],minval=-10.0,maxval=10.0)

initial_noise_image = tf.placeholder(tf.float32, shape=[1,size, size, 3])

noise_image = tf.Variable(tf.random_uniform([1,size,size,3],minval=-123.68,maxval=151.061), name='noise_image', dtype='float32')
noise_image_init = noise_image.assign(initial_noise_image)


noise_image_clip  = noise_image.assign(tf.clip_by_value(noise_image, -123.68, 151.061))

noise_image_flip_lr = tf.reverse(noise_image, tf.constant([2]))

noise_image_noised = tf.add(noise_image, aug_noise)
noise_image_pad = tf.pad(noise_image_noised, padding_16, "REFLECT")
noise_image_jitter_16 = tf.add(noise_image_pad, jitter_16)
noise_image_scaled = tf.image.resize_images(noise_image_jitter_16, scale_tensor)
noise_image_rotate = tf.contrib.image.rotate(noise_image_scaled, rotate_f)
noise_image_jitter_8 = tf.add(noise_image_rotate, jitter_8)
noise_image_cropped = tf.image.crop_to_bounding_box(noise_image_jitter_8, 16, 16, size, size )






net  = choose_net(net_name, noise_image_cropped)

net_logits  = net[pre_softmax]
net_softmax  = net['prob']
tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'noise_image')

noise_cost_softmax = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits= net_logits, labels = noise_l_strip ))
noise_cost_logits = -net_logits[0][noise_l_strip[0]]
noise_l2_loss = tf.nn.l2_loss(noise_image)

noise_cost = noise_cost_logits
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
grads = optimizer.compute_gradients(noise_cost,tvars)
update = optimizer.apply_gradients(grads)

sess.run(tf.global_variables_initializer())



max_iter_noise = int(1e+6)
noise = np.zeros((batch_size,size,size,3))
noise_image_all = np.zeros((1000, size, size, 3))
saturation = 0
noise_image_base  = np.load('ilsvrc_mean.npy')

curr_time = time.time()


stop


for cls_no in range(1000):
	## Picking a cls label
	noise_l_train = cls_no*np.ones((batch_size,1 ))
	for sam_no in range(1):
		## initializing a noise image
		flip_confid = np.random.uniform(low = 0.55, high= 0.99)
		initial_noise_image_train = noise_image_base + np.random.randn(1,size,size,3)
		_ = sess.run(noise_image_init, feed_dict={initial_noise_image: initial_noise_image_train})
		for j in range(1500):
			rand_rotate_value = 0.0174533* np.random.randint(low =-5, high =6, size= 1)
			rand_scale_value = np.random.choice([ 1, 0.975, 1.025, 0.95, 1.05], size = 1 )
			_,_, train_logits, train_softmax = sess.run([noise_image_clip, update, net_logits, net_softmax], feed_dict={noise_l: noise_l_train, scale_f: rand_scale_value, rotate_f: rand_rotate_value})
			train_pred = np.argmax(train_softmax, axis=1)
			train_confid =  [train_softmax[i, cls_no] for i in range(batch_size)]
			train_evid = [train_logits[i, cls_no] for i in range(batch_size)]
			if np.mean(train_confid) > flip_confid or j == 1499: 
				train_noise_image, train_noise_loss= sess.run([noise_image, noise_cost], feed_dict={noise_l: noise_l_train, scale_f: rand_scale_value, rotate_f: rand_rotate_value})
				noise_image_all[cls_no + sam_no, ...] = train_noise_image
				print 'cls', cls_no, 'iter', j, 'pred', np.mean(train_pred), 'confid', np.mean(train_confid), 'evid', np.mean(train_evid) 
				if j == 1499:
					saturation += 1
				break

np.save('proxy_data/noise_images_res152.npy', noise_image_all)


for cls_no in [383]:
	## Picking a cls label
	noise_l_train = cls_no*np.ones((batch_size,1 ))
	for sam_no in range(1):
		## initializing a noise image
		flip_confid = np.random.uniform(low = 0.55, high= 0.99)
		initial_noise_image_train = noise_image_base + np.random.randn(1,size,size,3)
		_ = sess.run(noise_image_init, feed_dict={initial_noise_image: initial_noise_image_train})
		for j in range(2500):
			rand_rotate_value = 0.0174533* np.random.randint(low =-5, high =6, size= 1)
			rand_scale_value = np.random.choice([ 1, 0.975, 1.025, 0.95, 1.05], size = 1 )
			_,_, train_logits, train_softmax = sess.run([noise_image_clip, update, net_logits, net_softmax], feed_dict={noise_l: noise_l_train, scale_f: rand_scale_value, rotate_f: rand_rotate_value})
			train_pred = np.argmax(train_softmax, axis=1)
			train_confid =  [train_softmax[i, cls_no] for i in range(batch_size)]
			train_evid = [train_logits[i, cls_no] for i in range(batch_size)]
			if j % 10 == 0: 
				train_noise_image, train_noise_loss= sess.run([noise_image, noise_cost], feed_dict={noise_l: noise_l_train, scale_f: rand_scale_value, rotate_f: rand_rotate_value})
				noise_image_all[cls_no + sam_no, ...] = train_noise_image
				save_img_rgb_2(train_noise_image, cls_no, j)
				print 'cls', cls_no, 'iter', j, 'pred', np.mean(train_pred), 'confid', np.mean(train_confid), 'evid', np.mean(train_evid) 
				if j == 2499:
					break