import sys
sys.path.insert(0,'tensorflow-classification')
import matplotlib.pyplot as plt

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
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
config = tf.ConfigProto()
config.gpu_options.allow_growth = False
sess = tf.Session(config=config)

## Arguments

net_name = 'vggf'
img_list = None
batch_size = 32
z_dim = 10
gt_labels = '/data1/krishna/tfff/DG-UAP/classification/utils/gt.txt'
img_list = '/data1/krishna/tfff/DG-UAP/classification/utils/ilsvrc_test_n.txt'
adv_im_path = None
use_generator = True


def validate_arguments(network, adv_im_path, img_list, gt_labels):
	nets = ['vggf', 'caffenet', 'vgg16', 'vgg19', 'googlenet', 'resnet152']

	if not(network in nets):
		print ('invalid network')
		exit (-1)
	if adv_im_path is None and use_generator ==  False:
		print ('no path to perturbation')
		exit (-1)
	if img_list is None or gt_labels is None:
		print ('provide image list and labels')
		exit (-1)


def choose_net(network, input_image, adv_image):
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

	pert_load = adv_image
	# preprocessing if necessary
	if (pert_load.shape[1] == 224 and size == 227):
		pert_load = fff_utils.upsample(np.squeeze(pert_load))
	elif (pert_load.shape[1] == 227 and size == 224):
		pert_load = fff_utils.downsample(np.squeeze(pert_load))
	elif (pert_load.shape[1] not in [224, 227]):
		print(pert_load.shape[1])
		raise Exception("Invalid size of input perturbation")


	input_batch = tf.concat([input_image, tf.add(input_image,adv_image)], 0)
	with tf.variable_scope('target_classifier'):
		t_net = MAP[network](input_batch)
	return t_net



size = 224  
if net_name == 'caffenet':
	size = 227

if use_generator:
	with tf.variable_scope('generator'):
		G = Generator()
		adv_image = G.generate()
else:
	adv_image = np.load(adv_im_path)

input_image = tf.placeholder(shape=[None, size, size, 3],dtype='float32', name='input_image')

net = choose_net(net_name, input_image, adv_image)

sess.run(tf.global_variables_initializer())
gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'generator')
gen_loader = tf.train.Saver(gvars)
gen_loader.restore(sess, 'weights_gen/best_valid-11000')

imgs = open(img_list).readlines()#[::10]
gt_labels = open(gt_labels).readlines()#[::10]
fool_rate = 0
top_1 = 0
top_1_real = 0
isotropic,size = get_params(net_name)
batch_im = np.zeros((batch_size, size,size,3))


img_loader = loader_func(net_name,sess,isotropic,size)

for i in range(len(imgs)/batch_size):
	lim = min(batch_size,len(imgs)-i*batch_size)
	for j in range(lim):
		im = img_loader(imgs[i*batch_size+j].strip())
		batch_im[j] = np.copy(im)
	gt = np.array([int(gt_labels[i*batch_size+j].strip()) for j in range(lim)])
	softmax_scores = sess.run(net['prob'], feed_dict={input_image: batch_im, G.zn:np.random.uniform(low=-1. , high=1. ,size = (batch_size,z_dim))})
	true_predictions = np.argmax(softmax_scores[:batch_size],axis=1)
	ad_predictions = np.argmax(softmax_scores[batch_size:],axis=1)
	if i!=0 and i%100 == 0:
		print 'iter: {:5d}\ttop-1_real: {:04.2f}\ttop-1: {:04.2f}\tfooling-rate: {:04.2f}'.format(i,
		(top_1_real/float(i*batch_size))*100,(top_1/float(i*batch_size))*100, (fool_rate)/float(i*batch_size)*100)
	top_1 +=np.sum(ad_predictions == gt)
	top_1_real +=np.sum(true_predictions == gt)
	fool_rate +=np.sum(true_predictions != ad_predictions)
print 'Real Top-1 Accuracy = {:.2f}'.format(top_1_real/float(len(imgs))*100)
print 'Top-1 Accuracy = {:.2f}'.format(top_1/float(len(imgs))*100)
print 'Fooling Rate = {:.2f}'.format(fool_rate/float(len(imgs))*100)