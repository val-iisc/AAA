
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
from scipy.misc import imread, imresize


## Setting the configrations
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
config = tf.ConfigProto()
config.gpu_options.allow_growth = False
sess = tf.Session(config=config)

## Arguments

net_name = 'vggf'
train_type = 'with_proxy_data'

img_list_file = '/data1/krishna/tfff/data/ilsvrc_train_new.txt'
batch_size = 32
z_dim = 10
noise_image = imread('data/gaussian_noise.png')

if 'vgg' in net_name or net_name == 'caffenet':
	pre_softmax_layer = 'fc8'
elif net_name == 'resnet152':
	pre_softmax_layer = 'fc1000'
elif net_name == 'googlenet':
	pre_softmax_layer = 'loss3_classifier'

softmax_layer = 'prob'

def validate_arguments(args):
	nets = ['vggf', 'caffenet', 'vgg16', 'vgg19', 'googlenet',  'resnet152']

	if not(args.network in nets):
		print ('invalid network')
		exit (-1)

def choose_net(network,train_type, input_image, adv_image, range_image):
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
	with tf.variable_scope('target_classifier'):
		t_net = MAP[network](input_image)
		tf.get_variable_scope().reuse_variables()
		t_net_adv = MAP[network](tf.add(input_image,adv_image))
		t_net_rng = MAP[network](tf.add(range_image,adv_image))
	return t_net,t_net_adv, t_net_rng

def not_optim_layers(network):
	if network == 'vggf':
		return ['norm1', 'pool1', 'norm2', 'pool2', 'pool5', 'fc6', 'fc7', 'fc8', 'prob']
	if network == 'caffenet':
		return ['norm1', 'pool1', 'norm2', 'pool2', 'pool5', 'fc6', 'fc7', 'fc8', 'prob']
	elif network == 'vgg16':
		return ['pool1', 'pool2', 'pool3', 'pool4', 'pool5', 'fc6', 'fc7', 'fc8', 'prob']
	elif network == 'vgg19':
		return ['pool1', 'pool2', 'pool3', 'pool4', 'pool5', 'prob']
	elif network == 'googlenet':
		return ['pool1_3x3_s2', 'pool1_norm1', 'conv2_norm2', 'pool2_3x3_s2', 'pool3_3x3_s2', 'pool4_3x3_s2', 'pool5_7x7_s1', 'loss3_classifier', 'prob']
	elif network == 'resnet152':
		return ['bn_conv1', 'pool1', 'pool5', 'pool5_r', 'fc1000', 'prob']


def log_loss_ce(tar_prob_vec, adv_prob_vec, top_prob):
	size = batch_size
	for i in xrange(size):
		if i==0:
			loss= -tf.log(1 - adv_prob_vec[i][top_prob[i][0]] + 1e-4)
			mean = adv_prob_vec[i][top_prob[i][0]]
		else:
			loss = loss - tf.log(1 - adv_prob_vec[i][top_prob[i][0]]+ 1e-4)
			mean = mean + adv_prob_vec[i][top_prob[i][0]]
	mean = (mean/size)
	gen_loss = loss
	return gen_loss,mean

def log_loss(prob_vec, adv_prob_vec, top_prob):
	size = batch_size
	for i in xrange(size):
		if i==0:
			loss= adv_prob_vec[i][top_prob[i][0]]
		else:
			loss = loss + adv_prob_vec[i][top_prob[i][0]]
	mean = (loss/size)
	gen_loss = -tf.log(1-mean)
	return gen_loss,mean

def get_update_operation_func(train_type,in_im,sess,update,batch_size,size,img_list,
	inp_noise = np.random.uniform(low=-1. , high=1. ,size = (batch_size,z_dim))):
	if train_type == 'no_data':
		def updater(noiser,sess=sess,update=update):
			return sess.run(update,feed_dict={in_im:noiser, zn:inp_noise})
	elif train_type =='with_range':
		def updater(noiser,sess=sess,update=update,in_im=in_im,batch_size = batch_size,size=size):
			image_i = 'data/gaussian_noise.png'
			for j in range(batch_size):
				noiser[j:j+1] = np.copy(func.img_preprocess(image_i,size=size,augment=True))
			return sess.run(update,feed_dict={in_im:noiser, zn:inp_noise})
	elif train_type =='with_data':
		def updater(noiser,sess=sess,update=update,in_im=in_im,batch_size = batch_size,size=size,img_list=img_list):
			rander = np.random.randint(low=0,high=(len(img_list)-batch_size))
			for j in range(batch_size):
				noiser[j:j+1] = np.copy(func.img_preprocess(img_list[rander+j].strip(),size=size,augment=True))
			return sess.run(update,feed_dict={in_im:noiser, zn:inp_noise})

	return updater




val_data_prefix = '/data1/krishna/tfff/data/'
size = 224  
# getting the validation set
if net_name == 'caffenet':
	data_path = os.path.join('ilsvrc_small_val', 'caffenet_preprocessed.npy')
	size = 227
elif 'vgg' in net_name:
	data_path = os.path.join('ilsvrc_small_val','vgg_preprocessed.npy')
elif net_name =='googlenet':
	data_path = os.path.join('ilsvrc_small_val','googlenet_preprocessed.npy')
elif net_name =='resnet152':
	data_path = os.path.join('ilsvrc_small_val','resnet_preprocessed.npy')
	
imgs = np.load(val_data_prefix + data_path)
print('Loaded mini Validation Set')



if train_type == 'with_data':
	img_list = open(img_list_file).readlines()
elif train_type == 'with_proxy_data' :
	img_list = np.load('proxy_data/noise_images_res152.npy')

print "Loaded Proxy data for {:} ...".format(net_name)


data_l = tf.placeholder(tf.float32, shape=[batch_size,1])
zn = tf.placeholder(tf.float32, shape=[batch_size,z_dim])


with tf.variable_scope('pertub_generator'):
	G_N = Generator(zn)
	adv_image = G_N.generate()


input_image = tf.placeholder(shape=[None, size, size, 3],dtype='float32', name='input_image')
range_image = tf.placeholder(shape=[None, size, size, 3],dtype='float32', name='range_image')

net, adv_net, rng_net  = choose_net(net_name,train_type, input_image, adv_image, range_image)


adv_softmax = adv_net[softmax_layer]
true_softmax  = net[softmax_layer]

_, true_pred = tf.nn.top_k(true_softmax, 1)


opt_layers = not_optim_layers(net_name)

act_loss = -losses.l2_all(rng_net,opt_layers)
fooling_loss, adv_mean = log_loss(true_softmax, adv_softmax, true_pred)


gen_loss =  act_loss


gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'pertub_generator')
tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'pertub_generator')

optimizer = tf.train.AdamOptimizer(learning_rate=2e-4)
grads = optimizer.compute_gradients(gen_loss,tvars)
update = optimizer.apply_gradients(grads)


sess.run(tf.global_variables_initializer())
gen_loader = tf.train.Saver(gvars)
gen_saver_iter = tf.train.Saver(gvars) 
gen_saver_epoch = tf.train.Saver(gvars)
#gen_loader.restore(sess, 'weights_gen/best_valid-48000')

s1 = tf.summary.scalar('activation_loss', act_loss)
s2 = tf.summary.scalar('fooling_loss', fooling_loss)
s3 = tf.summary.scalar('confidence', adv_mean)

s4 = tf.summary.image('input_image', input_image)
s5 = tf.summary.image('adversarial_perturbaiton', adv_image)

fooling_acc = tf.placeholder(tf.float32)
s6 = tf.summary.scalar('fooling_rate',fooling_acc )
merge_op = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('logs/train', sess.graph)
valid_writer = tf.summary.FileWriter('logs/valid')


fool_rate = 0
max_iter = int(1e+6)
iteration_no = -1
no_samples = img_list.shape[0]
train_idxs_all = range(no_samples)
np.random.shuffle(train_idxs_all)
train_range_batch = np.zeros((batch_size,size,size,3))



while(iteration_no < max_iter):
	np.random.shuffle(train_idxs_all)
	## one epoch of training
	for i in range(no_samples/batch_size):
		iteration_no+=1
		train_idxs = train_idxs_all[i*batch_size: (i+1)*batch_size]
		train_images = img_list[train_idxs, ...]
		train_zn = np.random.uniform(low=-1. , high=1. ,size = (batch_size,z_dim))
		for j in range(batch_size):
			train_range_batch[j:j+1] = np.copy(func.img_preprocess(noise_image,size=size,augment=True))
		## Training step
		_, train_gen_loss, train_true_softmax, train_adv_softmax  = sess.run([update, gen_loss, true_softmax, adv_softmax], 
			feed_dict={input_image:train_images, zn:train_zn, range_image: train_range_batch})
		## Current Summary
		train_true_pred = np.argmax(train_true_softmax, axis = 1)
		train_adv_pred = np.argmax(train_adv_softmax, axis = 1)
		flip = np.sum([train_true_pred[j] != train_adv_pred[j] for j in range(batch_size) ])*(100.0/batch_size)
		print 'iter',iteration_no,'train loss', train_gen_loss, 'fool rate', flip
		## Writing summary
		if iteration_no%100 == 0 :
			train_summary = sess.run(merge_op, feed_dict={input_image:train_images, 
				zn:train_zn, fooling_acc: flip, range_image: train_range_batch})
			train_writer.add_summary(train_summary, iteration_no)
		## Validation
		if iteration_no% 100 == 0 :
			iters = int(math.ceil(1000/batch_size))
			temp = 0.0
			for j in range(iters):
				l = j*batch_size
				h = min((j+1)*batch_size,1000)
				valid_imgs = imgs[l:h]
				valid_zn = np.random.uniform(low=-1. , high=1. ,size = (batch_size,z_dim))
				valid_true_softmax, valid_adv_softmax = sess.run([true_softmax, adv_softmax],
					feed_dict={input_image: valid_imgs, zn: valid_zn})
				valid_true_pred = np.argmax(valid_true_softmax,axis=1)
				valid_adv_pred = np.argmax(valid_adv_softmax,axis=1)
				not_flip = np.sum(valid_true_pred==valid_adv_pred)
				temp += not_flip
			current_rate = ((iters*batch_size-temp)/1000.0)*100.0
			print 'iter',iteration_no, 'val fool rate', current_rate
			gen_saver_iter.save(sess, 'weights_gen/iteration',global_step = iteration_no,  write_meta_graph = False)	
			## saving the best performing models and adv images 
			valid_summary = sess.run(merge_op, feed_dict={input_image:valid_imgs, fooling_acc: current_rate, 
				zn: valid_zn, range_image: train_range_batch})
			valid_writer.add_summary(valid_summary, iteration_no)
			if current_rate>fool_rate:
				print('best_performance_till_now')
				fool_rate = current_rate
				gen_saver_epoch.save(sess, 'weights_gen/best_valid',global_step = iteration_no,  write_meta_graph = False)
				im = sess.run(adv_image, feed_dict={
					zn: valid_zn })
				name = 'perturbations/'+net_name+'_'+train_type+'.npy'
				np.save(name,im)