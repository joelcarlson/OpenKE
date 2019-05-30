#coding:utf-8
import numpy as np
import tensorflow as tf
from .Model import Model
import logging


l1 = logging.getLogger('root')
l1.setLevel(logging.WARNING)
# l1.setLevel(logging.DEBUG)

gv_log = logging.FileHandler('y_and_res.log')
gv_log.setLevel(logging.DEBUG)
l1.addHandler(gv_log)

class ComplEx_freeze(Model):

	def embedding_def(self):
		config = self.get_config()
		
		# Real is first half of embedding, Im is second
		real_idx = config.hidden_size // 2
		im_idx = config.hidden_size
		logging.warning("real_idx {}".format(real_idx))

		ent1_initilializer = tf.constant_initializer(np.array(config.ent_embedding_initializer)[:,0:real_idx] , verify_shape=True)	
		ent2_initilializer = tf.constant_initializer(np.array(config.ent_embedding_initializer)[:,real_idx:im_idx] , verify_shape=True)	
		rel1_initilializer = tf.constant_initializer(np.array(config.rel_embedding_initializer)[:,0:real_idx] , verify_shape=True)	
		rel2_initilializer = tf.constant_initializer(np.array(config.rel_embedding_initializer)[:,real_idx:im_idx] , verify_shape=True)	


		self.ent1_embeddings = tf.get_variable(name = "ent1_embeddings",\
		  shape = [config.entTotal, config.hidden_size//2],\
		  initializer = ent1_initilializer,\
		  trainable = True) #initialize with old embeddings
		self.ent2_embeddings = tf.get_variable(name = "ent2_embeddings",\
		  shape = [config.entTotal, config.hidden_size//2],\
		  initializer = ent2_initilializer,\
		  trainable = True) #initialize with old embeddings		
		self.rel1_embeddings = tf.get_variable(name = "rel1_embeddings",\
          shape = [config.relTotal, config.hidden_size//2],\
		  initializer = rel1_initilializer,\
		  trainable = True) #initialize with old embeddings
		self.rel2_embeddings = tf.get_variable(name = "rel2_embeddings",\
          shape = [config.relTotal, config.hidden_size//2],\
		  initializer = rel2_initilializer,\
		  trainable = True) #initialize with old embeddings				  		

		self.parameter_lists = {"ent_re_embeddings":self.ent1_embeddings, \
								"ent_im_embeddings":self.ent2_embeddings, \
								"rel_re_embeddings":self.rel1_embeddings, \
								"rel_im_embeddings":self.rel2_embeddings}

		

		  
	r'''
	ComplEx extends DistMult by introducing complex-valued embeddings so as to better model asymmetric relations. 
	It is proved that HolE is subsumed by ComplEx as a special case.
	'''
	def _calc(self, e1_h, e2_h, e1_t, e2_t, r1, r2):
		return e1_h * e1_t * r1 + e2_h * e2_t * r1 + e1_h * e2_t * r2 - e2_h * e1_t * r2

	def loss_def(self):
		#Obtaining the initial configuration of the model
		config = self.get_config()

		batch_size = config.batch_size
		negative_ent = config.negative_ent
		negative_rel = config.negative_rel
		#To get positive triples and negative triples for training
		#To get labels for the triples, positive triples as 1 and negative triples as -1
		#The shapes of h, t, r, y are (batch_size, 1 + negative_ent + negative_rel)
		h, t, r = self.get_all_instance()
		# y = self.get_all_labels()

		logging.warning("h dim: {}".format(h.shape)) # (neg_ent + neg_rel + 1)*batch_size (+1 is from 1 pos_ent per set of negs)
		# logging.warning("y dim: {}".format(y.shape))
		#Embedding entities and relations of triples
		e1_h = tf.nn.embedding_lookup(self.ent1_embeddings, h)
		e2_h = tf.nn.embedding_lookup(self.ent2_embeddings, h)
		e1_t = tf.nn.embedding_lookup(self.ent1_embeddings, t)
		e2_t = tf.nn.embedding_lookup(self.ent2_embeddings, t)
		r1 = tf.nn.embedding_lookup(self.rel1_embeddings, r)
		r2 = tf.nn.embedding_lookup(self.rel2_embeddings, r)
		#Calculating score functions for all positive triples and negative triples
		res = tf.reduce_sum(self._calc(e1_h, e2_h, e1_t, e2_t, r1, r2), 1, keep_dims = False)	

		# Labels are simply a list of 1s as long as the batch size, with an accompanying zero
		labels = tf.stack(tf.split(tf.tile([1,0],[batch_size]), batch_size))

		# Get positive and negative scores. Positive scores are the first N_batch size, and
		# the remaining are the negative scores. for each positive score there are negative_ent + negative_rel
		# negative scores
		pos_scores = tf.split(res[0:batch_size], batch_size)
		neg_scores = tf.split(res[batch_size:], batch_size)

		# shortcut to save computation time
		logsumexp_neg_scores = tf.math.reduce_logsumexp(neg_scores, 1, keep_dims=True)
		logits = tf.concat([pos_scores, logsumexp_neg_scores], axis=1)
		loss_func = tf.losses.softmax_cross_entropy(onehot_labels=labels,
		                                       logits=logits,
		                                       reduction=tf.losses.Reduction.SUM)

		# To convert from (-1, 1) coding we add 1, then divide by 2 to go to (0, 1) coding 
		# y_cross_ent = tf.cast(tf.math.divide(tf.math.add(y, tf.constant(1, dtype=tf.float32)), tf.constant(2, dtype=tf.float32)), dtype=tf.int32)


		logging.warning("Res dim: {}".format(res.shape))
		# logging.warning("- y * res dim: {}".format((- y * res).shape))
		l1.debug("res : {}".format(res))
		# l1.debug("y : {}".format(y))
		# l1.debug("y2 : {}".format(y_cross_ent)) # Convert y to cross entropy range
		l1.debug("------")

		
		# cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_cross_ent,logits=res)	
		
		# loss_func = tf.reduce_mean(cross_entropy)
		# self.ld_res = res
		# self.ld_y = y
		# # self.ld_y2 = y_cross_ent
		# # self.ld_softplus = tf.nn.softplus(- y * res)
		# self.ld_loss_func = loss_func


		# For freezing embeddings using a typical regularizer such as this is not particularly meaningful, as it is tabulating the 
		# function for many vectors that we have no wish to change
		regul_func = tf.reduce_mean(e1_h ** 2) + tf.reduce_mean(e1_t ** 2) + tf.reduce_mean(e2_h ** 2) + tf.reduce_mean(e2_t ** 2) + tf.reduce_mean(r1 ** 2) + tf.reduce_mean(r2 ** 2)

		
		# I am imagining some future scenario where a part of the loss function is something that
		# Penalizes distributional differences between positive and negative samples, since we can almost guarantee 
		# that negative samples will be drawn from the (much larger) training set. For now, I just
		# wish to be able to track the mean magnitude of the newly produced vectors
		self.pos_ent_mean_magnitude = tf.reduce_mean(tf.reduce_mean(tf.math.abs(e1_h[0:batch_size,]), 1)) # Mean of means of embeddings
		self.pos_ent_min = tf.reduce_min(e1_h[0:batch_size,])
		self.pos_ent_max = tf.reduce_max(e1_h[0:batch_size,])
		self.pos_ent_sd = tf.reduce_mean(tf.math.reduce_std(e1_h[0:batch_size,], 1)) # mean of sds of embeddings


		# Another option is to clamp max norm of the weight vectors using something like the keras.constrains.MaxNorm function after weight update
		# See: 
		# https://stats.stackexchange.com/questions/257996/what-is-maxnorm-constraint-how-is-it-useful-in-convolutional-neural-networks
		# https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/keras/constraints.py 
		# http://cs231n.github.io/neural-networks-2/#reg		

		#Calculating loss to get what the framework will optimize
		self.loss =  loss_func + config.lmbda * regul_func

	def predict_def(self):
		config = self.get_config()
		predict_h, predict_t, predict_r = self.get_predict_instance()
		predict_h_e1 = tf.nn.embedding_lookup(self.ent1_embeddings, predict_h)
		predict_t_e1 = tf.nn.embedding_lookup(self.ent1_embeddings, predict_t)
		predict_r_e1 = tf.nn.embedding_lookup(self.rel1_embeddings, predict_r)
		predict_h_e2 = tf.nn.embedding_lookup(self.ent2_embeddings, predict_h)
		predict_t_e2 = tf.nn.embedding_lookup(self.ent2_embeddings, predict_t)
		predict_r_e2 = tf.nn.embedding_lookup(self.rel2_embeddings, predict_r)
		self.predict = -tf.reduce_sum(self._calc(predict_h_e1, predict_h_e2, predict_t_e1, predict_t_e2, predict_r_e1, predict_r_e2), 1, keep_dims = True)

