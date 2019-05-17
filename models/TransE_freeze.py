#coding:utf-8
import numpy as np
import os
import tensorflow as tf
import logging
from .Model import Model

class TransE_freeze(Model):	
	r'''
	TransE is the first model to introduce translation-based embedding, 
	which interprets relations as the translations operating on entities.
	'''
	def _calc(self, h, t, r):
		return abs(h + r - t)

	def embedding_def(self):
		#Obtaining the initial configuration of the model
		config = self.get_config()
		
		#Defining required parameters of the model, including embeddings of entities and relations
		# self.ent_embeddings = tf.get_variable(name = "ent_embeddings", shape = [config.entTotal, config.hidden_size], initializer = ent_embedding_initializer, trainable=FALSE) #initialize with old embeddings
		# self.rel_embeddings = tf.get_variable(name = "rel_embeddings", shape = [config.relTotal, config.hidden_size], initializer = rel_embedding_initializer, trainable=FALSE) #initialize with old embeddings
		self.ent_embeddings = tf.get_variable(name = "ent_embeddings",\
		  shape = [config.entTotal, config.hidden_size],\
		  initializer = config.ent_embedding_initializer,\
		  trainable = True) #initialize with old embeddings
		self.rel_embeddings = tf.get_variable(name = "rel_embeddings",\
          shape = [config.relTotal, config.hidden_size],\
		  initializer = config.rel_embedding_initializer,\
		  trainable = True) #initialize with old embeddings

		logging.warning('Initialized embddings in transE')
		# Initialize to be the size of the difference between new and old entity2id and relation2id files, respectively
		# self.test_ent_embeddings = tf.get_variable(name = "test_ent_embeddings",\
		#     shape = [12, config.hidden_size],\
		#     initializer = tf.contrib.layers.xavier_initializer(uniform = False))
		# self.test_rel_embeddings = tf.get_variable(name = "test_rel_embeddings",\
		# 	shape = [9, config.hidden_size],\
		# 	initializer = tf.contrib.layers.xavier_initializer(uniform = False))

		self.parameter_lists = {"ent_embeddings":self.ent_embeddings, \
								"rel_embeddings":self.rel_embeddings}#, \
								# "test_ent_embeddings":self.test_ent_embeddings, \
								# "test_rel_embeddings":self.test_rel_embeddings								
								# }

	def loss_def(self):
		#Obtaining the initial configuration of the model
		config = self.get_config()
		#To get positive triples and negative triples for training
		# h: head
		# t: tail
		# r: relation?
		#The shapes of pos_h, pos_t, pos_r are (batch_size, 1)
		#The shapes of neg_h, neg_t, neg_r are (batch_size, negative_ent + negative_rel)
		pos_h, pos_t, pos_r = self.get_positive_instance(in_batch = True)
		neg_h, neg_t, neg_r = self.get_negative_instance(in_batch = True)
		#Embedding entities and relations of triples, e.g. p_h, p_t and p_r are embeddings for positive triples
		# Our goal then is to get a positive embedding from the train list, for either an 
		# entity, or relationship, and compare it against existing embeddings
		# fn = "./res/pos_h.txt"
		# if os.path.exists(fn):
		#     append_write = 'a' # append if already exists
		# else:
		#     append_write = 'w' # make a new file if not
		# with open(fn, append_write) as f:
		# 	f.write(pos_h)
		# 	f.write("\n")

		fake_pos_r = tf.constant(1347*np.ones([24158,1], dtype=np.int))		#1348: No, 1344: yes, 1345: yes, 1347: yes
		logging.warning('made it to fake pos r')
		p_h = tf.nn.embedding_lookup(self.ent_embeddings, pos_h) # What if pos_h is larger than the size of the embedding?
		p_t = tf.nn.embedding_lookup(self.ent_embeddings, pos_t)
		p_r = tf.nn.embedding_lookup(self.rel_embeddings, pos_r)
		n_h = tf.nn.embedding_lookup(self.ent_embeddings, neg_h)
		n_t = tf.nn.embedding_lookup(self.ent_embeddings, neg_t)
		n_r = tf.nn.embedding_lookup(self.rel_embeddings, neg_r)

		self.p_h = p_h
		self.pos_h = pos_h
		self.pos_r = pos_r
		self.fake_pos_r = fake_pos_r

		# print("Pos_h")
		# print(pos_h)		
		# print(p_h)
		# # print(pos_h.eval())
		# # print(p_h.eval())

		# print("pos_t")
		# print(pos_h)		
		# print(p_t)


		# print("pos_r")
		# print(pos_r)		
		# print(p_r)

		# print("neg_h")
		# print(neg_h)		
		# print(n_h)

		# print("neg_t")
		# print(neg_h)		
		# print(n_t)


		# print("neg_r")
		# print(neg_r)		
		# print(n_r)		

		#Calculating score functions for all positive triples and negative triples
		#The shape of _p_score is (batch_size, 1, hidden_size)
		#The shape of _n_score is (batch_size, negative_ent + negative_rel, hidden_size)
		_p_score = self._calc(p_h, p_t, p_r)
		_n_score = self._calc(n_h, n_t, n_r)
		#The shape of p_score is (batch_size, 1)
		#The shape of n_score is (batch_size, 1)
		p_score =  tf.reduce_sum(tf.reduce_mean(_p_score, 1, keep_dims = False), 1, keep_dims = True)
		n_score =  tf.reduce_sum(tf.reduce_mean(_n_score, 1, keep_dims = False), 1, keep_dims = True)
		#Calculating loss to get what the framework will optimize
		self.loss = tf.reduce_sum(tf.maximum(p_score - n_score + config.margin, 0))

	def predict_def(self):
		predict_h, predict_t, predict_r = self.get_predict_instance()
		predict_h_e = tf.nn.embedding_lookup(self.ent_embeddings, predict_h)
		predict_t_e = tf.nn.embedding_lookup(self.ent_embeddings, predict_t)
		predict_r_e = tf.nn.embedding_lookup(self.rel_embeddings, predict_r)
		self.predict = tf.reduce_mean(self._calc(predict_h_e, predict_t_e, predict_r_e), 1, keep_dims = False)