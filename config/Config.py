#coding:utf-8
import numpy as np
import tensorflow as tf
import os
import time
import datetime
import ctypes
import json
import logging

# create file handler which logs even debug messages
l1 = logging.getLogger('root')
# l1.setLevel(logging.DEBUG)
l1.setLevel(logging.WARNING)

gv_log = logging.FileHandler('debug.log')
gv_log.setLevel(logging.DEBUG)
l1.addHandler(gv_log)

class Config(object):
	'''
	use ctypes to call C functions from python and set essential parameters.
	'''
	def __init__(self):
		base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../release/Base.so'))
		self.lib = ctypes.cdll.LoadLibrary(base_file)
		self.lib.sampling.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
		self.lib.getHeadBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
		self.lib.getTailBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
		self.lib.testHead.argtypes = [ctypes.c_void_p]
		self.lib.testTail.argtypes = [ctypes.c_void_p]
		self.lib.getTestBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
		self.lib.getValidBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
		self.lib.getBestThreshold.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
		self.lib.test_triple_classification.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
		self.test_flag = False
		self.in_path = None
		self.out_path = None
		self.bern = 0
		self.hidden_size = 100
		self.train_times = 0
		self.margin = 1.0
		self.nbatches = 100
		self.negative_ent = 1
		self.negative_rel = 0
		self.workThreads = 1
		self.alpha = 0.001
		self.lmbda = 0.000
		self.log_on = 1
		self.exportName = None
		self.importName = None
		self.export_steps = 0
		self.opt_method = "SGD"
		self.optimizer = None
		self.test_link_prediction = False
		self.test_triple_classification = False
		self.early_stopping = None # It expects a tuple of the following: (patience, min_delta)
		self.freeze_train_embeddings = False
		self.embedding_initializer_path = None

	def init_link_prediction(self):
		r'''
		import essential files and set essential interfaces for link prediction
		'''
		self.lib.importTestFiles()
		self.lib.importTypeFiles()
		self.test_h = np.zeros(self.lib.getEntityTotal(), dtype = np.int64)
		self.test_t = np.zeros(self.lib.getEntityTotal(), dtype = np.int64)
		self.test_r = np.zeros(self.lib.getEntityTotal(), dtype = np.int64)
		self.test_h_addr = self.test_h.__array_interface__['data'][0]
		self.test_t_addr = self.test_t.__array_interface__['data'][0]
		self.test_r_addr = self.test_r.__array_interface__['data'][0]

	def init_triple_classification(self):
		r'''
		import essential files and set essential interfaces for triple classification
		'''
		self.lib.importTestFiles()
		self.lib.importTypeFiles()

		self.test_pos_h = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
		self.test_pos_t = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
		self.test_pos_r = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
		self.test_neg_h = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
		self.test_neg_t = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
		self.test_neg_r = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
		self.test_pos_h_addr = self.test_pos_h.__array_interface__['data'][0]
		self.test_pos_t_addr = self.test_pos_t.__array_interface__['data'][0]
		self.test_pos_r_addr = self.test_pos_r.__array_interface__['data'][0]
		self.test_neg_h_addr = self.test_neg_h.__array_interface__['data'][0]
		self.test_neg_t_addr = self.test_neg_t.__array_interface__['data'][0]
		self.test_neg_r_addr = self.test_neg_r.__array_interface__['data'][0]

		self.valid_pos_h = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
		self.valid_pos_t = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
		self.valid_pos_r = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
		self.valid_neg_h = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
		self.valid_neg_t = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
		self.valid_neg_r = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
		self.valid_pos_h_addr = self.valid_pos_h.__array_interface__['data'][0]
		self.valid_pos_t_addr = self.valid_pos_t.__array_interface__['data'][0]
		self.valid_pos_r_addr = self.valid_pos_r.__array_interface__['data'][0]
		self.valid_neg_h_addr = self.valid_neg_h.__array_interface__['data'][0]
		self.valid_neg_t_addr = self.valid_neg_t.__array_interface__['data'][0]
		self.valid_neg_r_addr = self.valid_neg_r.__array_interface__['data'][0]
		self.relThresh = np.zeros(self.lib.getRelationTotal(), dtype = np.float32)
		self.relThresh_addr = self.relThresh.__array_interface__['data'][0]

	# prepare for train and test
	def init(self):
		self.trainModel = None
		if self.in_path != None:
			self.lib.setInPath(ctypes.create_string_buffer(self.in_path.encode(), len(self.in_path) * 2))
			self.lib.setBern(self.bern)
			self.lib.setWorkThreads(self.workThreads)
			self.lib.randReset()
			self.lib.importTrainFiles()
			logging.warning('Imported train')	
			self.relTotal = self.lib.getRelationTotal()
			self.entTotal = self.lib.getEntityTotal()
			self.trainTotal = self.lib.getTrainTotal()
			logging.warning('Got train total: {}'.format(self.trainTotal))	
			self.testTotal = self.lib.getTestTotal()
			logging.warning('Got test total: {}'.format(self.testTotal))	
			self.validTotal = self.lib.getValidTotal()
			logging.warning('Got val total: {}'.format(self.validTotal))	
			self.batch_size = int(self.lib.getTrainTotal() / self.nbatches)
			logging.warning('Set batch size: {}'.format(self.batch_size))
			self.batch_seq_size = self.batch_size * (1 + self.negative_ent + self.negative_rel)
			self.batch_h = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = np.int64) # 1d list of 0s of shape (1 + self.negative_ent + self.negative_rel,)
			self.batch_t = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = np.int64)
			self.batch_r = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = np.int64)
			self.batch_y = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = np.float32)			
			self.batch_h_addr = self.batch_h.__array_interface__['data'][0]
			self.batch_t_addr = self.batch_t.__array_interface__['data'][0]
			self.batch_r_addr = self.batch_r.__array_interface__['data'][0]
			self.batch_y_addr = self.batch_y.__array_interface__['data'][0]			
			if self.freeze_train_embeddings:
				self.ent_embedding_initializer = self.set_ent_embedding_initializer(self.embedding_initializer_path)
				self.rel_embedding_initializer = self.set_rel_embedding_initializer(self.embedding_initializer_path)
				logging.warning('Initialized embeddings from: {}'.format(self.embedding_initializer_path))

		if self.test_link_prediction:
			self.init_link_prediction()
		if self.test_triple_classification:
			self.init_triple_classification()

	def set_freeze_train_embeddings(self, freeze_train_embeddings):
		self.freeze_train_embeddings = freeze_train_embeddings

	def set_embedding_initializer_path(self, embedding_initializer_path):
		self.embedding_initializer_path = embedding_initializer_path



	# def set_test_in_path(self, path):
	# 	self.in_path = path

	def get_ent_total(self):
		return self.entTotal

	def get_rel_total(self):
		return self.relTotal

	def set_lmbda(self, lmbda):
		self.lmbda = lmbda

	def set_optimizer(self, optimizer):
		self.optimizer = optimizer

	def set_opt_method(self, method):
		self.opt_method = method

	def set_test_link_prediction(self, flag):
		self.test_link_prediction = flag

	def set_test_triple_classification(self, flag):
		self.test_triple_classification = flag

	def set_log_on(self, flag):
		self.log_on = flag

	def set_alpha(self, alpha):
		self.alpha = alpha

	def set_in_path(self, path):
		self.in_path = path

	def set_out_files(self, path):
		self.out_path = path

	def set_bern(self, bern):
		self.bern = bern

	def set_dimension(self, dim):
		self.hidden_size = dim

	def set_train_times(self, times):
		self.train_times = times

	def set_nbatches(self, nbatches):
		self.nbatches = nbatches

	def set_margin(self, margin):
		self.margin = margin

	def set_work_threads(self, threads):
		self.workThreads = threads

	def set_ent_neg_rate(self, rate):
		self.negative_ent = rate

	def set_rel_neg_rate(self, rate):
		self.negative_rel = rate

	def set_import_files(self, path):
		self.importName = path

	def set_export_files(self, path, steps = 0):
		self.exportName = path
		self.export_steps = steps

	def set_export_steps(self, steps):
		self.export_steps = steps

	def set_early_stopping(self, early_stopping):
		self.early_stopping = early_stopping

	# call C function for sampling
	def sampling(self):
		self.lib.sampling(self.batch_h_addr, self.batch_t_addr, self.batch_r_addr, self.batch_y_addr, self.batch_size, self.negative_ent, self.negative_rel)

	# save model
	def save_tensorflow(self):
		with self.graph.as_default():
			with self.sess.as_default():
				self.saver.save(self.sess, self.exportName)
	# restore model
	def restore_tensorflow(self):
		with self.graph.as_default():
			with self.sess.as_default():
				self.saver.restore(self.sess, self.importName)


	def export_variables(self, path = None):
		with self.graph.as_default():
			with self.sess.as_default():
				if path == None:
					self.saver.save(self.sess, self.exportName)
				else:
					self.saver.save(self.sess, path)

	def import_variables(self, path = None):
		with self.graph.as_default():
			with self.sess.as_default():
				if path == None:
					self.saver.restore(self.sess, self.importName)
				else:
					self.saver.restore(self.sess, path)

	def get_parameter_lists(self):
		return self.trainModel.parameter_lists

	def get_parameters_by_name(self, var_name):
		with self.graph.as_default():
			with self.sess.as_default():
				if var_name in self.trainModel.parameter_lists:
					return self.sess.run(self.trainModel.parameter_lists[var_name])
				else:
					return None

	def get_parameters(self, mode = "numpy"):
		res = {}
		lists = self.get_parameter_lists()
		for var_name in lists:
			if mode == "numpy":
				res[var_name] = self.get_parameters_by_name(var_name)
			else:
				res[var_name] = self.get_parameters_by_name(var_name).tolist()
		return res

	def save_parameters(self, path = None):
		if path == None:
			path = self.out_path

		embedding_dict = self.get_parameters("list")	
		# OpenKE saves embeddings for ComplEx in a 4 key dict, we 
		# want to conform to our own format, so we will reshape this dictionary before saving:
		if "ent_re_embeddings" in embedding_dict.keys():
			embedding_dict["ent_embeddings"] = [re+im for (re,im) in\
			 zip(embedding_dict["ent_re_embeddings"], embedding_dict["ent_im_embeddings"])]
			embedding_dict["rel_embeddings"] = [re+im for (re,im) in\
			 zip(embedding_dict["rel_re_embeddings"], embedding_dict["rel_im_embeddings"])]			 

			del embedding_dict['ent_re_embeddings']
			del embedding_dict['ent_im_embeddings']			
			del embedding_dict['rel_re_embeddings']
			del embedding_dict['rel_im_embeddings']					


		dir_name = os.path.dirname(path)	
		os.makedirs(dir_name, exist_ok=True)	
		f = open(path, "w")
		f.write(json.dumps(embedding_dict))
		f.close()

	def set_parameters_by_name(self, var_name, tensor):
		with self.graph.as_default():
			with self.sess.as_default():
				if var_name in self.trainModel.parameter_lists:
					self.trainModel.parameter_lists[var_name].assign(tensor).eval()

	def set_parameters(self, lists):
		for i in lists:
			self.set_parameters_by_name(i, lists[i])

	def set_model(self, model):
		self.model = model
		self.graph = tf.Graph()
		with self.graph.as_default():
			self.sess = tf.Session()
			with self.sess.as_default():
				initializer = tf.contrib.layers.xavier_initializer(uniform = True)
				with tf.variable_scope("model", reuse=None, initializer = initializer):
					self.trainModel = self.model(config = self)
					if self.optimizer != None:
						pass
					elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
						self.optimizer = tf.train.AdagradOptimizer(learning_rate = self.alpha, initial_accumulator_value=0.1)
					elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
						self.optimizer = tf.train.AdadeltaOptimizer(self.alpha)
					elif self.opt_method == "Adam" or self.opt_method == "adam":
						self.optimizer = tf.train.AdamOptimizer(self.alpha)
					else:
						self.optimizer = tf.train.GradientDescentOptimizer(self.alpha)
					grads_and_vars = self.optimizer.compute_gradients(self.trainModel.loss)
					if self.freeze_train_embeddings:
						# The below based vaguely on this SO question:
						# https://stackoverflow.com/questions/35803425/update-only-part-of-the-word-embedding-matrix-in-tensorflow
						# Goal: Take indices from indexedSlices object, which encodes which values are involved in the forward pass
						# and are trainable, mask any gradients which apply to values we don;t want to change (i.e. those 
						# created during initial training) and apply masked gradient update (impacting only those embeddings created 
						# during test/val)

						# Get the grads and vars for each embedding						
						if len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)) == 2: # TransE, DistMult
							ent_grads_and_var = self.optimizer.compute_gradients(self.trainModel.loss, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[0]) # Ent embeddings						
							rel_grads_and_var = self.optimizer.compute_gradients(self.trainModel.loss, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[1]) # Rel embeddings	

							# Extract the gradients for entities and relationships
							ent_grads = ent_grads_and_var[0][0]
							rel_grads = rel_grads_and_var[0][0]

							# Create a mask of 1s and 0s for whether or not the gradient corresponds to a value in the new data or not
							# That is, if the index of the gradient (and by extension the entity or relation) is greater than or equal to the
							# length of the training embedding (self.xxx_embedding_length) then we set it to 1, else 0. If the value is 0, then 
							# the gradient will not be propogated
							ent_mask = tf.cast(ent_grads.indices >= tf.constant(self.ent_embedding_length, dtype=tf.int64), tf.float32)
							rel_mask = tf.cast(rel_grads.indices >= tf.constant(self.rel_embedding_length, dtype=tf.int64), tf.float32)

							# Mask the gradients using the above derived mask
							# The mask has to be reshaped to conform to the shape of the gradients.values
							ent_grads_masked = tf.reshape(ent_mask, [tf.shape(ent_mask)[0],1]) * ent_grads.values
							rel_grads_masked = tf.reshape(rel_mask, [tf.shape(rel_mask)[0],1]) * rel_grads.values

							# Reconstruct the grad and var tuple for ent and rel
							# This reconstruction is required because tuples are immutable
							# We should probbaly find a more principled way of doing this without relying on indices that have no names. makes it all a bit opaque
							ent_indexedSlices = tf.IndexedSlices(values=ent_grads_masked, indices=grads_and_vars[0][0].indices, dense_shape=grads_and_vars[0][0].dense_shape)
							ent_variable = grads_and_vars[0][1]
							ent_grads_and_var_tuple = (ent_indexedSlices,ent_variable)

							rel_indexedSlices = tf.IndexedSlices(values=rel_grads_masked, indices=grads_and_vars[1][0].indices, dense_shape=grads_and_vars[1][0].dense_shape)
							rel_variable = grads_and_vars[1][1]
							rel_grads_and_var_tuple = (rel_indexedSlices,rel_variable)

							# swap in the newly reconstructed embedding grad+var tuples
							grads_and_vars[0] = ent_grads_and_var_tuple
							grads_and_vars[1] = rel_grads_and_var_tuple

							self.train_op = self.optimizer.apply_gradients(grads_and_vars)

						# Hack together ComplEx for evaluation purposes, fix this section up later
						# If the performance is good
						if len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)) == 4: # ComplEx , find a better way to know what model we have chosen
							
							ent1_grads_and_var = self.optimizer.compute_gradients(self.trainModel.loss,\
							 tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[0]) # Ent real embeddings						
							ent2_grads_and_var = self.optimizer.compute_gradients(self.trainModel.loss,\
							 tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[1]) # Ent im embeddings							
							rel1_grads_and_var = self.optimizer.compute_gradients(self.trainModel.loss,\
							 tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[2]) # Rel real embeddings	
							rel2_grads_and_var = self.optimizer.compute_gradients(self.trainModel.loss,\
							 tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[3]) # Rel im embeddings

							# Extract the gradients for entities and relationships
							ent1_grads = ent1_grads_and_var[0][0]
							ent2_grads = ent2_grads_and_var[0][0]
							rel1_grads = rel1_grads_and_var[0][0]
							rel2_grads = rel2_grads_and_var[0][0]

							# Create a mask of 1s and 0s for whether or not the gradient corresponds to a value in the new data or not
							# That is, if the index of the gradient (and by extension the entity or relation) is greater than or equal to the
							# length of the training embedding (self.xxx_embedding_length) then we set it to 1, else 0. If the value is 0, then 
							# the gradient will not be propogated
							ent1_mask = tf.cast(ent1_grads.indices >= tf.constant(self.ent_embedding_length, dtype=tf.int64), tf.float32)
							ent2_mask = tf.cast(ent2_grads.indices >= tf.constant(self.ent_embedding_length, dtype=tf.int64), tf.float32)
							rel1_mask = tf.cast(rel1_grads.indices >= tf.constant(self.rel_embedding_length, dtype=tf.int64), tf.float32)
							rel2_mask = tf.cast(rel2_grads.indices >= tf.constant(self.rel_embedding_length, dtype=tf.int64), tf.float32)

							# Mask the gradients using the above derived mask
							# The mask has to be reshaped to conform to the shape of the gradients.values
							ent1_grads_masked = tf.reshape(ent1_mask, [tf.shape(ent1_mask)[0],1]) * ent1_grads.values
							ent2_grads_masked = tf.reshape(ent2_mask, [tf.shape(ent2_mask)[0],1]) * ent2_grads.values
							rel1_grads_masked = tf.reshape(rel1_mask, [tf.shape(rel1_mask)[0],1]) * rel1_grads.values
							rel2_grads_masked = tf.reshape(rel2_mask, [tf.shape(rel2_mask)[0],1]) * rel2_grads.values
							
							# Reconstruct the grad and var tuple for ent and rel
							# This reconstruction is required because tuples are immutable
							# We should probbaly find a more principled way of doing this without relying on indices that have no names. makes it all a bit opaque
							ent1_indexedSlices = tf.IndexedSlices(values=ent1_grads_masked, indices=grads_and_vars[0][0].indices, dense_shape=grads_and_vars[0][0].dense_shape)
							ent1_variable = grads_and_vars[0][1]
							ent1_grads_and_var_tuple = (ent1_indexedSlices,ent1_variable)

							ent2_indexedSlices = tf.IndexedSlices(values=ent2_grads_masked, indices=grads_and_vars[1][0].indices, dense_shape=grads_and_vars[1][0].dense_shape)
							ent2_variable = grads_and_vars[1][1]
							ent2_grads_and_var_tuple = (ent2_indexedSlices,ent2_variable)							

							rel1_indexedSlices = tf.IndexedSlices(values=rel1_grads_masked, indices=grads_and_vars[2][0].indices, dense_shape=grads_and_vars[2][0].dense_shape)
							rel1_variable = grads_and_vars[2][1]
							rel1_grads_and_var_tuple = (rel1_indexedSlices,rel1_variable)

							rel2_indexedSlices = tf.IndexedSlices(values=rel2_grads_masked, indices=grads_and_vars[3][0].indices, dense_shape=grads_and_vars[3][0].dense_shape)
							rel2_variable = grads_and_vars[3][1]
							rel2_grads_and_var_tuple = (rel2_indexedSlices,rel2_variable)														

							# swap in the newly reconstructed embedding grad+var tuples
							grads_and_vars[0] = ent1_grads_and_var_tuple
							grads_and_vars[1] = ent2_grads_and_var_tuple
							grads_and_vars[2] = rel1_grads_and_var_tuple
							grads_and_vars[3] = rel2_grads_and_var_tuple

							# Pass a few things for debugging
							self.ent1_variable_before = ent1_variable
							self.ent1_grads = ent1_grads
							self.ent1_mask = ent1_mask
							self.ent1_grads_masked = ent1_grads_masked

							logging.debug("ent1_var: {}".format(ent1_variable))

							self.train_op = self.optimizer.apply_gradients(grads_and_vars)	
							self.ent1_variable_after = ent1_variable									
						else:
							logging.warning('Models currently supported: TransE_freeze, DistMult_freeze, ComplEx_freeze')	

					else: 						
						self.train_op = self.optimizer.apply_gradients(grads_and_vars)
				self.saver = tf.train.Saver()
				self.sess.run(tf.initialize_all_variables())

	def train_step(self, batch_h, batch_t, batch_r, batch_y):
		feed_dict = {
			self.trainModel.batch_h: batch_h,
			self.trainModel.batch_t: batch_t,
			self.trainModel.batch_r: batch_r,
			self.trainModel.batch_y: batch_y
		}
		# _, loss,ent1_grads,ent1_mask,ent1_grads_masked,ent1_variable_before,ent1_variable_after, ld_res,ld_loss_func, ld_y  = self.sess.run([self.train_op, self.trainModel.loss, self.ent1_grads, self.ent1_mask, self.ent1_grads_masked, self.ent1_variable_before, self.ent1_variable_after,self.trainModel.ld_res, self.trainModel.ld_loss_func, self.trainModel.ld_y], feed_dict)	
		_, loss, pos_ent_mean_magnitude, pos_ent_min, pos_ent_max, pos_ent_sd = self.sess.run([self.train_op, self.trainModel.loss, self.trainModel.pos_ent_mean_magnitude, self.trainModel.pos_ent_min, self.trainModel.pos_ent_max, self.trainModel.pos_ent_sd], feed_dict)	
		# if len(np.where(ent1_grads.indices == 4627)[0] > 0):
		# 	check_this_one = np.where(ent1_grads.indices == 4627)[0][0]
		# 	l1.debug("ent1_grads.values.shape : {}".format(ent1_grads.values.shape))
		# 	l1.debug("ent1_grads.values ({}) : {}".format(check_this_one, ent1_grads.values[check_this_one][0:10]))
		# 	l1.debug(sum([sum(abs(vect)) for vect in ent1_grads.values]))
		# 	l1.debug("ent1_grads.indices : {}".format(ent1_grads.indices[check_this_one]))
		# 	l1.debug("max(ent1_grads.indices) : {}".format(max(ent1_grads.indices)))
		# 	l1.debug("min(ent1_grads.indices) : {}".format(min(ent1_grads.indices)))		
		# 	l1.debug("ent1_mask.shape : {}".format(ent1_mask.shape))
		# 	l1.debug("ent1_mask : {}".format(ent1_mask))
		# 	l1.debug("sum(ent1_mask) : {}".format(sum(ent1_mask)))
		# 	l1.debug("ent1_grads_masked.shape : {}".format(ent1_grads_masked.shape))
		# 	l1.debug(sum([sum(abs(vect)) for vect in ent1_grads_masked]))				
		# 	l1.debug("ent1_grads_masked : {}".format(ent1_grads_masked[check_this_one][0:10]))
		# 	l1.debug("ent1_variable_before : {}".format(ent1_variable_before[check_this_one][0:10]))
		# 	l1.debug("ent1_variable_after : {}".format(ent1_variable_after[check_this_one][0:10]))	
		# 	l1.debug("ent1_variable_before == ent1_variable_after: {}".format(ent1_variable_before == ent1_variable_after))	
		
		# l1.debug("res = {}".format(", ".join([str(x) for x in ld_res])))
		# l1.debug("y = {}".format(", ".join([str(x) for x in ld_y])))		
		# l1.debug("loss = {}".format(ld_loss_func))				
		# l1.debug("------")
		self.pos_ent_mean_magnitude = pos_ent_mean_magnitude
		self.pos_ent_min = pos_ent_min
		self.pos_ent_max = pos_ent_max
		self.pos_ent_sd = pos_ent_sd				
		return loss

	def test_step(self, test_h, test_t, test_r):
		feed_dict = {
			self.trainModel.predict_h: test_h,
			self.trainModel.predict_t: test_t,
			self.trainModel.predict_r: test_r,
		}
		predict = self.sess.run(self.trainModel.predict, feed_dict)
		return predict

	def run(self):
		with self.graph.as_default():
			with self.sess.as_default():
				if self.importName != None:
					self.restore_tensorflow()
				if self.early_stopping is not None:
					patience, min_delta = self.early_stopping
					best_loss = np.finfo('float32').max
					wait_steps = 0
				for times in range(self.train_times):
					t_init = time.time()
					loss = 0.0
					pos_ent_mean_magnitude = 0.0
					pos_ent_min = 0.0
					pos_ent_max = 0.0
					pos_ent_sd = 0.0			
					for batch in range(self.nbatches):
						self.sampling()
						loss += self.train_step(self.batch_h, self.batch_t, self.batch_r, self.batch_y)
						pos_ent_mean_magnitude += self.pos_ent_mean_magnitude
						pos_ent_min += self.pos_ent_min
						pos_ent_max += self.pos_ent_max
						pos_ent_sd += self.pos_ent_sd
					if self.log_on:						
						t_end = time.time()
						pos_ent_mean_magnitude /= (self.nbatches)
						pos_ent_min /= (self.nbatches)
						pos_ent_max /= (self.nbatches)
						pos_ent_sd /= (self.nbatches)
						print('Epoch: {}, loss: {}, time: {}, mag: {}, sd: {}, [{}, {}]'.format(times,\
							round(loss, 2),\
							round(t_end - t_init, 0),\
							round(pos_ent_mean_magnitude, 3),\
							round(pos_ent_sd, 3),\
							round(pos_ent_min, 3),\
							round(pos_ent_max, 3)))
					# if self.exportName != None and (self.export_steps!=0 and times % self.export_steps == 0):
					# 	self.save_tensorflow()
					# print("times: {} , export_steps: {}, div: , out_path:{}".format(times, self.export_steps,  self.out_path))
					if times > 0:
						if self.out_path != None and (self.export_steps!=0 and times % self.export_steps == 0):
							self.save_parameters(self.out_path + "_{}".format(times))						
					if self.early_stopping is not None:
						if loss + min_delta < best_loss:
							best_loss = loss
							wait_steps = 0
						elif wait_steps < patience:
							wait_steps += 1
						else:
							print('Early stopping. Losses have not been improved enough in {} times'.format(patience))
							break
				if self.exportName != None:
					self.save_tensorflow()
				if self.out_path != None:
					self.save_parameters(self.out_path)

	def test(self):
		with self.graph.as_default():
			with self.sess.as_default():
				if self.importName != None:
					self.restore_tensorflow()
				if self.test_link_prediction:
					total = self.lib.getTestTotal()
					for times in range(total):
						self.lib.getHeadBatch(self.test_h_addr, self.test_t_addr, self.test_r_addr)
						res = self.test_step(self.test_h, self.test_t, self.test_r)
						self.lib.testHead(res.__array_interface__['data'][0])

						self.lib.getTailBatch(self.test_h_addr, self.test_t_addr, self.test_r_addr)
						res = self.test_step(self.test_h, self.test_t, self.test_r)
						self.lib.testTail(res.__array_interface__['data'][0])
						if self.log_on:
							print(times)
					self.lib.test_link_prediction()
				if self.test_triple_classification:
					self.lib.getValidBatch(self.valid_pos_h_addr, self.valid_pos_t_addr, self.valid_pos_r_addr, self.valid_neg_h_addr, self.valid_neg_t_addr, self.valid_neg_r_addr)
					res_pos = self.test_step(self.valid_pos_h, self.valid_pos_t, self.valid_pos_r)
					res_neg = self.test_step(self.valid_neg_h, self.valid_neg_t, self.valid_neg_r)
					self.lib.getBestThreshold(self.relThresh_addr, res_pos.__array_interface__['data'][0], res_neg.__array_interface__['data'][0])

					self.lib.getTestBatch(self.test_pos_h_addr, self.test_pos_t_addr, self.test_pos_r_addr, self.test_neg_h_addr, self.test_neg_t_addr, self.test_neg_r_addr)

					res_pos = self.test_step(self.test_pos_h, self.test_pos_t, self.test_pos_r)
					res_neg = self.test_step(self.test_neg_h, self.test_neg_t, self.test_neg_r)
					self.lib.test_triple_classification(self.relThresh_addr, res_pos.__array_interface__['data'][0], res_neg.__array_interface__['data'][0])

	def predict_head_entity(self, t, r, k):
		r'''This mothod predicts the top k head entities given tail entity and relation.
		
		Args: 
			t (int): tail entity id
			r (int): relation id
			k (int): top k head entities
		
		Returns:
			list: k possible head entity ids 	  	
		'''
		self.init_link_prediction()
		if self.importName != None:
			self.restore_tensorflow()
		test_h = np.array(range(self.entTotal))
		test_r = np.array([r] * self.entTotal)
		test_t = np.array([t] * self.entTotal)
		res = self.test_step(test_h, test_t, test_r).reshape(-1).argsort()[:k]
		print(res)
		return res

	def predict_tail_entity(self, h, r, k):
		r'''This mothod predicts the top k tail entities given head entity and relation.
		
		Args: 
			h (int): head entity id
			r (int): relation id
			k (int): top k tail entities
		
		Returns:
			list: k possible tail entity ids 	  	
		'''
		self.init_link_prediction()
		if self.importName != None:
			self.restore_tensorflow()
		test_h = np.array([h] * self.entTotal)
		test_r = np.array([r] * self.entTotal)
		test_t = np.array(range(self.entTotal))
		res = self.test_step(test_h, test_t, test_r).reshape(-1).argsort()[:k]
		print(res)
		return res

	def predict_relation(self, h, t, k):
		r'''This methods predict the relation id given head entity and tail entity.
		
		Args:
			h (int): head entity id
			t (int): tail entity id
			k (int): top k relations
		
		Returns:
			list: k possible relation ids
		'''
		self.init_link_prediction()
		if self.importName != None:
			self.restore_tensorflow()
		test_h = np.array([h] * self.relTotal)
		test_r = np.array(range(self.relTotal))
		test_t = np.array([t] * self.relTotal)
		res = self.test_step(test_h, test_t, test_r).reshape(-1).argsort()[:k]
		print(res)
		return res

	def predict_triple(self, h, t, r, thresh = None):
		r'''This method tells you whether the given triple (h, t, r) is correct of wrong
	
		Args:
			h (int): head entity id
			t (int): tail entity id
			r (int): relation id
			thresh (fload): threshold for the triple
		'''
		self.init_triple_classification()
		if self.importName != None:
			self.restore_tensorflow()
		res = self.test_step(np.array([h]), np.array([t]), np.array([r]))
		if thresh != None:
			if res < thresh:
				print("triple (%d,%d,%d) is correct" % (h, t, r))
			else:
				print("triple (%d,%d,%d) is wrong" % (h, t, r))
			return
		self.lib.getValidBatch(self.valid_pos_h_addr, self.valid_pos_t_addr, self.valid_pos_r_addr, self.valid_neg_h_addr, self.valid_neg_t_addr, self.valid_neg_r_addr)
		res_pos = self.test_step(self.valid_pos_h, self.valid_pos_t, self.valid_pos_r)
		res_neg = self.test_step(self.valid_neg_h, self.valid_neg_t, self.valid_neg_r)
		self.lib.getBestThreshold(self.relThresh_addr, res_pos.__array_interface__['data'][0], res_neg.__array_interface__['data'][0])
		if res < self.relThresh[r]:
			print("triple (%d,%d,%d) is correct" % (h, t, r))
		else:
			print("triple (%d,%d,%d) is wrong" % (h, t, r))


	def set_ent_embedding_initializer(self, embedding_path):
		# This function needs to take a path for the embedding file produced by the initial training
		# And a list of the entities in the new (val or test, or oov or whatever) data (that are not in the old data?)
		# and create a new matrix that is composed of the training embeddings with random values initialized for the 
		# new embeddings append to it		
		# We also need to get the indices for updates

		# Need to fix this - right now it leaves the file open. Just use a with statement 
		try:
			embs = open(embedding_path, 'r')
    	# Store configuration file values
		except FileNotFoundError:
			raise Exception('Entity embedding file not found: {}'.format(embedding_path))

		embedding_dict = json.loads(embs.read())	

		# If the embeddings were produced by OpenKE, we will have to combine them
		# if "ent_re_embeddings" in embedding_dict.keys():
		# 	embedding_dict["ent_embeddings"] = [re+im for (re,im) in\
		# 	 zip(embedding_dict["ent_re_embeddings"], embedding_dict["ent_im_embeddings"])]
		# 	del embedding_dict['ent_re_embeddings']
		# 	del embedding_dict['ent_im_embeddings']				

		ent_embedding = embedding_dict["ent_embeddings"]
		self.ent_embedding_length = len(ent_embedding)

		# Compare to length of the training embedding to the total number of entities to see how many 
		# new rows we need to append to the embdding initializer
		if self.entTotal > self.ent_embedding_length:
			print("New entities found:")
			print("-- Total Entities in embedding file: {}".format(self.ent_embedding_length))			
			print("-- Total Entities in data: {} ".format(self.entTotal))
			
			required_new_vectors = self.entTotal - self.ent_embedding_length

			# Perform Xavier initialization for the new embeddings:  sqrt(6 / (fan_in + fan_out))
			# Not clear whether we should initialize to the same fan size as the original embeddings
			# (i.e. self.ent_embedding_length)
			# Or the fan size for the original + new embeddings (i.e. self.entTotal) 
			ent_bound = np.sqrt(6 / (self.entTotal + self.hidden_size)) 
			# new_ent_embedding = [np.random.uniform(-ent_bound,ent_bound,self.hidden_size).tolist()\
			# 	for x in range(required_new_vectors)]

			# PyTorch-BigGraph initalizes with draws from a standard normal, so we will too
			# Updated: We initialize the sd to be 0.4 to be in accordance with empirical embedding sd
			new_ent_embedding = [np.random.normal(loc=0.0, scale=0.4, size=self.hidden_size).tolist()\
				for x in range(required_new_vectors)]
			
			ent_embedding = ent_embedding + new_ent_embedding
			# self.ent_update_slices = [self.ent_embedding_length - idx for idx in range(required_new_vectors)]

		return ent_embedding

	def set_rel_embedding_initializer(self, embedding_path):
		# TODO: Combine this and the set_ent_embedding_initializer, lots of duplicated code
		try:
			embs = open(embedding_path, 'r')
    	# Store configuration file values
		except FileNotFoundError:
			raise Exception('Relation embedding file not found: {}'.format(embedding_path))

		embedding_dict = json.loads(embs.read())	

		# If the embeddings were produced by OpenKE, we will have to combine them
		# if "rel_re_embeddings" in embedding_dict.keys():			
		# 	embedding_dict["rel_embeddings"] = [re+im for (re,im) in\
		# 	 zip(embedding_dict["rel_re_embeddings"], embedding_dict["rel_im_embeddings"])]
		
		# 	del embedding_dict['rel_re_embeddings']
		# 	del embedding_dict['rel_im_embeddings']		

		rel_embedding = embedding_dict["rel_embeddings"]
		self.rel_embedding_length = len(rel_embedding)

		if self.relTotal > self.rel_embedding_length :
			print("New relationships found:")
			print("-- Total Relationships in embedding file: {}".format(len(rel_embedding)))			
			print("-- Total Relationships in data: {} ".format(self.relTotal))
			
			required_new_vectors = self.relTotal - self.rel_embedding_length 
			# TODO: Find a good way to initialize the vectors
			# new_rel_embedding = tf.Variable(name="new_rel_embedding",\
			# 			  shape = [self.relTotal - len(rel_embedding), self.hidden_size],\
			# 			  initializer = tf.contrib.layers.xavier_initializer(uniform = False))
			# print(new_rel_embedding.initialized_value())
			rel_bound = np.sqrt(6 / (self.relTotal + self.hidden_size)) # Xavier init:  sqrt(6 / (fan_in + fan_out))
			# new_rel_embedding = [np.random.uniform(-rel_bound, rel_bound, self.hidden_size).tolist()\
			# 	for x in range(required_new_vectors)]
			new_rel_embedding = [np.random.normal(loc=0.0, scale=0.4, size=self.hidden_size).tolist()\
				for x in range(required_new_vectors)]			
			rel_embedding = rel_embedding + new_rel_embedding	
			# self.rel_update_slices = [self.rel_embedding_length  - idx for idx in range(required_new_vectors)]

		return rel_embedding

