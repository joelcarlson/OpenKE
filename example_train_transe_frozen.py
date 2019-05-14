import config
import models
import tensorflow as tf
import numpy as np
import os



"""
Method:

Run the normal transe example (example_train_transe.py)
Write the embeddings as a file that can be read
use embeddings to initialize embedding layer in TransE_freeze.py

Append random embeddings for new entities and relations
set config freeze_train_embeddings = true
figure out how to update only the embeddings for a certain set of indices
figure out how to make sure that we only see examples using new items to speed convergence
compare new+old embeddings
"""
os.environ['CUDA_VISIBLE_DEVICES']='7'
#Input training files from benchmarks/FB15K/ folder.
con = config.Config()
#True: Input test files from the same folder.
con.set_in_path("./benchmarks/FB15K/")
con.set_test_link_prediction(True)
con.set_test_triple_classification(True)
con.set_work_threads(8)
con.set_train_times(10)
con.set_nbatches(20)
con.set_alpha(0.001)
con.set_margin(1.0)
con.set_bern(0)
con.set_dimension(100)
con.set_ent_neg_rate(1)
con.set_rel_neg_rate(0)
con.set_opt_method("SGD")
con.set_freeze_train_embeddings(True)
con.set_rel_embedding_initializer("new path")
con.set_rel_embedding_initializer("new _ path")


#Models will be exported via tf.Saver() automatically.
con.set_export_files("./res/model.vec.tf", 0)
#Model parameters will be exported to json files automatically.
con.set_out_files("./res/embedding.vec.json")
#Initialize experimental settings.
con.init()
#Set the knowledge embedding model
con.set_model(models.TransE_freeze)
#Train the model.
con.run()
#To test models after training needs "set_test_flag(True)".
# con.test()
con.predict_head_entity(152, 9, 5)
con.predict_tail_entity(151, 9, 5)
con.predict_relation(151, 152, 5)
con.predict_triple(151, 152, 9)
con.predict_triple(151, 152, 8)
