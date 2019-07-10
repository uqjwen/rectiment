import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

fc_layer = tf.contrib.layers.fully_connected

class Model(object):

	def __init__(self,
				 base_model,
				 vocab_size,
				 num_class,
				 num_user,
				 num_item,
				 user_mean_freq,
				 item_mean_freq,
				 embedding_size,
				 u_neighbor=None,
				 p_neighbor=None,
				 batch_size=None):
		
		self.vocab_size = vocab_size
		self.num_user = num_user
		self.num_item = num_item
		self.user_mean_freq = user_mean_freq
		self.item_mean_freq = item_mean_freq
		self.num_class = num_class
		self.embedding_size = embedding_size
		self.batch_size = batch_size
		self.base_model = base_model
		self.u_neighbor = u_neighbor
		self.p_neighbor = p_neighbor
		
		# PLACEHOLDERS
		self.inputs = tf.placeholder(tf.int32, shape=[self.batch_size, None])
		self.input_len = tf.placeholder(tf.int32, shape=[self.batch_size])

		# self.polarity = tf.placeholder(tf.float32, shape = [self.batch_size, None])
		
		self.users = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
		self.user_counts = tf.placeholder(tf.float32, shape=[self.batch_size, 1])
		
		self.items = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
		self.item_counts = tf.placeholder(tf.float32, shape=[self.batch_size, 1])

		self.targets = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_class])
		
		self.keep_prob = tf.placeholder(tf.float32)
		
		# EMBEDDINGS
		self.embedding = tf.get_variable("embedding", [self.vocab_size, self.embedding_size], 
								initializer=xavier_initializer())
		self.user_embedding = tf.get_variable("user_embedding", [self.num_user, self.embedding_size],
								initializer=xavier_initializer())
		self.item_embedding = tf.get_variable("item_embedding", [self.num_item, self.embedding_size],
								initializer=xavier_initializer())

		inputs_embedding = tf.nn.embedding_lookup(self.embedding, self.inputs)

		if base_model == 'cnn':
			encode = self.get_cnn_stack(inputs_embedding)
			h = reduce_max(encode, axis=1)
		elif base_model == 'rnn_cnn':
			encode = self.get_lstm_cnn(inputs_embedding)
			h = reduce_max(encode, axis=1)
		elif base_model == 'att_cnn':
			encode = self.get_cnn_stack(inputs_embedding)
			encode, self.atts = self.get_att_cnn(encode)
			h = encode


		# print("encode shape: ", encode.shape)
		# h = tf.reduce_max(encode, axis=1)

		print("encode reduce_max: ", h.shape)
		W0 = tf.get_variable("W0", [embedding_size, num_class], initializer=xavier_initializer())
		b0 = tf.Variable(tf.constant(0.0, shape=[num_class]))

		scores = tf.nn.xw_plus_b(h, W0, b0)


		ui_scores = self.get_user_item() ## actually it is logits
		# ui_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits = ui_scores, labels = self.targets)
		# ui_loss = tf.reduce_mean(ui_loss)

		scores += ui_scores
		losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.targets)
		self.loss = tf.reduce_mean(losses)#+ui_loss








		
		
		self.predictions = tf.argmax(scores, 1)
		self.updates = tf.train.AdadeltaOptimizer(1,0.95,1e-6).minimize(self.loss)		
		self.updates = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(self.loss)

		# self.hcsc_scores, self.hcsc_loss = self.get_hcsc_loss(encode, embedding_size)
		# self.predictions = tf.argmax(self.hcsc_scores, 1)
		# # self.updates = tf.train.AdadeltaOptimizer(1,0.95,1e-6).minimize(self.loss)		
		# self.updates = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(self.hcsc_loss)




		self.count = tf.cast(tf.equal(self.predictions, tf.argmax(self.targets, 1)), 'float')










	def get_hcsc_loss(self, encode, embedding_size):
		W0 = tf.get_variable("W1", [embedding_size, self.num_class], initializer=xavier_initializer())
		b0 = tf.Variable(tf.constant(0.0, shape=[self.num_class]))
		add_loss = 0
		
		# ATTENTION POOLING
		
		self.user_embed = tf.nn.embedding_lookup(self.user_embedding, self.users)
		self.prod_embed = tf.nn.embedding_lookup(self.item_embedding, self.items)
		
		# 1. SPECIFIC VECTORS
		vu_spec, self.a_ud = self.reduce_spec(encode, self.user_embed, 1, embedding_size, "user")
		vp_spec, self.a_pd = self.reduce_spec(encode, self.prod_embed, 1, embedding_size, "prod")
		
		# 2. SHARED VECTORS			  
		mean = tf.reduce_mean(encode, 1)
		
		user_mean = fc_layer(mean, embedding_size, activation_fn=None)
		vu_share_weights = tf.matmul(user_mean, tf.transpose(self.user_embedding, [1,0]))
		vu_share_weights = tf.expand_dims(tf.nn.softmax(vu_share_weights), -1)
		vu_share = tf.multiply(vu_share_weights, self.user_embedding)
		vu_share = tf.expand_dims(tf.reduce_sum(vu_share, 1), -2)
		vu_share, self.a_us = self.reduce_spec(encode, vu_share, 1, embedding_size, "user1")
		
		prod_mean = fc_layer(mean, embedding_size, activation_fn=None)
		vp_share_weights = tf.matmul(prod_mean, tf.transpose(self.user_embedding, [1,0]))
		vp_share_weights = tf.expand_dims(tf.nn.softmax(vp_share_weights), -1)
		vp_share = tf.multiply(vp_share_weights, self.user_embedding)
		vp_share = tf.expand_dims(tf.reduce_sum(vp_share, 1), -2)
		vp_share, self.a_ps = self.reduce_spec(encode, vp_share, 1, embedding_size, "prod1")
		
		# 3. GATE
		freq_u_norm = self.user_counts / self.user_mean_freq
		self.lambda_u = tf.Variable(tf.constant(1.0, shape=[1, embedding_size]))
		self.LU = tf.reduce_mean(self.lambda_u, -1)
		self.k_u = tf.Variable(tf.constant(1.0, shape=[1, embedding_size]))
		self.KU = tf.reduce_mean(self.k_u, -1)
		gate_u = 1 - tf.exp(-tf.pow(freq_u_norm / tf.nn.relu(self.lambda_u), tf.nn.relu(self.k_u)))
		self.gu = tf.reduce_mean(gate_u, -1)
		vu = gate_u * vu_spec + (1-gate_u) * vu_share
		
		freq_p_norm = self.item_counts / self.item_mean_freq # batch, 1
		self.lambda_p = tf.Variable(tf.constant(1.0, shape=[1, embedding_size]))
		self.LP = tf.reduce_mean(self.lambda_p, -1)
		self.k_p = tf.Variable(tf.constant(1.0, shape=[1, embedding_size]))
		self.KP = tf.reduce_mean(self.k_p, -1)
		gate_p = 1 - tf.exp(-tf.pow(freq_p_norm / tf.nn.relu(self.lambda_p), tf.nn.relu(self.k_p)))
		self.gp = tf.reduce_mean(gate_p, -1)
		vp = gate_p * vp_spec + (1-gate_p) * vp_share
		
		Wg = tf.get_variable("Wg", [embedding_size*2, embedding_size], initializer=xavier_initializer())
		bg = tf.Variable(tf.constant(0.0, shape=[embedding_size]))
		h0 = tf.concat([vu, vp], -1)
		gate = tf.sigmoid(tf.nn.xw_plus_b(h0, Wg, bg))
		self.gup = tf.reduce_mean(gate, -1)
		h = gate * vu + (1-gate) * vp

		# h = tf.reduce_max(encode, axis=1)
		print("encode shape: ", encode.shape, h.shape)

		
		scores_u_spec = tf.nn.xw_plus_b(vu_spec, W0, b0)
		loss_u_spec = tf.nn.softmax_cross_entropy_with_logits(logits=scores_u_spec, labels=self.targets)
		add_loss += tf.reduce_mean(loss_u_spec)
		
		scores_p_spec = tf.nn.xw_plus_b(vp_spec, W0, b0)
		loss_p_spec = tf.nn.softmax_cross_entropy_with_logits(logits=scores_p_spec, labels=self.targets)
		add_loss += tf.reduce_mean(loss_p_spec)
		
		scores_u_share = tf.nn.xw_plus_b(vu_share, W0, b0)
		loss_u_share = tf.nn.softmax_cross_entropy_with_logits(logits=scores_u_share, labels=self.targets)
		add_loss += tf.reduce_mean(loss_u_share)
		
		scores_p_share = tf.nn.xw_plus_b(vp_share, W0, b0)
		loss_p_share = tf.nn.softmax_cross_entropy_with_logits(logits=scores_p_share, labels=self.targets)
		add_loss += tf.reduce_mean(loss_p_share)
		
		scores_u = tf.nn.xw_plus_b(vu, W0, b0)
		loss_u = tf.nn.softmax_cross_entropy_with_logits(logits=scores_u, labels=self.targets)
		add_loss += tf.reduce_mean(loss_u)
		
		scores_p = tf.nn.xw_plus_b(vp, W0, b0)
		loss_p = tf.nn.softmax_cross_entropy_with_logits(logits=scores_p, labels=self.targets)
		add_loss += tf.reduce_mean(loss_p)
		
		hcsc_scores = tf.nn.xw_plus_b(h, W0, b0)
		
		losses = tf.nn.softmax_cross_entropy_with_logits(logits=hcsc_scores, labels=self.targets)
		hcsc_loss = tf.reduce_mean(losses) + add_loss
		
		# predictions = tf.argmax(scores, 1)
		# self.predictions = predictions
		# self.count = tf.cast(tf.equal(predictions, tf.argmax(self.targets, 1)), 'float')
		
		# optimizer = tf.train.AdadeltaOptimizer(1.0, 0.95, 1e-6)
		# grads_and_vars = optimizer.compute_gradients(self.loss)
		# capped_grads_and_vars = []
		# for gv in grads_and_vars:
		#	 capped_grads_and_vars.append((tf.clip_by_norm(gv[0], clip_norm=3, axes=[0]), gv[1]))
		# self.updates = optimizer.apply_gradients(capped_grads_and_vars)
		

		# self.updates = tf.train.AdadeltaOptimizer(1.0,0.95,1e-6).minimize(self.loss)		
		return hcsc_scores, hcsc_loss


	def reduce_spec(self, x, y, axis, size, name):
		X = tf.get_variable("X" + name, shape=[size, size], initializer=xavier_initializer())
		Y = tf.get_variable("Y" + name, shape=[size, size], initializer=xavier_initializer())
		b = tf.Variable(tf.zeros([size]))
		z = tf.get_variable("z" + name, shape=[size], initializer=xavier_initializer())
		
		sem = tf.tensordot(x, X, 1)
		sem.set_shape(x.shape)
		
		user = tf.tensordot(y, Y, 1)
		
		weights = tf.nn.tanh(sem + user + b)
		
		weights = tf.tensordot(weights, z, 1)
		weights.set_shape(x.shape[:-1])
		
		ret_weights = weights
		weights = tf.nn.softmax(weights)
		weights = tf.expand_dims(weights, -1)
		
		attended = tf.multiply(x, weights)
		attended = tf.reduce_sum(attended, axis)

		return attended, ret_weights


	def get_att_cnn(self, inputs_): #batch_size, maxlen, embed_size
		users = tf.nn.embedding_lookup(self.user_embedding, self.users)
		items = tf.nn.embedding_lookup(self.item_embedding, self.items) #batch_size 1 embed_size

		ui = tf.concat([users, items], axis=-1) # batch_size 1 2*embed_size

		ui_trans = tf.keras.layers.Dense(self.embedding_size)(ui)  #batch_size 1 embed_size

		input_trans = tf.keras.layers.Dense(self.embedding_size)(inputs_) #batch_size maxlen embed_size 


		weights = tf.nn.tanh(ui_trans+input_trans)



		weights = tf.keras.layers.Dense(1)(weights) #batch_size, maxlen, 1
		
		# polarity = tf.expand_dims(self.polarity, -1) 
		# weights = weights+polarity*10


		ret_weights = tf.squeeze(weights, -1)

		weights = tf.nn.softmax(weights, axis=1)
		attended = weights*inputs_ 

		attended = tf.reduce_sum(attended, axis=1)

		return attended, ret_weights





	def get_lstm_cnn(self,inputs_):
		fw_cell = tf.nn.rnn_cell.LSTMCell(self.embedding_size/4)
		bw_cell = tf.nn.rnn_cell.LSTMCell(self.embedding_size/4)
		fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=self.keep_prob)
		bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=self.keep_prob)
		
		outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs_, 
													 sequence_length=self.input_len, dtype=tf.float32)
		encode1 = tf.concat(outputs, 2)

		print('encode1 of shape: ', encode1.shape)
			
			
		kernel_sizes = [3,5,7]
		num_filters = [50,50,50]
			
		outputs = []
		for kernel_size,num_filter in zip(kernel_sizes, num_filters):
			conv = tf.keras.layers.Conv1D(num_filter, kernel_size, padding = 'same')(inputs_)
			hidden = tf.nn.dropout(tf.nn.relu(conv),self.keep_prob)
			outputs.append(hidden)
		encode2 = tf.concat(outputs, axis=-1)
		return tf.concat([encode1, encode2], axis=-1)


	def get_cnn(self,inputs_):
		kernel_sizes = [3,5,7]
		num_filters = [100,100,100]

		outputs = []
		for kernel_size,num_filter in zip(kernel_sizes, num_filters):
			conv = tf.keras.layers.Conv1D(num_filter, kernel_size, padding = 'same')(inputs_)
			hidden = tf.nn.dropout(tf.nn.relu(conv),self.keep_prob)
			outputs.append(hidden)
		encode = tf.concat(outputs, axis=-1)
		return encode

	def get_cnn_stack(self, inputs_):
		conv1 = tf.keras.layers.Conv1D(128, 3, padding = 'same')(inputs_)
		conv2 = tf.keras.layers.Conv1D(128, 5, padding = 'same')(inputs_) #batch_size, maxlen, 128

		hidden1 = tf.concat([conv1,conv2], axis=-1) #batch_size, maxlen, 256
		hidden2 = tf.nn.dropout(tf.nn.relu(hidden1), self.keep_prob)

		conv3 = tf.keras.layers.Conv1D(300, 5, padding = 'same')(hidden2)
		encode = tf.nn.dropout(tf.nn.relu(conv3), self.keep_prob)

		return encode

	def get_user_item(self):
		# print("neighbor matrix of shape: ", self.u_neighbor.shape)
		u_neighbors = tf.nn.embedding_lookup(self.u_neighbor,self.users)# batch_size, 1, 20
		un_embed = tf.nn.embedding_lookup(self.user_embedding, u_neighbors)#batch_size, 1, 20, 300 
		un_embed = tf.squeeze(un_embed,1) #batch_size, 20, 300

		# pool_size = self.u_neighbor.shape[-1]
		# user = tf.keras.layers.MaxPool1D(pool_size)(un_embed) #batch_size, 1, 300

		user = tf.nn.embedding_lookup(self.user_embedding, self.users)
		user,self.u_atts = self.get_att_neighbors(user, un_embed)

		# user = tf.nn.embedding_lookup(self.user_embedding,self.users)



		p_neighbors = tf.nn.embedding_lookup(self.p_neighbor,self.items)# batch_size, 1, 20
		pn_embed = tf.nn.embedding_lookup(self.item_embedding, p_neighbors)#batch_size, 1, 20, 300 
		pn_embed = tf.squeeze(pn_embed,1)
		
		# pool_size = self.p_neighbor.shape[-1]
		# item = tf.keras.layers.MaxPool1D(pool_size)(pn_embed)

		item = tf.nn.embedding_lookup(self.item_embedding, self.items)
		user,self.i_atts = self.get_att_neighbors(item, pn_embed)

		# item = tf.nn.embedding_lookup(self.item_embedding, self.items)


		latent = tf.concat([user,item,user*item], axis=-1)
		num_layers = 2
		for i in range(num_layers):
			units = self.embedding_size//(2**i)
			print("user-item dense layer units: ", units)
			latent = tf.keras.layers.Dense(units, 'relu')(latent)
		latent = tf.keras.layers.Dense(self.num_class)(latent)
		return tf.squeeze(latent, 1)


	def get_att_neighbors(self,embedding,n_embeddings):

		##embedding, batch_size, 1, embed_size
		##n_embedding, batch_size, 20, embed_size
		t_embed = tf.keras.layers.Dense(self.embedding_size)(embedding)
		t_n_embed = tf.keras.layers.Dense(self.embedding_size)(n_embeddings)

		weights = tf.nn.tanh(t_embed + t_n_embed) # batch_size, 20 ,embed

		weights = tf.keras.layers.Dense(1)(weights) #batch_size,20,1

		ret_weights = tf.squeeze(weights, -1)

		atts = tf.nn.softmax(weights, axis=1) #batchsize, 20,1

		attended = atts*n_embeddings
		attended = tf.reduce_sum(attended, axis=1, keep_dims = True) #batchsize, 1, 300 

		return attended, ret_weights


	def step(self,
			 session,
			 inputs,
			 targets,
			 users,
			 items,
			 input_len,
			 user_counts, 
			 item_counts,
			 training=True):
		
		# max_len = np.max([len(x) for x in inputs])
		# pad = 3
		# inputs = self.add_pad(inputs, max_len, pad)
		
		input_feed = {}
		input_feed[self.inputs] = inputs
		input_feed[self.input_len] = input_len
		input_feed[self.targets] = targets
		input_feed[self.users] = users
		input_feed[self.items] = items
		input_feed[self.user_counts] = user_counts
		input_feed[self.item_counts] = item_counts
		# input_feed[self.polarity] = polarity
		
		if training:
			input_feed[self.keep_prob] = 0.5
			output_feed = [self.loss, self.updates]
		else:
			input_feed[self.keep_prob] = 1.0
			output_feed = [self.predictions, self.count, self.atts]
		
		outputs = session.run(output_feed, input_feed)
		
		return outputs
