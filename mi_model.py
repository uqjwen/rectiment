import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

fc_layer = tf.contrib.layers.fully_connected

class Model(object):

	def __init__(self,
				 flags,
				 data_loader,
				 data_size,
				 batch_size=None):
		
		self.vocab_size 	= data_loader.vocab_size
		self.num_user 		= data_loader.num_user
		self.num_item 		= data_loader.num_item
		self.num_class 		= flags.num_class
		self.embedding_size = flags.embed_size
		self.batch_size 	= batch_size
		self.base_model 	= flags.base_model
		self.u_neighbor 	= data_loader.u_neighbors
		self.p_neighbor 	= data_loader.p_neighbors
		self.maxlen 		= flags.maxlen
		
		# PLACEHOLDERS
		self.inputs = tf.placeholder(tf.int32, shape=[self.batch_size, None])
		self.input_len = tf.placeholder(tf.int32, shape=[self.batch_size])


		# self.polarity = tf.placeholder(tf.float32, shape = [self.batch_size, None])
		self.sent 			= tf.placeholder(tf.int32, shape=[self.batch_size])
		self.sent_neighbors = tf.placeholder(tf.int32, shape=[self.batch_size, None])

		
		self.users = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
		
		self.items = tf.placeholder(tf.int32, shape=[self.batch_size, 1])

		self.targets = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_class])
		
		self.keep_prob = tf.placeholder(tf.float32)
		
		# EMBEDDINGS
		self.embedding = tf.get_variable("embedding", [self.vocab_size, self.embedding_size], 
								initializer=xavier_initializer())
		self.user_embedding = tf.get_variable("user_embedding", [self.num_user, self.embedding_size],
								initializer=xavier_initializer())
		self.item_embedding = tf.get_variable("item_embedding", [self.num_item, self.embedding_size],
								initializer=xavier_initializer())


		self.sent_embedding = tf.get_variable('sent_embedding',[data_size, self.embedding_size],
								initializer=xavier_initializer())


		inputs_embedding = tf.nn.embedding_lookup(self.embedding, self.inputs)

		if self.base_model == 'cnn':
			encode = self.get_cnn_stack(inputs_embedding)
			h = reduce_max(encode, axis=1)
		elif self.base_model == 'hcsc':
			encode = self.get_hcsc(inputs_embedding)
			h = encode
			# h = reduce_max(encode, axis=1)
		elif self.base_model == 'att_cnn':
			encode_1 = self.get_cnn_stack(inputs_embedding)
			# encode_2 = tf.reduce_max(encode_1,1)
			h, self.atts = self.get_att_cnn(encode_1)
			# h = tf.concat([encode_2,encode_3],axis=-1)


		# print("encode shape: ", encode.shape)
		# h = tf.reduce_max(encode, axis=1)


		# temp = tf.keras.layers.Dense(embedding_size//4, 'relu')(h)
		# scores = tf.keras.layers.Dense(num_class)(h)



		W0 = tf.get_variable("W0", [self.embedding_size, self.num_class], initializer=xavier_initializer())
		b0 = tf.Variable(tf.constant(0.0, shape=[self.num_class]))
		scores = tf.nn.xw_plus_b(h, W0, b0)

		# h2 = self.get_sent_neighbors(h)

		sent_neighbor_emb = tf.nn.embedding_lookup(self.sent_embedding, self.sent_neighbors)
		h2 = tf.reduce_max(sent_neighbor_emb, 1)
		scores_h2 = tf.nn.xw_plus_b(h2, W0, b0)




		ui_scores = self.get_user_item() ## actually it is logits
		# ui_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits = ui_scores, labels = self.targets)
		# ui_loss = tf.reduce_mean(ui_loss)
		if flags.ui_variant:
			scores += ui_scores
		if flags.co_variant:
			scores += scores_h2
		self.scores = scores[:,0]
		if flags.orient == 'acc':
			losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.targets)
			# rmse_scores = tf.cast(tf.argmax(scores,-1),tf.float32)
			# rmse_target = tf.cast(tf.argmax(self.targets,-1), tf.float32)
			# mse = tf.reduce_mean(tf.square(rmse_scores - rmse_target))
			# rmse = tf.sqrt(mse)
			self.loss = tf.reduce_mean(losses)
			# self.loss += rmse
		else:
			# scores = tf.squeeze(scores, -1)
			# targets = tf.cast(self.targets,tf.float32)
			targets = tf.argmax(self.targets, axis=-1)
			targets = tf.cast(targets,tf.float32)
			# self.scores = tf.nn.sigmoid(self.scores)

			mse = tf.reduce_mean(tf.square(targets - self.scores))
			self.loss = mse
			# self.loss = tf.sqrt(mse)


		sent_latent = tf.nn.embedding_lookup(self.sent_embedding, self.sent)
		sent_loss = tf.reduce_mean(tf.square(sent_latent - h))

		self.loss += sent_loss

		# mse_score 	= tf.argmax(scores, axis=-1)
		# mse_target 	= tf.argmax(self.targets, axis=-1)
		# mse_loss 	= tf.reduce_mean(tf.square(mse_score - mse_target))
		# mse_loss 	= tf.cast(mse_loss, tf.float32)
		# rmse_loss 	= tf.sqrt(mse_loss)
		# self.loss 	+= rmse_loss









		
		
		self.predictions = tf.argmax(scores, 1)
		self.updates = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(self.loss)

		# self.hcsc_scores, self.hcsc_loss = self.get_hcsc_loss(encode, embedding_size)
		# self.predictions = tf.argmax(self.hcsc_scores, 1)
		# # self.updates = tf.train.AdadeltaOptimizer(1,0.95,1e-6).minimize(self.loss)		
		# self.updates = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(self.hcsc_loss)




		# self.count = tf.cast(tf.equal(self.predictions, tf.argmax(self.targets, 1)), 'float')











	def get_att_cnn(self, inputs_): #batch_size, maxlen, embed_size
		users = tf.nn.embedding_lookup(self.user_embedding, self.users)
		items = tf.nn.embedding_lookup(self.item_embedding, self.items) #batch_size 1 embed_size

		ui = tf.concat([users, items], axis=-1) # batch_size 1 2*embed_size
		# ui = users*items

		ui_trans = tf.keras.layers.Dense(self.embedding_size)(ui)  #batch_size 1 embed_size

		input_trans = tf.keras.layers.Dense(self.embedding_size)(inputs_) #batch_size maxlen embed_size 


		weights = tf.nn.tanh(ui_trans+input_trans)



		weights = tf.keras.layers.Dense(1)(weights) #batch_size, maxlen, 1
		
		# polarity = tf.expand_dims(self.polarity, -1) 
		atts = weights#+polarity*10 

		ret_weights = tf.squeeze(atts,-1)

		atts = tf.nn.softmax(atts, axis=1)

		# weights = tf.expand_dims(self.polarity,-1)


		# ret_weights = tf.squeeze(weights, -1)*self.mask

		# sum_weights = tf.reduce_sum(tf.exp(ret_weights)*self.mask,axis=-1, keep_dims = True) #batch_size, 1

		# exp_weights = tf.exp(ret_weights)/sum_weights  # batch_size, maxlen

		# atts = exp_weights*self.mask

		# atts = tf.expand_dims(atts, -1)
		# weights = tf.nn.softmax(weights, axis=1)



		attended = atts*inputs_ 

		attended = tf.reduce_sum(attended, axis=1)

		return attended, ret_weights





	def get_hcsc(self,inputs_):
		fw_cell = tf.nn.rnn_cell.LSTMCell(self.embedding_size/4)
		bw_cell = tf.nn.rnn_cell.LSTMCell(self.embedding_size/4)
		fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=self.keep_prob)
		bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=self.keep_prob)
		
		outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs_, 
													 sequence_length=self.input_len, dtype=tf.float32)
		encode1 = tf.concat(outputs, 2)

			
			
		kernel_sizes = [3,5,7]
		num_filters = [50,50,50]
			
		outputs = []
		for kernel_size,num_filter in zip(kernel_sizes, num_filters):
			conv = tf.keras.layers.Conv1D(num_filter, kernel_size, padding = 'same')(inputs_)
			hidden = tf.nn.dropout(tf.nn.relu(conv),self.keep_prob)
			outputs.append(hidden)
		encode2 = tf.concat(outputs, axis=-1)

		encode = tf.concat([encode1, encode2], axis=-1)

		# return tf.reduce_max(encode, axis=1)

		user_embed = tf.nn.embedding_lookup(self.user_embedding, self.users)
		item_embed = tf.nn.embedding_lookup(self.item_embedding, self.items)


		vu_spec, self.a_ud = self.reduce_spec(encode, user_embed, 1, self.embedding_size, "user")
		vp_spec, self.a_pd = self.reduce_spec(encode, item_embed, 1, self.embedding_size, "prod")



		mean = tf.reduce_mean(encode, 1)
		
		user_mean = fc_layer(mean, self.embedding_size, activation_fn=None)
		vu_share_weights = tf.matmul(user_mean, tf.transpose(self.user_embedding, [1,0]))
		vu_share_weights = tf.expand_dims(tf.nn.softmax(vu_share_weights), -1)
		vu_share = tf.multiply(vu_share_weights, self.user_embedding)
		vu_share = tf.expand_dims(tf.reduce_sum(vu_share, 1), -2)
		vu_share, self.a_us = self.reduce_spec(encode, vu_share, 1, self.embedding_size, "user1")
		
		prod_mean = fc_layer(mean, self.embedding_size, activation_fn=None)
		vp_share_weights = tf.matmul(prod_mean, tf.transpose(self.item_embedding, [1,0]))
		vp_share_weights = tf.expand_dims(tf.nn.softmax(vp_share_weights), -1)
		vp_share = tf.multiply(vp_share_weights, self.item_embedding)
		vp_share = tf.expand_dims(tf.reduce_sum(vp_share, 1), -2)
		vp_share, self.a_ps = self.reduce_spec(encode, vp_share, 1, self.embedding_size, "prod1")

		vp = 0.5*vp_spec + 0.5*vp_share
		vu = 0.5*vu_spec + 0.5*vu_share


		Wg = tf.get_variable("Wg", [self.embedding_size*2, self.embedding_size], initializer=xavier_initializer())
		bg = tf.Variable(tf.constant(0.0, shape=[self.embedding_size]))
		h0 = tf.concat([vu, vp], -1)
		gate = tf.sigmoid(tf.nn.xw_plus_b(h0, Wg, bg))
		self.gup = tf.reduce_mean(gate, -1)
		h = gate * vu + (1-gate) * vp

		return h

		# return tf.concat([encode1, encode2], axis=-1)


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
		# hidden2 = self.get_cnn(inputs_)

		conv3 = tf.keras.layers.Conv1D(300, 5, padding = 'same')(hidden2)
		encode = tf.nn.dropout(tf.nn.relu(conv3), self.keep_prob)

		return encode

	def get_user_item(self):
		# print("neighbor matrix of shape: ", self.u_neighbor.shape)
		u_neighbors = tf.nn.embedding_lookup(self.u_neighbor,self.users)# batch_size, 1, 20
		un_embed = tf.nn.embedding_lookup(self.user_embedding, u_neighbors)#batch_size, 1, 20, 300 
		un_embed = tf.squeeze(un_embed,1) #batch_size, 20, 300

		pool_size = self.u_neighbor.shape[-1]
		user = tf.keras.layers.MaxPool1D(pool_size)(un_embed) #batch_size, 1, 300

		# user = tf.nn.embedding_lookup(self.user_embedding, self.users)
		# user,self.u_atts = self.get_att_neighbors(user, un_embed)

		# user = tf.nn.embedding_lookup(self.user_embedding,self.users)



		p_neighbors = tf.nn.embedding_lookup(self.p_neighbor,self.items)# batch_size, 1, 20
		pn_embed = tf.nn.embedding_lookup(self.item_embedding, p_neighbors)#batch_size, 1, 20, 300 
		pn_embed = tf.squeeze(pn_embed,1)
		
		pool_size = self.p_neighbor.shape[-1]
		item = tf.keras.layers.MaxPool1D(pool_size)(pn_embed)

		# item = tf.nn.embedding_lookup(self.item_embedding, self.items)
		# user,self.i_atts = self.get_att_neighbors(item, pn_embed)

		# item = tf.nn.embedding_lookup(self.item_embedding, self.items)


		latent = tf.concat([user,item,user*item], axis=-1)
		num_layers = 2
		for i in range(num_layers):
			units = self.embedding_size//(2**i)
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
			 flags,
			 inputs,
			 targets,
			 users,
			 items,
			 input_len,
			 sent,
			 sent_neighbors,
			 training=True):
		
		# max_len = np.max([len(x) for x in inputs])
		# pad = 3
		# inputs = self.add_pad(inputs, max_len, pad)
		
		input_feed = {}
		input_feed[self.inputs] 		= inputs
		input_feed[self.input_len] 		= input_len
		input_feed[self.targets] 		= targets
		input_feed[self.users] 			= users
		input_feed[self.items] 			= items
		# input_feed[self.polarity] 		= polarity
		input_feed[self.sent]			= sent 
		input_feed[self.sent_neighbors] = sent_neighbors
		# input_feed[self.polarity]		= pt
		
		if training:
			input_feed[self.keep_prob] = 0.5
			output_feed = [self.loss, self.updates, self.scores]
		else:
			input_feed[self.keep_prob] = 1.0
			if flags.orient == 'acc':
				output_feed = [self.predictions]
			else:
				output_feed = [self.scores]
		
		outputs = session.run(output_feed, input_feed)
		
		return outputs
