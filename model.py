import numpy as np 
import sys
import tensorflow as tf 
import os
from tensorflow.contrib.layers import xavier_initializer
from data_loader import Data_Loader
from sklearn.metrics import accuracy_score, f1_score


class Model():
	def __init__(self, flags, emb_mat):
		self.maxlen 		= flags.maxlen
		self.num_user 		= flags.num_user
		self.num_item 		= flags.num_item
		self.num_class 		= flags.num_class
		self.emb_size 		= flags.emb_size
		self.dropout 		= flags.dropout
		self.hidden_size 	= flags.hidden_size


		self.user = tf.placeholder(tf.int32, shape=[None])
		self.item = tf.placeholder(tf.int32, shape=[None])
		self.label = tf.placeholder(tf.int32, shape=[None])

		self.text = tf.placeholder(tf.int32, shape=[None, self.maxlen])

		self.training = tf.placeholder(tf.bool)

		# self.user_embedding = tf.Variable(tf.random.uniform([self.num_user, self.emb_size],-0.5,0.5))
		# self.item_embedding = tf.Variable(tf.random.uniform([self.num_item, self.emb_size],-0.5,0.5))

		self.user_embedding = tf.get_variable('user_embedding', [self.num_user, self.emb_size],
								initializer = xavier_initializer())
		self.item_embedding = tf.get_variable('item_embedding', [self.num_item, self.emb_size],
								initializer = xavier_initializer())
		# self.text_embedding = tf.get_variable('text_embedding', )
		self.text_embedding = tf.Variable(emb_mat)


		text_latent = self.get_text_latent()

		# print('-----------------------------------------\n',text_latent.shape)

		# text_latent = tf.layers.dense(text_latent, self.hidden_size//2, tf.nn.relu)
		text_latent = tf.keras.layers.Dense(self.hidden_size//2,'relu')(text_latent)

		self.logits = tf.keras.layers.Dense(self.num_class)(text_latent)

		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.logits, labels = self.label)
		self.cost = tf.reduce_mean(loss)


		self.global_step 	= tf.Variable(0, trainable = False)

		self.lr 			= tf.train.exponential_decay(0.0001, self.global_step, decay_steps=200, decay_rate=0.1)

		# self.train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.cost)
		self.train_op 		= tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cost)



	def get_text_latent(self):
		latent = tf.nn.embedding_lookup(self.text_embedding, self.text) # batch_size maxlen 
		kernels = [3,5,7]
		temp_size = self.hidden_size//len(kernels)
		outputs = []
		for kernel in kernels:
			conv = tf.keras.layers.Conv1D(temp_size, kernel,1,'same')(latent)
			h = tf.nn.relu(conv)
			h = tf.layers.dropout(conv, self.dropout, training = self.training)
			outputs.append(h)

		conv_concat = tf.concat(outputs, axis=-1)
		print('----------------------------\n',conv_concat.shape)

		conv2 = tf.keras.layers.Conv1D(self.hidden_size, 5, 1, 'same')(conv_concat)
		conv2 = tf.nn.relu(conv2)
		text_latent = tf.keras.layers.MaxPool1D(self.maxlen,1)(conv2)
		return tf.squeeze(text_latent,1)



tf.flags.DEFINE_string('domain', 'imdb', 'laptop or restaurant')
tf.flags.DEFINE_integer('emb_size', 100, 'embedding size')
tf.flags.DEFINE_integer('maxlen', 300, 'maximum review length')
tf.flags.DEFINE_integer('batch_size',128,'training mini batch size')
tf.flags.DEFINE_integer('hidden_size',128,'hidden size cnn network')
tf.flags.DEFINE_integer('num_class',10,'number of target class')
tf.flags.DEFINE_integer('num_user',0,'number of user')
tf.flags.DEFINE_integer('num_item',0,'number of item')
tf.flags.DEFINE_float('dropout',0.5,'dropout rate')
tf.flags.DEFINE_integer('epoch',100,'epoch for training')
# tf.flags.DEFINE_string('dir','./ckpt_imbd','directory for saving ckeckpoint')

flags = tf.flags.FLAGS 
flags(sys.argv)


ckpt_dir = './ckpt_' + flags.domain+'/'
if not os.path.exists(ckpt_dir):
	os.makedirs(ckpt_dir)


def train():
	data_loader = Data_Loader(flags)
	emb_mat = data_loader.emb_mat
	flags.num_user = data_loader.num_user
	flags.num_item = data_loader.num_item
	flags.num_class = data_loader.num_class

	model = Model(flags, emb_mat)
	best_acc = 0
	best_f1score = 0
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver()
		ckpt = tf.train.get_checkpoint_state(ckpt_dir)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			print(" [*] loading parameters success!!!")
		else:
			print(" [!] loading parameters failed ...")


		for i in range(flags.epoch):
			data_loader.reset_pointer()
			batch = data_loader.train_size//flags.batch_size+1
			for b in range(batch):
				user,item,rate,text = data_loader.next()
				feed_dict = {model.user:user,
							model.item:item,
							model.label:rate,
							model.text:text,
							model.training:True}
				logits,loss,_ = sess.run([model.logits,model.cost, model.train_op], feed_dict = feed_dict)
				# print(logits)
				sys.stdout.write('\repoch:{}, batch:{}, loss:{}'.format(i,b,loss))
				sys.stdout.flush()
				# break
			lr = sess.run(model.lr, feed_dict = {model.global_step:i})
			dev_data = data_loader.dev()
			acc, f1score =  dev(sess, model, dev_data)
			print("\nacc: ", acc, "f1score: ", f1score)
			if f1score>best_f1score:
				best_f1score = f1score
				saver.save(sess, ckpt_dir+'model.ckpt', global_step = i)


			# break
def dev(sess, model, dev_data):
	user, item, label, text = dev_data
	feed_dict = {model.user:user,
				model.item:item,
				model.label:label,
				model.text:text,
				model.training:False}
	logits = sess.run(model.logits, feed_dict = feed_dict)
	ypred = np.argmax(logits, axis=-1)
	acc= accuracy_score(label, ypred)
	f1score = f1_score(label, ypred, average='micro')
	return acc, f1score

def rmse(ytrue, ypred):
	assert len(ytrue) == len(ypred)
	mse = np.sum((ytrue-ypred)**2)/len(ytrue)
	rmse = mse**0.5
	return rmse

	# (ytrue - ypred)**2/


if __name__ == '__main__':
	train()
