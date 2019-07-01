from keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np 
from keras.utils.np_utils import to_categorical
import os 
import tensorflow as tf 
import sys

class Data_Loader():
	def __init__(self, flags):
		self.domain = flags.domain
		filename = './data/'+self.domain+'_30000.pkl'
		fr = open(filename, 'rb')
		data = pickle.load(fr)
		fr.close()

		self.batch_size = flags.batch_size
		self.user = data['user']
		self.item = data['item']
		self.rate = data['rate']

		self.num_user = len(set(self.user))
		self.num_item = len(set(self.item))
		self.num_class = 10 if self.domain == 'imdb' else 5


		self.word2idx = data['word2idx']
		self.vocab_size = len(self.word2idx)+1
		self.emb_size = flags.emb_size
		self.p_text = data['p_text']


		# text_len = [len(text) for text in self.p_text]
		self.maxlen = flags.maxlen

		self.p_text = pad_sequences(self.p_text, self.maxlen)

		self.emb_mat = self.get_emb_mat()


		# print(max(text_len), np.mean(text_len))
		self.train_text_split()

		self.shuffle()

	def get_emb_mat(self):

		emb_mat_file = './data/'+self.domain+'_emb_mat.npy'
		if os.path.exists(emb_mat_file):
			emb_mat = np.load(emb_mat_file)
		else:
			emb_mat = np.random.uniform(-0.5,0.5,(self.vocab_size, self.emb_size))
			fr = open('/media/wenjh/Ubuntu 16.0/Downloads/glove.6B/glove.6B.'+str(self.emb_size)+'d.txt')
			glove = fr.readlines()
			fr.close()
			for line in glove:
				line = line.strip()
				listfromline = line.split()
				word = listfromline[0]
				vect = list(map(float,listfromline[1:]))
				if word in self.word2idx:
					emb_mat[self.word2idx[word]] = vect
			np.save(emb_mat_file, emb_mat)
		return emb_mat.astype(np.float32)
	def train_text_split(self):
		if self.domain == 'imdb':
			test_size 	= 9112
			dev_size 	= 8381
			train_size 	= 67426
			# print(train_size+test_size+dev_size)
			# print(len(self.p_text))

			assert train_size+dev_size+test_size == len(self.p_text)
		idx = np.cumsum([test_size, dev_size, train_size])


		split = ['test','dev','train']
		attrs = ['user','item','rate','p_text']
		names = self.__dict__
		for i,(index,sub_split) in enumerate(zip(idx,split)):
			begin = 0 if i == 0 else idx[i-1]
			end = index
			for attr in attrs:
				# names[attr][begin:end]
				setattr(self,sub_split+'_'+attr, np.array(names[attr][begin:end]))
		self.train_size = train_size
	def reset_pointer(self):
		self.pointer = 0
	def next(self):
		begin = self.pointer*self.batch_size
		end = (self.pointer+1)*self.batch_size
		self.pointer += 1
		if end>=self.train_size:
			self.pointer = 0
			end = self.train_size
		# end = min(end,self.train_size)
		return self.train_user[begin:end],\
				self.train_item[begin:end],\
				self.train_rate[begin:end],\
				self.train_p_text[begin:end]
	def dev(self):
		return self.dev_user, self.dev_item, self.dev_rate,self.dev_p_text

	def shuffle(self):
		names = self.__dict__
		attrs = ['user','item','rate']
		pmtt = np.random.permutation(self.train_size)
		for attr in attrs:
			names['train_'+attr] = names['train_'+attr][pmtt]
		split = ['train','dev','test']
		for sp in split:
			print(sp)
			names[sp+'_rate'] = names[sp+'_rate']-1









if __name__ == '__main__':
	tf.flags.DEFINE_string('domain', 'imdb', 'laptop or restaurant')
	tf.flags.DEFINE_integer('emb_size', 100, 'embedding size')
	tf.flags.DEFINE_integer('maxlen', 300, 'maximum review length')
	tf.flags.DEFINE_integer('batch_size',64,'training mini batch size')

	FLAGS = tf.flags.FLAGS 
	FLAGS(sys.argv)

	data_loader = Data_Loader(FLAGS)
	data_loader.reset_pointer()
	res = data_loader.next()
	for item in res:
		print(len(item),item)