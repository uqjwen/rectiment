import pickle 
import numpy as np 
import tensorflow as tf 
import utils
import os
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
class Data_Loader():
	def __init__(self, flags):
		files = ['train', 'dev', 'test']
		filenames = [flags.data_dir+'/'+file+'.txt' for file in files]
		if flags.domain == 'imdb':
			self.size_split = [67426, 8381, 9112]

		self.batch_size = flags.batch_size

		self.vec_file = flags.data_dir+'/glove.840B.300d.txt'
		self.embed_size = flags.embed_size
		self.num_class = flags.num_class


		pickle_file = flags.data_dir+'/data.pkl'

		# x_dict = utils.my_get_dict(filenames)
		self.get_data(filenames, pickle_file)

		self.split()

		# self.get_neighbors_from_rating()
		self.get_neighbors_from_embed(flags.ckpt_dir)

	def get_emb_mat(self, filename, embed_size):
		fr = open(filename,'r', encoding='utf-8', errors = 'ignore')

		# for line in fr.readline()
		embed_mat = np.random.uniform(-0.5,0.5,(len(self.x_dict)+1, embed_size)).astype(np.float32)
		for line in fr:
			line = line.strip()
			listfromline = line.split()
			token,vector = listfromline[0],listfromline[1:]
			if token in self.x_dict:
				index = self.x_dict[token]
				embed_mat[index] = list(map(float,vector))
		fr.close()
		return embed_mat

	def get_data(self,source_files, save_file):
		names = self.__dict__
		dict_data = ['x_dict', 'u_dict', 'p_dict', 'u_freq', 'p_freq']
		dat_data = ['x','y','l','u','p','uc','pc']
		if not os.path.exists(save_file):
			self.x_dict = utils.my_get_dict(source_files)
			print('finish getting x_dict')
			self.u_dict, self.p_dict, self.u_freq, self.p_freq = utils.my_get_up_dict(source_files)
			print('finish getting dict data')
			data = utils.my_get_flat_data(source_files, self.x_dict, self.u_dict, self.p_dict, self.u_freq, self.p_freq)
			# data = utils.my_get_flat_data(source_files, {}, {}, {}, {}, {})
			print('finish getting training data')

			self.embed_mat = self.get_emb_mat(self.vec_file, self.embed_size)

			print('finish getting embedding matrix')

			for i,dat in enumerate(dat_data):
				names[dat+'_dat'] = data[i]

			names['y_dat'] = to_categorical(names['y_dat'],self.num_class)

			pickle_data = {}
			pickle_data['data'] = [names[dat+'_dat'] for dat in dat_data]

			for item in dict_data:
				pickle_data[item] = names[item]

			pickle_data['embed_mat'] = self.embed_mat

			fr = open(save_file,'wb')
			pickle.dump(pickle_data,fr)
			fr.close()
		else:
			fr = open(save_file, 'rb')
			pickle_data = pickle.load(fr)
			fr.close()
			for i,dat in enumerate(dat_data):
				names[dat+'_dat'] = pickle_data['data'][i]
			for item in dict_data:
				names[item] = pickle_data[item]
			self.embed_mat = pickle_data['embed_mat']

		lens = [len(x) for x in self.x_dat]
		print("mean: ",np.mean(lens), 'max: ', np.max(lens))
		self.maxlen = np.max(lens)
		self.maxlen = 500
		self.x_dat = pad_sequences(self.x_dat, self.maxlen, padding = 'post')
		self.l_dat[self.l_dat > self.maxlen] = self.maxlen


		self.vocab_size = len(self.x_dict)+1
		self.num_user 	= len(self.u_dict)
		self.num_item 	= len(self.p_dict)
		print("num_user: ", self.num_user, "num_item: ", self.num_item)


	def split(self):
		names = self.__dict__
		dat_data = ['x','y','l','u','p','uc','pc']
		files = ['train', 'dev', 'test']
		my_split = np.cumsum(self.size_split)
		for i,(to_split, file) in enumerate(zip(my_split,files)):
			# end = to_split if i!=0 else my_split[-1]
			end = to_split
			begin = 0 if i==0 else my_split[i-1]
			# print(file, begin,end, to_split)
			for dat in dat_data:
				names[file+'_'+dat] = names[dat+'_dat'][begin:end]# self.train_x...||self.dev_x...||self.test_x

		self.train_size = len(self.train_x)



		pmtt = np.random.permutation(self.train_size)
		for dat in dat_data:
			names['train_'+dat] = names['train_'+dat][pmtt]


	def reset_pointer(self):
		self.pointer = 0

	def __next__(self):
		names = self.__dict__
		dat_data = ['x','y','l','u','p','uc','pc']
		begin = self.pointer*self.batch_size
		end  = (self.pointer+1)*self.batch_size
		self.pointer+=1
		end = min(end,self.train_size)

		res_dat = [names['train_'+dat][begin:end]  for dat in dat_data]
		return res_dat

	def val(self):
		names = self.__dict__
		dat_data = ['x','y','l','u','p','uc','pc']

		res_dat = [names['test_'+dat] for dat in dat_data]
		return res_dat

	def get_neighbors_from_rating(self,n_neighbors=20):
		self.u_dat
		self.p_dat 
		self.y_dat 
		up_mat = np.zeros((self.num_user, self.num_item))
		for u,p,r in zip(self.u_dat, self.p_dat, self.y_dat):
			r = np.argmax(r)
			up_mat[u,p] = r 
		uu_mat = self.get_nn_mat(up_mat)
		pp_mat = self.get_nn_mat(up_mat.T)
		self.u_neighbors = np.argsort(uu_mat,axis=-1)[:,-n_neighbors:]
		self.p_neighbors = np.argsort(pp_mat,axis=-1)[:,-n_neighbors:]


	def get_nn_mat(self,mat):
		return np.dot(mat,mat.T)

	def get_neighbors_from_embed(self, ckpt_dir, n_neighbors=20):
		u_mat = np.load(ckpt_dir+'/user.npy')
		p_mat = np.load(ckpt_dir+'/item.npy')

		uu_mat = self.get_nn_mat(u_mat)
		pp_mat = self.get_nn_mat(p_mat)

		self.u_neighbors = np.argsort(uu_mat,axis=-1)[:,-n_neighbors:]
		self.p_neighbors = np.argsort(pp_mat,axis=-1)[:,-n_neighbors:]

		


if __name__ == '__main__':
	flags = tf.flags.FLAGS 
	tf.flags.DEFINE_string('domain','imdb','domain of the dataset')
	tf.flags.DEFINE_string('ckpt_dir',flags.domain+'_ckpt', 'dir of checkpoint')
	tf.flags.DEFINE_string('data_dir', flags.domain, 'dir of dataset')
	tf.flags.DEFINE_string('base_model', 'cnn', 'base model')
	tf.flags.DEFINE_integer('embed_size',300, 'embedding size')
	tf.flags.DEFINE_integer('epoch', 100, 'epochs for training')
	tf.flags.DEFINE_integer('batch_size', 64, 'mini batchsize for training')
	num_class = 10 if flags.domain == 'imdb' else 5
	tf.flags.DEFINE_integer('num_class', num_class, 'number of target class')

	data_loader = Data_Loader(flags)

	data_loader.reset_pointer()
	data_loader.__next__()
	data_loader.val()