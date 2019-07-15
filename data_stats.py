import numpy as np 
import scipy.stats as ss 
import sys 
import tensorflow as tf 
import pickle
import os 


class my_stats():
	def __init__(self, flags):

		stat_file = flags.data_dir+'/stats.txt'
		if not os.path.exists(stat_file):
			data_file = flags.data_dir+'/data.pkl'
			# neighbors = data_dir+'/x_neighbors_item.npy'
			sims_file = flags.data_dir+'/u_neighbors.npy'
			fr = open(data_file, 'rb')
			data = pickle.load(fr)
			fr.close()

			self.sims = np.load(sims_file)

			_, self.y_dat, _, self.u_dat, self.p_dat = data['data']

			self.y_dat = np.argmax(self.y_dat, axis=-1)

			self.data_size = len(self.y_dat)

			y1,y2,y3 = self.get_stats()

			ys = np.array([y1,y2,y3])
			np.savetxt(flags.data_dir+'/stats.txt', ys, fmt='%d')
		else:
			ys = np.genfromtxt(stat_file)
		# for i in range(len(ys)):
		# 	for j in range(i+1, len(ys)):
		print('\n')
		for i,ys1 in enumerate(ys):
			for j,ys2 in enumerate(ys[i+1:]):
				stat_val, p_val = ss.ttest_ind(ys1, ys2, equal_var = False)
				r,p = ss.pearsonr(ys1, ys2)
				print(p_val, r)


	def get_stats(self):
		to_random = []
		to_neighbors = []
		to_sim_neighbors = []
		for i,(y,u,p) in enumerate(zip(self.y_dat, self.u_dat, self.p_dat)):
			sys.stdout.write('\r{}/{}'.format(i, self.data_size))
			index = list(np.where(self.p_dat == p)[0])
			index.remove(i)


			random_index = np.random.randint(0,self.data_size)
			random_y = self.y_dat[random_index]

			random_neighbor_y = self.y_dat[np.random.choice(index)]

			labels = self.y_dat[index]
			users = self.u_dat[index].flatten() #users who have rated the item
			sims = self.sims[u].flatten()
			sims = sims[users]
			sims = np.exp(sims)/sum(np.exp(sims))

			debug = self.y_dat[index]


			random_sim_y = np.random.choice(labels,p= sims)

			to_sim_neighbors.append(abs(y - random_sim_y))
			to_neighbors.append(abs(y - random_neighbor_y))
			to_random.append(abs(y - random_y))

		return to_random, to_neighbors, to_sim_neighbors







if __name__ == '__main__':
	flags = tf.flags.FLAGS 
	tf.flags.DEFINE_string('domain', 'yelp', 'domain of dataset')
	data_dir = '/home/wenjh/data/'+flags.domain
	tf.flags.DEFINE_string('data_dir', data_dir, 'directory of dataset')
	flags(sys.argv)

	stats = my_stats(flags)
