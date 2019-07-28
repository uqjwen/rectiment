import numpy as np 
import scipy.stats as ss 
import sys 
import tensorflow as tf 
import pickle
import os 


class my_stats():
	def __init__(self, flags):

		stat_file = flags.data_dir+'/num_stats.txt'
		self.flags = flags
		if not os.path.exists(stat_file):
			data_file = flags.data_dir+'/data.pkl'
			# neighbors = data_dir+'/x_neighbors_item.npy'
			sims_file = flags.data_dir+'/u_neighbors.npy'
			ssd_file = flags.data_dir+'/x_neighbor_item.npy'

			fr = open(data_file, 'rb')
			data = pickle.load(fr)
			fr.close()

			self.sims = np.load(sims_file)
			self.ssd = np.load(ssd_file)

			_, self.y_dat, _, self.u_dat, self.p_dat = data['data']

			self.y_dat = np.argmax(self.y_dat, axis=-1)

			self.data_size = len(self.y_dat)

			# y1,y2,y3 = self.get_stats()
			# for k in range(5,21):
			self.get_stats_k(21)

		else:
			ys = np.genfromtxt(stat_file)
		# for i in range(len(ys)):
		# 	for j in range(i+1, len(ys)):
		
		# print('\n')
		# for i,ys1 in enumerate(ys):
		# 	for j,ys2 in enumerate(ys[i+1:]):
		# 		stat_val, p_val = ss.ttest_ind(ys1, ys2, equal_var = False)
		# 		r,p = ss.pearsonr(ys1, ys2)
		# 		print(p_val, r)

	def get_stats_k(self, k):
		to_random = []
		to_neighbors = []
		to_sim_neighbors = []
		for i,(y,u,p) in enumerate(zip(self.y_dat, self.u_dat, self.p_dat)):
			sys.stdout.write('\r{}/{}'.format(i, self.data_size))
			index = list(np.where(self.p_dat == p)[0])
			index.remove(i)
			random_index = np.random.randint(0,self.data_size, k)
			neighbors_index = np.random.choice(index, k)



			users = self.u_dat[index].flatten()
			sims = self.sims[u].flatten()
			sims = sims[users]
			arg_index = np.argsort(sims)[::-1][:min(k,len(sims))]
			sim_neighbors_index = np.array(index)[arg_index]

			if len(sim_neighbors_index)<k:
				sim_neighbors_index = np.concatenate([sim_neighbors_index, [i]*(k-len(sims))])



			# sim_neighbors_index = self.ssd[i][:k]








			num_random = self.y_dat[random_index] == y
			num_neighbor = self.y_dat[neighbors_index] == y
			num_sim_neighbor = self.y_dat[sim_neighbors_index] == y

			to_random.append(num_random)
			to_neighbors.append(num_neighbor)
			to_sim_neighbors.append(num_sim_neighbor)

		to_random = np.array(to_random)
		to_neighbors = np.array(to_neighbors)
		to_sim_neighbors = np.array(to_sim_neighbors)


		res_dat = [to_random, to_neighbors, to_sim_neighbors]
		
		res = []

		# for dat in res_dat:
		# 	temp = np.mean(np.sum(dat, axis=1))
		# 	res.append(temp)
		# fr = open(self.flags.data_dir+'/num_stats.txt','a')
		# fr.write('\t'.join(map(str,np.round(res,5))))
		# fr.write('\n')
		# print('\n',k)


		for i in range(5,k):
			temp_res = []
			for dat in res_dat:
				temp = np.mean(np.sum(dat[:,:i],axis=1))
				temp_res.append(temp)
			res.append(temp_res)
		res = np.array(res)

		np.savetxt(self.flags.data_dir+'/num_stats.txt', res, fmt='%.3f')





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

		ys = np.array([to_random,to_neighbors,to_sim_neighbors])
		np.savetxt(self.flags.data_dir+'/stats.txt', ys, fmt='%d')

		# return to_random, to_neighbors, to_sim_neighbors







if __name__ == '__main__':
	flags = tf.flags.FLAGS 
	tf.flags.DEFINE_string('domain', 'yelp', 'domain of dataset')
	data_dir = '/home/wenjh/data/'+flags.domain
	tf.flags.DEFINE_string('data_dir', data_dir, 'directory of dataset')
	flags(sys.argv)

	stats = my_stats(flags)
