import os 
import tensorflow as tf 
import numpy as np 
import hcsc
import my_model
import mi_model
import utils
from data_loader import Data_Loader 
import sys 
from sklearn.metrics import accuracy_score
import time

def get_best_res(filename):
	fr = open(filename)
	data = fr.readlines()
	fr.close()
	res = []
	for line in data:
		line = line.strip()
		listfromline = line.split()
		res.append(float(listfromline[1]))
	return min(res), max(res)


def train(sess,model,data_loader, flags):
	saver = tf.train.Saver(max_to_keep=1)

	ckpt = tf.train.get_checkpoint_state(flags.ckpt_dir)
	if ckpt and ckpt.model_checkpoint_path:
		saver.restore(sess, ckpt.model_checkpoint_path)
		print(" [*] loading parameters success !!!")
	else:
		print(" [!] loading parameters failed  ...")

	res_filename = flags.ckpt_dir+'/res.txt'
	time_filename = flags.ckpt_dir+'/time.txt'
	if os.path.exists(res_filename):
		best_rmse,best_acc = get_best_res(res_filename)
	else:
		best_rmse,best_acc = 10,0

	res_loss = []
	times = []
	fr = open(res_filename,'a')
	fr_time = open(time_filename,'a')
	for i in range(flags.epoch):
		data_loader.reset_pointer()
		batches = data_loader.train_size//flags.batch_size+1
		for b in range(batches):
			data = data_loader.__next__()
			# for item in data:
			# 	print(item.shape)
			start_time = time.time()
			x_batch, y_batch, l_batch, u_batch, p_batch, s_batch, n_batch = data
			# print('y_batch of shape:', y_batch.shape)
			loss, updats,scores = model.step(sess, flags, x_batch, y_batch, u_batch, p_batch,\
											l_batch, s_batch, n_batch)
			end_time = time.time()
			time_span = np.round(end_time - start_time, 5)

	
			trained_batches = i*batches+b
			sys.stdout.write('\repoch:{}, batch:{}/{}, trained_batches:{}, loss:{}'.format(i,b,batches,trained_batches, loss))
			sys.stdout.flush()
			if trained_batches!=0 and trained_batches%100==0:
				res, another = eval(sess, model, data_loader, flags)
				res = round(res,5)
				loss = round(loss,5)
				another = round(another,5)
				fr.write(str(loss)+'\t'+str(res)+'\t'+str(another))
				fr.write('\n')
				fr_time.write(str(time_span)+'\n')

				res_loss.append([loss, res])
				print("res", res)
				if flags.orient == 'acc':
					if res>best_acc:
						best_acc = res
						saver.save(sess, flags.ckpt_dir+'/model.ckpt', global_step  = trained_batches)
				else:
					if res<best_rmse:
						best_rmse = res
						saver.save(sess, flags.ckpt_dir+'/model.ckpt', global_step  = trained_batches)

				# break
		# break
	fr.close()
	fr_time.close()
	res_loss = np.array(res_loss)
	# np.savetxt(flags.ckpt_dir+'/res.txt', res_loss)
	np.savetxt(flags.ckpt_dir+'/times.txt', [np.mean(times)])





def eval(sess, model, data_loader, flags):
	data = data_loader.val()

	batch_size = flags.batch_size 
	test_size = len(data[0])
	batches = test_size//batch_size+1
	ypred = []
	ytrue = []
	print("\nevaluating...")
	for b in range(batches):
		begin = b*batch_size
		end = min(test_size, (b+1)*batch_size)
		batch_data = [sub_data[begin:end] for sub_data in data]
		x_val, y_val, l_val, u_val,p_val, s_val, n_val= batch_data
		outputs = model.step(sess, flags, x_val, y_val, u_val, p_val, l_val, s_val, n_val, training = False)

		sys.stdout.write('\rbatch: {}/{} '.format(b,batches))
		sys.stdout.flush()
		ypred.extend(list(outputs[0].flatten()))
		# print(y_val)
		ytrue.extend(list(np.argmax(y_val, axis=-1)))
		# if flags.orient == 'acc':
		# 	ytrue.extend(list(np.argmax(y_val, axis=-1)))
		# else:
		# 	ytrue.extend(list(y_val.flatten()))
	if flags.orient == 'acc':
		res = accuracy_score(ypred, ytrue)
		another = np.mean((np.array(ytrue) - np.array(ypred))**2)**0.5
		# rmse = np.mean(np.abs((np.array(ytrue) - np.array(ypred))))
		print('rmse: ', another)
		# np.savetxt('ytrueypred',[ytrue,ypred])
	else:
		# print(len(ytrue), len(ypred))
		ypred = np.array(ypred)
		ytrue = np.array(ytrue)
		# ypred = np.round(ypred)
		ypred[ypred<0] = 0
		ypred[ypred>9] = 9

		res = np.mean((ytrue - ypred)**2)**0.5
		ypred = np.round(ypred)
		another = accuracy_score(ypred,ytrue)
		print('acc: ', accuracy_score(ypred,ytrue))


	return res, another

def test(sess, model, data_loader, flags):
	saver = tf.train.Saver(max_to_keep = 1)
	ckpt = tf.train.get_checkpoint_state(flags.ckpt_dir)
	if ckpt and ckpt.model_checkpoint_path:
		saver.restore(sess, ckpt.model_checkpoint_path)
		print(' [*] loading parameters success !!!')
	else:
		print(' [!] loading parameters failed  ...')
		return 
	data = data_loader.val()
	batch_size = flags.batch_size 
	data_size = len(data[0])
	print("data_size: ", data_size)
	batches = data_size//batch_size+1
	ypred = []
	ytrue = []
	print("evaluating...")
	# data = data_loader.train_dat()

	# to_save = ['user', 'item', 'word']
	# outputs = sess.run([model.user_embedding, model.item_embedding, model.embedding])
	# for file,embed in zip(to_save, outputs):
	# 	np.save(flags.ckpt_dir+'/'+file, embed)

	# dat_data = ['x','y','l','u','p','uc','pc']
	# fr = open(flags.ckpt_dir+'/atts.txt','a')

	for b in range(batches):
		begin = b*batch_size
		end = min(data_size, (b+1)*batch_size)
		batch_data = [sub_data[begin:end] for sub_data in data]
		x_val, y_val, l_val, u_val,p_val,m_val = batch_data

		outputs = model.step(sess, x_val, y_val, u_val, p_val, l_val, m_val, training = False)
		sys.stdout.write('\rbatch: {}/{}'.format(b,batches))
		sys.stdout.flush()
		ypred.extend(list(outputs[0].flatten()))
		# print(y_val)
		ytrue.extend(list(np.argmax(y_val, axis=-1)))

		# atts.append(list(outputs[-1].flatten()))
		# atts = outputs[-1]
		# for att in atts:
		# 	att = list(map(str,np.round(att,3)))
		# 	fr.write(' '.join(att))
		# 	fr.write('\n')
		# break

	# fr.close()
	res = accuracy_score(ypred, ytrue)
	
	print('\n',res)
	np.savetxt(flags.ckpt_dir+'/res.npy',np.array([res]))

	return res




def main():
	flags = tf.flags.FLAGS 

	tf.flags.DEFINE_string('domain','yelp','domain of the dataset')
	tf.flags.DEFINE_string('base_model', 'att_cnn', 'base model')
	tf.flags.DEFINE_integer('embed_size',300, 'embedding size')
	tf.flags.DEFINE_integer('maxlen',300, 'maximum length of sequences')
	tf.flags.DEFINE_integer('epoch', 5, 'epochs for training')
	tf.flags.DEFINE_integer('batch_size', 128, 'mini batchsize for training')
	tf.flags.DEFINE_integer('x_neighbors',5,'number of collaborative x_dat')
	tf.flags.DEFINE_string('train_test', 'train', 'training or test')
	tf.flags.DEFINE_boolean('co_variant',True,'including collaborative sentences')
	tf.flags.DEFINE_boolean('ui_variant', True, 'including user-item interaction')
	tf.flags.DEFINE_string('orient','acc','acc or rmse, which to be optimized')
	tf.flags.DEFINE_string('ckpt_dir',flags.domain+'_ckpt'+'_'+flags.orient, 'dir of checkpoint')
	tf.flags.DEFINE_string('data_dir', '/home/wenjh/data/'+flags.domain, 'dir of dataset')

	num_class = 10 if flags.domain == 'imdb' else 5
	tf.flags.DEFINE_integer('num_class', num_class, 'number of target class')

	flags(sys.argv)
	data_loader = Data_Loader(flags)


	data_size = len(data_loader.x_dat)


	model = mi_model.Model(flags,
						data_loader,
						data_size,
						None)
	sess = tf.Session()
	tf.set_random_seed(1234)
	np.random.seed(1234)

	sess.run(tf.global_variables_initializer())
	sess.run(model.embedding.assign(data_loader.embed_mat))


	if not os.path.exists(flags.ckpt_dir):
		os.makedirs(flags.ckpt_dir)

	if flags.train_test == 'train':
		train(sess, model, data_loader, flags)
	else:
		test(sess, model, data_loader, flags)

if __name__ == '__main__':
	main()