import os 
import tensorflow as tf 
import numpy as np 
import hcsc
import my_model
import utils
from data_loader import Data_Loader 
import sys 
from sklearn.metrics import accuracy_score
def train(sess,model,data_loader, flags):
	saver = tf.train.Saver(max_to_keep=1)

	ckpt = tf.train.get_checkpoint_state(flags.ckpt_dir)
	if ckpt and ckpt.model_checkpoint_path:
		saver.restore(sess, ckpt.model_checkpoint_path)
		print(" [*] loading parameters success !!!")
	else:
		print(" [!] loading parameters failed  ...")

	best_acc= 0
	for i in range(flags.epoch):
		data_loader.reset_pointer()
		batches = data_loader.train_size//flags.batch_size+1
		for b in range(batches):
			data = data_loader.__next__()
			# for item in data:
			# 	print(item.shape)
			x_batch, y_batch, l_batch, u_batch, p_batch, uc_batch, pc_batch =data
			# print(x_batch[0])
			loss, updats = model.step(sess, x_batch, y_batch, u_batch, p_batch,\
									l_batch, uc_batch, pc_batch)
			sys.stdout.write('\repoch:{}, batch:{}/{}, loss:{}'.format(i,b,batches,loss))
			sys.stdout.flush()
			# break
			trained_batches = i*batches+b
			if trained_batches!=0 and trained_batches%200==0:
				acc = eval(sess, model, data_loader, flags)
				print("acc: ", acc)
				if acc>best_acc:
					best_acc = acc 
					saver.save(sess, flags.ckpt_dir+'/model.ckpt', global_step = trained_batches)
					np.save(flags.ckpt_dir+'/res.npy',acc)

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
	test_size = len(data[0])
	batches = test_size//batch_size+1
	ypred = []
	ytrue = []
	print("evaluating...")
	to_save = ['user', 'item', 'word']
	outputs = sess.run([model.user_embedding, model.item_embedding, model.embedding])
	for file,embed in zip(to_save, outputs):
		np.save(flags.ckpt_dir+'/'+file, embed)

	for b in range(batches):
		begin = b*batch_size
		end = min(test_size, (b+1)*batch_size)
		batch_data = [sub_data[begin:end] for sub_data in data]
		x_val, y_val, l_val, u_val,p_val,uc_val,pc_val = batch_data

		outputs = model.step(sess, x_val, y_val, u_val, p_val, l_val, uc_val, pc_val, training = False)
		sys.stdout.write('\rbatch: {}/{}'.format(b,batches))
		sys.stdout.flush()
		ypred.extend(list(outputs[0].flatten()))
		# print(y_val)
		ytrue.extend(list(np.argmax(y_val, axis=-1)))

	res = accuracy_score(ypred, ytrue)
	return res




def eval(sess, model, data_loader, flags):
	data = data_loader.val()

	batch_size = flags.batch_size 
	test_size = len(data[0])
	batches = test_size//batch_size+1
	ypred = []
	ytrue = []
	print("evaluating...")
	for b in range(batches):
		begin = b*batch_size
		end = min(test_size, (b+1)*batch_size)
		batch_data = [sub_data[begin:end] for sub_data in data]
		x_val, y_val, l_val, u_val,p_val,uc_val,pc_val = batch_data

		outputs = model.step(sess, x_val, y_val, u_val, p_val, l_val, uc_val, pc_val, training = False)
		sys.stdout.write('\rbatch: {}/{}'.format(b,batches))
		sys.stdout.flush()
		ypred.extend(list(outputs[0].flatten()))
		# print(y_val)
		ytrue.extend(list(np.argmax(y_val, axis=-1)))

	res = accuracy_score(ypred, ytrue)
	return res


def main():
	flags = tf.flags.FLAGS 

	tf.flags.DEFINE_string('domain','imdb','domain of the dataset')
	tf.flags.DEFINE_string('base_model', 'cnn', 'base model')
	tf.flags.DEFINE_integer('embed_size',300, 'embedding size')
	tf.flags.DEFINE_integer('epoch', 10, 'epochs for training')
	tf.flags.DEFINE_integer('batch_size', 128, 'mini batchsize for training')
	tf.flags.DEFINE_integer('maxlen',100, 'maximum len of sequences')
	tf.flags.DEFINE_string('train_test', 'train', 'training or test')
	tf.flags.DEFINE_string('ckpt_dir',flags.domain+'_ckpt_'+str(flags.maxlen), 'dir of checkpoint')
	tf.flags.DEFINE_string('data_dir', flags.domain, 'dir of dataset')
	num_class = 10 if flags.domain == 'imdb' else 5
	tf.flags.DEFINE_integer('num_class', num_class, 'number of target class')

	flags(sys.argv)
	data_loader = Data_Loader(flags)

	user_mean_freq = np.mean([data_loader.u_freq[x] for x in data_loader.u_freq])
	item_mean_freq = np.mean([data_loader.p_freq[x] for x in data_loader.p_freq])
	model = my_model.Model(flags.base_model,
						data_loader.vocab_size,
						flags.num_class,
						data_loader.num_user,
						data_loader.num_item,
						user_mean_freq,
						item_mean_freq,
						flags.embed_size,
						data_loader.u_neighbors,
						data_loader.p_neighbors,
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