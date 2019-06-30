import numpy as np 
from nltk import word_tokenize
import re
from collections import Counter
import pickle
from nltk.corpus import stopwords
def clean_str(string):
	"""
	Tokenization/string cleaning for all datasets except for SST.
	Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
	"""

	string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
	string = re.sub(r"\s{2,}", " ", string)
	return string.strip()


# def processfile(filename):
def processfile(filename):
	fr = open(filename)
	data = fr.readlines()
	fr.close()
	users = []
	items = []
	texts = []
	rates = []
	for line in data:
		line = line.strip()
		# listfromline = line.split('\t\t')
		user,item,rate,text = line.split('\t\t')
		text = clean_str(text)
		text = word_tokenize(text)
		# print(text)
		users.append(user)
		items.append(item)
		rates.append(int(rate))
		texts.append(text)
	return users, items,rates, texts


def processfiles(domain,filenames):
	users = []
	items = []
	rates = []
	texts = []
	for filename in filenames:
		print('processing ... ',filename)
		user,item, rate, text = processfile(filename)
		users.extend(user)
		items.extend(item)
		rates.extend(rate)
		texts.extend(text)
		# break
	# print(texts)
	# user_c = Counter(users)
	# item_c = Counter(items)
	# print(user_c.most_common()[-10:])
	# print(item_c.most_common()[-10:])
	print('finish processing....\nconverting....')
	p_user,p_item,p_text,word_c, word2idx = get_dict(users, items, texts)

	data = {}
	data['user'] = p_user 
	data['item'] = p_item
	data['rate'] = rates
	data['r_text'] = texts
	data['p_text'] = p_text

	data['word2idx'] = word2idx
	data['word_counter'] = word_c
	fr = open(domain+'.pkl','wb')
	pickle.dump(data,fr)
	fr.close()


def get_dict(user,item,text):
	user_set = set(user)
	item_set = set(item)
	user_dict = dict((u,i) for i,u in enumerate(user_set))
	new_user = [user_dict[u] for u in user]
	item_dict = dict((v,j) for j,v in enumerate(item_set))
	new_item = [item_dict[v] for v in item]

	word_c = Counter()
	for t in text:
		word_c += Counter(t)

	word2idx =  dict((word, i+1) for i,word in enumerate(word_c))#for i,word in enumerate(word_c)
	idx2word =  dict((i+1, word) for i,word in enumerate(word_c))

	new_text = [[word2idx[word] for word in t] for t in text]

	return new_user, new_item, new_text, word_c, word2idx


def filter_vocab(domain):
	filename = domain+'.pkl'
	fr = open(filename,'rb')
	data = pickle.load(fr)
	fr.close()

	max_words = 30000
	stoplist = stopwords.words('english')

	word_c = data['word_counter']
	vocab = []

	for tp in word_c.most_common():
		if len(vocab) > max_words:
			break
		if tp[0] not in stoplist:
			vocab.append(tp[0])


	word2idx = dict((word,i+1) for i,word in enumerate(vocab))

	r_text = data['r_text']
	p_text = [[word2idx[word] for word in text if word in word2idx] for text in r_text]

	data['p_text'] = p_text
	data['r_text'] = ''
	data['word2idx'] = word2idx
	fr = open(domain+'_'+str(max_words)+'.pkl','wb')
	pickle.dump(data,fr)
	fr.close()




def raw(domain):
	split = ['test','dev','train']
	year = ['2013','2014']
	if domain == 'imdb':
		filenames = ['imdb.'+s+'.txt.ss' for s in split]
	elif domain == 'yelp':
		filenames = ['yelp-'+y+'-seg-20-20.'+s+'.ss' for s in split for y in year  ]

	# print(filenames)
	processfiles(domain,filenames)

if __name__ == '__main__':
	domain = 'imdb'
	# raw(domain)
	filter_vocab(domain)