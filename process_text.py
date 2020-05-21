#gotta import all of this. If you're missing one of these..well then fuck

import re
import csv
import time
import json
import logging
import hashlib
import pandas as pd
import multiprocessing
from datetime import datetime
from itertools import groupby
from collections import Counter
from nltk.stem import PorterStemmer
from gensim.models.word2vec import Word2Vec


def Main():
	start=time.time()
	###NAME OF DATASET AND MODEL
	context=""

	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

	###CHANGE PATH TO THE CORPUS OF YOUR CHOICE
	df1=pd.read_csv("", engine='python')
	###SET THE TYPE OF THE CORPUS
	df1['type']=""

	df1.index = df1['name'].str.len()
	df1 = df1.sort_index(ascending=False).reset_index(drop=True)

	###COPY AND PASTE ABOVE TO INCLIUDE MORE CORPUSES AS SEEN UNDER


	"""df2=pd.read_csv()
	df2['type']=""

	df2.index = df2['name'].str.len()
	df2 = df2.sort_index(ascending=False).reset_index(drop=True)

	df3=pd.read_csv()
	df3['type']=""


	df3.index = df3['name'].str.len()
	df3 = df3.sort_index(ascending=False).reset_index(drop=True)"""


	framelist=[df1]



	data=pd.concat(framelist, sort=True)
	data = data.reset_index(drop=True)


	hashmap={}

	#this function is where all the cool shit happens
	corpus=process_corpus(data, hashmap)

	#uncomment only to check 200 most common words(replace 200 with number of choice)
	#print(check_most_common(corpus, 200))

	
	end=time.time()
	print("["+str(get_time())+"]: Processing finished. Elapsed time: ", end-start)


	print("["+str(get_time())+']: Saving processed corpus to "corpus_'+context+'.csv"')

	with open("corpus_"+context+".csv", "w") as f:
		writer = csv.writer(f)
		writer.writerows(corpus)

	print("["+str(get_time())+']: Saving hashmap to "hashmap_'+context+'.json"')
	with open('hashmap_'+context+'.json', 'w') as f:
		json.dump(hashmap, f)


	#UNCOMMENT ONLY FOR TESTING AND VERIFYING THE DATA
	#exit()

	#training part
	model=trainer(corpus, context)

	end=time.time()
	print("["+str(get_time())+"]: Entire process finished. Elapsed time: ", end-start)



def process_corpus(df, hashmap):
	content=[]
	print("["+str(get_time())+"]: Processing corpuses...")

	for i in df.index: 
		if not pd.isnull(df['name'][i]):
			if "(" in df['name'][i]:		#removes parts of index lables that are within parantheses
				df['name'][i]=df['name'][i].split("(", 1)[0]

			#cleans up index label and hashes it
			df['name'][i] = (str(df['name'][i]).rstrip().replace(" & ", "_").replace('-', '_').replace(" -- ", "_").replace("/", "_").replace(" ", "_").replace('’', "").replace("'", "").replace(',', '').replace('.', '').lower())
			hashed_id=df['type'][i]+"_"+hashlib.md5(df['name'][i].encode('utf-8')).hexdigest()+"_id"
			hashmap[df['name'][i]]=hashed_id
			df['name'][i]=hashed_id


	dehashmap={v: k for k, v in hashmap.items()}
	#with open('symbollist.txt', mode='wt') as fsym:
	#	fsym.write('\n'.join(df['name'].tolist()))


	print("["+str(get_time())+"]: Processing documents...")

	document_count=0
	#substitutes every occuring synonym with their respective labels
	df=substitute_synonyms(df, hashmap)


	df.to_csv("substituted_synonyms.csv", index=None)

	stops=[]
	with open('stopwords_complete.txt', 'r') as f:
		stops=f.read().splitlines()
	word_count=0
	for i in df.index:
		document_count+=1
		

		#here the general processing occurs
		sentences, word_count=corpus_processing(df.iloc[i], word_count, stops, dehashmap)
		content.extend(sentences)
		document_count+=1
		
	print("*************Processed ", word_count, " sentence(s)*************")
	print("Processed ", document_count, " documents(s).")

	#with open('druglist.txt', mode='wt') as fid:
		#fid.write('\n'.join(df['name'].tolist()))
	with open("processed_corpus.csv", "w") as f:
		writer = csv.writer(f)
		writer.writerows(content)

	return remove_single_occuring_words(content)

#core processer. Removes numbers(not in words), stopwords, stemming and tokenize the words
def corpus_processing(dataframe, word_count, stops, dehashmap):
	document=str(dataframe['text']).split('.')[:-1]
	clean_document=[]
	ps=PorterStemmer()


	for sentences in document:

		sentences=re.sub(r"[^a-zA-Z]*[^a-zA-Z\d_]", " ", sentences)

		#doing this again as the above regex sometimes won't remove all numbers
		sentences=re.sub(r"(\b\d*\b)+(\d+\w)", " ", sentences)

		words = sentences.split()
		words = [word for word in words if len(word) > 1 and not word in stops]
		words = [ps.stem(word.lower()) for word in words]
		if not pd.isnull(dataframe['name']):
			words.insert(0, dehashmap[dataframe['name']])
			words.insert(0, dataframe['name'])


		if len(words)>2:
			if len(words)>50:
				remove_single_occuring_words(words)
				clean_document.append([i for i, j in groupby(words)])
			else:
				clean_document.append([i for i, j in groupby(words)])

		word_count+=1
		#####COMMENT OUT THIS IF YOU RUN ANY PYTHON VERSION UNDER 3.3, BUT HOLY SHIT PLEASE UPGRADE ALREADY
		print("*************Processed ", word_count, " sentence(s)*************", end="\r", flush=True)

	return clean_document, word_count


#no idea why this still uses NLTK tokenizer. Could in theory just use split() as we don't care about any other tokens than words
"""def tokenizer(text):
	data = [] 
	print("["+str(datetime.now())+"]: Tokenizing words...")
	tokenize_count=0
	# iterate through each sentence in the file 
	for i in text: 
		temp = [] 
      
		# tokenize the sentence into words 
		for j in word_tokenize(i): 
			temp.append(j.lower())

			tokenize_count+=1
			#####COMMENT OUT THIS IF YOU RUN ANY PYTHON VERSION UNDER 3.3, BUT HOLY SHIT PLEASE UPGRADE ALREADY
			print("*************Tokenized ", tokenize_count, " word(s)*************", end="\r", flush=True)
		
		data.append(temp)

	print("*************Tokenized ", tokenize_count, " word(s)*************")
	print("["+str(datetime.now())+"]: Tokenizing finished.")

	return data"""

#used when processing to limit list sizes to no longer than 50 words
def list_slicer(list, name):
	new_list=[]
	while(len(list)>50):
		new_list.append(list[:50])
		list=list[50:]

	new_list.append(list)

	if not pd.isnull(name):
		for x in range(len(new_list)):
			new_list[x].insert(0, name)
			new_list[x].append(name)
			new_list[x]=[i for i, j in groupby(new_list)]

	return new_list


#trains...obviously
def trainer(corpus, context):
	print("training")
	model = Word2Vec(size=5000, window=10, min_count=2, workers=multiprocessing.cpu_count())
	model.build_vocab(corpus)
	model.train(corpus, total_examples=model.corpus_count, epochs=5000)
	model_name = "MODEL_"+context+".model"
	model.save(model_name)
	return model

#takes all synonyms of a label and replaces it with the label ID
def substitute_synonyms(df, hashmap):
	substitute_count=0

	df['text']=df['text'].str.replace('-','').replace("'", '').replace(" . ", "").replace("_ ", " ")
	df['text']=df['text'].str.lower()
	df['text']=df['text'].replace(to_replace=r"\[(.*?)\]", value="", regex=True)


	for i in df.index:

		synonym_list=str(df['synonyms'][i]).split(', ')

		#here it attempts to compensate for synonyms that are not or only in plural form
		[synonym_list.append(i+'s') for i in synonym_list if i[-1] != 's' and (i+'s') not in synonym_list]
		[synonym_list.append(i[:-1]) for i in synonym_list if i[-1]== 's' and i[:-1] not in synonym_list]

		#removing short synonyms as they might unintentionally occur inside words, like "AP" in "rapid"
		[synonym_list.remove(i) for i in synonym_list if len(i)<3]

		#smart thing to sort the list so that the shortest instances come last so they don't overlap potentially longer instances
		synonym_list.sort(key=len, reverse=True)




		for x in synonym_list:
			x=x.replace('-','').replace("'", '').replace(" . ", "").lower()
			#df['text']=df['text'].replace(to_replace=r"\b%s\b" % x, value= " "+df['name'][i]+" ", regex=True)
			try:
				df['text']=df['text'].replace(to_replace=r"%s" % x , value= " "+df['name'][i]+"  ", regex=True)

				substitute_count+=1
				#####COMMENT OUT THIS IF YOU RUN ANY PYTHON VERSION UNDER 3.3, BUT HOLY SHIT PLEASE UPGRADE ALREADY
				print("Substituted", substitute_count, " time(s).", end="\r", flush=True)
			except:
				pass

	#UNCOMMENT this if you want to revert to unhashed labels
	#df=dehash(df, hashmap)
	return df
#dehashes the label, used mostly in debugging
def dehash(df, hashmap):
	for i in df.index:
		dehash=None
		name=str(df['name'][i])
		for key, value in hashmap.items():
			if value==df['name'][i]:
				dehash=key

		df['text']=df['text'].str.replace(name, dehash)
		df['name'][i]=dehash

	
	return df
#used to check if stopword list works fine
def check_most_common(corpus, length):
	counts = Counter(x for xs in corpus for x in xs)
	return counts.most_common(length)

#used to remove unnecessary words that don't impact the model
def remove_single_occuring_words(corpus):
	counting = Counter(x for xs in corpus for x in xs)
	stopwords=[word for (word, count) in counting.items() if count==1]
	print("["+str(get_time())+"]: Number of words occuring only once: ", len(stopwords))
	print("["+str(get_time())+"]: Removing...")
	return [[word for word in sentence if not word in stopwords] for sentence in corpus]

def get_time():
	return datetime.now().strftime("%H:%M:%S")

if __name__ == '__main__':
	Main()