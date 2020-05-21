import time
import json
import gensim
import numpy as np
import pandas as pd

def Main():
	start=time.time()
	###DEFINE MODEL NAME HERE
	model = gensim.models.Word2Vec.load("MODELNAME.model")

	#arbitrary labels of the dataset to be created
	#input_data=pd.read_csv("symbollist.txt", header=None, index_col=None)
	column_list = []
	index_list = []


	#dimensions of the vectors and the specified top n most correlated
	dimensions=5000
	length=5

	hashmap={}
	###DEFINE HASHMAP NAME HERE
	with open('hashmap_.json', 'r') as f:
		hashmap=json.loads(f.read())

	#initialize lists containing the vectors extracted from the model
	list_of_column_vectors=[]
	list_of_index_vectors=[]

	index_remove=[]
	column_remove=[]
	#add vectors to the lists
	for id in column_list:
		try:
			temp=model.wv[hashmap[id]]
			list_of_column_vectors.append(temp)
		except:
			column_remove.append(id)

	for sym in index_list:
		try:
			temp2=model.wv[hashmap[sym]]
			list_of_index_vectors.append(temp2)
		except:
			index_remove.append(sym)

	column_list = [x for x in column_list if x not in column_remove]
	index_list = [x for x in index_list if x not in index_remove]


	#print("Index_list: ", len(index_list))
	#print("Index_Vector: ", len(list_of_index_vectors))
	#print("Column_list: ", len(column_list))
	#print("Column_Vector: ", len(list_of_column_vectors))

	#initialize matrix
	matrix = [[0 for x in range(len(index_list))] for y in range(len(column_list))]

	#perform a pearson correlation between each vector and add the result to the matrix in its respective position
	for j, index_word in enumerate(index_list):
		for i, column_word in enumerate(column_list):
			#print(j, i)
			temp=np.corrcoef(list_of_column_vectors[i], list_of_index_vectors[j])[1, 0]
			#temp=np.linalg.norm(list_of_column_vectors[i]-list_of_index_vectors[j])
			#print("Correlation between ", column_word.upper(), "and ", index_word.upper(), ": ", temp)
			if(temp>0.05):
				matrix[i][j]=temp
			else:
				matrix[i][j]=0
	matrix=np.transpose(matrix)
	#save matrix to pandas dataframe
	print("Saving dataset to Pandas dataframe...")
	df=pd.DataFrame(matrix, columns=column_list, index=index_list)


	#find the top n correlated vectors and list them by their label
	"""print("Finding top n for each company...")
	top_matrix=[[0 for x in range(length)] for y in range(len(column_list))]
	for i, column in enumerate(df.columns):
		top=df.nlargest(length, column)
		for l in range(length):
			top_matrix[i][l]=top.index.values[l]

	#transpose the matrix for easier reading
	print("Transponsing matrix")
	top_matrix=list(map(list, zip(*top_matrix)))

	#save as pandas dataframe
	print("Saving dataset as Pandas dataframe")
	df_top=pd.DataFrame(top_matrix, columns=column_list)"""


	#save to .csv files

	timestring= time.strftime("%Y%m%d-%H%M%S")
	print("Saving files to .csv")

	###REPLACE "A" WITH ROWS AND "B" WITH COLUMNS
	df.to_csv(str(len(index_list))+"_X_x_"+str(len(column_list))+"_B_"+timestring+".csv", index=True)
	#df_top.to_csv("top_matrix.csv", index=True)
	end=time.time()
	print("Done. Elapsed time ", end-start)

Main()