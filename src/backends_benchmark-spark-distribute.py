import sys
import re
from contextlib import closing, contextmanager
import time
import os
import sys
sys.path.append("/home/ubuntu/Wordbatch")

# test whether path works
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from wordbatch.extractors.extractors import WordHash, WordBag
from wordbatch.pipelines import WordBatch, Apply, ApplyBatch
from wordbatch.transformers import Tokenizer, Dictionary
from wordbatch.batcher import Batcher
import os
import json
from sklearn.feature_extraction.text import HashingVectorizer
import warnings
import pandas as pd
import multiprocessing

# http://sifaka.cs.uiuc.edu/~wang296/Data/LARA/TripAdvisor/
tripadvisor_dir= "/home/ubuntu/Wordbatch/data/json"
# sys.path.append(os.path.abspath(".."))

import json
#from pyspark import SparkConf, SparkContext
# Configure below to allow Dask / Spark
# scheduler_ip= "169.254.93.14"
# from dask.distributed import Client
# #dask-scheduler --host 169.254.93.14
# #dask-worker 169.254.93.14:8786 --nprocs 16
# dask_client = Client(scheduler_ip+":8786")
#
from pyspark import SparkContext, SparkConf

# os.environ['PYSPARK_PYTHON'] = '/home/USERNAME/anaconda3/envs/ENV_NAME/bin/python'

conf= SparkConf().setAll([('spark.worker.memory', '8g'), ('spark.executor.memory', '8g'), ('spark.driver.memory', '8g')]).setMaster("spark://10.24.223.190:7077").set("spark.executorEnv.PYTHONPATH", "/home/ubuntu/Wordbatch:$PYTHONPATH").set("spark.eventLog.enabled", "false").set("spark.driver.maxResultSize", "8g")
spark_context = SparkContext(conf=conf)

@contextmanager
def timer(name):
	t0 = time.time()
	yield
	print(name + " done in " + str(time.time() - t0) + "s")

texts = []
for jsonfile in os.listdir(tripadvisor_dir):
	with open(tripadvisor_dir + "/" + jsonfile, 'r') as inputfile:
		for line in inputfile:
			try:
				line = json.loads(line.strip())
			except:
				continue

			if 'text' in line:
				texts.append(line['text'])
# 	pd.to_pickle(texts, "tripadvisor_data.pkl")
# else:
# 	texts= pd.read_pickle("tripadvisor_data.pkl")
print(len(texts))
non_alphanums = re.compile('[\W+]')
nums_re= re.compile("\W*[0-9]+\W*")
triples_re= re.compile(r"(\w)\1{2,}")
trash_re= [re.compile("<[^>]*>"), re.compile("[^a-z0-9' -]+"), re.compile(" [.0-9'-]+ "), re.compile("[-']{2,}"),
		   re.compile(" '"),re.compile("  +")]
from nltk.stem.porter import PorterStemmer
stemmer= PorterStemmer()

def normalize_text(text):
	text= text.lower()
	text= nums_re.sub(" NUM ", text)
	text= " ".join([word for word in non_alphanums.sub(" ",text).strip().split() if len(word)>1])
	return text

print(len(texts))
backends= [
	# ['serial', ""],
	# ['multiprocessing', ""],
	# ['loky', ""],
	# ['dask', dask_client], # Uncomment once configured
	['spark', spark_context], # Uncomment once configured
	# ['ray', ray]
]

tasks= [
	"ApplyBatch",
]

data_sizes= [10, 40000, 80000, 160000, 320000, 640000]

for task in tasks:
	for data_size in data_sizes:
		texts_chunk = texts[:data_size]
		print("Task:", task, "Data size:", data_size)
		for backend in backends:
			batcher = Batcher(procs=multiprocessing.cpu_count(), minibatch_size=1000, backend=backend[0], backend_handle=backend[1])
			try:
				with timer("Completed: ["+task+","+str(len(texts_chunk))+","+backend[0]+"]"), warnings.catch_warnings():
					warnings.simplefilter("ignore")
					if task=="ApplyBatch":
						hv = HashingVectorizer(decode_error='ignore', n_features=2 ** 25, preprocessor=normalize_text,
											   ngram_range=(1, 2), norm='l2')
						t= ApplyBatch(hv.transform, batcher=batcher).transform(texts_chunk)
						print(t.shape, t.data[:5])

					if task=="WordBag":
						wb = WordBatch(normalize_text=normalize_text,
									   dictionary=Dictionary(min_df=10, max_words=1000000, verbose=0),
									   tokenizer= Tokenizer(spellcor_count=2, spellcor_dist=2, stemmer= stemmer),
									   extractor=WordBag(hash_ngrams=0, norm= 'l2', tf= 'binary', idf= 50.0),
									   batcher= batcher,
									   verbose= 0)
						t = wb.fit_transform(texts_chunk)
						print(t.shape, t.data[:5])
			except:
				print("Failed: ["+task+","+str(len(texts_chunk))+","+backend[0]+"]")
		print("")