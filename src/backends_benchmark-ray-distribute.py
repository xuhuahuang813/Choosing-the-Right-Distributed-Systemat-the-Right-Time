import sys
import re
from contextlib import closing, contextmanager
import time
import os
import sys
import json
import warnings
import pandas as pd
import multiprocessing
from sklearn.feature_extraction.text import HashingVectorizer
from nltk.stem.porter import PorterStemmer
import ray

ray.init(address='10.24.223.190:6379')

@ray.remote
def apply_batch_transform(texts_chunk):
    import socket
    hostname = socket.gethostname()
    print(f"Processing on node: {hostname}")
    hv = HashingVectorizer(
        decode_error='ignore',
        n_features=2 ** 25,
        preprocessor=normalize_text,
        ngram_range=(1, 2),
        norm='l2'
    )
    return hv.transform(texts_chunk)

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(name + " done in " + str(time.time() - t0) + "s")

tripadvisor_dir = "/home/ubuntu/Wordbatch/data/json"

texts = []
for jsonfile in os.listdir(tripadvisor_dir):
    with open(os.path.join(tripadvisor_dir, jsonfile), 'r') as inputfile:
        for line in inputfile:
            try:
                line = json.loads(line.strip())
            except:
                continue

            if 'text' in line:
                texts.append(line['text'])

print(len(texts))
non_alphanums = re.compile('[\W+]')
nums_re = re.compile("\W*[0-9]+\W*")
triples_re = re.compile(r"(\w)\1{2,}")
trash_re = [
    re.compile("<[^>]*>"), re.compile("[^a-z0-9' -]+"), re.compile(" [.0-9'-]+ "),
    re.compile("[-']{2,}"), re.compile(" '"), re.compile("  +")
]
stemmer = PorterStemmer()

def normalize_text(text):
    text = text.lower()
    text = nums_re.sub(" NUM ", text)
    text = " ".join([word for word in non_alphanums.sub(" ", text).strip().split() if len(word) > 1])
    return text

print(len(texts))

tasks = ["ApplyBatch"]
data_sizes = [10, 40000, 80000, 160000, 320000, 640000]

for task in tasks:
    for data_size in data_sizes:
        texts_chunk = texts[:data_size]
        print("Task:", task, "Data size:", data_size)
        if task == "ApplyApplyBatchBatch":
            with timer(f"Completed: [{task},{len(texts_chunk)},ray]"):
                future = apply_batch_transform.remote(texts_chunk)
                t = ray.get(future)
                print(t.shape, t.data[:5])
        print("")
