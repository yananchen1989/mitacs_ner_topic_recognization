# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 20:39:37 2022

@author: Doug
"""

################
# LIBRARIES
################
import time
from timeit import default_timer as timer

import pandas as pd
import numpy as np
import json

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

import torch
import numba
from numba import jit, cuda

################
# ENSURE TORCH WORKS
################
print("Is Torch correctly detecting the GPU/CUDA on your system?")
print(torch.__version__)
print(torch.zeros(1).cuda())
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

################
# QUICK TEST
################
print("Begin Quick Test")
# To run on CPU
def func(a):
    for i in range(10000000):
        a[i]+= 1

# To run on GPU
# Set Numba to choose when/how to optimize
# More info - https://numba.pydata.org/numba-doc/latest/user/jit.html
# Also - https://numba.pydata.org/numba-doc/latest/user/5minguide.html
@jit

def func2(x):
    return x+1

if __name__=="__main__":
    n = 10000000
    a = np.ones(n, dtype = np.float64)
    start = timer()
    func(a)
    print("Without GPU:", timer()-start)
    start = timer()
    func2(a)
    numba.cuda.profile_stop()
    print("With GPU:", timer()-start)
    
################
# LONG TEST
################
print("Begin Long Test - should be run overnight (~4-6h for CPU if not more)")
start_time_fxn = time.time()

target_categories = ['cond-mat.supr-con', 'math.QA', 'quant-ph', 'stat.CO']

def clean_text(sent):
    tokens = sent.replace('\n',' ').strip().split()
    tokens_clean = [ii for ii in tokens if  "$" not in ii and ii and '\\' not in ii]
    return ' '.join(tokens_clean)

def make_df():
    infos = []
    # include your local path to kaggle file
    with open('S:/Applications/Coding/Projects/The Quantum Insider/Run Files Here/arxiv-metadata-oai-snapshot_2.json', 'r') as f: # /Users/yanan/Downloads/
        for line in f:
            js = json.loads(line)
            if (js['categories'] == 'quant-ph') :
                js['abstract'] = clean_text(js['abstract'])
                infos.append(js)

    df = pd.DataFrame(infos) # 78160
    #df['abstract'] = df['abstract'].map(lambda x: x.lower())
    df.drop_duplicates(['abstract'], inplace=True)
    df['yymm'] = pd.to_datetime(df['update_date'].map(lambda x: '-'.join(x.split('-')[:2] )))
    return df 

df = make_df()
print(df.sample(1)["abstract"].tolist()[0])

end_time_fxn = time.time()
time_to_run_fxn = end_time_fxn - start_time_fxn

################
#RUN GPU VS CPU VERSIONS (Run overnight)
################

print("GPU VERSION BEGIN")
start_time_gpu = time.time()
embedding_model = SentenceTransformer("all-mpnet-base-v2", device='cuda')
embeddings = embedding_model.encode(df.sample(248)['abstract'].tolist(), show_progress_bar=True, batch_size=64)
topic_model = BERTopic(embedding_model=embedding_model, verbose=True, min_topic_size=20)
topics, probs = topic_model.fit_transform(df['abstract'].tolist())
end_time_gpu = time.time()
time_to_run_gpu = end_time_gpu - start_time_gpu

print("CPU VERSION BEGIN")
start_time_cpu = time.time()
embedding_model = SentenceTransformer("all-mpnet-base-v2", device='cpu')
embeddings = embedding_model.encode(df.sample(248)['abstract'].tolist(), show_progress_bar=True, batch_size=64)
topic_model = BERTopic(embedding_model=embedding_model, verbose=True, min_topic_size=20)
topics, probs = topic_model.fit_transform(df['abstract'].tolist())
end_time_cpu = time.time()
time_to_run_cpu = end_time_cpu - start_time_cpu

print('FXN TIME: %s seconds' % (time_to_run_fxn))
print('CPU TIME: %s hours' % (time_to_run_cpu/60/60))
print('GPU TIME: %s minutes' % (time_to_run_gpu/60))