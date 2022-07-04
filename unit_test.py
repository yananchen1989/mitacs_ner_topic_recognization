#############  ########## CUDA_VISIBLE_DEVICES
import pandas as pd 
import json,random,multiprocessing
import numpy as np 
import datasets,argparse
from utils.process_func import * 
# ds_nerd = datasets.load_dataset('dfki-nlp/few-nerd', "supervised", cache_dir='/scratch/w/wluyliu/yananc/cache')
# # ds_notes = datasets.load_dataset('conll2012_ontonotesv5', "english_v12", cache_dir='/scratch/w/wluyliu/yananc/cache')
# # ds_conll = datasets.load_dataset('conll2003', cache_dir='/scratch/w/wluyliu/yananc/cache')

# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--binomial",
#     type=float,
#     choices=[0.8, 0.3, 0.5, 0.15],
#     default=1
# )
# args = parser.parse_args()

# prepare few nerd dataset
# file_list = {}
# for dsn in ['dev','test','train']:

# import glob
# files = glob.glob("/scratch/w/wluyliu/yananc/fewnerd_augmented/fewnerd_*")

# for ff in files:
#     with open(ff, 'r') as f:
#         file = f.readlines()

#     split_ix = [0] + [i for i in range(len(file)) if file[i] == '\n']

#     with open('/gpfs/fs0/scratch/w/wluyliu/yananc/fewnerd_augmented/{}.json'.format(ff.split('/')[-1]), 'w') as f:
#         ix = 0
#         for i, j in zip(split_ix[0:-1], split_ix[1:]):

#             tokens = file[i:j]
#             dic = {}
#             dic['id'] = ix
#             dic['tokens'] = [ii.strip().split('\t')[0].strip() for ii in tokens if ii!='\n']
#             dic['tags'] = [ii.strip().split('\t')[1].strip() for ii in tokens if ii!='\n']
#             json_string = json.dumps(dic)

#             f.write(json_string+'\n')
#             ix += 1
#     # file_list[dsn] = '/gpfs/fs0/scratch/w/wluyliu/yananc/fewnerd_augmented/{}.json'.format(file.split('/')[-1])
#     print(ff.split('/')[-1])
import argparse,multiprocessing
import logging
import math
import os
import random

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import pandas as pd
import transformers
from accelerate import Accelerator, DistributedType
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)

file_list = {}
for dsn in ['dev','test','train']:
    file_list[dsn] = '/gpfs/fs0/scratch/w/wluyliu/yananc/few_nerd_supervised/{}.json'.format(dsn)
raw_datasets = datasets.load_dataset('json', data_files=file_list, cache_dir='/scratch/w/wluyliu/yananc/cache')
tags_column = 'tags_coarse'

tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2', cache_dir='/scratch/w/wluyliu/yananc/cache', local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

for ent in tags_coarse:
    tokenizer.add_tokens('<{}>'.format(ent))

def gpt_format(example):
    example['text'] = ' '.join(["<{}>{}".format(l, t) for t, l in zip(example['tokens'], example[tags_column])]) + tokenizer.eos_token
    return example  
    

dataset_ix = raw_datasets.map(map_func, 
                batched=False,
                num_proc= multiprocessing.cpu_count() ,
                load_from_cache_file=not True, remove_columns=['tags'],
                desc = "Running ix mapping ==>")

processed_datasets_gpt = dataset_ix.map(gpt_format, 
                batched=False,
                num_proc= multiprocessing.cpu_count() ,
                load_from_cache_file=False, 
                desc = "Running t5 mapping ==>")   



from transformers import pipeline

model_ver = "debugcnt_-1_epoch_6_ppl_2.4175580531029297"
# model_ver = "debugcnt_1024_epoch_12_ppl_3.4826605776680615"
gpt2 = transformers.GPT2LMHeadModel.from_pretrained("/scratch/w/wluyliu/yananc/finetunes/gpt2_fewnerd/{}".format(model_ver))
gpt2.trainable = False
gpt2.config.pad_token_id=50256




gen_nlp  = pipeline("text-generation", model=gpt2, tokenizer=tokenizer, device=0, return_full_text=True)


# ixs = list(range(len(processed_datasets_gpt['train'])))
# random.shuffle(ixs)

print("begin to generate")
infos = []
for example  in processed_datasets_gpt['train']:
     
    prompt = ' '.join(example['text'].replace(tokenizer.pad_token,'').split()[:5])
    result_gpt = gen_nlp(prompt, max_length=256, do_sample=False, temperature=0.5)

    if example['id'] % 100:
        print("generation ==> ", result_gpt[0]['generated_text'].strip())
        print("reference ==> ", example['text'])
        print()

    infos.append((example['id'], result_gpt[0]['generated_text'].strip()))


df = pd.DataFrame(infos, columns=['id','gen_text'])
df.to_csv("/scratch/w/wluyliu/yananc/gpt_gen_fewnerd_nosample.csv", index=False)


























'''

def t5_format(example):
    source_ll = []
    target_ll = []
    length = min(len(example['tokens']), len(tokenizer_t5.additional_special_tokens) )
    mask_binomial = np.random.binomial(size=length, n=1, p = args.binomial)
    for i in range( length ):
        source_ll.append(tokenizer_t5.additional_special_tokens[i] + example['tokens'][i] )
        if mask_binomial[i]:
            target_ll.append(tokenizer_t5.additional_special_tokens[i] + example[tags_column][i] )
        else:
            target_ll.append(tokenizer_t5.additional_special_tokens[i] + example['tokens'][i] )
    example['text1'] = ' '.join(source_ll)
    example['text2'] = ' '.join(target_ll)

    return example


from utils.process_func import * 
dataset_ix = raw_datasets.map(map_func, 
                batched=False,
                num_proc= multiprocessing.cpu_count() ,
                load_from_cache_file=not True, remove_columns=['tags'],
                desc = "Running ix mapping ==>")

processed_datasets_gpt = dataset_ix.map(gpt_format, 
                batched=False,
                num_proc= multiprocessing.cpu_count() ,
                load_from_cache_file=False, 
                desc = "Running t5 mapping ==>")


for split in ['train', 'test']:
    infos = []
    for ix, text in zip(processed_datasets_gpt[split]['id'], processed_datasets_gpt[split]['text_gpt']):
        infos.append((ix, text))

    df = pd.DataFrame(infos, columns=['ix','text'])

    df.to_csv("fewnerd_{}_gpt.csv".format(split), index=False, sep='\t')



tokenizer = AutoTokenizer.from_pretrained('gpt2', cache_dir='/scratch/w/wluyliu/yananc/cache', local_files_only=True)

tokenizer_neo = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B', cache_dir='/scratch/w/wluyliu/yananc/cache', local_files_only=True)





text = "<O>The <organization>Swedish <organization>national <organization>men <organization>'s <organization>ice <organization>hockey"





import transformers
from accelerate import Accelerator, DistributedType
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2', cache_dir='/scratch/w/wluyliu/yananc/cache', local_files_only=True)

tags_coarse = ['O',
 'art',
 'building',
 'event',
 'location',
 'organization',
 'other',
 'person',
 'product']

for ent in tags_coarse:
    tokenizer.add_tokens('<{}>'.format(ent))
tokenizer.convert_tokens_to_ids([tokenizer.eos_token])
tokenizer.pad_token = tokenizer.eos_token
    

sent = "<O> The <organization> Swedish <organization> national <organization> men <organization> 's <organization> ice <organization> hockey <organization> team <O> , <O> affectionately <O> known <O> as <O> `` <organization> Tre <organization> Kronor <O> `` <O> ( <other> English <O> : <organization> Three <organization> Crowns <O> ; <O> the <O> national <O> symbol <O> of <location> Sweden <O> ) <O> , <O> is <O> regarded <O> as <O> one <O> of <O> the <O> best <O> in <O> the <O> world <O> ." + tokenizer.eos_token
tokenizer(sent,  truncation=True, padding='max_length', max_length=16)

def clean_gen_span(span):
    for iden in tokenizer_t5.additional_special_tokens + [tokenizer_t5.unk_token, tokenizer_t5.eos_token, tokenizer_t5.pad_token]:
        span = span.replace(iden, '')
    return span.strip()


processed_datasets_t5_shuffle = processed_datasets_t5.shuffle()


bs = 128
ii = 0 
output_texts = []
while ii <= len(processed_datasets_t5_shuffle['train']):
    text1s = processed_datasets_t5_shuffle['train'][ii:ii+bs]['text2']
    text2s = processed_datasets_t5_shuffle['train'][ii:ii+bs]['text1']

    text2s_ori = []
    for t in text2s:
        text2_decode = tokenizer_t5.decode(tokenizer_t5.encode(t), clean_up_tokenization_spaces=True, skip_special_tokens=True)
        text2s_ori.append(text2_decode)


    inputs = tokenizer_t5(text1s, return_tensors='pt', padding=True, truncation=True)

    output = t5_nerd.generate(input_ids=inputs['input_ids'].to(device), 
                   attention_mask=inputs['attention_mask'].to(device), do_sample=False, max_length=1024,
                   top_p=0.9, top_k=0, temperature=1.2 ) 

    output_decode = tokenizer_t5.batch_decode(output, clean_up_tokenization_spaces=True, skip_special_tokens=True)
    output_decode_ori = tokenizer_t5.batch_decode(output, clean_up_tokenization_spaces=True)

    output_texts.extend([dec.replace('</s>','').replace('<pad>','') for dec in output_decode_ori])
    
    # for l, p in zip(text2s_ori, output_decode): 
    #     print(l)
    #     print(p)
    #     print()

    print(ii, inputs['input_ids'].shape)

    ii += bs
    torch.cuda.empty_cache()

assert len(output_texts) == len(processed_datasets_t5_shuffle['train'])




'''


























