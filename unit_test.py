#############  ########## CUDA_VISIBLE_DEVICES
import pandas as pd 
import json,random
import numpy as np 
import datasets
# ds_nerd = datasets.load_dataset('dfki-nlp/few-nerd', "supervised", cache_dir='/scratch/w/wluyliu/yananc/cache')
# # ds_notes = datasets.load_dataset('conll2012_ontonotesv5', "english_v12", cache_dir='/scratch/w/wluyliu/yananc/cache')
# # ds_conll = datasets.load_dataset('conll2003', cache_dir='/scratch/w/wluyliu/yananc/cache')




# prepare few nerd dataset
# file_list = {}
# for dsn in ['dev','test','train']:


#     with open("/scratch/w/wluyliu/yananc/few_nerd_supervised/{}.txt".format(dsn), 'r') as f:
#         file = f.readlines()

#     split_ix = [0] + [i for i in range(len(file)) if file[i] == '\n']

#     with open('./few_nerd_supervised/{}.json'.format(dsn), 'w') as f:
#         ix = 0
#         for i, j in zip(split_ix[0:-1], split_ix[1:]):

#             tokens = file[i:j]
#             dic = {}
#             dic['id'] = ix
#             dic['tokens'] = [ii.strip().split('\t')[0] for ii in tokens if ii!='\n']
#             dic['tags'] = [ii.strip().split('\t')[1] for ii in tokens if ii!='\n']
#             json_string = json.dumps(dic)

#             f.write(json_string+'\n')
#             ix += 1
#     file_list[dsn] = '/gpfs/fs0/scratch/w/wluyliu/yananc/few_nerd_supervised/{}.json'.format(dsn)
gpu = 0


file_list = {}
for dsn in ['dev','test','train']:
    file_list[dsn] = '/gpfs/fs0/scratch/w/wluyliu/yananc/few_nerd_supervised/{}.json'.format(dsn)
raw_datasets = datasets.load_dataset('json', data_files=file_list, cache_dir='/scratch/w/wluyliu/yananc/cache')




tags_column = 'tags_coarse'

from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline
import datasets,multiprocessing
tokenizer_t5 = AutoTokenizer.from_pretrained("t5-base", cache_dir="/scratch/w/wluyliu/yananc/cache", local_files_only=True)
print(tokenizer_t5)

t5_nerd = AutoModelWithLMHead.from_pretrained("/scratch/w/wluyliu/yananc/finetunes/t5_nerd_da_coarse/epoch_9")
gen_nlp  = pipeline("text2text-generation", model=t5_nerd, tokenizer=tokenizer_t5, device=gpu)
 

def t5_format(example):
    source_ll = []
    target_ll = []
    for i in range( min(len(example['tokens']), len(tokenizer_t5.additional_special_tokens) )):
        source_ll.append(tokenizer_t5.additional_special_tokens[i] + example['tokens'][i] )
        target_ll.append(tokenizer_t5.additional_special_tokens[i] + example[tags_column][i] )

    example['text1'] = ' '.join(source_ll)
    example['text2'] = ' '.join(target_ll)

    return example

from utils.process_func import * 
dataset_ix = raw_datasets.map(map_func, 
                batched=False,
                num_proc= multiprocessing.cpu_count() ,
                load_from_cache_file=not True, remove_columns=['tags'],
                desc = "Running ix mapping ==>")

processed_datasets_t5 = dataset_ix.map(t5_format, 
                batched=False,
                num_proc= multiprocessing.cpu_count() ,
                load_from_cache_file=False, 
                desc = "Running t5 mapping ==>")

# processed_datasets_t5.save_to_disk("/scratch/w/wluyliu/yananc/few_nerd_supervised")





import torch
device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")

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
    # text2_ids = tokenizer_t5.encode(text2)
    # text2_decode = tokenizer_t5.decode(tokenizer_t5.encode(text2), clean_up_tokenization_spaces=True, skip_special_tokens=True)

    inputs = tokenizer_t5(text1s, return_tensors='pt', padding=True, truncation=True)

    output = t5_nerd.generate(input_ids=inputs['input_ids'].to(device), 
                   attention_mask=inputs['attention_mask'].to(device), do_sample=False, max_length=1024,
                   top_p=0.9, top_k=0, temperature=1.2 )

    # output_decode = tokenizer_t5.decode(output, clean_up_tokenization_spaces=True, skip_special_tokens=True)

    output_decode_ori = tokenizer_t5.batch_decode(output, clean_up_tokenization_spaces=True)

    output_texts.extend([dec.replace('</s>','').replace('<pad>','') for dec in output_decode_ori])
    

    print(ii, inputs['input_ids'].shape)

    ii += bs
    torch.cuda.empty_cache()




with open('/scratch/w/wluyliu/yananc/few_nerd_supervised/da_coarse.json', 'w') as f:

    for ii, text1, text2, text_gen, tags in zip(processed_datasets_t5_shuffle['train'][:len(output_texts)]['id'], \
                                      processed_datasets_t5_shuffle['train'][:len(output_texts)]['text2'], \
                                      processed_datasets_t5_shuffle['train'][:len(output_texts)]['text1'], \
                                      output_texts, \
                                      processed_datasets_t5_shuffle['train'][:len(output_texts)][tags_column]):
        idens = []
        ix = 0
        for tag, i in zip(text1.split(), text2.split()):
            iden = "<extra_id_{}>".format(ix)
            iden_ = "<extra_id_{}>".format(ix+1)

            if iden in text_gen:
                span = text_gen.split(iden)[1].split(iden_)[0]  
                span = clean_gen_span(span)
                if not span:
                    span = tokenizer_t5.unk_token
            else:
                span = tokenizer_t5.unk_token

            print(tag.replace(iden, ''), '==>', i.replace(iden, ''), '--->', span)
            idens.append(span)
            ix += 1
        print(idens)
        dic = {}
        dic['id'] = ii
        dic['tokens'] = idens
        dic[tags_column] = tags[:len(idens)]
        assert len(dic[tags_column]) == len(dic['tokens'])
        print()

        json_string = json.dumps(dic)
        f.write(json_string+'\n')
        print('\n\n') 



# file_list = {}
# file_list['da_coarse'] = '/gpfs/fs0/scratch/w/wluyliu/yananc/few_nerd_supervised/da_coarse.json'
# raw_datasets = datasets.load_dataset('json', data_files=file_list, cache_dir='/scratch/w/wluyliu/yananc/cache')
















    































