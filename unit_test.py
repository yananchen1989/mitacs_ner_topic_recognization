#############  ########## CUDA_VISIBLE_DEVICES
import pandas as pd 
import json,random
import numpy as np 
import datasets
# ds_nerd = datasets.load_dataset('dfki-nlp/few-nerd', "supervised", cache_dir='/scratch/w/wluyliu/yananc/cache')
# # ds_notes = datasets.load_dataset('conll2012_ontonotesv5', "english_v12", cache_dir='/scratch/w/wluyliu/yananc/cache')
# # ds_conll = datasets.load_dataset('conll2003', cache_dir='/scratch/w/wluyliu/yananc/cache')


# for ix in range(len(ds_nerd['train'])):
#     sent = ds_nerd['train'][ix]['tokens']
#     label_coarse = ds_nerd['train'][ix]['ner_tags']
#     label_fine = ds_nerd['train'][ix]['fine_ner_tags']
#     break 

# for i,j in zip(ds_nerd['train'][ix]['tokens'], ds_nerd['train'][ix]['ner_tags']):
#     print(i, j)




file_list = {}
for dsn in ['dev','test','train']:
    with open("/scratch/w/wluyliu/yananc/few_nerd_supervised/{}.txt".format(dsn), 'r') as f:
        file = f.readlines()

    split_ix = [0] + [i for i in range(len(file)) if file[i] == '\n']

    with open('./few_nerd_supervised/{}.json'.format(dsn), 'w') as f:
        ix = 0
        for i, j in zip(split_ix[0:-1], split_ix[1:]):
            # print(ix)
            tokens = file[i:j]
            dic = {}
            dic['id'] = ix
            dic['tokens'] = [ii.strip().split('\t')[0] for ii in tokens if ii!='\n']
            dic['tags'] = [ii.strip().split('\t')[1] for ii in tokens if ii!='\n']
            json_string = json.dumps(dic)

            f.write(json_string+'\n')
    file_list[dsn] = '/gpfs/fs0/scratch/w/wluyliu/yananc/few_nerd_supervised/{}.json'.format(dsn)



# for each sample, write to each row




from transformers import T5Tokenizer, AutoModelWithLMHead, pipeline
import datasets
tokenizer_t5 = T5Tokenizer.from_pretrained("t5-base", cache_dir="/scratch/w/wluyliu/yananc/cache", local_files_only=True)
print(tokenizer_t5)

t5_nerd = AutoModelWithLMHead.from_pretrained("/scratch/w/wluyliu/yananc/finetunes_ner/t5_nerd/epoch_6")
gen_nlp  = pipeline("text2text-generation", model=t5_nerd, tokenizer=tokenizer_t5, device=0)




file_list = {}
for dsn in ['dev','test','train']:
    file_list[dsn] = '/gpfs/fs0/scratch/w/wluyliu/yananc/few_nerd_supervised/{}.json'.format(dsn)
raw_datasets = datasets.load_dataset('json', data_files=file_list, cache_dir='/scratch/w/wluyliu/yananc/cache')

import multiprocessing
tags_column = 'tags'
def t5_format(example):
    source_ll = []
    target_ll = []
    for i in range( min(len(example['tokens']), len(tokenizer_t5.additional_special_tokens) )):
        source_ll.append(tokenizer_t5.additional_special_tokens[i] + example['tokens'][i] )
        target_ll.append(tokenizer_t5.additional_special_tokens[i] + example[tags_column][i] )

    example['text1'] = ' '.join(source_ll)
    example['text2'] = ' '.join(target_ll)

    return example

processed_datasets_t5 = raw_datasets.map(t5_format, 
                batched=False,
                num_proc= multiprocessing.cpu_count() ,
                load_from_cache_file= False, 
                desc = "Running t5 mapping ==>")













import seqeval
y_true = []
y_pred = [] 
for ii in range(len(processed_datasets_t5['test'])):

    text1 = processed_datasets_t5['test'][ii]['text1']
    text2 = processed_datasets_t5['test'][ii]['text2']


    result_t5 = gen_nlp(text1, max_length=512, \
                                            do_sample=False, \
                                            # top_p=0.9, top_k=0, temperature=1.2,\
                                            # repetition_penalty=1.2, num_return_sequences= 8,\
                                            clean_up_tokenization_spaces=True)


    ref = tokenizer_t5.decode(tokenizer_t5.encode(text2), clean_up_tokenization_spaces=True, skip_special_tokens=True)
    gen = result_t5[0]['generated_text']

    if len(ref.split()) != len(gen.split()):
        break 
    



































import spacy
nlp_split_sents =  spacy.load('en_core_web_sm')




sent = "Honeywell  officials said the new partnership with Microsofts  Azure Quantum \
 also gives organizations a new way to familiarize themselves with quantum computing. The company is"


sent = "The  U.S. Department of Energy  ( DOE ) recently announced plans to provide \
    $30 million for Quantum Information Science (QIS) research that helps scientists understand \
    how nature works on an extremely small scale 100,000 times smaller than the diameter of a human hair. \
    The investor is "


sent = "Seeing a gap in the market,  UGhent  spinoff \
    called QustomDotan  advanced materials startup that manufactures on-chip grade \
    and cadmium-free quantum dotsis ready to build the next generation of applications using its novel IP. \
    The University is "


sample = random.sample(jxml, 1)
sents = [i.text.replace('\n','').strip() for i in nlp_split_sents(sample[0]['post_content']).sents]


sent = "Startup BeitTech from Poland, Swiss outfits ID Quantique and RQuanTech, German-based InfiniQuant,\
             and QTech from the Netherlands are there to be taken seriously in the QC industry. \
             The company is "


sent = "This research was funded by ORNL’s Laboratory Directed Research and Development program.\
       The Laboratory is "


sent = "In collaboration with scientists at the Air Force Research Laboratory, they are now developing tiny,\
        specialized silicon chips similar to those common in microelectronics in pursuit of \
        even better photonic performance. The Laboratory is  "


sent = "Conventional computer “bits” have a value of either 0 or 1, but quantum bits, \
            called “qubits,” can exist in a superposition of quantum states labeled 0 and 1. \
            The company is "

result_t5 = gen_nlp(sent, max_length=64, \
                                        do_sample=True, top_p=0.9, top_k=0, temperature=1.2,\
                                        repetition_penalty=1.2, num_return_sequences= 32,\
                                        clean_up_tokenization_spaces=True)

result_t5 = gen_nlp(sent, max_length=64, \
                                        do_sample=False, 
                                        clean_up_tokenization_spaces=True)
for ii in result_t5:
    print(ii['generated_text'])











    





















import spacy
ner_spacy_model = spacy.load('en_core_web_lg', disable=["tok2vec", "tagger", "attribute_ruler", "lemmatizer"])


import requests
import re
from collections import Counter

content = '''
The West lost self-confidence — and both Russian and Chinese leaders rubbed it in, putting out the word that these chaotic democratic systems were a spent force.
And then a totally unexpected thing happened: Russia and China each overreached.
Vladimir Putin invaded Ukraine and, to his surprise, invited an indirect war with NATO and the West. China insisted that it was smart enough to have its own local solution to a pandemic, leaving millions of Chinese underprotected or unprotected and, in effect,
'''

article = ner_spacy_model(content)
labels = [x.label_ for x in article.ents]
Counter(labels)


sentences = [x for x in article.sents]



texts = [
    "Net income was $9.4 million compared to the prior year of $2.7 million.",
    "Revenue exceeded twelve billion dollars, with a loss of $1b.",
]

nlp = spacy.load("en_core_web_sm")
for doc in nlp.pipe(texts, disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"]):
    # Do something with the doc here
    print([(ent.text, ent.label_) for ent in doc.ents])



from spacy import displacy
displacy.render(ner_spacy_model(content.strip()), jupyter=True, style='ent')

text = "When Sebastian Thrun started working on self-driving cars at Google in 2007, few people outside of the company took him seriously."

nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
html = displacy.render(doc, style="ent", page=True)


with open("ner_spacy_test.html", "w") as ff:
    ff.write(html+'------------\n') 












