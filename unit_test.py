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
for dsn in ['dev','test','train']
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
            # dic['ner_tags_coarse'] = [ii.split('-')[0] if '-' in ii else 'O' for ii in dic['tags']  ]
            # dic['ner_tags_fine'] = [ii.split() for ii in dic['tags']  ]
            # dic['ner_tags_coarse_ix'] = [tag_ix_coarse[ii] for ii in dic['ner_tags_coarse'] ]
            # dic['ner_tags_fine_ix'] = [tag_ix_map[ii] for ii in dic['tags']  ]
            
            json_string = json.dumps(dic)

            f.write(json_string+'\n')
    file_list[dsn] = '/gpfs/fs0/scratch/w/wluyliu/yananc/few_nerd_supervised/{}.json'.format(dsn)


dataset = load_dataset('json', data_files=file_list)





















tag_map = {}
with open("/home/w/wluyliu/yananc/nlp4quantumpapers/utils/few_nerd_tag_map.tsv", 'r') as f:
    for line in f:
        tokens = line.strip().split('\t')
        tag = tokens[0]
        tag_tokens = tokens[1].split('-')
        if tag_tokens == ['O']:
            continue
        if tag_tokens[1] == 'other':
            tags_ = tag_tokens[0]
        else:
            tags_ = tag_tokens[1]
        tag_map[tag] = tags_


ix = 0
ix_tag_map = {}
with open("/home/w/wluyliu/yananc/nlp4quantumpapers/utils/few_nerd_tag_map.tsv", 'r') as f:
    for line in f:
        tokens = line.strip().split('\t')
        ix_tag_map[ix] = tokens[0]
        ix += 1

tag_ix_map = {v:k for k, v in ix_tag_map.items()}


# tag_ix_fine = {'O':0}
# tag_ix_coarse = {'O':0}

# ix = 1
# for k, w in ix_tag_map.items():
#     if '-' in w:
#         tag_fine = w.split('-')[1]
#         if tag not in tag_ix_fine:
#             tag_ix_fine[tag] = ix
#             ix += 1
#         else:
#             continue

# ix = 1
# for k, w in ix_tag_map.items():
#     if '-' in w:
#         tag = w.split('-')[0]
#         if tag not in tag_ix_coarse:
#             tag_ix_coarse[tag] = ix
#             ix += 1
#         else:
#             continue






def sep_trunk(df_tmp):
    results = []
    for tag in df_tmp['tags'].unique():
        if tag == 'O':
            continue 

        df_tmp_f = df_tmp.loc[df_tmp['tags']==tag]

        list_of_df = [d for _, d in df_tmp_f.groupby(df_tmp_f.index - np.arange(len(df_tmp_f)))]
        mentions = [' '.join(df_tag['tokens'].tolist()) for df_tag in list_of_df]
        results.append((' ; '.join(mentions), tag_map[tag]))
    return results



def get_ft(dsn):
    path = "/scratch/w/wluyliu/yananc/few_nerd_supervised"
    infos = []
    buffer = []
    with open("{}/{}.txt".format(path, dsn), 'r') as f:
        for line in f:
            if line =='\n':
                
                tokens = []
                tags = []
                ents = {}

                for j in buffer:
                    token = j.split('\t')[0]
                    tag = j.split('\t')[1]
                    tokens.append(token)
                    tags.append(tag)
            
                sent = ' '.join(tokens)

                buffer = []
                # print('----------------------------------')
                df_tmp = pd.DataFrame(zip(tokens, tags), columns=['tokens','tags'])

                ner_tag = sep_trunk(df_tmp)
                for ii in ner_tag:
                    infos.append((sent , ii[1], ii[0]))
                
            else:
                buffer.append(line.strip())
    df_ft = pd.DataFrame(infos, columns=['sent', 'tag', 'span'])
    df_ft.drop_duplicates(inplace=True)
    return df_ft

df_ft_train = get_ft('train')
df_ft_dev = get_ft('dev')
df_ft_test = get_ft('test')



df_ft_train_dev = pd.concat([df_ft_train, df_ft_dev])

df_ft_train_dev.sample(frac=1).to_csv("/scratch/w/wluyliu/yananc/df_nerd_train.csv", index=False)
df_ft_test.sample(frac=1).to_csv("/scratch/w/wluyliu/yananc/df_nerd_test.csv", index=False)


from sklearn.model_selection import train_test_split
import random

df = pd.read_csv("/scratch/w/wluyliu/yananc/sentence_level_annotations.csv")

df.drop_duplicates(['sent'], inplace=True)

for ix, row in df.sample(10).iterrows():
    print(row['sent'])
    print(row['tag'])
    print(row['span'])
    print()


tags = [i for i in df['tag'].unique() if i != '###']
df['tag'] = df['tag'].map(lambda x: random.sample(tags, 1)[0] if x=='###' else x)
df['text1'] = df['sent'] + ' ' + df['tag'].map(lambda x: "The {} is ".format(x))
df['text2'] = df['span'].map(lambda x: x.replace(';', ' ; '))

df_train, df_test = train_test_split(df[['text1', 'text2']], test_size=0.1)


df_train.to_csv("/scratch/w/wluyliu/yananc/df_tqi_train.csv", index=False)
df_test.to_csv("/scratch/w/wluyliu/yananc/df_tqi_test.csv", index=False)




from transformers import T5Tokenizer, AutoModelWithLMHead, pipeline
tokenizer_t5 = T5Tokenizer.from_pretrained("t5-base", cache_dir="/scratch/w/wluyliu/yananc/cache", local_files_only=True)
print(tokenizer_t5)

t5_nerd = AutoModelWithLMHead.from_pretrained("/scratch/w/wluyliu/yananc/finetunes/t5_tqi/epoch_6")
gen_nlp  = pipeline("text2text-generation", model=t5_nerd, tokenizer=tokenizer_t5, device=0)

df_ft_test = pd.read_csv("/scratch/w/wluyliu/yananc/df_nerd_test.csv")




for ix, row in df_test.sample(100).iterrows():
    result_t5 = gen_nlp(row['text1'], max_length=64, \
                                            do_sample=False, \
                                            # top_p=0.9, top_k=0, temperature=1.2,\
                                            # repetition_penalty=1.2, num_return_sequences= 8,\
                                            clean_up_tokenization_spaces=True)
    print(row['text1'])
    print('REFER ===>', row['text2'])
    print('GEN ===>', result_t5[0]['generated_text'])
    print()


df = pd.read_csv("sentence_level_annotations.csv")
for ix, row in df.sample(50).iterrows():
    print(row['sent'])
    print(row['tag'], '===>', row['span'])
    # assert row['span'] in row['sent']
    print()










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












