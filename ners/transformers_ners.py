import json,random,string
import pandas as pd 

# huggingface transformers
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline


gpu = 0
# conll2003
# tokenizer_bert = AutoTokenizer.from_pretrained("dslim/bert-base-NER", cache_dir='/scratch/w/wluyliu/yananc/cache')
# model_bert = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER", cache_dir='/scratch/w/wluyliu/yananc/cache')

# conll2003
tokenizer_roberta = AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english", cache_dir='/scratch/w/wluyliu/yananc/cache')
model_roberta = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english", cache_dir='/scratch/w/wluyliu/yananc/cache')
nlp_roberta = pipeline("ner", model=model_roberta, tokenizer=tokenizer_roberta, \
                    aggregation_strategy="simple", device=gpu)


tokenizer_roberta_nerd_fine = AutoTokenizer.from_pretrained("/scratch/w/wluyliu/yananc/finetunes/roberta_nerd_fine")
model_roberta_nerd_fine = AutoModelForTokenClassification.from_pretrained("/scratch/w/wluyliu/yananc/finetunes/roberta_nerd_fine")
nlp_roberta_nerd_fine = pipeline("ner", model=model_roberta_nerd_fine, tokenizer=tokenizer_roberta_nerd_fine,\
                     aggregation_strategy="simple", device=gpu)


tokenizer_roberta_nerd_coarse = AutoTokenizer.from_pretrained("/scratch/w/wluyliu/yananc/finetunes/roberta_nerd_coarse")
model_roberta_nerd_coarse = AutoModelForTokenClassification.from_pretrained("/scratch/w/wluyliu/yananc/finetunes/roberta_nerd_coarse")
nlp_roberta_nerd_coarse = pipeline("ner", model=model_roberta_nerd_coarse, tokenizer=tokenizer_roberta_nerd_coarse,\
                    aggregation_strategy="simple", device=gpu)


def check_entity_ft(sent, nlp_roberta, fmark):
    ner_results = nlp_roberta(sent)
    if not ner_results:
        return []
    infos = []
    for ii in ner_results:
        if ii['word'].strip() in string.punctuation:
            continue
        infos.append((ii['entity_group'], ii['word'].strip(), fmark))
    return infos


# spacy
# https://spacy.io/models/en#en_core_web_lg
# python -m spacy download en_core_web_lg
# python -m spacy download en_core_web_trf
import spacy
#ner_spacy_model = spacy.load('en_core_web_lg', disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
nlp_split_sents = spacy.load('en_core_web_sm')


# flair 
from flair.data import Sentence
from flair.models import SequenceTagger
import torch, flair, string

if gpu >= 0:
    flair.device = torch.device('cuda:{}'.format(gpu))
else:
    flair.device = torch.device('cpu')

tagger_conll = SequenceTagger.load("flair/ner-english-large")
tagger_ontonotes = SequenceTagger.load("flair/ner-english-ontonotes-large")

def check_entity_flair(content, tagger, fmark):
    sentence = Sentence(content)
    tagger.predict(sentence)
    # ii.start_position, ii.end_position
    ners = sentence.get_spans('ner')
    infos = []
    for ii in sentence.get_spans('ner'):
        if (ii.tag, ii.text) in infos:
            continue
        if ii.text.strip() in string.punctuation:
            continue
        infos.append((ii.tag, ii.text.strip(), fmark))
    return infos




############## main ########################


with open('/home/w/wluyliu/yananc/nlp4quantumpapers/articles_full.json', 'r') as f:
    jxml = json.load(f)

tag_map = {'PERSON':'PER', 'ORGANIZATION':'PER', 'LOCATION':'LOC', 'GPE':'LOC'}
pick_from_nerd = ['EDUCATION','COMPANY','SOFTWARE','Government'.upper() ]

def ner_tag(content):
    sents = [i.text.replace('\n','').strip() for i in nlp_split_sents(content).sents]

    infos = []
    for sent in sents:
        if not sent:
            continue
        flair_ontonotes_ner_results = check_entity_flair(sent, tagger_ontonotes, 'flair_onto')
        infos.extend(flair_ontonotes_ner_results)

        flair_conll_ner_results = check_entity_flair(sent, tagger_conll, 'flair_conll')
        infos.extend(flair_conll_ner_results)

        ft_ner_ori = check_entity_ft(sent, nlp_roberta, 'ft_ori')
        infos.extend(ft_ner_ori)

        ft_ner_nerd_coarse = check_entity_ft(sent, nlp_roberta_nerd_coarse, 'nerd_coarse') 
        infos.extend(ft_ner_nerd_coarse)

        ft_ner_nerd_fine = check_entity_ft(sent, nlp_roberta_nerd_fine, 'nerd_fine') 
        infos.extend(ft_ner_nerd_fine)

        print(sent) 

    
    df = pd.DataFrame(infos, columns=['ner','span', 'fmark'])
    df['ner'] = df['ner'].map(lambda x: x.upper())
    df['ner'] = df['ner'].map(lambda x: tag_map[x] if x in tag_map else x )

    df = df.loc[df['ner'].str()]
    df.drop_duplicates(['ner','span', 'fmark'], inplace=True)

    df_pretrain = df.loc[df['fmark'].isin(['flair_onto', 'flair_conll', 'ft_ori'])]
    df_nerd = df.loc[df['fmark'].isin(['nerd_coarse', 'nerd_fine'])]
    df_nerd = df_nerd.loc[df_nerd['ner'].isin(pick_from_nerd)]

    df_pretrain.drop_duplicates(['ner','span'], inplace=True)
    df_nerd.drop_duplicates(['ner','span'], inplace=True)
    

    df_pretrain.sort_values(by=['ner'], ascending=True, inplace=True)
    df_nerd.sort_values(by=['ner'], ascending=True, inplace=True)
    return df_pretrain, df_nerd

content = random.sample(jxml, 1)[0]['post_content']
df_pretrain, df_nerd = ner_tag(content)


for ix, row in df_pretrain.iterrows():
    print(row['ner'], '\t', row['span'])


print()
for ix, row in df_nerd.iterrows():
    print(row['ner'], '\t', row['span'])



labels_candidates = ['company', 'investor', 'university','Venture Capital Investor', 'academic organization', 
                        'goverment organization']



















