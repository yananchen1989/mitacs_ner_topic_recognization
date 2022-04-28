import json,random,string
import pandas as pd 

# huggingface transformers
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline


gpu = 1
# conll2003
# tokenizer_bert = AutoTokenizer.from_pretrained("dslim/bert-base-NER", cache_dir='/scratch/w/wluyliu/yananc/cache')
# model_bert = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER", cache_dir='/scratch/w/wluyliu/yananc/cache')

# load off-the-shelf model : roberta-large fine-tuned on conll2003
tokenizer_roberta = AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english", cache_dir='/scratch/w/wluyliu/yananc/cache')
model_roberta = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english", cache_dir='/scratch/w/wluyliu/yananc/cache')
nlp_roberta = pipeline("ner", model=model_roberta, tokenizer=tokenizer_roberta, \
                    aggregation_strategy="simple", device=gpu)

# load off-the-shelf model : roberta-large fine-tuned on nerd-fine-grained
tokenizer_roberta_nerd_fine = AutoTokenizer.from_pretrained("/scratch/w/wluyliu/yananc/finetunes/roberta_nerd_fine")
model_roberta_nerd_fine = AutoModelForTokenClassification.from_pretrained("/scratch/w/wluyliu/yananc/finetunes/roberta_nerd_fine")
nlp_roberta_nerd_fine = pipeline("ner", model=model_roberta_nerd_fine, tokenizer=tokenizer_roberta_nerd_fine,\
                     aggregation_strategy="simple", device=gpu)

# load off-the-shelf model : roberta-large fine-tuned on nerd-coarse-grained
tokenizer_roberta_nerd_coarse = AutoTokenizer.from_pretrained("/scratch/w/wluyliu/yananc/finetunes/roberta_nerd_coarse")
model_roberta_nerd_coarse = AutoModelForTokenClassification.from_pretrained("/scratch/w/wluyliu/yananc/finetunes/roberta_nerd_coarse")
nlp_roberta_nerd_coarse = pipeline("ner", model=model_roberta_nerd_coarse, tokenizer=tokenizer_roberta_nerd_coarse,\
                    aggregation_strategy="simple", device=gpu)


def check_entity_ft(content, nlp_roberta, fmark):
    ner_results = nlp_roberta(content)

    infos = []
    for ii in ner_results:
        infos.append((ii['entity_group'], ii['word'].strip(), ii['start'], ii['end'], fmark))
    return infos



# spacy
# https://spacy.io/models/en#en_core_web_lg
# python -m spacy download en_core_web_lg
# python -m spacy download en_core_web_trf
import spacy
en_core_web_lg = spacy.load('en_core_web_lg', disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
nlp_split_sents = spacy.load('en_core_web_sm')

def check_entity_spacy(content, en_core_web_lg, fmark ):
    doc = en_core_web_lg(content)
    infos = []
    for ent in doc.ents:
        infos.append((ent.label_, ent.text, ent.start_char, ent.end_char, fmark))

    return infos


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

    infos = []
    for ii in sentence.get_spans('ner'):
        infos.append((ii.tag, ii.text.strip(), ii.start_position, ii.end_position, fmark))
 
    return infos






############## main ########################
import spacy,re
from spacy import displacy

def find_index_of_entities(span, raw_text):
    indexs = []
    for match in re.finditer(span, raw_text):
        indexs.append((match.start(), match.end()))
    return indexs



nlp = spacy.blank('en')

def check_overlap(index_exist, sel_index):
    if not index_exist:
        return 0 

    for ii in index_exist:
        if (sel_index[0]>=ii[0] and sel_index[0]<=ii[1]) or (sel_index[1]>=ii[0] and sel_index[1]<=ii[1]):
            return 1 
    return 0




with open('/home/w/wluyliu/yananc/nlp4quantumpapers/articles_full.json', 'r') as f:
    jxml = json.load(f)

tag_map = {'PERSON':'PER', 'ORGANIZATION':'ORG', 'LOCATION':'LOC', 'GPE':'LOC'}
pick_from_nerd = ['EDUCATION','COMPANY','SOFTWARE','Government'.upper() ]


# sents = [i.text.replace('\n','').strip() for i in nlp_split_sents(content).sents]

def ner_combine(content):
    infos = []

    flair_ontonotes_ner_results = check_entity_flair(content, tagger_ontonotes,'flaironto')
    infos.extend(flair_ontonotes_ner_results)

    flair_conll_ner_results = check_entity_flair(content, tagger_conll,'flairconll')
    infos.extend(flair_conll_ner_results)

    ft_ner_ori = check_entity_ft(content, nlp_roberta, 'ftori')
    infos.extend(ft_ner_ori)


    # spacy_ner_results = check_entity_spacy(content, en_core_web_lg, 'spacy')
    # infos.extend(spacy_ner_results)

    # ft_ner_nerd_coarse = check_entity_ft(sent, nlp_roberta_nerd_coarse) 
    # infos.extend(ft_ner_nerd_coarse)

    # ft_ner_nerd_fine = check_entity_ft(sent, nlp_roberta_nerd_fine) 
    # infos.extend(ft_ner_nerd_fine)

        # print(sent) 

    df = pd.DataFrame(infos, columns=['ner','span', 'start', 'end', 'fmark'])
    df['ner'] = df['ner'].map(lambda x: x.upper())
    df['ner'] = df['ner'].map(lambda x: tag_map[x] if x in tag_map else x )

    df = df.loc[~df['span'].isin([i for i in string.punctuation])]
    df.drop_duplicates(inplace=True)


    df_org = df.loc[df['ner']=='ORG']
    return df_org
# df_pretrain = df.loc[df['fmark'].isin(['flair_onto', 'flair_conll', 'ft_ori'])]
# df_nerd = df.loc[df['fmark'].isin(['nerd_coarse', 'nerd_fine'])]
# df_nerd = df_nerd.loc[df_nerd['ner'].isin(pick_from_nerd)]

# df_pretrain.drop_duplicates(['ner','span'], inplace=True)
# df_nerd.drop_duplicates(['ner','span'], inplace=True)

# df_pretrain.sort_values(by=['ner'], ascending=True, inplace=True)
# df_nerd.sort_values(by=['ner'], ascending=True, inplace=True)
# return df_pretrain, df_nerd

samples = random.sample(jxml, 100)

df_ll = []
for sample in jxml:
    content = sample['post_content'].encode("ascii", "ignore").decode("utf-8").strip()
    df_org = ner_combine(content)
    df_org['article_ID'] = sample['article_ID']



    doc = nlp.make_doc(content)
    ents = []
    index_exist = []

    for ix, row in df_org.iterrows():
        if len(row['span'])<=1:
            continue
        overlap = check_overlap(index_exist, (row['start'], row['end']))

        if overlap:
            continue

        ent = doc.char_span(row['start'], row['end'], label=row['ner'])
        print(row['span'], '==>', row['start'], row['end'])
        if ent is None:
            continue
        ents.append(ent)
        index_exist.append((row['start'], row['end']))

    doc.ents = ents
    html = displacy.render(doc, style="ent", page=True)

    with open("ner_spacy_test.html", "a") as ff:
        ff.write('article_ID:{}'.format(sample['article_ID'])+'\n'+ html+'------------\n\n') 











labels_candidates = ['company', 'investor', 'university','Venture Capital Investor', 'academic organization', 
                        'goverment organization']


