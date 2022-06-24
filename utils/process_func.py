import string,re,json
import pandas as pd
from nltk.corpus import stopwords
sw = set(stopwords.words("english"))
import datasets
def make_df(json_path, target_cates):
    infos = []
    with open(json_path, 'r') as f: 
        for line in f:
            js = json.loads(line)
            if target_cates:
                if js['categories'] in target_cates: # ion trap // 
                    infos.append(js)
            else:
                infos.append(js)
    df = pd.DataFrame(infos) # 78160
    df.drop_duplicates(['abstract'], inplace=True)
    df['yymm'] = pd.to_datetime(df['update_date'].map(lambda x: '-'.join(x.split('-')[:2] )))
    df['abstract_clean'] = df['abstract'].map(lambda x: remove_latex(x))
    df['abstract_stem'] = df['abstract_clean'].map(lambda x: clean_title(x))
    return df 
    
def remove_latex(sent):
    tokens = sent.replace('\n',' ').replace('</s>', '').strip().split()
    tokens_clean = [ii for ii in tokens if  "$" not in ii and ii and '\\' not in ii]
    return ' '.join(tokens_clean)

from nltk.stem.porter import PorterStemmer
def clean_title(title):
    title = title.lower()
    title = re.sub(r'[^\w\s]',' ',title)
    title = title.replace('\n',' ')
    title = title.replace('_line_', ' ')
    # 
    tokens = [ PorterStemmer().stem(w.strip()) \
               for w in title.split(' ') if w not in sw and w and not w.isdigit() \
             and w not in string.punctuation and w not in string.ascii_lowercase and len(w) >=2 and len(w)<=15]
    if not tokens:
        return ""
    else:
        return " ".join(tokens)




def load_dsn(dsn):
    if dsn == 'cc':
        cc_news = datasets.load_dataset('cc_news', split="train", cache_dir='/scratch/w/wluyliu/yananc/cache')
        dfi = pd.DataFrame(random.sample(cc_news['text'], 100000), columns=['abstract'])
    
    elif dsn == 'arxiv':
        dfi = make_df('/home/w/wluyliu/yananc/nlp4quantumpapers/arxiv-metadata-oai-snapshot_2.json', ['quant-ph'])

    elif dsn == 'subgroup':
        # cate sub
        df = pd.read_csv("/home/w/wluyliu/yananc/nlp4quantumpapers/artificially_labeled_abstracts.csv")
        dfi = df.loc[df['Assigned_group']=='Full stack and quantum computers']
        dfi['abstract_clean'] = dfi['abstract'].map(lambda x: remove_latex(x))
        dfi['abstract_stem'] = dfi['abstract_clean'].map(lambda x: clean_title(x))

    elif dsn == '20news':
        from sklearn.datasets import fetch_20newsgroups
        newsgroups_train = fetch_20newsgroups(subset='train')
        newsgroups_test = fetch_20newsgroups(subset='test')
        df_train = pd.DataFrame(zip(newsgroups_train['data'], list(newsgroups_train['target'])), columns=['abstract','label'])
        df_test = pd.DataFrame(zip(newsgroups_test['data'], list(newsgroups_test['target'])), columns=['abstract','label'])

        dfi = pd.concat([df_train, df_test])
        dfi['abstract_clean'] = dfi['abstract'].map(lambda x: remove_latex(x))
        dfi['abstract_stem'] = dfi['abstract_clean'].map(lambda x: clean_title(x))
    return dfi



#################### NER ###################


def map_func(example):
    # tag_fine_ix = []
    # tag_coarse_ix = []
    tags_coarse = []
    for tag in example['tags']:
        # tag_fine_ix.append(tag_map_fine[tag])
        if tag != 'O':
            # tag_coarse_ix.append(tag_map_coarse[tag.split('-')[0]])
            tags_coarse.append(tag.split('-')[0])
        else:
            # tag_coarse_ix.append(tag_map_coarse[tag])
            tags_coarse.append(tag)
    example['tags_coarse'] = tags_coarse
    example['tags_fine'] = example['tags']
    # example['tag_fine_ix'] = tag_fine_ix 
    # example['tag_coarse_ix'] = tag_coarse_ix

    for ii, jj in example.items():
        if ii == 'id':
            continue
        assert len(jj) == len(example['tokens']) 

    return example


# def sep_trunk(df_tmp):
    

    


#     for tag in df_tmp['tag'].unique():
#         if tag == 'O':
#             continue 
#         df_candidates_mention = df_tags.loc[df_tags['tag'] == tag]
#         df_tmp_f = df_tmp.loc[df_tmp['tag']==tag]

#         list_of_df = [d for _, d in df_tmp_f.groupby(df_tmp_f.index - np.arange(len(df_tmp_f)))]
        
        
#         for dfi in list_of_df:
#             ixi = list(dfi.index)
#             candidate = df_candidates_mention.sample(1)['span'].tolist()[0]
#             candidate_tokens = candidate.split()






    #     mentions = [' '.join(df_tag['token'].tolist()) for df_tag in list_of_df]
    #     results.append((' ; '.join(mentions), tag))
    # return results



