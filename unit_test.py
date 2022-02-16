# import json
# import pandas as pd 

# with open('articles_full.json', 'r') as f:
#     jfull = json.load(f)




# df = pd.DataFrame(jfull)

# df['article_ID'] = df['article_ID'].astype(int)



# with open('articles_sample.xml.json', 'r') as f:
#     jxml = json.load(f)


# for d in jxml:
#     if d['article_id'] == '9882':
#         print(d)
#         break 

# df_ent = pd.DataFrame(jxml)

# df_ent['article_id'] = df_ent['article_id'].astype(int)

# df_ = pd.merge(df, df_ent, left_on='article_ID', right_on='article_id', how='inner')

# row = df_.sample(1)
# print(row['post_content'].tolist()[0])
# print(row['Name'].tolist()[0])




#############  arxiv ##########
import pandas as pd 
import json

target_categories = ['cond-mat.supr-con', 'math.QA', 'quant-ph', 'stat.CO']


infos = []
with open('arxiv-metadata-oai-snapshot_2.json', 'r') as f:
    for line in f:
        js = json.loads(line)
        if js['categories'] in target_categories:
            infos.append(js)

df = pd.DataFrame(infos)


df['yymm'] = df['update_date'].map(lambda x: '-'.join(x.split('-')[:2] ))



df_dates_cnt = df.yymm.value_counts().reset_index()

df_dates_cnt.sort_values(by=['index'], ascending=False)













