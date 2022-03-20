# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 21:27:44 2022

@author: Doug
"""
import json
import pandas as pd
import nltk
import numpy as np
#nltk.download('stopwords')
#nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.util import ngrams
import collections

snowball = SnowballStemmer(language='english')
stopword = stopwords.words('english')


#import Text collection

arxiv_tbl = pd.read_csv('quant_ph.csv')

#Select a manageable amount of abstracts
#arxiv_tbl = arxiv_tbl.loc[0:5000,]


arxiv_tbl.head()

taxonomy_tbl_raw = pd.read_csv('tqi_taxonomy.csv', index_col=False)


taxonomy_tbl = taxonomy_tbl_raw.loc[taxonomy_tbl_raw['Classification'] == 'Secondary Classification']
taxonomy_tbl




#Mapping subclasses

subclasses = taxonomy_tbl["SubClassification"].unique()
subclass_to_type_map = {}

for s in subclasses:
    if s not in subclass_to_type_map:
        subclass_to_type_map[s] = []
    types = taxonomy_tbl.loc[taxonomy_tbl["SubClassification"] == s]["Type"].tolist()
    for t in types:
        subclass_to_type_map[s].append(t.lower())

print(subclass_to_type_map)












def prep_abstract(text):
    
    #tokenization
    tok_word = word_tokenize(text)

    # Lower case conversion
    lower_text = text.lower()
    tok_word_lwr = word_tokenize(lower_text)

    # Stop words removal
    removing_stopwords = [word for word in tok_word_lwr if word not in stopword]
    #print (removing_stopwords)

    # Word Stemming
    final_words = [snowball.stem(word) for word in removing_stopwords]
    
    return(final_words)


#testing above function

text = arxiv_tbl.iat[2,10]

abstract1 = prep_abstract(text)

print(text)
print(abstract1)












# isolate keywords from taxonomy
taxonomy_words = pd.DataFrame(taxonomy_tbl.iloc[:,3])
print(taxonomy_words)

#Lower case conversion
taxonomy_words['Type'] = taxonomy_words['Type'].str.lower()

taxonomy_list = taxonomy_words['Type'].tolist()

for term in range(len(taxonomy_list)):
    taxonomy_list[term] = word_tokenize(taxonomy_list[term])
    
print(taxonomy_list)

for i in range(len(taxonomy_list)):
    taxonomy_list[i] = [snowball.stem(word) for word in taxonomy_list[i]]

print("")
print(taxonomy_list)

# Lower case conversion

#lower_text = taxonomy_list.lower()
#tok_word_lwr = word_tokenize(lower_text)
#print(tok_word_lwr)








def word_counts(text, taxonomy_list):
    out = []
    counts =  nltk.FreqDist(text)   # this counts ALL word occurences
    for x in taxonomy_list:
        out.append(sum([counts[y] for y in x])) # this returns what was counted for *words
    return out

print(taxonomy_list)

example = word_counts(abstract1,taxonomy_list)
print(example)













# Return list of counted taxonomy
x = []

for i in range(len(arxiv_tbl.index)):
    text = arxiv_tbl.iat[i,10]
    abstract = prep_abstract(text)
    abst_count = word_counts(abstract, taxonomy_list)
    x.append(abst_count)

print(x)







# Convert list to the category
category_list = []

for cat in x:
    if sum(cat) < 4:
        category_list.append(0)
    else:
        cat_index = cat.index(max(cat))
        cat = taxonomy_list[cat_index]
        category_list.append(cat)

print(category_list)










def find_first_likely(stem_list):
    for tax_word in taxonomy_words['Type'].tolist():
        if stem_list != 0:
            for stem_word in stem_list:
                if tax_word.lower().startswith(stem_word):
                    return tax_word
    return 0

def find_all_likely(stem_list):
    likely_cats = []
    for tax_word in taxonomy_words['Type'].tolist():
        if stem_list != 0:
            for stem_word in stem_list:
                if tax_word.lower().startswith(stem_word):
                    likely_cats.append(tax_word)
    return likely_cats

def find_group(cat_key):
    for x in subclass_to_type_map:
        if cat_key in subclass_to_type_map[x]:
            return x

arxiv_tbl["Assigned_cat"] = category_list
arxiv_tbl["Assigned_cat_first"] = arxiv_tbl["Assigned_cat"].apply(find_first_likely)
arxiv_tbl["Assigned_cat_all"] = arxiv_tbl["Assigned_cat"].apply(find_all_likely)
arxiv_tbl["Assigned_group"] = arxiv_tbl["Assigned_cat_first"].apply(find_group)

arxiv_tbl





arxiv_tbl_cats = arxiv_tbl[arxiv_tbl["Assigned_cat_first"] != 0]
arxiv_tbl_cats
# arxiv_tbl_cats








########## RESULTS
arxiv_tbl_cats["Assigned_group"].value_counts()
arxiv_tbl_cats["Assigned_cat_first"].value_counts()
arxiv_tbl_cats.groupby("Assigned_group")["Assigned_cat_first"].value_counts()
arxiv_tbl_cats.to_csv('artificially_labeled_abstracts.csv', index=False)

######### UNIQUE IDS - temporary method
pd.unique(arxiv_tbl_cats['Assigned_cat_first'])
#unique_id_dict = {}

for category_type in pd.unique(arxiv_tbl_cats['Assigned_cat_first']):
    print(category_type)
    unique_id_list = (arxiv_tbl_cats.loc[arxiv_tbl_cats['Assigned_cat_first'] == category_type, 'id'].values)
    print(unique_id_list)
    csv_filepath = 'filtered_ids/%s.csv' % (category_type)
    id_df = pd.DataFrame(unique_id_list)
    id_df.columns = ['id']
    id_df.to_csv(csv_filepath, index=False, header=True)
    
    
    #Commented out for now - excel character limit reached, cannot store everything this way
    #unique_id_dict[category_type] = np.ndarray.tolist(unique_id_list)
    

#unique_id_dict['trapped ion']
#len(unique_id_dict['trapped ion'])
#id_filtered_results = pd.DataFrame.from_dict(unique_id_dict, orient='index')
#id_filtered_results.to_csv('filtered_ids.csv', index=False)
