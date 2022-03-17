import string,re,json
import pandas as pd
from nltk.corpus import stopwords
sw = set(stopwords.words("english"))

def make_df(json_path, target_cates):
    infos = []
    with open(json_path, 'r') as f: 
        for line in f:
            js = json.loads(line)
            if js['categories'] in target_cates: # ion trap // 
                infos.append(js)
    df = pd.DataFrame(infos) # 78160
    df.drop_duplicates(['abstract'], inplace=True)
    df['yymm'] = pd.to_datetime(df['update_date'].map(lambda x: '-'.join(x.split('-')[:2] )))
    return df 
    
def remove_latex(sent):
    tokens = sent.replace('\n',' ').strip().split()
    tokens_clean = [ii for ii in tokens if  "$" not in ii and ii and '\\' not in ii]
    return ' '.join(tokens_clean)

from nltk.stem.porter import PorterStemmer
def clean_title(title):
    title = title.lower()
    title = re.sub(r'[^\w\s]',' ',title)
    title = title.replace('\n',' ')
    title = title.replace('_line_', ' ')
    # PorterStemmer().stem()
    tokens = [ w.strip() \
               for w in title.split(' ') if w not in sw and w and not w.isdigit() \
             and w not in string.punctuation and w not in string.ascii_lowercase and len(w) >=2 and len(w)<=15]
    if not tokens:
        return ""
    else:
        return " ".join(tokens)