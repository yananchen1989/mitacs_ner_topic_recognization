import string

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