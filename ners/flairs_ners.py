




content = jxml[38]['post_content'].strip()

sentence = Sentence(content)
tagger.predict(sentence)

ners = sentence.get_spans('ner')








# python -m spacy download en_core_web_sm
# import spacy
# ner_model = spacy.load('en_core_web_sm')

# doc = ner_model(content)
# ners = list(set([ii.text for ii in doc.ents]))










