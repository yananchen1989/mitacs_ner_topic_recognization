import time
from flair.data import Sentence
from flair.models import SequenceTagger

import torch, flair
flair.device = torch.device('cpu')
#flair.device = torch.device('cuda:0')
#  note that get_ners can only run one single GPU !!! 
tagger = SequenceTagger.load("flair/ner-english-fast")




text = '''
The distinctive non-classical features of quantum physics were first
discussed in the seminal paper by A. Einstein, B. Podolsky and N. Rosen (EPR)
in 1935. In his immediate response E. Schr\"odinger introduced the notion of
entanglement, now seen as the essential resource in quantum information as well
as in quantum metrology. Furthermore he showed that at the core of the EPR
argument is a phenomenon which he called steering. In contrast to entanglement
and violations of Bell's inequalities, steering implies a direction between the
parties involved. Recent theoretical works have precisely defined this
property. Here we present an experimental realization of two entangled Gaussian
modes of light by which in fact one party can steer the other but not
conversely. The generated one-way steering gives a new insight into quantum
physics and may open a new field of applications in quantum information.
'''


sentence = Sentence(text)
tagger.predict(sentence)
#ners = list(set([ii['text'] for ii in sentence.to_dict(tag_type='ner')['ner']]))

for ii in sentence.to_dict(tag_type='ner')['ner']:
    print(ii)




