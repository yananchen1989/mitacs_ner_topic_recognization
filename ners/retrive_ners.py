import time
from flair.data import Sentence
from flair.models import SequenceTagger

import torch, flair
flair.device = torch.device('cpu')
#flair.device = torch.device('cuda:0')
#  note that get_ners can only run one single GPU !!! 
tagger = SequenceTagger.load("flair/ner-english-fast")




text = '''
The Josephson junction quantum computer was demonstrated in 1999 by Nakamura
and the coworkers. In this computer a Cooper pair box, which is a small superconducting
island electrode is weakly coupled to a bulk superconductor. Weak coupling between the
superconductors create a Josephson junction between them which behaves as a capacitor.
If the Cooper box is small as a quantum dot, the charging current breaks into discrete
transfer of individual Cooper pairs, so that ultimately it is possible to just transfer a single
Cooper pair across the junction. Like quantum dot, computers in Josephson junction
computers, qubits are controlled electrically. Josephson junction’s quantum computers
are one of the promising candidates for future developments.
'''

def get_ners(text):
    sentence = Sentence(text)
    tagger.predict(sentence)
    ners = list(set([ii['text'] for ii in sentence.to_dict(tag_type='ner')['entities']]))
    return ners


ners = get_ners(text)