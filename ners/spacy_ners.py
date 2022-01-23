# python -m spacy download en_core_web_sm

import spacy
ner_model = spacy.load('en_core_web_sm')

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

doc = ner_model(text)
ners = list(set([ii.text for ii in doc.ents]))