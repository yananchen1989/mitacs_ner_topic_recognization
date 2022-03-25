from keybert import KeyBERT
import pandas as pd 
from sentence_transformers import SentenceTransformer



#"all-mpnet-base-v2", "all-MiniLM-L6-v2", "paraphrase-albert-small-v2"

sentence_model = SentenceTransformer("allenai/scibert_scivocab_uncased")
kw_model = KeyBERT(model=sentence_model)

sentence_model= SentenceTransformer("/Users/yanan/Downloads/finetune/arxiv_scibert_quantph")
kw_model = KeyBERT(model=sentence_model)




sent = '''
This model is one of the possible geometrical interpretations of Quantum
Mechanics where found to every image Path correspondence the geodesic
trajectory of classical test particles in the random geometry of the stochastic
fields background. We are finding to the imagined Feynman Path a classical
model of test particles as geodesic trajectory in the curved space of Projected
Hilbert space on Bloch's sphere.
'''

sent = '''
We demonstrate a simple method for the determination of the magnetic field in
an ion trap using laser-cooled Be+ ions. The method is not based on magnetic
resonance and thus does not require delivering radiofrequency (RF) radiation to
the trap. Instead, stimulated Raman spectroscopy is used, and only an easily
generated optical sideband of the laser cooling wave is required. The d.c.
magnetic vector, averaged over the Be+ ion ensemble, is determined.
Furthermore, the field strength can be minimized and an upper limit for the
field gradient can be determined. The resolution of the method is 0.04 G at
present. The relevance for precision rovibrational spectroscopy of molecular
hydrogen ions is briefly discussed.
'''

sent = '''
This lecture series on Quantum Integer Programming (QuIP) -- created by
Professor Sridhar Tayur, David E. Bernal, and Dr. Davide Venturelli, a
collaboration between CMU and USRA, with the support from Amazon Braket during
Fall 2020 -- is intended for students and researchers interested in Integer
Programming and the potential of near term quantum and quantum-inspired
computing in solving optimization problems.
  Originally created for Tepper School of Business course 47-779 (at CMU),
these were also used for the course ID5840 (at IIT-Madras, by Professors Anil
Prabhakar and Prabha Mandayam) whose students (listed at the beginning of each
lecture) were scribes. Dr. Vikesh Siddhu, post-doc in CMU Quantum Computing
Group, assisted during the lectures, student projects, and with proof-reading
this scribe.
  Through these lectures one will learn to formulate a problem and map it to a
Quadratic Unconstrained Binary Optimization (QUBO) problem, understand various
mapping and techniques like the Ising model, Graver Augmented Multiseed
Algorithm (GAMA), Simulated or Quantum Annealing and QAOA, and ideas on how to
solve these Integer problems using these quantum and classical methods.
'''



kws = kw_model.extract_keywords(sent, keyphrase_ngram_range=(1, 3), stop_words='english', \
                            use_mmr=True, diversity=0.4, highlight=False,  top_n=5)

for ii in kws:
    print(ii[0], ii[1])
print()