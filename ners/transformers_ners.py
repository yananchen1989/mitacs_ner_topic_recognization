from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

nlp = pipeline("ner", model=model, tokenizer=tokenizer)

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

ner_results = nlp(text)
print(ner_results)