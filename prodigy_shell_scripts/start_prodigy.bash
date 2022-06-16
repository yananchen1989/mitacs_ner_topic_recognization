#python -m prodigy <recipe> <database> <tokenizing> <article jsonl input file> --label group,university,investor,lab,gov,company
python -m prodigy ner.manual tmp blank:en split_4.jsonl --label group,university,investor,lab,gov,company
