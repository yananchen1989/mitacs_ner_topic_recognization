#python3 -m prodigy <recipe> <database> <tokenizing> <article jsonl input file> --label group,university,investor,lab,gov,company
python3 -m prodigy ner.manual tmp blank:en data/prodigy_articles.jsonl --label group,university,investor,lab,gov,company
