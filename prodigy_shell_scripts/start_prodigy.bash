#python -m prodigy <recipe> <database> <tokenizing> <article jsonl input file> --label group,university,investor,lab,gov,company
python -m prodigy ner.manual tmp blank:en prodigy_articles.jsonl --label group,university,investor,lab,gov,company
