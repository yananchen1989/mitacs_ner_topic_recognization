import csv
import json
import re
from collections import defaultdict
import spacy
from spacy.matcher import Matcher
from nltk import tokenize


def load_articles(filename):
    """Loads the articles from the json file.

    Args:
        filename (str): The path to the json file containing the articles.

    Returns:
        list[dict]: A list of articles represented in Python dictionary form.
    """
    # special_character_mappings = {
    #     ord("’"): "'",
    #     ord("–"): "-",
    #     ord("”"): '"',
    #     ord("“"): '"',
    #     ord("‘"): "'",
    #     ord("—"): "-",
    # }
    # article["post_content"] = article["post_content"].translate(
    #     special_character_mappings
    # )
    with open(filename, "r", encoding="utf-8") as file:
        articles = json.load(file)
        for article in articles:
            article["post_content"] = (
                article["post_content"]
                .encode("ascii", "ignore")
                .decode("utf-8")
                .strip()
            )

    return articles


def create_dict_and_patterns(filename):
    """Creates the entity translation dictionary from the classes given in the csv file.
    Also creates the regular expression patterns for each entity.

    Args:
        filename (str): The path to the csv file containing the entity classes.

    Returns:
        dict: A translation of the class each named entity belongs to.
        dict: A dictionary of the regular expression patterns for each entity.
    """
    entity_translation = {}
    patterns = {}
    with open(filename, "r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            entity_translation[row[0]] = row[1]
            patterns[row[0]] = re.compile(
                f"(?<![\w\d]){row[0]}(?![\w\d])", re.IGNORECASE
            )
    return entity_translation, patterns


def create_dict_and_token_matcher(filename, nlp):
    """Generates token matcher using Spacy library.

    Args:
        filename (str): The path to the csv file containing the entity classes.
        nlp (Spacy Model): The spacy model used to tokenize the article.

    Returns:
        Matcher: The created matcher object used to token match entities
    """
    matcher = Matcher(nlp.vocab)
    patterns = defaultdict(list)
    with open(filename, "r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            pattern = [{"LOWER": f"{word.lower()}"} for word in row[0].split(" ")]
            patterns[row[1]].append(pattern)
            # pattern.append({"IS_PUNCT": True})
            # patterns[row[1]].append(pattern)

    for entity_type in patterns:
        matcher.add(entity_type, patterns[entity_type])
    return matcher


def token_match_entities(matcher, nlp, article):
    """Token matches entities for a single article.

    Args:
        matcher (Matcher): The matcher object used to match the entities.
        nlp (Spacy model): The spacy model used to tokenize the article.
        article (dict): The article json object represented in Python dictionary form
    Returns:
        dict: The matches for each of the entity classes within the post_content of the article.
    """

    doc = nlp(article["post_content"])
    matches = matcher(doc)
    all_matches = defaultdict(list)
    for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id]
        entity = doc[start:end].text
        all_matches[string_id].append(entity)
    return all_matches


def match_entities(entity_translation, article, patterns):
    """Matches entities for a single article.

    Args:
        entity_translation (dict): A translation of the class each named entity belongs to.
        article (dict): The article json object represented in Python dictionary form
        patterns (dict): A dictionary of the regular expression patterns for each entity.

    Returns:
        dict: The matches for each of the entity classes within the post_content of the article.
    """
    matches = defaultdict(list)
    for entity in entity_translation:
        if patterns[entity].search(article["post_content"]):
            matches[entity_translation[entity]].append(entity)
    return matches


def output_prodigy_patterns(entity_translation, output_filename):
    """Outputs the patterns used in prodigy format.

    Args:
        entity_translation (dict): A translation of the class each named entity belongs to.
        output_filename (str): The path to the output file.
    """
    prodigy_output = []
    labels = set()
    for entity in entity_translation:
        info = {
            "label": entity_translation[entity],
            "pattern": [{"lower": f"{word.lower()}"} for word in entity.split(" ")],
            # "pattern": [{"lower": {"regex": f"(?<![\w\d]){entity}(?![\w\d])"}}],
        }
        labels.add(entity_translation[entity])
        prodigy_output.append(info)
    for entity_type in labels:
        print("Label:", entity_type)
    with open(output_filename, "w", encoding="utf-8") as file:
        for item in prodigy_output:
            json.dump(item, file)
            file.write("\n")


def output_validation_prodigy(articles, output_filename):
    """Outputs the validation data in prodigy format.

    Args:
        entity_translation (dict): A translation of the class each named entity belongs to.
        patterns (dict): A dictionary of the regular expression patterns for each entity.
        articles (list[dict]): A list of articles represented in Python dictionary form.
        output_filename (str): The path to the output file.
    """
    prodigy_output = []
    for article in articles:
        print(f"Writing article {article['id']} to file")
        info = {"text": article["post_content"]}
        prodigy_output.append(info)
    with open(output_filename, "w", encoding="utf-8") as file:
        for item in prodigy_output:
            json.dump(item, file)
            file.write("\n")


def export_prodigy_sentence_level(input_filename, output_filename, entities):
    """Exports the given prodigy database output to sentence level annotations.

    Args:
        input_filename (str): Name of the prodigy database file to be exported (jsonl).
        output_filename (str): Name of the output file to be written (csv).
        entities (list[str]): List of the types of entities present in the database.
    """
    header = ["sent", "tag", "span"]
    with open(input_filename, "r", encoding="utf-8") as in_file:
        with open(output_filename, "a", encoding="utf-8", newline="") as out_file:
            writer = csv.DictWriter(out_file, fieldnames=header)
            out_file.seek(0)
            out_file.truncate()
            out_file.write(",".join(header) + "\n")
            entity_indices = {entity: i for i, entity in enumerate(entities)}
            for line in in_file:
                annotation = json.loads(line)
                if annotation["answer"] == "accept":
                    sentences = tokenize.sent_tokenize(annotation["text"])
                    sentence_borders = [len(sentences[0])]

                    for sentence in sentences[1:]:
                        sentence_borders.append(
                            len(sentence) + sentence_borders[-1] + 1
                        )
                    entity_present = set()
                    spans = annotation["spans"]
                    for i, sentence in enumerate(sentences):
                        entity_count = 0
                        sentence_annotation_objects = [
                            {"sent": sentence, "tag": entity, "span": []}
                            for entity in entities
                        ]
                        entity_type_present = set()
                        for span in spans:
                            if span["end"] <= sentence_borders[i]:
                                entity_type_present.add(entity_indices[span["label"]])
                                entity_count += 1
                                entity_present.add(i)
                                obj = sentence_annotation_objects[
                                    entity_indices[span["label"]]
                                ]
                                obj["span"].append(
                                    annotation["text"][span["start"] : span["end"]]
                                )
                                # print(span, sentence_borders[i], i)
                                # print(
                                #     f"sent: {sentence}, tag: {span['label']}, span: {annotation['text'][span['start']:span['end']]}"
                                # )
                        for i in range(len(sentence_annotation_objects)):
                            if i in entity_type_present:
                                sentence_annotation_objects[i]["span"] = ";".join(
                                    sentence_annotation_objects[i]["span"]
                                )
                                writer.writerow(sentence_annotation_objects[i])
                        spans = spans[entity_count:]
                        if spans == []:
                            break
                    for i in range(len(sentences)):
                        if i not in entity_present:
                            writer.writerow(
                                {"sent": sentences[i], "tag": "###", "span": "###"}
                            )
                            # print(f"sent: {sentences[i]}, tag: ###, span: ###")


def output_prodigy_sentence_level_tokens(input_filenames, output_filename):
    """Outputs the prodigy annotations to sentence level tokens for entity recoginition training. Takes in a list of
    input file names and outputs a single file concatenating all articles together in the output file.

    Args:
        input_filenames (list[str]): List of filenames of the prodigy database exports in jsonl format.
        output_filename (str): A jsonl file containing a collection of annotated sentence objects for each article.
    """
    with open(output_filename, "w", encoding="utf-8") as out_file:
        for input_filename in input_filenames:
            with open(input_filename, "r", encoding="utf-8") as in_file:
                id = 0
                for line in in_file:
                    annotation = json.loads(line)
                    if annotation["answer"] == "accept":
                        spans = iter(annotation["spans"])
                        current_span = next(spans)
                        sentences = tokenize.sent_tokenize(annotation["text"])
                        all_tokens = annotation["tokens"]
                        tokenizer = spacy.blank("en").tokenizer
                        current_token_id = 0
                        sentence_objects = []
                        for sentence in sentences:
                            obj = {"id": id, "tokens": [], "tags": []}
                            sentence_tokens = tokenizer(sentence)
                            for i in range(len(sentence_tokens)):
                                if (
                                    all_tokens[i + current_token_id]["text"]
                                    != sentence_tokens[i].text
                                ):
                                    for _ in range(i, len(all_tokens)):
                                        if (
                                            all_tokens[i + current_token_id]["text"]
                                            == sentence_tokens[i].text
                                        ):
                                            break
                                        else:
                                            current_token_id += 1
                                else:
                                    obj["tokens"].append(
                                        all_tokens[i + current_token_id]["text"]
                                    )
                                    if (
                                        current_span["token_start"]
                                        <= all_tokens[i + current_token_id]["id"]
                                        <= current_span["token_end"]
                                    ):

                                        obj["tags"].append(current_span["label"])
                                        # print(
                                        #     f"{all_tokens[i+current_token_id]['text']}: {current_span['label']}"
                                        # )
                                        if (
                                            current_span["token_end"]
                                            == all_tokens[i + current_token_id]["id"]
                                        ):
                                            try:
                                                current_span = next(spans)
                                            except StopIteration:
                                                continue
                                    else:
                                        obj["tags"].append("O")
                                        # print(
                                        #     f"{all_tokens[i+current_token_id]['text']}: O"
                                        # )

                            current_token_id += i + 1
                            sentence_objects.append(obj)
                        for obj in sentence_objects:
                            json.dump(obj, out_file)
                            out_file.write("\n")
                    id += 1

    return


def merge_sentence_level_annotations(filenames, output_filename):
    """Combines the sentence level annotations from multiple files into one file.

    Args:
        filenames (list[str]): List of the filenames of the sentence level annotations to be merged.
        output_filename (str): File to write the merged annotations to (csv).
    """
    header = ["sent", "tag", "span"]
    with open(output_filename, "w", encoding="utf-8") as out_file:
        out_file.write(",".join(header) + "\n")
        for filename in filenames:
            with open(filename, "r", encoding="utf-8") as in_file:
                next(in_file)
                out_file.write(in_file.read())
            out_file.write("\n")


def output_validation(entity_translation, patterns, articles, output_filename):
    """Creates format for validation of the entity matches for people to view in Excel worksheet. Records locations of
    entities within article text and asks for the confirmation of correct prediction of the named entity.

    Args:
        entity_translation (dict): A translation of the class each named entity belongs to.
        patterns (dict): A dictionary of the regular expression patterns for each entity.
        articles (list[dict]): A list of articles to explore find the named entities of interest.
        output_filename (str): The csv file to output the results of the found entities to.
    """
    header = ["Article ID", "Article Text", "Entity", "Tag", "Start", "End", "Length"]
    with open(output_filename, "a", newline="", encoding="utf-8") as file:
        file.seek(0)
        file.truncate()
        file.write(",".join(header) + "\n")
        writer = csv.DictWriter(file, fieldnames=header)
        for article in articles:
            print(f"Writing article {article['id']} to file")
            for entity in patterns:
                info = {
                    "Article ID": None,
                    "Article Text": None,
                    "Entity": None,
                    "Tag": None,
                    "Start": None,
                    "End": None,
                    "Length": None,
                }
                for match in patterns[entity].finditer(article["post_content"]):
                    info["Article ID"] = article["id"]
                    info["Article Text"] = article["post_content"]
                    info["Entity"] = entity
                    info["Tag"] = entity_translation[entity]
                    info["Start"] = match.start() + 1
                    info["End"] = match.end() + 1
                    info["Length"] = len(entity)
                    writer.writerow(info)


def match_all_articles(entity_translation, articles, output_filename):
    """Matches all the entities across a collection of articles using the entity translation dictionary.

    Args:
        entity_translation (dict): A translation of the class each named entity belongs to.
        articles (list[dict]): A list of articles to explore find the named entities of interest.
        output_filename (str): The csv file to output the results of the found entities to.
    """
    patterns = {}
    header = [
        "article_id",
        "group_centre",
        "university",
        "investor",
        "National Lab",
        "Government Agency",
        "Quantum_comp",
    ]
    matches = {
        "group_centre": [],
        "university": [],
        "investor": [],
        "National Lab": [],
        "Government Agency": [],
        "Quantum_comp": [],
    }
    with open(output_filename, "a", newline="", encoding="utf-8") as file:
        file.seek(0)
        file.truncate()
        file.write(",".join(header) + "\n")
        writer = csv.DictWriter(file, fieldnames=header)
        for entity in entity_translation:
            patterns[entity] = re.compile(
                f"(?<![\w\d]){entity}(?![\w\d])", re.IGNORECASE
            )
        for article in articles:
            matches = {
                "group_centre": [],
                "university": [],
                "investor": [],
                "National Lab": [],
                "Government Agency": [],
                "Quantum_comp": [],
            }
            for entity in patterns:
                if patterns[entity].search(article["post_content"]):
                    matches[entity_translation[entity]].append(entity)
            for match in matches:
                if len(matches[match]) == 0:
                    matches[match] = ""
                elif len(matches[match]) == 1:
                    matches[match] = matches[match][0]
                else:
                    matches[match] = f'{",".join(matches[match])}'
            matches["article_id"] = article["id"]
            print(f"Writing article {article['id']} to file")
            writer.writerow(matches)


if __name__ == "__main__":
    ENTITIES = ["group", "university", "investor", "lab", "gov", "company"]
    output_prodigy_sentence_level_tokens(
        [
            "data/annotations_split3_first_50_Ian.jsonl",
            "data/Douglas_Annotations.jsonl",
        ],
        "sentence_level_tokens.jsonl",
    )
    # entity_translation, patterns = create_dict_and_patterns("data/QI-NERs.csv")
    # data = load_articles("data/articles_full.json")
    # matches = match_entities(entity_translation, data[0], patterns)

    # export_prodigy_sentence_level(
    #     "data/annotations_split3_first_50_Ian.jsonl",
    #     "sentence_level_annotations_1.csv",
    #     ENTITIES,
    # )
    # export_prodigy_sentence_level(
    #     "data/Douglas_Annotations.jsonl", "sentence_level_annotations_2.csv", ENTITIES
    # )
    # merge_sentence_level_annotations(
    #     ["sentence_level_annotations_1.csv", "sentence_level_annotations_2.csv"],
    #     "sentence_level_annotations.csv",
    # )
