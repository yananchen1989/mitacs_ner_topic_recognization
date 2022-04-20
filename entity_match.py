import csv
import json
import re
from collections import defaultdict


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


def match_all_articles(entity_translation, articles, output_filename):
    """Matches all the entities across a collection of articles using the entity translation dictionary.

    Args:
        entity_translation (dict): A translation of the class each named entity belongs to.
        articles (list[dict]): A list of articles to explore find the named entities of interest.
        output_filename (str): The csv file to output the results of the entity matching to.
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
    with open(output_filename, "w") as file:
        file.write(",".join(header) + "\n")
    for entity in entity_translation:
        patterns[entity] = re.compile(f"(?<![\w\d]){entity}(?![\w\d])", re.IGNORECASE)
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
        with open(output_filename, "a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=header)
            writer.writerow(matches)


if __name__ == "__main__":
    entity_translation, patterns = create_dict_and_patterns("QI-NERs.csv")
    with open("articles_full.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    matches = match_entities(entity_translation, data[0], patterns)
    print(matches)

