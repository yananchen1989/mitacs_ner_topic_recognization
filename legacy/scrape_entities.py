import csv
import json
import os
import argparse

# import matplotlib.pyplot as plt
# import seaborn as sns
from dotenv import load_dotenv
from linkedin_api import Linkedin

ALL_JSON_PATHS = {
    "name": "name",
    "country": ["headquarter", "country"],
    "city": ["headquarter", "city"],
    "logo": ["logo", "image", "com.linkedin.common.VectorImage", "rootUrl"],
    "linkedin": "url",
    "founded": "foundedOn",
    "website": "companyPageUrl",
    "employeeRange": "staffCountRange",
    "tags": "associatedHashtags",
    "description": "description",
}

COMPANY_KEYS = {
    "name": ALL_JSON_PATHS["name"],
    "country": ALL_JSON_PATHS["country"],
    "city": ALL_JSON_PATHS["city"],
    "logo": ALL_JSON_PATHS["logo"],
    "linkedin": ALL_JSON_PATHS["linkedin"],
    "founded": ALL_JSON_PATHS["founded"],
    "website": ALL_JSON_PATHS["website"],
    "employeeRange": ALL_JSON_PATHS["employeeRange"],
    "tags": ALL_JSON_PATHS["tags"],
}

INVESTOR_KEYS = {
    "name": ALL_JSON_PATHS["name"],
    "country": ALL_JSON_PATHS["country"],
    "city": ALL_JSON_PATHS["city"],
    "linkedin": ALL_JSON_PATHS["linkedin"],
    "founded": ALL_JSON_PATHS["founded"],
    "website": ALL_JSON_PATHS["website"],
}

GROUP_KEYS = {
    "name": ALL_JSON_PATHS["name"],
    "country": ALL_JSON_PATHS["country"],
    "city": ALL_JSON_PATHS["city"],
    "logo": ALL_JSON_PATHS["logo"],
    "founded": ALL_JSON_PATHS["founded"],
    "website": ALL_JSON_PATHS["website"],
    "description": ALL_JSON_PATHS["description"],
}

GOVERNMENT_KEYS = {
    "name": ALL_JSON_PATHS["name"],
    "country": ALL_JSON_PATHS["country"],
    "logo": ALL_JSON_PATHS["logo"],
    "linkedin": ALL_JSON_PATHS["linkedin"],
    "founded": ALL_JSON_PATHS["founded"],
    "website": ALL_JSON_PATHS["website"],
    "employeeRange": ALL_JSON_PATHS["employeeRange"],
    "description": ALL_JSON_PATHS["description"],
}

UNIVERSITY_KEYS = {
    "name": ALL_JSON_PATHS["name"],
    "linkedin": ALL_JSON_PATHS["linkedin"],
    "founded": ALL_JSON_PATHS["founded"],
    "website": ALL_JSON_PATHS["website"],
    "description": ALL_JSON_PATHS["description"],
}

ALL_KEYS = {
    "company": COMPANY_KEYS,
    "investor": INVESTOR_KEYS,
    "group": GROUP_KEYS,
    "government": GOVERNMENT_KEYS,
    "university": UNIVERSITY_KEYS,
}

parser = argparse.ArgumentParser(description="Scrape entities from LinkedIn")
parser.add_argument(
    "-i",
    "--input_file",
    help="Input inference file from execution of inference.py (csv)",
    type=str,
)
parser.add_argument(
    "-o",
    "--output_file",
    help="Output filename for scraped entities (jsonl)",
)
args = parser.parse_args()
# Debug Function for testing of given named entities
def parse_dictionary(entities_filename, entity_type):
    """Creates a list of the specified named entity type from a csv file of named entities. Where the first column
    is the entity and the second column is the entity type.

    Args:
        entities_filename (str): A csv file of named entities
        entity_type (str): The type of named entity to create a list of

    Returns:
        list[str]: The list of named entities of the given type
    """
    with open(entities_filename, "r") as file:
        reader = csv.reader(file)
        next(reader)
        entities = [row[0] for row in reader if row[1] == entity_type]
    return entities


def scrape_entity(entity, output_filename, entity_type):
    """Scrape entity information from LinkedIn by name and appends the data to the given output file (jsonl)

    Args:
        entity (str): The entity name to scrape
        output_filename (str): The output file name to append the scraped data to (jsonl)
        entity_type (str): The type of entity to scrape
    """
    email = os.environ.get("email")
    password = os.environ.get("linkedpassword")
    api = Linkedin(email, password)
    try:
        entity_info = api.get_company(entity.replace(" ", "-").lower())
        scrape_data = {"entity_type": entity_type}
        for key, value in ALL_KEYS[entity_type].items():
            if isinstance(value, list):
                item = entity_info
                for v in value:
                    item = item.get(v)
                    if item is not None:
                        continue
                    else:
                        break
                scrape_data[key] = item
            else:
                scrape_data[key] = entity_info.get(value, None)
    except KeyError:
        scrape_data = {"entity_type": entity_type, "name": entity}
        for key in ALL_KEYS[entity_type].keys():
            if key != "name":
                scrape_data[key] = None
    finally:
        with open(output_filename, "a", encoding="utf-8") as file:
            json.dump(scrape_data, file)
            file.write("\n")


def get_entities_from_inference_file(input_filename):
    """Gather entities from given inference file after model execution (csv) in order to call scrape_entities()
    function to scrape information from LinkedIn.

    Args:
        input_filename (str): Inference file path from execution of inference.py

    Returns:
        list[str]: The list of named entities from the inference file
        list[str]: The list of the corresponding entity types from the inference file
    """
    with open(input_filename, "r") as file:
        reader = csv.reader(file)
        next(reader)
        entities = []
        entity_types = []
        for row in reader:
            entities.append(row[2])
            entity_types.append(row[0])
    return entities, entity_types


def scrape_entities(entities, output_filename, entity_types):
    """Scrape a list of entities from LinkedIn by name and creates a new file to store the data in jsonl format

    Args:
        entities (list[str]): List of entity names to scrape on LinkedIn
        output_filename (str): File name to export the data to (jsonl)
        entity_types (list[str]): List of the entity types that correspond to the entities list
    """
    email = os.environ.get("email")
    password = os.environ.get("linkedpassword")
    api = Linkedin(email, password)
    with open(output_filename, "w", encoding="utf-8") as file:
        for entity, entity_type in zip(entities, entity_types):
            try:
                entity_info = api.get_company(entity.replace(" ", "-").lower())
                scrape_data = {"entity_type": entity_type}
                for key, value in ALL_KEYS[entity_type].items():
                    if isinstance(value, list):
                        item = entity_info
                        for v in value:
                            item = item.get(v)
                            if item is not None:
                                continue
                            else:
                                break
                        scrape_data[key] = item
                    else:
                        scrape_data[key] = entity_info.get(value, None)
                json.dump(scrape_data, file)
            except KeyError:
                scrape_data = {"entity_type": entity_type, "name": entity}
                for key in ALL_KEYS[entity_type].keys():
                    if key != "name":
                        scrape_data[key] = None
                json.dump(scrape_data, file)
            finally:
                file.write("\n")


def create_scrape_checklist(entities_filename, output_filename):
    """Creates a checklist of the entities from a file to see if information was gathered on them and outputs a
    csv file with the entity name and whether or not it was scraped.

    Args:
        entities_filename (str): The file name of the file containing the entities to check (jsonl)
        output_filename (str): The file name of the file to output the checklist to (csv)
    """
    with open(entities_filename, "r", encoding="utf-8") as in_file:
        with open(output_filename, "w", encoding="utf-8", newline="") as out_file:
            header = ["entity", "found"]
            writer = csv.DictWriter(out_file, fieldnames=header)
            writer.writeheader()
            for line in in_file:
                entity_object = json.loads(line)
                keys = ALL_KEYS[entity_object["entity_type"]].keys()
                found = False
                for key in keys:
                    if key != "name" and entity_object[key] is not None:
                        found = True
                        break
                if found:
                    writer.writerow({"entity": entity_object["name"], "found": True})
                else:
                    writer.writerow({"entity": entity_object["name"], "found": False})


# def create_visualizations(
#     company_information_filemame, checklist_filename, entity_type
# ):
#     """Creates visualizations of the company information to various png files to showcase the feartures and the amount
#     of companies that were found on LinkedIn

#     Args:
#         company_information_filemame (str): The input file name containing the company information in jsonl format
#         checklist_filename (str): The input file name containing the checklist in csv format
#     """
#     query_count = {"Found": []}
#     with open(checklist_filename, "r") as file:
#         reader = csv.reader(file)
#         next(reader)
#         for row in reader:
#             if row[1] == "True":
#                 query_count["Found"].append("Found")
#             else:
#                 query_count["Found"].append("Not Found")
#     ax = sns.countplot(x="Found", data=query_count)
#     ax.bar_label(ax.containers[0])
#     plt.title(f"{entity_type} Information Found LinkedIn Querying")
#     plt.savefig(f"entity_scraping/{entity_type}_scrape_checklist.png")
#     company_info_objects = []
#     with open(company_information_filemame, "r") as file:
#         for line in file:
#             company_info_objects.append(json.loads(line))
#     feature_query_count = {"Feature": []}
#     for company_info in company_info_objects:
#         if len(company_info) > 1:
#             for key, _ in company_info.items():
#                 if key != "Name":
#                     feature_query_count["Feature"].append(key)
#     plt.clf()
#     plt.figure(figsize=(10, 16))
#     ax = sns.countplot(x="Feature", data=feature_query_count)
#     plt.xticks(rotation=90)
#     plt.title(f"{entity_type} Features Found LinkedIn Querying")
#     plt.savefig(f"entity_scraping/{entity_type}_scrape_features.png")
#     plt.clf()
"""
# Example Command Line Execution
python scrape_entities.py -i df_res.csv -o entity_information.jsonl
"""
if __name__ == "__main__":

    load_dotenv()
    # scrape_entities(
    #     parse_dictionary("data/QI-NERs.csv", "university"),
    #     "entity_scraping/university_scrape_info_linkedin.jsonl",
    #     [
    #         "university"
    #         for _ in range(len(parse_dictionary("data/QI-NERs.csv", "university")))
    #     ],
    # )
    entities, entity_types = get_entities_from_inference_file(args.input_file)
    scrape_entities(entities, args.output_file, entity_types)
    print(f"Entity scraping complete in file {args.output_file}")
    # create_scrape_checklist(
    #     "entity_scraping/university_scrape_info_linkedin.jsonl",
    #     "entity_scraping/university_scrape_checklist.csv",
    # )
