######
# Convert Label Studio completions to train and dev data,
# in the json format required for spacy train CLI
#
# Completions are NOT included if:
#   - there is more than one completion for the task, or
#   - the task was skipped (cancelled)
######

import spacy
import srsly
from spacy.gold import docs_to_json, biluo_tags_from_offsets, spans_from_biluo_tags
import argparse
from zipfile import ZipFile
import json
import random


##
# Return a list of spacy docs, converted from Label Studio completions.
# Skip cancelled tasks and tasks with multiple completions.
##
def ls_to_spacy_json(ls_completions):
    nlp = spacy.load('en_core_web_sm')

    # Load the Label Studio completions
    with ZipFile(ls_completions, 'r') as zip:
        result_file = zip.read('result.json')
        label_studio_json = json.loads(result_file)

    gold_docs = []
    entity_cnt = 0
    for task in label_studio_json:
        completions = task['completions']

        # don't include skipped tasks or tasks with multiple completions
        if len(completions) == 1:
            completion = completions[0]
            if 'was_cancelled' in completion:
                continue

            raw_text = task['data']['reddit']
            annotated_entities = []
            for result in completion['result']:
                ent = result['value']
                start_char_offset = ent['start']
                end_char_offset = ent['end']
                ent_label = ent['labels'][0]
                entity = (start_char_offset, end_char_offset, ent_label)
                annotated_entities.append(entity)

            doc = nlp(raw_text)
            tags = biluo_tags_from_offsets(doc, annotated_entities)
            entities = spans_from_biluo_tags(doc, tags)
            doc.ents = entities
            gold_docs.append(doc)
            entity_cnt += len(annotated_entities)

    print("{} entities in {} docs.".format(str(entity_cnt), len(gold_docs)))
    return gold_docs


def entity_count(docs):
    count = 0
    for doc in docs:
        count += len(doc.ents)
    return count

##
# Split the docs into training and dev,
# and save to 2 files
##
def save_train_dev_data(gold_docs, split, train_file, dev_file):
    # shuffle the docs
    random.seed(27)
    random.shuffle(gold_docs)

    # split the gold data into training and evaluation
    num_training_tasks = round(len(gold_docs) * split / 100)
    train_docs = gold_docs[:num_training_tasks]
    dev_docs = gold_docs[num_training_tasks:]

    print("{} training entities".format(str(entity_count(train_docs))))
    print("{} dev entities".format(str(entity_count(dev_docs))))

    srsly.write_json(train_file, [docs_to_json(train_docs)])
    srsly.write_json(dev_file, [docs_to_json(dev_docs)])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("ls_completions", help="completions exported from Label Studio")
    parser.add_argument("split", type=int, help="percentage of data to use for training")
    parser.add_argument("train_file", help="file to save training data")
    parser.add_argument("dev_file", help="file to save dev data")
    return parser.parse_args()


def main(args):
    gold_docs = ls_to_spacy_json(args.ls_completions)
    save_train_dev_data(gold_docs, args.split, args.train_file, args.dev_file)


if __name__ == '__main__':
    main(parse_args())
