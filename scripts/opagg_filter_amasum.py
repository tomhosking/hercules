import argparse
import logging
import sys
print('Running Amasum filter dataset script with args:')
print(sys.argv)

parser = argparse.ArgumentParser(
    description="Filter Amasum dataset",
)

parser.add_argument(
 "--input_path", type=str, metavar="PATH", default='./data/amasum/min_10_revs_filt_complete/', help="Source dataset"
)

parser.add_argument("--split_constituents", action="store_true", help="")


args = parser.parse_args()

import jsonlines, os
from math import floor

from tqdm import tqdm
from collections import defaultdict

max_toks = 0

ds_name = 'amasum-%%cat%%-filtered'
LIMIT_MIN_REVIEWS = 100 # default 100
LIMIT_MAX_REVIEWS = 100 # default 50
LIMIT_MAX_ENTITIES = {
    'train': 10000,
    'valid': 1000,
    'test': 1000
}

LIMIT_TOKENS = 25 # default 25

LIMIT_PRONOUNS = None # default 1 for space, 0 for amasum?
LIMIT_FIRST_PERSON = 0
LIMIT_ASPECTS = False
FILTER_CHARS = True
SPLIT_CONSTITUENTS = args.split_constituents


if LIMIT_TOKENS is not None:
    ds_name = ds_name + '-{:}toks'.format(LIMIT_TOKENS)
if LIMIT_PRONOUNS is not None:
    ds_name = ds_name + '-{:}pronouns'.format(LIMIT_PRONOUNS)
if LIMIT_FIRST_PERSON is not None:
    ds_name = ds_name + '-{:}firstperson'.format(LIMIT_FIRST_PERSON)
if SPLIT_CONSTITUENTS:
    ds_name = ds_name + '-split'
if LIMIT_ASPECTS:
    ds_name = ds_name + '-aspects'
if FILTER_CHARS:
    ds_name = ds_name + '-charfilt'

import numpy as np

np.random.seed(1234)


all_sents = []

if LIMIT_PRONOUNS is not None:
    from flair.data import Sentence
    from flair.models import SequenceTagger

    flair_log = logging.getLogger("flair")
    flair_log.setLevel(30)

    # load the NER tagger
    tagger = SequenceTagger.load('pos-fast')

if LIMIT_ASPECTS:
    space_aspect_list = ["building", "cleanliness", "food", "location", "rooms", "service"]
    aspect_keywords = defaultdict(list)
    for aspect in space_aspect_list:
        with open(f"./data/opagg/aspect-seeds/{aspect}.txt") as f:
            keywords = [line.strip().split()[1] for line in f.readlines()]
        aspect_keywords[aspect] = keywords

    all_aspect_keywords = [kw for kws in aspect_keywords.values() for kw in kws]
    keywords_to_aspect = {kw: aspect for aspect, kws in aspect_keywords.items() for kw in kws}
    
    all_aspect_keywords += ['good', 'bad', 'ok', 'great', 'poor', 'fine', 'excellent', 'terrible', 'awful', 'disappointing']
    
if SPLIT_CONSTITUENTS:
    from allennlp.predictors.predictor import Predictor
    import allennlp_models.structured_prediction
    from allennlp.models.archival import load_archive
    import torch
    predictor = Predictor.from_archive(
        load_archive("https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz",
        cuda_device=torch.cuda.current_device()),
    )

filter_cats = {
    'electronics' : 'Electronics',
    'home-kitchen': 'Home & Kitchen',
    'sports-outdoors': 'Sports & Outdoors',
    'shoes': 'Shoes',
    'tools': 'Tools & Home Improvement'
}

# INclude in top 10:
# ('Electronics', 4175),
#  ('Home & Kitchen', 3994),
#  ('Sports & Outdoors', 2966),
#  ('Tools & Home Improvement', 1930),
#  ('Kitchen & Dining', 1910),
#  ('Shoes', 1797),
#  ('Clothing, Shoes & Jewelry', 1756),
#  ('Beauty & Personal Care', 1694),
#  ('Sports & Fitness', 1542),
#  ('Computers & Accessories', 1525),

# Unused:
#  ('Toys & Games', 1201),
#  ('Pet Supplies', 1197),
#  ('Shoes & Bags', 1184)
              

# filter_cats = set([x[0] for x in filter_cats])

def filter_condition(x):
    if "i bought" in x.lower() or "i purchased" in x.lower():
        return False
    if FILTER_CHARS:
        alphanum_ratio = sum(i.isalpha()for i in x)/len(x)
        if alphanum_ratio < 0.5:
            return False
    if LIMIT_TOKENS is not None and len(x.replace("\n", " ").split()) > LIMIT_TOKENS:
        return False 
    if LIMIT_ASPECTS:
        overlap = set(x.replace("\n", " ").lower().split()) & set(all_aspect_keywords)
        if len(overlap) == 0:
            return False
    if LIMIT_FIRST_PERSON is not None:
        if (x.lower().split().count('i') + x.lower().split().count('we') + x.lower().split().count('my')  + x.lower().split().count('our') + x.lower().split().count('me') + x.lower().split().count('us')) > LIMIT_FIRST_PERSON:
            return False
    if LIMIT_PRONOUNS is not None:
        sentence = Sentence(x)
        tagger.predict(sentence)
        pos_count = len([1 for span in sentence.get_labels('pos') if span.value in ['PRP','$PRP']])
        if pos_count > LIMIT_PRONOUNS:
            return False
    return True

import os,json
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import jsonlines
from torchseq.utils.functions import batchify

for cat_slug, filter_cat in filter_cats.items():
    
    ds_name_cat = ds_name.replace('%%cat%%', cat_slug)


    os.makedirs(f'./data/opagg/amasum-filtered/{ds_name_cat}-all/', exist_ok=True)
    os.makedirs(f'./data/opagg/amasum-filtered-spaceformat/{ds_name_cat}-all/', exist_ok=True)


    print('Category: ', filter_cat, ' Slug: ', ds_name_cat)

    for split in ['train','valid','test']:
        all_sentences = []
        spaceformat = []
        entity_count = 0
        for file in tqdm(os.listdir(args.input_path + '/' + split + '/'), "Processing {:}".format(split)):
            if entity_count > LIMIT_MAX_ENTITIES[split]:
                break
            with open(args.input_path + '/' + split + '/' + file) as f:
                data = json.load(f)
                entity_id = file.split('.')[0]
                if len(data['customer_reviews']) < LIMIT_MIN_REVIEWS:
                    continue
                if 'categories' not in data['product_meta']:
                    continue
                curr_cats = set(data['product_meta']['categories'])
                if len(curr_cats & set([filter_cat])) == 0:
                    continue
                entity_count += 1
                for review_id, review in enumerate(data['customer_reviews'][:LIMIT_MAX_REVIEWS]):
                    sentences = sent_tokenize(review['text'])
                    if SPLIT_CONSTITUENTS:
                        sents_split = []
                        for i, batch_inputs in batchify(sentences, batch_size=24):

                            parses = predictor.predict_batch_json([{'sentence': sent} for sent in batch_inputs])
                            for parse, sent in zip(parses, batch_inputs):
                                subsents = [node['word'] for node in parse['hierplane_tree']['root']['children'] if node['nodeType'] == 'S']
                                if len(subsents) > 0:
                                    sents_split.extend(subsents)
                                else:
                                    sents_split.append(sent)
                        sentences = sents_split
                    for sent in sentences:
                        if filter_condition(sent):
                            all_sentences.append({
                                'sentence': sent,
                                'entity_id': entity_id,
                                'review_id': review_id,
                                'rating': review['rating'],
                                'categories': data['product_meta']['categories']
                            })
                            
                spaceformat.append({
                    'entity_id': entity_id,
                    'reviews': [
                        {
                            'review_id': str(rev_id),
                            'sentences': sent_tokenize(review['text']),
                            'rating': review['rating']
                        } for rev_id, review in enumerate(data['customer_reviews'])
                    ],
                    
                })
                        
        split_txt = 'dev' if split == 'valid' else split
        with jsonlines.open(f'./data/opagg/amasum-filtered/{ds_name_cat}-all/reviews.{split_txt}.jsonl', 'w') as writer:
            writer.write_all(all_sentences)
        with jsonlines.open(f'./data/opagg/amasum-filtered-spaceformat/{ds_name_cat}-all/reviews.{split_txt}.jsonl', 'w') as writer:
            writer.write_all(spaceformat)
        print(split, entity_count, len(all_sentences))
        print(f'./data/opagg/amasum-filtered/{ds_name_cat}-all/reviews.{split_txt}.jsonl')
