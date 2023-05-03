import argparse

import sys
print('Running Space filter dataset script with args:')
print(sys.argv)

parser = argparse.ArgumentParser(
    description="Filter Space dataset",
)

parser.add_argument(
 "--input_path", type=str, metavar="PATH", default='./data/space/', help="Source dataset"
)

parser.add_argument("--split_constituents", action="store_true", help="")
parser.add_argument("--aspectsv2", action="store_true", help="")

args = parser.parse_args()

import json
with open(args.input_path + "/space_train.json") as f:
    space = json.load(f)

import jsonlines, os
from math import floor

from tqdm import tqdm
from collections import defaultdict
import torch

from torchseq.utils.functions import batchify

from nltk import word_tokenize

max_toks = 0

ds_name = 'space-filtered'
LIMIT_MIN_REVIEWS = 10
LIMIT_MAX_REVIEWS = 50
LIMIT_MAX_ENTITIES = 4000
LIMIT_TOKENS = 25

LIMIT_PRONOUNS = 1
LIMIT_FIRST_PERSON = None
LIMIT_ASPECTS = True
LIMIT_ASPECTS_V2 = args.aspectsv2
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
if LIMIT_ASPECTS_V2:
    ds_name = ds_name + 'v2'
if FILTER_CHARS:
    ds_name = ds_name + '-charfilt'

import numpy as np

np.random.seed(1234)

space_filtered = []

i = 0
for entity in space:
    if len(entity['reviews']) >= LIMIT_MIN_REVIEWS:
        if i > LIMIT_MAX_ENTITIES:
            break
        i += 1
        entity['reviews'] = entity['reviews'][:LIMIT_MAX_REVIEWS]
        space_filtered.append(entity)
        
np.random.shuffle(space_filtered)

dev_count = floor(len(space_filtered) * 0.05)

os.makedirs(f'./data/opagg/space-filtered/{ds_name}-clusters/', exist_ok=True)
os.makedirs(f'./data/opagg/space-filtered/{ds_name}-all/', exist_ok=True)

clusters = []
all_sents = []

if SPLIT_CONSTITUENTS:
    from allennlp.predictors.predictor import Predictor
    import allennlp_models.structured_prediction
    from allennlp.models.archival import load_archive
    predictor = Predictor.from_archive(
        load_archive("https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz",
        cuda_device=torch.cuda.current_device()),
    )

if LIMIT_PRONOUNS is not None:
    from flair.data import Sentence
    from flair.models import SequenceTagger
    import logging

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
    all_aspect_keywords += ['amazing','special', 'fantastic', 'wonderful']
    if LIMIT_ASPECTS_V2:
        all_aspect_keywords += ['rooms', 'bed','beds', 'cookie', 'cookies', 'cheap', 'expensive', 'positive', 'negative','quick','slow','fast','better','worse','worn','new','modern','lovely','wifi','recommend', 'restaurant','restaurants','shuttle','airport','parking','light','dark', 'luxurious', 'luxury', 'price','priced','overpriced','tired','huge','tiny']

def filter_condition(x):
    if FILTER_CHARS:
        alphanum_ratio = sum(i.isalpha()for i in x)/len(x)
        if alphanum_ratio < 0.5:
            return False
    if LIMIT_TOKENS is not None and len(x.replace("\n", " ").split()) > LIMIT_TOKENS:
        return False 
    if LIMIT_ASPECTS:
        overlap = set(word_tokenize(x.replace("\n", " ").lower())) & set(all_aspect_keywords)
        if len(overlap) == 0:
            return False
    if LIMIT_FIRST_PERSON is not None:
        if (x.lower().split().count('i') + x.lower().split().count('we') + x.lower().split().count('my')  + x.lower().split().count('our') + x.lower().split().count('me') + x.lower().split().count('us')) > LIMIT_FIRST_PERSON:
            return False
    if LIMIT_PRONOUNS is not None:
        tagged = Sentence(x)
        tagger.predict(tagged)
        
        pos_count = len([1 for span in tagged.get_labels('pos') if span.value in ['PRP','$PRP']])
        if pos_count > LIMIT_PRONOUNS:
            return False
    return True


print('Start..')
# Start on train split
sents = [{'sentence': sent.replace("\n", " "), 'entity_id': row['entity_id'], 'review_id': review['review_id'], 'rating': review['rating']} for row in space_filtered[:-dev_count] for review in row['reviews'] for sent in review['sentences'] if len(sent.split()) < 100]
print('{:} sents input'.format(len(sents)))

# Expand by splitting into constituent S nodes
if SPLIT_CONSTITUENTS:
    sents_split = []
    for i, batch_inputs in tqdm(batchify(sents, batch_size=24), desc='Splitting train'):
        
        parses = predictor.predict_batch_json([{'sentence': sent['sentence']} for sent in batch_inputs])
        for parse, sent in zip(parses, batch_inputs):
            subsents = [node['word'] for node in parse['hierplane_tree']['root']['children'] if node['nodeType'] == 'S']
            if len(subsents) > 0:
                sents_split.extend([{**sent, 'sentence': subsent} for subsent in subsents])
            else:
                sents_split.append(sent)
    sents = sents_split
    
# tagged = None
# if LIMIT_PRONOUNS:
#     print('Tagging...')
#     tagged = [Sentence(x['sentence']) for x in sents]
#     tagger.predict(tagged, mini_batch_size=256, verbose=True)
#     print('..done!')
    
# Apply filter
sents_filt = [x for x in tqdm(sents, desc='Train filtering') if filter_condition(x['sentence'])]
# sents_filt = [x for i, x in tqdm(enumerate(sents), desc='Train filtering') if filter_condition(x['sentence'], tagged[i] if tagged is not None else None)]

# for row in tqdm():
#     reviews_whole = []
#     :
# #         reviews_whole.append(" ".join(review['sentences']).replace("\n", " "))
        
#         if SPLIT_CONSTITUENTS:
#             sents = []
#             sents_orig = review['sentences']
#             for sent in sents_orig:
#                 parse = predictor.predict(sent)
#                 subsents = [node['word'] for node in parse['hierplane_tree']['root']['children'] if node['nodeType'] == 'S']
#                 if len(subsents) > 0:
#                     sents.extend(subsents)
#                 else:
#                     sents.append(sent)
#         else:
#             sents = review['sentences']
        
#         if len(sents) == 0:
#             continue
#         for sent in sents:
            
#             all_sents.append()

#     clusters.append({'reviews': reviews_whole})
    
with jsonlines.open(f'./data/opagg/space-filtered/{ds_name}-all/reviews.train.jsonl', 'w') as f:
    f.write_all(sents_filt)
    
print('Train: ', len(sents_filt))
    

# all_sents = []
# for row in tqdm(space_filtered[-dev_count:]):
#     reviews_whole = []
#     for review in row['reviews']:
# #         reviews_whole.append(" ".join(review['sentences']).replace("\n", " "))
#         if SPLIT_CONSTITUENTS:
#             sents = []
#             sents_orig = review['sentences']
#             for sent in sents_orig:
#                 parse = predictor.predict(sent)
#                 subsents = [node['word'] for node in parse['hierplane_tree']['root']['children'] if node['nodeType'] == 'S']
#                 if len(subsents) > 0:
#                     sents.extend(subsents)
#                 else:
#                     sents.append(sent)
#         else:
#             sents = review['sentences']
#         sents = [x.replace("\n", " ") for x in sents if filter_condition(x.replace("\n", " "))]
#         if len(sents) == 0:
#             continue
#         for sent in sents:
#             all_sents.append({'sentence': sent, 'entity_id': row['entity_id'], 'review_id': review['review_id'], 'rating': review['rating']})
#         reviews_whole.extend(sents)
#     clusters.append({'reviews': reviews_whole})
    
# with jsonlines.open(f'./data/opagg/space-filtered/{ds_name}-clusters/space_reviews.dev.jsonl', 'w') as f:
#     f.write_all(clusters)


# Start on dev split
sents = [{'sentence': sent.replace("\n", " "), 'entity_id': row['entity_id'], 'review_id': review['review_id'], 'rating': review['rating']} for row in space_filtered[-dev_count:] for review in row['reviews'] for sent in review['sentences'] if len(sent.split()) < 100]
print('{:} sents input for dev'.format(len(sents)))

# Expand by splitting into constituent S nodes
if SPLIT_CONSTITUENTS:
    sents_split = []
    for i, batch_inputs in tqdm(batchify(sents, batch_size=24), desc='Splitting dev'):
        
        parses = predictor.predict_batch_json([{'sentence': sent['sentence']} for sent in batch_inputs])
        for parse, sent in zip(parses, batch_inputs):
            subsents = [node['word'] for node in parse['hierplane_tree']['root']['children'] if node['nodeType'] == 'S']
            if len(subsents) > 0:
                sents_split.extend([{**sent, 'sentence': subsent} for subsent in subsents])
            else:
                sents_split.append(sent)
    sents = sents_split
    
# tagged = None
# if LIMIT_PRONOUNS:
#     print('Tagging...')
#     tagged = [Sentence(x['sentence']) for x in sents]
#     tagger.predict(tagged, mini_batch_size=32, verbose=True)
#     print('..done!')
    
# Apply filter
sents_filt = [x for x in tqdm(sents, desc='Dev filtering') if filter_condition(x['sentence'])]
# sents_filt = [x for i, x in tqdm(enumerate(sents), desc='Train filtering') if filter_condition(x['sentence'], tagged[i] if tagged is not None else None)]

with jsonlines.open(f'./data/opagg/space-filtered/{ds_name}-all/reviews.dev.jsonl', 'w') as f:
    f.write_all([x for x in sents_filt])
    
print('Dev: ', len(sents_filt))
