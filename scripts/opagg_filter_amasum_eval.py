import argparse
import logging
import sys, os
print('Running Amasum Eval filter dataset script with args:')
print(sys.argv)

parser = argparse.ArgumentParser(
    description="Filter Amasum eval dataset",
)

parser.add_argument(
 "--input_path", type=str, metavar="PATH", default='./data/amasum/min_10_revs_filt_complete/', help="Source dataset"
)

# parser.add_argument(
#  "--dataset", type=str, metavar="PATH", default='space', help="Source dataset"
# )

parser.add_argument("--split_constituents", action="store_true", help="")


args = parser.parse_args()

import numpy as np
from collections import Counter
np.random.seed(123)
import json
from tqdm import tqdm
import jsonlines, os
from math import floor
from nltk.tokenize import sent_tokenize
from collections import defaultdict

EVAL_LIMIT = 50 # default 200
MIN_EVAL_REVIEWS = 100 # default 100
OUTPUT_SPLIT = args.split_constituents

if OUTPUT_SPLIT:
    from allennlp.predictors.predictor import Predictor
    import allennlp_models.structured_prediction
    from allennlp.models.archival import load_archive
    import torch
    predictor = Predictor.from_archive(
        load_archive("https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz",
        cuda_device=torch.cuda.current_device()),
    )
    from torchseq.utils.functions import batchify

filter_cats = {
    'electronics' : 'Electronics',
    'home-kitchen': 'Home & Kitchen',
    'sports-outdoors': 'Sports & Outdoors',
    'shoes': 'Shoes',
    'tools': 'Tools & Home Improvement'
}

# top-10:
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

#  unused:
#  ('Toys & Games', 1201),
#  ('Pet Supplies', 1197),
#  ('Shoes & Bags', 1184)
              

# filter_cats = set([x[0] for x in filter_cats])

for cat_slug, filter_cat in filter_cats.items():
    
    cat_str = '-'+cat_slug

    print('Category: ', filter_cat, ' Slug: ', cat_str)

    split_str = '-split' if OUTPUT_SPLIT else ''
    os.makedirs(f'./data/opagg/amasum{cat_str}-eval{split_str}/', exist_ok=True)

    for split in ['valid','test']:
        category_count = Counter()
        entities = []
        entity_count = 0
        num_reviews = []
        file_list = os.listdir(f'{args.input_path}/{split}/')
        np.random.shuffle(file_list)

        raw_objs = []
        for file in tqdm(file_list):
            with open(f'{args.input_path}/{split}/' + file) as f:
                data = json.load(f)
            entity_id = file.split('.')[0]
            if 'categories' not in data['product_meta']:
                    continue
            curr_cats = set(data['product_meta']['categories'])
            if len(curr_cats & set([filter_cat])) == 0:
                continue
            raw_objs.append({'entity_id': entity_id, **data})

        # print(Counter([len(x['website_summaries']) for x in raw_objs]))

        for data in tqdm(sorted(raw_objs, key=lambda obj: len(obj['website_summaries']), reverse=True)):
            if entity_count >= 100 and split == 'valid':
                break
            elif  entity_count >= EVAL_LIMIT:
                break
            
            # entity_id = file.split('.')[0]
            entity_id = data['entity_id']
            if len(data['customer_reviews']) < MIN_EVAL_REVIEWS:
                continue
            if 'categories' not in data['product_meta']:
                    continue
            curr_cats = set(data['product_meta']['categories'])
            if len(curr_cats & set([filter_cat])) == 0:
                continue
            entity_count += 1
            num_reviews.append(len(data['customer_reviews']))
            for cat in curr_cats:
                category_count[cat] += 1
    #             for review_id, review in enumerate(data['customer_reviews'][:100]):
    #                 sentences = sent_tokenize(review['text'])
    #                 for sent in sentences:
    #                     all_sentences.append({
    #                         'sentence': sent,
    #                         'entity_id': entity_id,
    #                         'review_id': review_id,
    #                         'rating': review['rating']
    #                     })

            if OUTPUT_SPLIT:
                reviews = []
                for review_id, rev in enumerate(data['customer_reviews']):
                    sents = []
                    for i, batch in batchify(sent_tokenize(rev['text']), 24):
                        parses = predictor.predict_batch_json([{'sentence': sent} for sent in batch])
                        for parse, sent in zip(parses, batch):
                            subsents = [node['word'] for node in parse['hierplane_tree']['root']['children'] if node['nodeType'] == 'S']
                            if len(subsents) > 0:
                                sents.extend(subsents)
                            else:
                                sents.append(sent)
                    reviews.append({
                        'review_id': review_id,
                        'sentences': sents,
                        'rating': rev['rating']
                    })

            else:
                reviews = [
                    {
                        'review_id': review_id,
                        'sentences': sent_tokenize(review['text']),
                        'rating': review['rating']
                    }
                    for review_id, review in enumerate(data['customer_reviews'])
                ]
            entities.append({
                'entity_id': entity_id,
                'categories': data['product_meta']['categories'],
                'reviews': reviews,
                'summaries': [
                        (summ['verdict'] + ". "+ ". ".join(summ['pros'])+ ". " + ". ".join(summ['cons'])) .replace('.. ','. ')
                    for summ in data['website_summaries']
                ],
            })
        split_txt = 'dev' if split == 'valid' else split

        # with jsonlines.open(f'./data/opagg/amasum{cat_str}-eval{split_str}/{split_txt}.jsonl', 'w') as writer:
        #     writer.write_all(entities)
        print(split, entity_count, np.min(num_reviews))
        # print(category_count.most_common(30))
        print(Counter([len(x['summaries']) for x in entities]))
    #     break