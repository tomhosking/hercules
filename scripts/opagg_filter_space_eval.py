# EVAL

ds_name = 'space'

parser = argparse.ArgumentParser(
    description="Filter Space dataset",
)

parser.add_argument(
 "--input_path", type=str, metavar="PATH", default='./data/space/', help="Source dataset"
)

args = parser.parse_args()

import json
with open(f'{args.input_path}/space_summ.json') as f:
    space_eval = json.load(f)
    
with open(f'{args.input_path}/space_summ_splits.txt') as f:
    split_ids = {x.strip().split('\t')[0]: x.strip().split('\t')[1] for x in f.readlines()}

import jsonlines, os, torch
from tqdm import tqdm

eval_rows_dev = []
eval_rows_test = []

summary_aspect = 'general'

from allennlp.predictors.predictor import Predictor
import allennlp_models.structured_prediction
from allennlp.models.archival import load_archive
predictor = Predictor.from_archive(
    load_archive("https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz",
    cuda_device=torch.cuda.current_device()),
)
from torchseq.utils.functions import batchify

OUTPUT_SPLIT = False

for ent in tqdm(space_eval):
#     reviews = [[sent for sent in review['sentences']] for review in ent['reviews']]
    summaries = ent['summaries'][summary_aspect]
    
    if OUTPUT_SPLIT:
        reviews = []
        for rev in ent['reviews']:
            sents = []
            for i, batch in batchify(rev['sentences'], 24):
                parses = predictor.predict_batch_json([{'sentence': sent} for sent in batch])
                for parse, sent in zip(parses, batch):
                    subsents = [node['word'] for node in parse['hierplane_tree']['root']['children'] if node['nodeType'] == 'S']
                    if len(subsents) > 0:
                        sents.extend(subsents)
                    else:
                        sents.append(sent)
                        
            reviews.append({**rev, 'sentences': sents})
            
    if split_ids[ent['entity_id']] == 'dev':
        eval_rows_dev.append({
            'entity_id': ent['entity_id'],
            'reviews': reviews if OUTPUT_SPLIT else ent['reviews'],
            'summaries': summaries
        })
    elif split_ids[ent['entity_id']] == 'test':
        eval_rows_test.append({
            'entity_id': ent['entity_id'],
            'reviews': reviews if OUTPUT_SPLIT else ent['reviews'],
            'summaries': summaries
        })
        
print(len(eval_rows_dev), len(eval_rows_test))

split_str = '-split' if OUTPUT_SPLIT else ''
os.makedirs(f'./data/opagg/{ds_name}-eval{split_str}/', exist_ok=True)

with jsonlines.open(f'./data/opagg/{ds_name}-eval{split_str}/dev.jsonl', 'w') as f:
    f.write_all(eval_rows_dev)
with jsonlines.open(f'./data/opagg/{ds_name}-eval{split_str}/test.jsonl', 'w') as f:
    f.write_all(eval_rows_test)