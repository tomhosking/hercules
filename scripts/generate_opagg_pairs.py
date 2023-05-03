from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import scipy

import numpy as np
import jsonlines, os

from tqdm import tqdm
from collections import defaultdict
from nltk.tokenize import word_tokenize
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet


import pkg_resources
from symspellpy import SymSpell, Verbosity


import argparse

import sys
print('Running OpAgg dataset script with args:')
print(sys.argv)

parser = argparse.ArgumentParser(
    description="WikiAnswers to 3way script",
)
parser.add_argument(
 "--data_dir", type=str, metavar="PATH", default='./data/', help="Path to data folder"
)

parser.add_argument(
 "--dataset", type=str, metavar="PATH", default='space', help="Source dataset"
)

parser.add_argument(
 "--similarity_method", type=str, default='tfidf', help=""
)

parser.add_argument(
 "--min_overlap", type=float,default=0.85, help=""
)
parser.add_argument(
 "--num_distractors", type=int,  default=3, help=""
)
parser.add_argument(
 "--num_cores", type=int,  default=4, help=""
)
parser.add_argument(
 "--max_samples", type=int,  default=None, help=""
)
parser.add_argument(
 "--max_tokens", type=int,  default=None, help=""
)
parser.add_argument(
 "--min_sents", type=int,  default=None, help=""
)
parser.add_argument(
 "--include_self_fraction", type=float,default=None, help=""
)

parser.add_argument("--include_self", action="store_true", help="")
parser.add_argument("--ignore_stopwords", action="store_true", help="")
parser.add_argument("--enforce_aspect_consistency", action="store_true", help="")
parser.add_argument("--enforce_rating_consistency", action="store_true", help="")
parser.add_argument("--enforce_category_consistency", action="store_true", help="")
parser.add_argument("--enforce_negation_consistency", action="store_true", help="")
parser.add_argument("--filter_irrelevant", action="store_true", help="")
parser.add_argument("--include_self_if_no_match", action="store_true", help="")
parser.add_argument("--check_rte", action="store_true", help="")
parser.add_argument("--spellcheck", action="store_true", help="")
parser.add_argument("--expand_src_wordnet", action="store_true", help="")
parser.add_argument("--expand_tgt_wordnet", action="store_true", help="")
parser.add_argument("--expand_include_similar", action="store_true", help="")

args = parser.parse_args()

INPUT_DATASET_NAME = args.dataset

dataset_slug = INPUT_DATASET_NAME + '-' + args.similarity_method + '-overlap' + "{:0.2f}".format(args.min_overlap).replace('.','') + '-num' + str(args.num_distractors)
if args.include_self:
    dataset_slug += '-includeself'
if args.include_self_if_no_match:
    dataset_slug += '-includeselfifnomatch'
if args.include_self_fraction:
    dataset_slug += '-includeself{:0.2f}'.format(args.include_self_fraction) 
if args.ignore_stopwords:
    dataset_slug += '-ignorestopwords'
if args.enforce_aspect_consistency:
    dataset_slug += '-enforceaspects'
if args.enforce_rating_consistency:
    dataset_slug += '-enforceratings'
if args.enforce_category_consistency:
    dataset_slug += '-enforcecategories'
if args.enforce_negation_consistency:
    dataset_slug += '-enforcenegation'
if args.filter_irrelevant:
    dataset_slug += '-filterirrelevant'
if args.check_rte:
    dataset_slug += '-checkrte'
if args.spellcheck:
    dataset_slug += '-spellcheck'
if args.expand_src_wordnet:
    dataset_slug += '-expandsrcwordnet'
if args.expand_tgt_wordnet:
    dataset_slug += '-expandtgtwordnet'
if args.expand_include_similar:
    dataset_slug += '-expandsimilar'
if args.max_samples is not None:
    dataset_slug += '-LIMIT'+str(args.max_samples)
if args.max_tokens is not None:
    dataset_slug += '-TOKS'+str(args.max_tokens)
if args.min_sents is not None:
    dataset_slug += '-minsents'+str(args.min_sents)

print('Dataset slug will be:')
print(dataset_slug)
    
if INPUT_DATASET_NAME == 'space':
    DATASET_ALL = f'opagg/{INPUT_DATASET_NAME}-filtered-all'
    DATASET_NOISED = f'opagg/{INPUT_DATASET_NAME}-noised/' + dataset_slug
elif INPUT_DATASET_NAME == 'space-25toks-1pronouns':
    DATASET_ALL = f'opagg/space-filtered/space-filtered-25toks-1pronouns-charfilt-all'
    DATASET_NOISED = f'opagg/space-noised/' + dataset_slug
elif INPUT_DATASET_NAME == 'space-25toks-1pronouns-aspects':
    DATASET_ALL = f'opagg/space-filtered/space-filtered-25toks-1pronouns-aspects-charfilt-all'
    DATASET_NOISED = f'opagg/space-noised/' + dataset_slug
elif INPUT_DATASET_NAME == 'space-25toks-1pronouns-aspectsv2':
    DATASET_ALL = f'opagg/space-filtered/space-filtered-25toks-1pronouns-aspectsv2-charfilt-all'
    DATASET_NOISED = f'opagg/space-noised/' + dataset_slug
elif INPUT_DATASET_NAME == 'space-25toks-1pronouns-split-aspectsv2':
    DATASET_ALL = f'opagg/space-filtered/space-filtered-25toks-1pronouns-split-aspectsv2-charfilt-all'
    DATASET_NOISED = f'opagg/space-noised/' + dataset_slug
elif INPUT_DATASET_NAME == 'space-25toks-1firstperson-aspects':
    DATASET_ALL = f'opagg/space-filtered/space-filtered-25toks-1firstperson-aspects-charfilt-all'
    DATASET_NOISED = f'opagg/space-noised/' + dataset_slug
elif INPUT_DATASET_NAME == 'space-20toks-1pronouns-aspects':
    DATASET_ALL = f'opagg/space-filtered/space-filtered-20toks-1pronouns-aspects-charfilt-all'
    DATASET_NOISED = f'opagg/space-noised/' + dataset_slug
elif INPUT_DATASET_NAME == 'space-15toks-1pronouns-aspects':
    DATASET_ALL = f'opagg/space-filtered/space-filtered-15toks-1pronouns-aspects-charfilt-all'
    DATASET_NOISED = f'opagg/space-noised/' + dataset_slug
elif INPUT_DATASET_NAME == 'oposumplus-25toks':
    DATASET_ALL = f'opagg/oposumplus-filtered/oposumplus-filtered-25toks-charfilt-all'
    DATASET_NOISED = f'opagg/oposumplus-noised/' + dataset_slug
elif INPUT_DATASET_NAME == 'amasum-25toks':
    DATASET_ALL = f'opagg/amasum-filtered/amasum-filtered-25toks-charfilt-all'
    DATASET_NOISED = f'opagg/amasum-noised/' + dataset_slug
elif INPUT_DATASET_NAME == 'amasum-25toks-1pronouns':
    DATASET_ALL = f'opagg/amasum-filtered/amasum-filtered-25toks-1pronouns-charfilt-all'
    DATASET_NOISED = f'opagg/amasum-noised/' + dataset_slug
elif INPUT_DATASET_NAME == 'amasum-electronics-25toks-1pronouns':
    DATASET_ALL = f'opagg/amasum-filtered/amasum-filtered-electronics-25toks-1pronouns-charfilt-all'
    DATASET_NOISED = f'opagg/amasum-noised/' + dataset_slug

elif INPUT_DATASET_NAME == 'amasum-electronics-25toks-0pronouns':
    DATASET_ALL = f'opagg/amasum-filtered/amasum-electronics-filtered-25toks-0pronouns-charfilt-all'
    DATASET_NOISED = f'opagg/amasum-noised/' + dataset_slug
elif INPUT_DATASET_NAME == 'amasum-shoes-25toks-0pronouns':
    DATASET_ALL = f'opagg/amasum-filtered/amasum-shoes-filtered-25toks-0pronouns-charfilt-all'
    DATASET_NOISED = f'opagg/amasum-noised/' + dataset_slug
elif INPUT_DATASET_NAME == 'amasum-home-kitchen-25toks-0pronouns':
    DATASET_ALL = f'opagg/amasum-filtered/amasum-home-kitchen-filtered-25toks-0pronouns-charfilt-all'
    DATASET_NOISED = f'opagg/amasum-noised/' + dataset_slug
elif INPUT_DATASET_NAME == 'amasum-sports-outdoors-25toks-0pronouns':
    DATASET_ALL = f'opagg/amasum-filtered/amasum-sports-outdoors-filtered-25toks-0pronouns-charfilt-all'
    DATASET_NOISED = f'opagg/amasum-noised/' + dataset_slug
elif INPUT_DATASET_NAME == 'amasum-tools-25toks-0pronouns':
    DATASET_ALL = f'opagg/amasum-filtered/amasum-tools-filtered-25toks-0pronouns-charfilt-all'
    DATASET_NOISED = f'opagg/amasum-noised/' + dataset_slug

elif INPUT_DATASET_NAME == 'amasum-electronics-25toks-0firstperson':
    DATASET_ALL = f'opagg/amasum-filtered/amasum-electronics-filtered-25toks-0firstperson-charfilt-all'
    DATASET_NOISED = f'opagg/amasum-noised/' + dataset_slug
elif INPUT_DATASET_NAME == 'amasum-shoes-25toks-0firstperson':
    DATASET_ALL = f'opagg/amasum-filtered/amasum-shoes-filtered-25toks-0firstperson-charfilt-all'
    DATASET_NOISED = f'opagg/amasum-noised/' + dataset_slug
elif INPUT_DATASET_NAME == 'amasum-home-kitchen-25toks-0firstperson':
    DATASET_ALL = f'opagg/amasum-filtered/amasum-home-kitchen-filtered-25toks-0firstperson-charfilt-all'
    DATASET_NOISED = f'opagg/amasum-noised/' + dataset_slug
elif INPUT_DATASET_NAME == 'amasum-sports-outdoors-25toks-0firstperson':
    DATASET_ALL = f'opagg/amasum-filtered/amasum-sports-outdoors-filtered-25toks-0firstperson-charfilt-all'
    DATASET_NOISED = f'opagg/amasum-noised/' + dataset_slug
elif INPUT_DATASET_NAME == 'amasum-tools-25toks-0firstperson':
    DATASET_ALL = f'opagg/amasum-filtered/amasum-tools-filtered-25toks-0firstperson-charfilt-all'
    DATASET_NOISED = f'opagg/amasum-noised/' + dataset_slug
else:
    raise Exception("Unknown input dataset!")

DATA_PATH = args.data_dir
MIN_OVERLAP = args.min_overlap

os.makedirs(os.path.join(DATA_PATH, DATASET_NOISED), exist_ok=True)

space_aspect_list = ["building", "cleanliness", "food", "location", "rooms", "service"]
aspect_keywords = defaultdict(list)
for aspect in space_aspect_list:
    with open(args.data_dir + f"/opagg/aspect-seeds/{aspect}.txt") as f:
        keywords = [line.strip().split()[1] for line in f.readlines()]
    aspect_keywords[aspect] = keywords

all_aspect_keywords = [kw for kws in aspect_keywords.values() for kw in kws]
keywords_to_aspect = {kw: aspect for aspect, kws in aspect_keywords.items() for kw in kws}

def filter_irrelevant(sentence):
    toks = sentence.split()
    for kw in all_aspect_keywords:
        if kw in toks:
            return True
    return False

def get_aspect_label(sentence):
    labels = set()
    for kw, aspect in keywords_to_aspect.items():
        if kw in sentence.split():
            labels.add(aspect)
    if len(labels) == 0:
        labels.add("UNK")
    return labels


# Load spellcheck
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")

sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# Load wordnet synsets (not used, except to avoid multithread issues)
from itertools import chain

synsets = {}
lemmas_in_wordnet = set(chain(*[x.lemma_names() for x in wordnet.all_synsets()]))

for lemma in lemmas_in_wordnet:
    synsets[lemma] = wordnet.synsets(lemma)

    
def spellcheck(q):
    output_q = []
    for tok in word_tokenize(q):
        suggestions = sym_spell.lookup(tok.lower(), Verbosity.CLOSEST,
                               max_edit_distance=1, include_unknown=True, ignore_token=r"[^A-Za-z]+")
        output_q.append(suggestions[0]._term)
    return " ".join(output_q)

def expand_sentence(sent, expand_similar=False):
    tag_to_tag = {
        'NN': 'n',
        'JJ': 'a'
    }
    # toks = [spellcheck(tok) for tok in word_tokenize(sent.lower())]
    toks = word_tokenize(sent.lower())
    pos_tags = nltk.pos_tag(toks)
    candidates = [[]]
    for tok, tag in zip(toks, pos_tags):
        replace = [tok]
        if tag[1] in ['NN','JJ']:
            synsets = [syns for syns in wordnet.synsets(tok.lower(), pos=tag_to_tag.get(tag[1], None))]
            if len(synsets) > 0 and len(synsets[0].hypernyms()) > 0:
                replace.append(synsets[0].hypernyms()[0].lemmas()[0].name().replace('_', ' '))
#                     break
            if len(synsets) > 0 and expand_similar:
                replace.extend([x.lemmas()[0].name().replace('_', ' ') for x in synsets[0].similar_tos()])

        candidates = [candidate + [expansion] for candidate in candidates for expansion in replace]
    return [" ".join(x) for x in candidates]

with jsonlines.open(os.path.join(DATA_PATH, DATASET_ALL, 'reviews.train.jsonl')) as f:
    space_filtered = [x for x in f if (not args.filter_irrelevant or filter_irrelevant(x['sentence']))]
with jsonlines.open(os.path.join(DATA_PATH, DATASET_ALL, 'reviews.dev.jsonl')) as f:
    space_filtered_dev = [x for x in f if (not args.filter_irrelevant or filter_irrelevant(x['sentence']))]


sent_counts_train = defaultdict(int)
sent_counts_dev = defaultdict(int)
for row in space_filtered:
    sent_counts_train[(row['entity_id'], row['review_id'])] +=1
for row in space_filtered_dev:
    sent_counts_dev[(row['entity_id'], row['review_id'])] +=1


all_sents = [spellcheck(x['sentence']) if args.spellcheck else x['sentence'] for x in space_filtered if args.min_sents is None or sent_counts_train[(x['entity_id'], x['review_id'])] >= args.min_sents]
all_sents_dev = [spellcheck(x['sentence']) if args.spellcheck else x['sentence'] for x in space_filtered_dev if args.min_sents is None or sent_counts_dev[(x['entity_id'], x['review_id'])] >= args.min_sents]

if args.enforce_category_consistency:
    all_categories = [x['categories'] for x in space_filtered]
    all_categories_dev = [x['categories'] for x in space_filtered_dev]
else:
    all_categories = None
    all_categories_dev = None

all_sent_ratings = [x['rating'] for x in space_filtered]
all_sent_ratings_dev = [x['rating'] for x in space_filtered_dev]


print('Loaded {:} train and {:} dev sentences'.format(len(all_sents), len(all_sents_dev)))
 

embedded = None
embedded_dev = None
all_sents_bow = [set(word_tokenize(sent)) for sent in all_sents]
all_sents_dev_bow = [set(word_tokenize(sent)) for sent in all_sents_dev]

if args.similarity_method != 'bow':
    print('Fitting tfidf')
    if args.similarity_method == 'tfidf2':
        vectorizer = TfidfVectorizer(
                stop_words=('english' if args.ignore_stopwords else None),
            min_df=10,
            max_df=0.5 if args.ignore_stopwords else 1.0,
            sublinear_tf=False,
            ngram_range=(1,2),
            norm=None,
        )
    else:
        vectorizer = TfidfVectorizer(
            stop_words=('english' if args.ignore_stopwords else None),
            min_df=5,
            # max_df=0.8 if args.ignore_stopwords else 1.0,
        )

    embedded = vectorizer.fit_transform(all_sents)

    print('Transforming dev')
    embedded_dev = vectorizer.transform(all_sents_dev)
    # all_sents_bow = [None for sent in all_sents]
    # all_sents_dev_bow = [None for sent in all_sents_dev]

    import time
    start = time.time()
    print('Finding nearest neighbours...')
    print('** This can take a long time!! ~12h for a dataset with 1m input samples **')
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=args.num_distractors * 25, algorithm='auto', metric='cosine').fit(embedded)
    distances, indices = nbrs.radius_neighbors(embedded[:args.max_samples], radius=1-MIN_OVERLAP)

    nbrs_dev = NearestNeighbors(n_neighbors=args.num_distractors * 25, algorithm='auto', metric='cosine').fit(embedded_dev)
    distances_dev, indices_dev = nbrs_dev.radius_neighbors(embedded_dev[:args.max_samples], radius=1-MIN_OVERLAP)

    end = time.time()
    print('done in {:}s'.format(end-start))


# print('Finding nearest neighbours')

# nbrs = NearestNeighbors(n_neighbors=6, algorithm='auto', n_jobs=-1).fit(embedded)
# distances, indices = nbrs.kneighbors(embedded)

# print('Finding nearest neighbours (dev)')

# nbrs_dev = NearestNeighbors(n_neighbors=6, algorithm='auto', n_jobs=-1).fit(embedded_dev)
# distances_dev, indices_dev = nbrs_dev.kneighbors(embedded_dev)

print('Building dataset')

if args.check_rte:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    rte_tokenizer = AutoTokenizer.from_pretrained("tomhosking/deberta-v3-base-debiased-nli")

    rte_model = AutoModelForSequenceClassification.from_pretrained("tomhosking/deberta-v3-base-debiased-nli").cuda()

    ENTAILMENT_LABEL = (
        rte_model.config.label2id["ENTAILMENT"]
        if "ENTAILMENT" in rte_model.config.label2id
        else rte_model.config.label2id["entailment"]
    )

def directional_dotprod(x,Y):
    numerator = x.dot(Y.T).toarray()
    denominator = scipy.sparse.linalg.norm(x)**2# *np.sqrt(scipy.sparse.linalg.norm(Y, axis=-1))

    return numerator/(denominator+1e-10)

def bow_subsumption(x, Y):
    scores = [len(x&b) - len(x-b) for b in Y]
    return scores

def check_rte(hypothesis, premise):
    batch = [[premise, hypothesis]]
    hf_inputs = rte_tokenizer(batch, return_tensors='pt', padding=True).to('cuda')

    outputs = rte_model(**hf_inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    pred = torch.argmax(probs, dim=-1).item()
    # if pred == ENTAILMENT_LABEL:
    #     print(hypothesis, premise)
    #     print(len(hypothesis.split()), len(premise.split()))
    #     print(pred, ENTAILMENT_LABEL)
    #     exit()
    return pred == ENTAILMENT_LABEL

def get_matches(i,x, all_sents, all_sent_ratings, all_categories, all_sents_bow, distances, indices, rands, randints):
    this_sims = {}
    
    if args.similarity_method[:5] == 'tfidf':
        # this_sims = directional_dotprod(embedded[i], embedded)[0]
        this_sims, topk_indices = 1 - distances[i], indices[i]
    elif args.similarity_method == 'bow':
        this_sims = bow_subsumption(all_sents_bow[i], all_sents_bow)

        topk_indices = np.argsort(this_sims)[-(args.num_distractors * 10):]
        this_sims = [this_sims[ix] for ix in topk_indices]
    else:
        raise Exception('Unknown similarity method! {:}'.format(args.similarity_method))
    candidate_sents = [all_sents[ix] for ix in topk_indices]
    
    matches = [
            (all_sents[ix], this_sims[j]) for j,ix in enumerate(topk_indices)
            if (
                this_sims[j] >= MIN_OVERLAP) 
                # and len(all_sents[ix].split()) >= len(x.split()) 
                and (args.include_self or all_sents[ix] != x)
                and (args.max_tokens is None or len(x.split()) <= args.max_tokens)
                and (all_sents[ix] not in candidate_sents[:j]) # prevent duplicates
                # and (not args.enforce_rating_consistency or abs(all_sent_ratings[i] - all_sent_ratings[ix])) < 2  # this was like this for the successful space model..!! Check if it's still ok?
                and (not args.enforce_rating_consistency or abs(all_sent_ratings[i] - all_sent_ratings[ix]) < 2)
                and (not args.enforce_category_consistency or len(set(all_categories[i]) & set(all_categories[ix])) > 0)
                and (not args.enforce_aspect_consistency or len(get_aspect_label(x) & get_aspect_label(all_sents[ix])) > 0) 
                and (not args.enforce_negation_consistency or ('not' in all_sents_bow[ix] and 'not' in all_sents_bow[i]) or ('not' not in all_sents_bow[ix] and 'not' not in all_sents_bow[i])) # if src contains a negation, make sure tgt does too
                and (not args.check_rte or check_rte(hypothesis=x, premise=all_sents[ix])
            )
        ][-args.num_distractors:][::-1]
    if args.expand_src_wordnet and len(matches) < args.num_distractors:
        expansions = expand_sentence(x, expand_similar=args.expand_include_similar)
        num_expansions = args.num_distractors - len(matches)
        extra_matches = [expansions[j % len(expansions)] for j in randints[i][:num_expansions]]
        matches.extend([(x, 1.0) for x in extra_matches])
    if len(matches) == 0 and (args.include_self_if_no_match or (args.include_self_fraction is not None and rands[i] > args.include_self_fraction)):
        matches = [(x, 1.0)]

    if len(matches) > args.num_distractors:
        print('Matches is too long!')
        exit()
    return matches

from joblib import Parallel, delayed
import multiprocessing
num_cores = args.num_cores
print('using {:} cores'.format(num_cores))

all_sents = np.asarray(all_sents)
all_sent_ratings = np.asarray(all_sent_ratings)
all_sents_bow = np.asarray(all_sents_bow)
all_sents_dev = np.asarray(all_sents_dev)
all_sent_ratings_dev = np.asarray(all_sent_ratings_dev)
all_sents_dev_bow = np.asarray(all_sents_dev_bow)

rands_train = np.random.uniform(size=len(all_sents))
rands_dev = np.random.uniform(size=len(all_sents_dev))
randints_train = np.random.randint(1000, size=(len(all_sents),args.num_distractors))
randints_dev = np.random.randint(1000, size=(len(all_sents_dev),args.num_distractors))

print('Building train data')
# most_similar = [get_matches(i, x, all_sents, all_sent_ratings, all_sents_bow, embedded) for i,x in enumerate(tqdm(all_sents))]
most_similar = Parallel(n_jobs=num_cores, backend='threading', batch_size='auto')(delayed(get_matches)(i, x, all_sents, all_sent_ratings, all_categories, all_sents_bow, distances, indices, rands_train, randints_train) for i,x in enumerate(tqdm(all_sents[:args.max_samples])))
print('Building dev data')
# most_similar_dev =[get_matches(i, x, all_sents_dev, all_sent_ratings_dev, all_sents_dev_bow, embedded_dev) for i,x in enumerate(tqdm(all_sents))]
most_similar_dev = Parallel(n_jobs=num_cores, backend='threading', batch_size='auto')(delayed(get_matches)(i, x, all_sents_dev, all_sent_ratings_dev, all_categories_dev, all_sents_dev_bow, distances_dev, indices_dev, rands_dev, randints_dev) for i,x in enumerate(tqdm(all_sents_dev[:args.max_samples])))

# most_similar_dev = []
# for i,x in enumerate(tqdm(all_sents_dev)):

#     most_similar_dev.append(get_matches(i, x, all_sents_dev, all_sent_ratings_dev, all_sents_dev_bow, embedded_dev))
    
rands_dev = np.random.randint(500, size=len(most_similar_dev))

def expand_target(x, i, rands):
    expansions = expand_sentence(x)
    return [expansions[j % len(expansions)] for j in rands[i][:min(args.num_distractors, len(expansions))]]
    
with jsonlines.open(os.path.join(DATA_PATH, DATASET_NOISED, 'reviews.train.jsonl'), 'w') as f:
    f.write_all([{'target': expand_target(x, i, randints_train) if args.expand_tgt_wordnet else x, 'sources': [tgts[0] for tgts in most_similar[i]]} for i,x in enumerate(all_sents[:args.max_samples]) if len(most_similar[i]) > 0])

# dev sources should be deterministic
with jsonlines.open(os.path.join(DATA_PATH, DATASET_NOISED, 'reviews.dev.jsonl'), 'w') as f:
    f.write_all([{'target': [expand_target(x, i, randints_dev)[0]] if args.expand_tgt_wordnet else x, 'sources': [most_similar_dev[i][rands_dev[i] % len(most_similar_dev[i])][0]]}  for i,x in enumerate(all_sents_dev[:args.max_samples]) if len(most_similar_dev[i]) > 0])