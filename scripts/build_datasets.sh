python scripts/opagg_filter_space.py --aspectsv2 --input_path ./data/space/
python scripts/opagg_filter_space_eval.py  --input_path ./data/space/
python scripts/opagg_filter_amasum.py --input_path ./data/amasum
python scripts/opagg_filter_amasum_eval.py --input_path ./data/amasum

python ./scripts/generate_opagg_pairs.py --similarity_method tfidf2 --min_overlap 0.6 --ignore_stopwords --dataset space-25toks-1pronouns-aspects --num_distractors 5 --num_cores 4 --enforce_rating_consistency
python ./scripts/generate_opagg_pairs.py --similarity_method tfidf2 --min_overlap 0.6 --ignore_stopwords --dataset amasum-shoes-25toks-0pronouns --num_distractors 5 --enforce_category_consistency --enforce_rating_consistency --num_cores 4 --enforce_negation_consistency
python ./scripts/generate_opagg_pairs.py --similarity_method tfidf2 --min_overlap 0.6 --ignore_stopwords --dataset amasum-tools-25toks-0pronouns --num_distractors 5 --enforce_category_consistency --enforce_rating_consistency --num_cores 4 --enforce_negation_consistency
python ./scripts/generate_opagg_pairs.py --similarity_method tfidf2 --min_overlap 0.6 --ignore_stopwords --dataset amasum-home-kitchen-25toks-0pronouns --num_distractors 5 --enforce_category_consistency --enforce_rating_consistency --num_cores 4 --enforce_negation_consistency
python ./scripts/generate_opagg_pairs.py --similarity_method tfidf2 --min_overlap 0.6 --ignore_stopwords --dataset amasum-sports-outdoors-25toks-0pronouns --num_distractors 5 --enforce_category_consistency --enforce_rating_consistency --num_cores 4 --enforce_negation_consistency