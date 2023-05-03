python scripts/opagg_filter_space.py --split_constituents  --input_path ~/datasets/space/space_train.json
python scripts/opagg_filter_space.py --aspectsv2 --input_path ~/datasets/space/space_train.json 
python scripts/opagg_filter_amasum.py --input_path ~/datasets/amasum
python scripts/opagg_filter_amasum.py --split_constituents --input_path ~/datasets/amasum
python scripts/opagg_filter_amasum_eval.py --split_constituents --input_path ~/datasets/amasum