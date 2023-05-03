# Hercules: Attributable and Scalable Opinion Summarization

![Idealised diagram of Hercules](/web/explanation_mini.png)

Code for the paper "Attributable and Scalable Opinion Summarization", Hosking et al. (ACL 2023).

By representing sentences from reviews as paths through a discrete hierarchy, we can generate abstractive summaries that are informative, attributable and scale to hundreds of input reviews.


## Setup

Create a fresh environment:
```
conda create -n herculesenv python=3.9
conda activate herculesenv
```
or
```
python3 -m venv herculesenv
source herculesenv/bin/activate
```

Then install dependencies:
```
pip install -r requirements.txt
```

Download data/models:
 - SPACE -> `./data/`
 - AmaSum -> `./data/`
 - [Trained checkpoints](http://tomho.sk/hercules/models/) -> `./models`


## Evaluation with trained models

## Training from scratch

## Training on a new dataset

Filter, generate pairs

Train model

## Citation