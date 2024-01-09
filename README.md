# TUM-ProjectWeek24-HarMeme

## Resources:
- Hateful Meme Competition: https://www.drivendata.org/competitions/64/hateful-memes/page/205/
- Bert Text Classification: https://www.tensorflow.org/text/tutorials/classify_text_with_bert
- Visual Bert: https://huggingface.co/docs/transformers/model_doc/visual_bert
- HarMeme: https://github.com/di-dimitrov/harmeme
- Lime: https://github.com/marcotcr/lime/blob/master/doc/notebooks/

## Installation
### Spacy
- pip install spacy
- python -m spacy download en_core_web_sm



# Findings 

## Defining Harmful content
Possible Labels:
- [not harmful]
- [somewhat harmful, individual]
- [somewhat harmful, organization]
- [somewhat harmful, community]
- [somewhat harmful, society]
- [very harmful, community]
- [very harmful, society]
- [very harmful, individual]
- [very harmful, organization]

For machine learning we used a Binary Classification for harmful content.

0:
- [not harmful]

1:
- [somewhat harmful, individual]
- [somewhat harmful, organization]
- [somewhat harmful, community]
- [somewhat harmful, society]
- [very harmful, community]
- [very harmful, society]
- [very harmful, individual]
- [very harmful, organization]

The reason for this classification for harmful speech is that with only using "very harmful" as part of harmful content (classifying as 1) is that precisions, recall and subsequently F1-Score are very low for 1 ("harmful content"). We suspect this is because of the small sample size of harmful content in the test data, which has in total only about 354 observations of which only 21 correspond to being labeled as "very harmful". Using this defintion of "harmful content" is not sufficient for our analysis. Modifying the defintion of harmful content as everything but "not harmful" gives us 124 observations of harmful content. This significantly improves out precision, recall and F1-Scores.

## Used Classifiers
### AdaBoost
### Decision Trees
### Gradient Boosting Classifier
### BERT
### Neural-Network (Pytorch)


## Explainable AI
### LIME
- https://github.com/marcotcr/lime
## Classifying Pictures
