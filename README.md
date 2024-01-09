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
|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0            | 0.81      | 0.84   | 0.82     | 230     |
| 1            | 0.68      | 0.64   | 0.66     | 124     |
| accuracy     |           |        | 0.77     | 354     |
| macro avg    | 0.75      | 0.74   | 0.74     | 354     |
| weighted avg | 0.77      | 0.77   | 0.77     | 354     |

### Decision Trees
|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0            | 0.82      | 0.86   | 0.84     | 230     |
| 1            | 0.71      | 0.65   | 0.68     | 124     |
| accuracy     |           |        | 0.77     | 354     |
| macro avg    | 0.76      | 0.75   | 0.76     | 354     |
| weighted avg | 0.78      | 0.78   | 0.78     | 354     |

### Gradient Boosting Classifier
|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0            | 0.81      | 0.88   | 0.85     | 230     |
| 1            | 0.74      | 0.63   | 0.68     | 124     |
| accuracy     |           |        | 0.77     | 354     |
| macro avg    | 0.78      | 0.75   | 0.76     | 354     |
| weighted avg | 0.79      | 0.79   | 0.79     | 354     |

### BERT

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0            | 0.85      | 0.85   | 0.85     | 230     |
| 1            | 0.72      | 0.72   | 0.72     | 124     |
| accuracy     |           |        | 0.80     | 354     |
| macro avg    | 0.78      | 0.78   | 0.78     | 354     |
| weighted avg | 0.80      | 0.80   | 0.80     | 354     |
#### Additional Information
|           | precision |
|-----------|---------|
| Test Loss | 0.4565      |
| Test Accuracy | 0.8023      |
| Test Precision | 0.7177      |
| Test Recall | 0.7177      |
| Test F1-Score | 0.7177      |

- Graphs for the loss function and accuracy are shown in the jupiter notebook in the bert file.
  - In it we can see that the loss function is increasing after the first epoch, which means that the model is not learning.
- Increasing the batch or epoch size does not improve the results for BERT, which is expected from BERT.

#### Addition of LoRA
Fine Tuning with the help of LoRA (Label Refinery) does not improve the results of BERT. The results are as follows:

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0            | 0.65      | 1.00   | 0.79     | 230     |
| 1            | 0.00      | 0.00   | 0.00     | 124     |
| accuracy     |           |        | 0.65     | 354     |
| macro avg    | 0.32      | 0.50   | 0.39     | 354     |
| weighted avg | 0.42      | 0.65   | 0.51     | 354     |

- Note: 0.00 and 1.00 results due to zero division errors, which is 
usually caused by a data set which is not big 
enough or has no occurrences of the label in 
question. 
- Evolution of the loss function over time is shown in the jupiter notebook in the lora bert file.
  - In it we can see that the loss function is decreasing over time, which is a good sign.
- These results can partially be explained by our low datasize and calculation power.

### Neural-Network (Pytorch)
|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0            | 0.79      | 0.75   | 0.77     | 230     |
| 1            | 0.58      | 0.64   | 0.61     | 124     |
| accuracy     |           |        | 0.71     | 354     |
| macro avg    | 0.69      | 0.69   | 0.69     | 354     |
| weighted avg | 0.72      | 0.71   | 0.71     | 354     |

Also has one of the shortest training times of all classifiers.

## Explainable AI
### LIME
- https://github.com/marcotcr/lime
## Classifying Pictures
