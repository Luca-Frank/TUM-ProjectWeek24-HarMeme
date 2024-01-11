# TUM-ProjectWeek24-HarMeme

## Resources:
- Hateful Meme Competition: https://www.drivendata.org/competitions/64/hateful-memes/page/205/
- Bert Text Classification: https://www.tensorflow.org/text/tutorials/classify_text_with_bert
- Visual Bert: https://huggingface.co/docs/transformers/model_doc/visual_bert
- HarMeme: https://github.com/di-dimitrov/harmeme
- Lime: https://github.com/marcotcr/lime/blob/master/doc/notebooks/
- Classifier Script: https://github.com/facebookresearch/mmf/blob/main/tools/scripts/features/extract_features_vmb.py
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

### SVM (Support Vector Machine)
#### Michael
|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0            | 0.74      | 0.91   | 0.82     | 577     |
| 1            | 0.74      | 0.43   | 0.54     | 327     |
| accuracy     |           |        | 0.74     | 904     |
| macro avg    | 0.74      | 0.67   | 0.68     | 904     |
| weighted avg | 0.74      | 0.74   | 0.72     | 904     |

- Additional Charts are included in the Jupyter Notebook in the SVM file.
- Second more improved SVM model:

#### Luca
![](/plots/SVM_Precision_Recall_Curve.jpeg)
![](/plots/SVM_ROC_Curve.jpeg)
![](/plots/SVM_PCA_PredClass.jpeg)
![](/plots/SVM_PCA_TrueClass.jpeg)
### Six models performance on binary text classification
![Six models performance on binary text classification](/plots/models_performance.jpg)

### Multi text classification on harful speech based on bert(individual, organization, community, society)
|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| individual   | 0.79      | 0.84   | 0.82     | 37      |
| organization | 0.91      | 0.87   | 0.89     | 46      |
| community    | 0.00      | 0.00   | 0.00     | 4       |
| society      | 0.58      | 0.70   | 0.64     | 20      |
| accuracy     |           |        | 0.79     | 107     |
| macro avg    | 0.57      | 0.60   | 0.59     | 107     |
| weighted avg | 0.77      | 0.79   | 0.78     | 107     |
- Because of small and not balabced dateset, some performance of bert model not perform well.

### Dataset we used
![Dataset](/plots/dataset_memes.jpg)
 
## Explainable AI

### LIME
#### Random Forest (Luca)
##### Optimizing Hyperparameters
To optimize the hyperparameters we used the GridSearch implemented in the sklearn library. We found that with these hyperparameters for our Random Forest Model we can maximize the F1-Score:

##### X-AI
We started using explainable AI methods first on simpler models where we can quickly implement methods to explain the decisions of the models in order to get a first overview of the impactful variables in our data.

From the lime library we used the LimeTextExplainer. This allows us to calculate for each text the probability of being labeled as harmful. Further it gives us the words with its corresponding weight. This we can use to get a deeper insight into what words have a large impact on our model to determine wether a meme-caption is harmful or harmless. 

The way this is calculated is with the function `explain_instance()` function. Here we specified to extract the top 5 features for the text which have the most explaining power when it comes to classifying the text as a harmful meme or a harmless meme.

First we were intrested which words appear very often, therefore often posess a high explaining power for our model. We counted the overall appearence among the top 5 explaining features and colored the graph by whether or not the model predicted this word to be in a harmfull or harmless meme.

![Top 20 Words that had the most impact on determining wether a Random Forest Model classifies a text as harmful or not harmful](/plots/RandomForest_LimeTop20Words_Features2.png)

This however allows us to only look at the number of occurences for each word. The explainers provided by LIME also have a value associated with it. The values reange between 1 and -1 and represent the weight each word has on influencing the classification. A positive value means that this word nudged the classification to label the text as 1, in our case a harmful meme. We can therefore conclude that positive values are more associated with harmful memes while negative values are associated with harmless memes.

The next graphs show the top 10 features with the highest average positive or negative weight. This allows us to see which words a generally associated with harmfull and harmless memes.

#### Inception_v3 (Pretrained pytorch model)

### Deceptron2

