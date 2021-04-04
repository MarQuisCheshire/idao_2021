##IDAO 2021 Track 1


Our final model was trained in 4 steps

1 Simple training with extra branches for domain adaptation\
2 Training with higher weight for regression\
3 Training with higher weight for regression + Unlabeled losses (pseudo labels and variance reduction for classifier and regression model respectively)\
4 Finetuning

For each next step we were choosing the best classifier and regression model from all previous tries basing on the results estimated on the 12 test-like-samples from the train partition of the dataset.\
Finetuning was performed using 48 random train samples and 12 test-like-samples from the train partition of the dataset.