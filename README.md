#IDAO 2021 Track 1

Our final model was trained in 4 steps

1 Simple training with extra branches for domain adaptation\
2 Training with higher weight for regression\
3 Training with higher weight for regression + Unlabeled losses (pseudo labels and variance reduction for classifier and regression model respectively)\
4 Finetuning

For each next step we were choosing the best classifier and regression model from all previous tries basing on the results estimated on the 12 test-like-samples from the train partition of the dataset.\
Finetuning was performed using 48 random train samples and 12 test-like-samples from the train partition of the dataset.

You can find all our checkpoints from each step here (So, you can start from any stage) 
1) https://www.dropbox.com/sh/zdbfkn05sljgwfm/AAAf77Qy3ucStKD2KEEqeqLra?dl=0
2) https://www.dropbox.com/sh/3j069a4mix6wl9a/AABOkUP_LnU4NFoKCEqDKsVKa?dl=0
3) https://www.dropbox.com/sh/b9vtdf0nacpae82/AAATwSQ_waVuyDavIxh8RBJba?dl=0
4) https://www.dropbox.com/sh/sufow6gkrebwut6/AADnxB2gAogj9IsP_JsZ3PwGa?dl=0