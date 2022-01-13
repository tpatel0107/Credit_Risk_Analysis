# Credit_Risk_Analysis
Module 17
# Purpose-Build and evaluate several machine learning models to assess credit risk, using data from LendingClub; a peer-to-peer lending services company

# Overview-Credit risk is an inherently unbalanced classification problem, as the number of good loans easily outnumber the number of risky loans. To obtain the best machine learning algorithm that would assess high credit risk, we ran the following models:

1) Logistic Regression model with various sampling techniques so that the results are not biased. I used Random Oversampling/SMOTE Over-sampling to increase proportional sample size of the High credit Risk (minority). In addition, we performed under-sampling with Cluster Centroids resample to lower proportional size of Low credit risk for loans(majority).
2) Balanced Random Forest model is classification technique to address imbalanced data. The algorithm applies an under-sampling strategy based on clustering techniques for each data bootstrap decision tree.
3) EasyEnsemble is probably the most straightforward way to further exploit the majority class examples ignored by under-sampling. This method independently samples several subsets. For each subset, a classifier is trained on and all generated classifiers are combined for the final decision.

# Results- In search of finding the best machine learning algorithm to predict High Credit Risk for loans, we ran 3 machine learning models: a) Logistics Regression with 4 different types of sampling options, 2) Balanced Random forest and 3) EasyEnsemble AdaBoost.

Based on our observation, Random Oversampling provided the best results among 4 sampling options. Balanced Accuracy score was the highest at 0.66. However, precision for High credit risk was very low at 0.01 and Low risk is almost 1.00. This means, we may not be getting all the instances of High credit risk loan applications despite oversampling. F1 score was also low for High Credit Risk i.e 0.02 only. While low Risk F1 was 0.80 which is high. However, Logistics Regression is not the best model to predict High credit risk for loans.
To further review a model that best predicts High credit risk, our search continued to Balanced Random forest and Easy Ensemble Adaboost.
We recommend EasyEnsemble AdaBoost model to be the best predictor of high-risk applications. This is one of the best models as the algorithm learns and aggregates all 100 results into 1 Final result.

The Balanced Accuracy score is 0.925 closest to 1, which is the best predictor of High credit Risk. F1 score for high risk was 0.14 and low risk was 0.97 with an average of 0.97 that balances both the concerns of precision and recall in one number. Precision score for high credit risk was 0.07 highest when compared to other models. This was achieved by under-sampling majority class ie. Low Credit Risk. There must be lots of False positives. Recall or sensitivity even though highest may not matter in this case.

Please see below the Summary Results of the 6 algorithm and definitions of metrics
