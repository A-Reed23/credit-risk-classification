Supervise machine learning


*Introduction*

This analysis will attempt to create a model that has the ability to predict the quality of loans based upon the interest rate, loan size, number of accounts, debt to income ratio, previous negative factors, total income and total debt. We will attempt machine learning from the data, and then oversampling the data to increase the high-risk loans giving the model more data to learn from. Hopefully this will result in a model that successfully predicts whether the loan is a good investment or is likely to default. 
The variable that we want to learn to predict from the data is the loan status, which is 0 for healthy, and 1 for high risk. 
  

*Summary Of first model processing* 

The first thing that was done was to examine the data and see what information we had. The loan status column was then set to be the y, then removed from the training data. We reviewed the data and saw that the average income was $49,222 with an average debt to income ratio of 0.377, and an average loan size of $9.805 at a 7.29% interest rate. The average borrower had 3.8 bank accounts.  

The data had 75,036 healthy loans, and 2,500 high risk loans. This is a highly imbalanced data set. We proceeded to split the data into training data and testing data. This was done with sklearn's test_train_split with random state set to one for reproducibility. The data was stratified, to improve the repetitiveness of the sampling, as compared to the entire data set [1]. A logistic regression was carried out using sklearn's logistic regression function, and the data was fit to the model. Predictions were then made using sklearn's* predict* classifier. These were tested against the testing section of the data. The model had an overall score of 99.14% on the training data and 99.24% on the testing data suggesting that the model was making relatively good predictions.  

*Analysis Of first Model* 

The overall accuracy of the model was 99.24%, however the balanced accuracy score was 94.43%, which suggests that the imbalance of the data set had an effect. Additionally, this suggests that the model is less accurate with the high-risk loans. The value counts that we did earlier showed that the data is about a 75%/25% split between healthy loans and high-risk loans. 

We can see that having a very large number of healthy loans (18,759 of 75,036 healthy loans in the data (values counts vs support numbers) has made the model extremely precise for healthy loans. Calculating the ratio of samples in the data to samples used to test, we find that 25% of both classes were used. 

For the healthy loans, the f1-score, the precision, and the recall are very good, meaning that the model does a good job identifying healthy loans. However, the scores for the high-risk loans are not so good. This could be due to the training data having fewer samples for the model to learn from.  

The lowest score for the high-risk loans is the precision (0.87), which means that the model is misclassifying things that are not high risk as high-risk loans. The recall is similar to the precision in that it measures the ratio of true positives over true positives and false negatives (things that should be positive) This was 0.89 which means that the model is slightly better at not having false positives than it is at not having false negatives. The f1 score (the harmonic mean of the preceding) is 0.88. 

{The f1-score is the harmonic mean of precision (the ratio of true positives over all positives (TRUE AND FALSE) [3]) and recall (which is the ratio of true positive over the total number of positives (TRUE POSITIVE + FALSE NEGATIVE) in sample (things that are positive and were categorized correctly, or incorrectly)[3]). } 

*Attempting to improve the model through oversampling the minority class* 

Using the random over sampler from sklearn, set to oversample the minority class (as opposed to not majority, which would oversample all non-majority classes) results in a model that has scores across the board of 0.99. The same result is reached with oversampling of 'not-majority'. This seems like it is an improvement. However, the improvement was reached by resampling the minority class so that the total data went from 2,500 to 75,036 minority samples. This has the disadvantage of making an outlier look like 1 of 30 (if you divide 75,036 by 2,500 you get 30.0144, or each record was reproduced an average of 30 times). It seems likely that the model has been overfit. 

*Conclusions* 

The data consists of 77,536 loan records. Of note the interest rate in the data ranges from a minimum of 5.25% to 13.235%, with the median being 7.172%, and the mean being 7.29%. The 75th percentile for the interest rate was 7.528%. If I was given the data and asked to analyze it without machine learning, the first things I would look at are the interest rate, the debt to income, and derogatory marks. 

The number of loans that were charging over 10% interest was 2,709 (or 3.49% of loans). The over 10% interest borrowers tend to have a higher debt ratio than the data as a whole (which was 0.377 vs 0.646). While these borrowers were more likely to have an above average income (dataset median $48,100 vs the high interest median of $84,900), they were above 75% of borrowers in derogatory marks, number of accounts, debt to income ratio, and loan size. This is likely due to being assessed as poor credit risks, before being given the loan (and explaining the high interest rate). 

Having a debt to income of 0.5 or higher or at least one derogatory mark results in about 75% of the loans being high-risk, and we can see that some loans were getting extremely high interest rates, probably because they had been assessed as not the best risk. This suggests that the method of loan assessment is fairly good. 


*Refrences:*

https://stackoverflow.com/questions/55548675/difference-between-balanced-accuracy-score-and-accuracy-score
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score
https://stackoverflow.com/questions/40008015/problems-importing-imblearn-python-package-on-ipython-notebook
https://imbalanced-learn.org/dev/references/generated/imblearn.over_sampling.RandomOverSampler.html

*Thanks*
Thanks to Dale Linn and Ahmand Mosua
