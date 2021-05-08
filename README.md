# Home-Credit-Default-Risk-AML-Project [![](<img src="https://user-images.githubusercontent.com/61594650/117523778-16ed5e00-af88-11eb-8914-31021617d53a.png" width="50" height="50">)](https://youtu.be/TXh4owKBE3o)


This is an academic project which we completed as a part of Applied Machine Learning Course from Indiana University, Bloomington

## **Abstract**
The objective of this project is to use machine learning methodologies on historical loan application data to predict whether or not an applicant will be able to repay a loan. As an extension to EDA and hyper-tuned model, this phase provided valuable insights when feature engineering was modified to handle data leakage employing a better data processing flow. Multiple experiments were conducted applying feature selection techniques including RFE, SelecKbest, Variance threshold to Logistic regression, Gradient Boosting, XGBoost, LightGBM & SVM models , further handling class imbalance using SMOTE for XGBoost , monitoring error generalization with early stopping and building high performance Neural Networks . Our results in this phase show that the best performing algorithm was Logistic Regerssion with varaince threshold selection with test ROC_AOC as 75.22%. The lowest performing was SVM model with test AUC(Area under ROC) as 67.21%. Our best score in Kaggle submission was for Logistic Regression with SelectKBest with score of 0.72158 for private and 0.72592 for public.

The files in this repository are:

1. Group1_Phase1_EDA_Baseline.html
2. Group1_Phase1_EDA_Baseline.ipynb
3. Group1_Phase2_Feature_Engr_Hyperparameter_Tuning.html
4. Group1_Phase2_Feature_Engr_Hyperparameter_Tuning.ipynb
5. Group1_Phase3_PyTorch Deep Learning.ipynb
6. Group1_Phase3_PyTorch Deep Learning.pptx
7. Group1_Phase3_PyTorch Deep Learning.html


## **Discussion of Results** <br>
As you can see in the Experimental Results Final results we have performed various feature selection technique like (RFE, PCA, Variance Threshold, SelectKBest, SMOTE) on certain specific models with 132 highly correlated features. Below is brief description of results attained in these experiments.

Our best model turned out to be Logistic Regression with SelectKBest with 74.86% ROC score. Our hopes were higher on XGBoost classifier but it stood out to be second best in our models.

Our Deep Learning of simple network preformed model better than the multilayer network. The ROC score came as 74.6% for the simple network. For multilayer network our score came as 59.38%.

Compared to traditional machine learning model, deep learning model trained on completed dataset much faster.

Below has more details or the various classifiers executed in this project.

### **Logistic Regression :** <br> 
This model was chosen as the baseline model trained with imbalanced dataset and later performed feature selection using RFE, SelectKBest, PCA & Variance Threshold technique on it. The baseline training accuracy for this model was encouraging which let us to perform the prior mentioned feature selection on these models. The best model for logistic regression we had is with Variance Threshold, with training accuracy as 92.56% and test accuracy as 92.2%. A 75.22% ROC score resulted with best parameters for this model. The same model was run with other feature selection performed very closer to the best model.

### **Gradient Boosting :** <br> 
Boosting didn't help in achieving better results than the baseline model. The results were not good enough to continue in implementing & evaluating other feature selection technique. Training accuracy of 94.75% and test accuracy of 91.95% was achieved in this model. Test ROC under the curve for this model came out to 72.12%

### **XGBoost :** <br> 
By far this model resulted in the second best model with RFE hence we continued to explore other feature selection techniques on this. The best performing model for XGBoost was with Variance Threshold. The accuracy of the training and test are 93.1% and test 92.36%. Test ROC under the curve is 73.88%. The other feature selection were very closer to the best XGBoost model. We also performed XGBoost with SMOTE as the dataset had oversampled records. The ROC score has promising result with 74.23%.

### **Light BGM :** <br> 
Our expectation was this model would give us better and faster results than XGBoost, however it was slightly lower compared to XGBoost. Both RFE and variance threshold feature selection resulted in same ROC score of 72.2. The training accuracy came as 92.81% and test accuracy 92.28% was achieved.

### **Random Forest :** <br> 
On our last decision tree models, the best Random Forest was with variance threshold which produced training accuracy of 92.51% and test accuracy of 92.36%. Test ROC score came out as 72.43%. Random forest performed better compared to LightBGM but lower than XGBoost.

### **SVM :** <br> 
This was the lowest performing model in our experiment. Hence we didn't decide to continue on SVM with other feature selection techniques. The ROC score achieved for this model was way lower i.e. is 67.21%.

## **Conclusion** <br>
In the final phase, after proving our hypothesis that tuned machine learning techniques can outperform baseline models to aid Home Credit in their evaluation of loan applications, we believe expanding our framework will create a more robust environment with improved performance.

Logistic regression, XGBoost, Random Forest and LightGBM were selected to run with RFE, PCA, SelectKBest and Variance Threshold for feature selection, and SMOTE for data imbalance. The best performance for each algorithm was included in the classification ensemble using soft voting. The resulting Kaggle score was 0.72592 ROC_AUC.

Single and Multi-layer deep learning models, including linear, sigmoid, ReLu, and hidden layers were added with binary CXE, custom hinge loss using adam & sgd optimizer. The deep learning Kaggle score fell short of the ensemble model; additional experimentation will result in a better performing deep learning models. By combining and continuing to refine our extended loss function, we can further demonstrate our effectiveness.

## **Contributors** <br>
1. Anitha Ganapathy
2. Archana Krishnamurthy
3. Bathurunnisha Abdul Jabar
4. Rajesh Thanji
