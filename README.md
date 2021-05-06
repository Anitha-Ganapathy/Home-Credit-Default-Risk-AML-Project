# Home-Credit-Default-Risk-AML-Project

Abstract
The objective of this project is to use machine learning methodologies on historical loan application data to predict whether or not an applicant will be able to repay a loan. As an extension to EDA and hyper-tuned model, this phase provided valuable insights when feature engineering was modified to handle data leakage employing a better data processing flow. Multiple experiments were conducted applying feature selection techniques including RFE, SelecKbest, Variance threshold to Logistic regression, Gradient Boosting, XGBoost, LightGBM & SVM models , further handling class imbalance using SMOTE for XGBoost , monitoring error generalization with early stopping and building high performance Neural Networks . Our results in this phase show that the best performing algorithm was Logistic Regerssion with varaince threshold selection with test ROC_AOC as 75.22%. The lowest performing was SVM model with test AUC(Area under ROC) as 67.21%. Our best score in Kaggle submission was for Logistic Regression with SelectKBest with score of 0.72158 for private and 0.72592 for public.

The files in this repository are:

1. Group1_Phase3_PyTorch Deep Learning.html
2. Group1_Phase3_PyTorch Deep Learning.ipynb
3. Group1_Phase3_PyTorch Deep Learning.pptx
