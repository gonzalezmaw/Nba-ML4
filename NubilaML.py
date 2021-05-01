import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from SLR import SLR
from RFR import RFR
from SVMC import SVMC
from NBC import NBC
from KNN import KNNC
from RFC import RFC
from PRM import PRM
from MRFR import MRFR
from MLPR import MLPR
from MLPC import MLPC
from ClusteringModels import ClusteringModels
from QuickClassifiers import QuickClassifiers
from QuickRegressors import QuickRegressors
from ClassifierModels import ClassifierModels
from RegressorModels import RegressorModels

from Pyspark import Pyspark
from RegressorsPyspark import RegressorsPyspark
from ClassifiersPyspark import ClassifiersPyspark
from ClusteringPyspark import ClusteringPyspark
from TimeSeries import TimeSeries
from AnomalyDetection import AnomalyDetection

from PIL import Image
# from BigdataPyspark import BigdataPyspark


img = Image.open('NubilaIcono.png')

st.set_page_config(page_title="Nubila ML", page_icon=img)
st.title("Machine Learning for Everyone")

st.sidebar.write("""
# **Nubila ML**

""")


modelSelect_name = st.sidebar.selectbox(
    "Select a Model", ("Simple Linear Regression", "Polynomial Regression", "Multiple Linear Regressor", "Support Vector Machines Regressor", "Bayesian Ridge Regressor",
                       "Decision Tree Regressor", "Extra Trees Regressor", "Random Forest Regressor", "K-Nearest Neighbors Regressor", "Gradient Boosting Regressor", "Extreme Gradient Boosting Regressor", "Gaussian Process Regressor", "Stochastic Gradient Descent Regressor",
                       "Light Gradient Boosting Machine Regressor", "CatBoost Regressor", "AdaBoost Regressor", "Bagging Regressor", "Passive Aggressive Regressor",
                       "Elastic Net Regressor", "Lasso Regressor", "Ridge Regressor", "Huber Regressor", "Kernel Ridge Regressor",
                       "Tweedie Regressor", "TheilSen Regressor", "Orthogonal Matching Pursuit Regressor", "Histogram Gradient Boosting Regressor", "Least Angle Regressor",
                       "Lasso Least Angle Regressor", "Automatic Relevance Determination Regressor", "Random Sample Consensus Regressor", "Perceptron Regressor", "Natural Gradient Boosting Regressor",
                       "Random Forest Classifier", "Support Vector Machines Classifier", "Logistic Regression Classifier", "Naive Bayes Classifier", "Decision Tree Classifier", "Extra Trees Classifier",
                       "K-Nearest Neighbors Classifier", "Gradient Boosting Classifier", "Extreme Gradient Boosting Classifier", "Gaussian Process Classifier", "Stochastic Gradient Descent Classifier",
                       "Light Gradient Boosting Machine Classifier", "CatBoost Classifier", "AdaBoost Classifier", "Bagging Classifier", "Passive Aggressive Classifier",
                       "Linear Discriminant Analysis Classifier", "Quadratic Discriminant Analysis Classifier", "Linear Support Vector Machine Classifier", "Ridge Classifier", "Natural Gradient Boosting Classifier",
                       "Quick Comparison Regressors", "Quick Comparison Classifiers",
                       "NN - Multi-Layer Perceptron",
                       "K-Means Clustering", "Hierarchical Clustering", "Spectral Clustering",
                       "Bigdata Analysis", "Time Series", "Anomaly Detection"))


if modelSelect_name == "Simple Linear Regression":
    st.write("""
        ## **Simple Linear Regression Model**
        """)
    st.write("""
    ### **Simple Regression Method**
    """)
    SLR()

elif modelSelect_name == "Multiple Linear Regressor":
    st.write("""
        ## **Multiple Linear Regression Model**
        """)

    st.write("""
        ### **Multiple Regression Method**
        """)
    RegressorModels(modelSelect_name)

elif modelSelect_name == "Support Vector Machines Regressor":
    st.write("""
        ## **Support Vector Machines Regression Model**
        """)

    st.write("""
        ### **Multiple Regression Method**
        """)
    RegressorModels(modelSelect_name)

elif modelSelect_name == "Bayesian Ridge Regressor":
    st.write("""
        ## **Bayesian Ridge Regression Model**
        """)

    st.write("""
        ### **Multiple Regression Method**
        """)
    RegressorModels(modelSelect_name)

elif modelSelect_name == "Decision Tree Regressor":
    st.write("""
        ## **Decision Tree Regression Model**
        """)

    st.write("""
        ### **Multiple Regression Method**
        """)
    RegressorModels(modelSelect_name)

elif modelSelect_name == "Extra Trees Regressor":
    st.write("""
        ## **Extra Trees Regression Model**
        """)

    st.write("""
        ### **Multiple Regression Method**
        """)
    RegressorModels(modelSelect_name)

elif modelSelect_name == "Random Forest Regressor":
    st.write("""
        ## **Random Forest Regression Model**
        """)

    modelSelect_Type = st.sidebar.selectbox(
        "Select a Method", ("Single Variable Regression", "Multiple Variable Regression"))

    if modelSelect_Type == "Single Variable Regression":

        st.write("""
        ### **Simple Regression Method**
        """)
        RFR()
    elif modelSelect_Type == "Multiple Variable Regression":
        st.write("""
        ### **Multiple Regression Method**
        """)
        # MRFR()
        RegressorModels(modelSelect_Type)


elif modelSelect_name == "Gradient Boosting Regressor":
    st.write("""
        ## **Gradient Boosting Regression Model**
        """)
    st.write("""
    ### **Multivariate Regression Method**
    """)

    RegressorModels(modelSelect_name)


elif modelSelect_name == "Extreme Gradient Boosting Regressor":
    st.write("""
        ## **Extreme Gradient Boosting Regression Model**
        """)
    st.write("""
    ### **Multivariate Regression Method**
    """)

    RegressorModels(modelSelect_name)


elif modelSelect_name == "Gaussian Process Regressor":
    st.write("""
        ## **Gaussian Process Regression Model**
        """)
    st.write("""
    ### **Multivariate Regression Method**
    """)
    RegressorModels(modelSelect_name)

elif modelSelect_name == "Stochastic Gradient Descent Regressor":
    st.write("""
        ## **Stochastic Gradient Descent Regression Model**
        """)
    st.write("""
    ### **Multivariate Regression Method**
    """)

    RegressorModels(modelSelect_name)


elif modelSelect_name == "K-Nearest Neighbors Regressor":
    st.write("""
        ## **K-Nearest Neighbors Regression Model**
        """)

    st.write("""
        ### **Multiple Regression Method**
        """)
    RegressorModels(modelSelect_name)


elif modelSelect_name == "Light Gradient Boosting Machine Regressor":
    st.write("""
        ## **Light Gradient Boosting Machine Model**
        """)

    st.write("""
        ### **Multiple Regression Method**
        """)
    RegressorModels(modelSelect_name)

elif modelSelect_name == "CatBoost Regressor":
    st.write("""
        ## **CatBoost Regression Model**
        """)

    st.write("""
        ### **Multiple Regression Method**
        """)
    RegressorModels(modelSelect_name)

elif modelSelect_name == "AdaBoost Regressor":
    st.write("""
        ## **AdaBoost Regression Model**
        """)

    st.write("""
        ### **Multiple Regression Method**
        """)
    RegressorModels(modelSelect_name)

elif modelSelect_name == "Bagging Regressor":
    st.write("""
        ## **Bagging Regression Model**
        """)

    st.write("""
        ### **Multiple Regression Method**
        """)
    RegressorModels(modelSelect_name)

elif modelSelect_name == "Passive Aggressive Regressor":
    st.write("""
        ## **Passive Aggressive Regression Model**
        """)

    st.write("""
        ### **Multiple Regression Method**
        """)
    RegressorModels(modelSelect_name)

elif modelSelect_name == "Elastic Net Regressor":
    st.write("""
        ## **Elastic Net Regression Model**
        """)

    st.write("""
        ### **Multiple Regression Method**
        """)
    RegressorModels(modelSelect_name)

elif modelSelect_name == "Lasso Regressor":
    st.write("""
        ## **Lasso Regression Model**
        """)

    st.write("""
        ### **Multiple Regression Method**
        """)
    RegressorModels(modelSelect_name)

elif modelSelect_name == "Ridge Regressor":
    st.write("""
        ## **Ridge Regression Model**
        """)

    st.write("""
        ### **Multiple Regression Method**
        """)
    RegressorModels(modelSelect_name)

elif modelSelect_name == "Huber Regressor":
    st.write("""
        ## **Huber Regression Model**
        """)

    st.write("""
        ### **Multiple Regression Method**
        """)
    RegressorModels(modelSelect_name)

elif modelSelect_name == "Kernel Ridge Regressor":
    st.write("""
        ## **Kernel Ridge Regression Model**
        """)

    st.write("""
        ### **Multiple Regression Method**
        """)
    RegressorModels(modelSelect_name)

elif modelSelect_name == "Tweedie Regressor":
    st.write("""
        ## **Tweedie Regression Model**
        """)

    st.write("""
        ### **Multiple Regression Method**
        """)
    RegressorModels(modelSelect_name)

elif modelSelect_name == "TheilSen Regressor":
    st.write("""
        ## **TheilSen Regression Model**
        """)

    st.write("""
        ### **Multiple Regression Method**
        """)
    RegressorModels(modelSelect_name)

elif modelSelect_name == "Orthogonal Matching Pursuit Regressor":
    st.write("""
        ## **Orthogonal Matching Pursuit Regression Model**
        """)

    st.write("""
        ### **Multiple Regression Method**
        """)
    RegressorModels(modelSelect_name)

elif modelSelect_name == "Histogram Gradient Boosting Regressor":
    st.write("""
        ## **Histogram Gradient Boosting Regression Model**
        """)

    st.write("""
        ### **Multiple Regression Method**
        """)
    RegressorModels(modelSelect_name)

elif modelSelect_name == "Least Angle Regressor":
    st.write("""
        ## **Least Angle Regression Model**
        """)

    st.write("""
        ### **Multiple Regression Method**
        """)
    RegressorModels(modelSelect_name)

elif modelSelect_name == "Lasso Least Angle Regressor":
    st.write("""
        ## **Lasso Least Angle Regression Model**
        """)

    st.write("""
        ### **Multiple Regression Method**
        """)
    RegressorModels(modelSelect_name)

elif modelSelect_name == "Automatic Relevance Determination Regressor":
    st.write("""
        ## **Automatic Relevance Determination Regression Model**
        """)

    st.write("""
        ### **Multiple Regression Method**
        """)
    RegressorModels(modelSelect_name)

elif modelSelect_name == "Random Sample Consensus Regressor":
    st.write("""
        ## **Random Sample Consensus Regression Model**
        """)

    st.write("""
        ### **Multiple Regression Method**
        """)
    RegressorModels(modelSelect_name)

elif modelSelect_name == "Perceptron Regressor":
    st.write("""
        ## **Perceptron Regression Model**
        """)

    st.write("""
        ### **Multiple Regression Method**
        """)
    RegressorModels(modelSelect_name)

elif modelSelect_name == "Natural Gradient Boosting Regressor":
    st.write("""
        ## **Natural Gradient Boosting Regression Model**
        """)

    st.write("""
        ### **Multiple Regression Method**
        """)
    RegressorModels(modelSelect_name)

#  CLASSIFICATION CASES
elif modelSelect_name == "Support Vector Machines Classifier":
    st.write("""
        ## **Support Vector Machines Classification Model**
        """)
    st.write("""
    ### **Multivariate Classification Method**
    """)
    # SVMC()
    ClassifierModels(modelSelect_name)


elif modelSelect_name == "Logistic Regression Classifier":
    st.write("""
        ## **Logistic Regression Classification Model**
        """)
    st.write("""
    ### **Multivariate Classification Method**

    """)
    # SVMC()
    ClassifierModels(modelSelect_name)

elif modelSelect_name == "Naive Bayes Classifier":
    st.write("""
        ## **Naive Bayes Classification Model**

        """)
    st.write("""
    ### **Multivariate Classification Method**

    """)
    # NBC()
    ClassifierModels(modelSelect_name)

elif modelSelect_name == "Decision Tree Classifier":
    st.write("""
        ## **Decision Tree Classification Model**

        """)
    st.write("""
    ### **Multivariate Classification Method**

    """)
    # NBC()
    ClassifierModels(modelSelect_name)

elif modelSelect_name == "Extra Trees Classifier":
    st.write("""
        ## **Extra Trees Classification Model**

        """)
    st.write("""
    ### **Multivariate Classification Method**

    """)

    ClassifierModels(modelSelect_name)


elif modelSelect_name == "K-Nearest Neighbors Classifier":
    st.write("""
        ## **K-Nearest Neighbors Classification Model**

        """)
    st.write("""
    ### **Multivariate Classification Method**

    """)
    # KNNC()
    ClassifierModels(modelSelect_name)

elif modelSelect_name == "Random Forest Classifier":
    st.write("""
        ## **Random Forest Classification Model**

        """)
    st.write("""
    ### **Multivariate Classification Method**

    """)
    # RFC()
    ClassifierModels(modelSelect_name)

elif modelSelect_name == "Gradient Boosting Classifier":
    st.write("""
        ## **Gradient Boosting Classification Model**

        """)
    st.write("""
    ### **Multivariate Classification Method**

    """)
    # RFC()
    ClassifierModels(modelSelect_name)


elif modelSelect_name == "Extreme Gradient Boosting Classifier":
    st.write("""
        ## **Extreme Gradient Boosting Classification Model**

        """)
    st.write("""
    ### **Multivariate Classification Method**

    """)
    # RFC()
    ClassifierModels(modelSelect_name)

elif modelSelect_name == "Gaussian Process Classifier":
    st.write("""
        ## **Gaussian Process Classification Model**

        """)
    st.write("""
    ### **Multivariate Classification Method**

    """)

    ClassifierModels(modelSelect_name)

elif modelSelect_name == "Stochastic Gradient Descent Classifier":
    st.write("""
        ## **Stochastic Gradient Descent Classification Model**

        """)
    st.write("""
    ### **Multivariate Classification Method**

    """)

    ClassifierModels(modelSelect_name)

elif modelSelect_name == "Light Gradient Boosting Machine Classifier":
    st.write("""
        ## **Light Gradient Boosting Machine Classification Model**

        """)
    st.write("""
    ### **Multivariate Classification Method**

    """)

    ClassifierModels(modelSelect_name)

elif modelSelect_name == "CatBoost Classifier":
    st.write("""
        ## **CastBoost Classification Model**

        """)
    st.write("""
    ### **Multivariate Classification Method**

    """)

    ClassifierModels(modelSelect_name)

elif modelSelect_name == "AdaBoost Classifier":
    st.write("""
        ## **AdaBoost Classification Model**

        """)
    st.write("""
    ### **Multivariate Classification Method**

    """)
    ClassifierModels(modelSelect_name)

elif modelSelect_name == "Bagging Classifier":
    st.write("""
        ## **Bagging Classification Model**

        """)
    st.write("""
    ### **Multivariate Classification Method**

    """)

    ClassifierModels(modelSelect_name)


elif modelSelect_name == "Passive Aggressive Classifier":
    st.write("""
        ## **Passive Aggressive Classification Model**

        """)
    st.write("""
    ### **Multivariate Classification Method**

    """)
    ClassifierModels(modelSelect_name)

elif modelSelect_name == "Linear Discriminant Analysis Classifier":
    st.write("""
        ## **Linear Discriminant Analysis Classification Model**

        """)
    st.write("""
    ### **Multivariate Classification Method**

    """)
    ClassifierModels(modelSelect_name)

elif modelSelect_name == "Quadratic Discriminant Analysis Classifier":
    st.write("""
        ## **Quadratic Discriminant Analysis Classification Model**

        """)
    st.write("""
    ### **Multivariate Classification Method**

    """)
    ClassifierModels(modelSelect_name)

elif modelSelect_name == "Linear Support Vector Machine Classifier":
    st.write("""
        ## **Linear Support Vector Machine Classification Model**

        """)
    st.write("""
    ### **Multivariate Classification Method**

    """)
    ClassifierModels(modelSelect_name)

elif modelSelect_name == "Ridge Classifier":
    st.write("""
        ## **Ridge Classifier Classification Model**

        """)
    st.write("""
    ### **Multivariate Classification Method**

    """)
    ClassifierModels(modelSelect_name)

elif modelSelect_name == "Natural Gradient Boosting Classifier":
    st.write("""
        ## **Natural Gradient Boosting Classification Model**

        """)
    st.write("""
    ### **Multivariate Classification Method**

    """)
    ClassifierModels(modelSelect_name)

elif modelSelect_name == "Polynomial Regression":
    st.write("""
        ## **Polynomial Regression Model**

        """)
    st.write("""
    ### **Single-Variable Regression Method**

    """)
    PRM()

elif modelSelect_name == "Random Forest Regression":
    st.write("""
        ## **Multiple Random Forest Regression Model**

        """)
    st.write("""
    ### **Multiple Variable Regression Method**

    """)
    MRFR()

elif modelSelect_name == "NN - Multi-Layer Perceptron":

    modelSelect_Type = st.sidebar.selectbox(
        "Select a Method", ("Neural Network Regression", "Neural Network Classification"))

    if modelSelect_Type == "Neural Network Regression":

        st.write("""
        ## **Multi-Layer Perceptron Regressor Model**

        """)

        st.write("""
    ### **Neural Network (supervised) Regression Method**

    """)
        MLPR()
    else:
        st.write("""
        ## **Multi-Layer Perceptron Classification Model**

        """)
        st.write("""
    ### **Neural Network (supervised) Classification Method**

    """)
        MLPC()

elif modelSelect_name == "K-Means Clustering":
    st.write("""
        ## **K-Means Clustering Model**

        """)

    st.write("""
    ### **Unsupervised Learning Method**

    """)
    ClusteringModels(modelSelect_name)


elif modelSelect_name == "Hierarchical Clustering":
    st.write("""
        ## **Hierarchical Clustering Model**

        """)

    st.write("""
    ### **Unsupervised Learning Method**

    """)
    ClusteringModels(modelSelect_name)

elif modelSelect_name == "Spectral Clustering":
    st.write("""
        ## **Spectral Clustering Model**

        """)

    st.write("""
    ### **Unsupervised Learning Method**

    """)
    ClusteringModels(modelSelect_name)

elif modelSelect_name == "Quick Comparison Classifiers":
    st.write("""
        ## **Quick Comparison Classifiers**

        """)
    QuickClassifiers()

elif modelSelect_name == "Quick Comparison Regressors":
    st.write("""
        ## **Quick Comparison Regressors**

        """)
    QuickRegressors()

elif modelSelect_name == "Bigdata Analysis":
    st.write("""
        ## **Bigdata Analysis**


        """)

    SelectMethod = st.sidebar.selectbox("Select a method", ("Regression",
                                                            "Classification", "Clustering", "Regressor Ranking", "Classifier Ranking"))

    if SelectMethod == "Regression":
        selectModelRegressor = st.sidebar.selectbox("Select a model", ("Linear Regressor", "Generalized Linear Regressor", "Decision Tree Regressor",
                                                                       "Random Forest Regressor", "Gradient-Boosted Tree Regressor"))

        RegressorsPyspark(selectModelRegressor)

    elif SelectMethod == "Classification":
        selectModelClassifier = st.sidebar.selectbox("Select a model", ("Decision Tree Classifier", "Logistic Regression Classifier",
                                                                        "Random Forest Classifier", "Navy Bayes Classifier"))

        ClassifiersPyspark(selectModelClassifier)

    elif SelectMethod == "Regressor Ranking" or SelectMethod == "Classifier Ranking":
        Pyspark(SelectMethod)

    elif SelectMethod == "Clustering":
        selectModelClustering = st.sidebar.selectbox(
            "Select a model", ("K-Means", "Gaussian Mixture"))
        ClusteringPyspark(selectModelClustering)


elif modelSelect_name == "Time Series":
    st.write("""
    ## **Time Series**

    """)
    TimeSeries()

elif modelSelect_name == "Anomaly Detection":
    st.write("""
    ## **Anomaly Detection in Time Series**

    """)
    AnomalyDetection()
