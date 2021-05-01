import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PowerTransformer
#from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import Normalizer

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from ngboost import NGBClassifier
from ngboost.distns import k_categorical, Bernoulli

"""
The installation of the NGBoost library uninstalls the latest version of scikit-learn-0.24.0 and 
installs the version of scikit-learn-0.23.2, this apparently brings problems with the arguments 
of the Logistic Regression and Gaussian Process Classifier models, but nothing that cannot be solved
(delet n_jobs=-1). Also brings problem with Tweedie Regressor, Huber Regressor, perceptron
"""

from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score, recall_score, precision_score
from streamlit import caching

import time

# Next Models
"""
from sklearn.ensemble import VotingClassifier
"""


def QuickClassifiers():
    try:
        parameter_test_size = st.sidebar.slider(
            "Test size (fraction)", 0.02, 0.80, 0.2)
        st.sidebar.write("Test size: ", parameter_test_size)

        st.sidebar.info("""
                                [More information](http://gonzalezmaw.pythonanywhere.com/)
                                """)

        uploaded_file = st.file_uploader("Choose a CSV file")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            df = df.interpolate(method='linear', axis=0).bfill()
            df = df.dropna()
            df = df.drop_duplicates()

            """
            ##Example for data pre-processing
            #Drop irrelevant columns (Ticket and PassengerId)
            data.drop(['Ticket', 'PassengerId'], axis=1, inplace=True) 
            
            #Remap Sex column to zeros and ones
            gender_mapper = {'male': 0, 'female': 1} 
            data['Sex'].replace(gender_mapper, inplace=True)
            
            #Check if a passenger had a unique title (like doctor) or had something more generic (like Mr., Miss.) 
            #can be extracted from the Name column
            data['Title'] = data['Name'].apply(lambda x: x.split(',')[1].strip().split(' ')[0]) 
            data['Title'] = [0 if x in ['Mr.', 'Miss.', 'Mrs.'] else 1 for x in data['Title']] 
            data = data.rename(columns={'Title': 'Title_Unusual'}) 
            data.drop('Name', axis=1, inplace=True) 

            #Check if cabin information was known — if the value of Cabin column is not NaN
            data['Cabin_Known'] = [0 if str(x) == 'nan' else 1 for x in data['Cabin']] 
            data.drop('Cabin', axis=1, inplace=True) 
            
            #Create dummy variables from the Embarked column — 3 options
            emb_dummies = pd.get_dummies(data['Embarked'], drop_first=True, prefix='Embarked') 
            data = pd.concat([data, emb_dummies], axis=1) 
            data.drop('Embarked', axis=1, inplace=True) 
            
            #Fill Age values with the simple mean
            data['Age'] = data['Age'].fillna(int(data['Age'].mean())) 
            """

            # Taking N% of the data for training and (1-N%) for testing:
            num = int(len(df)*(1-parameter_test_size))
            # training data:
            data = df
            train = df[:num]
            # Testing data:
            test = df[num:]

            showData = st.checkbox('Show Dataset')

            if showData:
                st.subheader('Dataset')
                st.write(df)

            st.subheader("Data Processing:")

            col1, col2, col3, col4 = st.beta_columns(4)

            with col1:
                st.write("Shape of dataset:")
                st.info(df.shape)
            with col2:
                st.write("Complete data:")
                st.info(len(data))
            with col3:
                st.write("Data to train:")
                st.info(len(train))
            with col4:
                st.write("Data to test:")
                st.info(len(test))

            st.write("Descriptive statistics:")
            st.write(df.describe())

            col1, col2 = st.beta_columns(2)

            with col1:
                columsList = df.columns.tolist()

                columsSelectX = st.multiselect(
                    "Select X columns for the analysis", columsList)

            with col2:

                index2 = []
                # index = list()
                select2 = ""
                for x in columsList:
                    for y in columsSelectX:
                        if x == y:
                            index = columsList.index(x)
                            index2.append(index)
                            select2 = np.delete(columsList, index2)

                if select2 is not "":
                    columSelectY = st.selectbox(
                        "Select the Target for the analysis", select2)
                else:
                    columSelectY = st.selectbox(
                        "Select the Target for the analysis", columsList)

            if len(columsSelectX) > 0 and columSelectY != None:

                X = df[columsSelectX]
                y = df[columSelectY]  # Selecting the last column as Y

                Stan_Scaler = st.checkbox(
                    'Data Scaler')
                if Stan_Scaler:
                    Select_Type = st.selectbox(
                        "Select a method", ("MinMax Scaler", "Standard Scaler", "MaxAbs Scaler", "Robust Scaler", "Power Transformer", "Normalizer"))
                    if Select_Type == "Standard Scaler":
                        sc = StandardScaler()
                    elif Select_Type == "MinMax Scaler":
                        sc = MinMaxScaler()
                    elif Select_Type == "MaxAbs Scaler":
                        sc = MaxAbsScaler()
                    elif Select_Type == "Robust Scaler":
                        sc = RobustScaler()
                    elif Select_Type == "Power Transformer":
                        sc = PowerTransformer()
                    elif Select_Type == "Normalizer":
                        sc = Normalizer()

                    X = sc.fit_transform(X)
                    # Selecting the last column as Y
                    y = df[columSelectY]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=parameter_test_size, random_state=0)

                if st.button('Analysis'):

                    # Support Vector Machine Classifer (SVC)
                    classifier = SVC(random_state=0)
                    start_time_1 = time.time()
                    classifier.fit(X_train, y_train)
                    y_pred = classifier.predict(X_test)
                    # acc = round(accuracy_score(y_test, y_pred), 5)
                    time_1 = round(time.time() - start_time_1, 4)
                    acc1 = round(accuracy_score(y_test, y_pred), 4)
                    mod_1 = "Support Vector Machine"
                    f1 = round(f1_score(y_test, y_pred, average="weighted"), 4)
                    prec_1 = round(precision_score(
                        y_test, y_pred, average="weighted"), 4)
                    fpr, tpr, threshold = roc_curve(
                        y_test, y_pred, pos_label=2)
                    auc_1 = round(auc(fpr, tpr), 4)
                    f1 = round(f1_score(y_test, y_pred, average="weighted"), 4)
                    rec_1 = round(recall_score(
                        y_test, y_pred, average="weighted"), 4)
                    # st.write(rec_1)

                    # Logistic Regression Classifier
                    classifier = LogisticRegression(random_state=0)
                    start_time_2 = time.time()
                    classifier.fit(X_train, y_train)
                    y_pred = classifier.predict(X_test)
                    time_2 = round(time.time() - start_time_2, 4)
                    acc2 = round(accuracy_score(y_test, y_pred), 4)
                    mod_2 = "Logistic Regression"
                    prec_2 = round(precision_score(
                        y_test, y_pred, average="weighted"), 4)
                    fpr, tpr, threshold = roc_curve(
                        y_test, y_pred, pos_label=2)
                    auc_2 = round(auc(fpr, tpr), 4)
                    f2 = round(f1_score(y_test, y_pred, average="weighted"), 4)
                    rec_2 = round(recall_score(
                        y_test, y_pred, average="weighted"), 4)
                    # st.write(rec_1)

                    # Naive Bayes Classifier
                    classifier = GaussianNB()
                    start_time_3 = time.time()
                    classifier.fit(X_train, y_train)
                    y_pred = classifier.predict(X_test)
                    time_3 = round(time.time() - start_time_3, 4)
                    acc3 = round(accuracy_score(y_test, y_pred), 4)
                    mod_3 = "Naive Bayes"
                    prec_3 = round(precision_score(
                        y_test, y_pred, average="weighted"), 4)
                    fpr, tpr, threshold = roc_curve(
                        y_test, y_pred, pos_label=2)
                    auc_3 = round(auc(fpr, tpr), 4)
                    f3 = round(f1_score(y_test, y_pred, average="weighted"), 4)
                    rec_3 = round(recall_score(
                        y_test, y_pred, average="weighted"), 4)
                    # st.write(rec_1)

                    # Decision Tree Classifier
                    classifier = DecisionTreeClassifier(random_state=0)
                    start_time_4 = time.time()
                    classifier.fit(X_train, y_train)
                    y_pred = classifier.predict(X_test)
                    time_4 = round(time.time() - start_time_4, 4)
                    acc4 = round(accuracy_score(y_test, y_pred), 4)
                    mod_4 = "Decision Tree"
                    prec_4 = round(precision_score(
                        y_test, y_pred, average="weighted"), 4)
                    fpr, tpr, threshold = roc_curve(
                        y_test, y_pred, pos_label=2)
                    auc_4 = round(auc(fpr, tpr), 4)
                    f4 = round(f1_score(y_test, y_pred, average="weighted"), 4)
                    rec_4 = round(recall_score(
                        y_test, y_pred, average="weighted"), 4)
                    # st.write(rec_1)

                    # Random Forest Classifier
                    classifier = RandomForestClassifier(
                        n_jobs=-1, random_state=0)
                    start_time_5 = time.time()
                    classifier.fit(X_train, y_train)
                    y_pred = classifier.predict(X_test)
                    time_5 = round(time.time() - start_time_5, 4)
                    acc5 = round(accuracy_score(y_test, y_pred), 4)
                    mod_5 = "Random Forest"
                    prec_5 = round(precision_score(
                        y_test, y_pred, average="weighted"), 4)
                    fpr, tpr, threshold = roc_curve(
                        y_test, y_pred, pos_label=2)
                    auc_5 = round(auc(fpr, tpr), 4)
                    f5 = round(f1_score(y_test, y_pred, average="weighted"), 4)
                    rec_5 = round(recall_score(
                        y_test, y_pred, average="weighted"), 4)
                    # st.write(rec_1)

                    # K Neighbors Classifier
                    classifier = KNeighborsClassifier(n_jobs=-1)
                    start_time_6 = time.time()
                    classifier.fit(X_train, y_train)
                    y_pred = classifier.predict(X_test)
                    time_6 = round(time.time() - start_time_6, 4)
                    acc6 = round(accuracy_score(y_test, y_pred), 4)
                    mod_6 = "K Nearest Neighbors"
                    prec_6 = round(precision_score(
                        y_test, y_pred, average="weighted"), 4)
                    fpr, tpr, threshold = roc_curve(
                        y_test, y_pred, pos_label=2)
                    auc_6 = round(auc(fpr, tpr), 4)
                    f6 = round(f1_score(y_test, y_pred, average="weighted"), 4)
                    rec_6 = round(recall_score(
                        y_test, y_pred, average="weighted"), 4)

                    # Gradient Boosting Classifier
                    classifier = GradientBoostingClassifier(random_state=0)
                    start_time_7 = time.time()
                    classifier.fit(X_train, y_train)
                    y_pred = classifier.predict(X_test)
                    time_7 = round(time.time() - start_time_7, 4)
                    acc7 = round(accuracy_score(y_test, y_pred), 4)
                    mod_7 = "Gradient Boosting"
                    prec_7 = round(precision_score(
                        y_test, y_pred, average="weighted"), 4)
                    fpr, tpr, threshold = roc_curve(
                        y_test, y_pred, pos_label=2)
                    auc_7 = round(auc(fpr, tpr), 4)
                    f7 = round(f1_score(y_test, y_pred, average="weighted"), 4)
                    rec_7 = round(recall_score(
                        y_test, y_pred, average="weighted"), 4)

                    # Extreme Gradient Boosting Classifier
                    classifier = XGBClassifier(n_jobs=-1, random_state=0)
                    start_time_8 = time.time()
                    classifier.fit(X_train, y_train)
                    y_pred = classifier.predict(X_test)
                    time_8 = round(time.time() - start_time_8, 4)
                    acc8 = round(accuracy_score(y_test, y_pred), 4)
                    mod_8 = "Extreme Gradient Boosting"
                    prec_8 = round(precision_score(
                        y_test, y_pred, average="weighted"), 4)
                    fpr, tpr, threshold = roc_curve(
                        y_test, y_pred, pos_label=2)
                    auc_8 = round(auc(fpr, tpr), 4)
                    f8 = round(f1_score(y_test, y_pred, average="weighted"), 4)
                    rec_8 = round(recall_score(
                        y_test, y_pred, average="weighted"), 4)

                    # Gaussian Process Classifier
                    kernel = DotProduct() + WhiteKernel()
                    start_time_9 = time.time()
                    classifier = GaussianProcessClassifier(
                        kernel=kernel, random_state=0)
                    classifier.fit(X_train, y_train)
                    y_pred = classifier.predict(X_test)
                    time_9 = round(time.time() - start_time_9, 4)
                    acc9 = round(accuracy_score(y_test, y_pred), 4)
                    mod_9 = "Gaussian Process"
                    prec_9 = round(precision_score(
                        y_test, y_pred, average="weighted"), 4)
                    fpr, tpr, threshold = roc_curve(
                        y_test, y_pred, pos_label=2)
                    auc_9 = round(auc(fpr, tpr), 4)
                    f9 = round(f1_score(y_test, y_pred, average="weighted"), 4)
                    rec_9 = round(recall_score(
                        y_test, y_pred, average="weighted"), 4)

                    # Stochastic Gradient Descent
                    classifier = SGDClassifier(n_jobs=-1, random_state=0)
                    start_time_10 = time.time()
                    classifier.fit(X_train, y_train)
                    y_pred = classifier.predict(X_test)
                    time_10 = round(time.time() - start_time_10, 4)
                    acc10 = round(accuracy_score(y_test, y_pred), 4)
                    mod_10 = "Stochastic Gradient Descent"
                    prec_10 = round(precision_score(
                        y_test, y_pred, average="weighted"), 4)
                    fpr, tpr, threshold = roc_curve(
                        y_test, y_pred, pos_label=2)
                    auc_10 = round(auc(fpr, tpr), 4)
                    f10 = round(
                        f1_score(y_test, y_pred, average="weighted"), 4)
                    rec_10 = round(recall_score(
                        y_test, y_pred, average="weighted"), 4)

                    # Light Boosting Machine Classifier
                    classifier = LGBMClassifier(n_jobs=-1, random_state=0)
                    start_time_11 = time.time()
                    classifier.fit(X_train, y_train)
                    y_pred = classifier.predict(X_test)
                    time_11 = round(time.time() - start_time_11, 4)
                    acc11 = round(accuracy_score(y_test, y_pred), 4)
                    mod_11 = "Light Gradient Boosting Machine"
                    prec_11 = round(precision_score(
                        y_test, y_pred, average="weighted"), 4)
                    fpr, tpr, threshold = roc_curve(
                        y_test, y_pred, pos_label=2)
                    auc_11 = round(auc(fpr, tpr), 4)
                    f11 = round(
                        f1_score(y_test, y_pred, average="weighted"), 4)
                    rec_11 = round(recall_score(
                        y_test, y_pred, average="weighted"), 4)

                    # CatBost Classifier
                    classifier = CatBoostClassifier(random_state=0)
                    start_time_12 = time.time()
                    classifier.fit(X_train, y_train)
                    y_pred = classifier.predict(X_test)
                    time_12 = round(time.time() - start_time_12, 4)
                    acc12 = round(accuracy_score(y_test, y_pred), 4)
                    mod_12 = "CatBost"
                    prec_12 = round(precision_score(
                        y_test, y_pred, average="weighted"), 4)
                    fpr, tpr, threshold = roc_curve(
                        y_test, y_pred, pos_label=2)
                    auc_12 = round(auc(fpr, tpr), 4)
                    f12 = round(
                        f1_score(y_test, y_pred, average="weighted"), 4)
                    rec_12 = round(recall_score(
                        y_test, y_pred, average="weighted"), 4)

                    # AdaBoost Classifier
                    classifier = AdaBoostClassifier(random_state=0)
                    start_time_13 = time.time()
                    classifier.fit(X_train, y_train)
                    y_pred = classifier.predict(X_test)
                    time_13 = round(time.time() - start_time_13, 4)
                    acc13 = round(accuracy_score(y_test, y_pred), 4)
                    mod_13 = "AdaBoost"
                    prec_13 = round(precision_score(
                        y_test, y_pred, average="weighted"), 4)
                    fpr, tpr, threshold = roc_curve(
                        y_test, y_pred, pos_label=2)
                    auc_13 = round(auc(fpr, tpr), 4)
                    f13 = round(
                        f1_score(y_test, y_pred, average="weighted"), 4)
                    rec_13 = round(recall_score(
                        y_test, y_pred, average="weighted"), 4)

                    # Extra Tree Classifier
                    classifier = ExtraTreeClassifier(random_state=0)
                    start_time_14 = time.time()
                    classifier.fit(X_train, y_train)
                    y_pred = classifier.predict(X_test)
                    time_14 = round(time.time() - start_time_14, 4)
                    acc14 = round(accuracy_score(y_test, y_pred), 4)
                    mod_14 = "Extra Tree"
                    prec_14 = round(precision_score(
                        y_test, y_pred, average="weighted"), 4)
                    fpr, tpr, threshold = roc_curve(
                        y_test, y_pred, pos_label=2)
                    auc_14 = round(auc(fpr, tpr), 4)
                    f14 = round(
                        f1_score(y_test, y_pred, average="weighted"), 4)
                    rec_14 = round(recall_score(
                        y_test, y_pred, average="weighted"), 4)

                    # Bagging Classifier
                    classifier = BaggingClassifier(random_state=0)
                    start_time_15 = time.time()
                    classifier.fit(X_train, y_train)
                    y_pred = classifier.predict(X_test)
                    time_15 = round(time.time() - start_time_15, 4)
                    acc15 = round(accuracy_score(y_test, y_pred), 4)
                    mod_15 = "Bagging"
                    prec_15 = round(precision_score(
                        y_test, y_pred, average="weighted"), 4)
                    fpr, tpr, threshold = roc_curve(
                        y_test, y_pred, pos_label=2)
                    auc_15 = round(auc(fpr, tpr), 4)
                    f15 = round(
                        f1_score(y_test, y_pred, average="weighted"), 4)
                    rec_15 = round(recall_score(
                        y_test, y_pred, average="weighted"), 4)

                    # Passive Aggressive Classifier
                    classifier = PassiveAggressiveClassifier(random_state=0)
                    start_time_16 = time.time()
                    classifier.fit(X_train, y_train)
                    y_pred = classifier.predict(X_test)
                    time_16 = round(time.time() - start_time_16, 4)
                    acc16 = round(accuracy_score(y_test, y_pred), 4)
                    mod_16 = "Passive Aggressive"
                    prec_16 = round(precision_score(
                        y_test, y_pred, average="weighted"), 4)
                    fpr, tpr, threshold = roc_curve(
                        y_test, y_pred, pos_label=2)
                    auc_16 = round(auc(fpr, tpr), 4)
                    f16 = round(
                        f1_score(y_test, y_pred, average="weighted"), 4)
                    rec_16 = round(recall_score(
                        y_test, y_pred, average="weighted"), 4)

                    # Linear Discriminant Analysis
                    classifier = LinearDiscriminantAnalysis()
                    start_time_17 = time.time()
                    classifier.fit(X_train, y_train)
                    y_pred = classifier.predict(X_test)
                    time_17 = round(time.time() - start_time_17, 4)
                    acc17 = round(accuracy_score(y_test, y_pred), 4)
                    mod_17 = "Linear Discriminant Analysis"
                    prec_17 = round(precision_score(
                        y_test, y_pred, average="weighted"), 4)
                    fpr, tpr, threshold = roc_curve(
                        y_test, y_pred, pos_label=2)
                    auc_17 = round(auc(fpr, tpr), 4)
                    f17 = round(
                        f1_score(y_test, y_pred, average="weighted"), 4)
                    rec_17 = round(recall_score(
                        y_test, y_pred, average="weighted"), 4)

                    # Quadratic Discriminant Analysis
                    classifier = QuadraticDiscriminantAnalysis()
                    start_time_18 = time.time()
                    classifier.fit(X_train, y_train)
                    y_pred = classifier.predict(X_test)
                    time_18 = round(time.time() - start_time_18, 4)
                    acc18 = round(accuracy_score(y_test, y_pred), 4)
                    mod_18 = "Quadratic Discriminant Analysis"
                    prec_18 = round(precision_score(
                        y_test, y_pred, average="weighted"), 4)
                    fpr, tpr, threshold = roc_curve(
                        y_test, y_pred, pos_label=2)
                    auc_18 = round(auc(fpr, tpr), 4)
                    f18 = round(
                        f1_score(y_test, y_pred, average="weighted"), 4)
                    rec_18 = round(recall_score(
                        y_test, y_pred, average="weighted"), 4)

                    # Linear Support Vector Classifer (SVC)
                    classifier = LinearSVC(random_state=0)
                    start_time_19 = time.time()
                    classifier.fit(X_train, y_train)
                    y_pred = classifier.predict(X_test)
                    time_19 = round(time.time() - start_time_19, 4)
                    acc19 = round(accuracy_score(y_test, y_pred), 4)
                    mod_19 = "Linear Support Vector Machine"
                    f19 = round(
                        f1_score(y_test, y_pred, average="weighted"), 4)
                    prec_19 = round(precision_score(
                        y_test, y_pred, average="weighted"), 4)
                    fpr, tpr, threshold = roc_curve(
                        y_test, y_pred, pos_label=2)
                    auc_19 = round(auc(fpr, tpr), 4)
                    f19 = round(
                        f1_score(y_test, y_pred, average="weighted"), 4)
                    rec_19 = round(recall_score(
                        y_test, y_pred, average="weighted"), 4)

                    # Ridge Classifier
                    classifier = RidgeClassifier(random_state=0)
                    start_time_20 = time.time()
                    classifier.fit(X_train, y_train)
                    y_pred = classifier.predict(X_test)
                    time_20 = round(time.time() - start_time_20, 4)
                    acc20 = round(accuracy_score(y_test, y_pred), 4)
                    mod_20 = "Ridge"
                    f20 = round(
                        f1_score(y_test, y_pred, average="weighted"), 4)
                    prec_20 = round(precision_score(
                        y_test, y_pred, average="weighted"), 4)
                    fpr, tpr, threshold = roc_curve(
                        y_test, y_pred, pos_label=2)
                    auc_20 = round(auc(fpr, tpr), 4)
                    f20 = round(
                        f1_score(y_test, y_pred, average="weighted"), 4)
                    rec_20 = round(recall_score(
                        y_test, y_pred, average="weighted"), 4)

                    # Natutal Gradient Boosting
                    k = len(np.unique(y))
                    classifier = GradientBoostingClassifier(random_state=0)

                    try:
                        classifier = NGBClassifier(
                            Dist=k_categorical(k), verbose=False)
                        start_time_21 = time.time()
                        classifier.fit(X_train, y_train)
                        y_pred = classifier.predict(X_test)
                    except:
                        classifier = GradientBoostingClassifier(random_state=0)
                        start_time_21 = time.time()
                        classifier.fit(X_train, y_train)
                        y_pred = classifier.predict(X_test)

                    time_21 = round(time.time() - start_time_21, 4)
                    acc21 = round(accuracy_score(y_test, y_pred), 4)
                    mod_21 = "Natutal Gradient Boosting"
                    f21 = round(
                        f1_score(y_test, y_pred, average="weighted"), 4)
                    prec_21 = round(precision_score(
                        y_test, y_pred, average="weighted"), 4)
                    fpr, tpr, threshold = roc_curve(
                        y_test, y_pred, pos_label=2)
                    auc_21 = round(auc(fpr, tpr), 4)
                    f21 = round(
                        f1_score(y_test, y_pred, average="weighted"), 4)
                    rec_21 = round(recall_score(
                        y_test, y_pred, average="weighted"), 4)

                    #from ngboost import NGBClassifier

                    print(threshold)

                    st.subheader("Accuracy Scores:")
                    col1, col2 = st.beta_columns(2)

                    with col1:
                        st.info("Support Vector Machines: " + str(acc1))
                        st.info("Linear Support Vector Machines: " + str(acc19))
                        st.info("Logistic Regression: " + str(acc2))
                        st.info("Naive Bayes: " + str(acc3))
                        st.info("Gradient Boosting: " + str(acc7))
                        st.info("Extreme Gradient Boosting: " + str(acc8))
                        st.info("Light Boosting Machine: " + str(acc11))
                        st.info("AdaBoost: " + str(acc13))
                        st.info("Bagging: " + str(acc15))
                        st.info("Linear Discriminant Analysis: " + str(acc17))
                        st.info("Natural Gradient Boosting: " + str(acc21))

                    with col2:
                        st.info("Decision Tree: " + str(acc4))
                        st.info("Random Forest: " + str(acc5))
                        st.info("K Nearest Neighbors Classifier: " + str(acc6))
                        st.info("Gaussian Process Classifier: " + str(acc9))
                        st.info(
                            "Stochastic Gradient Descent Classifier: " + str(acc10))
                        st.info("CatBoost Classifier: " + str(acc12))
                        st.info("Extra Tree Classifier: " + str(acc14))
                        st.info("Passive Aggressive Classifier: " + str(acc16))
                        st.info("Quadratic Discriminant Analysis: " + str(acc18))
                        st.info("Ridge: " + str(acc20))

                    st.subheader(
                        "Models Runtime: Training + Testing (seconds):")

                    col1, col2 = st.beta_columns(2)

                    with col1:
                        st.info(
                            "Support Vector Machines Classifer: " + str(time_1))
                        st.info("Linear Support Vector Machines: " + str(time_19))
                        st.info("Logistic Regression Classifier: " + str(time_2))
                        st.info("Naive Bayes Classifier: " + str(time_3))
                        st.info("Gradient Boosting: " + str(time_7))
                        st.info("Extreme Gradient Boosting: " + str(time_8))
                        st.info("Light Boosting Machine: " + str(time_11))
                        st.info("AdaBoost: " + str(time_13))
                        st.info("Bagging: " + str(time_15))
                        st.info("Linear Discriminant Analysis: " + str(time_17))
                        st.info("Natural Gradient Boosting: " + str(time_21))

                    with col2:
                        st.info("Decision Tree: " + str(time_4))
                        st.info("Random Forest: " + str(time_5))
                        st.info("K Nearest Neighbors: " + str(time_6))
                        st.info("Gaussian Process: " + str(time_9))
                        st.info("Stochastic Gradient Descent: " + str(time_10))
                        st.info("CatBoost: " + str(time_12))
                        st.info("Extra Trees: " + str(time_14))
                        st.info("Passive Aggressive: " + str(time_16))
                        st.info(
                            "Quadratic Discriminant Analysis: " + str(time_18))
                        st.info("Ridge: " + str(time_20))

                    # figure, ax = plt.subplots()
                    fig = plt.figure(figsize=(15, 12))
                    ax1 = fig.add_subplot(121)

                    X_acc = ['SVMC', 'LSVMC', 'LRC', 'NBC', 'DTC', 'RFC',
                             'KNNC', 'GBC', 'XGBC', 'GPC', 'SGDC',
                             'LGBMC', 'CatBC', 'AdaBC', 'ETC', 'BC',
                             'PAC', 'LDAC', 'QDAC', 'RC', 'NGBC']
                    y_acc = [acc1, acc19, acc2, acc3, acc4, acc5,
                             acc6, acc7, acc8, acc9, acc10,
                             acc11, acc12, acc13, acc14, acc15,
                             acc16, acc17, acc18, acc20, acc21]

                    z_acc = [time_1, time_19, time_2, time_3, time_4, time_5,
                             time_6, time_7, time_8, time_9, time_10,
                             time_11, time_12, time_13, time_14, time_15,
                             time_16, time_17, time_18, time_20, time_21]

                    dfchart = [['SVMC', mod_1, acc1, auc_1, prec_1, rec_1, f1, time_1],
                               ['LSVMC', mod_19, acc19, auc_19,
                                prec_19, rec_19, f19, time_19],
                               ['LRC', mod_2, acc2, auc_2,
                                   prec_2, rec_2, f2, time_2],
                               ['NBC', mod_3, acc3, auc_3,
                                   prec_3, rec_3, f3, time_3],
                               ['DTC', mod_4, acc4, auc_4,
                                   prec_4, rec_4, f4, time_4],
                               ['RFC', mod_5, acc5, auc_5,
                                   prec_5, rec_5, f5, time_5],
                               ['KNNC', mod_6, acc6, auc_6,
                                   prec_6, rec_6, f6, time_6],
                               ['GBC', mod_7, acc7, auc_7,
                                   prec_7, rec_7, f7, time_7],
                               ['XGBC', mod_8, acc8, auc_8,
                                   prec_8, rec_8, f8, time_8],
                               ['GPC', mod_9, acc9, auc_9,
                                   prec_9, rec_9, f9, time_9],
                               ['SGDC', mod_10, acc10, auc_10,
                                prec_10, rec_10, f10, time_10],
                               ['LGBMC', mod_11, acc11, auc_11,
                                prec_11, rec_11, f11, time_11],
                               ['CatBC', mod_12, acc12, auc_12,
                                prec_12, rec_12, f12, time_12],
                               ['AdaBC', mod_13, acc13, auc_13,
                                prec_13, rec_13, f13, time_13],
                               ['ETC', mod_14, acc14, auc_14,
                                prec_14, rec_14, f14, time_14],
                               ['BC', mod_15, acc15, auc_15,
                                prec_15, rec_15, f15, time_15],
                               ['PAC', mod_16, acc16, auc_16,
                                prec_16, rec_16, f16, time_16],
                               ['LDAC', mod_17, acc17, auc_17,
                                prec_17, rec_17, f17, time_17],
                               ['QDAC', mod_18, acc18, auc_18,
                                prec_18, rec_18, f18, time_18],
                               ['RC', mod_20, acc20, auc_20,
                                prec_20, rec_20, f20, time_20],
                               ['NGBC', mod_21, acc21, auc_21,
                                prec_21, rec_21, f21, time_21]]

                    dfchart = pd.DataFrame(
                        dfchart, columns=['Rank', 'Classifier', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1-score', 'Time(sec)'])

                    df_New = dfchart.sort_values(
                        by='Accuracy', ascending=False)

                    st.write(df_New)

                    df_New2 = dfchart.sort_values(
                        by='Accuracy', ascending=True)

                    X_acc = df_New2.iloc[:, 0]
                    # Selecting the last column as Y
                    y_acc = df_New2.iloc[:, 2]
                    z_acc = df_New2.iloc[:, 7]

                    X_pos = np.arange(len(X_acc))

                    #x_min = min(y_acc)*0.975
                    x_min = min(y_acc)*0.995
                    x_max = max(y_acc)*1.005
                    #x_max = 1

                    # ticks = np.arange(min(y_acc)*0.975, max(y_acc)
                    # * 1.0225, (x_max-x_min)/5)
                    ticks = np.arange(min(y_acc)*0.995, max(y_acc)
                                      * 1.005, (x_max-x_min)/5)

                    ax1.barh(X_pos, y_acc, alpha=0.7, color='deepskyblue')

                    # Añadimos la etiqueta de nombre de cada lenguaje en su posicion correcta
                    plt.yticks(X_pos, X_acc)
                    plt.yticks(X_pos, X_acc, fontsize=15)
                    plt.xticks(ticks, fontsize=15)
                    #plt.xlim(min(y_acc)*0.975, max(y_acc)*1.0225)
                    plt.xlim(min(y_acc)*0.995, max(y_acc)*1.0005)
                    #plt.xlim(round(min(y_acc)*0.975, 3), 1.000)
                    plt.title("Classification Models Ranking", fontsize=20)
                    plt.xlabel("Accuracy Score", fontsize=20)
                    plt.ylabel("Classifier Models", fontsize=18)

                    ax2 = fig.add_subplot(122)

                    x_min = min(z_acc)*0.975
                    x_max = max(z_acc)*1.0225

                    ticks = np.arange(min(z_acc)*0.975, max(z_acc)
                                      * 1.0225, (x_max-x_min)/5)

                    ax2.barh(X_pos, z_acc,  alpha=0.7, color='lightgreen')

                    plt.yticks(X_pos, X_acc, fontsize=15)
                    plt.xticks(ticks, fontsize=15)
                    plt.xlim(min(z_acc)*0.975, max(z_acc)*1.0225)
                    plt.title("Classification Models Runtime", fontsize=20)
                    plt.xlabel(
                        "Runtime: Training + Testing (seconds)", fontsize=18)

                    st.pyplot(fig)

                    if st.button('Re-Run'):
                        caching.clear_cache()
                        st._RerunException

        else:
            st.info('Awaiting for CSV file to be uploaded.')

    except:
        st.info('ERROR - Please check your Dataset, parameters o selected model')

    return
