import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PowerTransformer
# from sklearn.preprocessing import QuantileTransformer
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
#from ngboost import NGBClassifier
from ngboost.distns import k_categorical, Bernoulli

from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

from sklearn.metrics import accuracy_score
from streamlit import caching

from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix

# Next Models
"""
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from catboost import CatBoostClassifier
"""

def ClassifierModels(ClassifierSelectModel):
    try:
        parameter_test_size = st.sidebar.slider(
            "Test size (fraction)", 0.02, 0.80, 0.2)
        st.sidebar.write("Test size: ", parameter_test_size)

        if ClassifierSelectModel == "Random Forest Classifier":
            max_depth = st.sidebar.slider("Maximum depth", 2, 20)
            st.sidebar.write("Maximum depth: ", max_depth)

            n_estimators = st.sidebar.slider("n estimators", 1, 100)
            st.sidebar.write("n estimators: ",  n_estimators)

        if ClassifierSelectModel == "K-Nearest Neighbors Classifier":
            K_parameter = st.sidebar.slider("K parameter", 1, 20, 5)
            st.sidebar.write("K parameter: ", K_parameter)

        if ClassifierSelectModel == "Support Vector Machines Classifier":
            param_C = st.sidebar.slider("C parameter", 0.01, 10.0, 1.0)
            st.sidebar.write("C parameter: ", param_C)
            kernelType = st.sidebar.selectbox("Kernel type",
                                              ("Linear", "Polynomial", "Gaussian", "Sigmoid"))
            if kernelType == "Polynomial":
                degreePoly = st.sidebar.slider("Polynomial degree", 1, 10, 2)
                st.sidebar.write("Polynomial degree: ", degreePoly)

        st.sidebar.info("""
                                [More information](http://gonzalezmaw.pythonanywhere.com/)
                                """)

        uploaded_file = st.file_uploader("Choose a CSV file")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            df = df.interpolate(method='linear', axis=0).bfill()
            df = df.dropna()
            df = df.drop_duplicates()

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

                dfp = X
                dfp[columSelectY] = y

                Transform_data = st.checkbox(
                    'Transform Dataset (Categorial --> Numerical)')
                columsList = dfp.columns

                if Transform_data:

                    ColCategories = st.multiselect(
                        "Categorical Columns", columsList)
                    df2 = dfp

                    if ColCategories:
                        # st.write(ColCategories)
                        le = preprocessing.LabelEncoder()
                        df2[ColCategories] = df[ColCategories].apply(
                            le.fit_transform)
                        st.write(df2)

                    X = df2[columsSelectX]
                    y = df2[columSelectY]

                    pairplot_X = st.checkbox('Show Pairplot')
                    if pairplot_X:
                        fig = sns.pairplot(df2)
                        st.pyplot(fig)

                else:
                    X = df[columsSelectX]
                    y = df[columSelectY]  # Selecting the last column as Y
                    pairplot_X = st.checkbox('Show Pairplot')
                    if pairplot_X:
                        fig = sns.pairplot(dfp)
                        st.pyplot(fig)

                Stan_Scaler = st.checkbox('Data Scaler')
                if Stan_Scaler:
                    Select_Type = st.selectbox(
                        "Select a method", ("Standard Scaler", "MinMax Scaler", "MaxAbs Scaler", "Robust Scaler", "Power Transformer", "Normalizer"))
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
                    y = df[columSelectY]  # Selecting the last column as Y

                # Header name

                # X_name = df.iloc[:, 0].name
                # y_name = df.iloc[:, -1].name

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=parameter_test_size, random_state=0)

                # Model-1: Support Vector Machines Classifier
                if ClassifierSelectModel == "Support Vector Machines Classifier":
                    if kernelType == "Linear":
                        kernelType1 = "linear"
                    elif kernelType == "Polynomial":
                        kernelType1 = "poly"
                    elif kernelType == "Gaussian":
                        kernelType1 = "rbf"
                    elif kernelType == "Sigmoid":
                        kernelType1 = "sigmoid"

                    if kernelType == "poly":
                        classifier = SVC(
                            C=param_C, kernel=kernelType1, degree=degreePoly, random_state=0)
                    else:
                        classifier = SVC(
                            C=param_C, kernel=kernelType1, random_state=0)

                # Model-2: Logistic Regression Classifier
                elif ClassifierSelectModel == "Logistic Regression Classifier":
                    # classifier = LogisticRegression(n_jobs=-1, random_state=0)
                    classifier = LogisticRegression(random_state=0)

                # Model-3: Naive Bayes Classifier
                elif ClassifierSelectModel == "Naive Bayes Classifier":
                    classifier = GaussianNB()

                # Model-4: Decision Tree Classifier
                elif ClassifierSelectModel == "Decision Tree Classifier":
                    classifier = DecisionTreeClassifier(random_state=0)

                # Model-5: Extra Trees Classifier
                elif ClassifierSelectModel == "Extra Trees Classifier":
                    classifier = ExtraTreeClassifier(random_state=0)

                # Model-6: Random Forest Classifier
                elif ClassifierSelectModel == "Random Forest Classifier":
                    classifier = RandomForestClassifier(
                        n_estimators=n_estimators, max_depth=max_depth, random_state=0)

                # Model-7: K-Nearest Neighbors Classifier
                elif ClassifierSelectModel == "K-Nearest Neighbors Classifier":
                    # K Neighbors Classifier
                    classifier = KNeighborsClassifier(n_neighbors=K_parameter)

                # Model-8: Gradient Boosting Classifier
                elif ClassifierSelectModel == "Gradient Boosting Classifier":
                    classifier = GradientBoostingClassifier(random_state=0)

                # Model-9: Extreme Gradient Boosting Classifier
                elif ClassifierSelectModel == "Extreme Gradient Boosting Classifier":
                    classifier = XGBClassifier(n_jobs=-1, random_state=0)

                # Model-10: Gaussian Process Classifier
                elif ClassifierSelectModel == "Gaussian Process Classifier":
                    kernel = DotProduct() + WhiteKernel()
                    classifier = GaussianProcessClassifier(
                        kernel=kernel, random_state=0)
                    """
                        classifier = GaussianProcessClassifier(
                            kernel=kernel, random_state=0, n_jobs=-1)
                        """
                # Model-11: Stochastic Gradient Descent Classifier
                elif ClassifierSelectModel == "Stochastic Gradient Descent Classifier":
                    classifier = SGDClassifier(n_jobs=-1, random_state=0)

                # Model-12: Light Gradient Boosting Machine Classifier
                elif ClassifierSelectModel == "Light Gradient Boosting Machine Classifier":
                    classifier = LGBMClassifier(n_jobs=-1, random_state=0)

                # Model-13: CatBoost Classifier
                elif ClassifierSelectModel == "CatBoost Classifier":
                    classifier = CatBoostClassifier(random_state=0)

                # Model-14: AdaBoost Classifier
                elif ClassifierSelectModel == "AdaBoost Classifier":
                    classifier = AdaBoostClassifier(random_state=0)

                # Model-15: Bagging Classifier
                elif ClassifierSelectModel == "Bagging Classifier":
                    classifier = BaggingClassifier(random_state=0)

                # Model-16: Passive Aggressive Classifier
                elif ClassifierSelectModel == "Passive Aggressive Classifier":
                    classifier = PassiveAggressiveClassifier(random_state=0)

                # Model-17: Linear Discriminant Analysis Classifier
                elif ClassifierSelectModel == "Linear Discriminant Analysis Classifier":
                    classifier = LinearDiscriminantAnalysis()

                # Model-18: Quadratic Discriminant Analysis Classifier
                elif ClassifierSelectModel == "Quadratic Discriminant Analysis Classifier":
                    classifier = QuadraticDiscriminantAnalysis()

                # Model-19: Linear Support Vector Machine Classifier
                elif ClassifierSelectModel == "Linear Support Vector Machine Classifier":
                    classifier = LinearSVC(random_state=0)

                # Model-20: Ridge Classifier
                elif ClassifierSelectModel == "Ridge Classifier":
                    classifier = RidgeClassifier(random_state=0)

                # Model-21: Natutal Gradient Boosting
                elif ClassifierSelectModel == "Natural Gradient Boosting Classifier":
                    classifier = GradientBoostingClassifier(random_state=0)
                    """
                    k = len(np.unique(y))
                    classifier = NGBClassifier(
                        Dist=k_categorical(k), verbose=False)
                        """

                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)
                acc1 = round(accuracy_score(y_test, y_pred), 3)

                # PLOT
                pca = PCA(2)
                X_projected = pca.fit_transform(X)

                x1 = X_projected[:, 0]
                x2 = X_projected[:, 1]

                st.subheader("Classification:")
                st.write("**Accuracy:**")
                st.info(acc1)

                st.set_option('deprecation.showPyplotGlobalUse', False)
                plt.figure()
                plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
                plt.xlabel("Principal Component X1")
                plt.ylabel("Principal Component X2")
                plt.title(ClassifierSelectModel)
                plt.colorbar()
                st.pyplot()
                st.set_option('deprecation.showPyplotGlobalUse', False)

                # st.info(classification_report(y_test, y_pred))

                uploaded_file_target = st.file_uploader(
                    "Choose a new CSV file for prediction")
                if uploaded_file_target is not None:
                    dfT = pd.read_csv(uploaded_file_target)
                    dfT = dfT.interpolate(method='linear', axis=0).bfill()
                    dfT = dfT.dropna()
                    dfT = dfT.drop_duplicates()
                    dfT = dfT

                    # Using all column except for the last column as X
                    X_new = dfT[columsSelectX]
                    # st.write(X_new)

                    columsList = df.columns

                    if Transform_data:
                        if ColCategories:
                            le = preprocessing.LabelEncoder()
                            dfT[ColCategories] = dfT[ColCategories].apply(
                                le.fit_transform)
                            X_new = dfT[columsSelectX]
                            # st.write(X_new)

                    if Stan_Scaler:
                        if Select_Type == "Standard Scaler":
                            sc = StandardScaler()
                        elif Select_Type == "MinMax Scaler":
                            sc = MinMaxScaler()
                        elif Select_Type == "MaxAbs Scaler":
                            sc = MaxAbsScaler()
                        elif Select_Type == "Robust Scaler":
                            sc = RobustScaler()
                        elif Select_Type == "Power Trasformer":
                            sc = PowerTrasformer()
                        elif Select_Type == "Normalizer":
                            sc = Normalizer()
                        # X_new = dfT2.iloc[:, :-1].dropna()
                        X_new = sc.fit_transform(X)

                        # st.write(X_new)

                    y_target = dfT[columSelectY].name
                    y_new = classifier.predict(X_new)
                    dfT_new = pd.DataFrame(X_new)
                    dfT_new[y_target] = y_new  # Agregar la columna

                    st.subheader("""
                                        **Prediction**
                                        """)

                    pca2 = PCA(2)
                    X_projected = pca2.fit_transform(X_new)

                    x1_new = X_projected[:, 0]
                    x2_new = X_projected[:, 1]

                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    plt.figure()
                    plt.scatter(x1_new, x2_new, c=y_new,
                                alpha=0.8, cmap="viridis")
                    plt.xlabel("Principal Component X1")
                    plt.ylabel("Principal Component X2")
                    plt.title(ClassifierSelectModel + " - Prediction")
                    plt.colorbar()
                    st.pyplot()

                    dfTChart = dfT[columsSelectX]
                    dfTChart[columSelectY] = dfT[columSelectY]
                    dfTChart['prediction'] = y_new
                    st.write(dfTChart)

                    # st.info(classification_report(y_test, y_pred))
                    # st.info(precision_)

                else:
                    st.info('Awaiting for CSV file to be uploaded.')

                if st.button('Re-Run'):
                    caching.clear_cache()
                    st._RerunException

        else:
            st.info('Awaiting for CSV file to be uploaded.')

    except:
        st.info('ERROR - Please check your Dataset, parameters o selected model')
    return
