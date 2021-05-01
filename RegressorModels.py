import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PowerTransformer
#from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import Normalizer

from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import TweedieRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from ngboost import NGBRegressor

# Next Models
"""
#from sklearn.cross_decomposition import PLSRegression #Review
#from sklearn.cross_decomposition import PLSCanonical  #Review
"""

from scipy.stats import pearsonr
from sklearn import preprocessing
from random import randint
from streamlit import caching

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from streamlit import caching


from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

from os import path

# Para verificar
from matplotlib.lines import Line2D
from scipy import stats


# from streamlit.ScriptRunner import StopException, RerunException


def simple_scatter_plot(x_data, y_data, output_filename, title_name, x_axis_label, y_axis_label):

    figure, ax = plt.subplots()
    sns.set(color_codes=True)
    plt.figure(1, figsize=(9, 6))
    plt.title(title_name)
    ax = sns.scatterplot(x=x_data, y=y_data)
    ax.set(xlabel=x_axis_label, ylabel=y_axis_label)
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    # plt.close()
    st.pyplot(figure)


def RegressorModels(RegressorSelectModel):
    try:
        parameter_test_size = st.sidebar.slider(
            "Test size (fraction)", 0.02, 0.80, 0.2)
        st.sidebar.write("Test size: ", parameter_test_size)

        if RegressorSelectModel == "Multiple Variable Regression":
            parameter_n_estimators = st.sidebar.slider(
                "n estimators", 1, 1000, 100)
            st.sidebar.write("n  estimators: ", parameter_n_estimators)

        if RegressorSelectModel == "Support Vector Machines Regressor":
            param_C = st.sidebar.slider("C parameter", 0.01, 10.0, 1.0)
            st.sidebar.write("C parameter: ", param_C)
            kernelType = st.sidebar.selectbox("Kernel type",
                                              ("Linear", "Polynomial", "Gaussian", "Sigmoid"))
            if kernelType == "Polynomial":
                degreePoly = st.sidebar.slider("Polynomial degree", 1, 10, 2)
                st.sidebar.write("Polynomial degree: ", degreePoly)

        if RegressorSelectModel == "K-Nearest Neighbors Regressor":
            K_parameter = st.sidebar.slider("K parameter", 1, 20, 5)
            st.sidebar.write("K parameter: ", K_parameter)

        if RegressorSelectModel == "Gradient Boosting Regressor":
            n_estimators = st.sidebar.slider("n estimators", 1, 1000, 100, 1)
            st.sidebar.write("n estimators: ",  n_estimators)

        if RegressorSelectModel == "Extreme Gradient Boosting Regressor":
            n_estimators = st.sidebar.slider("n estimators", 1, 1000, 100)
            st.sidebar.write("n estimators: ",  n_estimators)

        if RegressorSelectModel == "Stochastic Gradient Descent Regressor":
            max_iters = st.sidebar.slider(
                "Maximum number of iterations", 800, 5000, 1000, 100)
            st.sidebar.write("Maximum number of iterations: ",  max_iters)

        if RegressorSelectModel == "Bayesian Ridge Regressor":
            max_iters = st.sidebar.slider(
                "Maximum number of iterations", 100, 5000, 300, 100)
            st.sidebar.write("Maximum number of iterations: ",  max_iters)

        if RegressorSelectModel == "Multiple Linear Regressor":
            fit_intercept = st.sidebar.selectbox(
                "Fit intercept", ("True", "False"))

        st.sidebar.info("""
                        [More information](http://gonzalezmaw.pythonanywhere.com/)
                        """)

        uploaded_file = st.file_uploader("Choose a CSV file")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
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

                Stan_Scaler = st.checkbox(
                    'Data Scaler')
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
                    # Selecting the last column as Y
                    y = df[columSelectY]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=parameter_test_size, random_state=0)

                # Model-1: Support Vector Machine Regressor (SVR)
                if RegressorSelectModel == "Support Vector Machines Regressor":
                    if kernelType == "Linear":
                        kernelType1 = "linear"
                    elif kernelType == "Polynomial":
                        kernelType1 = "poly"
                    elif kernelType == "Gaussian":
                        kernelType1 = "rbf"
                    elif kernelType == "Sigmoid":
                        kernelType1 = "sigmoid"

                    if kernelType1 == "poly":
                        # st.write(degreePoly)
                        regressor = SVR(
                            C=param_C, kernel=kernelType1, degree=degreePoly)
                    else:
                        # st.write(degreePoly)
                        regressor = SVR(
                            C=param_C, kernel=kernelType1)

                # Model-2: Linear Regression
                elif RegressorSelectModel == "Multiple Linear Regressor":
                    regressor = LinearRegression(
                        fit_intercept=fit_intercept, n_jobs=-1)

                # Model-3: Bayesian Ridge Regressor
                elif RegressorSelectModel == "Bayesian Ridge Regressor":
                    regressor = BayesianRidge(n_iter=max_iters)

                # Model-4: Decision Tree Regressor
                elif RegressorSelectModel == "Decision Tree Regressor":
                    regressor = DecisionTreeRegressor(random_state=0)

                # Model-5: Random Forest Regressor
                elif RegressorSelectModel == "Multiple Variable Regression":
                    regressor = RandomForestRegressor(
                        n_estimators=parameter_n_estimators, random_state=0, n_jobs=-1)

                # Model-6: K-Nerest Neighbors Regressor
                elif RegressorSelectModel == "K-Nearest Neighbors Regressor":
                    regressor = KNeighborsRegressor(
                        n_neighbors=K_parameter, n_jobs=-1)

                # Model-7: Gradient Boosting Regressor
                elif RegressorSelectModel == "Gradient Boosting Regressor":
                    regressor = GradientBoostingRegressor(
                        n_estimators=n_estimators, random_state=0)

                # Model-8: Extreme Gradient Boosting Regressor
                elif RegressorSelectModel == "Extreme Gradient Boosting Regressor":
                    regressor = XGBRegressor(
                        n_estimators=n_estimators, n_jobs=-1, random_state=0)

                # Model-9: Gaussian Process Regressor
                elif RegressorSelectModel == "Gaussian Process Regressor":
                    kernel = DotProduct() + WhiteKernel()
                    regressor = GaussianProcessRegressor(
                        kernel=kernel, random_state=0)

                  # Model-10: Stochastic Gradient Descent Regressor
                elif RegressorSelectModel == "Stochastic Gradient Descent Regressor":
                    regressor = SGDRegressor(
                        max_iter=max_iters, random_state=0)

                # Model-11: Light Gradient Boosting Machine Regressor
                elif RegressorSelectModel == "Light Gradient Boosting Machine Regressor":
                    regressor = LGBMRegressor(random_state=0)

                # Model-12: CatBoost Regressor
                elif RegressorSelectModel == "CatBoost Regressor":
                    regressor = CatBoostRegressor(random_state=0)

                # Model-13: AdaBoost Regressor
                elif RegressorSelectModel == "AdaBoost Regressor":
                    regressor = AdaBoostRegressor(random_state=0)

                # MOdel-14: Extra Trees Regressor
                elif RegressorSelectModel == "Extra Trees Regressor":
                    regressor = ExtraTreeRegressor(random_state=0)

                # Model-15: Bagging Regressor
                elif RegressorSelectModel == "Bagging Regressor":
                    regressor = BaggingRegressor(random_state=0)

                # Model-16: Passive Aggressive Regressor
                elif RegressorSelectModel == "Passive Aggressive Regressor":
                    regressor = PassiveAggressiveRegressor(random_state=0)

                # Model-17: Elastic Net Regressor
                elif RegressorSelectModel == "Elastic Net Regressor":
                    regressor = ElasticNet(random_state=0)

                # Model-18: Lasso Regressor
                elif RegressorSelectModel == "Lasso Regressor":
                    regressor = Lasso(random_state=0)

                # Model-19: Ridge Regressor
                elif RegressorSelectModel == "Ridge Regressor":
                    regressor = Ridge(random_state=0)

                # Model-20: Huber Regressor
                elif RegressorSelectModel == "Huber Regressor":
                    regressor = HuberRegressor()

                # Model-21: Kernel Ridge Regressor
                elif RegressorSelectModel == "Kernel Ridge Regressor":
                    regressor = KernelRidge()

                # Model-22: Tweedie Regressor
                elif RegressorSelectModel == "Tweedie Regressor":
                    regressor = TweedieRegressor()

                # Model-23: TheilSen Regressor
                elif RegressorSelectModel == "TheilSen Regressor":
                    regressor = TheilSenRegressor(random_state=0)

                # Model-24: Orthogonal Matching Pursuit Regressor
                elif RegressorSelectModel == "Orthogonal Matching Pursuit Regressor":
                    regressor = OrthogonalMatchingPursuit()

                # Model-25: Histogram Gradient Boosting Regressor
                elif RegressorSelectModel == "Histogram Gradient Boosting Regressor":
                    regressor = HistGradientBoostingRegressor(random_state=0)

                # Model-26: Least Angle Regressor
                elif RegressorSelectModel == "Least Angle Regressor":
                    regressor = Lars(random_state=0)

                # Model-27: Lasso Least Angle Regressor
                elif RegressorSelectModel == "Lasso Least Angle Regressor":
                    regressor = LassoLars(random_state=0)

                # Model-28: Automatic Relevance Determination
                elif RegressorSelectModel == "Automatic Relevance Determination Regressor":
                    regressor = ARDRegression()

                # Model-29: Random Sample Consensus
                elif RegressorSelectModel == "Random Sample Consensus Regressor":
                    regressor = RANSACRegressor(random_state=0)

                # Model-30: Perceptron
                elif RegressorSelectModel == "Perceptron Regressor":
                    regressor = Perceptron(random_state=0)

                # Model-31: Natural Gradient Boosting
                elif RegressorSelectModel == "Natural Gradient Boosting Regressor":
                    regressor = NGBRegressor(random_state=0)

                regressor.fit(X_train, y_train)
                y_pred = regressor.predict(X_test)
                coefR2 = round(r2_score(y_test, y_pred), 4)

                st.subheader("""
                            **Regression**
                            """)

                col1, col2 = st.beta_columns(2)

                with col1:
                    correlation = round(pearsonr(y_pred, y_test)[0], 4)
                    st.write("Pearson correlation coefficient:")
                    st.info(correlation)

                with col2:
                    R2_score = coefR2
                    st.write("Regression coefficient:")
                    st.info(R2_score)

                output_filename = "rf_regression.png"
                title_name = "Real Values vs Predicted Values - correlation ({})".format(
                    correlation)
                x_axis_label = "Real Values"
                y_axis_label = "Predicted Values"

                # plot data

                simple_scatter_plot(y_test, y_pred,
                                    output_filename, title_name, x_axis_label, y_axis_label)

                showBloxplot = st.checkbox('Show Boxplot')
                numFiles = math.ceil(df.shape[1]/5)

                if showBloxplot:
                    fig, axs = plt.subplots(
                        ncols=5, nrows=numFiles, figsize=(20, 10))
                    index = 0
                    axs = axs.flatten()
                    for k, v in df.items():
                        # sns.boxplot(y=k, data=df, ax=axs[index], palette="Paired")
                        sns.boxplot(
                            y=k, data=df, ax=axs[index], palette="Set3")
                        index += 1
                    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
                    st.pyplot(fig)

                showOutPerc = st.checkbox('Show Outliers Percentage')
                if showOutPerc:
                    for k, v in df.items():
                        q1 = v.quantile(0.25)
                        q3 = v.quantile(0.75)
                        irq = q3 - q1
                        v_col = v[(v <= q1 - 1.5 * irq) |
                                  (v >= q3 + 1.5 * irq)]
                        perc = np.shape(v_col)[0] * 100.0 / np.shape(df)[0]
                        st.info("Column %s outliers = %.2f%%" % (k, perc))

                    # Columns like CRIM, ZN, RM, B seems to have outliers. Let's see the outliers percentage in every column.

                ymax = np.amax(y)
                # Let's remove MEDV outliers (MEDV = 50.0) before plotting more distributions
                df = df[~(df[y.name] >= ymax)]

                numFilesX = math.ceil(X.shape[1]/5)

                showHistograms = st.checkbox('Show Histograms')
                if showHistograms:
                    fig2, axs = plt.subplots(
                        ncols=5, nrows=numFiles, figsize=(20, 10))
                    index = 0
                    axs = axs.flatten()
                    for k, v in df.items():
                        sns.distplot(v, ax=axs[index], color="dodgerblue")
                        # sns.histplot(v, ax=axs[index], color="dodgerblue")
                        index += 1
                    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
                    st.pyplot(fig2)

                # fig, ax = plt.subplots()
                st.set_option('deprecation.showPyplotGlobalUse', False)
                showMapHeat = st.checkbox('Show Matrix Correlations')

                if showMapHeat:
                    plt.figure(figsize=(20, 10))
                    sns.heatmap(df.corr().abs(),  annot=True, cmap="Blues")
                    st.pyplot()

                showCorrChart = st.checkbox('Show Correlation Charts')
                if showCorrChart:
                    X = df[columsSelectX]
                    y = df[columSelectY]
                    # Let's scale the columns before plotting them against MEDV
                    min_max_scaler = preprocessing.MinMaxScaler()
                    column_sels = list(X.columns)
                    # st.info(column_sels)
                    # x = df.loc[:, column_sels]
                    # y = df[y.name]
                    X = pd.DataFrame(data=min_max_scaler.fit_transform(X),

                                     columns=column_sels)
                    fig, axs = plt.subplots(
                        ncols=5, nrows=numFilesX, figsize=(20, 10))
                    index = 0
                    axs = axs.flatten()
                    colors = "bgrcmyk"
                    # color_index = 0
                    for i, k in enumerate(column_sels):
                        # sns.regplot(y=y, x=X[k], ax=axs[i], color="blue")
                        sns.regplot(y=y, x=X[k], ax=axs[i],
                                    color=colors[randint(0, 6)])
                        # color_index += 1
                    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
                    st.pyplot(fig)

                st.subheader("""
                                **Prediction**
                                """)
                uploaded_file_target = st.file_uploader(
                    "Choose a new CSV file for prediction")
                if uploaded_file_target is not None:
                    dfT = pd.read_csv(uploaded_file_target)
                    dfT = dfT.interpolate(method='linear', axis=0).bfill()
                    dfT = dfT.dropna()
                    dfT = dfT.drop_duplicates()

                    # st.write(uploaded_file_target.name)
                    #routefile = path.abspath(uploaded_file_target.name)
                    # st.write(routefile)
                    # st.write(dfT)
                    # Using all column except for the last column as X

                    X = dfT[columsSelectX]
                    y = dfT[columSelectY]
                    X_new = dfT[columsSelectX]
                    dfT_new = pd.DataFrame(X_new)
                    # st.write(dfT)

                    if Stan_Scaler:
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
                        # X_new = dfT2.iloc[:, :-1].dropna()
                        X_new = sc.fit_transform(X)
                        # st.write(X_new)

                    y_new = regressor.predict(X_new)
                    y_target = dfT[columSelectY].name
                    y_target = dfT[columSelectY].name+"_pred"

                    dfT_new[y_target] = y_new  # Agregar la columna

                    # st.write(dfT_new)

                    col1, col2, col3 = st.beta_columns(3)

                    with col1:
                        error = round(mean_absolute_error(y, y_new), 4)
                        st.write("Mean absolute error:")
                        st.success(error)

                    with col2:
                        pearson = round(pearsonr(y, y_new)[0], 4)
                        st.write("Pearson correlation:")
                        st.success(pearson)

                    with col3:
                        r2 = round(r2_score(y, y_new), 4)
                        st.write("Regression coefficient:")
                        st.success(r2)
                    axs = sns.jointplot(
                        # x=dfT.iloc[:, -1], y=dfT_new.iloc[:, -1], kind="reg", color="deepskyblue")
                        # x=dfT.iloc[:, -1], y=dfT_new.iloc[:, -1], kind="reg", color="royalblue")
                        x=dfT[columSelectY], y=dfT_new[y_target], kind="reg", color="dodgerblue")
                    st.pyplot(axs)

                    dfT['prediction'] = y_new
                    st.write(dfT)

                    if st.button('Export Data'):
                        dfT_new.reset_index().to_csv('ExportData.csv', header=True, index=False)

                if st.button('Re-Run'):
                    caching.clear_cache()
                    st._RerunException

        else:
            st.info('Awaiting for CSV file to be uploaded.')

    except:
        st.info('ERROR - Please check your Dataset, parameters o selected model')
    return
