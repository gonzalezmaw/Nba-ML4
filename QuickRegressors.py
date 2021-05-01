import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from streamlit import caching
#from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_log_error
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_squared_log_error

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
# https://stanfordmlgroup.github.io/ngboost/1-useage.html

"""
The installation of the NGBoost library uninstalls the latest version of scikit-learn-0.24.0 and 
installs the version of scikit-learn-0.23.2, this apparently brings problems with the arguments 
of the Logistic Regression and Gaussian Process Classifier models, but nothing that cannot be solved
(delet n_jobs=-1). Also brings problem with Tweedie Regressor, Huber Regressor, perceptron
"""

#from sklearn.neural_network import MLPRegressor
#from sklearn.isotonic import IsotonicRegression

from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
import time

# Next Models
"""
# from sklearn.cross_decomposition import PLSRegression #Review
# from sklearn.cross_decomposition import PLSCanonical  #Review
"""


# Versions Scifi-learn < 0.24.0 is necesary this function of mean_absolute_percentage_error
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


# from streamlit.ScriptRunner import StopException, RerunException


def QuickRegressors():

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

                    # Support Vector Machine Regressor (SVR)
                    regressor = SVR(kernel="linear")
                    start_time_1 = time.time()
                    regressor.fit(X_train, y_train)
                    y_pred = regressor.predict(X_test)
                    R2_1 = round(r2_score(y_test, y_pred), 3)
                    time_1 = round(time.time() - start_time_1, 4)
                    mod_1 = "Support Vector Machine"
                    MAE_1 = round(mean_absolute_error(y_test, y_pred), 2)
                    MSE_1 = round(mean_squared_error(y_test, y_pred), 2)
                    MAPE_1 = round(mean_absolute_percentage_error(
                        y_test, y_pred)*100, 2)
                    #MSSE_1 = np.sqrt(mean_squared_error(y_test, y_pred,))
                    # st.write(MSSE_1)
                    RMSE_1 = round(mean_squared_error(
                        y_test, y_pred, squared=False), 2)
                    try:
                        RMSLE_1 = round(
                            np.sqrt(mean_squared_log_error(y_test, y_pred)), 4)
                    except:
                        RMSLE_1 = "error"
                    # st.write(RMSLE_1)

                    # Linear Regression
                    regressor = LinearRegression(n_jobs=-1)
                    start_time_2 = time.time()
                    regressor.fit(X_train, y_train)
                    y_pred = regressor.predict(X_test)
                    R2_2 = round(r2_score(y_test, y_pred), 3)
                    time_2 = round(time.time() - start_time_2, 4)
                    mod_2 = "Linear Regression"
                    MAE_2 = round(mean_absolute_error(y_test, y_pred), 2)
                    MSE_2 = round(mean_squared_error(y_test, y_pred), 2)
                    MAPE_2 = round(
                        mean_absolute_percentage_error(y_test, y_pred), 2)
                    MAPE_2 = round(mean_absolute_percentage_error(
                        y_test, y_pred)*100, 2)
                    RMSE_2 = round(mean_squared_error(
                        y_test, y_pred, squared=False), 2)
                    try:
                        RMSLE_2 = round(
                            np.sqrt(mean_squared_log_error(y_test, y_pred)), 4)
                    except:
                        RMSLE_2 = "error"

                    # Bayesian Ridge Regressor
                    regressor = BayesianRidge()
                    start_time_3 = time.time()
                    regressor.fit(X_train, y_train)
                    y_pred = regressor.predict(X_test)
                    R2_3 = round(r2_score(y_test, y_pred), 3)
                    time_3 = round(time.time() - start_time_3, 4)
                    mod_3 = "Bayesian Ridge"
                    MAE_3 = round(mean_absolute_error(y_test, y_pred), 2)
                    MSE_3 = round(mean_squared_error(y_test, y_pred), 2)
                    MAPE_3 = round(mean_absolute_percentage_error(
                        y_test, y_pred)*100, 2)
                    RMSE_3 = round(mean_squared_error(
                        y_test, y_pred, squared=False), 2)
                    try:
                        RMSLE_3 = round(
                            np.sqrt(mean_squared_log_error(y_test, y_pred)), 4)
                    except:
                        RMSLE_3 = "error"

                    # Decision Tree Regressor
                    regressor = DecisionTreeRegressor(random_state=0)
                    start_time_4 = time.time()
                    regressor.fit(X_train, y_train)
                    y_pred = regressor.predict(X_test)
                    R2_4 = round(r2_score(y_test, y_pred), 3)
                    time_4 = round(time.time() - start_time_4, 4)
                    mod_4 = "Decision Tree"
                    MAE_4 = round(mean_absolute_error(y_test, y_pred), 2)
                    MSE_4 = round(mean_squared_error(y_test, y_pred), 2)
                    MAPE_4 = round(mean_absolute_percentage_error(
                        y_test, y_pred)*100, 2)
                    RMSE_4 = round(mean_squared_error(
                        y_test, y_pred, squared=False), 2)
                    try:
                        RMSLE_4 = round(
                            np.sqrt(mean_squared_log_error(y_test, y_pred)), 4)
                    except:
                        RMSLE_4 = "error"

                    # Random Forest Regressor
                    regressor = RandomForestRegressor(
                        n_estimators=100, random_state=0, n_jobs=-1)
                    # regressor = RandomForestRegressor(
                    # n_estimators=100, random_state=0)
                    start_time_5 = time.time()
                    regressor.fit(X_train, y_train)
                    y_pred = regressor.predict(X_test)
                    R2_5 = round(r2_score(y_test, y_pred), 3)
                    time_5 = round(time.time() - start_time_5, 4)
                    mod_5 = "Random Forest"
                    MAE_5 = round(mean_absolute_error(y_test, y_pred), 2)
                    MSE_5 = round(mean_squared_error(y_test, y_pred), 2)
                    MAPE_5 = round(mean_absolute_percentage_error(
                        y_test, y_pred)*100, 2)
                    RMSE_5 = round(mean_squared_error(
                        y_test, y_pred, squared=False), 2)
                    try:
                        RMSLE_5 = round(
                            np.sqrt(mean_squared_log_error(y_test, y_pred)), 4)
                    except:
                        RMSLE_5 = "error"

                    # K Nearest Neighbors Regressor
                    regressor = KNeighborsRegressor(n_jobs=-1)
                    start_time_6 = time.time()
                    regressor.fit(X_train, y_train)
                    y_pred = regressor.predict(X_test)
                    R2_6 = round(r2_score(y_test, y_pred), 3)
                    time_6 = round(time.time() - start_time_6, 4)
                    mod_6 = "K Nearest Neighbors"
                    MAE_6 = round(mean_absolute_error(y_test, y_pred), 2)
                    MSE_6 = round(mean_squared_error(y_test, y_pred), 2)
                    MAPE_6 = round(mean_absolute_percentage_error(
                        y_test, y_pred)*100, 2)
                    RMSE_6 = round(mean_squared_error(
                        y_test, y_pred, squared=False), 2)
                    try:
                        RMSLE_6 = round(
                            np.sqrt(mean_squared_log_error(y_test, y_pred)), 4)
                    except:
                        RMSLE_6 = "error"

                    # Gradient Boosting Regressor
                    regressor = GradientBoostingRegressor(random_state=0)
                    start_time_7 = time.time()
                    regressor.fit(X_train, y_train)
                    y_pred = regressor.predict(X_test)
                    R2_7 = round(r2_score(y_test, y_pred), 3)
                    time_7 = round(time.time() - start_time_7, 4)
                    mod_7 = "Gradient Boosting"
                    MAE_7 = round(mean_absolute_error(y_test, y_pred), 2)
                    MSE_7 = round(mean_squared_error(y_test, y_pred), 2)
                    MAPE_7 = round(mean_absolute_percentage_error(
                        y_test, y_pred)*100, 2)
                    RMSE_7 = round(mean_squared_error(
                        y_test, y_pred, squared=False), 2)
                    try:
                        RMSLE_7 = round(
                            np.sqrt(mean_squared_log_error(y_test, y_pred)), 4)
                    except:
                        RMSLE_7 = "error"

                    # Extreme Gradient Boosting Regressor
                    regressor = XGBRegressor(n_jobs=-1, random_state=0)
                    start_time_8 = time.time()
                    regressor.fit(X_train, y_train)
                    y_pred = regressor.predict(X_test)
                    R2_8 = round(r2_score(y_test, y_pred), 3)
                    time_8 = round(time.time() - start_time_8, 4)
                    mod_8 = "Extreme Gradient Boosting"
                    MAE_8 = round(mean_absolute_error(y_test, y_pred), 2)
                    MSE_8 = round(mean_squared_error(y_test, y_pred), 2)
                    MAPE_8 = round(mean_absolute_percentage_error(
                        y_test, y_pred)*100, 2)
                    RMSE_8 = round(mean_squared_error(
                        y_test, y_pred, squared=False), 2)
                    try:
                        RMSLE_8 = round(
                            np.sqrt(mean_squared_log_error(y_test, y_pred)), 4)
                    except:
                        RMSLE_8 = "error"

                    # Gaussian Process Regressor
                    kernel = DotProduct() + WhiteKernel()
                    regressor = GaussianProcessRegressor(
                        kernel=kernel, random_state=0)
                    start_time_9 = time.time()
                    regressor.fit(X_train, y_train)
                    y_pred = regressor.predict(X_test)
                    R2_9 = round(r2_score(y_test, y_pred), 3)
                    time_9 = round(time.time() - start_time_9, 4)
                    mod_9 = "Gaussian Process"
                    MAE_9 = round(mean_absolute_error(y_test, y_pred), 2)
                    MSE_9 = round(mean_squared_error(y_test, y_pred), 2)
                    MAPE_9 = round(mean_absolute_percentage_error(
                        y_test, y_pred)*100, 2)
                    RMSE_9 = round(mean_squared_error(
                        y_test, y_pred, squared=False), 2)
                    try:
                        RMSLE_9 = round(
                            np.sqrt(mean_squared_log_error(y_test, y_pred)), 4)
                    except:
                        RMSLE_9 = "error"

                    # Stochastic Gradient Descent Regressor
                    regressor = SGDRegressor(random_state=0)
                    start_time_10 = time.time()
                    regressor.fit(X_train, y_train)
                    y_pred = regressor.predict(X_test)
                    R2_10 = round(r2_score(y_test, y_pred), 3)
                    time_10 = round(time.time() - start_time_10, 4)
                    mod_10 = "Stochastic Gradient Descent"
                    MAE_10 = round(mean_absolute_error(y_test, y_pred), 2)
                    MSE_10 = round(mean_squared_error(y_test, y_pred), 2)
                    MAPE_10 = round(mean_absolute_percentage_error(
                        y_test, y_pred)*100, 2)
                    RMSE_10 = round(mean_squared_error(
                        y_test, y_pred, squared=False), 2)
                    try:
                        RMSLE_10 = round(
                            np.sqrt(mean_squared_log_error(y_test, y_pred)), 4)
                    except:
                        RMSLE_10 = "error"

                    # Light Gradient Boosting Machine Regressor
                    regressor = LGBMRegressor(random_state=0)
                    start_time_11 = time.time()
                    regressor.fit(X_train, y_train)
                    y_pred = regressor.predict(X_test)
                    R2_11 = round(r2_score(y_test, y_pred), 3)
                    time_11 = round(time.time() - start_time_11, 4)
                    mod_11 = "Light Gradient Boosting Machine"
                    MAE_11 = round(mean_absolute_error(y_test, y_pred), 2)
                    MSE_11 = round(mean_squared_error(y_test, y_pred), 2)
                    MAPE_11 = round(mean_absolute_percentage_error(
                        y_test, y_pred)*100, 2)
                    RMSE_11 = round(mean_squared_error(
                        y_test, y_pred, squared=False), 2)
                    try:
                        RMSLE_11 = round(
                            np.sqrt(mean_squared_log_error(y_test, y_pred)), 4)
                    except:
                        RMSLE_11 = "error"

                    # CatBoost Regressor
                    regressor = CatBoostRegressor(random_state=0)
                    start_time_12 = time.time()
                    regressor.fit(X_train, y_train)
                    y_pred = regressor.predict(X_test)
                    R2_12 = round(r2_score(y_test, y_pred), 3)
                    time_12 = round(time.time() - start_time_12, 4)
                    mod_12 = "CatBoost"
                    MAE_12 = round(mean_absolute_error(y_test, y_pred), 2)
                    MSE_12 = round(mean_squared_error(y_test, y_pred), 2)
                    MAPE_12 = round(mean_absolute_percentage_error(
                        y_test, y_pred)*100, 2)
                    RMSE_12 = round(mean_squared_error(
                        y_test, y_pred, squared=False), 2)
                    try:
                        RMSLE_12 = round(
                            np.sqrt(mean_squared_log_error(y_test, y_pred)), 4)
                    except:
                        RMSLE_12 = "error"

                    # AdaBoost Regressor
                    regressor = AdaBoostRegressor(random_state=0)
                    start_time_13 = time.time()
                    regressor.fit(X_train, y_train)
                    y_pred = regressor.predict(X_test)
                    R2_13 = round(r2_score(y_test, y_pred), 3)
                    time_13 = round(time.time() - start_time_13, 4)
                    mod_13 = "AdaBoost"
                    MAE_13 = round(mean_absolute_error(y_test, y_pred), 2)
                    MSE_13 = round(mean_squared_error(y_test, y_pred), 2)
                    MAPE_13 = round(mean_absolute_percentage_error(
                        y_test, y_pred)*100, 2)
                    RMSE_13 = round(mean_squared_error(
                        y_test, y_pred, squared=False), 2)
                    try:
                        RMSLE_13 = round(
                            np.sqrt(mean_squared_log_error(y_test, y_pred)), 4)
                    except:
                        RMSLE_13 = "error"

                    # Extra Trees Regressor
                    regressor = ExtraTreeRegressor(random_state=0)
                    start_time_14 = time.time()
                    regressor.fit(X_train, y_train)
                    y_pred = regressor.predict(X_test)
                    R2_14 = round(r2_score(y_test, y_pred), 3)
                    time_14 = round(time.time() - start_time_14, 4)
                    mod_14 = "Extra Trees"
                    MAE_14 = round(mean_absolute_error(y_test, y_pred), 2)
                    MSE_14 = round(mean_squared_error(y_test, y_pred), 2)
                    MAPE_14 = round(mean_absolute_percentage_error(
                        y_test, y_pred)*100, 2)
                    RMSE_14 = round(mean_squared_error(
                        y_test, y_pred, squared=False), 2)
                    try:
                        RMSLE_14 = round(
                            np.sqrt(mean_squared_log_error(y_test, y_pred)), 4)
                    except:
                        RMSLE_14 = "error"

                    # Bagging Regressor
                    #regressor = BaggingRegressor(random_state=0, n_jobs=-1)
                    regressor = BaggingRegressor(random_state=0)
                    start_time_15 = time.time()
                    regressor.fit(X_train, y_train)
                    y_pred = regressor.predict(X_test)
                    R2_15 = round(r2_score(y_test, y_pred), 3)
                    time_15 = round(time.time() - start_time_15, 4)
                    mod_15 = "Bagging"
                    MAE_15 = round(mean_absolute_error(y_test, y_pred), 2)
                    MSE_15 = round(mean_squared_error(y_test, y_pred), 2)
                    MAPE_15 = round(mean_absolute_percentage_error(
                        y_test, y_pred)*100, 2)
                    RMSE_15 = round(mean_squared_error(
                        y_test, y_pred, squared=False), 2)
                    try:
                        RMSLE_15 = round(
                            np.sqrt(mean_squared_log_error(y_test, y_pred)), 4)
                    except:
                        RMSLE_15 = "error"

                    # Passive Aggressive Regressor
                    regressor = PassiveAggressiveRegressor(random_state=0)
                    start_time_16 = time.time()
                    regressor.fit(X_train, y_train)
                    y_pred = regressor.predict(X_test)
                    R2_16 = round(r2_score(y_test, y_pred), 3)
                    time_16 = round(time.time() - start_time_16, 4)
                    mod_16 = "Passive Aggressive"
                    MAE_16 = round(mean_absolute_error(y_test, y_pred), 2)
                    MSE_16 = round(mean_squared_error(y_test, y_pred), 2)
                    MAPE_16 = round(mean_absolute_percentage_error(
                        y_test, y_pred)*100, 2)
                    RMSE_16 = round(mean_squared_error(
                        y_test, y_pred, squared=False), 2)
                    try:
                        RMSLE_16 = round(
                            np.sqrt(mean_squared_log_error(y_test, y_pred)), 4)
                    except:
                        RMSLE_16 = "error"

                    # Elastic Net Regressor
                    regressor = ElasticNet(random_state=0)
                    start_time_17 = time.time()
                    regressor.fit(X_train, y_train)
                    y_pred = regressor.predict(X_test)
                    R2_17 = round(r2_score(y_test, y_pred), 3)
                    time_17 = round(time.time() - start_time_17, 4)
                    mod_17 = "Elastic Net"
                    MAE_17 = round(mean_absolute_error(y_test, y_pred), 2)
                    MSE_17 = round(mean_squared_error(y_test, y_pred), 2)
                    MAPE_17 = round(mean_absolute_percentage_error(
                        y_test, y_pred)*100, 2)
                    RMSE_17 = round(mean_squared_error(
                        y_test, y_pred, squared=False), 2)
                    try:
                        RMSLE_17 = round(
                            np.sqrt(mean_squared_log_error(y_test, y_pred)), 4)
                    except:
                        RMSLE_17 = "error"

                    # Lasso
                    regressor = Lasso(random_state=0)
                    start_time_18 = time.time()
                    regressor.fit(X_train, y_train)
                    y_pred = regressor.predict(X_test)
                    R2_18 = round(r2_score(y_test, y_pred), 3)
                    time_18 = round(time.time() - start_time_18, 4)
                    mod_18 = "Lasso Regression"
                    MAE_18 = round(mean_absolute_error(y_test, y_pred), 2)
                    MSE_18 = round(mean_squared_error(y_test, y_pred), 2)
                    MAPE_18 = round(mean_absolute_percentage_error(
                        y_test, y_pred)*100, 2)
                    RMSE_18 = round(mean_squared_error(
                        y_test, y_pred, squared=False), 2)
                    try:
                        RMSLE_18 = round(
                            np.sqrt(mean_squared_log_error(y_test, y_pred)), 4)
                    except:
                        RMSLE_18 = "error"

                    # Ridge
                    regressor = Ridge(random_state=0)
                    start_time_19 = time.time()
                    regressor.fit(X_train, y_train)
                    y_pred = regressor.predict(X_test)
                    R2_19 = round(r2_score(y_test, y_pred), 3)
                    time_19 = round(time.time() - start_time_19, 4)
                    mod_19 = "Ridge Regression"
                    MAE_19 = round(mean_absolute_error(y_test, y_pred), 2)
                    MSE_19 = round(mean_squared_error(y_test, y_pred), 2)
                    MAPE_19 = round(mean_absolute_percentage_error(
                        y_test, y_pred)*100, 2)
                    RMSE_19 = round(mean_squared_error(
                        y_test, y_pred, squared=False), 2)
                    try:
                        RMSLE_19 = round(
                            np.sqrt(mean_squared_log_error(y_test, y_pred)), 4)
                    except:
                        RMSLE_19 = "error"

                    # HuberRegressor
                    try:
                        regressor = HuberRegressor()
                        start_time_20 = time.time()
                        regressor.fit(X_train, y_train)
                    except:
                        regressor = LinearRegression(n_jobs=-1)
                        start_time_20 = time.time()
                        regressor.fit(X_train, y_train)

                    y_pred = regressor.predict(X_test)
                    R2_20 = round(r2_score(y_test, y_pred), 3)
                    time_20 = round(time.time() - start_time_20, 4)
                    mod_20 = "Huber"
                    MAE_20 = round(mean_absolute_error(y_test, y_pred), 2)
                    MSE_20 = round(mean_squared_error(y_test, y_pred), 2)
                    MAPE_20 = round(mean_absolute_percentage_error(
                        y_test, y_pred)*100, 2)
                    RMSE_20 = round(mean_squared_error(
                        y_test, y_pred, squared=False), 2)
                    try:
                        RMSLE_20 = round(
                            np.sqrt(mean_squared_log_error(y_test, y_pred)), 4)
                    except:
                        RMSLE_20 = "error"

                    # KernelRidge
                    regressor = KernelRidge()
                    start_time_21 = time.time()
                    regressor.fit(X_train, y_train)
                    y_pred = regressor.predict(X_test)
                    R2_21 = round(r2_score(y_test, y_pred), 3)
                    time_21 = round(time.time() - start_time_21, 4)
                    mod_21 = "Kernel Ridge"
                    MAE_21 = round(mean_absolute_error(y_test, y_pred), 2)
                    MSE_21 = round(mean_squared_error(y_test, y_pred), 2)
                    MAPE_21 = round(mean_absolute_percentage_error(
                        y_test, y_pred)*100, 2)
                    RMSE_21 = round(mean_squared_error(
                        y_test, y_pred, squared=False), 2)
                    try:
                        RMSLE_21 = round(
                            np.sqrt(mean_squared_log_error(y_test, y_pred)), 4)
                    except:
                        RMSLE_21 = "error"

                    # Tweedie Regressor
                    try:
                        regressor = TweedieRegressor()
                        start_time_22 = time.time()
                        regressor.fit(X_train, y_train)

                    except:
                        regressor = LinearRegression(n_jobs=-1)
                        start_time_22 = time.time()
                        regressor.fit(X_train, y_train)

                    y_pred = regressor.predict(X_test)

                    R2_22 = round(r2_score(y_test, y_pred), 3)
                    time_22 = round(time.time() - start_time_22, 4)
                    mod_22 = "Tweedie"
                    MAE_22 = round(mean_absolute_error(y_test, y_pred), 2)
                    MSE_22 = round(mean_squared_error(y_test, y_pred), 2)
                    MAPE_22 = round(mean_absolute_percentage_error(
                        y_test, y_pred)*100, 2)
                    RMSE_22 = round(mean_squared_error(
                        y_test, y_pred, squared=False), 2)
                    try:
                        RMSLE_22 = round(
                            np.sqrt(mean_squared_log_error(y_test, y_pred)), 4)
                    except:
                        RMSLE_22 = "error"

                    # TheilSen Regressor
                    regressor = TheilSenRegressor(random_state=0)
                    #regressor = TheilSenRegressor(n_jobs=-1, random_state=0)
                    start_time_23 = time.time()
                    regressor.fit(X_train, y_train)
                    y_pred = regressor.predict(X_test)
                    R2_23 = round(r2_score(y_test, y_pred), 3)
                    time_23 = round(time.time() - start_time_23, 4)
                    mod_23 = "TheilSen"
                    MAE_23 = round(mean_absolute_error(y_test, y_pred), 2)
                    MSE_23 = round(mean_squared_error(y_test, y_pred), 2)
                    MAPE_23 = round(mean_absolute_percentage_error(
                        y_test, y_pred)*100, 2)
                    RMSE_23 = round(mean_squared_error(
                        y_test, y_pred, squared=False), 2)
                    try:
                        RMSLE_23 = round(
                            np.sqrt(mean_squared_log_error(y_test, y_pred)), 4)
                    except:
                        RMSLE_23 = "error"

                    # OrthogonalMatchingPursuit
                    regressor = OrthogonalMatchingPursuit()
                    start_time_24 = time.time()
                    regressor.fit(X_train, y_train)
                    y_pred = regressor.predict(X_test)
                    R2_24 = round(r2_score(y_test, y_pred), 3)
                    time_24 = round(time.time() - start_time_24, 4)
                    mod_24 = "Orthogonal Matching Pursuit"
                    MAE_24 = round(mean_absolute_error(y_test, y_pred), 2)
                    MSE_24 = round(mean_squared_error(y_test, y_pred), 2)
                    MAPE_24 = round(mean_absolute_percentage_error(
                        y_test, y_pred)*100, 2)
                    RMSE_24 = round(mean_squared_error(
                        y_test, y_pred, squared=False), 2)
                    try:
                        RMSLE_24 = round(
                            np.sqrt(mean_squared_log_error(y_test, y_pred)), 4)
                    except:
                        RMSLE_24 = "error"

                    # HistGradientBoostingRegressor
                    regressor = HistGradientBoostingRegressor(random_state=0)
                    start_time_25 = time.time()
                    regressor.fit(X_train, y_train)
                    y_pred = regressor.predict(X_test)
                    R2_25 = round(r2_score(y_test, y_pred), 3)
                    time_25 = round(time.time() - start_time_25, 4)
                    mod_25 = "Histogram Gradient Boosting"
                    MAE_25 = round(mean_absolute_error(y_test, y_pred), 2)
                    MSE_25 = round(mean_squared_error(y_test, y_pred), 2)
                    MAPE_25 = round(mean_absolute_percentage_error(
                        y_test, y_pred)*100, 2)
                    RMSE_25 = round(mean_squared_error(
                        y_test, y_pred, squared=False), 2)
                    try:
                        RMSLE_25 = round(
                            np.sqrt(mean_squared_log_error(y_test, y_pred)), 4)
                    except:
                        RMSLE_25 = "error"

                    # Least Angle Regression
                    regressor = Lars(random_state=0)
                    start_time_26 = time.time()
                    regressor.fit(X_train, y_train)
                    y_pred = regressor.predict(X_test)
                    R2_26 = round(r2_score(y_test, y_pred), 3)
                    time_26 = round(time.time() - start_time_26, 4)
                    mod_26 = "Least Angle Regression"
                    MAE_26 = round(mean_absolute_error(y_test, y_pred), 2)
                    MSE_26 = round(mean_squared_error(y_test, y_pred), 2)
                    MAPE_26 = round(mean_absolute_percentage_error(
                        y_test, y_pred)*100, 2)
                    RMSE_26 = round(mean_squared_error(
                        y_test, y_pred, squared=False), 2)
                    try:
                        RMSLE_26 = round(
                            np.sqrt(mean_squared_log_error(y_test, y_pred)), 4)
                    except:
                        RMSLE_26 = "error"

                    # Lasso Least Angle Regression
                    regressor = LassoLars(random_state=0)
                    start_time_27 = time.time()
                    regressor.fit(X_train, y_train)
                    y_pred = regressor.predict(X_test)
                    R2_27 = round(r2_score(y_test, y_pred), 3)
                    time_27 = round(time.time() - start_time_27, 4)
                    mod_27 = "Lasso Least Angle Regression"
                    MAE_27 = round(mean_absolute_error(y_test, y_pred), 2)
                    MSE_27 = round(mean_squared_error(y_test, y_pred), 2)
                    MAPE_27 = round(mean_absolute_percentage_error(
                        y_test, y_pred)*100, 2)
                    RMSE_27 = round(mean_squared_error(
                        y_test, y_pred, squared=False), 2)
                    try:
                        RMSLE_27 = round(
                            np.sqrt(mean_squared_log_error(y_test, y_pred)), 4)
                    except:
                        RMSLE_27 = "error"

                    # Automatic Relevance Determination
                    regressor = ARDRegression()
                    start_time_28 = time.time()
                    regressor.fit(X_train, y_train)
                    y_pred = regressor.predict(X_test)
                    R2_28 = round(r2_score(y_test, y_pred), 3)
                    time_28 = round(time.time() - start_time_28, 4)
                    mod_28 = "Automatic Relevance Determination"
                    MAE_28 = round(mean_absolute_error(y_test, y_pred), 2)
                    MSE_28 = round(mean_squared_error(y_test, y_pred), 2)
                    MAPE_28 = round(mean_absolute_percentage_error(
                        y_test, y_pred)*100, 2)
                    RMSE_28 = round(mean_squared_error(
                        y_test, y_pred, squared=False), 2)
                    try:
                        RMSLE_28 = round(
                            np.sqrt(mean_squared_log_error(y_test, y_pred)), 4)
                    except:
                        RMSLE_28 = "error"

                    # Random Sample Consensus
                    regressor = RANSACRegressor(random_state=0)
                    start_time_29 = time.time()
                    regressor.fit(X_train, y_train)
                    y_pred = regressor.predict(X_test)
                    R2_29 = round(r2_score(y_test, y_pred), 3)
                    time_29 = round(time.time() - start_time_29, 4)
                    mod_29 = "Random Sample Consensus"
                    MAE_29 = round(mean_absolute_error(y_test, y_pred), 2)
                    MSE_29 = round(mean_squared_error(y_test, y_pred), 2)
                    MAPE_29 = round(mean_absolute_percentage_error(
                        y_test, y_pred)*100, 2)
                    RMSE_29 = round(mean_squared_error(
                        y_test, y_pred, squared=False), 2)
                    try:
                        RMSLE_29 = round(
                            np.sqrt(mean_squared_log_error(y_test, y_pred)), 4)
                    except:
                        RMSLE_29 = "error"

                    # Perceptron
                    try:
                        regressor = Perceptron(random_state=0)
                        start_time_30 = time.time()
                        regressor.fit(X_train, y_train)
                    except:
                        regressor = LinearRegression(n_jobs=-1)
                        start_time_30 = time.time()
                        regressor.fit(X_train, y_train)

                    y_pred = regressor.predict(X_test)
                    R2_30 = round(r2_score(y_test, y_pred), 3)
                    time_30 = round(time.time() - start_time_30, 4)
                    mod_30 = "Perceptron"
                    MAE_30 = round(mean_absolute_error(y_test, y_pred), 2)
                    MSE_30 = round(mean_squared_error(y_test, y_pred), 2)
                    MAPE_30 = round(mean_absolute_percentage_error(
                        y_test, y_pred)*100, 2)
                    RMSE_30 = round(mean_squared_error(
                        y_test, y_pred, squared=False), 2)
                    try:
                        RMSLE_30 = round(
                            np.sqrt(mean_squared_log_error(y_test, y_pred)), 4)
                    except:
                        RMSLE_30 = "error"

                    # Natural Gradient Boosting
                    regressor = NGBRegressor(random_state=0)
                    start_time_31 = time.time()
                    regressor.fit(X_train, y_train)
                    y_pred = regressor.predict(X_test)
                    R2_31 = round(r2_score(y_test, y_pred), 3)
                    time_31 = round(time.time() - start_time_31, 4)
                    mod_31 = "Natural Gradient Boosting"
                    MAE_31 = round(mean_absolute_error(y_test, y_pred), 2)
                    MSE_31 = round(mean_squared_error(y_test, y_pred), 2)
                    MAPE_31 = round(mean_absolute_percentage_error(
                        y_test, y_pred)*100, 2)
                    RMSE_31 = round(mean_squared_error(
                        y_test, y_pred, squared=False), 2)
                    try:
                        RMSLE_31 = round(
                            np.sqrt(mean_squared_log_error(y_test, y_pred)), 4)
                    except:
                        RMSLE_31 = "error"

                    """
                    regressor = MLPRegressor(random_state=0)
                    start_time_30 = time.time()
                    regressor.fit(X_train, y_train)
                    y_pred = regressor.predict(X_test)
                    R2_30 = round(r2_score(y_test, y_pred), 3)
                    time_30 = round(time.time() - start_time_30, 4)
                    mod_30 = "Random Sample Consensus"
                    MAE_30 = round(mean_absolute_error(y_test, y_pred), 2)
                    MSE_30 = round(mean_squared_error(y_test, y_pred), 2)
                    MAPE_30 = round(mean_absolute_percentage_error(
                        y_test, y_pred)*100, 2)
                    RMSE_30 = round(mean_squared_error(
                        y_test, y_pred, squared=False), 2)
                    try:
                        RMSLE_30 = round(
                            np.sqrt(mean_squared_log_error(y_test, y_pred)), 4)
                    except:
                        RMSLE_30 = "error"
                        """

                    # RMSLE_21 = round(
                    # np.sqrt(mean_squared_log_error(y_test, y_pred)), 4)

                    st.subheader("Regression Coefficients:")

                    col1, col2 = st.beta_columns(2)

                    with col1:
                        st.info("Support Vector Machines: " + str(R2_1))
                        st.info("Linear Regression: " + str(R2_2))
                        st.info("Bayesian Ridge: " + str(R2_3))
                        st.info("Gradient Boosting: " + str(R2_7))
                        st.info("Extreme Gradient Boosting: " + str(R2_8))
                        st.info("Light Gradient Boosting Machine: " + str(R2_11))
                        st.info("AdaBoost: " + str(R2_13))
                        st.info("Bagging: " + str(R2_15))
                        st.info("Elastic Net: " + str(R2_17))
                        st.info("Ridge Regression: " + str(R2_19))
                        st.info("Kernel Ridge: " + str(R2_21))
                        st.info("Theil Sen: " + str(R2_23))
                        st.info("Histogram Gradient Boosting: " + str(R2_25))
                        st.info("Lasso Least Angle Regression: " + str(R2_27))
                        st.info("Random Sample Consensus: " + str(R2_29))
                        st.info("Natural Gradient Boosting: " + str(R2_31))

                    with col2:
                        st.info("Decision Tree: " + str(R2_4))
                        st.info("Random Forest: " + str(R2_5))
                        st.info("K Nearest Neighbors: " + str(R2_6))
                        st.info("Gaussian Process: " + str(R2_9))
                        st.info("Stochastic Gradient Descent: " + str(R2_10))
                        st.info("CatBoost: " + str(R2_12))
                        st.info("Extra Trees: " + str(R2_14))
                        st.info("Passive Aggressive: " + str(R2_16))
                        st.info("Lasso Regression: " + str(R2_18))
                        st.info("Huber: " + str(R2_20))
                        st.info("Tweedie: " + str(R2_22))
                        st.info("Orthogonal Matching Pursuit: " + str(R2_24))
                        st.info("Least Angle Regression: " + str(R2_26))
                        st.info(
                            "Automatic Relevance Determination: " + str(R2_28))
                        st.info("Perceptron: " + str(R2_30))
                        #st.info("Multi-Layer Perceptron: " + str(R2_30))

                        # st.info("KNeighborsClassifier: " + str(R2_6))

                    st.subheader(
                        "Models Runtime: Training + Testing (seconds):")

                    col1, col2 = st.beta_columns(2)

                    with col1:
                        st.info("Support Vector Machines: " + str(time_1))
                        st.info("Linear Regression: " + str(time_2))
                        st.info("Bayesian Ridge: " + str(time_3))
                        st.info("Gradient Boosting: " + str(time_7))
                        st.info("Extreme Gradient Boosting: " + str(time_8))
                        st.info(
                            "Light Gradient Boosting Machine: " + str(time_11))
                        st.info("AdaBoost: " + str(time_13))
                        st.info("Bagging: " + str(time_15))
                        st.info("Elastic Net: " + str(time_17))
                        st.info("Ridge: " + str(time_19))
                        st.info("Kernel Ridge: " + str(time_21))
                        st.info("Theil Sen: " + str(time_23))
                        st.info("Histogram Gradient Boosting: " + str(time_25))
                        st.info("Lasso Least Angle Regression: " + str(time_27))
                        st.info("Random Sample Consensus: " + str(time_29))
                        st.info("Natural Gradient Boosting: " + str(time_31))

                    with col2:
                        st.info("Decision Tree: " + str(time_4))
                        st.info("Random Forest: " + str(time_5))
                        st.info("K Nearest Neighbors: " + str(time_6))
                        st.info("Gaussian Process: " + str(time_9))
                        st.info("Stochastic Gradient Descent: " + str(time_10))
                        st.info("CatBoost: " + str(time_12))
                        st.info("Extra Trees: " + str(time_14))
                        st.info("Passive Aggressive: " + str(time_16))
                        st.info("Lasso: " + str(time_18))
                        st.info("Huber: " + str(time_20))
                        st.info("Tweedie: " + str(time_22))
                        st.info("Orthogonal Matching Pursuit: " + str(time_24))
                        st.info("Least Angle Regression: " + str(time_26))
                        st.info(
                            "Automatic Relevance Determination: " + str(time_28))
                        st.info("Perceptron: " + str(time_30))
                        #st.info("Multi-Layer Perceptron: " + str(time_30))

                    # figure, ax = plt.subplots()
                    # figure = plt.figure()
                    # ax1 = figure.add_subplot(121)
                    fig = plt.figure(figsize=(15, 12))
                    ax1 = fig.add_subplot(121)

                    # plt.subplot(1, 2, 1)

                    X_acc = ['SVMR', 'LR', 'BRR', 'DTR', 'RFR',
                             'KNNR', 'GBR', 'XGBR', 'GPR', 'SGDR',
                             'LGBMR', 'CatBR', 'AdaBR', 'ETR', 'BR',
                             'PAR', 'ENR', 'Lasso', 'RR', 'HR',
                             'KRR', 'TR', 'TSR', 'OMPR', 'HGBR',
                             'LAR', 'LLAR', 'ARDR', 'RANSAC', 'PR', 'NGBR']
                    y_acc = [R2_1, R2_2, R2_3, R2_4, R2_5,
                             R2_6, R2_7, R2_8, R2_9, R2_10,
                             R2_11, R2_12, R2_13, R2_14, R2_15,
                             R2_16, R2_17, R2_18, R2_19, R2_20,
                             R2_21, R2_22, R2_23, R2_24, R2_25,
                             R2_26, R2_27, R2_28, R2_29, R2_30, R2_31]

                    z_acc = [time_1, time_2, time_3, time_4, time_5,
                             time_6, time_7, time_8, time_9, time_10,
                             time_11, time_12, time_13, time_14, time_15,
                             time_16, time_17, time_18, time_20, time_20,
                             time_21, time_22, time_23, time_24, time_25,
                             time_26, time_27, time_28, time_29, time_30, time_31]

                    # Mod_acc = [Mod_1, Mod_2, Mod_3, Mod_4, Mod_5,
                    # Mod_6, Mod_7, Mod_8, Mod_9, Mod_10]

                    dfchart = [['SVMR', mod_1, R2_1, MAE_1, MAPE_1,
                                MSE_1, RMSE_1, RMSLE_1, time_1],
                               ['LR', mod_2, R2_2, MAE_2, MAPE_2,
                                MSE_2, RMSE_2, RMSLE_2, time_2],
                               ['BRR', mod_3, R2_3, MAE_3, MAPE_3,
                                MSE_3, RMSE_3, RMSLE_3, time_3],
                               ['DTR', mod_4, R2_4, MAE_4, MAPE_4,
                                MSE_4, RMSE_4, RMSLE_4, time_4],
                               ['RFR', mod_5, R2_5, MAE_5, MAPE_5,
                                MSE_5, RMSE_5, RMSLE_5, time_5],
                               ['KNNR', mod_6, R2_6, MAE_6, MAPE_6,
                                MSE_6, RMSE_6, RMSLE_6, time_6],
                               ['GBR', mod_7, R2_7, MAE_7, MAPE_7,
                                MSE_7, RMSE_7, RMSLE_7, time_7],
                               ['XGBR', mod_8, R2_8, MAE_8, MAPE_8,
                                MSE_8, RMSE_8, RMSLE_8, time_8],
                               ['GPR', mod_9, R2_9, MAE_9, MAPE_9,
                                MSE_9, RMSE_9, RMSLE_9, time_9],
                               ['SGDR', mod_10, R2_10, MAE_10, MAPE_10,
                                MSE_10, RMSE_10, RMSLE_10, time_10],
                               ['LGBMR', mod_11, R2_11, MAE_11, MAPE_11,
                                MSE_11, RMSE_11, RMSLE_11, time_11],
                               ['CatBR', mod_12, R2_12, MAE_12, MAPE_12,
                                MSE_12, RMSE_12, RMSLE_12, time_12],
                               ['AdaBR', mod_13, R2_13, MAE_13, MAPE_13,
                                MSE_13, RMSE_13, RMSLE_13, time_13],
                               ['ETR', mod_14, R2_14, MAE_14, MAPE_14,
                                MSE_14, RMSE_14, RMSLE_14, time_14],
                               ['BR', mod_15, R2_15, MAE_15, MAPE_15,
                                MSE_15, RMSE_15, RMSLE_15, time_15],
                               ['PAR', mod_16, R2_16, MAE_16, MAPE_16,
                                MSE_16, RMSE_16, RMSLE_16, time_16],
                               ['ENR', mod_17, R2_17, MAE_17, MAPE_17,
                                MSE_17, RMSE_17, RMSLE_17, time_17],
                               ['Lasso', mod_18, R2_18, MAE_18, MAPE_18,
                                MSE_18, RMSE_18, RMSLE_18, time_18],
                               ['RR', mod_19, R2_19, MAE_19, MAPE_19,
                                MSE_19, RMSE_19, RMSLE_19, time_19],
                               ['HR', mod_20, R2_20, MAE_20, MAPE_20,
                                MSE_20, RMSE_20, RMSLE_20, time_20],
                               ['KRR', mod_21, R2_21, MAE_21, MAPE_21,
                                MSE_21, RMSE_21, RMSLE_21, time_21],
                               ['TR', mod_22, R2_22, MAE_22, MAPE_22,
                                MSE_22, RMSE_22, RMSLE_22, time_22],
                               ['TSR', mod_23, R2_23, MAE_23, MAPE_23,
                                MSE_23, RMSE_23, RMSLE_23, time_23],
                               ['OMPR', mod_24, R2_24, MAE_24, MAPE_24,
                                MSE_24, RMSE_24, RMSLE_24, time_24],
                               ['HGBR', mod_25, R2_25, MAE_25, MAPE_25,
                                MSE_25, RMSE_25, RMSLE_25, time_25],
                               ['LAR', mod_26, R2_26, MAE_26, MAPE_26,
                                MSE_26, RMSE_26, RMSLE_26, time_26],
                               ['LLAR', mod_27, R2_27, MAE_27, MAPE_27,
                                MSE_27, RMSE_27, RMSLE_27, time_27],
                               ['ARDR', mod_28, R2_28, MAE_28, MAPE_28,
                                MSE_28, RMSE_28, RMSLE_28, time_28],
                               ['RANSAC', mod_29, R2_29, MAE_29, MAPE_29,
                                MSE_29, RMSE_29, RMSLE_29, time_29],
                               ['PR', mod_30, R2_30, MAE_30, MAPE_30,
                                MSE_30, RMSE_30, RMSLE_30, time_30],
                               ['NGBR', mod_31, R2_31, MAE_31, MAPE_31,
                                MSE_31, RMSE_31, RMSLE_31, time_31]]

                    dfchart = pd.DataFrame(
                        dfchart, columns=['Rank', 'Regressor', 'R2', 'MAE', 'MAPE(%)', 'MSE', 'RMSE', 'RMSLE', 'Time(sec)'])

                    df_New = dfchart.sort_values(
                        by='R2', ascending=False)

                    st.write(df_New)

                    df_New2 = dfchart.sort_values(
                        by='R2', ascending=True)

                    X_acc = df_New2.iloc[:, 0]
                    # Selecting the last column as Y
                    y_acc = df_New2.iloc[:, 2]
                    z_acc = df_New2.iloc[:, 8]

                    X_pos = np.arange(len(X_acc))

                    x_min = min(y_acc)*0.975
                    x_max = max(y_acc)*1.0225

                    ticks = np.arange(min(y_acc)*0.975, max(y_acc)
                                      * 1.0225, (x_max-x_min)/5)

                    # ax.barh(X_pos, y_acc, align='center', alpha=0.5)
                    ax1.barh(X_pos, y_acc,  alpha=0.7, color='deepskyblue')

                    # ax.barh(X_acc, y_acc,  alpha=0.5)
                    # Añadimos la etiqueta de nombre de cada lenguaje en su posicion correcta
                    plt.yticks(X_pos, X_acc, fontsize=15)
                    plt.xticks(ticks, fontsize=15)
                    plt.xlim(min(y_acc)*0.975, max(y_acc)*1.0225)
                    plt.title("Regressor Models Ranking", fontsize=20)
                    plt.xlabel("Regression Coefficient (R2)", fontsize=18)
                    plt.ylabel("Regressor Models", fontsize=18)
                    # st.pyplot(figure)

                    # figure2, ax = plt.subplots()
                    ax2 = fig.add_subplot(122)
                    # plt.subplot(1, 2, 2)
                    # ax2.barh(X_pos, y_acc,  alpha=0.7, color='deepskyblue')

                    x_min = min(z_acc)*0.975
                    x_max = max(z_acc)*1.0225

                    ticks = np.arange(min(z_acc)*0.975, max(z_acc)
                                      * 1.0225, (x_max-x_min)/5)

                    # ax.barh(X_pos, y_acc, align='center', alpha=0.5)
                    ax2.barh(X_pos, z_acc,  alpha=0.7, color='lightgreen')

                    # ax.barh(X_acc, y_acc,  alpha=0.5)
                    # Añadimos la etiqueta de nombre de cada lenguaje en su posicion correcta
                    plt.yticks(X_pos, X_acc, fontsize=15)
                    plt.xticks(ticks, fontsize=15)
                    plt.xlim(min(z_acc)*0.975, max(z_acc)*1.0225)
                    plt.title("Regressor Models Runtime", fontsize=20)
                    plt.xlabel(
                        "Runtime: Training + Testing (seconds)", fontsize=18)
                    # plt.ylabel("Regressor Models", fontsize=17)

                    st.pyplot(fig)

                    if st.button('Re-Run'):
                        caching.clear_cache()
                        st._RerunException

        else:
            st.info('Awaiting for CSV file to be uploaded.')

    except:
        st.info('ERROR - Please check your Dataset, parameters o selected model')
    return
