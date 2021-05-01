import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PowerTransformer
#from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import Normalizer

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
from scipy import stats
from sklearn import preprocessing
from random import randint
import sklearn as sk
import sklearn.neural_network
import math
from streamlit import caching


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


def MLPR():  # Multi-Layer Perceptron Regressor
    try:
        parameter_test_size = st.sidebar.slider(
            "Test size (fraction)", 0.02, 0.80, 0.2)
        st.sidebar.write("Test size: ", parameter_test_size)

        activation = st.sidebar.selectbox(
            "Select a activation function", ("identity", "logistic sigmoid", "hyperbolic tan", "rectified linear unit"))

        if activation == "identity":
            activation = "identity"
        elif activation == "logistic sigmoid":
            activation = "logistic"
        elif activation == "hyperbolic tan":
            activation = "tanh"
        elif activation == "rectified linear unit":
            activation = "relu"

        solver = st.sidebar.selectbox("Select a solver", ("quasi-Newton optimizer",
                                                          "stochastic gradient descent", "stochastic gradient(adam)"))

        if solver == "quasi-Newton optimizer":
            solver = "lbfgs"
        elif solver == "stochastic gradient descent":
            solver = "sgd"
        elif solver == "stochastic gradient(adam)":
            solver = "adam"

        learning_rate_init = st.sidebar.text_input(
            "Initial learning rate", 0.001)
        lri = float(learning_rate_init)

        hidden_layer_sizes = st.sidebar.text_input(
            "Hidden layer sizes", "5,5,5")
        nn = list(eval(hidden_layer_sizes))

        max_iter = st.sidebar.number_input("Maximum number of iterations", 200)
        tol = st.sidebar.text_input("Tolerance for the optimization", 1e-4)
        tol = float(tol)

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
                        "Select a method", ("MinMax Scaler", "Standard Scaler", "MaxAbs Scaler",
                                            "Robust Scaler", "Power Transformer", "Normalizer"))
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

                # model = MLPRegressor(hidden_layer_sizes=100,
                # learning_rate_init = 0.001, max_iter = 200)
                # hidden_layer_sizes=nn[:]
                model = sk.neural_network.MLPRegressor(hidden_layer_sizes=nn[:],
                                                       activation=activation,
                                                       solver=solver,
                                                       alpha=0.0001,
                                                       batch_size='auto',
                                                       learning_rate='constant',
                                                       learning_rate_init=lri,
                                                       power_t=0.5,
                                                       max_iter=max_iter,
                                                       shuffle=True,
                                                       random_state=0,
                                                       tol=tol,
                                                       verbose=False,
                                                       warm_start=False,
                                                       momentum=0.9,
                                                       nesterovs_momentum=True,
                                                       early_stopping=False,
                                                       validation_fraction=0.1,
                                                       beta_1=0.9,
                                                       beta_2=0.999,
                                                       epsilon=1e-08,
                                                       n_iter_no_change=10,
                                                       max_fun=15000)

                # Train the model using the training sets
                model.fit(X_train, y_train)

                # Predict the response for test dataset
                y_pred = model.predict(X_test)

                st.subheader("""
                            **Regression**
                            """)

                col1, col2 = st.beta_columns(2)

                with col1:
                    correlation = round(pearsonr(y_pred, y_test)[0], 4)
                    st.write("Pearson correlation coefficient:")
                    st.info(correlation)

                with col2:
                    coefR2 = round(r2_score(y_test, y_pred), 4)
                    st.write("Regression coefficient:")
                    st.info(coefR2)

                output_filename = "rf_regression.png"
                title_name = "Real Values vs Predicted Values - correlation ({})".format(
                    correlation)
                x_axis_label = "Real Values"
                y_axis_label = "Predicted Values"

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

                    X = dfT[columsSelectX]
                    y = dfT[columSelectY]

                    X_new = dfT[columsSelectX]

                    dfT_new = pd.DataFrame(X_new)

                    # Using all column except for the last column as X
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

                    y_new = model.predict(X_new)
                    y_target = dfT[columSelectY].name+"_pred"

                    dfT_new[y_target] = y_new

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
                        x=dfT[columSelectY], y=dfT_new[y_target], kind="reg", color="dodgerblue")
                    st.pyplot(axs)

                    #dfp[columSelectY] = dfT[columsSelectX]
                    # dfp['prediction'] = y_new  # Agregar la columna
                    #X_new[columSelectY] = y
                    X[columSelectY] = dfT[columSelectY]
                    X['prediction'] = y_new

                    st.write(X)

                    if st.button('Re-Run'):
                        caching.clear_cache()
                        st._RerunException

        else:
            st.info('Awaiting for CSV file to be uploaded.')

    except:
        st.info('ERROR - Please check your Dataset, parameters o selected model')
        st.success(
            'All your data must be numeric and without empty or null spaces')
    return
