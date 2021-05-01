import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from scipy.stats import pearsonr
from scipy import stats
from sklearn import preprocessing
from random import randint
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


def MRFR():  # Multiple Random Forest Regressor
    try:
        parameter_test_size = st.sidebar.slider(
            "Test size (fraction)", 0.02, 0.80, 0.2)
        st.sidebar.write("Test size: ", parameter_test_size)

        parameter_n_estimators = st.sidebar.slider(
            "n estimators", 1, 1000, 100)
        st.sidebar.write("n  estimators: ", parameter_n_estimators)

        st.sidebar.info("""
                [More information](http://gonzalezmaw.pythonanywhere.com/)
                """)
        uploaded_file = st.file_uploader("Choose a CSV file")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            df = df.interpolate(method='linear', axis=0).bfill()
            df = df.dropna()
            df = df.drop_duplicates()
            # Using all column except for the last column as X
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]  # Selecting the last column as Y

            st.write("shape of dataset:", df.shape)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=parameter_test_size, random_state=0)
            st.write("Complete data: ", len(df))
            st.write("Data to train: ", len(X_train))
            st.write("Data to test: ", len(X_test))
            st.write(df.describe())

            showData = st.checkbox('Show Dataset')
            if showData:
                st.subheader('Dataset')
                st.write(df)

            Stan_Scaler = st.checkbox('Standard Scaler')
            if Stan_Scaler:
                sc = StandardScaler()
                X_train = sc.fit_transform(X_train)
                X_test = sc.transform(X_test)

            # Create a Regressor Model
            model = RandomForestRegressor(
                n_estimators=parameter_n_estimators, random_state=0)

            # Train the model using the training sets
            model.fit(X_train, y_train)

            # Predict the response for test dataset
            y_pred = model.predict(X_test)

            st.subheader("""
                    **Regression**
                    """)
            correlation = round(pearsonr(y_pred, y_test)[0], 5)
            st.write("Pearson correlation coefficient:")
            st.info(correlation)

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
                    #sns.boxplot(y=k, data=df, ax=axs[index], palette="Paired")
                    sns.boxplot(y=k, data=df, ax=axs[index], palette="Set3")
                    index += 1
                plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
                st.pyplot(fig)

            showOutPerc = st.checkbox('Show Outliers Percentage')
            if showOutPerc:
                for k, v in df.items():
                    q1 = v.quantile(0.25)
                    q3 = v.quantile(0.75)
                    irq = q3 - q1
                    v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
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
                    ncols=5, nrows=numFilesX, figsize=(20, 10))
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
                # Let's scale the columns before plotting them against MEDV
                min_max_scaler = preprocessing.MinMaxScaler()
                column_sels = list(X.columns)
                # st.info(column_sels)
                # x = df.loc[:, column_sels]
                # y = df[y.name]
                X = pd.DataFrame(data=min_max_scaler.fit_transform(X),

                                 columns=column_sels)
                fig, axs = plt.subplots(
                    ncols=5, nrows=numFiles, figsize=(20, 10))
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
                # st.write(dfT)
                # Using all column except for the last column as X
                X_new = dfT.iloc[:, :-1]
                y_new = model.predict(X_new)
                y_target = dfT.iloc[:, -1].name
                dfT_new = pd.DataFrame(X_new)
                dfT_new[y_target] = y_new  # Agregar la columna

                st.write(dfT_new)

                if st.button('Re-Run'):
                    caching.clear_cache()
                    st._RerunException

            # features_importance = model.feature_importances_

            # st.write("Feature ranking:")
            # for i, data_class in enumerate(feature_names):
            #    st.write("{}. {} ({})".format(
            #       i + 1, data_class, features_importance[i]))

            # acc = round(accuracy_score(y_test, y_pred), 5)

        else:
            st.info('Awaiting for CSV file to be uploaded.')

    except:
        st.info('ERROR - Please check your Dataset, parameters o selected model')
        st.success(
            'All your data must be numeric and without empty or null spaces')
    return
