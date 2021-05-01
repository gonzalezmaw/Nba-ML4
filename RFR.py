import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from streamlit import caching


def RFR():  # Random Forest Regressor
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

            #X = df.iloc[:, :-1].values
            X = df.iloc[:, 0].values
            X_name = df.iloc[:, 0].name
            y_name = df.iloc[:, -1].name
            y = df.iloc[:, -1].values  # Selecting the last column as Y

            #y_name = df.iloc[:, 1].name

            # Taking N% of the data for training and (1-N%) for testing:
            num = int(len(df)*(1-parameter_test_size))
            data = df

            # training data:
            train = df[:num]
            # Testing data:
            test = df[num:]

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

            showData = st.checkbox('Show Dataset')

            X_train = np.array(train[[X_name]])
            y_train = np.array(train[[y_name]])
            X_test = np.array(test[[X_name]])
            y_test = np.array(test[[y_name]])

            #Stan_Scaler = st.checkbox('Standard Scaler')
            # if Stan_Scaler:
            #sc = StandardScaler()
            #X_train = sc.fit_transform(X_train)
            #X_test = sc.transform(X_test)

            regressor = RandomForestRegressor(
                n_estimators=parameter_n_estimators, random_state=0)
            regressor.fit(X_train.reshape(-1, 1), y_train.reshape(-1, 1))

            y_pred = regressor.predict(X_test.reshape(-1, 1))

            df2 = pd.DataFrame({'Real Values': y_test.reshape(-1),
                                'Predicted Values': y_pred.reshape(-1)})

            # # Visualising the Random Forest Regression Results

            if showData:
                st.subheader('Dataset')
                st.write(df)
                figure1, ax = plt.subplots()
                ax.scatter(X, y, label="Dataset")
                plt.title('Dataset')
                plt.xlabel(X_name)
                plt.ylabel(y_name)
                plt.legend()
                st.pyplot(figure1)

            st.subheader("""
                **Regression**
                """)
            st.write(df2)

            st.subheader('Coefficient of determination ($R^2$):')
            resultR2 = round(r2_score(y_test, y_pred), 6)

            st.info(resultR2)

            # chart = st.line_chart(X_test, y_test)
            figure2, ax = plt.subplots()
            ax.scatter(X_test, y_test, label="Real values", color='green')
            ax.scatter(X_test, y_pred, label="Predicted values", color='red')
            plt.title('Random Forest Regression')
            plt.xlabel(X_name)
            plt.ylabel(y_name)
            plt.legend()
            st.pyplot(figure2)

            X_grid = np.arange(min(X), max(X), 0.01)
            X_grid = X_grid.reshape((len(X_grid), 1))
            figure3, ax2 = plt.subplots()
            ax2.plot(X_grid, regressor.predict(X_grid),
                     label="Prediction", color='black')
            plt.title('Random Forest Regression')
            plt.xlabel(X_name)
            plt.ylabel(y_name)
            plt.legend()
            st.pyplot(figure3)

            st.subheader("""
                        **Prediction**
                        """)

            col1, col2 = st.beta_columns(2)

            with col1:
                X_new = st.number_input('Input a number:')

                st.write("Result:")
                y_new = np.reshape(X_new, (1, -1))
                y_new = regressor.predict(y_new)
                st.success(round(y_new[0], 6))
            with col2:
                print("nothing")

            if st.button('Re-Run'):
                caching.clear_cache()
                st._RerunException

            # st.set_option('deprecation.showPyplotGlobalUse', False)
        else:
            st.info('Awaiting for CSV file to be uploaded.')
    except:
        st.info('ERROR - Please check your Dataset, parameters o selected model')
        st.success(
            'All your data must be numeric and without empty or null spaces')
    return
