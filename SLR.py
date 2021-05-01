import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from streamlit import caching


def SLR():  # Simple Linear Regression
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

            # Header name
            X_name = df.iloc[:, 0].name
            y_name = df.iloc[:, -1].name

            # Using all column except for the last column as X
            # X = df.iloc[:, :-1].values
            X = df.iloc[:, 0].values
            # Selecting the last column as Y
            y = df.iloc[:, -1].values

            # Taking N% of the data for training and (1-N%) for testing:
            num = int(len(df)*(1-parameter_test_size))
            # training data:
            data = df
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
            if showData:
                st.subheader('Dataset')
                st.write(df)
                figure1, ax = plt.subplots()

                ax.scatter(X, y, label="Dataset", s=15)
                plt.title('Dataset')
                plt.xlabel(X_name)
                plt.ylabel(y_name)
                plt.legend()
                st.pyplot(figure1)

            # st.write("Null values: ", df.info())

            # Training the model:
            regr = linear_model.LinearRegression()
            train_x = np.array(train[[X_name]])
            train_y = np.array(train[[y_name]])

            regr.fit(train_x, train_y)
            coefficients = regr.coef_
            intercept = regr.intercept_

            st.subheader("""
            **Regression**
            """)
            # Slope:
            st.write("Slope:")
            st.info(coefficients[0])
            # st.info(coefficients)
            # Inercept:
            st.write("Intercept:")
            st.info(intercept)

            # Predicting values for the whole dataset
            predicted_data = regr.predict(data[[X_name]])

            # Predicting values for the whole dataset
            predicted_train = regr.predict(train[[X_name]])

            # Predicting values for testing data
            predicted_test = regr.predict(test[[X_name]])

            st.write('Coefficient of determination ($R^2$):')
            resultR2 = r2_score(y, regr.predict(data[[X_name]]))

            st.info(round(resultR2, 6))

            figure2, ax = plt.subplots()
            # ax.scatter(X, y, label="Dataset", color='Blue')
            ax.scatter(data[X_name], data[y_name], label="Dataset", s=15)
            ax.plot(data[X_name], predicted_data, label="Dataset", color="Red")
            plt.title('Complete Dataset')
            plt.xlabel(X_name)
            plt.ylabel(y_name)
            # plt.legend()
            st.pyplot(figure2)

            figure3, ax = plt.subplots()
            ax.scatter(train[X_name], train[y_name],
                       label="Dataset", color="Green", s=15)
            ax.plot(train[X_name], predicted_train,
                    label="Dataset", color="Red")
            plt.title('Training Dataset')
            plt.xlabel(X_name)
            plt.ylabel(y_name)
            st.pyplot(figure3)

            figure4, ax = plt.subplots()
            ax.scatter(test[X_name], test[y_name],
                       label="Dataset", color="Green", s=15)
            ax.plot(test[X_name], predicted_test, label="Dataset", color="Red")
            plt.title('Test Dataset')
            plt.xlabel(X_name)
            plt.ylabel(y_name)
            st.pyplot(figure4)

            st.subheader("""
                        **Prediction**
                        """)

            col1, col2 = st.beta_columns(2)

            with col1:
                X_new = st.number_input('Input a number:')

            with col1:
                st.write("Result:")
                y_new = np.reshape(X_new, (1, -1))
                y_new = regr.predict(y_new)
                st.success(y_new[0])

            with col2:
                print("nothing")

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
