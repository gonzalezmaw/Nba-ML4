import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
from streamlit import caching


def PRM():  # Polynomial Regresssion Method
    try:
        parameter_test_size = st.sidebar.slider(
            "Test size (fraction)", 0.02, 0.80, 0.2)
        st.sidebar.write("Test size: ", parameter_test_size)

        polynomial_grade = st.sidebar.slider(
            "Polynomial grade:", 2, 10, 2)
        st.sidebar.write("Polynomial grade: ", polynomial_grade)

        st.sidebar.info("""
                [More information](http://gonzalezmaw.pythonanywhere.com/)
                """)
        uploaded_file = st.file_uploader("Choose a CSV file")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            df = df.dropna()
            df = df.drop_duplicates()

            # Header name

            X_name = df.iloc[:, 0].name
            y_name = df.iloc[:, -1].name
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]  # Selecting the last column as Y

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

            #st.write("Shape of dataset:", df.shape)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=parameter_test_size, random_state=0)
            #st.write("Complete data: ", len(df))
            #st.write("Data to train: ", len(X_train))
            #st.write("Data to test: ", len(X_test))
            # st.write(df.describe())

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

            # Create a model
            X_ = PolynomialFeatures(
                degree=polynomial_grade, include_bias=False).fit_transform(X)
            X_train_ = PolynomialFeatures(
                degree=polynomial_grade, include_bias=False).fit_transform(X_train)
            X_test_ = PolynomialFeatures(
                degree=polynomial_grade, include_bias=False).fit_transform(X_test)

            # Train the model using the training sets
            model = LinearRegression().fit(X_, y)
            # model = LinearRegression().fit(X_train_, y_train)

            st.subheader("""
                    **Regression**
                    """)
            r_sq = round(model.score(X_, y), 6)

            intercept = round(model.intercept_, 6)
            st.write("Intercept:")
            st.info(intercept)
            coefficients = model.coef_
            # st.table(coefficients)
            st.write("Coefficients:")
            st.info(coefficients)

            st.write('Coefficient of determination ($R^2$):')

            st.info(r_sq)

            # Predict the response for test dataset
            y_pred = model.predict(X_test_)

            # Predicting values for the whole dataset
            predicted_data = model.predict(X_)

            # Predicting values for the whole dataset
            predicted_train = model.predict(X_train_)
            # fitted = model.fit()
            # st.info(fitted.summary())

            # acc = accuracy_score(y_test, y_pred)
            # st.info(acc)
            figure2, ax = plt.subplots()
            # ax.scatter(X, y, label="Dataset", color='Blue')
            ax.scatter(X, y, label="Dataset", s=15)
            ax.plot(X, predicted_data, label="Dataset", color="Red")
            plt.title('Complete Dataset')
            plt.xlabel(X_name)
            plt.ylabel(y_name)
            # plt.legend()
            st.pyplot(figure2)

            figure3, ax = plt.subplots()
            # ax.scatter(X, y, label="Dataset", color='Blue')
            ax.scatter(X_train, y_train, label="Dataset", s=15, color="Green")
            ax.plot(X_train, predicted_train, label="Dataset", color="Red")
            plt.title('Training Dataset')
            plt.xlabel(X_name)
            plt.ylabel(y_name)
            # plt.legend()
            st.pyplot(figure3)

            figure4, ax = plt.subplots()
            # ax.scatter(X, y, label="Dataset", color='Blue')
            ax.scatter(X_test, y_test, label="Dataset", s=15, color="Green")
            ax.plot(X_test, y_pred, label="Dataset", color="Red")
            plt.title('Test Dataset')
            plt.xlabel(X_name)
            plt.ylabel(y_name)
            # plt.legend()
            st.pyplot(figure4)

            # for creating pipeline
            from sklearn.pipeline import Pipeline
            # creating pipeline and fitting it on data
            Input = [('polynomial', PolynomialFeatures(degree=2)),
                     ('modal', LinearRegression())]
            pipe = Pipeline(Input)
            pipe.fit(X, y)
            # poly_pred = pipe.predict(X)

            # st.info(poly_pred)
            # st.write(poly_pred.shape)
            # sorting predicted values with respect to predictor
            # sorted_zip = sorted(zip(X, poly_pred))
            # x_poly, poly_pred = zip(*sorted_zip)
            # st.info(x_poly)
            # plotting predictions

            # figure3, ax1 = plt.subplots()
            # plt.figure(figsize=(10, 6))
            # ax1.scatter(X, y, s=15)
            # plt.plot(x,y_pred,color='r',label='Linear Regression')
            # ax1.plot(X, poly_pred, color='red', label='Polynomial Regression')
            # plt.xlabel('Predictor', fontsize=16)
            # plt.ylabel('Target', fontsize=16)
            # plt.legend()
            # st.pyplot(figure3)
            st.subheader("""
                    **Prediction**
                    """)

            col1, col2 = st.beta_columns(2)

            with col1:
                X_new = st.number_input('Input a number:')
                matrix = np.array([X_new]).reshape(1, -1)
                X_2 = PolynomialFeatures(
                    degree=polynomial_grade, include_bias=False).fit_transform(matrix)

            with col1:
                st.write("Result:")
            # y_new = np.reshape(X_new, (1, -1))
                y_new = model.predict(X_2)
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
