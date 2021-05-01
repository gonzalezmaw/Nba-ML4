import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from streamlit import caching


def RFC():  # Random Forest Classifier
    try:
        parameter_test_size = st.sidebar.slider(
            "Test size (fraction)", 0.02, 0.80, 0.2)
        st.sidebar.write("Test size: ", parameter_test_size)

        max_depth = st.sidebar.slider("Maximum depth", 2, 20)
        st.sidebar.write("Maximum depth: ", max_depth)

        n_estimators = st.sidebar.slider("n estimators", 1, 100)
        st.sidebar.write("n estimators: ",  n_estimators)

        st.sidebar.info("""
                    [More information](http://gonzalezmaw.pythonanywhere.com/)
                    """)

        uploaded_file = st.file_uploader("Choose a CSV file")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            df = df.interpolate(method='linear', axis=0).bfill()
            df = df.dropna()
            df = df.drop_duplicates()

            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]  # Selecting the last column as Y

            st.subheader("Data Processing:")
            st.write("Shape of dataset:", df.shape)
            st.write('number of classes:', len(np.unique(y)))
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=parameter_test_size, random_state=0)
            st.write("Complete data: ", len(df))
            st.write("Data to train: ", len(X_train))
            st.write("Data to test: ", len(X_test))
            
            st.write("Descriptive statistics:")
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

            # Create a Random Forest Classifier
            classifier = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth, random_state=1234)
            # Train the model using the training sets
            classifier.fit(X_train, y_train)

            # Predict the response for test dataset
            y_pred = classifier.predict(X_test)

            acc = round(accuracy_score(y_test, y_pred), 5)

            # PLOT
            pca = PCA(2)
            X_projected = pca.fit_transform(X)

            x1 = X_projected[:, 0]
            x2 = X_projected[:, 1]

            st.subheader("Classification:")
            st.write("**Accuracy:**")
            st.info(acc)

            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.figure()
            # fig, ax = plt.subplots()
            # figure, ax = plt.subplots()
            plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
            plt.xlabel("Principal Component X1")
            plt.ylabel("Principal Component X2")
            plt.title('Random Forest Classification')
            plt.colorbar()
            st.pyplot()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.info(classification_report(y_test, y_pred))

            # st.write("**Precision:**")
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
                y_new = classifier.predict(X_new)
                y_target = dfT.iloc[:, -1].name
                dfT_new = pd.DataFrame(X_new)
                dfT_new[y_target] = y_new  # Agregar la columna

                st.subheader("""
                        **Prediction**
                        """)

                st.write(dfT_new)

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
                plt.title('Random Forest - Prediction')
                plt.colorbar()
                st.pyplot()

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
