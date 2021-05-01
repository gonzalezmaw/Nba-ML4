import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# import sklearn.cluster as Cluster
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering

from sklearn import preprocessing
from streamlit import caching
import collections

# Next Models
"""
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AffinityPropag
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.cluster import Birch
"""


def ClusteringModels(ClusteringSelectModel):

    try:
        clusters_number = st.sidebar.slider(
            "Clusters number", 2, 20, 3, 1)
        st.sidebar.write("Clusters number: ", clusters_number)

        st.sidebar.info("""
                                    [More information](http://gonzalezmaw.pythonanywhere.com/)
                                    """)

        uploaded_file = st.file_uploader("Choose a CSV file")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            #df = df.interpolate(method='linear', axis=0).bfill()
            #df = df.dropna()

            df = df.drop_duplicates()

            # X = df.iloc[:, :]
            # X = X.astype(str)
            st.subheader("Data Processing:")
            st.write("Shape of dataset:", df.shape)

            st.write(df.describe())

            showData = st.checkbox('Show Dataset')
            if showData:
                st.subheader('Dataset')
                st.write(df)
                # st.write(X)

            Transform_data = st.checkbox(
                'Transform Dataset (Categorial --> Numerical)')
            columsList = df.columns

            X = df.iloc[:, :-1]
            Xheader = list(X.columns)
            df = df.dropna(subset=Xheader)

            if Transform_data:

                ColCategories = st.multiselect(
                    "Categorical Columns", columsList)
                df2 = df

                if ColCategories:
                    # st.write(ColCategories)
                    le = preprocessing.LabelEncoder()
                    df2[ColCategories] = df[ColCategories].apply(
                        le.fit_transform)
                    st.write(df2)
                    X = df2.iloc[:, :-1]

                    # fig = sns.pairplot(X)
                    # st.pyplot(fig)

            pairplot_X = st.checkbox(
                'Show Pairplot')

            if pairplot_X:

                fig = sns.pairplot(X)
                st.pyplot(fig)

                # ColCateg = ColCategories.split(',')

                # df3 = df[ColCategories].apply(le.fit_transform)

                # ColCategories = st.selectbox(
                # "Categorical Column", columsList)

                # df2 = df

                # encoder = preprocessing.LabelEncoder()
                # encoder.fit(df[ColCateg])
                # encoder.fit(df["Gener", "Age"])
                # st.write(encoder.classes_)
                # df2[ColCateg] = encoder.transform(df[ColCateg])
                # df3 = le.fit_transform(df[["Gener"]])
                # df5 = le.inverse_transform(df3)
                # df[["Gener", "Age"]] = df3[["Gener", "Age"]]

            st.subheader("Clustering Analysis:")

            columsSelect = st.multiselect(
                "Select X columns for the clustering analysis", columsList)

            # RowsSelect = RowsSelect.split(',')
            # st.info(columsSelect)
            # st.write(df[columsSelect])
            if len(columsSelect) > 1:
                newdf = df[columsSelect].dropna()

                if ClusteringSelectModel == "K-Means Clustering":

                    clustering = KMeans(
                        n_clusters=clusters_number, init='k-means++', random_state=0).fit(newdf)
                    # u_labels = np.unique(kmeans.labels_)
                    np.unique(clustering.labels_, return_counts=True)
                    centroids = clustering.cluster_centers_
                    # st.write(centroids)

                elif ClusteringSelectModel == "Hierarchical Clustering":
                    clustering = AgglomerativeClustering(
                        n_clusters=clusters_number, affinity="euclidean").fit(newdf)

                elif ClusteringSelectModel == "Spectral Clustering":
                    clustering = SpectralClustering(
                        n_clusters=clusters_number, assign_labels="discretize", random_state=0).fit(newdf)

                clustering.labels_

                """
                        wcss = []
                        for i in range(1, 20):
                            kmeans = Cluster.KMeans(
                                n_clusters=i, random_state=0, n_jobs=-1).fit(newdf)
                            # kmeans = Cluster.KMeans(
                            # n_clusters=i, init='k-means++', random_state=0)
                            kmeans.fit(newdf)
                            # inertia method returns wcss for that model
                            wcss.append(kmeans.inertia_)
                            plt.figure(figsize=(10, 5))
                            fig1 = sns.lineplot(range(1, 20), wcss, color='red')
                            plt.title('The Elbow Method')
                            plt.xlabel('Number of clusters')
                            plt.ylabel('WCSS')
                            st.pyplot(fig1)
                            """

                if ClusteringSelectModel == "K-Means Clustering":
                    Sum_of_squared_distances = []
                    K = range(1, 15)
                    for k in K:
                        km = KMeans(
                            n_clusters=k, init='k-means++', random_state=0)

                        km = km.fit(newdf)
                        Sum_of_squared_distances.append(km.inertia_)

                    fig, ax = plt.subplots()
                    # ax.figure(figsize=(10, 5))
                    ax.plot(K, Sum_of_squared_distances, 'bx-')
                    plt.xlabel('K')
                    plt.ylabel('Sum_of_squared_distances')
                    plt.title('Elbow Method for Optimal K')
                    st.pyplot(fig)

                """
                elif ClusteringSelectModel == "Spectral Clustering":
                    Sum_of_squared_distances = []
                    K = range(1, 15)
                    for k in K:
                        km = SpectralClustering(
                            assign_labels="discretize", random_state=0)

                        km = km.fit(newdf)
                        Sum_of_squared_distances.append(km)

                    fig, ax = plt.subplots()
                    # ax.figure(figsize=(10, 5))
                    ax.plot(K, Sum_of_squared_distances, 'bx-')
                    plt.xlabel('K')
                    plt.ylabel('Sum_of_squared_distances')
                    plt.title('Elbow Method for Optimal K')
                    st.pyplot(fig)
                    """

                col1, col2 = st.beta_columns(2)

                with col1:
                    selectX = st.selectbox(
                        "Select the X axis to plot", columsSelect)
                    index = columsSelect.index(selectX)
                    select2 = np.delete(columsSelect, index)
                with col2:
                    selectY = st.selectbox(
                        "Select the Y axis to plot", select2)

                index_X = columsSelect.index(selectX)
                index_Y = columsSelect.index(selectY)

                figure2, ax = plt.subplots()

                scatter = ax.scatter(newdf[selectX], newdf[selectY],
                                     c=clustering.labels_, cmap='viridis')
                # scatter = ax.scatter(newdf[selectX], newdf[selectY],
                # c=kmeans.labels_, cmap='jet')
                plt.colorbar(scatter)

                if ClusteringSelectModel == "K-Means Clustering":
                    ax.scatter(centroids[:, index_X], centroids[:, index_Y],
                               marker='*', c='red', s=30)

                """
                elif ClusteringSelectModel == "Spectral Clustering":
                    ax.scatter(centroids[:, index_X], centroids[:, index_Y],
                            marker='*', c='red', s=30)
                            """
                # ax.scatter(centroids[selectX], centroids[selectY],
                # marker='*', c='red', s=30)
                plt.xlabel(selectX)
                plt.ylabel(selectY)
                plt.title(ClusteringSelectModel)
                # plt.legend()
                st.pyplot(figure2)

                st.subheader("Prediction:")
                uploaded_file_target = st.file_uploader(
                    "Choose a new CSV file for prediction")
                if uploaded_file_target is not None:
                    dfT = pd.read_csv(uploaded_file_target)
                    #dfT = dfT.interpolate(method='linear', axis=0).bfill()
                    # dfT = dfT.dropna()
                    dfT = dfT.drop_duplicates()
                    # Xheader = list(X.columns)
                    dfT = dfT.dropna(subset=Xheader)

                    dfT2 = dfT

                    # Using all column except for the last column as X
                    Transform_New_data = st.checkbox(
                        'Transform New Dataset (Categorial --> Numerical)')
                    columsList = df.columns

                    # try:

                    if Transform_New_data:
                        ColNew_Categories = st.multiselect(
                            "Categorical New Columns", columsList)

                        if ColNew_Categories:
                            le = preprocessing.LabelEncoder()
                            dfT2[ColCategories] = dfT[ColCategories].apply(
                                le.fit_transform)
                    # st.write(dfT2)
                    X_new = dfT2[columsSelect].dropna()

                    # y_target = dfT2.iloc[:, -1].name
                    if ClusteringSelectModel == "K-Means Clustering":
                        y_new = clustering.predict(X_new)

                    elif ClusteringSelectModel == "Hierarchical Clustering":
                        y_new = clustering.fit_predict(X_new)

                    elif ClusteringSelectModel == "Spectral Clustering":
                        y_new = clustering.fit_predict(X_new)

                    dfT2_new = pd.DataFrame(X_new)
                    dfT2_new['prediction'] = y_new  # Agregar la columna
                    st.write(dfT2_new)
                    # except:
                    # print("nothing")

                else:
                    st.info('Awaiting for CSV file to be uploaded.')

            else:
                st.info("You must select at least 2 columns")

            if st.button('Re-Run'):
                caching.clear_cache()
                st._RerunException

        else:
            st.info('Awaiting for CSV file to be uploaded.')

        # except:
        # st.info('ERROR - Please check your Dataset, parameters o selected model')

    except:
        st.info('ERROR - Please check your Dataset, parameters o selected model')

    return

# cargar el conjunto de datos
