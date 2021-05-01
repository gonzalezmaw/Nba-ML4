import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import streamlit as st
import time

from pyspark import SparkConf, SparkContext
from pyspark import SQLContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

from pyspark.ml.clustering import KMeans
from pyspark.ml.clustering import GaussianMixture

from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import Normalizer
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import MaxAbsScaler
from pyspark.ml.feature import RobustScaler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.ml.feature import PCA as PCAml
from pyspark.ml import Pipeline

from sklearn.decomposition import PCA as PCAp

# Next Models
"""
from pyspark.ml.clustering import LDA
from pyspark.ml.clustering import GaussianMixture
from pyspark.ml.clustering import PowerIterationClustering
"""

os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/bin/python3'


def ClusteringPyspark(ModelClustering):
    try:

        clusters_number = st.sidebar.slider(
            "Clusters number", 2, 20, 3, 1)
        st.sidebar.write("Clusters number: ", clusters_number)

        st.sidebar.info("""
                    [More information](http://gonzalezmaw.pythonanywhere.com/)
                    """)

        st.write('/Users/MarlonGonzalez/Documents/MachineLearning_2021/2008.csv')
        # st.write('/Users/MarlonGonzalez/Documents/MachineLearning_2021/BostonHousing.csv')
        # st.write(
        # '/Users/MarlonGonzalez/Documents/MachineLearning_2021/Regression_Pwf_Modeling.csv')
        st.write(
            '/Users/MarlonGonzalez/Documents/MachineLearning_2021/iris.csv')

        stringconnectionCSV = st.text_input(
            'CSV file string connection')

        if stringconnectionCSV != "":

            sizefile = os.path.getsize(stringconnectionCSV)
            sizefile = round(sizefile/1000000, 2)
            sizefile = "File size: " + str(sizefile) + " MB"
            st.info(sizefile)

            df = pd.read_csv(stringconnectionCSV, nrows=1000)
            st.write(df.head(1000))

            columsList = df.columns.tolist()

            columsSelectX = st.multiselect(
                "Select X columns for the Clustering analysis", columsList)

            selectScalerType = st.selectbox(
                "Select a scaler method", ("MinMax Scaler", "MaxAbs Scaler", "Normalizer", "Robust Scaler", "Standard Scaler"))

            # sampling = st.slider("Sampling fraction: ", 0.0001, 1.0000, 0.1)

            sampling = st.selectbox("Sampling fraction: ", ("1.0", "0.9", "0.8", "0.7",
                                                            "0.6", "0.5", "0.4", "0.3", "0.2", "0.1", "0.01", "0.001", "0.0001", "0.00001"))

            samplingNum = float(sampling)

            # st.write(SelectColumns)

            if len(columsSelectX) > 0:

                if st.button('Big data'):

                    conf = SparkConf().setMaster("local").setAppName("NubilaML")
                    sc = SparkContext(conf=conf).getOrCreate()
                    # sc = SparkContext.getOrCreate()
                    sqlContext = SQLContext(sc)

                    st.info(sc)
                    dfspark = sqlContext.read.format("csv").option("header", "true").option(
                        "inferSchema", "true").load(stringconnectionCSV)

                    dfspark = dfspark.sample(
                        fraction=samplingNum, withReplacement=False, seed=0)

                    SelectColumns = columsSelectX

                    st.info(dfspark[SelectColumns].printSchema)

                    # st.write(dfspark.dtypes['Qo'])
                    pandasdf = dfspark.limit(1).toPandas()
                    #pandasdf = pd.DataFrame(pandasdf, columns=['Data type'])
                    st.write(pandasdf.dtypes[SelectColumns])

                    #dataTypeDict = dict(pandasdf.dtypes[SelectColumns])
                    # st.info(dataTypeDict)

                    stringColumns = []
                    exampleDataTypes = []

                    for x in SelectColumns:
                        if pandasdf.dtypes[x] == "object":
                            stringColumns.append(x)
                            exampleDataTypes.append('integer')

                    if len(stringColumns) > 0:
                        # st.write()
                        st.info(stringColumns)
                        st.text_input(
                            'Example of data types (integer, double)', exampleDataTypes)

                    if len(stringColumns) == 1:
                        dfspark = dfspark.withColumn(
                            stringColumns[0], dfspark[stringColumns[0]].cast(exampleDataTypes[0]))
                    elif len(stringColumns) == 2:
                        dfspark = dfspark.withColumn(
                            stringColumns[0], dfspark[stringColumns[0]].cast(exampleDataTypes[0]))
                        dfspark = dfspark.withColumn(
                            stringColumns[1], dfspark[stringColumns[1]].cast(exampleDataTypes[1]))

                    elif len(stringColumns) == 3:
                        dfspark = dfspark.withColumn(
                            stringColumns[0], dfspark[stringColumns[0]].cast(exampleDataTypes[0]))
                        dfspark = dfspark.withColumn(
                            stringColumns[1], dfspark[stringColumns[1]].cast(exampleDataTypes[1]))
                        dfspark = dfspark.withColumn(
                            stringColumns[2], dfspark[stringColumns[2]].cast(exampleDataTypes[2]))

                    elif len(stringColumns) == 4:
                        dfspark = dfspark.withColumn(
                            stringColumns[0], dfspark[stringColumns[0]].cast(exampleDataTypes[0]))
                        dfspark = dfspark.withColumn(
                            stringColumns[1], dfspark[stringColumns[1]].cast(exampleDataTypes[1]))
                        dfspark = dfspark.withColumn(
                            stringColumns[2], dfspark[stringColumns[2]].cast(exampleDataTypes[2]))
                        dfspark = dfspark.withColumn(
                            stringColumns[3], dfspark[stringColumns[3]].cast(exampleDataTypes[3]))

                    elif len(stringColumns) == 5:
                        dfspark = dfspark.withColumn(
                            stringColumns[0], dfspark[stringColumns[0]].cast(exampleDataTypes[0]))
                        dfspark = dfspark.withColumn(
                            stringColumns[1], dfspark[stringColumns[1]].cast(exampleDataTypes[1]))
                        dfspark = dfspark.withColumn(
                            stringColumns[2], dfspark[stringColumns[2]].cast(exampleDataTypes[2]))
                        dfspark = dfspark.withColumn(
                            stringColumns[3], dfspark[stringColumns[3]].cast(exampleDataTypes[3]))
                        dfspark = dfspark.withColumn(
                            stringColumns[4], dfspark[stringColumns[4]].cast(exampleDataTypes[4]))

                    elif len(stringColumns) == 6:
                        dfspark = dfspark.withColumn(
                            stringColumns[0], dfspark[stringColumns[0]].cast(exampleDataTypes[0]))
                        dfspark = dfspark.withColumn(
                            stringColumns[1], dfspark[stringColumns[1]].cast(exampleDataTypes[1]))
                        dfspark = dfspark.withColumn(
                            stringColumns[2], dfspark[stringColumns[2]].cast(exampleDataTypes[2]))
                        dfspark = dfspark.withColumn(
                            stringColumns[3], dfspark[stringColumns[3]].cast(exampleDataTypes[3]))
                        dfspark = dfspark.withColumn(
                            stringColumns[4], dfspark[stringColumns[4]].cast(exampleDataTypes[4]))
                        dfspark = dfspark.withColumn(
                            stringColumns[5], dfspark[stringColumns[5]].cast(exampleDataTypes[5]))

                    elif len(stringColumns) == 7:
                        dfspark = dfspark.withColumn(
                            stringColumns[0], dfspark[stringColumns[0]].cast(exampleDataTypes[0]))
                        dfspark = dfspark.withColumn(
                            stringColumns[1], dfspark[stringColumns[1]].cast(exampleDataTypes[1]))
                        dfspark = dfspark.withColumn(
                            stringColumns[2], dfspark[stringColumns[2]].cast(exampleDataTypes[2]))
                        dfspark = dfspark.withColumn(
                            stringColumns[3], dfspark[stringColumns[3]].cast(exampleDataTypes[3]))
                        dfspark = dfspark.withColumn(
                            stringColumns[4], dfspark[stringColumns[4]].cast(exampleDataTypes[4]))
                        dfspark = dfspark.withColumn(
                            stringColumns[5], dfspark[stringColumns[5]].cast(exampleDataTypes[5]))
                        dfspark = dfspark.withColumn(
                            stringColumns[6], dfspark[stringColumns[6]].cast(exampleDataTypes[6]))

                    elif len(stringColumns) == 8:
                        dfspark = dfspark.withColumn(
                            stringColumns[0], dfspark[stringColumns[0]].cast(exampleDataTypes[0]))
                        dfspark = dfspark.withColumn(
                            stringColumns[1], dfspark[stringColumns[1]].cast(exampleDataTypes[1]))
                        dfspark = dfspark.withColumn(
                            stringColumns[2], dfspark[stringColumns[2]].cast(exampleDataTypes[2]))
                        dfspark = dfspark.withColumn(
                            stringColumns[3], dfspark[stringColumns[3]].cast(exampleDataTypes[3]))
                        dfspark = dfspark.withColumn(
                            stringColumns[4], dfspark[stringColumns[4]].cast(exampleDataTypes[4]))
                        dfspark = dfspark.withColumn(
                            stringColumns[5], dfspark[stringColumns[5]].cast(exampleDataTypes[5]))
                        dfspark = dfspark.withColumn(
                            stringColumns[6], dfspark[stringColumns[6]].cast(exampleDataTypes[6]))
                        dfspark = dfspark.withColumn(
                            stringColumns[7], dfspark[stringColumns[7]].cast(exampleDataTypes[7]))

                    elif len(stringColumns) == 9:
                        dfspark = dfspark.withColumn(
                            stringColumns[0], dfspark[stringColumns[0]].cast(exampleDataTypes[0]))
                        dfspark = dfspark.withColumn(
                            stringColumns[1], dfspark[stringColumns[1]].cast(exampleDataTypes[1]))
                        dfspark = dfspark.withColumn(
                            stringColumns[2], dfspark[stringColumns[2]].cast(exampleDataTypes[2]))
                        dfspark = dfspark.withColumn(
                            stringColumns[3], dfspark[stringColumns[3]].cast(exampleDataTypes[3]))
                        dfspark = dfspark.withColumn(
                            stringColumns[4], dfspark[stringColumns[4]].cast(exampleDataTypes[4]))
                        dfspark = dfspark.withColumn(
                            stringColumns[5], dfspark[stringColumns[5]].cast(exampleDataTypes[5]))
                        dfspark = dfspark.withColumn(
                            stringColumns[6], dfspark[stringColumns[6]].cast(exampleDataTypes[6]))
                        dfspark = dfspark.withColumn(
                            stringColumns[7], dfspark[stringColumns[7]].cast(exampleDataTypes[7]))
                        dfspark = dfspark.withColumn(
                            stringColumns[8], dfspark[stringColumns[8]].cast(exampleDataTypes[8]))

                    elif len(stringColumns) == 10:
                        dfspark = dfspark.withColumn(
                            stringColumns[0], dfspark[stringColumns[0]].cast(exampleDataTypes[0]))
                        dfspark = dfspark.withColumn(
                            stringColumns[1], dfspark[stringColumns[1]].cast(exampleDataTypes[1]))
                        dfspark = dfspark.withColumn(
                            stringColumns[2], dfspark[stringColumns[2]].cast(exampleDataTypes[2]))
                        dfspark = dfspark.withColumn(
                            stringColumns[3], dfspark[stringColumns[3]].cast(exampleDataTypes[3]))
                        dfspark = dfspark.withColumn(
                            stringColumns[4], dfspark[stringColumns[4]].cast(exampleDataTypes[4]))
                        dfspark = dfspark.withColumn(
                            stringColumns[5], dfspark[stringColumns[5]].cast(exampleDataTypes[5]))
                        dfspark = dfspark.withColumn(
                            stringColumns[6], dfspark[stringColumns[6]].cast(exampleDataTypes[6]))
                        dfspark = dfspark.withColumn(
                            stringColumns[7], dfspark[stringColumns[7]].cast(exampleDataTypes[7]))
                        dfspark = dfspark.withColumn(
                            stringColumns[8], dfspark[stringColumns[8]].cast(exampleDataTypes[8]))
                        dfspark = dfspark.withColumn(
                            stringColumns[9], dfspark[stringColumns[9]].cast(exampleDataTypes[9]))

                    if len(stringColumns) > 0:
                        st.info(dfspark[SelectColumns].printSchema)

                    df2 = dfspark.na.drop(subset=SelectColumns)
                    # df2 = df2.filter("ArrDelay is not NULL")
                    df2 = df2.dropDuplicates()

                    st.write("Descriptive statistics:")
                    st.write(df2[SelectColumns].describe().toPandas())

                    col1, col2, col3 = st.beta_columns(3)

                    with col1:
                        st.write("Number of columns:")
                        st.success(df[SelectColumns].shape[1])

                    with col2:
                        st.write("Number of rows:")
                        st.success(df2.count())
                    # st.info(dfspark.printSchema)
                    with col3:
                        st.write("Resilient Distributed Datasets:")
                        # st.success(dfspark.rdd.getNumPartitions())
                        st.success(df2.rdd.getNumPartitions())

                    assembler = VectorAssembler(
                        inputCols=columsSelectX, outputCol='features_')

                    # transformed_dfspark = assembler.transform(df2)

                    transformed_dfspark = assembler.transform(df2)
                    finalData = transformed_dfspark.select(
                        'features_')
                    if selectScalerType == "MinMax Scaler":
                        featureScaler = MinMaxScaler(inputCol="features_",
                                                     outputCol="features").fit(finalData)
                    elif selectScalerType == "MaxAbs Scaler":
                        featureScaler = MaxAbsScaler(inputCol="features_",
                                                     outputCol="features").fit(finalData)

                    elif selectScalerType == "Normalizer":
                        featureScaler = Normalizer(inputCol="features_",
                                                   outputCol="features", p=1.0)

                    elif selectScalerType == "Robust Scaler":
                        featureScaler = RobustScaler(inputCol="features_",
                                                     outputCol="features", withScaling=True, withCentering=False,
                                                     lower=0.25, upper=0.75).fit(finalData)

                    elif selectScalerType == "Standard Scaler":
                        featureScaler = StandardScaler(inputCol="features_",
                                                       outputCol="features", withMean=False, withStd=True).fit(finalData)

                    # Scale features.
                    # featureScaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures").fit(finalData)

                    # st.write(finalData.head(10))
                    # prueba = finalData.head(10)
                    # st.write(prueba.toPandas())

                    """
                    test_size = parameter_test_size
                    train_size = 1-test_size

                    trainingData, testingData = finalData.randomSplit(
                        [train_size, test_size], seed=0)
                    """

                    if ModelClustering == "K-Means":
                        # K-Means Clustering
                        clustering = KMeans(
                            k=clusters_number, featuresCol="features", predictionCol="prediction", seed=0)
                        start_time_1 = time.time()
                        pipeline = Pipeline(stages=[featureScaler, clustering])
                        clusteringModel = pipeline.fit(finalData)
                        results = clusteringModel.transform(finalData)
                        time_ = round(time.time() - start_time_1, 4)
                        ModelType = "K-Means Clustering"

                        clustering = KMeans(featuresCol="features_").setK(
                            clusters_number).setSeed(0)
                        model = clustering.fit(finalData)
                        centers = model.clusterCenters()
                        for center in centers:
                            print(center)

                        cost = np.zeros(19)
                        for k in range(2, 19):
                            #kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
                            if ModelClustering == "K-Means":
                                clustering = KMeans().setK(k).setSeed(0).setFeaturesCol("features_")

                                model = clustering.fit(finalData.sample(
                                    False, 0.1, seed=0))
                                cost[k] = model.summary.trainingCost

                        if ModelClustering == "K-Means":
                            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                            #fig, ax = plt.subplots()
                            ax.plot(range(2, 19), cost[2:19])
                            ax.set_xlabel('K')
                            ax.set_ylabel('Training Cost')
                            plt.title('Training Cost Method for Optimal K')
                            st.pyplot(fig)

                    elif ModelClustering == "Gaussian Mixture":
                        # Latent Dirichlet allocation Clustering
                        clustering = GaussianMixture(
                            k=clusters_number, featuresCol="features", predictionCol="prediction", seed=0)
                        start_time_1 = time.time()
                        pipeline = Pipeline(stages=[featureScaler, clustering])
                        clusteringModel = pipeline.fit(finalData)
                        results = clusteringModel.transform(finalData)
                        time_ = round(time.time() - start_time_1, 4)
                        ModelType = "Gaussian Mixture Clustering"

                        """
                        clustering = GaussianMixture(featuresCol="features_").setK(
                            clusters_number).setSeed(0)
                        model = clustering.fit(finalData)
                        centers = model.clusterCenters()
                        for center in centers:
                            print(center)
                        """

                        """
                        spark = SparkSession.builder.getOrCreate()
                        bla = [e.tolist() for e in centers]
                        #dfC = sc.parallelize(bla).toDF([SelectColumns])
                        dfC = spark.createDataFrame(bla, [SelectColumns])
                        st.write(dfC.toPandas())
                        """

                        # elif ModelClustering == "Gaussian Mixture":
                        #clustering = GaussianMixture().setK(k).setSeed(0).setFeaturesCol("features_")
                        #

                    st.subheader("""
                        **Clustering**
                        """)
                    col1, col2 = st.beta_columns(2)

                    with col1:
                        st.write("Model:")
                        st.info(ModelType + ": " + str(""))

                    with col2:
                        st.write("Models Simulation Time (seconds)")
                        st.info(ModelType + ": " + str(time_))

                    pd_finalData = df2[SelectColumns].toPandas()

                    X_new = pd_finalData.iloc[:, :]
                    #X_new = pd.DataFrame(X_new)
                    # st.write(X_new)

                    pca = PCAml(k=2, inputCol="features_", outputCol="PCA")
                    model = pca.fit(finalData)
                    X_transformed = model.transform(finalData).select("PCA")
                    print(X_transformed)

                    pca = PCAp(2)
                    X_projected = pca.fit_transform(X_new)

                    # st.write(X_projected)

                    x1_new = X_projected[:, 0]
                    # st.write(x1)
                    x2_new = X_projected[:, 1]

                    pd_results = results.toPandas()
                    # st.write(pd_results)

                    # Agregar la columna
                    pd_finalData['prediction'] = pd_results['prediction']
                    st.write(pd_finalData)

                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    plt.figure()
                    plt.scatter(
                        x1_new, x2_new, c=pd_results['prediction'], alpha=0.8, cmap="viridis")

                    plt.xlabel("Principal Component X1")
                    plt.ylabel("Principal Component X2")
                    plt.title(ModelType + " - Data")
                    plt.colorbar()
                    st.pyplot()

                    sc.stop()

                # sc.stop()
    except:
        sc.stop()
        st.info('ERROR - Please check your Dataset, parameters o selected model')

    return
