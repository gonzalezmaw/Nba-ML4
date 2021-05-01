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

from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier

from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import NaiveBayes

# Next Models
"""
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.classification import OneVsRest
from pyspark.ml.classification import FMClassifier
"""

from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import Normalizer
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import MaxAbsScaler
from pyspark.ml.feature import RobustScaler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import PCA as PCAml
from pyspark.ml import Pipeline

from sklearn import preprocessing
from sklearn.decomposition import PCA as PCAp

#from pyspark.sql.types import ArrayType, DoubleType
from pyspark.sql.types import FloatType
from pyspark.sql.functions import udf
from pyspark.ml.feature import VectorSlicer

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql.types import *

#from pyspark.mllib.linalg.distributed import RowMatrix

os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/bin/python3'


def ClassifiersPyspark(selectModelClassifier):
    try:
        parameter_test_size = st.sidebar.slider(
            "Test size (fraction)", 0.02, 0.80, 0.2)
        st.sidebar.write("Test size: ", parameter_test_size)

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

            col1, col2 = st.beta_columns(2)

            with col1:
                columsList = df.columns.tolist()
                # st.write(columsList)
                columsSelectX = st.multiselect(
                    "Select X columns for the analysis", columsList)

                # st.write(columsSelectX)

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

            selectScalerType = st.selectbox(
                "Select a scaler method", ("MinMax Scaler", "MaxAbs Scaler", "Normalizer", "Robust Scaler", "Standard Scaler"))

            # sampling = st.slider("Sampling fraction: ", 0.0001, 1.0000, 0.1)

            sampling = st.selectbox("Sampling fraction: ", ("1.0", "0.9", "0.8", "0.7",
                                                            "0.6", "0.5", "0.4", "0.3", "0.2", "0.1", "0.01", "0.001", "0.0001", "0.00001"))

            samplingNum = float(sampling)

            # st.write(SelectColumns)

            if len(columsSelectX) > 0 and columSelectY != None:

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
                    # st.info(dfspark.count())

                    # st.info(dfspark.printSchema)
                    # st.info(dfspark.dtypes)
                    # df_SelectX = DataFrame (your_list,columns=['Column_Name'])

                    SelectColumns = columsSelectX
                    # st.write(columsSelectX)
                    SelectColumns.append(columSelectY)
                    # st.write(columsSelectX)
                    # st.info(dfspark[SelectColumns].dtypes)
                    st.info(dfspark[SelectColumns].printSchema)

                    # st.write(dfspark.dtypes['Qo'])
                    pandasdf = dfspark.limit(1).toPandas()
                    st.write(pandasdf.dtypes[SelectColumns])

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

                    """
                    if len(stringColumns) > 0:
                        for valor_a, valor_b in zip(stringColumns_, exampleDataTypes_):
                            # convertColumn(dfspark, valor_a, valor_b)
                            st.write(str(valor_a) + str(valor_b))

                    """

                    """

                    if len(stringColumns) > 0:
                        for index, valor_a in enumerate(stringColumns_):
                            convertColumn(dfspark, valor_a,
                                        exampleDataTypes_[index])

                    """
                    """

                    if len(stringColumns) > 0:
                        for valor_a, valor_a in zip(stringColumns_, exampleDataTypes_):
                            dfspark = dfspark.withColumn(
                                valor_a, dfspark[valor_a].cast(valor_a))
                                """

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

                    # dfspark = dfspark.withColumn(
                        # "ArrTime", dfspark["ArrTime"].cast("integer"))

                    if len(stringColumns) > 0:
                        st.info(dfspark[SelectColumns].printSchema)

                    # df2 = dfspark.na.drop(subset=SelectColumns)
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
                        st.success(df2.rdd.getNumPartitions())

                    columsSelectX.remove(columSelectY)

                    assembler = VectorAssembler(
                        inputCols=columsSelectX, outputCol='Features_')

                    # transformed_dfspark = assembler.transform(df2)

                    transformed_dfspark = assembler.transform(df2)
                    finalData = transformed_dfspark.select(
                        'Features_', columSelectY)
                    if selectScalerType == "MinMax Scaler":
                        featureScaler = MinMaxScaler(inputCol="Features_",
                                                     outputCol="Features").fit(finalData)
                    elif selectScalerType == "MaxAbs Scaler":
                        featureScaler = MaxAbsScaler(inputCol="Features_",
                                                     outputCol="Features").fit(finalData)

                    elif selectScalerType == "Normalizer":
                        featureScaler = Normalizer(inputCol="Features_",
                                                   outputCol="Features", p=1.0)

                    elif selectScalerType == "Robust Scaler":
                        featureScaler = RobustScaler(inputCol="Features_",
                                                     outputCol="Features", withScaling=True, withCentering=False,
                                                     lower=0.25, upper=0.75).fit(finalData)

                    elif selectScalerType == "Standard Scaler":
                        featureScaler = StandardScaler(inputCol="Features_",
                                                       outputCol="Features", withMean=False, withStd=True).fit(finalData)

                    # Scale features.
                    # featureScaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures").fit(finalData)

                    # st.write(finalData.head(10))
                    # prueba = finalData.head(10)
                    # st.write(prueba.toPandas())

                    test_size = parameter_test_size
                    train_size = 1-test_size

                    trainingData, testingData = finalData.randomSplit(
                        [train_size, test_size], seed=0)

                    if selectModelClassifier == "Decision Tree Classifier":
                        # Decision Tree Classifier
                        classifier = DecisionTreeClassifier(
                            featuresCol="Features", labelCol=columSelectY, predictionCol=columSelectY+"_pred", seed=0)
                        start_time_1 = time.time()
                        pipeline = Pipeline(stages=[featureScaler, classifier])
                        classifierModel = pipeline.fit(trainingData)
                        results = classifierModel.transform(testingData)
                        results.select(columSelectY+"_pred",
                                       columSelectY, "Features")
                        evaluator = MulticlassClassificationEvaluator(
                            labelCol=columSelectY, predictionCol=columSelectY+"_pred", metricName="accuracy")
                        Acc_ = round(evaluator.evaluate(results), 4)
                        time_ = round(time.time() - start_time_1, 4)
                        ModelType = "Decision Tree Classifier"

                    elif selectModelClassifier == "Logistic Regression Classifier":
                        # Logistic Regression Classifier
                        classifier = LogisticRegression(
                            featuresCol="Features", labelCol=columSelectY, predictionCol=columSelectY+"_pred")
                        start_time_2 = time.time()
                        pipeline = Pipeline(stages=[featureScaler, classifier])
                        classifierModel = pipeline.fit(trainingData)
                        results = classifierModel.transform(testingData)
                        results.select(columSelectY+"_pred",
                                       columSelectY, "Features")
                        evaluator = MulticlassClassificationEvaluator(
                            labelCol=columSelectY, predictionCol=columSelectY+"_pred", metricName="accuracy")
                        Acc_ = round(evaluator.evaluate(results), 4)
                        time_ = round(time.time() - start_time_2, 4)
                        ModelType = "Logistic Regression Classifier"

                    elif selectModelClassifier == "Random Forest Classifier":
                        # Random Forest Classifier
                        classifier = RandomForestClassifier(
                            featuresCol="Features", labelCol=columSelectY, predictionCol=columSelectY+"_pred", seed=0)
                        start_time_3 = time.time()
                        pipeline = Pipeline(stages=[featureScaler, classifier])
                        classifierModel = pipeline.fit(trainingData)
                        results = classifierModel.transform(testingData)
                        results.select(columSelectY+"_pred",
                                       columSelectY, "Features")
                        evaluator = MulticlassClassificationEvaluator(
                            labelCol=columSelectY, predictionCol=columSelectY+"_pred", metricName="accuracy")
                        Acc_ = round(evaluator.evaluate(results), 4)
                        time_ = round(time.time() - start_time_3, 4)
                        ModelType = "Random Forest Classifier"

                    elif selectModelClassifier == "Support Vector Machine Classifier":
                        # ONLY SUPPORT BINARY CLASSIFICATION
                        # Support Vector Machine Classifier
                        classifier = LinearSVC(
                            featuresCol="Features", labelCol=columSelectY, predictionCol=columSelectY+"_pred")
                        start_time_4 = time.time()
                        pipeline = Pipeline(stages=[featureScaler, classifier])
                        classifierModel = pipeline.fit(trainingData)
                        results = classifierModel.transform(testingData)
                        results.select(columSelectY+"_pred",
                                       columSelectY, "Features")
                        evaluator = MulticlassClassificationEvaluator(
                            labelCol=columSelectY, predictionCol=columSelectY+"_pred", metricName="accuracy")
                        Acc_ = round(evaluator.evaluate(results), 4)
                        time_ = round(time.time() - start_time_4, 4)
                        ModelType = "Support Vector Machine Classifier"

                    elif selectModelClassifier == "Navy Bayes Classifier":
                        # Gradient Boosted Trees Regression to do the prediction
                        classifier = NaiveBayes(
                            featuresCol="Features", labelCol=columSelectY, predictionCol=columSelectY+"_pred")
                        start_time_5 = time.time()
                        pipeline = Pipeline(stages=[featureScaler, classifier])
                        classifierModel = pipeline.fit(trainingData)
                        results = classifierModel.transform(testingData)
                        results.select(columSelectY+"_pred",
                                       columSelectY, "Features")
                        evaluator = MulticlassClassificationEvaluator(
                            labelCol=columSelectY, predictionCol=columSelectY+"_pred", metricName="accuracy")
                        Acc_ = round(evaluator.evaluate(results), 4)
                        time_ = round(time.time() - start_time_5, 3)
                        ModelType = "Navy Bayes Classifier"

                    st.subheader("""
                        **Classification**
                        """)
                    col1, col2 = st.beta_columns(2)

                    with col1:
                        st.write("Accurancy")
                        st.info(ModelType + ": " + str(Acc_))

                    with col2:
                        st.write("Models Simulation Time (seconds)")
                        st.info(ModelType + ": " + str(time_))

                    pd_results = results.toPandas()

                    # st.write(pd_results[columSelectY+"_pred"])

                    axs = sns.jointplot(
                        x=pd_results[columSelectY], y=pd_results[columSelectY+"_pred"], kind="reg", color="dodgerblue")
                    st.pyplot(axs)

                    pca = PCAml(k=2, inputCol="Features_", outputCol="PCA")
                    model = pca.fit(testingData)
                    X_transformed = model.transform(testingData).select("PCA")
                    #X_transformed = model.transform(testingData)
                    # st.write(X_transformed.printSchema())

                    PCA = X_transformed.toPandas()
                    # st.write(PCA)

                    spark = SparkSession.builder.getOrCreate()
                    #df = spark.createDataFrame(PCA, FloatType())
                    df = spark.createDataFrame(PCA)
                    #from pyspark.sql import functions as f

                    slicer_1 = VectorSlicer(
                        inputCol="PCA", outputCol="features_one", indices=[0])

                    output = slicer_1.transform(df)
                    x1_new = output.select("features_one")

                    x1_new = x1_new.toPandas()
                    # st.write(x1_new)

                    slicer_2 = VectorSlicer(
                        inputCol="PCA", outputCol="features_two", indices=[1])

                    output = slicer_2.transform(df)
                    x2_new = output.select("features_two")

                    x2_new = x2_new.toPandas()

                    trainingData2, testingData2 = df2.randomSplit(
                        [train_size, test_size], seed=0)
                    print(trainingData2)

                    """
                    col1, col2 = st.beta_columns(2)

                    with col1:
                        selectX = st.selectbox(
                            "Select the X axis to plot", columsSelectX)
                        index = columsSelectX.index(selectX)
                        select2 = np.delete(columsSelectX, index)
                    with col2:
                        selectY = st.selectbox(
                            "Select the Y axis to plot", select2)

                    #index_X = columsSelectX.index(selectX)
                    #index_Y = columsSelectX.index(selectY)
                    """

                    # st.write(testingData.toPandas())
                    # st.write(testingData2.toPandas())

                    testingData2 = testingData2[columsSelectX].toPandas()

                    testingData2[columSelectY] = pd_results[columSelectY]
                    testingData2['prediction'] = pd_results[columSelectY+"_pred"]

                    st.write(testingData2)
                    #df = pd.DataFrame(testingData2)

                    X_new = testingData2.iloc[:, :]
                    #X_new = pd.DataFrame(X_new)
                    # st.write(X_new)

                    pca = PCAp(2)
                    X_projected = pca.fit_transform(X_new)

                    # st.write(X_projected)

                    x1_new_ = X_projected[:, 0]
                    # st.write(x1)
                    x2_new_ = X_projected[:, 1]

                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    plt.figure()
                    plt.scatter(
                        x1_new_, x2_new_, c=pd_results[columSelectY+"_pred"], alpha=0.8, cmap="viridis")

                    plt.xlabel("Principal Component X1")
                    plt.ylabel("Principal Component X2")
                    plt.title(ModelType + " - Test Data")
                    plt.colorbar()
                    st.pyplot()

                    sc.stop()
                    # st.write(x2_new)
                    # st.write(testingData.toPandas())
                    # st.write(pd_results[columSelectY+"_pred"])

                    """
                    AA = np.array(df.select('PCA').collect())
                    st.write(AA)
                    """

                    # or: to_array = F.udf(lambda v: list([float(x) for x in v]), T.ArrayType(T.FloatType()))
                    """
                    to_array = F.udf(lambda v: v.toArray().tolist(),
                                    T.ArrayType(T.FloatType()))
                    df = df.withColumn('PCA', to_array('PCA'))
                    st.write(df.toPandas())
                    """

                    # column_names

                    """
                    splits = [F.udf(lambda val: float(val[i].item()),
                                    FloatType()) for i in range(2)]
                    df = df.select(*[s(F.col('PCA')).alias(c)
                                    for c, s in zip('col1,col2', splits)])
                    st.write(df.toPandas())
                    """

                    """
                    firstelement = f.udf(lambda v: float(v[0]), FloatType())
                    st.write(df.withColumn("PCA", firstelement("PCA")).show())
                    """

                    #firstelement = udf(lambda v: float(v[0]), FloatType())
                    # st.write(df.select(firstelement('PCA')).show())

                    #from pyspark.ml.linalg.distributed import RowMatrix
                    #from pyspark.ml.feature import *
                    """
                    pca_features = x1_new.select(
                        "features_one").rdd.map(lambda row: row[0])
                    mat = RowMatrix(pca_features)
                    st.write(mat.toPandas())
                    """

                    #st.write(output.select("PCA", "features_one").toPandas())
                    # st.write(output.select("features_one").toPandas())

                    # x1_new = x1_new.withColumn(
                    # "features_one",  x1_new["features_one"].cast('double'))

                    #x1_new = x1_new.toPandas()

                    # st.write(x2_new)

                    # st.write(x1_new.dtypes["features_one"])
                    # st.write(x2_new.dtypes["features_two"])

                    #x1_new1 = x1_new.apply(pd.to_numeric, errors='coerce')
                    #x1_new["features_one"] = x1_new.features_one.astype(float)

                    #x2_new1 = x2_new.apply(pd.to_numeric, errors='coerce')
                    #x2_new["features_two"] = x2_new.features_two.astype(float)

                    #st.write(output.select("PCA", "features_two").toPandas())
                    # st.write(output.select("features_two").toPandas())

                    """
                    from pyspark.sql import functions as f
                    from pyspark.sql.types import FloatType
                    firstelement = f.udf(lambda v: float(v[0]), FloatType())
                    df.withColumn("PCA", firstelement("PCA")).show()
                    """

                    """
                    from pyspark.sql.functions import udf
                    from pyspark.sql.types import FloatType

                    firstelement = udf(lambda v: float(v[0]), FloatType())
                    st.write(df.select(firstelement('PCA')).show())
                    """

                    """
                    def ith_(v, i):
                        try:
                            return float(v[i])
                        except ValueError:
                            return None

                    ith = udf(ith_, DoubleType())

                    st.write(df.select(
                        ith("PCA", lit(1))).show())
                        """

                    #df2 = spark.sql("SELECT PCA FROM df")
                    # st.write(df2.show())

                    #pd3 = spark.createDataFrame(X_transformed, "array").toDF("PDA")

                    #df3 = spark.createDataFrame(X_transformed)

                    #import pyspark.sql.functions as f
                    # df.withColumn("first_two", f.array(
                    #    [f.col("PCA")[0], f.col("PCA")[1]])).show()

                    # st.write(X_transformed.withColumn("prob", $"PCA".getItem(0)))

                    #from pyspark.sql.functions import udf
                    #from pyspark.sql.types import FloatType
                    #firstelement = udf(lambda v: float(v[0]), FloatType())
                    # st.write(X_transformed.select(firstelement('PCA')).show())

                    #from pyspark.ml.feature import VectorSlicer
                    #slicer = VectorSlicer(inputCol="features", outputCol="features_one", indices=[0])

                    #result = model.transform(df).select("pcaFeatures")

                    # st.write(columsSelectX)
                    # st.write(PCA[0])
                    #X_projected = pca2.fit_transform(X_new)

                    #x1_new = PCA[:, 0]
                    # st.write(x1_new)
                    #x2_new = PCA[:, 1]
                    # st.write(x2_new)

                """

                    dfspark = dfspark.sample(fraction=0.001, withReplacement=False)
                    # st.info(dfspark.count())

                    dfspark = dfspark.withColumn(
                        "ArrDelay", dfspark["ArrDelay"].cast("integer"))

                    df2 = dfspark.na.drop(
                        subset=["ArrDelay", "DepDelay", "Distance"])
                    df2 = df2.filter("ArrDelay is not NULL")
                    df2 = df2.dropDuplicates()
                    # st.write(df2.head(10))
                    # st.info(df2.count())
                    # st.info(df2.printSchema)

                    mean = np.mean(df2.select("ArrDelay").collect())

                    # st.info(mean)
                    df2.rdd.getNumPartitions()

                    df2.select("ArrDelay").filter("ArrDelay > 60").take(5)
                    # df2.select("ArrDelay").filter("ArrDelay > 60").take(5)[0]

                    df2.filter("ArrDelay > 60").take(5)

                    # df2.select("ArrDelay").rdd.map(lambda x: (x-mean)**2).take(10)
                    # st.info(df2)

                    df3 = df2.select("Origin").rdd.distinct().take(5)
                    # st.write(df3)
                    # st.info(df3)

                    df3 = df2.groupBy("DayOfWeek").count().take(7)
                    # st.write(df3)
                    # st.table(df2.groupBy("DayOfWeek").count())

                    df3 = df2.groupBy("DayOfWeek").mean("ArrDelay").take(7)
                    # st.info(df3)

                    df3 = df2.orderBy(df2.ArrDelay.desc()).take(5)
                    # st.write(df3)

                    df3 = df2.select("ArrDelay").describe().take(5)

                    # st.info(df3)
                    # st.write(df3)
                    # st.write(df2.select("ArrDelay").describe().show())

                    df3 = df2.select("Origin").rdd.countByValue()
                    st.info(df3)
                    # st.write(df3) Aparece error de JSON

                    df3 = df2.select("ArrDelay").rdd.max()[0]
                    st.info(df3)

                    df3 = df2.select("Origin").rdd.collect()
                    st.info(df3)

                    """

                # sc.stop()
    except:
        sc.stop()
        st.info('ERROR - Please check your Dataset, parameters o selected model')

    return
