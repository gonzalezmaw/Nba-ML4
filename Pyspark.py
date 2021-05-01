from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import RobustScaler
from pyspark.ml.feature import MaxAbsScaler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import Normalizer
from pyspark.ml.feature import MinMaxScaler
import os
from pyspark import SparkConf, SparkContext
# from pyspark.sql.types import StringType
from pyspark import SQLContext
# from pyspark.sql import SparkSession

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import streamlit as st
import time

from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.regression import IsotonicRegression
# from pyspark.ml.regression import AFTSurvivalRegression
# from pyspark.ml.regression import FMRegressor #Factorization Machine Regressor'

from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier

from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import NaiveBayes

# Next Classifier Models
"""
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.classification import OneVsRest
from pyspark.ml.classification import FMClassifier
"""


# Next Regressor Models
"""
from pyspark.ml.regression import AFTSurvivalRegression
from pyspark.ml.regression import FMRegressor
"""


# import sys
# print(sys.executable)

# Set spark environments
os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/bin/python3'

# sc.stop()pyt


# import findspark
# findspark.init()

# Write a custom function to convert the data type of DataFrame columns


def Pyspark(RankingMethod):

    try:
        parameter_test_size = st.sidebar.slider(
            "Test size (fraction)", 0.02, 0.80, 0.2)
        st.sidebar.write("Test size: ", parameter_test_size)

        st.sidebar.info("""
                        [More information](http://gonzalezmaw.pythonanywhere.com/)
                        """)

        st.write('/Users/MarlonGonzalez/Documents/MachineLearning_2021/2008.csv')
        # st.write('/Users/MarlonGonzalez/Documents/MachineLearning_2021/BostonHousing.csv')
        st.write(
            '/Users/MarlonGonzalez/Documents/MachineLearning_2021/Regression_Pwf_Modeling.csv')
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

                    try:
                        if sc != None:
                            sc.stop()
                    except:
                        pass

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
                    SelectColumns.append(columSelectY)
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
                        st.write()
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

                    # st.write(SelectColumns)

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

                    if RankingMethod == "Regressor Ranking":
                        stringRanking = "Regressor"
                        stringMetric = "Regression Coefficient (R2)"
                        # Linear Regression
                        regressor = LinearRegression(
                            featuresCol="Features", labelCol=columSelectY,  maxIter=100, regParam=0.3, elasticNetParam=0.8)
                        start_time_1 = time.time()
                        pipeline = Pipeline(stages=[featureScaler, regressor])
                        regressorModel = pipeline.fit(trainingData)
                        results = regressorModel.transform(testingData)
                        results.select("Prediction", columSelectY, "Features")
                        # rfresults.show()
                        # Using RMSE to evaluate the model
                        evaluator = RegressionEvaluator(
                            labelCol=columSelectY, predictionCol="prediction", metricName="r2")
                        R2_1 = round(evaluator.evaluate(results), 4)
                        evaluator = RegressionEvaluator(
                            labelCol=columSelectY, predictionCol="prediction", metricName="mae")
                        mae_1 = round(evaluator.evaluate(results), 2)
                        evaluator = RegressionEvaluator(
                            labelCol=columSelectY, predictionCol="prediction", metricName="mse")
                        mse_1 = round(evaluator.evaluate(results), 2)
                        evaluator = RegressionEvaluator(
                            labelCol=columSelectY, predictionCol="prediction", metricName="rmse")
                        rmse_1 = round(evaluator.evaluate(results), 2)
                        time_1 = round(time.time() - start_time_1, 4)
                        rmod_1 = "Linear Regression"
                        # st.write(mae_1)

                        # Generalized Linear Regression
                        regressor = GeneralizedLinearRegression(
                            featuresCol="Features", labelCol=columSelectY,  family="gaussian", link="identity", maxIter=25, regParam=0.3)
                        start_time_2 = time.time()
                        pipeline = Pipeline(stages=[featureScaler, regressor])
                        regressorModel = pipeline.fit(trainingData)
                        results = regressorModel.transform(testingData)
                        results.select("Prediction", columSelectY, "Features")
                        # rfresults.show()
                        # Using RMSE to evaluate the model
                        evaluator = RegressionEvaluator(
                            labelCol=columSelectY, predictionCol="prediction", metricName="r2")
                        R2_2 = round(evaluator.evaluate(results), 4)
                        evaluator = RegressionEvaluator(
                            labelCol=columSelectY, predictionCol="prediction", metricName="mae")
                        mae_2 = round(evaluator.evaluate(results), 2)
                        evaluator = RegressionEvaluator(
                            labelCol=columSelectY, predictionCol="prediction", metricName="mse")
                        mse_2 = round(evaluator.evaluate(results), 2)
                        evaluator = RegressionEvaluator(
                            labelCol=columSelectY, predictionCol="prediction", metricName="rmse")
                        rmse_2 = round(evaluator.evaluate(results), 2)
                        time_2 = round(time.time() - start_time_2, 4)
                        rmod_2 = "Generalized Linear"

                        # Decision Tree Regression
                        regressor = DecisionTreeRegressor(
                            featuresCol="Features", labelCol=columSelectY, maxDepth=5, maxBins=32, seed=0)
                        start_time_3 = time.time()
                        pipeline = Pipeline(stages=[featureScaler, regressor])
                        regressorModel = pipeline.fit(trainingData)
                        results = regressorModel.transform(testingData)
                        results.select("Prediction", columSelectY, "Features")
                        evaluator = RegressionEvaluator(
                            labelCol=columSelectY, predictionCol="prediction", metricName="r2")
                        R2_3 = round(evaluator.evaluate(results), 4)
                        evaluator = RegressionEvaluator(
                            labelCol=columSelectY, predictionCol="prediction", metricName="mae")
                        mae_3 = round(evaluator.evaluate(results), 2)
                        evaluator = RegressionEvaluator(
                            labelCol=columSelectY, predictionCol="prediction", metricName="mse")
                        mse_3 = round(evaluator.evaluate(results), 2)
                        evaluator = RegressionEvaluator(
                            labelCol=columSelectY, predictionCol="prediction", metricName="rmse")
                        rmse_3 = round(evaluator.evaluate(results), 2)
                        time_3 = round(time.time() - start_time_3, 4)
                        rmod_3 = "Decision Tree"

                        # Random Forest Regression
                        regressor = RandomForestRegressor(
                            featuresCol="Features", labelCol=columSelectY,  maxDepth=5, maxBins=32, numTrees=20, seed=0)
                        start_time_4 = time.time()
                        pipeline = Pipeline(stages=[featureScaler, regressor])
                        regressorModel = pipeline.fit(trainingData)
                        results = regressorModel.transform(testingData)
                        results.select("Prediction", columSelectY, "Features")
                        evaluator = RegressionEvaluator(
                            labelCol=columSelectY, predictionCol="prediction", metricName="r2")
                        R2_4 = round(evaluator.evaluate(results), 4)
                        evaluator = RegressionEvaluator(
                            labelCol=columSelectY, predictionCol="prediction", metricName="mae")
                        mae_4 = round(evaluator.evaluate(results), 2)
                        evaluator = RegressionEvaluator(
                            labelCol=columSelectY, predictionCol="prediction", metricName="mse")
                        mse_4 = round(evaluator.evaluate(results), 2)
                        evaluator = RegressionEvaluator(
                            labelCol=columSelectY, predictionCol="prediction", metricName="rmse")
                        rmse_4 = round(evaluator.evaluate(results), 2)
                        time_4 = round(time.time() - start_time_4, 4)
                        rmod_4 = "Random Forest"

                        # Gradient Boosted Trees Regression to do the prediction
                        regressor = GBTRegressor(
                            featuresCol="Features", labelCol=columSelectY, maxDepth=5, maxBins=32, maxIter=20, seed=0)
                        start_time_5 = time.time()
                        pipeline = Pipeline(stages=[featureScaler, regressor])
                        regressorModel = pipeline.fit(trainingData)
                        results = regressorModel.transform(testingData)
                        results.select("Prediction", columSelectY, "Features")
                        evaluator = RegressionEvaluator(
                            labelCol=columSelectY, predictionCol="prediction", metricName="r2")
                        R2_5 = round(evaluator.evaluate(results), 4)
                        evaluator = RegressionEvaluator(
                            labelCol=columSelectY, predictionCol="prediction", metricName="mae")
                        mae_5 = round(evaluator.evaluate(results), 2)
                        evaluator = RegressionEvaluator(
                            labelCol=columSelectY, predictionCol="prediction", metricName="mse")
                        mse_5 = round(evaluator.evaluate(results), 2)
                        evaluator = RegressionEvaluator(
                            labelCol=columSelectY, predictionCol="prediction", metricName="rmse")
                        rmse_5 = round(evaluator.evaluate(results), 2)
                        time_5 = round(time.time() - start_time_5, 3)
                        rmod_5 = "Gradient Boosted Trees"

                        # Isotoic Regression to do the prediction
                        regressor = IsotonicRegression(
                            featuresCol="Features", labelCol=columSelectY)
                        start_time_6 = time.time()
                        pipeline = Pipeline(stages=[featureScaler, regressor])
                        regressorModel = pipeline.fit(trainingData)
                        results = regressorModel.transform(testingData)
                        results.select("Prediction", columSelectY, "Features")
                        evaluator = RegressionEvaluator(
                            labelCol=columSelectY, predictionCol="prediction", metricName="r2")
                        R2_6 = round(evaluator.evaluate(results), 4)
                        evaluator = RegressionEvaluator(
                            labelCol=columSelectY, predictionCol="prediction", metricName="mae")
                        mae_6 = round(evaluator.evaluate(results), 2)
                        evaluator = RegressionEvaluator(
                            labelCol=columSelectY, predictionCol="prediction", metricName="mse")
                        mse_6 = round(evaluator.evaluate(results), 2)
                        evaluator = RegressionEvaluator(
                            labelCol=columSelectY, predictionCol="prediction", metricName="rmse")
                        rmse_6 = round(evaluator.evaluate(results), 2)
                        time_6 = round(time.time() - start_time_6, 4)
                        rmod_6 = "Isotoic Regression"

                    # Classifier Models
                    elif RankingMethod == "Classifier Ranking":
                        stringRanking = "Classifier"
                        stringMetric = "Accuracy"
                        # Decision tree classifier
                        classifier = DecisionTreeClassifier(
                            featuresCol="Features", labelCol=columSelectY, seed=0)
                        start_time_11 = time.time()
                        pipeline = Pipeline(stages=[featureScaler, classifier])
                        classifierModel = pipeline.fit(trainingData)
                        results = classifierModel.transform(testingData)
                        results.select("Prediction", columSelectY, "Features")
                        evaluator = MulticlassClassificationEvaluator(
                            labelCol=columSelectY, predictionCol="prediction", metricName="accuracy")
                        Acc_1 = round(evaluator.evaluate(results), 4)
                        evaluator = MulticlassClassificationEvaluator(
                            labelCol=columSelectY, predictionCol="prediction", metricName="weightedPrecision")
                        prec_1 = round(evaluator.evaluate(results), 4)
                        evaluator = MulticlassClassificationEvaluator(
                            labelCol=columSelectY, predictionCol="prediction", metricName="weightedRecall")
                        rec_1 = round(evaluator.evaluate(results), 4)
                        evaluator = MulticlassClassificationEvaluator(
                            labelCol=columSelectY, predictionCol="prediction", metricName="f1")
                        f1 = round(evaluator.evaluate(results), 4)

                        time_11 = round(time.time() - start_time_11, 4)
                        cmod_1 = "Decision tree"

                        # Logistic Regression Classifier
                        classifier = LogisticRegression(
                            featuresCol="Features", labelCol=columSelectY)
                        start_time_12 = time.time()
                        pipeline = Pipeline(stages=[featureScaler, classifier])
                        classifierModel = pipeline.fit(trainingData)
                        results = classifierModel.transform(testingData)
                        results.select("Prediction", columSelectY, "Features")
                        evaluator = MulticlassClassificationEvaluator(
                            labelCol=columSelectY, predictionCol="prediction", metricName="accuracy")
                        Acc_2 = round(evaluator.evaluate(results), 4)
                        evaluator = MulticlassClassificationEvaluator(
                            labelCol=columSelectY, predictionCol="prediction", metricName="weightedPrecision")
                        prec_2 = round(evaluator.evaluate(results), 4)
                        evaluator = MulticlassClassificationEvaluator(
                            labelCol=columSelectY, predictionCol="prediction", metricName="weightedRecall")
                        rec_2 = round(evaluator.evaluate(results), 4)
                        evaluator = MulticlassClassificationEvaluator(
                            labelCol=columSelectY, predictionCol="prediction", metricName="f1")
                        f2 = round(evaluator.evaluate(results), 4)
                        time_12 = round(time.time() - start_time_12, 4)
                        cmod_2 = "Logistic Regression"

                        # Random Forest Classifier
                        classifier = RandomForestClassifier(
                            featuresCol="Features", labelCol=columSelectY, seed=0)
                        start_time_13 = time.time()
                        pipeline = Pipeline(stages=[featureScaler, classifier])
                        classifierModel = pipeline.fit(trainingData)
                        results = classifierModel.transform(testingData)
                        results.select("Prediction", columSelectY, "Features")
                        evaluator = MulticlassClassificationEvaluator(
                            labelCol=columSelectY, predictionCol="prediction", metricName="accuracy")
                        Acc_3 = round(evaluator.evaluate(results), 4)
                        evaluator = MulticlassClassificationEvaluator(
                            labelCol=columSelectY, predictionCol="prediction", metricName="weightedPrecision")
                        prec_3 = round(evaluator.evaluate(results), 4)
                        evaluator = MulticlassClassificationEvaluator(
                            labelCol=columSelectY, predictionCol="prediction", metricName="weightedRecall")
                        evaluator = MulticlassClassificationEvaluator(
                            labelCol=columSelectY, predictionCol="prediction", metricName="f1")
                        f3 = round(evaluator.evaluate(results), 4)
                        rec_3 = round(evaluator.evaluate(results), 4)
                        time_13 = round(time.time() - start_time_13, 4)
                        cmod_3 = "Random Forest"

                        """
                            # Only Binary Classification
                            # Gradient-boosted tree classifier
                            classifier = GBTClassifier(
                                featuresCol="Features", labelCol=columSelectY, maxDepth=5, maxBins=32, maxIter=20, seed=0)
                            start_time_14 = time.time()
                            pipeline = Pipeline(stages=[featureScaler, classifier])
                            classifierModel = pipeline.fit(trainingData)
                            results = classifierModel.transform(testingData)
                            results.select("Prediction", columSelectY, "Features")
                            evaluator = MulticlassClassificationEvaluator(
                                labelCol=columSelectY, predictionCol="prediction", metricName="accuracy")
                            Acc_4 = round(evaluator.evaluate(results), 4)
                            evaluator = MulticlassClassificationEvaluator(
                                labelCol=columSelectY, predictionCol="prediction", metricName="weightedRecall")
                            rec_4 = round(evaluator.evaluate(results), 4)
                            evaluator = MulticlassClassificationEvaluator(
                            labelCol=columSelectY, predictionCol="prediction", metricName="f1")
                            f4 = round(evaluator.evaluate(results), 4)
                            time_14 = round(time.time() - start_time_14, 4)
                            """

                        """
                            # Only Support Binary Classification
                            # Linear Support Vector Machine Classifier
                            classifier = LinearSVC(
                                featuresCol="Features", labelCol=columSelectY)
                            start_time_15 = time.time()
                            pipeline = Pipeline(stages=[featureScaler, classifier])
                            classifierModel = pipeline.fit(trainingData)
                            results = classifierModel.transform(testingData)
                            results.select("Prediction", columSelectY, "Features")
                            evaluator = MulticlassClassificationEvaluator(
                                labelCol=columSelectY, predictionCol="prediction", metricName="accuracy")
                            Acc_5 = round(evaluator.evaluate(results), 4)
                            evaluator = MulticlassClassificationEvaluator(
                                labelCol=columSelectY, predictionCol="prediction", metricName="weightedRecall")
                            rec_5 = round(evaluator.evaluate(results), 4)
                            evaluator = MulticlassClassificationEvaluator(
                            labelCol=columSelectY, predictionCol="prediction", metricName="f1")
                            f5 = round(evaluator.evaluate(results), 4)
                            time_15 = round(time.time() - start_time_15, 4)
                            """

                        # Naive Bayes Classifier
                        classifier = NaiveBayes(
                            featuresCol="Features", labelCol=columSelectY)
                        start_time_16 = time.time()
                        pipeline = Pipeline(stages=[featureScaler, classifier])
                        classifierModel = pipeline.fit(trainingData)
                        results = classifierModel.transform(testingData)
                        results.select("Prediction", columSelectY, "Features")
                        evaluator = MulticlassClassificationEvaluator(
                            labelCol=columSelectY, predictionCol="prediction", metricName="accuracy")
                        Acc_6 = round(evaluator.evaluate(results), 4)
                        evaluator = MulticlassClassificationEvaluator(
                            labelCol=columSelectY, predictionCol="prediction", metricName="weightedPrecision")
                        prec_6 = round(evaluator.evaluate(results), 4)
                        evaluator = MulticlassClassificationEvaluator(
                            labelCol=columSelectY, predictionCol="prediction", metricName="weightedRecall")
                        rec_6 = round(evaluator.evaluate(results), 4)
                        evaluator = MulticlassClassificationEvaluator(
                            labelCol=columSelectY, predictionCol="prediction", metricName="f1")
                        f6 = round(evaluator.evaluate(results), 4)
                        time_16 = round(time.time() - start_time_16, 4)
                        cmod_6 = "Naive Bayes"

                        """
                            This model show error
                            # Factorization Machines Classifier
                            classifier = FMClassifier(
                                featuresCol="Features", labelCol=columSelectY, stepSize=0.001)
                            start_time_17 = time.time()
                            pipeline = Pipeline(stages=[featureScaler, classifier])
                            classifierModel = pipeline.fit(trainingData)
                            results = classifierModel.transform(testingData)
                            results.select("Prediction", columSelectY, "Features")
                            evaluator = MulticlassClassificationEvaluator(
                                labelCol=columSelectY, predictionCol="prediction", metricName="accuracy")
                            Acc_7 = round(evaluator.evaluate(results), 4)
                            time_17 = round(time.time() - start_time_17, 4)
                            """

                    st.subheader(stringMetric + " :")
                    col1, col2 = st.beta_columns(2)

                    with col1:
                        if RankingMethod == "Regressor Ranking":
                            st.info("Linear Regressor: " + str(R2_1))
                            st.info("Generalized Linear Regressor: " + str(R2_2))
                            st.info("Decision Tree Regressor: " + str(R2_3))
                        elif RankingMethod == "Classifier Ranking":
                            st.info(
                                "Logistic Regression Classifier: " + str(Acc_1))
                            st.info("Decision Tree Classifier: " + str(Acc_2))

                    with col2:
                        if RankingMethod == "Regressor Ranking":
                            st.info("Random Forest Regressor: " + str(R2_4))
                            st.info(
                                "Gradient-Boosted Tree Regressor: " + str(R2_5))
                            st.info("Isotonic Regressor: " + str(R2_6))
                        elif RankingMethod == "Classifier Ranking":
                            st.info("Random Forest Classifier: " + str(Acc_3))
                            # st.info("Gradient-Boosted Tree Classifier: " + str(Acc_4))
                            # st.info(
                            # "Support Vector Machine Classifier: " + str(Acc_5))
                            st.info("Naive Bayes Classifier: " + str(Acc_6))
                            # st.info(
                            # "Factorization Machines Classifier: " + str(Acc_7))

                    st.subheader("Models Simulation Time (seconds):")

                    col1, col2 = st.beta_columns(2)

                    with col1:
                        if RankingMethod == "Regressor Ranking":
                            st.info("Linear Regressor: " + str(time_1))
                            st.info(
                                "Generalized Linear Regressor: " + str(time_2))
                            st.info("Decision Tree Regressor: " + str(time_3))
                        elif RankingMethod == "Classifier Ranking":
                            st.info(
                                "Logistic Regression Classifier: " + str(time_11))
                            st.info("Decision Tree Classifier: " + str(time_12))

                    with col2:
                        if RankingMethod == "Regressor Ranking":
                            st.info("Random Forest Regressor: " + str(time_4))
                            st.info(
                                "Gradient-Boosted Tree Regressor: " + str(time_5))
                            st.info("Isotonic Regressor: " + str(time_6))
                        elif RankingMethod == "Classifier Ranking":
                            st.info("Random Forest Classifier: " + str(time_13))
                            # st.info("Gradient-Boosted Tree Classifier: " + str(time_14))
                            # st.info(
                            # "Support Vector Machine Classifier: " + str(time_15))
                            st.info("Naive Bayes Classifier: " + str(time_16))
                            # st.info(
                            # "Factorization Machines Classifier: " + str(time_17))

                    fig = plt.figure(figsize=(15, 12))
                    ax1 = fig.add_subplot(121)

                    if RankingMethod == "Regressor Ranking":

                        X_acc = ['LR', 'GLR', 'DTR', 'RFR', 'GBTR', 'IR']
                        y_acc = [R2_1, R2_2, R2_3, R2_4, R2_5, R2_6]
                        z_acc = [time_1, time_2, time_3,
                                 time_4, time_5, time_6]

                        dfchart = [['LR', rmod_1, R2_1, mae_1, mse_1, rmse_1, time_1],
                                   ['GLR', rmod_2, R2_2, mae_2,
                                       mse_2, rmse_2, time_2],
                                   ['DTR', rmod_3, R2_3, mae_3,
                                       mse_3, rmse_3, time_3],
                                   ['RFR', rmod_4, R2_4, mae_4,
                                       mse_4, rmse_4, time_4],
                                   ['GBTR', rmod_5, R2_5, mae_5,
                                    mse_5, rmse_5, time_5],
                                   ['IR', rmod_6, R2_6, mae_6, mse_6, rmse_6, time_6]]

                        dfchart = pd.DataFrame(
                            dfchart, columns=['Model', 'Regressor', 'R2', 'MAE', 'MSE', 'RMSE', 'Time(sec)'])

                        df_New = dfchart.sort_values(
                            by='R2', ascending=False)

                        st.write(df_New)
                        df_New2 = dfchart.sort_values(
                            by='R2', ascending=True)

                    elif RankingMethod == "Classifier Ranking":
                        X_acc = ['LRC', 'DTC', 'RFR', 'NBC']
                        y_acc = [Acc_1, Acc_2, Acc_3, Acc_6]
                        z_acc = [time_11, time_12, time_13, time_16]

                        dfchart = [['LRC', cmod_1, Acc_1, prec_1, rec_1, f1, time_11],
                                   ['DTC', cmod_2, Acc_2, prec_2,
                                       rec_2, f2, time_12],
                                   ['RFR', cmod_3, Acc_3, prec_3,
                                       rec_3, f3, time_13],
                                   ['NBC', cmod_6, Acc_6, prec_6, rec_6, f6, time_16]]

                        dfchart = pd.DataFrame(
                            dfchart, columns=['Model', 'Classifier', 'Accuracy', 'Precision', 'Recall', 'F1', 'Time(sec)'])

                        df_New = dfchart.sort_values(
                            by='Accuracy', ascending=False)
                        st.write(df_New)
                        df_New2 = dfchart.sort_values(
                            by='Accuracy', ascending=True)

                    X_acc = df_New2.iloc[:, 0]
                    # Selecting the last column as Y
                    y_acc = df_New2.iloc[:, 2]
                    z_acc = df_New2.iloc[:, 6]

                    X_pos = np.arange(len(X_acc))

                    x_min = min(y_acc)*0.975
                    x_max = max(y_acc)*1.0225

                    ticks = np.arange(min(y_acc)*0.975, max(y_acc)
                                      * 1.0225, (x_max-x_min)/5)

                    ax1.barh(X_pos, y_acc,  alpha=0.7, color='deepskyblue')

                    plt.yticks(X_pos, X_acc, fontsize=15)
                    plt.xticks(ticks, fontsize=15)
                    plt.xlim(min(y_acc)*0.975, max(y_acc)*1.0225)
                    plt.title(stringRanking + " Models Ranking", fontsize=20)
                    plt.xlabel(stringMetric, fontsize=18)
                    plt.ylabel(stringRanking + " Models", fontsize=18)

                    ax2 = fig.add_subplot(122)

                    x_min = min(z_acc)*0.975
                    x_max = max(z_acc)*1.0225

                    ticks = np.arange(min(z_acc)*0.975, max(z_acc)
                                      * 1.0225, (x_max-x_min)/5)

                    ax2.barh(X_pos, z_acc,  alpha=0.7, color='lightgreen')

                    plt.yticks(X_pos, X_acc, fontsize=15)
                    plt.xticks(ticks, fontsize=15)
                    plt.xlim(min(z_acc)*0.975, max(z_acc)*1.0225)
                    plt.title(stringRanking + " Models Time", fontsize=20)
                    plt.xlabel("Time (Seconds)", fontsize=18)

                    st.pyplot(fig)

                    # rute = "/Users/MarlonGonzalez/Documents/MachineLearning_2021/2008.csv"
                    # delimiter = ","

                    # df = pd.read_csv(
                    #    "/Users/MarlonGonzalez/Documents/MachineLearning_2021/2008.csv", nrows=1000000)

                    # dfspark = sqlContext.read.format("csv").option("header", "true").option(
                    #    "inferSchema", "true").load("/Users/MarlonGonzalez/Documents/MachineLearning_2021/2008.csv")

                    # st.write(dfspark.head(10))

                    sc.stop()

                # columsSelect = st.checkbox(
                # "Select X columns for the clustering analysis", columsList)

                """

                        dfspark = dfspark.sample(
                            fraction=0.001, withReplacement=False)
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

    # st.write("Error")

    # pdf = df.toPandas()

    """




            sc.stop()

            """

    return
