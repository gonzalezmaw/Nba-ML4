import os
import streamlit as st
from datetime import datetime, timedelta
from fbprophet import Prophet  # Faceebok Library
# https://facebook.github.io/prophet/docs/uncertainty_intervals.html
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
from pandas import to_datetime
from pandas import DataFrame
from sklearn.metrics import mean_absolute_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose  # pip install statsmodels

import xlrd  # pip install xlrd #library to work with Excel

# ARIMA
import warnings
import itertools
import statsmodels.api as sm
import matplotlib.dates as mdates

# Regressors

from sklearn.model_selection import KFold
from sklearn.model_selection import TimeSeriesSplit

from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import TweedieRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from ngboost import NGBRegressor

# from sklearn.metrics import mean_squared_error, r2_score


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))*100


def TimeSeries():

    try:
        plt.rcParams.update({'figure.figsize': (10, 7), 'figure.dpi': 120})
        excel_date = 42139
        dt = datetime.fromordinal(
            datetime(1900, 1, 1).toordinal() + excel_date - 2)

        # from datetime import datetime

        excel_date = 42139
        python_date = datetime(*xlrd.xldate_as_tuple(excel_date, 0))
        date = python_date.date()
        # st.write(date)

        d1 = datetime(1900, 1, 1)

        strn = '2015/5/15'
        # date_object = datetime.strptime(strn, '%Y-%m-%d')
        date_object = to_datetime(strn)
        year = int(date_object.strftime('%Y'))
        # st.write(year)
        month = int(date_object.strftime('%m'))
        # st.write(month)
        days = int(date_object.strftime('%d'))
        # st.write(days)

        dnow = datetime(year, month, days)

        # date_object = datetime.strptime(strn, '%Y')
        datevalue = (dnow - d1).days + 2
        # st.write(datevalue)
        # st.write(date_object)

        # tt = dt.timetuple()

        # st.write(tt)
        # try:

        # method = st.sidebar.selectbox("select a method", ('SARIMA', 'Prophet - Facebook', 'Regressors'))
        method = st.sidebar.selectbox(
            "select a method", ('SARIMA', 'Prophet - Facebook'))

        if method == "Prophet - Facebook":
            UncInt = st.sidebar.slider(
                "select the confidence intervals", 0.50, 1.00, 0.80, 0.01)
            st.sidebar.write("confidence intervals: ", UncInt)

            st.write("""### **Prophet - Facebook Model**""")
        elif method == "SARIMA":
            st.write(
                """### **Seasonal Auto Regressive Integrated Moving Average (SARIMA) Model**""")

        elif method == "Regressors":
            st.write(
                """### **Regression Models**""")

        st.sidebar.info("""
            [More information](http://gonzalezmaw.pythonanywhere.com/)
            """)

        uploaded_file = st.file_uploader("Choose a CSV file")
        if uploaded_file is not None:

            # filePath = __file__
            # t.write(filePath)
            df = pd.read_csv(uploaded_file)
            # df2 = pd.read_csv(uploaded_file, parse_dates=['date'], index_col='date'))

            df = df.dropna()
            df = df.drop_duplicates()

            col1, col2 = st.beta_columns(2)

            with col1:
                st.write("Shape of dataset:")
                st.info(df.shape)
            with col2:
                st.write("Complete data:")
                st.info(len(df))

            dataset = st.checkbox("show Dataset")
            if dataset:
                st.write(df)

            col1, col2 = st.beta_columns(2)

            with col1:
                columsList = df.columns.tolist()

                columSelectX = st.selectbox(
                    "Select Date columns for the analysis", columsList)

            with col2:
                index = columsList.index(columSelectX)
                select = np.delete(columsList, index)
                columSelectY = st.selectbox(
                    "Select the Target for the analysis", select)

            if len(columSelectX) > 0 and columSelectY != None:

                X = df[columSelectX]
                y = df[columSelectY]  # Selecting the last column as Y

                # Selecting the last column as Y
                # y = df[columSelectX]
                # y[columSelectX] = to_datetime(y[columSelectX])
                # y.set_index(columSelectX, inplace=True)

                # df[columSelectX] = to_datetime(df[columSelectX])
                # st.write(df)

                df2 = pd.DataFrame()
                df2[columSelectX] = df[columSelectX]
                df2[columSelectY] = df[columSelectY]
                df2[columSelectX] = pd.to_datetime(df[columSelectX])
                st.write(df2)

                df2 = df2.sort_values(
                    by=columSelectX, ascending=True)

                dfChart = pd.DataFrame()
                dfChart[columSelectX] = pd.to_datetime(df2[columSelectX])
                dfChart[columSelectY] = df[columSelectY]

                # set date as index
                dfChart.set_index(columSelectX, inplace=True)
                dfChart.index
                # st.write(dfChart)
                #
                y_ = dfChart[columSelectY].resample('MS').mean()
                # y = dfChart[columSelectY].resample('w').mean()

                # plot data
                # fig, ax = plt.subplots(figsize=(15, 7))
                # dfChart.plot(ax=ax)

                # set ticks every week
                # ax.xaxis.set_major_locator(mdates.WeekdayLocator())
                # ax.xaxis.set_major_locator(mdates.YearLocator())
                # set major ticks format
                # ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
                # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

            # st.write(df2)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            ax = dfChart.plot()
            ax.grid(b=True, which='major', color='lightgrey', linestyle='-')
            # y.plot()
            st.pyplot()
            # st.write(y)
            # st.write(dfChart)

            # plt.ylabel(columSelectY, fontsize=15)
            # df2 = df

            multiplicative = st.checkbox("Show Multiplicative Decomposition")

            # import statsmodels.api as sm

            # graphs to show seasonal_decompose

            """
            def seasonal_decompose(y):
                decomposition = sm.tsa.seasonal_decompose(
                    y, model='additive', extrapolate_trend='freq', freq=30)
                fig = decomposition.plot()
                fig.set_size_inches(14, 7)
                st.pyplot()

            seasonal_decompose(y)
            seasonal_decompose(dfChart[columSelectY])
            """

            if multiplicative:

                """
                Multiplicative Time Series:
                Value = Base Level x Trend x Seasonality x Error
                """

                # df.iloc[:, 1]
                # dfChart[columSelectY]

                """
                result_mul = seasonal_decompose(
                    y, model='multiplicative', extrapolate_trend='freq', freq=30)
                    """

                result_mul = seasonal_decompose(y_, model='multiplicative')

                # Extract the Components ----
                # Actual Values = Product of (Seasonal * Trend * Resid)
                df_reconstructed = pd.concat(
                    [result_mul.seasonal, result_mul.trend, result_mul.resid, result_mul.observed], axis=1)
                df_reconstructed.columns = [
                    'seas', 'trend', 'resid', 'actual_values']

                # Plot
                # fig = plt.figure(figsize=(15, 12))
                fig = plt.rcParams.update({'figure.figsize': (10, 10)})
                fig = result_mul.plot().suptitle('Multiplicative Decompose', fontsize=22)
                # fig = result_mul.plot()
                # ax1 = fig3.add_subplot(141)
                # fig.grid()
                st.pyplot()
                st.write(df_reconstructed)

            additive = st.checkbox("Show Additive Decomposition")

            if additive:

                # Additive Decomposition
                """
                Additive time series:
                Value = Base Level + Trend + Seasonality + Error
                """
                # dfChart[columSelectY]
                """
                result_add = seasonal_decompose(
                    y, model='additive', extrapolate_trend='freq', freq=30)
                    """

                result_add = seasonal_decompose(
                    y_, model='additive')

                fig2 = plt.rcParams.update({'figure.figsize': (10, 10)})
                # ax2 = fig2.add_subplot(142)
                fig2 = result_add.plot().suptitle('Additive Decompose', fontsize=22)
                # fig2.grid()
                st.pyplot()

                # Extract the Components ----
                # Actual Values = Product of (Seasonal + Trend + Resid)
                df2_reconstructed = pd.concat(
                    [result_add.seasonal, result_add.trend, result_add.resid, result_add.observed], axis=1)
                df2_reconstructed.columns = [
                    'seas', 'trend', 'resid', 'actual_values']
                st.write(df2_reconstructed)

            if method == "Prophet - Facebook":

                # prepare expected column names
                df2.columns = ['ds', 'y']
                df2['ds'] = pd.to_datetime(df2['ds'])
                # define the model
                model = Prophet()
                # fit the model

                model = Prophet(interval_width=UncInt)

                model.fit(df2)

                # define the period for which we want a prediction
                """
                    future = list()
                    for i in range(1, 13):
                        date = '1968-%02d' % i
                        future.append([date])
                    future = DataFrame(future)
                    future.columns = ['ds']
                    future['ds'] = to_datetime(future['ds'])
                    """

                st.subheader("""**Training**""")
                future = list()
                future = df2.iloc[:, 0]
                future = DataFrame(future)
                # future.columns = ['ds']
                # future['ds'] = to_datetime(future['ds'])
                forecast = model.predict(future)
                # st.write(forecast)

                # forecast = model.predict(df2.iloc[:, 0])
                # summarize the forecast
                # st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
                st.set_option('deprecation.showPyplotGlobalUse', False)
                # plot forecast
                fig, ax = plt.subplots()
                ax = model.plot(forecast)
                # st.write(ax)
                plt.xlabel('Date', fontsize=17)
                plt.ylabel(columSelectY, fontsize=17)
                plt.title('Time Series - Training', fontsize=20)
                st.pyplot()

                # df.sort_index(inplace=True)
                # Multiplicative Decomposition

                # calculate MAE between expected and predicted values for december
                y_true = df2['y'].values
                y_pred = forecast['yhat'].values

                col1, col2 = st.beta_columns(2)

                with col1:
                    mae = mean_absolute_error(y_true, y_pred)
                    st.success('Mean Absolute Error: %.4f' % mae)
                with col2:
                    mape = mean_absolute_percentage_error(y_true, y_pred)
                    st.success('Mean Absolute Percentage Error: %.3f' % mape)

                # plot expected vs actual
                # st.write(df2)
                dfPredChart = DataFrame()
                dfPredChart[columSelectX] = df2['ds']
                dfPredChart[columSelectY] = df2['y'].values
                dfPredChart['Predicted'] = forecast['yhat'].values

                dfPredChart.set_index(columSelectX, inplace=True)
                ax = dfPredChart[[columSelectY, 'Predicted']].plot(
                    figsize=(12, 8))
                ax.grid(b=True, which='major',
                        color='lightgrey', linestyle='-')
                st.pyplot()
                st.write(dfPredChart)

                """
                st.set_option('deprecation.showPyplotGlobalUse', False)
                # plt.plot(y_true, label=columSelectY)
                # plt.plot(y_pred, label='Predicted')
                # plt.xlabel('Date', fontsize=15)
                # plt.ylabel(columSelectY, fontsize=17)
                plt.legend()
                st.pyplot()
                """

                # define the period for which we want a prediction
                """
                future = list()
                for i in range(1, 13):
                    date = '1969-%02d' % i
                    future.append([date])
                future = DataFrame(future)
                future.columns = ['ds']
                future['ds'] = to_datetime(future['ds'])
                """

                st.subheader("""**Prediction**""")
                # use the model to make a forecast
                years = st.number_input(
                    'Select a period prediction', min_value=1, value=3)
                future = model.make_future_dataframe(
                    periods=years*12, freq='MS')
                forecast = model.predict(future)

                # summarize the forecast
                st.set_option('deprecation.showPyplotGlobalUse', False)
                # plot forecast
                fig, ax = plt.subplots()
                model.plot(forecast)
                plt.xlabel('Date', fontsize=17)
                plt.ylabel(columSelectY, fontsize=17)
                plt.title('Time Series - Prediction', fontsize=20)
                st.pyplot()
                st.subheader("""**Results**""")
                st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

                model.plot_components(forecast)
                st.pyplot()

                # plt.plot(forecast['Date'], forecast['furniture_trend'], 'b-')

            elif method == "SARIMA":

                st.subheader("""**Training**""")
                # Define the p, d and q parameters to take any value between 0 an.d 2
                p = d = q = range(0, 2)

                # Generate all different combinations of p, q and q triplets
                pdq = list(itertools.product(p, d, q))

                # Generate all different combinations of seasonal p, q and q triplets
                seasonal_pdq = [(x[0], x[1], x[2], 12)
                                for x in list(itertools.product(p, d, q))]

                """
                st.write('Examples of parameter combinations for Seasonal ARIMA...')
                st.write('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
                st.write('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
                st.write('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
                st.write('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
                """

                # specify to ignore warning messages
                warnings.filterwarnings("ignore")

                """
                future = list()
                for i in range(1, 13):
                    date = '1969-%02d' % i
                    future.append([date])
                future = DataFrame(future)
                future.columns = ['ds']
                future['ds'] = to_datetime(future['ds'])
                """

                paramList = list()
                aicList = list()
                param_seasonalList = list()

                # st.write("Aqui")
                # st.write(dfChart)
                # y = dfChart[columSelectY].resample('MS').mean()
                # y = dfChart[columSelectY]
                # furniture = furniture.set_index('Order Date')
                # furniture.index

                for param in pdq:
                    for param_seasonal in seasonal_pdq:
                        # try:
                        mod = sm.tsa.statespace.SARIMAX(y,
                                                        order=param,
                                                        seasonal_order=param_seasonal,
                                                        enforce_stationarity=False,
                                                        enforce_invertibility=False)

                        results = mod.fit()

                        paramList.append([param])

                        param_seasonalList.append([param_seasonal])
                        aicList.append([results.aic])

                        # st.write('ARIMA{}x{} - AIC:{}'.format(param,param_seasonal, results.aic))
                        # except:
                        # st.write("Error")
                        # continue

                # df3 = pd.DataFrame()
                df3 = pd.DataFrame(paramList)
                df3["Seasonal Component"] = pd.DataFrame(param_seasonalList)
                df3["AIC"] = pd.DataFrame(aicList)
                df3.columns = ['Parameters', 'Seasonal Component', 'AIC']

                df3_New = df3.sort_values(
                    by='AIC', ascending=True)

                st.write(df3_New)
                # st.write(df3_New.iloc[0, 0])
                # st.write(df3_New.iloc[0, 1])
                # st.write(df3_New.iloc[0, 2])

                # st.write(y)
                model = sm.tsa.statespace.SARIMAX(y,
                                                  order=df3_New.iloc[0, 0],
                                                  seasonal_order=df3_New.iloc[0, 1],
                                                  enforce_stationarity=False,
                                                  enforce_invertibility=False)

                results = model.fit()

                summary = pd.DataFrame(results.summary().tables[1])

                st.write(summary)
                results.plot_diagnostics(figsize=(16, 8))

                diagnostic = st.checkbox("Show Diagnostic Charts")
                if diagnostic:
                    results.plot_diagnostics(figsize=(15, 12))
                    st.pyplot()
                # pred = results.get_prediction(df.iloc[:, 0], dynamic=False)
                # pred = results.get_prediction(
                    # start=pd.to_datetime('2005-01-01'), dynamic=False)

                df['Predicted'] = results.predict(
                    start=0, end=len(df2)-1, dynamic=False)

                df[columSelectX] = pd.to_datetime(df[columSelectX])

                # st.write(df)
                """
                df[[columSelectY, 'forecast']].plot(figsize=(12, 8))
                st.pyplot()
                st.write(df)
                """

                # calculate MAE between expected and predicted values for december
                y_true = df[columSelectY].values
                y_pred = df['Predicted'].values

                col1, col2 = st.beta_columns(2)

                with col1:
                    mae = mean_absolute_error(y_true, y_pred)
                    st.success('Mean Absolute Error: %.4f' % mae)
                with col2:
                    mape = mean_absolute_percentage_error(y_true, y_pred)
                    st.success('Mean Absolute Percentage Error: %.3f' % mape)

                # Get the chart in date
                dfPredChart = DataFrame()
                dfPredChart[columSelectX] = pd.to_datetime(df[columSelectX])
                dfPredChart[columSelectY] = (df[columSelectY])
                dfPredChart['Predicted'] = results.predict(
                    start=0, end=len(df2)-1, dynamic=False)

                dfPredChart.set_index(columSelectX, inplace=True)
                ax = dfPredChart[[columSelectY, 'Predicted']].plot(
                    figsize=(12, 8))
                # plt.plot(dfPredChart[columSelectY], label=columSelectY)
                # plt.plot(dfPredChart['forecast'], label='Predicted')
                # plt.ylabel(columSelectY, fontsize=17)
                # plt.legend()
                plt.title('Time Series - Training', fontsize=20)
                ax.grid(b=True, which='major',
                        color='lightgray', linestyle='-')
                st.pyplot()
                st.write(dfPredChart)

                # st.write(df2)

                pred = results.get_prediction(
                    start=len(df2)-50-1, end=len(df2)-1, dynamic=False)
                pred_ci = pred.conf_int()

                pred_dynamic = results.get_prediction(
                    start=len(df2)-50-1, end=len(df2)-1, dynamic=True, full_results=True)
                pred_dynamic_ci = pred_dynamic.conf_int()

                # st.write(pred_ci)
                # st.write(pred_dynamic_ci)

                # Get forecast 3*12 steps ahead in future
                # pred_uc = results.get_forecast(steps=years*12)
                # pred_uc = results.get_forecast(steps=1*12)

                # Get confidence intervals of forecasts
                # pred_ci = pred_uc.conf_int()
                # st.write("pred_uc")

                # st.write(pred_ci)

                # date += datetime.timedelta(days=1)
                strn = df[columSelectX].iloc[-1]
                strn = max(df[columSelectX])
                df.set_index(columSelectX, inplace=True)
                st.write(df)

                st.subheader("""**Prediction**""")
                years = st.number_input(
                    'Select a period prediction (years)', min_value=1, value=3)

                pred_uc = results.get_forecast(steps=years*12)

                pred_ci = pred_uc.conf_int()

                date = to_datetime(strn)
                # date += timedelta(days=1)

                dateList = []
                for i in range(1, years*12+1):
                    date = date + relativedelta(months=+1)
                    dateList.append(date.date())

                dateList = np.array(dateList)
                # datelistdf = pd.DataFrame(dateList)
                pred_ci[columSelectX] = dateList
                pred_ci[columSelectX] = pd.to_datetime(pred_ci[columSelectX])

                pred_ci.set_index(columSelectX, inplace=True)
                pred_ci['Forecast'] = pred_ci.iloc[:, :].mean(axis=1)
                st.write(pred_ci)

                # datelistdf = pd.DataFrame(dateList, columns=[columSelectX])

                # dfPredChart[columSelectX] = dateList

                # future_df2 = pd.concat([df, datelistdf])

                """
                df['forecast'] = results.predict(start=90, end=103, dynamic=True)
                df[[columSelectY, 'forecast']].plot(figsize=(12, 8), legend=True)
                st.pyplot()
                """

                # Forecast for the next 1 years

                df["Forecast"] = results.predict(start=len(df)-1,
                                                 end=(len(df)-1) + years * 12,
                                                 typ='levels').rename('Forecast')

                """
                prueba = pd.DataFrame
                prueba = results.predict(start=len(df),
                                        end=(len(df)-1) + 1 * 12,
                                        typ='levels').rename('Forecast')
                st.write("prueba")
                st.write(prueba)
                """

                """

                # df["Forecast"] = forecast
                # Plot the forecast values
                df[[columSelectY, "Forecast"]].plot(figsize=(12, 5), legend=True)
                # forecast.plot(legend=True)
                st.pyplot()
                """
                # df.set_index(columSelectX, inplace=True)
                # st.write("Aqui Chart")
                # st.write(dfChart)
                fig, ax = plt.subplots()

                # ax.plot_date(x=df[columSelectX], y=df[columSelectY], marker='o')

                # dfChart.set_index(columSelectX, inplace=True)
                # dfChart[[columSelectY, 'Forecast']].plot(figsize=(12, 8))
                # ax = dfChart[[columSelectY]].plot(figsize=(12, 8))
                # ax.pred_ci[['Forecast']].plot(figsize=(12, 8))

                # dfChart.plot()

                ax.plot_date(dfChart.index, dfChart, '-')
                # ax.plot_date(dfChart.index, dfChart, marker='o')
                # ax = df[columSelectY].ax.plot_date(label=columSelectY, figsize=(20, 15))
                # pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
                ax.plot_date(
                    pred_ci.index, pred_ci["Forecast"], '-', color='dodgerblue')
                ax.fill_between(pred_ci.index,
                                pred_ci.iloc[:, 0],
                                pred_ci.iloc[:, 1], color='dodgerblue', alpha=.25)
                ax.set_xlabel('Date')
                ax.set_ylabel(columSelectY)
                ax.grid(b=True, which='major',
                        color='lightgrey', linestyle='-')
                # plt.grid(axis='both', linestyle='-', color='white')
                # plt.grid()
                plt.title('Time Series - Prediction', fontsize=20)
                plt.legend()
                st.pyplot()

                """
                # df.set_index(columSelectX, inplace=True)
                from pandas.tseries.offsets import DateOffset
                future_dates = [df.index[-1] +
                                DateOffset(months=x)for x in range(0, 24)]
                future_datest_df = pd.DataFrame(
                    index=future_dates[1:], columns=df.columns)

                future_datest_df.tail()

                future_df = pd.concat([df, prueba, future_datest_df])
                # future_df = pd.concat([df, prueba])
                # df['Forecast'] = prueba
                st.write(future_df)

                future_df['Forecast'] = results.get_forecast(steps=3*12)

                future_df[[columSelectY, 'Forecast']].plot(figsize=(12, 8))
                st.pyplot()
                """
            elif method == "Regressors":

                data = pd.DataFrame
                data = df2

                dateColum = data[columSelectX]
                # st.write(dateColum)

                # create 12 month moving average
                data['MA12'] = data[columSelectY].rolling(12).mean()
                # st.write(data)

                # plot the data and MA

                fig = px.line(data, x=columSelectX, y=[
                    columSelectY, "MA12"], template='plotly_dark')
                # fig = px.line(df2, x=columSelectX, y=[columSelectY, "MA12"])
                # fig.show()
                st.plotly_chart(fig)

                # extract month and year from dates
                month = [i.month for i in data[columSelectX]]

                year = [i.year for i in data[columSelectX]]

                # create a sequence of numbers
                data['Series'] = np.arange(1, len(data)+1)
                # drop unnecessary columns and re-arrange
                data.drop([columSelectX, 'MA12'], axis=1, inplace=True)
                data['Month'] = month
                data['Year'] = year
                data = data[['Series', 'Year', 'Month', columSelectY]]
                # check the head of the dataset

                yearSplit = st.number_input(
                    'Select a year for data split', min_value=min(year)+1, max_value=max(year)-1, value=max(year)-2)

                # split data into train-test set
                train = data[data['Year'] < yearSplit]
                test = data[data['Year'] >= yearSplit]

                # check shape

                col1, col2, col3, col4 = st.beta_columns(4)

                with col1:
                    st.write("Shape of dataset:")
                    st.info(data.shape)
                with col2:
                    st.write("Complete data:")
                    st.info(len(data))
                with col3:
                    st.write("Data to train:")
                    st.info(len(train))

                with col4:
                    st.write("Data to test:")
                    st.info(len(test))

                # st.write(data)

                X_data = data[['Series', 'Year', 'Month']]
                y_data = data[columSelectY]

                X_train = train[['Series', 'Year', 'Month']]
                # X_train = train[['Series', 'Year']]
                y_train = train[columSelectY]

                X_test = test[['Series', 'Year', 'Month']]
                # X_test = test[['Series', 'Year']]
                y_test = test[columSelectY]

                # kf=KFold(n_splits=3,shuffle=True,random_state=0)
                cv = TimeSeriesSplit(n_splits=10)

                # for train, test in enumerate(cv.split(X, y)):

                # Linear Regression
                regressor = LinearRegression(n_jobs=-1)

                # regressor = Lars(random_state=0)
                # regressor = DecisionTreeRegressor(random_state=0)

                # Random Forest Regressor
                # regressor = RandomForestRegressor(
                # n_estimators=100, random_state=0, n_jobs=-1)

                # Gradient Boosting Regressor
                # regressor = GradientBoostingRegressor(random_state=0)

                # Extreme Gradient Boosting Regressor
                # regressor = XGBRegressor(n_jobs=-1, random_state=0)

                # Light Gradient Boosting Machine Regressor
                # regressor = LGBMRegressor(random_state=0)

                # CatBoost Regressor
                # regressor = CatBoostRegressor(random_state=0)

                # AdaBoost Regressor
                # regressor = AdaBoostRegressor(random_state=0)

                # Extra Trees Regressor
                # regressor = ExtraTreeRegressor(random_state=0)

                # Lasso
                # regressor = Lasso(random_state=0)

                # Lasso Least Angle Regression
                # regressor = LassoLars(random_state=0)

                # K Nearest Neighbors Regressor
                # regressor = KNeighborsRegressor(n_jobs=-1)

                # regressor.fit(X_train, y_train)
                regressor.fit(X_train, y_train)

                y_pred = regressor.predict(X_data)

                coefR2 = round(r2_score(y_data, y_pred), 4)

                MAE = round(mean_absolute_error(y_data, y_pred), 2)
                # st.write(y_pred)
                # coefR2 = round(r2_score(y_test, y_pred), 4)
                # MAE = round(mean_absolute_error(y_test, y_pred), 2)

                data['Forecast'] = y_pred

                # st.write(coefR2)
                # st.write(MAE)

                dfForecast = pd.DataFrame(dateColum)
                dfForecast[columSelectY] = data[columSelectY]
                dfForecast['Forecast'] = data['Forecast']
                st.write(dfForecast)

                fig = px.line(dfForecast, x=columSelectX, y=[
                    columSelectY, 'Forecast'], template='plotly_dark')
                # fig = px.line(df2, x=columSelectX, y=[columSelectY, "MA12"])
                # fig.show()
                st.plotly_chart(fig)

                # st.write(max(year))

        else:
            st.info('Awaiting for CSV file to be uploaded.')
    except:
        st.info('ERROR - Please check your Dataset, parameters o selected model')

    return
