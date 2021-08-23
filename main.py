import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import yfinance as yf # https://pypi.org/project/yfinance/
from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.momentum import RSIIndicator
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split



option = st.sidebar.selectbox('Select one symbol', ( 'TCS', 'BTC-USD','ETH'))
import datetime
today = datetime.date.today()
before = today - datetime.timedelta(days=700)
start_date = st.sidebar.date_input('Start date', before)
end_date = st.sidebar.date_input('End date', today)
if start_date < end_date:
    st.sidebar.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
else:
    st.sidebar.error('Error: End date must fall after start date.')

df = yf.download(option,start= start_date,end= end_date, progress=False)

#data pridiction
x = df.loc[:,'High':'Volume']
y = df.loc[:,'Open']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.8,random_state = 0)
LR = LinearRegression()
LR.fit(x_train,y_train)
LR.score(x_test,y_test)
Test_data = x_test
prediction = LR.predict(Test_data)





# Bollinger Bands
indicator_bb = BollingerBands(df['Close'])
bb = df
bb['bb_upper'] = indicator_bb.bollinger_hband()
bb['bb_lower'] = indicator_bb.bollinger_lband()
bb = bb[['Close','bb_upper','bb_lower']]

# Moving Average Convergence Divergence
macd = MACD(df['Close']).macd()

# Resistence Strength Indicator
rsi = RSIIndicator(df['Close']).rsi()


# Set up main app #

# Plot the prices and the bolinger bands
st.write('Stock Bollinger Bands')
st.line_chart(bb)

progress_bar = st.progress(0)

# Plot MACD
st.write('Stock Moving Average Convergence Divergence (MACD)')
st.area_chart(macd)

# Plot RSI
st.write('Stock RSI ')
st.line_chart(rsi)

st.write('Recent data ')
st.dataframe(df.tail(10))
from io import *
import base64

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = to_excel(df)
    b64 = base64.b64encode(val) 
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="download.xlsx">Download excel file</a>' 

st.markdown(get_table_download_link(df), unsafe_allow_html=True)
