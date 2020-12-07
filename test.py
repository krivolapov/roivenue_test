# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 13:39:26 2020

@author: Max
"""

import pandas as pd
df = pd.read_csv('input/data.csv',encoding = 'unicode_escape')

# calculate totall price for items
df['TotalPrice'] = df['UnitPrice'] * df['Quantity']
# convert into tatetime format
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
# add Month-Year format for charts
df['Mon_Year'] = df['InvoiceDate'].dt.strftime('%b-%Y')

df.drop('StockCode', axis=1, inplace=True)
df.drop('Description', axis=1, inplace=True)
df.drop('Country', axis=1, inplace=True)

df_total_price = df.groupby(['InvoiceNo','InvoiceDate','CustomerID','Mon_Year' ], as_index=False)[['TotalPrice']].sum() 

df_total_price_pos = df_total_price[df_total_price['TotalPrice'] > 0]
df_total_price_neg = df_total_price[df_total_price['TotalPrice'] < 0]

group = pd.DataFrame({'count' : df_total_price_pos.groupby( [ "InvoiceDate", "CustomerID"] ).size()}).reset_index()