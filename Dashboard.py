import pandas as pd
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Output, Input
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas_datareader as web
import multiprocessing as mp
import time as tm
import datetime as dt

start_date='2017-08-18'
end_date=dt.date.today()
tickers=['SPY','AMZN','GOOGL','TSLA','AMD']

df = web.get_data_yahoo(tickers, start = start_date, end = end_date)['Adj Close']
Vol = web.get_data_yahoo(tickers, start = start_date, end = end_date)['Volume']
print(f'Downloaded {df.shape[0]} rows of data.')

Vol1= Vol.stack().reset_index()
Vol1
Vol2 = Vol1.rename(columns={0:'Volume'}).drop(columns=['Date','Symbols'])
Vol2

df_ret = df.pct_change().fillna(0)
df_ret1 = df_ret.stack().reset_index()
df_ret2=df_ret1.rename(columns={0:'Rets'}).drop(columns=['Date','Symbols'])
df_ret2

df1 = df.stack().reset_index()
df2 = df1.rename(columns={0:'Price'})

df_total = pd.concat([df2,df_ret2,Vol2],axis=1)
df_total

newframe = pd.DataFrame.copy(df)

def statistics(df):
    for i in df:
        df[i+'daily_rets'] = df[i].pct_change()
        df[i+'momentum'] = df[i]/df[i].shift(1)
        df[i+'21_days_MA'] = df[i].rolling(window=21).mean()
        df[i+'difference']=df[i].diff()


t = tm.time()
statistics(newframe)
print(tm.time()-t)

t1 = tm.time()
p = mp.Process(target = statistics, args =(newframe))
p.start()
print(tm.time()-t1)

newframe.to_csv('my_stocks')

update_price = web.get_data_yahoo(tickers, start = dt.date.today(), end = dt.date.today())['Adj Close']
update_price.to_csv('my_stocks.csv')


def correlation_from_covariance(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation

fig_corr = px.imshow(correlation_from_covariance(df.pct_change().dropna().cov()))

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUMEN],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}])
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Stage1 Dashboard",
                        className='text-center text-primary, mb-3'),
                width=12)
    ]),
    dbc.Row([
        dbc.Col([dcc.Dropdown(id='my-dpdn', multi=False, value='AMZN',
                              options=[{'label': x, 'value': x}
                                       for x in sorted(df1['Symbols'].unique())]),
                 dcc.Graph(id='line-fig', figure={})
                 ], width={'size': 5, 'offset': 1, 'order': 1}),
        dbc.Col([dcc.Dropdown(id='my-dpdn2', multi=True, value=tickers,
                              options=[{'label': x, 'value': x}
                                       for x in sorted(df1['Symbols'].unique())]),
                 dcc.Graph(id='line-fig2', figure={})
                 ], width={'size': 5, 'offset': 1, 'order': 2}),
    ]),
    dbc.Row([
        dbc.Col([
            html.P('Select Company Stock:',
                   style={'textDecoration': 'underline'}),
            dcc.Checklist(id='my-checklist', value=tickers,
                          options=[{'label': x, 'value': x}
                                   for x in sorted(df1['Symbols'].unique())],
                          labelClassName='mr-3 text-success'),
            dcc.Graph(id='my-hist', figure={})

        ], width={'size': 5, 'offset': 1}),
        dbc.Col([
            html.P('Select Company Stock',
                   style={'textDecoration': 'underline'}),
            dcc.Checklist(id='my-checklist2', value=['SPY','AMZN'],
                          options=[{'label': x, 'value': x}
                                   for x in sorted(df1['Symbols'].unique())],
                          labelClassName='mr-3 text-success'),
            dcc.Graph(id='my-bar', figure={})

        ], width={'size': 5, 'offset': 1}),

    ]),
    dbc.Row([
        html.H1("Correlation Heatmap"),
        dcc.Graph(figure=fig_corr)
    ])


])


# Line chart - Single
@app.callback(
    Output('line-fig', 'figure'),
    Input('my-dpdn', 'value')
)
def update_graph(stock_slctd):
    dff = df_total[df_total['Symbols']==stock_slctd]
    figln = px.line(dff, x='Date', y='Price')
    return figln

# Line chart - multiple
@app.callback(
    Output('line-fig2', 'figure'),
    Input('my-dpdn2', 'value')
)
def update_graph(stock_slctd):
    dff = df_total[df_total['Symbols'].isin(stock_slctd)]
    figln2 = px.line(dff, x='Date', y='Rets', color='Symbols')
    return figln2


# Histogram
@app.callback(
    Output('my-hist', 'figure'),
    Input('my-checklist', 'value')
)
def update_graph(stock_slctd):
    dff = df_total[df_total['Symbols'].isin(stock_slctd)]
    dff = dff[dff['Date']=='2022-08-18']
    fighist = px.histogram(dff, x='Symbols', y='Volume')
    return fighist


#scatterplot
@app.callback(
    Output('my-bar', 'figure'),
    Input('my-checklist2', 'value')
)
def update_graph(stock_slctd):
    dff = df_total[df_total['Symbols'].isin(stock_slctd)]
    fighist = px.scatter(dff, x=df.SPY.pct_change().fillna(0),
                         y=df.AMZN.pct_change().fillna(0)).update_layout(
                             xaxis_title="SPY Returns", yaxis_title="AMZN Returns")
    return fighist


print(df1['Symbols'])
if __name__ == '__main__':
    app.run_server(debug=True, port=8000)
