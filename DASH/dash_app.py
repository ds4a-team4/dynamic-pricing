import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import pandas as pd

import torch
import torch.nn as nn
import pickle
import numpy as np
import datetime
import configparser

from sqlalchemy import create_engine


class Environment:
    """
    Simulates an environment for the RL agent
    Init arguments:
    model: a model to predict demand as a function of price and other
           covariates (see the function predict_demand)
    df: dataframe with historical data, including the competitors'
        prices

    Properties:
    self.model: the model used to predict demand
    self.data: the dataframe with historical data
    self.N: max number os iterations considering the length of self.data
    self.t: indicates the current time, starting at 0
    self.done: indicates that the simulation is done, i.e. the simulation
               has been run on the entire self.data dataframe
    self.orders: the number of orders predicted for the current time instant
    self.olist_price: the price selected due to and action

    Methods:
    reset(self): resets the simulation to the initial conditions
    step(self, act): performs a step in the simulation considering
                     an action ranging from 0 to 9 
                     (choice of a price)
    )
    """

    def __init__(self, model, df):

        self.model = model
        self.data = df
        self.N = len(self.data) - 1
        self.reset()

    def reset(self):
        self.t = 0
        self.done = False
        self.orders = 0
        self.olist_price = 0
        self.profits = 0
        return [self.olist_price, self.orders] + self.data.iloc[self.t].tolist()

    def step(self, act):
        if act == 0:
            self.olist_price = self.data['base_cost'][self.t] * 1.05
        elif act == 1:
            self.olist_price = self.data['base_cost'][self.t] * 1.075
        elif act == 2:
            self.olist_price = self.data['base_cost'][self.t] * 1.10
        elif act == 3:
            self.olist_price = self.data['base_cost'][self.t] * 1.125
        elif act == 4:
            self.olist_price = self.data['base_cost'][self.t] * 1.15
        elif act == 5:
            self.olist_price = self.data['base_cost'][self.t] * 1.175
        elif act == 6:
            self.olist_price = self.data['base_cost'][self.t] * 1.20
        elif act == 7:
            self.olist_price = self.data['base_cost'][self.t] * 1.225
        elif act == 8:
            self.olist_price = self.data['base_cost'][self.t] * 1.25
        elif act == 9:
            self.olist_price = self.data['base_cost'][self.t] * 1.275

        # Calculate demand
        self.orders = predict_demand(
            self.model, self.data.iloc[self.t], self.olist_price)

        reward = (self.olist_price + self.data['freight_value']
                  [self.t] - self.data['base_cost'][self.t])*self.orders
        self.profits += reward

        # set next time
        self.t += 1

        if (self.t == self.N):
            self.done = True

        # obs, reward, done
        return [self.olist_price, self.orders] + self.data.iloc[self.t].tolist(), reward, self.done


class Q_Network(nn.Module):
    """
    Neural Network architecture for the agent
    Depends on:
    obs_len: length of the "observation" vector provided by the environment,
             which is the input of the network
    hidden_size: number of neurons used for the 4 hidden layers
    actions_n: length of the output of the network, consisting of
               Q-values for each possible action
    """

    def __init__(self, obs_len, hidden_size, actions_n):

        super(Q_Network, self).__init__()
        self.fc_val = nn.Sequential(
            nn.BatchNorm1d(num_features=obs_len),
            nn.Linear(obs_len, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.Linear(hidden_size, actions_n),
        )

    def forward(self, x):
        h = self.fc_val(x)
        return (h)


def predict_demand(model, df_row, olist_price):
    """
    Auxiliary function to use the model to obtain the predicted
    demand, given a price and other covariates on a dataframe row
    """

    year = df_row.year
    month = df_row.month
    dayofweek = df_row.dayofweek
    day = df_row.day
    olist_price = olist_price
    freight_value = df_row.freight_value
    competition_price = df_row.competition_price
    stock = df_row.stock
    black_friday = df_row.black_friday
    carnival = df_row.carnival
    christmas = df_row.christmas
    friday = df_row.friday
    mothers_day = df_row.mothers_day
    new_year = df_row.new_year
    others = df_row.others
    valentines = df_row.valentines

    X = np.array([year, month, dayofweek, day, olist_price, freight_value,
                  competition_price, stock, black_friday, carnival, christmas,
                  friday, mothers_day, new_year, others, valentines]).reshape(1, -1)

    #X = xgboost.DMatrix(X)

    orders = model.predict(X)

    return max(orders[0], 0)


# Parameters of the trained NN
hidden_size = 30
input_size = 2 + 16
output_size = 10

# Loading of the trained NN parameters
Q = Q_Network(input_size, hidden_size, output_size)
Q.load_state_dict(torch.load('./Q_state.torch',
                             map_location=torch.device('cpu')))

# Loading of the model for predicting demand
with open('./lr_cellphone_C.pkl', 'rb') as f:
    model = pickle.load(f)

# Loading of preprocessed data for with results for the entire period
#df = pd.read_csv('./cellphones_final_results.csv')

config = configparser.ConfigParser()
config.read('rds.conf')
uri = config.get('rds', 'uri')
engine = create_engine(uri)
df = pd.read_sql_table("rl_results", con=engine)

############ DASH APP ##############

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.SPACELAB],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)


server = app.server
app.title = 'Dynamic Pricing'

app.layout = dbc.Container(
    html.Div(
        [
            html.Div(
                [
                    html.Img(src=app.get_asset_url('logo.png'),
                             style={'height': '15%', 'width': '20%'})
                ],
                style={
                    'display': 'inline-block',
                    'text-align': 'center'
                }
            ),


            html.Div(
                [
                    html.Div(
                        [
                            html.H5('Product Group'),
                            dcc.Dropdown(
                                id='selector_group',
                                options=[{'label': str(item),
                                          'value': str(item)}
                                         for item in set(df['group'])],
                                multi=True,
                                value=list(set(df['group']))
                            )
                        ],
                        className='col s4 m4 l4'
                    ),
                    html.Div(
                        [
                            html.H5('Product Type'),
                            dcc.Dropdown(
                                id='selector_type',
                                options=[{'label': str(item),
                                          'value': str(item)}
                                         for item in set(df['type'])],
                                multi=True,
                                value=list(set(df['type']))
                            )
                        ],
                        className='col s4 m4 l4'
                    ),
                    html.Div(
                        [
                            html.H5('Product Price Category'),
                            dcc.Dropdown(
                                id='selector_price_range',
                                options=[{'label': str(item),
                                          'value': str(item)}
                                         for item in set(df['price_range'])],
                                multi=True,
                                value=list(set(df['price_range']))
                            )
                        ],
                        className='col s4 m4 l4'
                    )

                ], className="row",
                style={
                    'margin-top': '15px',
                    'margin-bottom': '20px'
                }
            ),

            html.Div(
                [

                    html.Div(
                        [
                            html.H5('Baseline'),
                            html.Div([
                                html.P(
                                    f'R$ {round(df.baseline_rewards.sum(),2)}')
                            ],
                                style={
                                    'color': '#0DC78B'
                            })
                        ],
                        className='col s4 m4 l4'
                    ),
                    html.Div(
                        [
                            html.H5('Dynamic Pricing'),
                            html.Div([
                                html.P(f'R$ {round(df.rl_rewards.sum(),2)}')
                            ],
                                style={
                                    'color': '#0DC78B'
                            })
                        ],
                        className='col s4 m4 l4'
                    ),
                    html.Div(
                        [
                            html.H5('Improvement'),
                            html.Div([
                                html.P(
                                    f'{round((df.rl_rewards.sum() / df.baseline_rewards.sum() - 1) * 100, 2)} %'
                                )
                            ],
                                style={
                                    'color': '#0DC78B'
                            })
                        ],
                        className='col s4 m4 l4'
                    )

                ], className="row",
                style={
                    'margin-top': '10px',
                }
            ),


            html.Div(
                [
                    html.Div(
                        [
                            dcc.Graph(id='graph1'
                                      )
                        ], className='col s12 m12 l12'
                    )
                ], className="row",
                style={
                    'margin-bottom': '55px'
                }
            ),

            html.Hr(),  # horizontal line

            html.Div(
                [
                    html.H3
                    (
                        children='Dynamic Pricing Prediction',
                        style={
                            'text-align': 'center',
                            'margin-top': '55px',
                            'margin-bottom': '15px',
                            'color': '#0C29D0'
                        }
                    ),
                    html.P
                    (
                        children='change the parameters and see the price suggestion in real time',
                        style={
                            'text-align': 'center',
                            'margin-bottom': '45px'
                        }
                    )
                ]
            ),

            html.Div(
                [
                    html.Div(
                        [
                            html.H5('Date'),
                            dcc.Input(
                                id='date_input',
                                value=datetime.date.today().strftime('%m/%d/%Y'),
                                style={
                                    'color': '#0C29D0'
                                }
                            ),
                            html.Div(
                                [html.P('Invalid value')],
                                style={'display': 'none'},
                                id="date_invalid"
                            )
                        ],
                        className='col s4 m4 l4'
                    ),
                    html.Div(
                        [
                            html.H5('Stock Quantity'),
                            dcc.Input(
                                id='stock_input',
                                value='1',
                                style={
                                    'color': '#0C29D0'
                                }
                            ),
                            html.Div(
                                [html.P('Invalid value')],
                                style={'display': 'none'},
                                id="stock_invalid"
                            )
                        ],
                        className='col s4 m4 l4'
                    ),
                    html.Div(
                        [
                            html.H5('Cost'),
                            dcc.Input(
                                id='base_cost_input',
                                value='718',
                                style={
                                    'color': '#0C29D0'
                                }
                            ),
                            html.Div(
                                [html.P('Invalid value')],
                                style={'display': 'none'},
                                id="base_cost_invalid"
                            )
                        ],
                        className='col s4 m4 l4'
                    )

                ], className="row",
                style={
                    'margin-top': '15px',
                    'margin-bottom': '20px'
                }
            ),

            html.Div(
                [
                    html.Div(
                        [
                            html.H5('Freight Value'),
                            dcc.Input(
                                id='freight_value_input',
                                value='25',
                                style={
                                    'color': '#0C29D0'
                                }
                            ),
                            html.Div(
                                [html.P('Invalid value')],
                                style={'display': 'none'},
                                id="freight_value_invalid"
                            )

                        ],
                        className='col s4 m4 l4'
                    ),
                    html.Div(
                        [
                            html.H5('Competitor\'s Price'),
                            dcc.Input(
                                id='competition_price_input',
                                value='898',
                                style={
                                    'color': '#0C29D0'
                                }
                            ),
                            html.Div(
                                [html.P('Invalid value')],
                                style={'display': 'none'},
                                id="competition_price_invalid"
                            )
                        ],
                        className='col s4 m4 l4'
                    ),
                    html.Div(
                        [
                            html.Div([
                                html.Div(id="message_output")
                            ],
                                style={
                                    'color': '#0DC78B'
                            })

                        ],
                        className='col s4 m4 l4'
                    )
                ], className="row",
                style={
                    'margin-top': '15px',
                    'margin-bottom': '20px'
                }
            ),


            html.Div(
                [

                    html.Div(
                        [
                            html.H5('Suggested Price'),
                            html.Div([
                                html.Div(id="o_price_output")
                            ],
                                style={
                                    'color': '#0DC78B'
                            })
                        ],
                        className='col s4 m4 l4'
                    ),
                    html.Div(
                        [
                            html.H5('Predicted Orders (Day after)'),
                            html.Div([
                                html.Div(id="orders_output")
                            ],
                                style={
                                    'color': '#0DC78B'
                            })
                        ],
                        className='col s4 m4 l4'
                    ),
                    html.Div(
                        [
                            html.H5('Predicted Profit (Day after)'),
                            html.Div([
                                html.Div(id="profit_output")
                            ],
                                style={
                                    'color': '#0DC78B'
                            })
                        ],
                        className='col s4 m4 l4'
                    )

                ], className="row",
                style={
                    'margin-top': '15px',
                }
            ),

            html.Div(
                [
                    html.P
                    (
                        children='Powered by Team4 - Brazil - DS4A',
                        style={
                            'text-align': 'center',
                            'padding-top': '150px'
                        }
                    )
                ]
            )
        ],
        style={
            'margin-bottom': '55px',
        }
    )
)


@app.callback(
    Output(component_id='graph1', component_property='figure'),
    [
        Input(component_id='selector_group', component_property='value'),
        Input(component_id='selector_type', component_property='value'),
        Input(component_id='selector_price_range', component_property='value')
    ]
)
def update_graph(selector_group, selector_type, selector_price_range):
    """
    Callback to update plot according to the selected product category
    In the current version, it only works with electronics - cellphones - C price category
    """
    data = []
    if 'electronics' in selector_group and'cellphones' in selector_type and 'C' in selector_price_range:
        data.append({'x': df.index, 'y': df.rl_rewards, 'type': 'line',
                     'name': 'Dynamic Pricing Profits', 'line': dict(color='#EDAD00')})
        data.append({'x': df.index, 'y': df.baseline_rewards, 'type': 'line',
                     'name': 'Baseline Profits', 'line': dict(color='#6A00A3')})
    figure = {
        'data': data,
        'layout': {
            'title': 'Baseline x Reinforcement Learning',
            'xaxis': dict(
                title='TIME',
                titlefont=dict(
                    family='Helvetica, monospace',
                    size=14,
                    color='#312F4F'
                )),
            'yaxis': dict(
                title='PROFITS',
                titlefont=dict(
                    family='Helvetica, monospace',
                    size=14,
                    color='#312F4F'
                ))
        }
    }
    return figure


@app.callback(
    [
        Output("orders_output", "children"),
        Output("o_price_output", "children"),
        Output("profit_output", "children"),
        Output("message_output", "children"),
        Output("date_invalid", "style"),
        Output("freight_value_invalid", "style"),
        Output("competition_price_invalid", "style"),
        Output("stock_invalid", "style"),
        Output("base_cost_invalid", "style"),
    ],
    [
        Input("date_input", "value"),
        Input("freight_value_input", "value"),
        Input("competition_price_input", "value"),
        Input("stock_input", "value"),
        Input("base_cost_input", "value"),
    ],
)
def update_output(date_input, freight_value_input, competition_price_input,
                  stock_input, base_cost_input):
    """
    Callback to update the results fields (Suggested Price, Predicted Orders
    and predicted profits), using the RL agent and the demand model with the 
    input data provided (date, stocks, cost, shipping value and competitors' 
    price)
    """

    # Input Data validation
    try:
        date = datetime.datetime.strptime(date_input, '%m/%d/%Y')
        date_invalid = {'display': 'none'}
    except:
        date = None
        date_invalid = {'display': 'block', 'color': '#FF0000'}

    try:
        freight_value = float(freight_value_input)
        freight_value_invalid = {'display': 'none'}
    except:
        freight_value = None
        freight_value_invalid = {'display': 'block', 'color': '#FF0000'}

    try:
        competition_price = float(competition_price_input)
        competition_price_invalid = {'display': 'none'}
    except:
        competition_price = None
        competition_price_invalid = {'display': 'block', 'color': '#FF0000'}

    try:
        stock = float(stock_input)
        stock_invalid = {'display': 'none'}
    except:
        stock = None
        stock_invalid = {'display': 'block', 'color': '#FF0000'}

    try:
        base_cost = float(base_cost_input)
        base_cost_invalid = {'display': 'none'}
    except:
        base_cost = None
        base_cost_invalid = {'display': 'block', 'color': '#FF0000'}

    # RL Agent
    if date and freight_value and competition_price and stock and base_cost:
        year = date.year
        month = date.month
        dayofweek = date.weekday()
        day = date.day
        friday = 1 if dayofweek == 4 else 0
        black_friday = 0
        carnival = 0
        christmas = 1 if (day == 25 and month == 12) else 0
        mothers_day = 1 if ((8 <= day <= 14) and (
            month == 5) and (dayofweek == 6)) else 0
        new_year = 1 if (day == 1 and month == 1) else 0
        others = 0
        valentines = 1 if (day == 12 and month == 6) else 0

        date_next = date + datetime.timedelta(days=1)
        year_next = date_next.year
        month_next = date_next.month
        dayofweek_next = date_next.weekday()
        day_next = date_next.day
        friday_next = 1 if dayofweek_next == 4 else 0

        df_dict = {
            'year': [year, year_next],
            'month': [month, month_next],
            'dayofweek': [dayofweek, dayofweek_next],
            'day': [day, day_next],
            'freight_value': [freight_value, freight_value],
            'competition_price': [competition_price, competition_price],
            'stock': [stock, stock],
            'black_friday': [black_friday, black_friday],
            'carnival': [carnival, carnival],
            'christmas': [christmas, christmas],
            'friday': [friday, friday_next],
            'mothers_day': [mothers_day, mothers_day],
            'new_year': [new_year, new_year],
            'others': [others, others],
            'valentines': [valentines, valentines],
            'base_cost': [base_cost, base_cost]
        }

        df_rl = pd.DataFrame.from_dict(df_dict)

        test_env = Environment(model, df_rl)
        pobs = test_env.reset()
        pact_history = []
        done = False

        try:
            Q.eval()
            pact = Q(torch.from_numpy(
                np.array(pobs, dtype=np.float32).reshape(1, -1)))
            pact = np.argmax(pact.data.cpu())
            pact_history.append(pact)
            obs, reward, done = test_env.step(pact.numpy())
            orders = obs[1]
            o_price = obs[0]
            profit = reward
            message_output = ''
        except:
            orders = 0
            o_price = 0
            profit = 0
            message_output = 'Something went wrong'
    else:
        orders = 0
        o_price = 0
        profit = 0
        message_output = ''

    orders_output = str(round(orders, 0))
    o_price_output = 'R$ '+str(round(o_price, 2))
    profit_output = 'R$ '+str(round(profit, 2))

    output = (
        orders_output,
        o_price_output,
        profit_output,
        message_output,
        date_invalid,
        freight_value_invalid,
        competition_price_invalid,
        stock_invalid,
        base_cost_invalid,
    )
    return output


if __name__ == '__main__':
    app.run_server(debug=True)
