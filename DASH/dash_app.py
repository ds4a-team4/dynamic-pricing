import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import pandas as pd
from datetime import datetime as dt

import torch
import torch.nn as nn
import pickle
import numpy as np
import datetime


class Environment:

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

        # act = 0: stay, 1: raise, 2: lower
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


hidden_size = 30
input_size = 2 + 16
output_size = 10

Q = Q_Network(input_size, hidden_size, output_size)
Q.load_state_dict(torch.load('./Q_state.torch',
                             map_location=torch.device('cpu')))

with open('./lr_cellphone_C.pkl', 'rb') as f:
    model = pickle.load(f)

df = pd.read_csv('./cellphones_final_results.csv')

# app = dash.Dash()
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SPACELAB])
server = app.server
app.title = 'DS4A'

# Boostrap CSS.
# app.css.append_css({'external_url': 'https://codepen.io/amyoshino/pen/jzXypZ.css'})

# Materialize CSS
# external_css = ["https://cdnjs.cloudflare.com/ajax/libs/materialize/0.100.2/css/materialize.min.css"]
# for css in external_css:
#     app.css.append_css({"external_url": css})

# image1 = 'assets/cloud2.jpeg' # replace with your own image
# encoded_image = base64.b64encode(open(image_filename, 'rb').read())

app.layout = dbc.Container(
    html.Div(
        [
            html.Div(
                [
                    html.H1
                    (
                        children='Olist Dynamic Pricing',
                        style={
                            'text-align': 'center',
                            'margin-top': '15px',
                            'margin-bottom': '45px',
                            'color': '#0C29D0',
                            'background-image': 'url("/assets/background-cloud.jpg")'
                        }
                    )
                ]
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
                    #                     html.Div(
                    #                         [   html.H5('Date Range'),
                    #                             dcc.DatePickerRange(
                    #                                 id='my-date-picker-range',
                    #                                 min_date_allowed=dt(df.year.min(), 8, 5),
                    #                                 max_date_allowed=dt(2017, 9, 19),
                    #                                 initial_visible_month=dt(2017, 8, 5),
                    #                                 end_date=dt(2017, 8, 25).date()
                    #                             ),
                    #                             html.Div(id='output-container-date-picker-range')
                    #                         ],
                    #                         className='col s6 m6 l6'
                    #                     )

                ], className="row",
                style={
                    'margin-top': '15px',
                    'margin-bottom': '15px'
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
                            html.H5('RL'),
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
                            html.H5('DELTA'),
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
                    'margin-top': '15px',
                    #                                 'margin-bottom': '15px'
                }
            ),


            html.Div(
                [
                    html.Div([
                        dcc.Graph(id='graph1'
                                  )
                    ], className='col s12 m12 l12'
                    )

                ], className="row"
            ),


            html.Div(
                [
                    html.Div(
                        [
                            html.H5('Date'),
                            dcc.Input(
                                id='date_input',
                                value=datetime.date.today().strftime('%m/%d/%Y')
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
                            html.H5('Freight Value'),
                            dcc.Input(
                                id='freight_value_input',
                                value='25'
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
                                value='898'
                            ),
                            html.Div(
                                [html.P('Invalid value')],
                                style={'display': 'none'},
                                id="competition_price_invalid"
                            )
                        ],
                        className='col s4 m4 l4'
                    )
                    #                     html.Div(
                    #                         [   html.H5('Date Range'),
                    #                             dcc.DatePickerRange(
                    #                                 id='my-date-picker-range',
                    #                                 min_date_allowed=dt(df.year.min(), 8, 5),
                    #                                 max_date_allowed=dt(2017, 9, 19),
                    #                                 initial_visible_month=dt(2017, 8, 5),
                    #                                 end_date=dt(2017, 8, 25).date()
                    #                             ),
                    #                             html.Div(id='output-container-date-picker-range')
                    #                         ],
                    #                         className='col s6 m6 l6'
                    #                     )

                ], className="row",
                style={
                    'margin-top': '15px',
                    'margin-bottom': '15px'
                }
            ),

            html.Div(
                [
                    html.Div(
                        [
                            html.H5('Stock Quantity'),
                            dcc.Input(
                                id='stock_input',
                                value='1'
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
                                value='718'
                            ),
                            html.Div(
                                [html.P('Invalid value')],
                                style={'display': 'none'},
                                id="base_cost_invalid"
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
                    #                     html.Div(
                    #                         [   html.H5('Date Range'),
                    #                             dcc.DatePickerRange(
                    #                                 id='my-date-picker-range',
                    #                                 min_date_allowed=dt(df.year.min(), 8, 5),
                    #                                 max_date_allowed=dt(2017, 9, 19),
                    #                                 initial_visible_month=dt(2017, 8, 5),
                    #                                 end_date=dt(2017, 8, 25).date()
                    #                             ),
                    #                             html.Div(id='output-container-date-picker-range')
                    #                         ],
                    #                         className='col s6 m6 l6'
                    #                     )

                ], className="row",
                style={
                    'margin-top': '15px',
                    'margin-bottom': '15px'
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
                    #                                 'margin-bottom': '15px'
                }
            ),

        ]
    )
)


@app.callback(
    Output(component_id='graph1', component_property='figure'),
    [Input(component_id='selector_group', component_property='value')])
def update_graph(selector):
    data = []
    if 'electronics' in selector:
        data.append({'x': df.index, 'y': df.rl_rewards, 'type': 'line',
                     'name': 'RL Profits', 'line': dict(color='#EDAD00')})
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
                    size=20,
                    color='#312F4F'
                )),
            'yaxis': dict(
                title='PROFITS',
                titlefont=dict(
                    family='Helvetica, monospace',
                    size=20,
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

    if date and freight_value and competition_price and stock and base_cost:
        year = date.year
        month = date.month
        dayofweek = date.weekday()
        day = date.day
        friday = 1 if dayofweek == 4 else 0
        black_friday = 0
        carnival = 0
        christmas = 0
        mothers_day = 0
        new_year = 0
        others = 0
        valentines = 0

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

    orders_output = str(orders)
    o_price_output = str(o_price)
    profit_output = str(profit)

    # orders_output = ''
    # o_price_output = ''
    # profit_output = ''
    # message_output = ''

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
