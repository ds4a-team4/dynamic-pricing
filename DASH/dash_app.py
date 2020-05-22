
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import pandas as pd
from datetime import datetime as dt

df = pd.read_csv('../models/cellphones/cellphones_final_results.csv')

# app = dash.Dash()
app = dash.Dash(external_stylesheets=[dbc.themes.SPACELAB])
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
                            'color':'#0C29D0',
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
                                options= [{'label': str(item),
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
                                options= [{'label': str(item),
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
                                options= [{'label': str(item),
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
                                html.P(f'R$ {round(df.baseline_prices.sum(),2)}')
                            ],
                            style={
                                    'color':'#0DC78B'
                                })
                        ],
                        className='col s4 m4 l4'
                    ),
                    html.Div(
                        [   
                            html.H5('RL'),
                            html.Div([
                                html.P(f'R$ {round(df.rl_prices.sum(),2)}')
                            ],
                            style={
                                    'color':'#0DC78B'
                                })
                        ],
                        className='col s4 m4 l4'
                    ),
                    html.Div(
                        [   
                            html.H5('DELTA'),
                            html.Div([
                                html.P(f'{round(df.rl_prices.sum() / df.baseline_prices.sum() * 100,2)} %')
                            ],
                            style={
                                    'color':'#0DC78B'
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
                        ], className= 'col s12 m12 l12'
                    )

                ], className="row"
            )
        ]
    )
)

@app.callback(
    Output(component_id='graph1', component_property='figure'),
    [Input(component_id='selector_group', component_property='value')])
def update_graph(selector):
    data = []
    if 'electronics' in selector:
        data.append({'x': df.index, 'y': df.baseline_orders, 'type': 'line', 'name': 'Baseline Orders', 'line': dict(color='#6A00A3')})
        data.append({'x': df.index, 'y': df.rl_orders, 'type': 'line', 'name': 'RL Orders', 'line': dict(color='#EDAD00')})
    figure = {
        'data': data,
        'layout': {
            'title': 'Baseline x Reinforcement Learning',
            'xaxis' : dict(
                title='TIME',
                titlefont=dict(
                family='Helvetica, monospace',
                size=20,
                color='#312F4F'
            )),
            'yaxis' : dict(
                title='ORDERS',
                titlefont=dict(
                family='Helvetica, monospace',
                size=20,
                color='#312F4F'
            ))
        }
    }
    return figure

if __name__ == '__main__':
    app.run_server(debug=True)
