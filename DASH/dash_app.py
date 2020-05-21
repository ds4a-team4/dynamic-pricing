
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import pandas as pd

df = pd.read_csv('../models/cellphones/cellphones_final_results.csv')

# app = dash.Dash()
app = dash.Dash(external_stylesheets=[dbc.themes.YETI])
app.title = 'DS4A'

# Boostrap CSS.
# app.css.append_css({'external_url': 'https://codepen.io/amyoshino/pen/jzXypZ.css'})  

# Materialize CSS
# external_css = ["https://cdnjs.cloudflare.com/ajax/libs/materialize/0.100.2/css/materialize.min.css"]
# for css in external_css:
#     app.css.append_css({"external_url": css})



app.layout = dbc.Container(
    html.Div(
        [
            html.Div(
                [
                    html.Div(
                        html.H1
                        (
                            children='Olist Dynamic Pricing',
                            style={
                                'text-align': 'center',
                                'margin-top': '15px',
                                'margin-bottom': '25px'
                            }
                        )
                    ) 
                ]
            ),

            html.Div(
                [
                    html.Div(
                        [   html.P('Product Group'),
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
                        [   html.P('Product Type'),
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
                        [   html.P('Price Range'),
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

                ], className="row",
                        style={
                                'margin-top': '15px',
                                'margin-bottom': '15px'
                            }
            ),

#             html.Div(
#                 [

#                     html.Div(
#                         [   html.P('Baseline'),
#                             dcc.Input(
#                                 placeholder='Baseline',
#                                 type='text',
#                                 value=''
#                             ) 
#                         ],
#                         className='col s4 m4 l4'
#                     ),
#                     html.Div(
#                         [   html.P('RL Model'),
#                             dcc.Input(
#                                 placeholder='RL Model',
#                                 type='text',
#                                 value=''
#                             ) 
#                         ],
#                         className='col s4 m4 l4'
#                     ),
#                     html.Div(
#                         [   html.P('Delta'),
#                             dcc.Input(
#                                 placeholder='Delta',
#                                 type='text',
#                                 value=''
#                             ) 
#                         ],
#                         className='col s4 m4 l4'
#                     )                        

#                 ], className="row",
#                         style={
#                                 'margin-top': '15px',
#                                 'margin-bottom': '15px'
#                             }
#             ),


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
        data.append({'x': df.index, 'y': df.baseline_orders, 'type': 'line', 'name': 'Baseline Orders'})
        data.append({'x': df.index, 'y': df.rl_orders, 'type': 'line', 'name': 'RL Orders'})
    figure = {
        'data': data,
        'layout': {
            'title': 'Baseline x Reinforcement Learning',
            'xaxis' : dict(
                title='TIME',
                titlefont=dict(
                family='Helvetica, monospace',
                size=20,
                color='#7f7f7f'
            )),
            'yaxis' : dict(
                title='ORDERS',
                titlefont=dict(
                family='Helvetica, monospace',
                size=20,
                color='#7f7f7f'
            ))
        }
    }
    return figure

if __name__ == '__main__':
    app.run_server(debug=True)
