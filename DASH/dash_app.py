
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd

df = pd.read_csv('../models/cellphones/cellphonedata.csv')

app = dash.Dash()
app.title = 'dynamic-pricing'

# Boostrap CSS.
# app.css.append_css({'external_url': 'https://codepen.io/amyoshino/pen/jzXypZ.css'})  

# Materialize CSS
external_css = ["https://cdnjs.cloudflare.com/ajax/libs/materialize/0.100.2/css/materialize.min.css"]
for css in external_css:
    app.css.append_css({"external_url": css})

app.layout = html.Div(
    html.Div([
            html.Div([
                    html.H1
                    (
                        children='Dynamic Pricing',
                        className = "col s7 m7 l7",
                        style={
                            'text-align': 'center'
                        }
                    ),
                    html.Img
                    (
                        src="https://vagas.byintera.com/wp-content/uploads/2020/03/logo-olist.png",
                        className='col s5 m5 l5',
                        style={
                            'height': '15%',
                            'width': '10%',
                            'float': 'right',
                            'position': 'relative',
                            'margin-right': 50

                        }
                    )
                ], className = "row"
            ),

            html.Div([
                    html.Div([
                            html.P('Choose Price'),
                            dcc.Checklist
                            (
                                    id = 'selector_choice',
                                    options=[
                                        {'label': 'Olist', 'value': 'Olist CellPhone Prices'},
                                        {'label': 'Competition', 'value': 'Competition CellPhone Prices'},
                                    ],
                                    value=['Olist CellPhone Prices', '0'],
                                    labelStyle={'display': 'inline-block'}
                            ),
#                             html.Div(
#                                 [
#                                     dcc.Dropdown(
#                                         id='year',
#                                         options= [{'label': str(item),
#                                                    'value': str(item)}
#                                                     for item in set(df['year'])],
#                                         multi=True,
#                                         value=list(set(df['year']))
#                                     )
#                                 ],
#                                 className='col s6 m6 l6',
#                                 style={'margin-top': '10'}
#                             )
                        ],
                        className='col s6 m6 l6'
                    ),
                ], className="row"
            ),

            html.Div([
                    html.Div([
                            dcc.Graph(id='graph1'
                            )
                        ], className= 'col s6 m6 l6'
                    )

                ], className="row"
            )
        ] 
    ), className="container",
    style={      
        'margin-left':35,
        'margin-right':35
    }
)

@app.callback(
    Output(component_id='graph1', component_property='figure'),
    [Input(component_id='selector_choice', component_property='value')])
def update_graph(selector):
    data = []
    if 'Olist CellPhone Prices' in selector:
        data.append({'x': df.index, 'y': df.olist_price, 'type': 'line', 'name': 'Olist Price'})
    if 'Competition CellPhone Prices' in selector:
        data.append({'x': df.index, 'y': df.competition_price, 'type': 'line', 'name': 'Competition Price'})
    figure = {
        'data': data,
        'layout': {
            'title': 'Graph 1',
            'xaxis' : dict(
                title='x Axis',
                titlefont=dict(
                family='Courier New, monospace',
                size=20,
                color='#7f7f7f'
            )),
            'yaxis' : dict(
                title='y Axis',
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
