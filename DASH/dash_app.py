
import dash
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash()

# Boostrap CSS.
app.css.append_css({'external_url': 'https://codepen.io/amyoshino/pen/jzXypZ.css'})  

app.layout = html.Div(
    html.Div([
            html.Div([
                    html.H1
                    (
                        children='Dynamic Pricing',
                        className = "seven columns",
                        style={
                            'margin-left': 500
                        }
                    ),
                    html.Img
                    (
                        src="https://vagas.byintera.com/wp-content/uploads/2020/03/logo-olist.png",
                        className='five columns',
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
                            html.P('Choose Product Group:'),
                            dcc.Checklist
                            (
                                    id = 'Groups',
                                    options=[
                                        {'label': 'Electronics', 'value': 'Smartphone'},
                                        {'label': 'Beauty', 'value': 'Shampoo'}
                                    ],
                                    value=['Smartphone', '0'],
                                    labelStyle={'display': 'inline-block'}
                            )
                        ],
                        className='six columns',
                        style={'margin-top': '10'}
                    ),
                ], className="row"
            ),

            html.Div([
                    html.Div([
                            dcc.Graph(id='graph1'
                            )
                        ], className= 'six columns'
                    )

                ], className="row"
            )
        ] 
    ), 
    style={
        'padding': '5px 35px'
    }
)

@app.callback(
    dash.dependencies.Output('graph1', 'figure'),
    [dash.dependencies.Input('Groups', 'value')])
def update_image_src(selector):
    data = []
    if 'Smartphone' in selector:
        data.append({'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'line', 'name': 'Smartphone'})
    if 'Shampoo' in selector:
        data.append({'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'line', 'name': 'Shampoo'})
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
