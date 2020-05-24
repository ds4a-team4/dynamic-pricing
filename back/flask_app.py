import flask
from RL import create_results
import pandas as pd
from sqlalchemy import create_engine
import secrets

app = flask.Flask(__name__)

def get_result_df():
    uri = secrets.URI
    engine = create_engine(uri)
    df = pd.read_sql_table("rl_results", con=engine)
    return df




@app.route('/', methods=['GET'])
def home():
    return "<h1>Dynamic Pricing API</h1><p>Endpoint list</p>"

@app.route('/api/v1/dataframe/html', methods=['GET'])
def api_dfhtml():
    df = get_result_df()
    return df.to_html(index = False)

@app.route('/api/v1/dataframe/json', methods=['GET'])
def api_dfjson():
    df = get_result_df()
    return flask.Response(df.to_json(orient="records"), mimetype='application/json')

@app.route('/api/v1/process', methods=['GET'])
def api_process():
    create_results()
    return "Processed"

    

app.run()