import torch
import torch.nn as nn
import pickle
import pandas as pd
import numpy as np
from Q_Network import Q_Network
from Environment import Environment
from sqlalchemy import create_engine
import secrets

def base_cost(offer):
    return 0.8 * offer

def create_results():
    with open('./models/lr_cellphone_C.pkl','rb') as f:
        model = pickle.load(f)

    #dataload
    uri = secrets.URI
    engine = create_engine(uri)
    data = pd.read_sql_table("cellphone_data", con=engine)
    data = data[data.price_category == 'C'].copy().reset_index(drop=True) 
    df = data[data.price_category == 'C'].copy().reset_index(drop=True)
    df.drop(columns = ['ds','price_category'],inplace=True)
    df['base_cost'] = df.offer.apply(lambda x: base_cost(x))
    baseline_prices = df['olist_price'].values
    y_pred = []
    for row in df.itertuples():
        y_pred.append(Environment.predict_demand(None,model, row, 0))
    baseline = (df.olist_price + df.freight_value - df['base_cost']) * y_pred
    df.drop(columns = ['offer'],inplace=True)
    df.drop(columns=['olist_price','y'], inplace=True)


    hidden_size = 30
    input_size = 2 + df.shape[1]
    output_size = 10 #5

    Q = Q_Network(input_size, hidden_size, output_size)
    Q.load_state_dict(torch.load('./models/Q_state.torch',map_location=torch.device('cpu')))

    test_env = Environment(model, df)
    test_acts=[]
    test_rewards = []
    orders = []
    o_prices = []
    pobs = test_env.reset()
    profits_2 = 0
    pact_history = []
    done = False

    while not done:
        Q.eval()
        
        pact = Q(torch.from_numpy(np.array(pobs, dtype=np.float32).reshape(1, -1)))
        pact = np.argmax(pact.data.cpu())
        pact_history.append(pact)
        test_acts.append(pact.item())
        
        obs, reward, done = test_env.step(pact.numpy())
        orders.append(obs[1])
        o_prices.append(obs[0])
        test_rewards.append(reward)
        profits_2 += reward
        pobs = obs

    # results
    results = df.iloc[:-1].copy()
    results['baseline_prices'] = baseline_prices[:-1]
    results['baseline_orders'] = y_pred[:-1]

    results['baseline_rewards'] = baseline[:-1]
    results['rl_prices'] = o_prices
    results['rl_orders'] = orders
    results['rl_actions'] = test_acts
    results['rl_rewards'] = test_rewards
    results['group'] = 'electronics'
    results['type'] = 'cellphones'
    results['price_range'] = 'C'
    results.to_sql("rl_results",engine,if_exists='replace',index=False)