import torch
import torch.nn as nn
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go

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
    
    def predict_demand(self, model, df_row, olist_price):
        
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
                    friday, mothers_day, new_year, others, valentines]).reshape(1,-1)
                            
        orders = model.predict(X)
        
        return max(orders[0],0)

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
        self.orders = self.predict_demand(self.model, self.data.iloc[self.t], self.olist_price)        

        reward = (self.olist_price + self.data['freight_value'][self.t] - self.data['base_cost'][self.t])*self.orders
        self.profits += reward

        # set next time
        self.t += 1
        
        if (self.t == self.N):
            self.done=True

        return [self.olist_price, self.orders] + self.data.iloc[self.t].tolist(), reward, self.done # obs, reward, done 
