## Models
In this section, we describe in more detail a few of our modeling process and decisions.

## Forecasts
In order to train an Agent based on Reinforcement Learning, it is a good idea to create an environment where one could perform a good amount of simulations to train the agent on.  
With this in mind, we experimented with a few algorithms to predict incoming orders based on different features, such as date
parameters (day of the week, month, holidays, etc.), product's prices, shipping values, inventory levels, etc. We then used this forecasting algorithm to act as a scenario simulator to our Reinforcement Learning agent.  
Among the regression techniques, we experimented with: Linear Regression, Decision Trees, Random Forests, Support Vector Machines,
eXtreme Gradient Boosting and Facebook's fbprophet library.

## Reinforcement Learning - Environment and Agent
In this folder we hold the base code to our reinforcement Learning Agent.

## Cellphones
The cellphones folder contains the final data and models used in our selected category to experiment with the aforementioned techniques.
