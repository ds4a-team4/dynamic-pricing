# Dynamic Pricing with Deep Reinforcement Learning
Dynamic Pricing repository for the DS4A Correlation One practicum project.  

## Project Description
Our project consists in the application of Reinforcement Learning algorithms to predict optimal pricing policies for an e-commerce platform.
We gathered timeseries information for products' with respect to their sales, price changes, inventory levels, market prices, and others to assemble an competitive pricing policy.
We started by performing EDAs on different datasets, gathering information about the data's nature and possible flaws and shortcomings, such as data sparsity. We then proceed to some ETL processes to clean and process the data before injecting it to different forecasting algorithms. Such models were used to simulate an environment in which we would train our Deep Reinforcement Learning algorithm to choose pricing policies. To wrap-up, we display our results in an web application developed using Plotly's Dash.

See below an index for our file structure:

## File Structure
## back
In the back folder we host the Flask API.  

## DASH
In the DASH folder we host the front-end to our web application code.

## EDA
Folder that contains our Exploratory Data Analysis process.

## ETL
Contains the routines to Extract, Transform and Load our data into more aggregated levels.

## data
Collection of datasets.

## models
Folder that hosts both the simulator's code to forecast orders levels and the Reinforcement Learning algorithms that were used.

## models_dash
Folder that contains the last version of our simulator.  

## File Structure
├── back  -- Flask API  
│   ├── models  
├── DASH -- Hosts our front end  
│   └── assets  
├── data  -- Our Datasets 
├── EDA  -- Exploratory data analysis  
├── ETL  
├── models  -- Model collection and testing  
│   └── cellphones  
└── models_dash  -- Final Model using Reinforcement Learning    
    └── cellphones  