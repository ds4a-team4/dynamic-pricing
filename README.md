# Dynamic Pricing with Deep Reinforcement Learning
Dynamic Pricing repository for the DS4A Correlation One practicum project.  

## Project Description
Our project consists in the application of Reinforcement Learning algorithms to predict optimal pricing policies for an e-commerce platform.
We gathered timeseries information for products' with respect to their sales, price changes, inventory levels, market prices, and others to assemble an competitive pricing policy.
We started by performing EDAs on different datasets, gathering information about the data's nature and possible flaws and shortcomings, such as data sparsity. We then proceed to some ETL processes to clean and process the data before injecting it to different forecasting algorithms. Such models were used to simulate an environment in which we would train our Deep Reinforcement Learning algorithm to choose pricing policies. To wrap-up, we display our results in an web application developed using Plotly's Dash.

See below an index for our file structure:

## File Structure
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

## File Structure
├── DASH -- Hosts our front end  
│   └── dash_app.py  
├── EDA -- Exploratory data analysis  
│   ├── EDA_Orders.ipynb  
│   ├── EDA_product_info.ipynb  
│   ├── EDA_product_type.ipynb  
│   ├── EDA_timeseries.ipynb  
│   └── timeseries_report.html  
├── ETL -- Preprocessing our data  
│   ├── ETL.ipynb  
│   ├── ETL_full.ipynb  
│   ├── cellphone2017.ipynb  
│   ├── cellphone_etl.ipynb  
│   ├── electronics_etl.ipynb  
│   └── read_athena.ipynb  
├── README.md  
├── data -- Our Datasets  
│   ├── Celular_data.csv  
│   ├── celular_over50.csv  
│   └── timeseries644.csv  
├── models  -- Model collection and testing  
│   ├── 004 Test_cellphones.ipynb  
│   ├── 004.1_by_gtin.ipynb  
│   ├── 004.2_by_cluster.ipynb  
│   ├── 004.3_by_product.ipynb  
│   ├── 005 Test_cellphones.ipynb  
│   ├── Linear_Regression.ipynb  
│   ├── Linear_Regression_CellPhones.ipynb  
│   ├── Model_Selection.ipynb  
│   ├── bottom-up.ipynb  
│   ├── cellphone_prophet.ipynb  
│   ├── cellphones  
│   │   ├── CrazyRL.ipynb  
│   │   ├── cellphone_model.ipynb  
│   │   ├── cellphonedata.csv  
│   │   └── holidays.csv  
│   ├── electronics_prophet-daily.ipynb  
│   ├── electronics_prophet-weekly.ipynb  
│   └── hierarchy_price_electronics.ipynb  
└── requirements.txt  
