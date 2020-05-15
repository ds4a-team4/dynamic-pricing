# dynamic-pricing  
Dynamic Pricing repository for the DS4A Correlation One practicum project.  

## Project Description
How to analyse our project:  
We first started with the ETL, processing our data for posterior analysis and processing. We formatted the data as a timeseries.  
After we explored the data performing the EDA. In this EDA we got to understand more about the data and saw some shortcomings, as the non continuity of orders per day.  

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