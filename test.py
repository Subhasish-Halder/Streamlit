import pandas as pd
telecom_cust = pd.read_csv('Telco_Customer_Churn.csv')
print(telecom_cust.isnull().sum())