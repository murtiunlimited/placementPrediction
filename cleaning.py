import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# loading data 
df = pd.read_csv("./dataset/Placement_Data_Full_Class.csv")
df.drop('sl_no',axis=1,inplace=True)


le = LabelEncoder()
cat_cols =['gender','ssc_b','hsc_b','hsc_s','degree_t','workex','specialisation','status']
for col in cat_cols:
    df[col] =le.fit_transform(df[col])

# Scale Numerical columns 
scaler = StandardScaler()
num_cols = ['ssc_p','hsc_p','degree_p','etest_p','mba_p']
df[num_cols]=scaler.fit_transform(df[num_cols])
df.to_csv("./dataset/preprocessed_placement_data.csv",index=False)