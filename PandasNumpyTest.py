import pandas as pd
import  numpy as np
df = pd.read_csv(
    "C:\\Sayed\\Python\\py-master\\pandas\\2_dataframe_basics\\weather_data.csv",
     parse_dates=["day"])
print(df.head(10))
rows, colms = df.shape
print(rows, colms)
print(df[2:3])
print(df[:])
print(df[1:])
print(df.columns)
print(df['event'])
print(df[['day','event']])
print(df.describe())
print(df[df['temperature']>=32])

data_frame={
    'day':['12','23'],
    'score':[12,45.6],
    'name':['omi',12.5]
}
df_new= pd.DataFrame(data_frame)
print(df_new)
print(df)
print(type(df['day'][0]))
# new_df=df.fillna(0)
# print(new_df)
new_df = df.fillna({
    'day':max(df['day']),
    'temperature': df['temperature'].mean(),
    'windspeed': df['windspeed'].max(),
    'event':'Rain'
})
new_df = new_df.replace({
    0:np.nan
})
new_df= new_df.fillna({
    'windspeed':new_df['windspeed'].mean()
})

new_df = new_df.replace({
    'temperature': '[A-Za-z]',
    'windspeed': '[A-Za-z]'
}, '', regex=True)

new_df =new_df.replace(['Rain','Sunny','Snow'],[1,2,3])

print(new_df)
