import pandas as pd
directory = r'C:\Users\tmccl\OneDrive\Documents\GitHub\leek-growth-model\Weather Station'
df_url = pd.read_csv(r'https://sms2.soilmoisturesense.com/weather/allpress.php?station=2428&uname=allpress&pass=C4pef36')
filename = df_url.loc[0, 'Local Date'] + '_allpress_sensors.csv'
print(filename)
df_url.to_csv(rf'{directory}/{filename}')