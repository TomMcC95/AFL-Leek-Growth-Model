import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

sd_data_nulls = pd.read_excel(r'F:\\Farm Data\Growth Model\AFL Leek Growth Data.xlsx', sheet_name='growth')

sd_data_dc = pd.get_dummies(sd_data_nulls, prefix='', prefix_sep='', columns=['variety'])
sd_data_dc = pd.get_dummies(sd_data_dc, prefix='', prefix_sep='', columns=['method'])
sd_data_dc = pd.get_dummies(sd_data_dc, prefix='', prefix_sep='', columns=['inputs'])
sd_data_dc = pd.get_dummies(sd_data_dc, prefix='', prefix_sep='', columns=['protection'])
print(sd_data_dc)

x = sd_data_dc[['count','particle_size','organic_matter','solar_radiation','heat_units','diameter^0.625','Drilled','Blocks','Superseedlings','Plant Tape','Modules','Bare Roots','Conventional','Organic','Baby','Fleece','Poly','Barley','Batter','Belton','Chiefton','Defender','Comanche','Fencer','Galvani','Gostar','Harston','Krypton','Lancaster','Lexton','Likestar','Linkton','Longton','Nun 70406','Oslo','Pluston','Runner','Shafton','Spheros','Stromboli','Sumstar','Triton','Volta']]
y = sd_data_dc[['standard_deviation']]
x = sm.add_constant(x)
sd_model = sm.OLS(y,x).fit()
print(sd_model.summary())