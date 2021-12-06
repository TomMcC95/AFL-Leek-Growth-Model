###Data & Package Import

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import numpy as np
import seaborn as sns
import statsmodels.api as sm
leek_data_all = pd.read_excel(r'C:\Users\tmccl\OneDrive\Documents\Python\AFL Leek Growth Data.xlsx', sheet_name='growth')

####STILL NEED STANDARD DEVIATION OF 2019 MEASUREMENTS OR EVEN BETTER TO DISCARD THEM ONCE ENOUGH DATA IS COLLECTED.

###Additional coloumns
leek_data_all['hu_per_mm'] = leek_data_all['heat_units']/leek_data_all['diameter']

###Filter Data
def filter(method="all", variety="all", inputs="all", protection="all"):
    ###Filter by Method
    method = method.title()
    if method != "All":
        filtered_leek_data = leek_data_all[leek_data_all['method'].str.contains(method)]
    elif method == "All":
        filtered_leek_data = leek_data_all

    ###Filter by Variety
    variety = variety.title()
    if variety != "All":
        filtered_leek_data = filtered_leek_data[filtered_leek_data['variety'].str.contains(variety)]

    ###Filter by Inputs    
    inputs = inputs.title()
    if inputs != "All":
        filtered_leek_data = filtered_leek_data[filtered_leek_data['inputs'].str.contains(inputs)]

    ###Filter by Protection
    protection = protection.title()
    if protection != "All":
        filtered_leek_data = filtered_leek_data[filtered_leek_data['protection'].str.contains(protection)]
    return filtered_leek_data

filtered_leek_data = filter("drilled", "krypton")

###Growth Model Creation###
x = filtered_leek_data[['count','organic_matter','diameter^0.625*10', 'solar_radiation']]
y = filtered_leek_data[['heat_units']]
x = sm.add_constant(x)
model = sm.OLS(y,x)
leek_model = model.fit()
model_summary = leek_model.summary()
print(model_summary)
model_fitted_y = leek_model.fittedvalues
filtered_leek_data['model_fitted_y'] = model_fitted_y
model_residuals = leek_model.resid

###Bias Graph###
plt.scatter(model_fitted_y,model_residuals)
plt.title('Proof of Bais')
plt.xlabel(r'Total Heat Units')
plt.ylabel(r'Residual Heat Units')
plt.show()

###Model Accuracy Grpah###
colors = np.where(filtered_leek_data["sample_date"]<'2020-5-1','b','r')
filtered_leek_data.plot.scatter(x="model_fitted_y",y="heat_units",c=colors)
plt.legend(['2019 Planting'])
plt.title('Model Accuracy')
plt.xlabel(r'Modelled Heat Units')
plt.ylabel(r'Actual Heat Units')
plt.plot([0, 2500], [0, 2500],color='green',linewidth=2)
plt.show()

###Soil OM Influence Graph
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter3D(filtered_leek_data['hu_per_mm'], filtered_leek_data['organic_matter'], filtered_leek_data['diameter^0.625*10'], cmap='hsv',)
plt.title('         Influence of Soil Organic Matter on Growth\
         \n')
plt.xlabel(r'x')
plt.ylabel(r'y')
#plt.show()


###Heat Map
f,ax = plt.subplots(figsize=(20, 20))
corr = filtered_leek_data.corr()
sns.heatmap(corr,mask=np.zeros_like(corr, dtype=np.bool),
           cmap=sns.diverging_palette(220, 10, as_cmap=True),
           square=True, ax=ax)
#plt.show()