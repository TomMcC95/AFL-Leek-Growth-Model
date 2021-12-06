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
leek_data_all = pd.read_excel(r'F:\\Farm Data\Growth Model\AFL Leek Growth Data.xlsx', sheet_name='growth')

####STILL NEED STANDARD DEVIATION OF 2019 MEASUREMENTS OR EVEN BETTER TO DISCARD THEM ONCE ENOUGH DATA IS COLLECTED.

###Additional coloumns
leek_data_all['hu_per_mm'] = leek_data_all['heat_units']/leek_data_all['diameter']

###Filter Data
#def filter:(method="all", variety="all", inputs="all", protection="all"):
###Filter by Method
response = input("Would You like to filter Methods? ")
response = response.lower()
if "y" in response:
    method = input("Which Method? ")
    method = method.title()
    filtered_leek_data = leek_data_all[leek_data_all['method'].str.contains(method)]
elif "n" in response:
    filtered_leek_data = leek_data_all
  
###Filter by Variety
response = input("Would You like to filter Variety? ")
response = response.lower()
if "y" in response:
    variety = input("Which Variety? ")
    variety = variety.title()
    filtered_leek_data = filtered_leek_data[filtered_leek_data['variety'].str.contains(variety)]
elif "n" in response:
    filtered_leek_data = filtered_leek_data

###Filter by Inputs    
response = input("Would You like to filter Inputs? ")
response = response.lower()
if "y" in response:
    inputs = input("Which Inputs? ")
    inputs = inputs.title()
    filtered_leek_data = filtered_leek_data[filtered_leek_data['inputs'].str.contains(inputs)]
elif "n" in response:
    filtered_leek_data = filtered_leek_data
   
###Filter by Protection
response = input("Would You like to filter Protection? ")
response = response.lower()
if "y" in response:
    protection = input("Which Protection? ")
    protection = protection.title()
    filtered_leek_data = filtered_leek_data[filtered_leek_data['protection'].str.contains(protection)]
elif "n" in response:
    filtered_leek_data = filtered_leek_data
   
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
#xpred = [40, 200, 0.7, int(32**0.625), 800, 7]
#filtered_leek_data['predicition'] = leek_model.predict(xpred)
#print(filtered_leek_data)

###Bias Graph###
plt.scatter(model_fitted_y,model_residuals)
plt.title('Proof of Bais')
plt.xlabel(r'Total Heat Units')
plt.ylabel(r'Residual Heat Units')
plt.show()

colors = np.where(filtered_leek_data["sample_date"]<'2020-1-1','y','k')
filtered_leek_data.plot.scatter(x="model_fitted_y",y="heat_units",c=colors)
plt.title('Model Accuracy')
plt.xlabel(r'Modelled Heat Units')
plt.ylabel(r'Actual Heat Units')
plt.plot([0, 2500], [0, 2500],color='green',linewidth=2)
plt.show()

###Model Accuracy Graph
fig, ax = plt.subplots()
ax.scatter(model_fitted_y, y)
plt.title('Model Accuracy')
plt.xlabel(r'Modeled Heat Units')
plt.ylabel(r'Actual Heat Units')
line = mlines.Line2D([0, 1, 2], [0, 1, 2], color='red')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.set_ylim(ymin=0)
ax.set_xlim(xmin=0)
plt.show()

###Soil OM Influence Graph
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter3D(filtered_leek_data['hu_per_mm'], filtered_leek_data['organic_matter'], filtered_leek_data['diameter^0.625*10'], cmap='hsv',)
plt.title('         Influence of Soil Organic Matter on Growth\
         \n')
plt.xlabel(r'x')
plt.ylabel(r'y')
plt.show()


###Heat Map
f,ax = plt.subplots(figsize=(20, 20))
corr = filtered_leek_data.corr()
sns.heatmap(corr,mask=np.zeros_like(corr, dtype=np.bool),
           cmap=sns.diverging_palette(220, 10, as_cmap=True),
           square=True, ax=ax)
#plt.show()