"""
Arina modeling of sales forecasting. Customers provide purchasing forecasts. 
This program runs these data against the actual figures from past quarters to make future projections.
"""



# load required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import itertools
import sys
from numpy import inf
import warnings
import math


df = pd.read_csv('book2.csv', header=0, index_col=0, parse_dates=True)
 
# create Series object
y = df[0:]
print(y)
y_train = y[:'1/1/2018']
y_test = y['1/1/2018':]

# define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 3)
 
# generate all different combinations of p, d and q triplets
pdq = list(itertools.product(p, d, q))
 
# generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
best_aic = np.inf
best_pdq = None
best_seasonal_pdq = None
tmp_model = None
best_mdl = None
'''
# fit model to data
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            tmp_mdl = sm.tsa.statespace.SARIMAX(y_train,
                                                order = param,
                                                seasonal_order = param_seasonal,
                                                enforce_stationarity=True,
                                                enforce_invertibility=True)
            res = tmp_mdl.fit()
            if res.aic < best_aic:
                best_aic = res.aic
                best_pdq = param
                best_seasonal_pdq = param_seasonal
                best_mdl = tmp_mdl
        except:
            print("Unexpected error:", sys.exc_info()[0])
            continue
print("Best SARIMAX{}x{}12 model - AIC:{}".format(best_pdq, best_seasonal_pdq, best_aic))

#Best SARIMAX(2, 2, 0)x(0, 2, 0, 12)12 model - AIC:348.56247355479053
'''
# fit model to data
res = sm.tsa.statespace.SARIMAX(y_train,
                                order=(2, 2, 1),
                                seasonal_order=(0, 2, 0, 12),
                                enforce_stationarity=True,
                                enforce_invertibility=True).fit()

 
# get forecast 120 steps ahead in future
pred_uc = res.get_forecast(steps=6)
 
# get confidence intervals of forecasts
pred_ci = pred_uc.conf_int()
 
# plot time series and long-term forecast
ax = y.plot(label='Observed', figsize=(16, 8), color='#006699');
pred_uc.predicted_mean.plot(ax=ax, label='Forecast', color='#ff0066');
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='#ff0066', alpha=.25);
ax.set_xlabel('Date');
ax.set_ylabel('Passengers');
plt.legend(loc='upper left')
plt.show()



print(pred_uc.predicted_mean)