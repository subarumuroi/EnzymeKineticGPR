#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 19:10:14 2024

@author: s

In Synthetic Biology, Metabolic engineering has the potential to manufacture
products through precision fermentation. This model uses thermodynamically 
constrained Enzyme Kinetic model data outputs and converge towards the
multidimensional solution space 
"""



trainFile = #"/Users/s/PhD/AllRuns_SingleTime_Train.csv"
testFile = #"/Users/s/PhD/AllRuns_SingleTime_Test.csv"


Adam_learning_rate = 0.0066
Nepochs = 2000
Nneurons_Dense = 320
activation = 'relu'
batch = 32

verbose = 0 # 1 => display model training/prediction progress data to console or 0 => don't

DataSplit = 'random' # split of data into training and test data 'manual' => Sub's separate files; 
                     #                                           'random' => Sub's separate files are first concatenated into one data set and then split randomly

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt

#prepare data

# load data
def LoadData(csvFile):
    
    df = pd.read_csv(csvFile)

    df_Inputs = df.iloc[2:8,1:].astype(float)
    df_Inputs = df_Inputs.reset_index(drop=True)

    df_Targets = df.iloc[17:,1:].astype(float) # selects whole profile for each test sample
    df_Targets = df_Targets.reset_index(drop=True)
    
    y_alongshore = df.iloc[17:,0].astype(float).values

    return df_Inputs, df_Targets, y_alongshore
# this is the manula spec of training and test data by Sub's two files
x_train, y_train, y_alongshore = LoadData(trainFile)
x_test, y_test, y_alongshore = LoadData(testFile)

#Smooth the data
def SmoothData(df,window_size=5,weights_type='gaussian', std_dev=1):
    
    '''
    df = dataframe
    window_size = size of moving average window
    weights_type = type of centred weight distribution
    std_dev = std dev of normal dist, increase for greater smooting

    '''
    
    if weights_type == 'gaussian':

        # Create centered weights based on the Gaussian distribution
        weights = norm.pdf(np.arange(window_size), loc=window_size // 2, scale=std_dev)
        df_Smoothed = df.rolling(window=window_size, center=True).apply(lambda x: np.dot(x, weights) / sum(weights), raw=True)
    
    elif weights_type == 'equal': # equal weight moving average

        df_Smoothed = df.apply(lambda column: column.rolling(window=window_size, center=True).mean(), axis=0)

    # drop NaNs
    df_Smoothed = df_Smoothed.dropna()
    df_Smoothed = df_Smoothed.reset_index(drop=True)
    
    return df_Smoothed

#Do the smoothing, you can tune the degree of smoothing with std_dev and window size
std_dev = .75 # std dev of normal dist, increase for greater smoothing (I think .75 was Nick's default val)
window_size = 5 # smoothing window size 5=> center point and 2 either side
dropid = window_size // 2 # divides window_size by 2 and returns the quotient rounded down to the nearest whole number (integer).

y_train_smooth = SmoothData(y_train,window_size,'gaussian',std_dev=std_dev)
y_test_smooth = SmoothData(y_test,window_size,'gaussian',std_dev=std_dev)

#Plot raw vs smooth comparison
idx = 10 # test sample shoreline profile to plot raw vs smoothed

fig,axs=plt.subplots(1,2, figsize=(10, 5))
# training data
axs[0].plot(y_alongshore,y_train.iloc[:,idx],label='Raw Data')
axs[0].plot(y_alongshore[dropid:-dropid],y_train_smooth.iloc[:,idx], color='r', label='Smoothed Data')
axs[0].set_xlabel('Alongshore position of transect (m)')
axs[0].set_ylabel('Crosshore position of shoreline (m)')
axs[0].set_title('Training Data Sample')
axs[0].legend()

# testing data
axs[1].plot(y_alongshore,y_test.iloc[:,idx],label='Raw Data')
axs[1].plot(y_alongshore[dropid:-dropid],y_test_smooth.iloc[:,idx], color='r',label='Smoothed Data')
axs[1].set_xlabel('Alongshore position of transect (m)')
axs[1].set_ylabel('Crosshore position of shoreline (m)')
axs[1].set_title('Test Data Sample')
axs[1].legend()

plt.show()

#Drop edges from alongshore profile coordinates that are lost due to the moving widnow smoothing
y_alongshore = y_alongshore[dropid:-dropid] 


#Standardise data for ML input
#%% transpose dfs to get to work with model code below
x_train, x_test, y_train, y_test = x_train.T, x_test.T, y_train_smooth.T, y_test_smooth.T

#%% Standardize features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)



# Save scaling parameters for scaling of single input test
scaling_mean = scaler.mean_
scaling_std = scaler.scale_

#Initialise dicts for skill metric collection
# Initialize dictionaries to store performance metrics
train_metrics = {'Model': [], 'MAE': [], 'MSE': [], 'R2': []}
test_metrics = {'Model': [], 'MAE': [], 'MSE': [], 'R2': []}


model_nn = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(6,)),  #input layer with 6var
    tf.keras.layers.Dense(Nneurons_Dense, activation=activation),
    tf.keras.layers.Dense(Nneurons_Dense, activation=activation),
    #tf.keras.layers.Dense(Nneurons_Dense, activation='tanh'),
    tf.keras.layers.Dense(51)  #output layer with entire shoreline
])
#optimizer
opt = tf.keras.optimizers.legacy.Adam(learning_rate = Adam_learning_rate, clipnorm = 1)
#compile the model
model_nn.compile(optimizer=opt, loss='mean_squared_error', metrics=[tf.keras.metrics.MeanSquaredError()])


# Define the EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss',  # Monitor the validation loss
                               patience=1200,           # Number of epochs with no improvement after which training will be stopped
                               restore_best_weights=True)  # Restore model weights from the epoch with the best value of the monitored quantity


fit=model_nn.fit(x_train_scaled, y_train, epochs=Nepochs, batch_size=batch, verbose=verbose, validation_data=(x_test_scaled, y_test), callbacks = [early_stopping])

# loss plot
plt.figure()
plt.plot(fit.history['loss'])
plt.title('Model Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


#Get model predictions
# Predictions on training set
y_train_pred_nn = model_nn.predict(x_train_scaled)

# Predictions on testing set
y_test_pred_nn = model_nn.predict(x_test_scaled)

model1_name = 'Neural Network'

#nerual net ends here


#Train the model
model2_name = 'Gaussian Process Regressor'
model_gpr=GaussianProcessRegressor(alpha = 1e-7, kernel = Matern(nu=0.8), random_state=42)
model_gpr.fit(x_train_scaled, y_train)



#Get model predictions
# Predictions on training set
y_train_pred_gpr = model_gpr.predict(x_train_scaled)

# Predictions on testing set
y_test_pred_gpr = model_gpr.predict(x_test_scaled)

#Compute skill metrics
def getMetrics(actual,predicted):

    metrics = {}
    metrics['mae'] = mean_absolute_error(actual, predicted)
    metrics['mse'] = mean_squared_error(actual, predicted)
    metrics['rmse'] = np.sqrt(metrics['mse']) # root mean square error
    metrics['si'] = 100*metrics['rmse'] / (np.max(actual) - np.min(actual)) # scatter index
    metrics['r2'] = r2_score(actual, predicted)
    
    return metrics

# Calculate metrics for training set
train_metrics_nn = getMetrics(y_train, y_train_pred_nn)
train_metrics_gpr = getMetrics(y_train, y_train_pred_gpr)

# Calculate metrics for testing set
test_metrics_nn = getMetrics(y_test, y_test_pred_nn)
test_metrics_gpr = getMetrics(y_test, y_test_pred_gpr)



y_train_pred_nn_df = pd.DataFrame(y_train_pred_nn)
y_train_pred_gpr_df = pd.DataFrame(y_train_pred_gpr)
y_test_pred_nn_df = pd.DataFrame(y_test_pred_nn)
y_test_pred_gpr_df = pd.DataFrame(y_test_pred_gpr)



def getMetricsEdo(actual, predicted):
    metrics_list = []
    for i in range(len(actual)):
        metrics = {}
        actual_row = actual.iloc[i]  # Accessing the ith row of the DataFrame
        predicted_row = predicted.iloc[i]
        metrics['mae'] = mean_absolute_error(actual_row, predicted_row)
        metrics['mse'] = mean_squared_error(actual_row, predicted_row)
        metrics['rmse'] = np.sqrt(metrics['mse'])  # root mean square error
        metrics['si'] = 100 * metrics['rmse'] / (np.max(actual_row) - np.min(actual_row))  # scatter index
        metrics['r2'] = r2_score(actual_row, predicted_row)
        metrics_list.append(metrics)
    return metrics_list

# Calculate metrics for training set
train_metrics_nn_list = getMetricsEdo(y_train, y_train_pred_nn_df)
train_metrics_gpr_list = getMetricsEdo(y_train, y_train_pred_gpr_df)

# Calculate metrics for testing set
test_metrics_nn_list = getMetricsEdo(y_test, y_test_pred_nn_df)
test_metrics_gpr_list = getMetricsEdo(y_test, y_test_pred_gpr_df)

#Plot evaluation
# def plot_actual_vs_predicted(ax, y_actual, y_pred, model_name, set_type, mae, mse, r2, y0):
'''
def plot_actual_vs_predicted(ax, y_actual, y_pred, model_name, set_type, metrics, x0, y0):
    
    ax.scatter(y_actual, y_pred, s=20, alpha=0.6, label=set_type)
    ax.set_title(f'{model_name}')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_aspect('equal', adjustable='box')  # Set aspect ratio to make axes square
    ax.set_xlim([min(y_actual.min().min(), y_pred.min().min()), max(y_actual.max().max(), y_pred.max().max())])
    ax.set_ylim([min(y_actual.min().min(), y_pred.min().min()), max(y_actual.max().max(), y_pred.max().max())])
    formatted_string = set_type+'\n'+"\n".join([f"{key.upper()}: {value:.4f}" for key, value in metrics.items()])
    ax.text(x0, y0, formatted_string, transform=ax.transAxes)

    ax.grid(True)
    ax.legend()
'''

#below is the code gpt suggested to fix above
def plot_actual_vs_predicted(ax, y_actual, y_pred, model_name, set_type, metrics, x0, y0):
    ax.scatter(y_actual, y_pred, s=20, alpha=0.6, label=set_type)
    ax.set_title(f'{model_name}')
    ax.set_xlabel('Actual Shoreline Change (m)')
    ax.set_ylabel('Predicted Shoreline Change (m)')
    ax.set_aspect('equal', adjustable='box')  # Set aspect ratio to make axes square
    ax.set_xlim([min(y_actual.min().min(), y_pred.min().min()), max(y_actual.max().max(), y_pred.max().max())])
    ax.set_ylim([min(y_actual.min().min(), y_pred.min().min()), max(y_actual.max().max(), y_pred.max().max())])
    
    # Format metrics string
    formatted_string = set_type + '\n' + "\n".join([f"{key.upper()}: {value:.3f}" if isinstance(value, float) else f"{key.upper()}: {value}" for key, value in metrics.items()])
    
    ax.text(x0, y0, formatted_string, transform=ax.transAxes)
    ax.grid(True)
    ax.legend()
    
    
    
    
#Skill scatter plot and shoreline profile comparison

test_index = 8 # index of test sample shoreline profile to compare

# skill scatter plot
plt.close('all')

plt.rcParams['font.size'] = 12


fig, axes = plt.subplots(1, 2, figsize=(15, 9)) # the left plot is the model evaluation plot for all training and test data, the second will be the shoreline profile for a user specified test sample


# Plot actual vs. predicted for neural net training set 
plot_actual_vs_predicted(axes[0], y_train, y_train_pred_nn, model1_name, 'NNTraining', train_metrics_nn, 0.05, 0.5)

# Plot actual vs. predicted for gpr training set
plot_actual_vs_predicted(axes[1], y_train, y_train_pred_gpr, model2_name, 'GPRTraining', train_metrics_gpr, 0.05, 0.5)


# Plot actual vs. predicted for nn testing set
plot_actual_vs_predicted(axes[0], y_test, y_test_pred_nn, model1_name, 'NNTesting', test_metrics_nn, 0.65, 0.1)
axes[0].plot([y_train.min().min(), y_train.max().max()], [y_train.min().min(), y_train.max().max()], 'k--', lw=2)  # Add 1:1 line

# Plot actual vs. predicted for gpr testing set
plot_actual_vs_predicted(axes[1], y_test, y_test_pred_gpr, model2_name, 'GPRTesting', test_metrics_gpr, 0.65, 0.1)
axes[1].plot([y_train.min().min(), y_train.max().max()], [y_train.min().min(), y_train.max().max()], 'k--', lw=2)  # Add 1:1 line

# select test sample to examine
x_input = [x_test.iloc[test_index]] 

# Scale the new input using the saved scaling parameters
x_input_scaled = (x_input - scaling_mean) / scaling_std
# get the actual test sample data
y_actual = y_test.iloc[test_index].values

# # run mode for the nn test sample
y_pred_nn = model_nn.predict(x_input_scaled)

# # run mode for the gpr test sample
y_pred_gpr = model_gpr.predict(x_input_scaled)

# add the current test sample to the model scatter plot from above
axes[0].scatter(y_actual, y_pred_nn, s=70, facecolors='none', color='r', label='Test Sample')
axes[0].legend(loc='upper left', bbox_to_anchor=(0.03, 0.97))

# add the current test sample to the model scatter plot from above
axes[1].scatter(y_actual, y_pred_gpr, s=70, facecolors='none', color='g', label='Test Sample')
axes[1].legend(loc='upper left', bbox_to_anchor=(0.03, 0.97))

plt.tight_layout


#plt.close('all')

plt.rcParams['font.size'] = 12


fig, axes = plt.subplots(1, figsize=(15, 6)) 

# get text input paramerters for the plot title
input_par = x_test.iloc[test_index].values
input_dict = {'H' : input_par[0],
              'T' : input_par[1],
              'Dir' : input_par[2],
              'Dist' : input_par[3],
              'L' : input_par[4],
              'W' : input_par[5],
}
plot_title = 'Test Sample: '+', '.join([f"{key}={value}" for key, value in input_dict.items()])

# plot the shoreline profile
axes.plot(y_alongshore,y_actual, 'k', label='Actual')
axes.plot(y_alongshore,y_pred_nn.T, 'r--', label='Predicted nn', lw=1)#nn
axes.plot(y_alongshore,y_pred_gpr.T, 'g--', label='Predicted gpr', lw=1)#gpr


# appearance
axes.set_xlabel('Alongshore position of transect (m)')
axes.set_ylabel('Crosshore position of shoreline (m)')
axes.set_title(plot_title)
axes.grid(True)
axes.legend()
fig.tight_layout()

'''
####
#This section contains an alternate plot which can display all on one page, also contains code that has CI but unfinished
####

test_index=1 # index of test sample shoreline profile to compare

# skill scatter plot
plt.close('all')
fig, axes = plt.subplots(3, 1, figsize=(15, 9)) # the left plot is the model evaluation plot for all training and test data, the second will be the shoreline profile for a user specified test sample


# Plot actual vs. predicted for neural net training set 
plot_actual_vs_predicted(axes[0], y_train, y_train_pred_nn, model1_name, 'NNTraining', train_metrics_nn, 0.05, 0.5)

# Plot actual vs. predicted for gpr training set
plot_actual_vs_predicted(axes[2], y_train, y_train_pred_gpr, model2_name, 'GPRTraining', train_metrics_gpr, 0.05, 0.5)


# Plot actual vs. predicted for nn testing set
plot_actual_vs_predicted(axes[0], y_test, y_test_pred_nn, model1_name, 'NNTesting', test_metrics_nn, 0.65, 0.1)
axes[0].plot([y_train.min().min(), y_train.max().max()], [y_train.min().min(), y_train.max().max()], 'k--', lw=2)  # Add 1:1 line

# Plot actual vs. predicted for gpr testing set
plot_actual_vs_predicted(axes[2], y_test, y_test_pred_gpr, model2_name, 'GPRTesting', test_metrics_gpr, 0.65, 0.1)
axes[2].plot([y_train.min().min(), y_train.max().max()], [y_train.min().min(), y_train.max().max()], 'k--', lw=2)  # Add 1:1 line

# select test sample to examine
x_input = [x_test.iloc[test_index]] 

# Scale the new input using the saved scaling parameters
x_input_scaled = (x_input - scaling_mean) / scaling_std
# get the actual test sample data
y_actual = y_test.iloc[test_index].values

# # run mode for the nn test sample
y_pred_nn = model_nn.predict(x_input_scaled)

# # run mode for the gpr test sample
y_pred_gpr = model_gpr.predict(x_input_scaled)

# add the current test sample to the model scatter plot from above
axes[0].scatter(y_actual, y_pred_nn, s=70, facecolors='none', color='r', label='nn Test Sample (see plot below)')
axes[0].legend(loc='upper left', bbox_to_anchor=(1, 1))

# add the current test sample to the model scatter plot from above
axes[2].scatter(y_actual, y_pred_gpr, s=70, facecolors='none', color='g', label='gpr Test Sample (see plot above)')
axes[2].legend(loc='upper left', bbox_to_anchor=(1, 1))


# get text input paramerters for the plot title
input_par = x_test.iloc[test_index].values
input_dict = {'H' : input_par[0],
              'T' : input_par[1],
              'Dir' : input_par[2],
              'Dist' : input_par[3],
              'L' : input_par[4],
              'W' : input_par[5],
}
plot_title = 'Test Sample: '+', '.join([f"{key}={value}" for key, value in input_dict.items()])

# plot the shoreline profile
axes[1].plot(y_alongshore,y_actual, 'k', label='Actual')
axes[1].plot(y_alongshore,y_pred_nn.T, 'r--', label='Predicted nn', lw=1)#nn
axes[1].plot(y_alongshore,y_pred_gpr.T, 'g--', label='Predicted gpr', lw=1)#gpr

# # add confidence intervals - not sure this is correct?
# ci_lower = y_pred - 1.96 * std_prediction  # 95% ci
# ci_upper = y_pred + 1.96 * std_prediction  # 95% ci
# axes[1].plot(y_alongshore,ci_upper.T,color='b', lw=0.5)
# axes[1].plot(y_alongshore,ci_lower.T,color='b', lw=0.5)
# # plot the cis
# axes[1].fill_between(
#     y_alongshore,
#     ci_lower.flatten(),
#     ci_upper.flatten(),
#     color='b',
#     alpha=0.2,
#     label=r"95% confidence interval",
# )

# appearance
axes[1].set_xlabel('Alongshore position of transect (m)')
axes[1].set_ylabel('Crosshore position of shoreline (m)')
axes[1].set_title(plot_title)
axes[1].grid(True)
axes[1].legend()
fig.tight_layout()
'''

