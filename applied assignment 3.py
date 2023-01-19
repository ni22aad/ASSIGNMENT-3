#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as skmet

def reading_data_from_file(name):
    '''
    
    this function read file name and then return the original dat and tranposed data

    '''
    data=pd.read_csv("data.csv",skiprows=3)
    data=data.drop(["Unnamed: 66"],axis=1)
    return data,data.T

def logistics(t, scale, growth, t0):
    """ 
    Computes logistics function with scale, growth raat
    and time of the turning point as free parameters
    """
    f = scale / (1.0 + np.exp(-growth * (t - t0)))
    return f
def exponential(t, n0, g):
    """Calculates exponential function with scale factor n0 and growth rate g."""
    t = t - 1960.0
    f = n0 * np.exp(g*t)
    return f
def norm(array):
    """ Returns array normalised to [0,1]. Array can be a numpy array
    or a column of a dataframe"""
    min_val = np.min(array)
    max_val = np.max(array)
    scaled = (array-min_val) / (max_val-min_val)
    return scaled
def norm_df(df, first=0, last=None):

    """
    Returns all columns of the dataframe normalised to [0,1] with the
    exception of the first (containing the names)
    Calls function norm to do the normalisation of one column, but
    doing all in one function is also fine.
    First, last: columns from first to last (including) are normalised.
    Defaulted to all. None is the empty entry. The default corresponds
    """
    # iterate over all numerical columns
    for col in df.columns[first:last]: # excluding the first column
        df[col] = norm(df[col])
    return df
def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for 
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.
    
    This routine can be used in assignment programs.
    """
    import itertools as iter
    
    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    
    uplow = []   # list to hold upper and lower limits for parameters
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        
    return lower, upper  
def logistic(t, n0, g, t0):
    """Calculates the logistic function with scale factor n0 and growth rate g"""
    f = n0 / (1 + np.exp(-g*(t - t0)))
    return f
def predict_future_data(data,sigma,param):
    year = np.arange(1960, 2031)
    low, up = err_ranges(year, logistic, param, sigma)
    forecast = logistic(year, *param)

    plt.figure()
    plt.plot(data["Year"],data['co2'], label="co2")
    plt.plot(year, forecast, label="forecast")
    plt.fill_between(year, low, up, color="yellow", alpha=0.7)
    plt.xlabel("year",fontsize=18)
    plt.title("india",fontsize=18)
    plt.ylabel("co2",fontsize=18)
    plt.legend(),
    plt.show()
    
def data_fitting(data,name):
    '''
    this  function will seprate data indicators and then plot graph show the good data fitting 
    then it will show confidence of data and error uper and lower limit
    '''
    # fit exponential growth
    data_for_fitting=pd.DataFrame()
    year = np.arange(1963, 2020)
    print(year)
    in_data=data[data["Country Name"]==name]
    in_data_fores_area=in_data[in_data["Indicator Code"]=="EN.ATM.CO2E.LF.KT"]
    in_data_fores_urban=in_data[in_data["Indicator Code"]=="SP.URB.TOTL"]
    in_data_fores_arable_land=in_data[in_data["Indicator Code"]=="EN.ATM.CO2E.LF.KT"]
    
    in_data_fores_area=in_data_fores_area.drop(["Country Name","Indicator Name","Country Code","Indicator Code"],axis=1).T
    in_data_fores_urban=in_data_fores_urban.drop(["Country Name","Indicator Name","Country Code","Indicator Code"],axis=1).T
    in_data_fores_arable_land=in_data_fores_arable_land.drop(["Country Name","Indicator Name","Country Code","Indicator Code"],axis=1).T
    
    in_data_fores_area=in_data_fores_area.dropna()
    in_data_fores_urban=in_data_fores_urban.dropna()
    in_data_fores_arable_land=in_data_fores_arable_land.dropna()
    
    
    data_for_fitting['co2']=in_data_fores_area
    data_for_fitting['urban']=in_data_fores_urban
    data_for_fitting['arable']=in_data_fores_arable_land
    data_for_fitting['Year']=pd.to_numeric(year)
    popt, covar = opt.curve_fit(logistic,data_for_fitting['Year'],data_for_fitting['urban'],p0=(2e9, 0.05, 1990.0))
    data_for_fitting["fit"] = logistic(data_for_fitting["Year"], *popt)
    sigma = np.sqrt(np.diag(covar))
    year = np.arange(1963, 2040)
    forecast = logistic(year, *popt)
    low, up = err_ranges(year, logistic, popt, sigma)
    
    data_for_fitting.plot("Year", ["urban", "fit"])
    plt.title(str(name)+" Urban Population Fitting")
    plt.ylabel('urban') 
    plt.show()
    plt.figure()
    plt.plot(data_for_fitting["Year"], data_for_fitting["urban"], label="Urban")
    plt.title(str(name)+" Urban Population Fitting")
    plt.plot(year, forecast, label="forecast")
    plt.fill_between(year, low, up, color="yellow", alpha=0.7)
    plt.xlabel("year")
    plt.ylabel("Urban Population")
    plt.legend()
    plt.show()
    
    popt, covar = opt.curve_fit(logistic,data_for_fitting['Year'],data_for_fitting['co2'],p0=(2e9, 0.05, 1990.0))
    data_for_fitting["fit"] = logistic(data_for_fitting["Year"], *popt)
    sigma = np.sqrt(np.diag(covar))
    forecast = logistic(year, *popt)
    low, up = err_ranges(year, logistic, popt, sigma)
    data_for_fitting.plot("Year", ["co2", "fit"])
    plt.title(str(name)+" Urban Population Fitting")
    plt.ylabel('co2') 
    plt.show()
    plt.figure()
    plt.plot(data_for_fitting["Year"], data_for_fitting["co2"], label="co2")
    plt.title(str(name)+" Urban Population Fitting")
    plt.plot(year, forecast, label="forecast")
    plt.fill_between(year, low, up, color="yellow", alpha=0.7)
    plt.xlabel("year")
    plt.ylabel("Urban Population")
    plt.legend()
    plt.show()
    
    popt, covar = opt.curve_fit(logistic,data_for_fitting['Year'],data_for_fitting['arable'],p0=(2e9, 0.05, 1990.0))
    data_for_fitting["fit"] = logistic(data_for_fitting["Year"], *popt)
    sigma = np.sqrt(np.diag(covar))
    forecast = logistic(year, *popt)
    low, up = err_ranges(year, logistic, popt, sigma)
    data_for_fitting.plot("Year", ["arable", "fit"])
    plt.title(str(name)+" Urban Population Fitting")
    plt.ylabel('arable') 
    plt.show()
    plt.figure()
    plt.plot(data_for_fitting["Year"], data_for_fitting["arable"], label="arable")
    plt.title(str(name)+" Urban Population Fitting")
    plt.plot(year, forecast, label="forecast")
    plt.fill_between(year, low, up, color="yellow", alpha=0.7)
    plt.xlabel("year")
    plt.ylabel("arable land")
    plt.legend()
    plt.show()
    return data_for_fitting
    
    
    
    
def kmeans_clustring(data,xlabel,ylabel):
    '''
    this function will show the comparison of different kmeans cluster we we used differenrt statistical methods and other tools

    '''
    
    df_ex = data[["co2", "arable"]].copy()
    # min and max operate column by column by default
    max_val = df_ex.max()
    min_val = df_ex.min()
    df_ex = (df_ex - min_val) / (max_val - min_val)
    # print(df_ex)
    # set up the clusterer for number of clusters
    ncluster = 3
    kmeans = cluster.KMeans(n_clusters=ncluster)
    # Fit the data, results are stored in the kmeans object
    kmeans.fit(df_ex) # fit done on x,y pairs
    labels = kmeans.labels_
    # print(labels) # labels is the number of the associated clusters of (x,y)
    # extract the estimated cluster centres
    cen = kmeans.cluster_centers_
    print(cen)
    # calculate the silhoutte score
    print(skmet.silhouette_score(df_ex, labels))
    # plot using the labels to select colour
    plt.figure(figsize=(10.0, 10.0))
    col = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", \
    "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    for l in range(ncluster): # loop over the different labels
        plt.plot(df_ex[labels==l]["co2"], df_ex[labels==l]["arable"], \
                 "o", markersize=3, color=col[l])
    #
    # show cluster centres
    for ic in range(ncluster):
        xc, yc = cen[ic,:]
        plt.plot(xc, yc, "dk", markersize=10,label="Cluster "+str(ic))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(" cluster of data")
    plt.show()
    
    df_ex = data[["co2", "urban"]].copy()
    # min and max operate column by column by default
    max_val = df_ex.max()
    min_val = df_ex.min()
    df_ex = (df_ex - min_val) / (max_val - min_val)
    # print(df_ex)
    # set up the clusterer for number of clusters
    ncluster = 3
    kmeans = cluster.KMeans(n_clusters=ncluster)
    # Fit the data, results are stored in the kmeans object
    kmeans.fit(df_ex) # fit done on x,y pairs
    labels = kmeans.labels_
    # print(labels) # labels is the number of the associated clusters of (x,y)
    # extract the estimated cluster centres
    cen = kmeans.cluster_centers_
    print(cen)
    # calculate the silhoutte score
    print(skmet.silhouette_score(df_ex, labels))
    # plot using the labels to select colour
    plt.figure(figsize=(10.0, 10.0))
    col = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", \
    "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    for l in range(ncluster): # loop over the different labels
        plt.plot(df_ex[labels==l]["co2"], df_ex[labels==l]["urban"], \
                 "o", markersize=3, color=col[l])
    #
    # show cluster centres
    for ic in range(ncluster):
        xc, yc = cen[ic,:]
        plt.plot(xc, yc, "dk", markersize=10,label="Cluster "+str(ic))
    plt.xlabel(xlabel)
    plt.ylabel("Urban")
    plt.legend()
    plt.title(" cluster of data")
    plt.show()
    
    
    
   
    
    

    
    
   
if __name__ == "__main__":  
    data,transposed_data=reading_data_from_file("data.csv")
    fiti=data_fitting(data,"China")
    kmeans_clustring(fiti,"co2","arable")
    fiti=data_fitting(data,"India")
    kmeans_clustring(fiti,"co2","forest")

    