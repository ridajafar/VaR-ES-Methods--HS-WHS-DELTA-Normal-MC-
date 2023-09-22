# Utilities file for Assignment 5 - group 7

import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm
from statistics import NormalDist
import datetime


def relevant_returns(stocks_data, indices_data, datestart, dateend, specific_names):
    # Returns relevant stocks and their returns

    # Compute relevant indices for time and companies
    start = np.where(np.isin(stocks_data['Date'], str(datestart)))[0][0]
    end = np.where(np.isin(stocks_data['Date'], str(dateend)))[0][0]
    stocks_indices = np.where(np.isin(indices_data['Name'], specific_names))[0]+1

    # Slice stocks to obtain the relevant ones
    stocks = stocks_data.iloc[start:end+1, stocks_indices].reset_index(drop=True)

    # Compute returns from the relevant stocks
    stocks_shifted = stocks.iloc[1:end + 1, :].reset_index(drop=True)
    stocks_shifted_back = stocks.iloc[0:-1, :].reset_index(drop=True)
    returns = np.log(stocks_shifted/stocks_shifted_back)

    return [returns, stocks]


def AnalyticalNormalMeasures(alpha, weights, portfolioValue, riskMeasureTimeIntervalInDay, returns):
    # Computes VaR and ES via Gaussian parametric approach

    # Compute mean and covariance matrix of the returns
    mu = -returns.mean() #mean of the loss
    std = returns.cov()
    delta = riskMeasureTimeIntervalInDay

    # Compute VaR
    VaR_std = norm.ppf(alpha)
    VaR = portfolioValue * (delta*np.dot(weights,mu) + np.sqrt(delta*np.dot(weights,np.dot(std,weights)))*VaR_std)

    # Compute ES
    ES_std = 1 / (1 - alpha) * norm.pdf(norm.ppf(alpha))
    ES = portfolioValue * (delta*np.dot(weights,mu) + np.sqrt(delta*np.dot(weights,np.dot(std,weights)))*ES_std)

    return [VaR, ES]


def HSMeasurements(returns, alpha, weights, portfolioValue, riskMeasureTimeIntervalInDay):
    # Computes VaR and ES via Historical simulation

    # Compute Loss for the risk factors
    L = -np.sum(returns*weights,axis=1)*portfolioValue

    # Sort the loss
    L_sorted = -np.sort(-L,axis=0)
    index = int(len(L)*(1-alpha))

    # Compute VaR and ES
    VaR = L_sorted[index-1] * np.sqrt(riskMeasureTimeIntervalInDay)
    ES = np.mean(L_sorted[0:index].T) * np.sqrt(riskMeasureTimeIntervalInDay)

    return [VaR, ES,L,L_sorted ]


def bootstrapStatistical(numberOfSamplesToBootstrap, returns):

    indices = np.random.choice(np.arange(len(returns)), size=numberOfSamplesToBootstrap, replace=True)
    samples = returns.iloc[indices,:].reset_index(drop=True)

    return samples


def WHSMeasurements(returns, alpha, lam, weights, portfolioValue, riskMeasureTimeIntervalInDay):
    # Computes VaR and ES via weighted Historical simulation

    # Compute weights
    delta_time = np.arange(1, len(returns) + 1)
    weights_order = np.array([(1 - lam) / (1 - lam ** len(returns)) * lam ** (len(returns) - delta) for delta in delta_time])

    # Compute Loss for the risk factors
    L = -np.sum(returns*weights,axis=1)*portfolioValue

    # Sort the loss
    L_sorted = np.sort(L)[::-1]
    indices = np.argsort(L)[::-1]
    weights_order_sorted = weights_order[indices]

    index = np.cumsum(weights_order_sorted) <= 1-alpha
    index = np.sum(index)

    # Compute VaR and ES
    VaR = L_sorted[index-1] * np.sqrt(riskMeasureTimeIntervalInDay)
    ES = np.sum(L_sorted[0:index]*weights_order_sorted[0:index])/np.sum(weights_order_sorted[0:index]) * np.sqrt(riskMeasureTimeIntervalInDay)

    return [VaR, ES]


def PrincCompAnalysis(Covariance, MeanReturns, weights, H, alpha, numberOfPrincipalComponents, portfolioValue):
    # Computes VaR and ES with PCA

    n = numberOfPrincipalComponents

    # Compute eigenvalues and rotate the mean and the weights
    eigenvalues, eigenvectors = np.linalg.eig(Covariance)
    ind = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[ind]
    eigenvectors = eigenvectors[:,ind]
    m = -MeanReturns
    mu_hat = np.dot(eigenvectors.T,m)
    weights_hat = np.dot(eigenvectors.T,weights)

    # Compute mu and sigma
    sigma = np.sqrt(np.dot(weights_hat[0:n]**2,eigenvalues[0:n]))
    mu = np.dot(weights_hat[0:n],mu_hat[0:n])

    # Compute VaR
    VaR_std = norm.ppf(alpha)
    VaR = portfolioValue * (H * mu + np.sqrt(H) * sigma * VaR_std)

    # Compute ES
    ES_std = 1 / (1 - alpha) * norm.pdf(norm.ppf(alpha))
    ES = portfolioValue * (H * mu + np.sqrt(H) * sigma * ES_std)

    return [VaR,ES]


def plausibilityCheck(returns, portfolioWeights, alpha, portfolioValue, riskMeasureTimeIntervalInDay):
    # Plausibility check for VaR

    # Compute correlation matrix
    C = returns.corr()

    # Compute  VaR
    sensitivity = -portfolioValue*portfolioWeights
    l = np.percentile(returns,(1-alpha)*100,axis=0)
    u = np.percentile(returns,alpha*100,axis=0)
    sVaR = sensitivity * (abs(l) + abs(u)) / 2
    VaR = np.sqrt(np.dot(sVaR,np.dot(C,sVaR)) * riskMeasureTimeIntervalInDay)

    return VaR


def black_scholes_put(S, K, T, r, q, sigma):
    d1 = (np.log(S/K) + (r - q + sigma**2/2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def FullMonteCarloVaR(logReturns, numberOfShares, numberOfPuts, stockPrice, strike, rate, dividend, volatility, timeToMaturityInYears, riskMeasureTimeIntervalInYears, alpha,NumberOfDaysPerYears):
    # Compute VaR with full valuation Monte-Carlo

    stockPrices = stockPrice * np.exp(logReturns * np.sqrt(riskMeasureTimeIntervalInYears*NumberOfDaysPerYears))

    # Compute Loss
    L_stock = - numberOfShares * ( stockPrices - stockPrice )
    P_tomorrow = black_scholes_put(stockPrices, strike, timeToMaturityInYears[0]-riskMeasureTimeIntervalInYears, rate[1], dividend, volatility).reset_index(drop=True)
    P_today = black_scholes_put(stockPrice, strike, timeToMaturityInYears[0], rate[0], dividend, volatility).reset_index(drop=True)
    L_der = - numberOfPuts * ( P_tomorrow.to_numpy() - P_today.to_numpy() )
    L = L_der + L_stock

    # Sort the loss
    L_sorted = -np.sort(-L, axis=0)
    index = int( len(L) * (1 - alpha) )

    # Compute VaR
    VaR = L_sorted[index - 1]

    return VaR


def DeltaNormalVaR(logReturns, numberOfShares, numberOfPuts, stockPrice, strike, rate, dividend, volatility, timeToMaturityInYears, riskMeasureTimeIntervalInYears, alpha, NumberOfDaysPerYears):
    # Compute VaR with Delta Normal method

    # Compute stock prices and returns
    new_returns = logReturns * np.sqrt(riskMeasureTimeIntervalInYears * NumberOfDaysPerYears)
    stockPrices = stockPrice * np.exp(new_returns)

    # Compute Loss
    L_stock = - numberOfShares * (stockPrices - stockPrice)
    d1 = (np.log(stockPrice / strike) + (rate - dividend + volatility ** 2 / 2) * timeToMaturityInYears) / (volatility * np.sqrt(timeToMaturityInYears))
    Delta = -np.exp(-dividend*timeToMaturityInYears)*norm.cdf(-d1)
    L_der = - numberOfPuts * (Delta * stockPrice) * new_returns
    L = L_der + L_stock

    # Sort the loss
    L_sorted = -np.sort(-L, axis=0)
    index = int(len(L) * (1 - alpha))

    # Compute VaR
    VaR = L_sorted[index - 1]

    return VaR


def black_scholes_call(S, K, T, r, q, sigma):
    d1 = (np.log(S/K) + (r - q + sigma**2/2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return - K * np.exp(-r * T) * norm.cdf(d2) + S * np.exp(-q * T) * norm.cdf(d1)