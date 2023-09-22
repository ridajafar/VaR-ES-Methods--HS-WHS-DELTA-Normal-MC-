# MAIN ASSIGNMENT 5 - GROUP 7

# Import the required packages
import numpy as np
import pandas as pd
import scipy
import datetime
import utilities

# Set the seed for reproducibility
np.random.seed(42)

# Import data
path = '/Users/redajafar/PycharmProjects/Assignment5'
stocks_data = pd.read_csv(path + '/EUROSTOXX50_2023_Dataset.csv')
stocks_data.fillna(axis=0, method='ffill', inplace=True) # setting missing values to the previous available one
indices_data = pd.read_csv(path + '/_indexes.csv')



# Exercise 0

# Parameters
datestart = datetime.date(2016, 3, 21)
dateend = datetime.date(2019, 3, 20)
specific_names = ['Adidas', 'Allianz', 'Munich Re', 'L\'Oréal'] # Change here if you need other stocks
alpha = 0.95
portfolioValue = 10**7
riskMeasureTimeIntervalInDay = 1

# Compute weights and returns
weights = np.ones(len(specific_names))/len(specific_names)
returns = utilities.relevant_returns(stocks_data, indices_data, datestart, dateend, specific_names)[0]

# Compute VaR and ES
[VaR, ES] = utilities.AnalyticalNormalMeasures(alpha, weights, portfolioValue, riskMeasureTimeIntervalInDay, returns)
print('Exercise 0 \n VaR: ', VaR, ', ES:', ES, '\n')



# Case Study 1
alpha_1 = 0.99


# Portfolio 1

# Parameters
specific_names_1 = ['Danone', 'Sanofi', 'TotalEnergies', 'Volkswagen Group']
n_shares_1 = np.array([20, 20, 25, 10])*10**3
numberOfSamplesToBootstrap = 200

# Compute returns, weights and portfolio value
[returns_1,stocks_1] = utilities.relevant_returns(stocks_data, indices_data, datestart, dateend, specific_names_1) # compute returns and stocks
stocks_end = stocks_1.iloc[len(stocks_1)-1,:] # select stocks
portfolioValue_1 = np.sum(stocks_end*n_shares_1) # compute portfolio value
weights_1 = stocks_end*n_shares_1/portfolioValue_1 # compute weights

# Compute VaR and ES for the Historical simulation
[VaR_1_1, ES_1_1, L, Ls] = utilities.HSMeasurements(returns_1, alpha_1, weights_1, portfolioValue_1, riskMeasureTimeIntervalInDay)
print('Exercise 1 \n Portfolio 1, Historical simulation -> VaR: ', VaR_1_1, ', ES:', ES_1_1, '\n')
print('Exercise 1 \n Portfolio 1, Historical simulation -> L: ', L, ', ES:', Ls, '\n')
# Perform Bootstrap
samples = utilities.bootstrapStatistical(numberOfSamplesToBootstrap, returns_1)

# Compute VaR and ES for the Bootstrapped Historical simulation
[VaR_1_2, ES_1_2] = utilities.HSMeasurements(samples, alpha_1, weights_1, portfolioValue_1, riskMeasureTimeIntervalInDay)
print(' Portfolio 1, Bootstrapped Historical simulation -> VaR: ', VaR_1_2, ', ES:', ES_1_2, '\n')

# Plausibility check 1
VaR_1_check = utilities.plausibilityCheck(returns_1, weights_1, alpha_1, portfolioValue_1, riskMeasureTimeIntervalInDay)
print(' Portfolio 1, check -> VaR: ', VaR_1_check, '\n')


# Portfolio 2

# Parameters
specific_names_2 = ['Adidas', 'Airbus', 'BBVA', 'BMW', 'Schneider Electric']
lam = 0.97

# Compute returns and weights
returns_2 = utilities.relevant_returns(stocks_data, indices_data, datestart, dateend, specific_names_2)[0]
weights_2 = np.ones(len(returns_2.columns))/len(returns_2.columns)

# Compute VaR and ES
[VaR_2, ES_2] = utilities.WHSMeasurements(returns_2, alpha_1, lam, weights_2, portfolioValue, riskMeasureTimeIntervalInDay)
print(' Portfolio 2, Weighted Historical simulation -> VaR: ', VaR_2, ', ES:', ES_2, '\n')

# Plausibility check 2
VaR_2_check = utilities.plausibilityCheck(returns_2, weights_2, alpha_1, portfolioValue, riskMeasureTimeIntervalInDay)
print(' Portfolio 2, check -> VaR: ', VaR_2_check, '\n')


# Portfolio 3

# Parameters
specific_names_3 = indices_data['Name'][0:21].tolist()
specific_names_3.pop(3)
numberOfPrincipalComponents = list(range(1,21))
H = 10

# Compute returns and weights
returns_3 = utilities.relevant_returns(stocks_data, indices_data, datestart, dateend, specific_names_3)[0]
weights_3 = np.ones(len(specific_names_3))/len(specific_names_3)
Covariance = returns_3.cov()
MeanReturns = returns_3.mean()

# Compute VaR and ES
for num in numberOfPrincipalComponents:
    [VaR_3, ES_3] = utilities.PrincCompAnalysis(Covariance, MeanReturns, weights_3, H, alpha_1, num, portfolioValue)
    print(' Portfolio 3, PCA with ', num, ' principal components -> VaR: ', VaR_3, ', ES:', ES_3, '\n')

# Plausibility check 3
VaR_3_check = utilities.plausibilityCheck(returns_3, weights_3, alpha_1, portfolioValue, H)
print(' Portfolio 3, check -> VaR: ', VaR_3_check, '\n')


# Exercise 2

# Parameters
notional = 25870000
specific_name = 'Vonovia'
dateput = datetime.date(2023, 4, 5)
dateend = datetime.date(2023, 1, 31)
datestart = datetime.date(2021, 2, 1) # 29-01 nostro
datetomorrow = datetime.date(2023, 2, 1)
strike = 25
dividend = 3.1/100
volatility = 15.4/100
NumberOfDaysPerYears = 365
riskMeasureTimeIntervalInYears = 10/NumberOfDaysPerYears
start = np.where(np.isin(stocks_data['Date'], str(datestart)))[0][0]
end = np.where(np.isin(stocks_data['Date'], str(dateend)))[0][0]
timeToMaturityInYears = np.zeros(2)
timeToMaturityInYears[0] = (dateput - dateend).days/NumberOfDaysPerYears
timeToMaturityInYears[1] = (dateput - datetomorrow).days/NumberOfDaysPerYears
alpha = 0.99

# Obtain rates
rates_data = pd.read_csv(path + '/Rates.csv')
rate = np.zeros(2)
rate[0] = rates_data.iloc[3,1]/100
rate[1] = (rate[0]*timeToMaturityInYears[0] + rates_data.iloc[1,1]/(100*NumberOfDaysPerYears))/timeToMaturityInYears[1]

# Compute returns
[returns,stocks] = utilities.relevant_returns(stocks_data, indices_data, datestart, dateend, specific_name)
stockPrice = stocks.iloc[-1]
numberOfShares = int(notional/stockPrice)
numberOfPuts = numberOfShares

# Compute VaR with MC
VaR_1 = utilities.FullMonteCarloVaR(returns, numberOfShares, numberOfPuts, stockPrice, strike, rate, dividend, volatility, timeToMaturityInYears, riskMeasureTimeIntervalInYears, alpha, NumberOfDaysPerYears)[0]
print('Exercise 2 \n Full Monte-Carlo VaR: ', VaR_1, '\n')

# Compute VaR with delta normal method
VaR_2 = utilities.DeltaNormalVaR(returns, numberOfShares, numberOfPuts, stockPrice, strike, rate[0], dividend, volatility, timeToMaturityInYears[0], riskMeasureTimeIntervalInYears, alpha, NumberOfDaysPerYears)[0]
print(' Delta Normal VaR: ', VaR_2, '\n')



# Exercise 3
np.random.seed(42)

# Parameters
notional = 50*10**6
T = 4.0
volatility = 0.25
recovery = 0.4
specific_name = ['Intesa Sanpaolo']
datestart = datetime.date(2023, 1, 31)
dateend = datestart

# Compute defaultable discount factors
SurvProb_data = pd.read_csv(path + '/Survival_Probabilities.csv')
survival_probabilities = SurvProb_data.iloc[0:4, 1].to_numpy()
Disc_data = pd.read_csv(path + '/Discounts.csv')
B_bar = (survival_probabilities * Disc_data.iloc[:, 1]).to_numpy()
survival_probabilities = np.insert(survival_probabilities, 0, 1)
dates = Disc_data.iloc[:, 0].to_numpy()
dates = np.insert(dates, 0, '2-Feb-2023')
TTMs = np.array([(datetime.datetime.strptime(x,'%d-%b-%Y').date() - datetime.datetime.strptime(y,'%d-%b-%Y').date()).days/365 for x, y in zip(dates[1:len(dates)], dates[0:-1])]) # check

# GBM parameters
S0 = utilities.relevant_returns(stocks_data, indices_data, datestart, dateend, specific_name)[1].iloc[0, 0]
M = 10**7
S = np.zeros((len(dates),M))
S[0,:] = S0*np.ones((1,M))
discounts = Disc_data.iloc[:,1].to_numpy()
discounts = np.insert(discounts, 0, 1)
forward_rates = -np.log(discounts[1:len(discounts)]/discounts[0:-1])/TTMs

# Simulate S
for i in range(1,len(dates)):
    g = np.random.randn(M)
    S[i,:] = S[i-1,:]*np.exp( (forward_rates[i-1]-volatility**2/2)*TTMs[i-1] + np.sqrt(TTMs[i-1]) * volatility * g )

payoffs = np.mean( np.maximum( S[1:,:] - S[0:-1,:] , np.zeros((len(TTMs),M))), axis=1 )

# Compute prices
price_without_default = np.sum( discounts[1:]*payoffs )
print('Exercise 3 \n price of Cliquet option without default: ', price_without_default, '\n')

price_with_default = np.sum( B_bar*payoffs )
print(' price of Cliquet option with default: ', price_with_default, '\n')