import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kurtosis, iqr, shapiro
from scipy.optimize import differential_evolution
import warnings

DateTimeID = ['periodStartDate']

usecols = [
    "adCampaign", "platformCode","currencyCode", "channelCode", "marketingInvestment",
    "impressions", "clicks", "visits", "conversions", "deliveries",
    "netRevenue", "grossProfit", "adGroup", "periodStartDate","businessUnit"
          ]

dtype = {
    'adCampaign' : str,
    'platformCode' : str,
    'businessUnit':str,
    'currencyCode':str,
    'channelCode' : str,
    'impressions' : int,
    'clicks' : int,
    'visits' : int,
    'conversions' : int,
    'deliveries' : int,
    'netRevenue' : float,
    'grossProfit' : float,
    'adGroup' : str,
    'marketingInvestment': float
}

stat_name = ['Mean','Std','Med','Sqew','Kurtosis','Var','Iqr','Shapiro-stat','Shapiro-Pr']
regr_name = ['a','b','c','a-err','b-err','c-err']
r2_stat_name = ['R2-mean','R2-std','R2-norm','R2-chi-sqr'] # func [line, log] if line -> c=NaN


# regression functions definition
def log_f(x, a, b, c):
    return a * (1-np.exp(-x/b)) + c #a * (1 - np.exp((x/b)))

def line_f(x, a, b):
    return a * x + b

def sine_f(x, a, b):
    return a * np.sin(b * x)

t_train = []
y_train = []

def main_statistics(df, precision = 3):
    """ Function input pandas series or numpy array
        input: dataframe, precision -> digits after point
        output: mean, std, median, skewness, kurtosis, variance, interquartile range, Shapiro-Wilk Test [Stat, Pr] """
    from scipy.stats import kurtosis, iqr, shapiro
    result = []
    count,mean, std, *all = df.describe()
    result.append(np.round(mean, precision))
    result.append(np.round(std, precision))
    result.append(np.round(df.median(), precision))
    result.append(np.round(df.skew(axis = 0, skipna = True), precision ))
    result.append(np.round(kurtosis(df, fisher=True), precision ))
    result.append(np.round(df.var(), precision))
    result.append((np.round(iqr(df, axis=0, keepdims=True), precision)).item())
    if len(df) > 2:
        stat, p = shapiro(df)
    else:
        stat = np.nan
        p = np.nan
    result.append(np.round(stat,precision))
    result.append(np.round(p,precision))
    return result

def regression_calc(df, function):
    """Calculate paramiters of regression function
    input: pandas dataframe [x,y]
    output: list[paramiters], list[std diviation err of paramiters]"""
    from scipy.optimize import curve_fit
    t_train = df[df.columns[0]]  #
    y_train = df[df.columns[1]]
    geneticParameters = generate_Initial_Parameters(t_train, y_train, function)
    popt, pcov = curve_fit(function, t_train, y_train, geneticParameters)

    error  = np.sqrt(np.diag(pcov))
    return popt, error

# function for genetic algorithm to minimize (sum of squared error)
def sumOfSquaredError(parameterTuple):
    function = log_f # it's bad staff #TODO
    warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
    val = function(t_train, *parameterTuple)
    return np.sum((y_train - val) ** 2.0)


# function for search initial value for regression parameters

def generate_Initial_Parameters(t_train, y_train,function):
    # min and max used for bounds
    maxX = max(t_train)
    minX = min(t_train)
    maxY = max(y_train)
    minY = min(y_train)
    maxXY = max(maxX, maxY)

    parameterBounds = []
    parameterBounds.append([-maxXY, maxXY]) # seach bounds for a
    parameterBounds.append([-maxXY, maxXY]) # seach bounds for b
    parameterBounds.append([-maxXY, maxXY]) # seach bounds for c

    # "seed" the numpy random number generator for repeatable results
    result = differential_evolution(sumOfSquaredError, parameterBounds, seed=3)
    return result.x



def select_business_unit(df, business_unit):
    """Select specific busines unti in dataframe """
    return df[df['businessUnit'] == business_unit]


def select_agg_resample_df(df, index, granularity, use_nan):
    """Function for select and group  data in dataframe
    input: pandas dataframe, name of index, granularity['week','month'] default as in dataframe
    return: aggregated dataframe with extra column with indexes [ROI, CPI, CPC]"""
    group_df = df.groupby(['periodStartDate']).agg({
    'netRevenue': 'sum','marketingInvestment': 'sum', 'visits': 'sum', 'conversions': 'sum',
    'deliveries': 'sum', 'impressions': 'sum', 'clicks': 'sum', 'grossProfit': 'sum' })   

    if granularity == 'week':
        group_agg_df = group_df.resample('W-MON').agg('sum')
    elif granularity == 'month':
        group_agg_df = group_df.resample('M', convention='end').agg('sum')
    elif granularity == 'day':
        group_agg_df = group_df
    else:
        raise ValueError("Incorrect aggregation period, shuld be 'day', 'week' or 'month'")

    group_agg_df['ROI'] = (group_agg_df['netRevenue']/group_agg_df['marketingInvestment'])
    group_agg_df['CPI'] = (group_agg_df['marketingInvestment']/group_agg_df['impressions'])
    group_agg_df['CPC'] = (group_agg_df['marketingInvestment']/group_agg_df['clicks'])
    group_agg_df['POI'] = (group_agg_df['marketingInvestment']/group_agg_df['grossProfit']) # Profit over investment

    if use_nan == False:
        group_agg_df.fillna(0)
    else:
        pass
    return group_agg_df


def stat_index_platform(df, platform_code, granularity, index, precision):
    """Calculate marketing indexis for all platforms
    input: dataframe, granularity, index[ROI, CPI, CPC], precision
    output: dataframe with statistics for all platforms"""
    statistics_ROI = []
    for i in platform_code:
        new_df = df[df['platformCode'] == i]

        selected_df = select_agg_resample_df(new_df, 'periodStartDate', granularity, use_nan = False)      

        stat = main_statistics(selected_df[index], precision)

        statistics_ROI.append(stat)
        
    statistics_df = pd.DataFrame(statistics_ROI, columns = stat_name)
    statistics_df = statistics_df.set_index([pd.Index(platform_code)])
    return statistics_df


def filter_df(df, alpha):
    return df.ewm(alpha=alpha, adjust=False).mean()


