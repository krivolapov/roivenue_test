import pandas as pd
import numpy as np
#from scipy.stats import kurtosis, iqr, Shapiro
from scipy.optimize import differential_evolution
import warnings
import scipy.stats as stat

settings_estimator = {'alpha'   :   0.1,
                      'corr_thd':   0.9,
                      'time_window_regr':12,
                      'period'  :   'week',
                      'margin'  :   2.0,
                      'level'   :   'platform',
                      'dependant_var':'grossProfit',
                      'independatnt_var': 'marketingInvestment',
                      'filter_type': 'EWM',
                      'optimization_type':'Keep_budget',
                      'optimization_settings':'Conservative',
                      'budget_change': 0,
                      'client': ''
                      }

DateTimeID = ['periodStartDate']

usecols_ = [
    "account","adCampaign", "platformCode","currencyCode", "channelCode", "marketingInvestment",
    "impressions", "clicks", "visits", "conversions", "deliveries",
    "netRevenue", "grossProfit", "adGroup", "periodStartDate","businessUnit"
          ]

dtype_ = {
    'account': str,
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
settings_optimizer = {}

settings_trainer = {}

class EstimatorClass:

    def __init__(self):
        self.dataset_size = 0
        print("Init estimator class")


    def load_settings(self, settings):
        self.settings = settings
        print("Load Settings")


    def load_data(self, dataframe):
        self.data = dataframe
        self.dataset_size = len(self.data)
        print("Load data")


    def regr_calc(self):
        print('Regression method')


    def corr_data(self):
        #df = self.data

        def filter_df(df, alpha):
            return df.ewm(alpha = alpha, adjust = False).mean()


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


        def log_f(x, a, b):
            return a * (1-np.exp(-x/b)) #a * (1 - np.exp((x/b)))


        # function for genetic algorithm to minimize (sum of squared error)
        def sumOfSquaredError(parameterTuple):
            function = log_f # it's bad staff #TODO
            warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
            val = function(t_train, *parameterTuple)
            return np.sum((y_train - val) ** 2.0)


        def generate_Initial_Parameters(t_train, y_train,function):
            """ Calculate intitial paramiters for curve_fit function"""
            # min and max used for bounds
            maxX = max(t_train)
            #minX = min(t_train)
            maxY = max(y_train)
            #minY = min(y_train)
            maxXY = max(maxX, maxY)

            parameterBounds = []
            parameterBounds.append([-maxXY, maxXY]) # search bounds for a
            parameterBounds.append([-maxXY, maxXY]) # search bounds for b
            #parameterBounds.append([-maxXY, maxXY]) # search bounds for c

            # "seed" the numpy random number generator for repeatable results
            result = differential_evolution(sumOfSquaredError, parameterBounds, seed=3)
            return result.x


        def regression_calc(df, function):
            """Calculate paramiters of regression function
            input: pandas dataframe [x,y]
            output: list[paramiters], list[std diviation err of paramiters]"""
            from scipy.optimize import curve_fit
            t_train = df[df.columns[0]]  #
            y_train = df[df.columns[1]]
            geneticParameters = generate_Initial_Parameters(t_train, y_train, function)
            popt, pcov = curve_fit(function, t_train, y_train, geneticParameters)

            # class method return
            error  = np.sqrt(np.diag(pcov))
            return popt, error    
 

        business_unit_code = self.data['businessUnit'].unique().tolist()
        business_unit_code.sort()
        channel_code_code = self.data['channelCode'].unique().tolist()
        channel_code_code.sort()
        currency_code_code = self.data['currencyCode'].unique().tolist()
        currency_code_code.sort()
        account_code = self.data['account'].unique().tolist()


        df_time_selected = self.data[self.data['periodStartDate'] >= self.data['periodStartDate'].max() - pd.DateOffset(weeks=12)]

        output = pd.DataFrame()

        for i in business_unit_code:
    
    
            df_selected = df_time_selected[df_time_selected['businessUnit'] == i]
   
            if self.settings['level'] == 'platform':
                component_code = df_selected['platformCode'].unique().tolist()
        
            elif self.settings['level'] == 'compaign':
                component_code = df_selected['adCampaign'].unique().tolist()
        
            elif self.settings['level'] == 'channel':
                component_code = df_selected['channelCode'].unique().tolist()
    
            else:
                print('Incorrect component settings')
        
            for z in component_code:
                temp_dict = dict([('Client', ''), ('Business_unit', ''), ('Platform', ''),
                          ('Corr_coeff', 0), ('Pr_value', 0), ('a', 0), ('b', 0),
                          ('Mean', 0), ('R_std', 0), ('Invest', 0), ('Profit', 0),
                          ('Prof_last_week', 0), ('Invest_last_week', 0),
                          ('Prof_prev_week', 0), ('Invest_prev_week', 0),
                          ('Max_week_profit', 0), ('Max_week_invest', 0),
                          ('X',[]),('Y',[]),('Regr_X',[]),('Regt_Y',[])])
        
        
                if self.settings['level'] == 'platform':
                    df_platform = df_selected[df_selected['platformCode'] == z]
        
                elif self.settings['level'] == 'compaign':
                    df_platform = df_selected[df_selected['adCampaign'] == z]
        
                elif self.settings['level'] == 'channel':
                    df_platform = df_selected[df_selected['channelCode'] == z]
        

                df_business_selected_agg = select_agg_resample_df( df_platform, 'periodStartDate', 'week', use_nan=False )

               
                temp_dict['Max_week_profit'] = df_business_selected_agg.grossProfit[:].max()
                temp_dict['Max_week_invest'] = df_business_selected_agg.marketingInvestment[:].max()
        
                temp_dict['Prof_prev_week'] = df_business_selected_agg.grossProfit[-2:-1].max()
                temp_dict['Invest_prev_week'] = df_business_selected_agg.marketingInvestment[-2:-1].max()
        
                temp_dict['Prof_last_week'] = df_business_selected_agg.grossProfit[-1].max()
                temp_dict['Invest_last_week'] = df_business_selected_agg.marketingInvestment[-1].max()
        
                df_business_selected_agg = df_business_selected_agg.iloc[:-1, :] # get rid of last week in dataset     
        
                if self.settings['filter_type'] == 'EWM':
                    df_business_selected_agg_filter = filter_df( df_business_selected_agg, self.settings['alpha'] )
            
                elif self.settings['filter_type'] == 'None':
                    df_business_selected_agg_filter = df_business_selected_agg
            
                elif self.settings['filter_type'] == 'Median':
                    print("Median Filter have Not implemented yet") 
        
                else:
                    print("Incorrect Filter settings") 

                temp_dict['Client'] = account_code[0]
                temp_dict['Business_unit'] = i
                temp_dict['Component'] = str(i) + '/' + str(z)

                temp_dict['Profit'] = np.round( df_business_selected_agg['grossProfit'].sum(), 2 )
                temp_dict['Invest'] = np.round( df_business_selected_agg['marketingInvestment'].sum(), 2)

                sper_crr, P_value = stat.spearmanr( df_business_selected_agg_filter['marketingInvestment'], df_business_selected_agg_filter['grossProfit'])
                temp_dict['Corr_coeff'] = sper_crr
                temp_dict['Pr_value'] = P_value


                sorted_df = df_business_selected_agg_filter
        
                temp_dict['Mean'] = sorted_df['grossProfit'].describe()[1]
        
                t_train = sorted_df['marketingInvestment']
                y_train = sorted_df['grossProfit']
        
                temp_dict['X'] = df_business_selected_agg['marketingInvestment'].to_list()
                temp_dict['Y'] = df_business_selected_agg['grossProfit'].to_list()

        
                if sper_crr > self.settings['corr_thd']:
                    #print(i + ' business unit ' + z + ' platform has \t' + str(sper_crr) + ' corr coefficient')
                    try:

                        regr, err = regression_calc(sorted_df[['marketingInvestment', 'grossProfit']], log_f)

                        R = df_business_selected_agg['grossProfit'] -  log_f(df_business_selected_agg['marketingInvestment'], *regr)
                        R_std = R.describe()['std']

                        a, b = regr
                        temp_dict['a'] = a
                        temp_dict['b'] = b
                        temp_dict['R_std'] = R_std

                        # add 25% to maximum value for visualization of the prediction (+-10 or 20%)
                        x_space = np.linspace(0, (max(temp_dict['X']) * 1.25), num = len(temp_dict['X']))
                        temp_dict['Regr_X'] = x_space.tolist() 
                        temp_dict['Regt_Y'] = log_f(x_space, a, b)
                        temp_dict['Up_line_Y'] = log_f(x_space, a,b) + self.settings['margin'] * R_std
                        temp_dict['Down_line_Y'] = log_f(x_space, a,b) - self.settings['margin'] * R_std
                            
                    except:
                        print("Regression wasn't calculated")
                        temp_dict['a'] = 0
                        temp_dict['b'] = 0
                else:
                    temp_dict['a'] = 0
                    temp_dict['b'] = 0

                output = output.append(temp_dict, ignore_index=True)
        
        
        print("Correlation calculation executed")


        return output


class OptimizerClass:
    def __init__(self):
        print("Optimizer class Init")

    def load_settings(self, settings):
        self.settings = settings
        print("Load optimizer settings")

    def load_data(self, data):
        self.data = data
        print("Load optimizer data")

    def optimization(self):
        #filterd_out = output[~output['Platform'].isin(dropped_ch)]

        def log_f(x, a, b):
            return a * (1-np.exp(-x/b)) #a * (1 - np.exp((x/b)))

        #filterd_out = self.data
        corr_df = self.data[self.data['Corr_coeff'] > self.settings['corr_thd']]
        non_corr_df = self.data[self.data['Corr_coeff'] <= self.settings['corr_thd']]

        corr_df.reset_index(inplace=True, drop=True)
        non_corr_df.reset_index(inplace=True, drop=True)

        control_inv = corr_df['Invest'].sum()
        non_control_inv = non_corr_df['Invest'].sum()
        control_prof = corr_df['Profit'].sum()
        non_control_prof = non_corr_df['Profit'].sum()

        # budget optimizer preprocessing 
        corr_df['plus_invest'] = corr_df['Invest_prev_week'] * (1 + self.settings['max_invest_chng'])
        corr_df['minus_invest'] = corr_df['Invest_prev_week'] * (1 - self.settings['max_invest_chng'])


        corr_df['slope_plus'] = 0.0
        corr_df['slope_minus'] = 0.0


        corr_df['opt_invest'] = 0.0
        corr_df['profit_change'] = 0.0

        

        for i in range(len(corr_df)):
            profit_base = float(log_f(corr_df['Invest_prev_week'][i], corr_df['a'][i], corr_df['b'][i]))
    
            profit_plus = float(log_f(corr_df['plus_invest'][i], corr_df['a'][i], corr_df['b'][i]))
            slope_plus = float((profit_plus - profit_base))/float((corr_df['plus_invest'][i] - corr_df['Invest_prev_week'][i]))
            corr_df['slope_plus'][i] = slope_plus
    

    
            profit_minus = float(log_f(corr_df['minus_invest'][i], corr_df['a'][i], corr_df['b'][i]))
            slope_minus = float((profit_minus - profit_base))/float((corr_df['minus_invest'][i] - corr_df['Invest_prev_week'][i]))
            corr_df['slope_minus'][i] = -slope_minus
    
 
        invest_pool = (corr_df['Invest_prev_week'] - corr_df['minus_invest']).sum()


        zzz = corr_df.sort_values(by=['slope_plus'], ascending=False)
        zzz.reset_index(inplace=True, drop=True)



        accum = invest_pool + self.settings['change_investment']
        
        
        for i in range (len(zzz)):
            temp = (zzz['plus_invest'][i] - zzz['minus_invest'][i] )
            if temp <= accum:
                zzz['opt_invest'][i] = zzz['plus_invest'][i]
                accum = accum - temp
            else:
                zzz['opt_invest'][i] = zzz['minus_invest'][i] + accum
                accum = 0.0   

             
        for i in range (len(zzz)):
            profit_base = float(log_f(zzz['Invest_prev_week'][i], zzz['a'][i], zzz['b'][i]))
            optim_profit = float(log_f(zzz['opt_invest'][i], zzz['a'][i], zzz['b'][i]))
            zzz['profit_change'][i] = optim_profit - profit_base

        prof_chng = zzz['profit_change'].sum()
        prof_last_week = zzz['Prof_prev_week'].sum()
        all_prof_last_week = self.data['Prof_prev_week'].sum()


        result = pd.concat([non_corr_df, zzz], ignore_index=True)

        stats = {
                  'Invest_last_week':0,
                  'Controlled_Profit_last_week' : prof_last_week,
                  'Total_profit_last_week' : all_prof_last_week,
                  'Expected_profit_change' : prof_chng,
                  'Controlled_investment' : control_inv,
                  'Non-controlled_investment' : non_control_inv,
                  'Controlled_profit' : control_prof,
                  'Non_controlled_profit' : non_control_prof,
                  'Confidence_interval': self.settings['margin']
                }



        return result, stats


class TrainerClass:
    def __init__(self):
        print("constructor")

    def load_settings(self, settings):
        self.settings = settings
        print("load settings")

    def get_settings(self):
        print(self.settings)

    def __start(self):
        print("private method")

    def call_private(self):
        self.__start()



def Optimizer(  performance_dataset,
                alpha=0.1,
                correlation_treshold=0.9, 
                time_window_regession=84, 
                period='week',  
                confidence_interval=2.0, 
                level='platform',
                dependant_variable='grossProfit',   
                independant_variable = 'marketingInvestment', 
                filter_type = 'EWM',   
                change_investment = 0,   
                maximum_investment_change = 0.1):   
    """
    Documentation for optimizer function
    #-----------------------------------------------------------------------------------------
    Input parameters description
    #-----------------------------------------------------------------------------------------
            name                 |  data-type      | description
    ------------------------------------------------------------------------------------------
    performance_dataset           pandas dataframe  initial dataframe  
    alpha                         float             EWM (exponential weathered mean) filter parameter
    correlation_treshold          float             Non-linear Spermann correlation coefficient, optional
    time_window_regession         int               time window for regression calculation (in case of 'week' period in should be number multiply of 7), optional
    period                        str               ['day','week','month'], mandatory
    confidence_interval           float             Number of standard deviations, optional (1:63%, 2:95%, 3:99%)
    level                         str               ['platform','channel','campaign'], optional
    dependant_variable            str               ['grossProfit','impressions','clicks','visits','conversions','deliveries'], optional
    independant_variable          str               Optional, there is only one possible variable, optional
    filter_type                   str               Exponential Weighted Mean by default ['None','Median','EWM'], optional
    change_investment             int               0 means no investment change, only reallocation, any number means add or remove investment e.g. -200000, 35000, optional
    maximum_investment_change     float             Maximum change of investment allocation on one component against the last value. The value can be 0.1-0.2, optional

    #-----------------------------------------------------------------------------------------
    Output: ([pandas dataframe], [dictionary]) 
    #------------------------------------------------------------------------------------------

    Dataframe (component's related output):

    ------------------------------------------------------------------------------------------
            column name           |  data-type   |   discrimination
    ------------------------------------------------------------------------------------------
    client                           float          Client ID
    business_unit                    float          Business_id
    platform                         float          Name of the platform
    independend_value_total          float          Total investment in the selected period
    independend_value_max            float
    independend_value_last_week      float
    independend_value_last_week      float
    independend_value_list           float
    dependent_value_total            float          Total profit in the selected period
    dependent_value_max              float          Dependant value list according function's settings ('dependant_variable')
    dependend_value_last_week        float          Dependant value list according function's settings ('dependant_variable')
    dependend_value_previous_week    float
    dependend_value_list             [float]    
    central_line_list_x              [float]        List of x values for regression visualization
    central_line_list_y              [float]        List of y values for regression 
    upper_line_list_y                [float]        List of y values for upper confidence interval (Regression value + residual standard deviation * confidence_interval)
    lower_line_list_y                [float]        List of y values for lower confidence interval (Regression value - residual standard deviation * confidence_interval)

    # diagnostic part of output dataframe (not for visualization)

    _corr_coeff                      [float]       Spearman's rank correlation coefficient for all components 
    _pr_value:                       [float]       Probability of Spearman's test
    _a:                              [float]       Regression coefficient
    _b:                              [float]       Regression coefficient
    _mean:                           [float]       Mean value of period
    _r_std:                          [float]       Residual vector standard deviation

    #------------------------------------------------------------------------------
    Dictionary (overall statistics for dataframe):
    #------------------------------------------------------------------------------
          key name                 | data-type | description
    -------------------------------------------------------------------------------
    {
    'Invest_last_week':               float
    'Controlled_Profit_last_week':    float         Marketing investment we able to control (all components with with proper regression model )
    'Total_profit_last_week':         float      
    'Expected_profit_change':         float         Expected profit change based on optimization settings and controlled components of dataset 
    'Controlled_investment' :         float
    'Non-controlled_investment':      float
    'Controlled_frofit':              float
    'Non_controlled':                 float
    'Confidence_interval':            float
    }

    """


    settings = {**settings_estimator}

    settings['alpha'] = alpha
    settings['corr_thd'] = correlation_treshold
    settings['time_window_regr'] = time_window_regession
    settings['period'] = period
    settings['margin'] = confidence_interval
    settings['level'] = level
    settings['dependant_var'] = dependant_variable
    settings['independatnt_var'] = independant_variable
    settings['filter_type'] = filter_type
    settings['change_investment'] = change_investment
    settings['maximum_investment_change'] = maximum_investment_change


    estimator = EstimatorClass()
    estimator.load_settings(settings)
    estimator.load_data(performance_dataset)
    regr_df = estimator.corr_data()

    optimizer = OptimizerClass()
    optimizer.load_settings(settings_estimator)
    optimizer.load_data(regr_df)
    opt_result, stats = optimizer.optimization()

    return opt_result, stats