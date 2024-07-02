from app.services.common_imports import *
from app.services import extrafunctions as extfun
api_key = "78K1C9N5E01K7XLV"
#### test
#### global variables
lst_all_sectors = ["Technology","Financial Services","Healthcare","Consumer Cyclical","Basic Materials",
                   "Industrials", "Communication Services", "Consumer Defensive", "Utilities", "Real Estate", "Energy"]

##### some configgs

###only daily supported for now
data_frequency = "d"  #### should be changebale from the app -- can be M (for monthly) or D

data_frequency = data_frequency.lower()
if data_frequency not in ["d", "m","w"]: data_frequency = "d"

x = {"d":252, "w":52, "m":12}

frequency_scaler = x[data_frequency]

risk_free_data, ff_factors = extfun.get_ff_data()

factors_dict = {"style":{}}
for factor in ff_factors.columns:
  factors_dict["style"][factor] = {"return_data": ff_factors[factor]}

all_factor_returns = pd.concat([i["return_data"] for j,i in  factors_dict["style"].items()], axis =1).dropna()
all_factor_returns

default_alpha = 0.000000000000001

#------------------------------ Class Construction Begins -----------------------------------
#------------------------------ Class Construction Begins -----------------------------------
class PortfolioConstruction:

  def __init__(self):

    self.all_equity_data = {}
    self.all_etf_data = {}
    self.startDate = None
    self.endDate = None


#------------------------------------ functions to add and define tickers #------------------------------------


  def add_equity_tickers(self, tickers, tickers_config ={}):

    for ticker in tickers:
      ticker_config = tickers_config.get(ticker,{})
      ticker_data = get_basic_ticker_data(ticker,ticker_config)
      ticker_data= ticker_data | get_ticker_info(ticker)
      if "sector" in ticker_config and ticker_config.get("sector", "") not in [None,""]:
        ticker_data["sector"] = ticker_config["sector"]
      self.all_equity_data[ticker] = ticker_data

    self.get_sector_weights()

  def add_tickers_using_dataset(self, price_data):

    price_data.set_index("Date", inplace = True)
    tickers = price_data.columns.tolist()

    price_data.index = pd.to_datetime(price_data.index)
    price_data = price_data.resample(data_frequency).last().astype(float)

    price_data.index = price_data.index.strftime('%Y-%m-%d')
    for ticker in tickers:
      #price_data[ticker] = price_data[ticker].resample(data_frequency).last().astype(float)
      self.all_equity_data[ticker] = {"price_data":price_data[ticker], "sector":"No Sector Data", "include":True}


    self.get_sector_weights()

  def select_tickers_to_use(self, tickers_to_use):

    for ticker in tickers_to_use:
      self.all_equity_data[ticker]["include"] = True

    for ticker in self.all_equity_data:
      if ticker not in tickers_to_use:
        self.all_equity_data[ticker]["include"] = False



  def add_or_update_tickers_configs(self, tickers_config):

    for ticker in self.all_equity_data:
      ticker_config = tickers_config.get(ticker,None)
      if ticker_config is None: continue
      else:
        for config in ticker_config:
          config_val = ticker_config[config]
          if config_val is not None: self.all_equity_data[ticker][config] = config_val

    self.get_sector_weights()



  def add_etf_tickers(self, tickers, ticker_config ={}):
    for ticker in tickers:
      ticker_limit = ticker_config.get(ticker,{})
      ticker_data = get_basic_ticker_data(ticker,ticker_config)
      ticker_data =ticker_data |  get_etf_info(ticker)
      self.all_equity_data[ticker] = ticker_data



  def set_model_period(self, startDate=None, endDate=None):
    self.startDate = startDate
    self.endDate = endDate


  def get_sector_weights(self):
    allsectors = [info['sector'] for info in self.all_equity_data.values() if 'sector' in info]
    allsectors = list(set(allsectors))
    for ticker in self.all_equity_data:
      self.all_equity_data[ticker]["sector_weights"] = pd.Series({sector: 1 if sector == self.all_equity_data[ticker]["sector"] else 0 for sector in allsectors}, name = ticker)


#------------------------------------ End of Ticker and Data setup #------------------------------------



#------------------------------------ MeanVariance Optimization #------------------------------------

  def MVO(self,optimize_for, hist_decay, max_risk = None, min_return = None, max_leverage =0, short_sell = False, portfolio_sector_exposures_limits = None, portfolio_factor_exposures_limits = None):

    all_data = self.all_equity_data | self.all_etf_data
    all_data = get_all_data_dict_for_usable_tickers(all_data)

    optimize_for = optimize_for.lower()
    allowed_optimize_for = ["risk", "return"]

    if optimize_for not in allowed_optimize_for:
      raise Exception(f"error, optimize variable must be in {allowed_optimize_for}, not {optimize_for}")


    if optimize_for in ["risk"] and min_return is None:

      raise Exception(f"error, need to provide minimum portfolio return when optimizing for risk or diversification")


    all_ticker_limits = get_clean_ticker_bounds_dict(all_data)

    expected_return_vector,std_vector,covar_df, excess_return_vector, expected_rf_rate, one_year_return_df = get_EWMA_std_return_cov(all_data, alpha = hist_decay, startDate = self.startDate, endDate = self.endDate)
    weights = cp.Variable(len(expected_return_vector))

    leverage_amt = cp.Variable(1)
    factor_loadings = factor_model(all_data, all_factor_returns, alpha = hist_decay)


    expected_return, portfolio_variance, excess_return, \
    diversification_ratio, portfolio_sector_exposures_dict, \
    portfolio_factor_exposure_dict = calculate_all(all_data, weights, expected_return_vector, excess_return_vector,covar_df, 
                                                   factor_loadings,leverage_amt = leverage_amt, expected_rf_rate = expected_rf_rate )


    if optimize_for == "return" : objective = cp.Maximize(expected_return)
    elif optimize_for == "risk": objective = cp.Minimize(portfolio_variance)
    elif optimize_for == "diversification ratio": objective = cp.Minimize(diversification_ratio)

    max_variance = None if max_risk is None else max_risk **2

    constraints = create_constraints(weights,ticker_limits = all_ticker_limits, variance= portfolio_variance, expected_return = expected_return, short_sell_allowed= short_sell,
                                     min_return = min_return, max_variance =  max_variance, portfolio_sector_exposures_dict = portfolio_sector_exposures_dict,
                                     portfolio_sector_exposures_limits =portfolio_sector_exposures_limits, portfolio_factor_exposure_dict = portfolio_factor_exposure_dict,
                                     portfolio_factor_exposures_limits =portfolio_factor_exposures_limits, leverage_amt = leverage_amt, max_leverage=max_leverage  )

    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Extract optimal weights
    if weights.value is None:
      return "infeasible optimization"
    optimal_weights = weights.value.round(2).tolist()
    optimal_weights = dict(zip(list(all_data.keys()), optimal_weights))

    optimal_port_data = calculate_portfolio_metrics(pd.Series(optimal_weights), expected_return_vector, covar_df, std_vector, all_data,  one_year_return_df)

    return optimal_port_data




#------------------------------------ Optimize ratio [Sharpe and diversification ratio] #------------------------------------



  def optimize_ratio(self,ratio, hist_decay, allow_short_sell = False, portfolio_sector_exposures_limits = None, portfolio_factor_exposures_limits = None):

    all_data = self.all_equity_data | self.all_etf_data
    all_data = get_all_data_dict_for_usable_tickers(all_data)

    ratio = ratio.lower()
    allowed_ratios = ["sharpe ratio", "diversification ratio"]

    if ratio not in allowed_ratios:
      raise Exception(f"error, ratio must be one of {allowed_ratios}, not {ratio}")

    all_ticker_limits = get_clean_ticker_bounds_dict(all_data)

    expected_return_vector,std_vector,covar_df,excess_return_vector, expected_rf_rate, one_year_return_df = get_EWMA_std_return_cov(all_data, alpha = hist_decay, startDate = self.startDate, endDate = self.endDate)

    weights = cp.Variable(len(expected_return_vector))

    factor_loadings = factor_model(all_data, all_factor_returns, alpha = hist_decay)
    expected_return, portfolio_variance, excess_return, \
    diversification_ratio, portfolio_sector_exposures_dict, \
    portfolio_factor_exposure_dict = calculate_all(all_data, weights, expected_return_vector, excess_return_vector,covar_df, 
                                                   factor_loadings )


    #### find min possible risk
    min_risk_objective = cp.Minimize(portfolio_variance)
    constraints = create_constraints( weights,  variance= portfolio_variance, expected_return = expected_return, ticker_limits = all_ticker_limits, short_sell_allowed= allow_short_sell,
                                     strict_bound = True, portfolio_sector_exposures_dict = portfolio_sector_exposures_dict, portfolio_sector_exposures_limits =portfolio_sector_exposures_limits, portfolio_factor_exposure_dict = portfolio_factor_exposure_dict, portfolio_factor_exposures_limits =portfolio_factor_exposures_limits    )
    problem = cp.Problem(min_risk_objective, constraints)
    problem.solve()
    if portfolio_variance.value is None:
      return "infeasible optimization"
    min_possible_volatility = round(portfolio_variance.value **0.5,2)

    ### find max possible risk -- not DCP so cant solve for it directly

    #1. find max possible return
    max_return_objective = cp.Maximize(expected_return)
    problem = cp.Problem(max_return_objective, constraints)
    problem.solve()
    max_possible_return = expected_return.value
    print(max_possible_return)
    #2. find min variance to get that max possible return
    min_risk_objective = cp.Minimize(portfolio_variance)
    constraints = create_constraints( weights, min_return = max_possible_return, ticker_limits = all_ticker_limits, variance= portfolio_variance, expected_return = expected_return, strict_bound = True,short_sell_allowed= allow_short_sell,
                                     portfolio_sector_exposures_dict = portfolio_sector_exposures_dict, portfolio_sector_exposures_limits =portfolio_sector_exposures_limits, portfolio_factor_exposure_dict = portfolio_factor_exposure_dict, portfolio_factor_exposures_limits =portfolio_factor_exposures_limits  )

    problem = cp.Problem(min_risk_objective, constraints)
    problem.solve()
    max_possible_volatility = round(portfolio_variance.value **0.5,2)
    print(min_possible_volatility, max_possible_volatility)


    ######## Sharpe Ratio Calcuation #########

    if ratio =="sharpe ratio":
      optimization_plot = {"risk":[], "return":[], "sharpe":[], "weights": []}

      ##### get efficient frontier
      objective = cp.Maximize(excess_return)

      best_ratio = 0
      for max_risk in np.linspace(min_possible_volatility,max_possible_volatility,200):

        constraints = create_constraints(weights, ticker_limits = all_ticker_limits, variance= portfolio_variance, expected_return = expected_return, strict_bound = True, short_sell_allowed= allow_short_sell,
                                         max_variance =  max_risk**2, portfolio_sector_exposures_dict = portfolio_sector_exposures_dict, portfolio_sector_exposures_limits =portfolio_sector_exposures_limits,portfolio_factor_exposure_dict = portfolio_factor_exposure_dict, portfolio_factor_exposures_limits =portfolio_factor_exposures_limits   )
        problem = cp.Problem(objective, constraints)
        problem.solve()

        if expected_return.value is not None:

          expect_return = round(expected_return.value,6)
          excess_return = round(objective.value,6)

          volatility = round(portfolio_variance.value **0.5,5)
          sharpe_ratio = excess_return/volatility
          optimization_plot["risk"].append(volatility)
          optimization_plot["return"].append(expect_return)
          _weights = weights.value.round(2).tolist()
          _weights = dict(zip(list(all_data.keys()), _weights))
          optimization_plot["weights"].append(_weights)
          optimization_plot["sharpe"].append(sharpe_ratio)


      max_sharpe_index = max(range(len(optimization_plot["sharpe"])), key=optimization_plot["sharpe"].__getitem__)

      best_sharpe_info_dict = {"weights":optimization_plot["weights"][max_sharpe_index],
                               "volatility":optimization_plot["risk"][max_sharpe_index] ,
                               "sharpe":optimization_plot["sharpe"][max_sharpe_index] ,
                               "return":optimization_plot["return"][max_sharpe_index] }

      print(optimization_plot)

      plt.plot(optimization_plot["risk"], optimization_plot["return"])
      return best_sharpe_info_dict


    ######## diversification ratio ###########

    if ratio == "diversification ratio":

      '''
      logic explained --
      find min and max range of risk values possible for portfolio of n assets
      for each possible risk value
      maximize the weighted sum of asset risk (i.e risk assuming no correlation)
      calculate diversification ratio for each iteration
      the largest value would be the maximum possible diversification ratio for the portfolio of n assest.
      '''

      objective = cp.Maximize(cp.sum(cp.multiply(weights, std_vector)))

      best_ratio = 0

      for possible_risk in np.linspace(min_possible_volatility,max_possible_volatility,250):

        constraints = create_constraints(weights,ticker_limits = all_ticker_limits, variance= portfolio_variance, expected_return = expected_return, strict_bound = True, max_variance =  possible_risk**2, portfolio_sector_exposures_dict = portfolio_sector_exposures_dict, portfolio_sector_exposures_limits =portfolio_sector_exposures_limits, portfolio_factor_exposure_dict = portfolio_factor_exposure_dict, portfolio_factor_exposures_limits =portfolio_factor_exposures_limits   )
        problem = cp.Problem(objective, constraints)
        problem.solve()

        if objective.value is not None:
          weight_array = weights.value.round(10).tolist()
          weight_array = dict(zip(list(all_data.keys()), weight_array))
          portfolio_metrics = calculate_portfolio_metrics(pd.Series(weight_array), expected_return_vector, covar_df, std_vector, all_data,one_year_return_df)
          if portfolio_metrics["diversification ratio"]> best_ratio:
            best_ratio = portfolio_metrics["diversification ratio"]
            port_maximizing_diversification_ratio = portfolio_metrics

      return port_maximizing_diversification_ratio



#------------------------------------ Risk Parity and Budgeting #------------------------------------


  def risk_parity(self, hist_decay = None, risk_budget:dict = None):
    '''
    risk budget would be a dict like {ticker: budget, ...}, the sum should not be 1 if all assets are given a budget
    if >1 will be scaled to 1
    if <1 and not all assets are given a budget, rest of budget will be distributed equaly among assets assuming
    if <1 and all assets are given a budget, will be scaled to 1
    '''



    all_data = self.all_equity_data | self.all_etf_data
    all_data = get_all_data_dict_for_usable_tickers(all_data)

    expected_return_vector,std_vector,covar_df, excess_return_vector, expected_rf_rate, one_year_return_df = get_EWMA_std_return_cov(all_data, alpha =  hist_decay, startDate = self.startDate, endDate = self.endDate)
    weights = cp.Variable(len(expected_return_vector))
    expected_return = calculate_portfolio_expected_return(weights, expected_return_vector)

    b = np.ones(len(expected_return_vector)) / len(expected_return_vector) #risk parity

    if risk_budget is not None:
      b = [risk_budget.get(asset,None) for asset in all_data]
      sum_budget = sum([x for x in b if x is not None])

      if sum_budget>=1:
        b = [x/sum_budget if x is not None else 0 for x in b]

      if sum_budget<1:
        num_no_budget = b.count(None)
        if num_no_budget>0:
          b = [x if x is not None else (1-sum_budget)/num_no_budget for x in b ]
        else: b = b/sum_budget

    print(b)

    # 0.5 var = b*log(weights)

    objective = 0.5 * cp.quad_form(weights, covar_df.values) - cp.sum(cp.multiply(b, cp.log(weights)))

    constr = [weights >= 0] # constraint
    problem = cp.Problem(cp.Minimize(objective), constr)
    problem.solve()

    weights = weights/cp.sum(weights)

    weight_array = weights.value.round(10).tolist()
    weight_array = dict(zip(list(all_data.keys()), weight_array))
    risk_parity_portfolio = calculate_portfolio_metrics(pd.Series(weight_array), expected_return_vector, covar_df, std_vector, all_data, one_year_return_df)
    print(x)

    b_w = cp.multiply(weights, covar_df.values @ weights)

    return risk_parity_portfolio



#------------------------------------ CVaR Optimization #------------------------------------


  def CVaR_Efficient_Frontier(self, beta=0.99, target_return = None, portfolio_sector_exposures_limits = None, portfolio_factor_exposures_limits = None):

    '''
    if target return is None it will generate the efficient frontier

    '''

    all_data = self.all_equity_data | self.all_etf_data
    all_data = get_all_data_dict_for_usable_tickers(all_data)

    all_ticker_limits = get_clean_ticker_bounds_dict(all_data)
    expected_return_vector,std_vector,covar_df,excess_return_vector, expected_rf_rate, one_year_return_df = get_EWMA_std_return_cov(all_data, startDate = self.startDate, endDate = self.endDate)
    weights = cp.Variable(len(expected_return_vector))


    factor_loadings = factor_model(all_data, all_factor_returns)
    expected_return, portfolio_variance, excess_return, \
    diversification_ratio, portfolio_sector_exposures_dict, \
    portfolio_factor_exposure_dict = calculate_all(all_data, weights, expected_return_vector, excess_return_vector,covar_df, 
                                                   factor_loadings )


    problem_constraints = create_constraints( weights, expected_return = expected_return, ticker_limits = all_ticker_limits, strict_bound = True, 
                                             portfolio_sector_exposures_dict = portfolio_sector_exposures_dict, portfolio_sector_exposures_limits =portfolio_sector_exposures_limits, portfolio_factor_exposure_dict = portfolio_factor_exposure_dict, portfolio_factor_exposures_limits =portfolio_factor_exposures_limits    )

    #### min possible return

    min_return_objective = cp.Minimize(expected_return)
    problem = cp.Problem(min_return_objective, problem_constraints)
    problem.solve()
    min_possible_return = expected_return.value

    ### max possible return

    max_return_objective = cp.Maximize(expected_return)
    problem = cp.Problem(max_return_objective, problem_constraints)
    problem.solve()
    max_possible_return = expected_return.value

    for return_i in np.linspace(min_possible_return, max_possible_return, 10):

      hist_returns = one_year_return_df.copy()

      problem_constraints = create_constraints( weights, expected_return = expected_return, ticker_limits = all_ticker_limits, strict_bound = True, 
                                             portfolio_sector_exposures_dict = portfolio_sector_exposures_dict, portfolio_sector_exposures_limits =portfolio_sector_exposures_limits, portfolio_factor_exposure_dict = portfolio_factor_exposure_dict, portfolio_factor_exposures_limits =portfolio_factor_exposures_limits    )
      weight_array = general_min_CVaR_problem(return_i,beta, expected_return, weights, problem_constraints, hist_returns, all_data )
      
      portfolio_metrics = calculate_portfolio_metrics(pd.Series(weight_array), expected_return_vector, covar_df, std_vector, all_data,  hist_returns, CVaR_betas=[beta])
      print(portfolio_metrics)




  def min_CVaR(self, target_return, beta=0.95, portfolio_sector_exposures_limits = None, portfolio_factor_exposures_limits = None):

    all_data = self.all_equity_data | self.all_etf_data
    all_data = get_all_data_dict_for_usable_tickers(all_data)

    all_ticker_limits = get_clean_ticker_bounds_dict(all_data)
    print(all_ticker_limits)
    expected_return_vector,std_vector,covar_df,excess_return_vector, expected_rf_rate, one_year_return_df = get_EWMA_std_return_cov(all_data, startDate = self.startDate, endDate = self.endDate)
    weights = cp.Variable(len(expected_return_vector))


    factor_loadings = factor_model(all_data, all_factor_returns)
    expected_return, portfolio_variance, excess_return, \
    diversification_ratio, portfolio_sector_exposures_dict, \
    portfolio_factor_exposure_dict = calculate_all(all_data, weights, expected_return_vector, excess_return_vector,covar_df, 
                                                   factor_loadings )

    problem_constraints = create_constraints( weights, expected_return = expected_return, ticker_limits = all_ticker_limits, strict_bound = True, 
                                             portfolio_sector_exposures_dict = portfolio_sector_exposures_dict, portfolio_sector_exposures_limits =portfolio_sector_exposures_limits, portfolio_factor_exposure_dict = portfolio_factor_exposure_dict, portfolio_factor_exposures_limits =portfolio_factor_exposures_limits    )


    hist_returns = one_year_return_df.copy()

    problem_constraints = create_constraints( weights, expected_return = expected_return, ticker_limits = all_ticker_limits, strict_bound = True, 
                                            portfolio_sector_exposures_dict = portfolio_sector_exposures_dict, portfolio_sector_exposures_limits =portfolio_sector_exposures_limits, portfolio_factor_exposure_dict = portfolio_factor_exposure_dict, portfolio_factor_exposures_limits =portfolio_factor_exposures_limits    )
    weight_array = general_min_CVaR_problem(target_return, beta, expected_return, weights,problem_constraints, hist_returns, all_data )
    
    portfolio_metrics = calculate_portfolio_metrics(pd.Series(weight_array), expected_return_vector, covar_df, std_vector, all_data,  hist_returns, CVaR_betas=[beta])

    return portfolio_metrics


#------------------------------------ Benchmark Methods #------------------------------------

  def benchmark_portfolios(self, model, hist_decay):

    '''
    no optimization needed here
    '''
    model = model.lower()
    allowed_models = ["equal weight", "inverse volatility", "return weighted",
                      "inverse variance"]

    if model not in allowed_models:
      raise Exception(f"error, model must be one of {allowed_models}, not {model}")


    all_data = self.all_equity_data | self.all_etf_data
    all_data = get_all_data_dict_for_usable_tickers(all_data)

    expected_return_vector,std_vector,covar_df, excess_return_vector , expected_rf_rate, one_year_return_df = get_EWMA_std_return_cov(all_data, alpha =  hist_decay, startDate = self.startDate, endDate = self.endDate)

    if model =="equal weight":
      weight_array = expected_return_vector*0+1
      weight_array = weight_array / np.sum(weight_array)

    if model == "inverse volatility":
      inv_vol = 1/ std_vector
      weight_array = inv_vol / np.sum(inv_vol)

    if model == "return weighted":
      weight_array = expected_return_vector / np.sum(expected_return_vector)

    if model == "inverse volatility":
      inv_var = 1/ std_vector**0.5
      weight_array = inv_var / np.sum(inv_var)

    calculate_portfolio_metrics(weight_array, expected_return_vector, covar_df, std_vector, all_data,  one_year_return_df)
    return weight_array


#------------------------------------ END OF CLASS METHODS #------------------------------------




#------------------------------------ Supporting Functions Begin #------------------------------------


def general_min_CVaR_problem(target_return, beta, expected_return_var, weights_var,problem_constraints, hist_returns, all_data ):

  z = cp.Variable(len(hist_returns))
  var = cp.Variable(1) ### variable to mimic the VaR of the portfolio for a given beta confidence
  cvar = var + 1.0 / (len(hist_returns) * (1 - beta)) * cp.sum(z)
  objective = cp.Minimize(cvar)

  problem_constraints +=[expected_return_var>= target_return]

  ### cvar related constraints:
  problem_constraints += [z+ hist_returns.values @ weights_var + var >=0, z>=0]
  problem = cp.Problem(objective, problem_constraints)
  problem.solve()



  if objective.value is not None:
    weight_array = weights_var.value.round(10).tolist()
    weight_array = dict(zip(list(all_data.keys()), weight_array))
    print(weight_array)
    return weight_array
  else:
    return "Infeasible"





#------------------------------------ function to calculate portfolio risk metrics #------------------------------------

def calculate_portfolio_metrics(weights,expected_returns, covar, std_vector, all_data, one_year_return_df, CVaR_betas =[0.95] ):
  all_data = get_all_data_dict_for_usable_tickers(all_data)

  portfolio_metrics = {}
  weights_dict = weights.to_dict()
  portfolio_metrics["weights"] = weights_dict

  port_vol = np.matmul(np.matmul(weights.values, covar), np.transpose(weights.values))**0.5
  portfolio_metrics["volatility"] = round(port_vol,4)

  port_expected_return = np.dot(weights,expected_returns )
  portfolio_metrics["return"] = round(port_expected_return,4)

  diversification_ratio = np.dot(weights,std_vector ) / port_vol
  portfolio_metrics["diversification ratio"] = round(diversification_ratio,4)

  ### sector exposures dict

  sector_exposures_df = pd.concat([all_data[ticker]["sector_weights"] for ticker in all_data], axis =1)
  sector_exposures = {sec: round(np.matmul( sector_exposures_df.loc[sec].values , weights),3) for sec in sector_exposures_df.index.tolist()}
  portfolio_metrics["sector_exposures"] = sector_exposures


  ### risk contribution breakdown

  weightXcovar = np.matmul(weights.values, covar)
  risk_contribution = (weightXcovar * weights.values)/port_vol
  portfolio_metrics["risk contribution"] = risk_contribution
  portfolio_metrics["risk contribution sum"] = np.sum(risk_contribution)


  ### CVaR Calculation



  for beta in CVaR_betas:
    beta_hist_returns = one_year_return_df.copy()
    for ticker in weights_dict:
      beta_hist_returns[ticker] *= weights_dict.get(ticker, 0)

    beta_hist_returns = beta_hist_returns.sum(axis =1)

    # Drop the first 252 days
    sorted_returns = beta_hist_returns.sort_values(ascending=False)
    print(sorted_returns)

    portfolio_metrics[f"{beta*100}% CVaR"] = sorted_returns[-1* int(len(sorted_returns)*(1-beta)):].mean()

  return portfolio_metrics


#------------------------------------ Function to build Constraint vectors #------------------------------------


def create_constraints(weights, strict_bound = False, variance = None, expected_return = None,portfolio_sector_exposures_dict = None, ticker_limits:dict ={}, portfolio_sector_exposures_limits:dict = None, max_variance = None, min_return =None, short_sell_allowed = False, max_leverage = 0, leverage_amt=None, portfolio_factor_exposure_dict = None, portfolio_factor_exposures_limits =None):


  if strict_bound:  
    constraints =[cp.sum(weights)==1+max_leverage]
  else:  constraints = [cp.sum(weights)<=1+max_leverage]

  if leverage_amt is not None: constraints +=[leverage_amt>=0]

  if variance is not None and max_variance is not None:
    constraints += [variance <= max_variance]

  if expected_return is not None and min_return is not None:
    constraints += [expected_return >= min_return]

  if portfolio_sector_exposures_limits is not None and portfolio_sector_exposures_dict is not None:
    for sector in portfolio_sector_exposures_limits:
      if portfolio_sector_exposures_limits[sector].get("maxWeight", None) is not None:
        constraints += [portfolio_sector_exposures_dict[sector] <= portfolio_sector_exposures_limits[sector]["maxWeight"]]

      if portfolio_sector_exposures_limits[sector].get("minWeight", None) is not None:
        constraints += [portfolio_sector_exposures_dict[sector] >= portfolio_sector_exposures_limits[sector]["minWeight"]]

  if portfolio_factor_exposures_limits is not None and portfolio_factor_exposure_dict is not None:
    for sector in portfolio_factor_exposures_limits:
      if portfolio_factor_exposures_limits[sector].get("maxWeight", None) is not None:
        constraints += [portfolio_factor_exposure_dict[sector] <= portfolio_factor_exposures_limits[sector]["maxWeight"]]

      if portfolio_factor_exposures_limits[sector].get("minWeight", None) is not None:
        constraints += [portfolio_factor_exposure_dict[sector] >= portfolio_factor_exposures_limits[sector]["minWeight"]]

  if ticker_limits is not None:
    for index, (ticker, value) in enumerate(ticker_limits.items()):
      if ticker_limits[ticker].get("maxWeight", None) is not None:
        constraints += [weights[index] <= ticker_limits[ticker]["maxWeight"]]
      if ticker_limits[ticker].get("minWeight", None) is not None:
        constraints += [weights[index] >= ticker_limits[ticker]["minWeight"]]

  if not short_sell_allowed:
    constraints += [weights>=0]

  return constraints



#------------------------------------ Function to calculate Factor Exposures #------------------------------------


def factor_model(all_data, factor_data, alpha = None):

  '''
  gets factor loading of assets to factors (i.e regression betas)
  '''

  tickers_lst = [i for i in all_data]

  f_loadings_lst = []

  if alpha is None: alpha =default_alpha

  for ticker in tickers_lst:

    ticker_returns = all_data[ticker]["price_data"].pct_change().dropna()

    style_factors = all_factor_returns[[style_factor for style_factor in  factors_dict['style']]]
    factors_df = pd.concat([i for i in [ticker_returns, style_factors] if len(i)!=2], axis = 1).dropna()
    weights = (1-alpha) ** np.arange(len(factors_df))[::-1]

    model = sm.WLS(factors_df[ticker], sm.add_constant(factors_df.drop(columns = ticker)), weights = weights).fit()
    f_loadings_lst.append(model.params.rename(ticker))

  factor_loadings = pd.concat(f_loadings_lst, axis = 1).fillna(0)
  print(factor_loadings)

  return factor_loadings


def get_all_data_dict_for_usable_tickers(all_data):

  return {key: value for key, value in all_data.items() if value.get('include')}



def cov_corr_calculator(df, alpha):
  '''
  df is a df with returns of different indices sectors etc
  '''
  tickers_lst =df.columns.tolist()

  temp_retuns = df.reset_index(drop = True )
  covar = pd.DataFrame(columns =tickers_lst, index = tickers_lst )
  corr = pd.DataFrame(columns =tickers_lst, index = tickers_lst )

  for ticker_x in tickers_lst:
    for ticker_y in tickers_lst:
      x_y_valid_returns = temp_retuns[list(set([ticker_x, ticker_y]))].dropna(axis = 0, how = "any")
      covar.loc[ticker_x,ticker_y] = x_y_valid_returns[ticker_x].ewm(alpha =alpha).cov(x_y_valid_returns[ticker_y]).iloc[-1] *frequency_scaler
      corr.loc[ticker_x,ticker_y] = x_y_valid_returns[ticker_x].ewm(alpha =alpha).corr(x_y_valid_returns[ticker_y]).iloc[-1]
  return covar, corr


def get_EWMA_std_return_cov(all_data, factor_model = False, alpha = None, startDate = None, endDate = None):

  all_prices = pd.concat([all_data[ticker]["price_data"] for ticker in all_data], axis = 1)
  tickers_lst = all_prices.columns.tolist()

  if startDate is not None:all_prices = all_prices[all_prices.index>= startDate]
  if endDate is not None:all_prices = all_prices[all_prices.index<= endDate]


  one_year_return_df = all_prices.copy()
  for stock in all_prices.columns:
    price_252_days_ago = one_year_return_df[stock].shift(252)
    one_year_return_df[stock] = (one_year_return_df[stock] / price_252_days_ago) - 1
  one_year_return_df = one_year_return_df.dropna()


  all_returns = all_prices.pct_change().dropna(axis = 0, how = "all")
  #all_returns["AAPL"] = all_returns["AAPL"]*-1

  ######

  df_excess_ret = all_returns.join(risk_free_data, how='left')

  # Forward fill missing risk-free rates
  df_excess_ret['RF'].fillna(method='ffill', inplace=True)

  # Calculate excess returns
  for col in tickers_lst: df_excess_ret[col] = df_excess_ret[col] - df_excess_ret['RF']

  if alpha is None: alpha =default_alpha
  expw_returns = all_returns.ewm(alpha =alpha)
  excess_returns = df_excess_ret.drop(columns="RF").ewm(alpha =alpha)
  excess_return_vector = excess_returns.mean().iloc[-1].transpose() *frequency_scaler
  expected_return_vector = expw_returns.mean().iloc[-1].transpose() *frequency_scaler
  std_vector = expw_returns.std().iloc[-1].transpose() *np.sqrt(frequency_scaler)
  expected_rf_rate = df_excess_ret[["RF"]].ewm(alpha =alpha).mean().iloc[-1].transpose().loc["RF"] *frequency_scaler

  print(expected_rf_rate)

  covar, corr = cov_corr_calculator(all_returns, alpha)

  return expected_return_vector,std_vector,covar, excess_return_vector, expected_rf_rate, one_year_return_df


def calculate_portfolio_excess_return(weights, excess_return_vector, leverage_amt = 0, expected_rf_rate = 0): 
  return cp.sum(cp.multiply(weights, excess_return_vector))  - leverage_amt * expected_rf_rate
def calculate_portfolio_expected_return(weights, expected_return_vector, leverage_amt = 0, expected_rf_rate = 0): 
  return cp.sum(cp.multiply(weights, expected_return_vector)) - leverage_amt * expected_rf_rate
def calculate_portfolio_variance(weights, covar_df):   return cp.quad_form(weights, covar_df.values) 
 

def calculate_portfolio_diversification_ratio(weights, covar_df):
    portfolio_variance = calculate_portfolio_variance(weights, covar_df)
    asset_volatilities = cp.sqrt(np.diag(covar_df.values))
    weighted_asset_volatilities = cp.multiply(weights, asset_volatilities)
    sum_weighted_asset_volatilities = cp.sum(weighted_asset_volatilities)
    diversification_ratio =  sum_weighted_asset_volatilities / portfolio_variance**0.5
    return diversification_ratio


def calculate_portfolio_sector_exposure(weights, all_data):
    sector_exposures_df = pd.concat([all_data[ticker]["sector_weights"] for ticker in all_data], axis =1)
    portfolio_sector_exposures_dict = {sec: cp.matmul( sector_exposures_df.loc[sec].values , weights) for sec in sector_exposures_df.index.tolist()}
    return portfolio_sector_exposures_dict

def calculate_portfolio_factor_exposure(weights, factor_loadings):
    portfolio_factor_exposures_dict = {factor: cp.matmul( factor_loadings.loc[factor].values , weights) for factor in factor_loadings.index.tolist()}
    return portfolio_factor_exposures_dict



def calculate_all(all_data, weights, expected_return_vector, exccess_return_vector,covar_df, factor_loadings,leverage_amt = 0, expected_rf_rate = 0 ):
  portfolio_variance = calculate_portfolio_variance(weights, covar_df)
  portfolio_excess_return = calculate_portfolio_excess_return(weights, exccess_return_vector, leverage_amt,expected_rf_rate )
  portfolio_expected_return = calculate_portfolio_expected_return(weights, expected_return_vector, leverage_amt,expected_rf_rate )
  diversification_ratio = calculate_portfolio_diversification_ratio(weights, covar_df)
  portfolio_sector_exposures_dict = calculate_portfolio_sector_exposure(weights, all_data)
  portfolio_factor_exposures_dict = calculate_portfolio_factor_exposure(weights, factor_loadings)
  return portfolio_expected_return, portfolio_variance, portfolio_excess_return,diversification_ratio, portfolio_sector_exposures_dict, portfolio_factor_exposures_dict





def download_stock_historical_price(ticker):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY_ADJUSTED&symbol={ticker}&outputsize=full&apikey={api_key}'
    r = requests.get(url)
    data = r.json()
    print(data)
    price_data = pd.DataFrame(data["Weekly Adjusted Time Series"]).loc[["5. adjusted close"]].T
    price_data.rename(columns={"5. adjusted close": ticker}, inplace=True)
    price_data.index = pd.to_datetime(price_data.index)
    price_data[ticker] = price_data[ticker].resample(data_frequency).last().astype(float)
    price_data = price_data.iloc[::-1]

    return price_data.dropna()



def get_basic_ticker_data(ticker, ticker_config, source_test = "yahoo"):

    current_date = datetime.datetime.now()
    start_date = current_date - datetime.timedelta(days=30 * 365)

    ticker_data = {}
    if source_test == "yahoo":
      price_data = yf.download(ticker, start=start_date, end=current_date)["Adj Close"].resample(data_frequency).last().rename(ticker).dropna()
      price_data.index = price_data.index.strftime('%Y-%m-%d')
    else:price_data = download_stock_historical_price(ticker)

    ticker_data["price_data"] = price_data

    ticker_data["maxWeight"] = ticker_config.get("maxWeight",None)
    ticker_data["minWeight"] = ticker_config.get("minWeight",None)

    return ticker_data

def get_etf_info(ticker):

    info = get_ticker_info(ticker)

    t = Ticker(ticker)
    sector_weights = t.fund_sector_weightings
    sector_weights.index.name = None
    sector_weights.rename(index = {"technology":"Technology", "realestate":"Real Estate", "consumer_cyclical":"Consumer Cyclical",
    "basic_materials":"Basic Materials","industrials":"Industrials", "communication_services" :"Communication Services",
    "consumer_defensive":"Consumer Defensive", "utilities":"Utilities","energy":"Energy", "healthcare":"Healthcare", "financial_services":"Financial Services"}, inplace = True)
    info["sector_weights"] = sector_weights
    return info


def get_clean_ticker_bounds_dict(all_equity_data):
  all_ticker_limits = {outer_k: {k: v for k, v in outer_v.items() if k in ['max', 'min']} for outer_k, outer_v in all_equity_data.items()}
  return all_ticker_limits



def filter_dict(original_dict, keys):
    return {key: original_dict[key] for key in keys if key in original_dict}

def get_ticker_info(ticker, test_source = "yahoo"):

    if test_source =="yahoo":
      relevant_info = ["quoteType", "country","marketCap", "currency","longName", "sector", "Ask", "Bid"]
      yf_ticker = yf.Ticker(ticker)
      info = yf_ticker.info
      info = filter_dict(info, relevant_info)

    else:
      relevant_info = ["MarketCapitalization", "Sector", "Country", "Currency"]
      url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={api_key}'
      r = requests.get(url)
      info = r.json()
      print(info)
      info = filter_dict(info, relevant_info)
    return info