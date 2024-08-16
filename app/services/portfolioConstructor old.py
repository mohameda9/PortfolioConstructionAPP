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




### class
class PortfolioConstruction:

  def __init__(self):

    self.all_equity_data = {}
    self.all_etf_data = {}
    self.startDate = None
    self.endDate = None

  def add_equity_tickers(self, tickers, tickers_config ={}):
    for ticker in tickers:
      ticker_config = tickers_config.get(ticker,{})
      ticker_data = get_basic_ticker_data(ticker,ticker_config)
      ticker_data= ticker_data | get_ticker_info(ticker)
      if "sector" in ticker_config and ticker_config.get("sector", "") not in [None,""]:
        ticker_data["sector"] = ticker_config["sector"]
      self.all_equity_data[ticker] = ticker_data

    self.get_sector_weights()

  def add_tickers_using_dataset(self, price_data, tickers_to_use=None):

    if tickers_to_use is not None:
      tickers_to_use = [i for i in tickers_to_use if i in price_data.columns.tolist()] + ["Date"]
      price_data = price_data[tickers_to_use]
    
    price_data.set_index("Date", inplace = True)
    tickers = price_data.columns.tolist()

    price_data.index = pd.to_datetime(price_data.index)
    price_data = price_data.resample(data_frequency).last().astype(float)

    price_data.index = price_data.index.strftime('%Y-%m-%d')
    for ticker in tickers:
      #price_data[ticker] = price_data[ticker].resample(data_frequency).last().astype(float)
      self.all_equity_data[ticker] = {"price_data":price_data[ticker], "sector":"No Sector Data"}


    self.get_sector_weights()
  
  def add_or_update_tickers_configs(self, tickers_config):

    for ticker in self.all_equity_data:
      print(ticker)
      ticker_config = tickers_config.get(ticker,None)
      print(ticker_config)
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


  def MVO(self,optimize_for, hist_decay, max_risk = None, min_return = None, portfolio_sector_exposures_limits = None):

    all_data = self.all_equity_data | self.all_etf_data

    optimize_for = optimize_for.lower()
    allowed_optimize_for = ["risk", "return"]

    if optimize_for not in allowed_optimize_for:
      raise Exception(f"error, optimize variable must be in {allowed_optimize_for}, not {optimize_for}")


    if optimize_for in ["risk"] and min_return is None:

      raise Exception(f"error, need to provide minimum portfolio return when optimizing for risk or diversification")


    all_ticker_limits = get_clean_ticker_bounds_dict(all_data)

    expected_return_vector,std_vector,covar_df, excess_return_vector = get_EWMA_std_return_cov(all_data, alpha = hist_decay, startDate = self.startDate, endDate = self.endDate)
    weights = cp.Variable(len(expected_return_vector))


    portfolio_sector_exposures_dict = calculate_portfolio_sector_exposure(weights, all_data)

    expected_return = calculate_portfolio_expected_return(weights, expected_return_vector)
    portfolio_variance = calculate_portfolio_variance(weights, covar_df)
    diversification_ratio = calculate_portfolio_diversification_ratio(weights, covar_df)

    if optimize_for == "return" : objective = cp.Maximize(expected_return)
    elif optimize_for == "risk": objective = cp.Minimize(portfolio_variance)
    elif optimize_for == "diversification ratio": objective = cp.Minimize(diversification_ratio)

    max_variance = None if max_risk is None else max_risk **2

    constraints = create_constraints(weights,ticker_limits = all_ticker_limits, variance= portfolio_variance, expected_return = expected_return, min_return = min_return, max_variance =  max_variance, portfolio_sector_exposures_dict = portfolio_sector_exposures_dict, portfolio_sector_exposures_limits =portfolio_sector_exposures_limits  )

    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Extract optimal weights
    if weights.value is None:
      return "infeasible optimization"
    optimal_weights = weights.value.round(2).tolist()
    optimal_weights = dict(zip(list(all_data.keys()), optimal_weights))

    optimal_port_data = calculate_portfolio_metrics(pd.Series(optimal_weights), expected_return_vector, covar_df, std_vector, all_data)

    return optimal_port_data

  def optimize_ratio(self,ratio, hist_decay, portfolio_sector_exposures_limits = None):

    all_data = self.all_equity_data | self.all_etf_data
    ratio = ratio.lower()
    allowed_ratios = ["sharpe ratio", "diversification ratio"]

    if ratio not in allowed_ratios:
      raise Exception(f"error, ratio must be one of {allowed_ratios}, not {ratio}")

    all_ticker_limits = get_clean_ticker_bounds_dict(all_data)

    expected_return_vector,std_vector,covar_df,excess_return_vector = get_EWMA_std_return_cov(all_data, alpha = hist_decay, startDate = self.startDate, endDate = self.endDate)

    weights = cp.Variable(len(expected_return_vector))
    portfolio_sector_exposures_dict = calculate_portfolio_sector_exposure(weights, all_data)
    expected_return = calculate_portfolio_expected_return(weights, expected_return_vector)
    portfolio_variance = calculate_portfolio_variance(weights, covar_df)
    excess_return = calculate_portfolio_excess_return(weights, excess_return_vector)


    #### find min possible risk
    min_risk_objective = cp.Minimize(portfolio_variance)
    constraints = create_constraints( weights, variance= portfolio_variance, expected_return = expected_return, ticker_limits = all_ticker_limits, strict_bound = True, portfolio_sector_exposures_dict = portfolio_sector_exposures_dict, portfolio_sector_exposures_limits =portfolio_sector_exposures_limits  )
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
    constraints = create_constraints( weights, min_return = max_possible_return, ticker_limits = all_ticker_limits, variance= portfolio_variance, expected_return = expected_return, strict_bound = True, portfolio_sector_exposures_dict = portfolio_sector_exposures_dict, portfolio_sector_exposures_limits =portfolio_sector_exposures_limits  )

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
      for max_risk in np.linspace(min_possible_volatility,max_possible_volatility,80):
        
        constraints = create_constraints(weights, ticker_limits = all_ticker_limits, variance= portfolio_variance, expected_return = expected_return, strict_bound = True, max_variance =  max_risk**2, portfolio_sector_exposures_dict = portfolio_sector_exposures_dict, portfolio_sector_exposures_limits =portfolio_sector_exposures_limits  )
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

        constraints = create_constraints(weights,ticker_limits = all_ticker_limits, variance= portfolio_variance, expected_return = expected_return, strict_bound = True, max_variance =  possible_risk**2, portfolio_sector_exposures_dict = portfolio_sector_exposures_dict, portfolio_sector_exposures_limits =portfolio_sector_exposures_limits  )
        problem = cp.Problem(objective, constraints)
        problem.solve()
      
        if objective.value is not None:
          weight_array = weights.value.round(10).tolist()
          weight_array = dict(zip(list(all_data.keys()), weight_array))
          portfolio_metrics = calculate_portfolio_metrics(pd.Series(weight_array), expected_return_vector, covar_df, std_vector, all_data)
          if portfolio_metrics["diversification ratio"]> best_ratio:
            best_ratio = portfolio_metrics["diversification ratio"]
            port_maximizing_diversification_ratio = portfolio_metrics
      
      return port_maximizing_diversification_ratio

      


    

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

    expected_return_vector,std_vector,covar_df, excess_return_vector = get_EWMA_std_return_cov(all_data, alpha =  hist_decay, startDate = self.startDate, endDate = self.endDate)

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

    calculate_portfolio_metrics(weight_array, expected_return_vector, covar_df, std_vector, all_data)
    return weight_array


def calculate_portfolio_metrics(weights,expected_returns, covar, std_vector, all_data):

  portfolio_metrics = {}
  portfolio_metrics["weights"] = weights.to_dict()

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


  return portfolio_metrics








def create_constraints(weights, strict_bound = False, variance = None, expected_return = None,portfolio_sector_exposures_dict = None, ticker_limits:dict ={}, portfolio_sector_exposures_limits:dict = None, max_variance = None, min_return =None, short_sell_allowed = False, max_leverage = 0 ):

  constraints = [] ### initialze

  if strict_bound:  constraints += [cp.sum(weights)==1+max_leverage]
  else:  constraints += [cp.sum(weights)<=1+max_leverage]

  if variance is not None and max_variance is not None:
    constraints += [variance <= max_variance]

  if expected_return is not None and min_return is not None:
    constraints += [expected_return >= min_return]

  if portfolio_sector_exposures_limits is not None and portfolio_sector_exposures_dict is not None:
    for sector in portfolio_sector_exposures_limits:
      if portfolio_sector_exposures_limits[sector].get("max", None) is not None:
        constraints += [portfolio_sector_exposures_dict[sector] <= portfolio_sector_exposures_limits[sector]["max"]]

      if portfolio_sector_exposures_limits[sector].get("min", None) is not None:
        constraints += [portfolio_sector_exposures_dict[sector] >= portfolio_sector_exposures_limits[sector]["min"]]

  if ticker_limits is not None:
    for index, (ticker, value) in enumerate(ticker_limits.items()):
      if ticker_limits[ticker].get("max", None) is not None:
        constraints += [weights[index] <= ticker_limits[ticker]["max"]]
      if ticker_limits[ticker].get("min", None) is not None:
        constraints += [weights[index] >= ticker_limits[ticker]["min"]]

  if not short_sell_allowed:
    constraints += [weights>=0]

  return constraints






def factor_model(all_data, factor_data):

  pass






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


  all_returns = all_prices.pct_change().dropna(axis = 0, how = "all")

  ###### 

  df_excess_ret = all_returns.join(risk_free_data, how='left')

  # Forward fill missing risk-free rates
  df_excess_ret['RF'].fillna(method='ffill', inplace=True)

  # Calculate excess returns
  for col in tickers_lst: df_excess_ret[col] = df_excess_ret[col] - df_excess_ret['RF']

  if alpha is None: alpha =0.000000000000001
  expw_returns = all_returns.ewm(alpha =alpha)
  excess_returns = df_excess_ret.drop(columns="RF").ewm(alpha =alpha)
  excess_return_vector = excess_returns.mean().iloc[-1].transpose() *frequency_scaler
  expected_return_vector = expw_returns.mean().iloc[-1].transpose() *frequency_scaler
  std_vector = expw_returns.std().iloc[-1].transpose() *np.sqrt(frequency_scaler)

  covar, corr = cov_corr_calculator(all_returns, alpha)

  return expected_return_vector,std_vector,covar, excess_return_vector


def calculate_portfolio_excess_return(weights, excess_return_vector): return cp.sum(cp.multiply(weights, excess_return_vector))
def calculate_portfolio_expected_return(weights, expected_return_vector): return cp.sum(cp.multiply(weights, expected_return_vector))
def calculate_portfolio_variance(weights, covar_df): return cp.quad_form(weights, covar_df.values)

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

    ticker_data["max"] = ticker_config.get("max",None)
    ticker_data["min"] = ticker_config.get("min",None)

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
