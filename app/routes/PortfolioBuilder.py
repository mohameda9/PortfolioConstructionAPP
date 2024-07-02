import pandas as pd
from fastapi import APIRouter, File, UploadFile
#from app.client import supabase
import io
from pydantic import BaseModel
from typing import Dict, List, Union
from app.services.portfolioConstructor import PortfolioConstruction
import csv
from io import StringIO, BytesIO




#### global instance of model -- APIs will update
model = PortfolioConstruction()


class inputTickersSetupConfigs(BaseModel):
    tickerstoUse: list
    tickers_configs: Dict



#### input to add-data api -- not used for now
class inputTickers(BaseModel):
    tickers: Dict[str, list] ### must be like {stocks :[s1,s2..], "ETFs":[e1,e2]}. need either stocks or ETFs
    ticker_configs: Dict ##### {stocks: {s1: {min:0.2, max:0.3}, s2: {min:0.3, max:0.7} }, ETFs: {e1: {min:0.3}, }}


class moreModelConstraints(BaseModel):
    sector_limits : Dict


class inputModelPeriod(BaseModel):
    startDate: str ### yyyy-mm-dd
    endDate: str ### yyyy-mm-dd


router = APIRouter()

#### update model lookback period (i.e start and end date)

@router.post("/update-model-date-settings")
async def update_model_periods(inputModelPeriod: inputModelPeriod):
    model.set_model_period(startDate= inputModelPeriod.startDate, endDate= inputModelPeriod.endDate)
    print("model date settings updated")


@router.post("/upload-price-data")
async def upload_price_datasets(file: UploadFile = File(...)):
    # contents = file.file.read()
    # buffer = BytesIO(contents)
    # df = pd.read_csv(buffer)
    # print(df)
    df = pd.read_csv(file.file)
    model.add_tickers_using_dataset(df)
    return df.head()


@router.post("/update-tickers-configs")
async def updateTickersConfigs(inputTickersSetupConfigs:inputTickersSetupConfigs):
    tickers_configs = inputTickersSetupConfigs.tickers_configs
    model.select_tickers_to_use(inputTickersSetupConfigs.tickerstoUse)
    model.add_or_update_tickers_configs(tickers_configs)





@router.post("/MVOptimization") 
async def MVOpt(optimize_for, moreModelConstraints:moreModelConstraints, min_return:float = None, max_risk:float = None, short_sell:bool = False,
                hist_decay:float = None):
    portfolio_sector_exposures_limits = moreModelConstraints.sector_limits
    print(portfolio_sector_exposures_limits)
    optimized_port = model.MVO(optimize_for=optimize_for, min_return=min_return,
              max_risk=max_risk, hist_decay=hist_decay, portfolio_sector_exposures_limits =portfolio_sector_exposures_limits)
    
    return optimized_port


@router.post("/RiskParity") 
async def MVOpt(hist_decay:float = None):
    optimized_port = model.risk_parity(hist_decay=hist_decay)
    
    return optimized_port




@router.post("/ratioOpt") 
async def ratioOpt(optimize_ratio, moreModelConstraints:moreModelConstraints,
                hist_decay:float = None):
    portfolio_sector_exposures_limits = moreModelConstraints.sector_limits
    print(portfolio_sector_exposures_limits)
    optimized_port = model.optimize_ratio(ratio=optimize_ratio,  hist_decay=hist_decay,
                                portfolio_sector_exposures_limits =portfolio_sector_exposures_limits)
    
    return optimized_port


@router.post("/min_CVaR")
async def min_CVaR(min_return:float, beta:float, moreModelConstraints:moreModelConstraints):
    portfolio_sector_exposures_limits = moreModelConstraints.sector_limits
    print(portfolio_sector_exposures_limits)
    optimized_port = model.min_CVaR(target_return=min_return, beta=beta, portfolio_sector_exposures_limits= portfolio_sector_exposures_limits)

    return optimized_port


@router.get("/test")
async def test():
    print("working")
    return {"1"}

@router.post("/test_post")
async def test_post(min_return:float, beta:float, moreModelConstraints:moreModelConstraints):
    return {"Msg": "Yay!"}





#### use this to add equities and etfs as well as max and min weight limits
@router.post("/add-data")
async def add_tickers(inputTickers: inputTickers):
    stock_tickers = inputTickers.tickers.get("stocks", None)
    etf_tickers = inputTickers.tickers.get("ETFs", None)
    stocklimits = inputTickers.ticker_limits.get("stocks", {})
    etflimits = inputTickers.ticker_limits.get("ETFs", {})

    if stock_tickers is not None:
        model.add_equity_tickers(stock_tickers, stocklimits)

    if etf_tickers is not None:
        model.add_etf_tickers(etf_tickers, etflimits)