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


class inputTickersConfigs(BaseModel):
    tickers_configs: Dict



#### input to add-data api -- not used for now
class inputTickers(BaseModel):
    tickers: Dict[str, list] ### must be like {stocks :[s1,s2..], "ETFs":[e1,e2]}. need either stocks or ETFs
    ticker_configs: Dict ##### {stocks: {s1: {min:0.2, max:0.3}, s2: {min:0.3, max:0.7} }, ETFs: {e1: {min:0.3}, }}


class moreModelParams(BaseModel):
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
    contents = file.file.read()
    buffer = BytesIO(contents)
    df = pd.read_csv(buffer)
    print(df)
    model.add_tickers_using_dataset(df)


@router.post("/update-tickers-configs")
async def updateTickersConfigs(inputTickersConfigs:inputTickersConfigs):
    tickers_configs = inputTickersConfigs.tickers_configs
    model.add_or_update_tickers_configs(tickers_configs)




@router.post("/MVOptimization") 
async def MVOpt(optimize_for, moreModelParams:moreModelParams, min_return:float = None, max_risk:float = None, short_sell:bool = False,
                hist_decay:float = None):
    portfolio_sector_exposures_limits = moreModelParams.sector_limits
    print(portfolio_sector_exposures_limits)
    optimized_port = model.MVO(optimize_for=optimize_for, min_return=min_return,
              max_risk=max_risk, hist_decay=hist_decay, portfolio_sector_exposures_limits =portfolio_sector_exposures_limits)
    
    return optimized_port



@router.post("/ratioOpt") 
async def ratioOpt(optimize_ratio, moreModelParams:moreModelParams,
                hist_decay:float = None):
    portfolio_sector_exposures_limits = moreModelParams.sector_limits
    print(portfolio_sector_exposures_limits)
    optimized_port = model.optimize_ratio(ratio=optimize_ratio,  hist_decay=hist_decay,
                                portfolio_sector_exposures_limits =portfolio_sector_exposures_limits)
    
    return optimized_port







@router.get("/test")
async def test(filename):
    return {"1"}






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