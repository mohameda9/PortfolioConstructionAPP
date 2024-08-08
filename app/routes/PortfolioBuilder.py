import pandas as pd
from fastapi import APIRouter, File, UploadFile
#from app.client import supabase
import io
from pydantic import BaseModel
from typing import Dict, List, Union
from app.services.portfolioConstructor import PortfolioConstruction
import csv
from io import StringIO, BytesIO
from typing import Any, Dict, Optional



class inputTickersSetupConfigs(BaseModel):
    tickers_configs: Dict


class DataRow(BaseModel):
    """Row from a dataset"""
    columns: List


class Data(BaseModel):
    """Dataset table"""
    data: List[DataRow]



class moreModelConstraints(BaseModel):
    sector_limits : Dict



class inputModelPeriod(BaseModel):
    startDate: str ### yyyy-mm-dd
    endDate: str ### yyyy-mm-dd


router = APIRouter()

#### update model lookback period (i.e start and end date)

@router.post("/update-model-date-settings")
async def update_model_periods(inputModelPeriod: inputModelPeriod):
    model = PortfolioConstruction()
    model.set_model_period(startDate= inputModelPeriod.startDate, endDate= inputModelPeriod.endDate)
    print("model date settings updated")





class RequestBody(BaseModel):
    params: Dict[str, Any]
    data: Data
    moreModelConstraints: moreModelConstraints
    inputTickersSetupConfigs: inputTickersSetupConfigs

class RequestBodyBenchMark(BaseModel):
    params: Dict[str, Any]
    data: Data
    inputTickersSetupConfigs: inputTickersSetupConfigs

class riskParityBody(BaseModel):
    params: Dict[str, Any]
    data: Data
    riskBudget: Dict


@router.post("/MVOptimization")
async def MVOpt(request: RequestBody):
    
    params = request.params
    data = request.data
    moreModelConstraints = request.moreModelConstraints
    inputTickersSetupConfigs = request.inputTickersSetupConfigs

    print("params:", params)
    print("moreModelConstraints:", moreModelConstraints)
    print("inputTickersSetupConfigs:", inputTickersSetupConfigs)

    optimize_for = params.get('optimize_for')
    min_return = params.get('min_return')
    max_risk = params.get('max_risk')
    short_sell = params.get('short_sell')
    hist_decay = params.get('hist_decay')
    if optimize_for =="risk":
        max_risk = None
        if min_return  is None: min_return=0.00001
    else:  
        min_return =None
        if max_risk  is None: max_risk=0.00001



    
    model = PortfolioConstruction()
    df = convert_to_df(data)


    model.add_tickers_using_dataset(df)

    tickers_configs = inputTickersSetupConfigs.tickers_configs
    model.add_or_update_tickers_configs(tickers_configs)

    model.set_model_period(startDate= params.get('startDate', None), endDate= params.get('endDate', None))

    portfolio_sector_exposures_limits = moreModelConstraints.sector_limits
    print(portfolio_sector_exposures_limits)
    optimized_port = model.MVO(optimize_for=optimize_for, short_sell=short_sell, min_return=min_return,
              max_risk=max_risk, hist_decay=hist_decay, portfolio_sector_exposures_limits=portfolio_sector_exposures_limits)
    print(optimized_port)

    return optimized_port




@router.post("/ratioOpt") 
async def ratioOpt(request: RequestBody):
    
    params = request.params
    data = request.data
    moreModelConstraints = request.moreModelConstraints
    inputTickersSetupConfigs = request.inputTickersSetupConfigs

    print("params:", params)
    print("moreModelConstraints:", moreModelConstraints)
    print("inputTickersSetupConfigs:", inputTickersSetupConfigs)

    optimize_ratio = params.get('optimize_for')
    short_sell = params.get('short_sell')
    hist_decay = params.get('hist_decay')


    model = PortfolioConstruction()
    df = convert_to_df(data)

    model.add_tickers_using_dataset(df)
    tickers_configs = inputTickersSetupConfigs.tickers_configs
    model.add_or_update_tickers_configs(tickers_configs)

    portfolio_sector_exposures_limits = moreModelConstraints.sector_limits
    print(portfolio_sector_exposures_limits)
    model.set_model_period(startDate= params.get('startDate', None), endDate= params.get('endDate', None))

    portfolio_sector_exposures_limits = moreModelConstraints.sector_limits
    print(portfolio_sector_exposures_limits)
    optimized_port = model.optimize_ratio(ratio=optimize_ratio, allow_short_sell = short_sell,  hist_decay=hist_decay,
                                portfolio_sector_exposures_limits =portfolio_sector_exposures_limits, abs_weight_constraint=5)
    
    print(optimized_port)
    return optimized_port





@router.post("/CVaR")
async def CVaRopt(request: RequestBody):
    params = request.params
    data = request.data
    moreModelConstraints = request.moreModelConstraints
    inputTickersSetupConfigs = request.inputTickersSetupConfigs

    print("params:", params)
    print("moreModelConstraints:", moreModelConstraints)
    print("inputTickersSetupConfigs:", inputTickersSetupConfigs)

    target_return = params.get('target_return')
    beta = params.get('beta', 0.95)
    short_sell = params.get('short_sell')

    model = PortfolioConstruction()
    df = convert_to_df(data)

    model.add_tickers_using_dataset(df)
    tickers_configs = inputTickersSetupConfigs.tickers_configs
    model.add_or_update_tickers_configs(tickers_configs)
    model.set_model_period(startDate= params.get('startDate', None), endDate= params.get('endDate', None))

    portfolio_sector_exposures_limits = moreModelConstraints.sector_limits
    print(portfolio_sector_exposures_limits)

    optimized_port = model.min_CVaR(target_return=target_return, beta=beta, short_sell=short_sell,
                                    portfolio_sector_exposures_limits=portfolio_sector_exposures_limits)

    print(optimized_port)
    return optimized_port






@router.post("/Benchmark")
async def Benchmark(request: RequestBodyBenchMark):
    params = request.params
    data = request.data
    inputTickersSetupConfigs = request.inputTickersSetupConfigs

    print("params:", params)
    print("inputTickersSetupConfigs:", inputTickersSetupConfigs)

    bm_type = params.get('bm_type') ## equal weight, inverse volatitly etc

    model = PortfolioConstruction()
    df = convert_to_df(data)

    model.add_tickers_using_dataset(df)
    tickers_configs = inputTickersSetupConfigs.tickers_configs
    model.add_or_update_tickers_configs(tickers_configs)
    model.set_model_period(startDate= params.get('startDate', None), endDate= params.get('endDate', None))


    optimized_port = model.benchmark_portfolios( model = bm_type,
                                    hist_decay = params.get('hist_decay'))
    print(optimized_port)
    return optimized_port









@router.post("/RiskParity") 
async def RiskParity(request: riskParityBody):
    
    params = request.params
    data = request.data

    print("params:", params)

    hist_decay = params.get('hist_decay')


    model = PortfolioConstruction()
    df = convert_to_df(data)

    model.add_tickers_using_dataset(df)
    model.set_model_period(startDate= params.get('startDate', None), endDate= params.get('endDate', None))

    risk_budget = request.riskBudget
    print(risk_budget)
    optimized_port = model.risk_parity(hist_decay=hist_decay, risk_budget=risk_budget)

    print(optimized_port)
    return optimized_port













def convert_to_df(data: Data):
    """Convert dataset to a Pandas dataframe"""
    # Extract data
    rows = [row.columns for row in data.data]

    # Convert to dataframe (assuming the first row contains headers)
    df = pd.DataFrame(rows[1:], columns=rows[0])

    return df
