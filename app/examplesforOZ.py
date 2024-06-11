####### format input to .post("/add-data")

exampleInput_toPOST_ADD_DATA = {
  "tickers": {
    "stocks": ["AAPL", "AMD", "TSLA"],
    "ETFs": ["XLK", "XLC"]
        },
  "ticker_configs": {"stocks":{"AAPL":{"min":0.1, "max":0.3, "sector":"Technology"}},
                    "ETFs":{"XLK":{"min":0.2, "max":0.6}}}
}

exampleInputTickerconfigs = {
  "tickers_configs": {"AAPL":{"min":0.1, "max":0.3, "sector":"Technology"}}
} 



moreModelParams_example = {
    "sector_configs": {"Technology":{"max":0.5}}
}