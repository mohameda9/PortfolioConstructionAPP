####### format input to .post("/add-data")

exampleInput_toPOST_ADD_DATA = {
  "tickers": {
    "stocks": ["AAPL", "AMD", "TSLA"],
    "ETFs": ["XLK", "XLC"]
        },
  "ticker_configs": {"stocks":{"AAPL":{"minWeight":0.1, "maxWeight":0.3, "sector":"Technology"}},
                    "ETFs":{"XLK":{"min":0.2, "max":0.6}}}
}

exampleinputTickersSetupConfigs = {
  "tickerstoUse": ["AAPL", "WMT", "IBM"],
  "tickers_configs": {"AAPL":{"minWeight":0.1, "maxWeight":0.3, "sector":"Technology"}}
} 


moreModelConstraints_example = {
    "sector_configs": {"Technology":{"maxWeight":0.5}}
} 

data ={ "data": {
    "data": [
      {"columns": ["Date", "AAPL", "WMT"]},
      {"columns": ["2023-01-01", "50", "20"]},
      {"columns": ["2023-01-02", "51", "23"]},
      {"columns": ["2023-01-03", "52", "22"]},
      {"columns": ["2023-01-04", "53", "27"]}
     
    ]
  }
}