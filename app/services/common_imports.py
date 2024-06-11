import pandas as pd
import warnings

# Ignore all warnings
warnings.filterwarnings('ignore')
import numpy as np
import yfinance as yf
import datetime
import cvxpy as cp
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from yahooquery import Ticker
import requests,io
import zipfile, os
import statsmodels.api as sm
print(os.getcwd())
from app.services import extrafunctions
