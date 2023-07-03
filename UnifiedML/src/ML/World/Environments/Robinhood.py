# Downloaded from https://stooq.com/db/h/
# However requires captcha to download
# import wget
# url = 'https://stooq.com/db/d/?b=h_us_txt'
# wget.download(url)
# Can try https://pandas-datareader.readthedocs.io/en/latest/readers/stooq.html   # Daily, not hourly
import pandas as pd


stooq_aapl = pd.read_csv("MarketHourly/us/nasdaq stocks/1/aapl.us.txt")
stooq_aapl.head()
