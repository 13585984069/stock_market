import yfinance as yf
import urllib
from urllib import request
import datetime

now = datetime.datetime.now()
two = datetime.timedelta(days=2)
one = datetime.timedelta(days=1)
yes = now-two
if now - yes >= two:
    print('yes')
# yes = now-oneday
# print(yes-now)
# print(yes)