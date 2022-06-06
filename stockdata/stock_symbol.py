import os
import django
import requests
import pickle
import yfinance as yf
from datetime import datetime, timedelta
from pandas import Timestamp
# download and store all the stock symbol and company in the database
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "stock_market.settings")
django.setup()


from stock.models import Stock
def main():
    r = requests.get('https://finnhub.io/api/v1/stock/symbol?exchange=US&token=c0ab4jn48v6tv7n8k3e0')
    pickle.dump(r.json(), open("data.p", "wb"))
    stock_data = pickle.load(open("data.p", "rb"))
    for data in stock_data:
        Stock.objects.get_or_create(symbol=data['symbol'], company=data['description'][:50].title())
    symbol_set = ['GE', 'AAPL', 'FB', 'IBM', 'BTX']  # pre upload data to database
    # pre_symbol = 'GE' # if happens cancel the loop and only use pre_symbol ='GE'
    for pre_symbol in symbol_set:
        load_data = Stock.objects.get(symbol=pre_symbol)
        data_his = yf.download(pre_symbol, period="1y", interval="1d", rounding=True,
                               end=datetime.now() - timedelta(days=1))
        for pdtimestamp, price_dict in data_his.to_dict('index').items():
            timestamp = Timestamp(pdtimestamp, tz='UTC')
            load_data.info_set.update_or_create(
                id="{}_{}".format(timestamp, pre_symbol),
                open_price=price_dict['Open'],
                high_price=price_dict['High'],
                low_price=price_dict['Low'],
                close_price=price_dict['Close'],
                adj_close_price=price_dict['Adj Close'],
                volume=price_dict['Volume'],
                date=timestamp,
                change=price_dict['Close'] - price_dict['Open'],
                change_percent=((price_dict['Close'] - price_dict['Open']) / price_dict['Open']) * 100,
            )
            load_data.save()


if __name__ == '__main__':
    main()
