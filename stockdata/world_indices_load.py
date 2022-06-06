import os
import django

# download and store all the stock symbol and company in the database
import pandas as pd

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "stock_market.settings")
django.setup()


from stock.models import WorldIndices
def load_indice():
    worldindices = [{'name':'HANG SENG', 'file':'^HSI'},
                    {'name': 'NASDAQ Composite', 'file': '^IXIC'},
                    {'name':'Shenzhen Component', 'file':'^SZ'},
                    {'name':'Russell 2000', 'file':'^RUT'}]
    for data in worldindices:
        WorldIndices.objects.get_or_create(name=data['name'], file=data['file'][:50].title())

def load_indice_values():
    load_data = WorldIndices.objects.all()
    for data in load_data:
        dir = '../stock/ML/Indices data/' + data.file + '.csv'
        read_data = pd.read_csv(dir)
        read_data = read_data.set_index(["Date"], drop=True)
        for pdtimestamp, price_dict in read_data.to_dict('index').items():
            data.info_set.update_or_create(
                id="{}_{}".format(pdtimestamp, data.name),
                open_price=price_dict['Open'],
                high_price=price_dict['High'],
                low_price=price_dict['Low'],
                close_price=price_dict['Close'],
                adj_close_price=price_dict['Adj Close'],
                volume=price_dict['Volume']/1000,
                date=pdtimestamp,
            )
            data.save()


if __name__ == '__main__':
    load_indice()
    load_indice_values()