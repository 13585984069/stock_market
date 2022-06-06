import numpy as np
from django.shortcuts import render, redirect
import django
from .models import Stock, StockInfo, WorldIndicesInfo, WorldIndices
from django.contrib import messages
import yfinance as yf
from pandas import Timestamp
from datetime import datetime, timedelta
import requests
from .ML import RNN_torch
import matplotlib.pyplot as plt
import os
import pandas as pd
# Create your views here.


def data_info(request):
    if not request.user.is_authenticated:
        return redirect('/')
    else:
        context = {}
        date_load = StockInfo.objects.all().order_by('-date')[0].date
        #date_load = datetime.strptime(str(StockInfo.objects.all().order_by('-date')[0].date), '%Y-%m-%d')
        volume_load = StockInfo.objects.all().filter(date=date_load).order_by('-volume')[:10]
        context['Max_volume'] = volume_load
        return render(request, 'stock.html', context)


def stock_search_data(request):
    stock_symbol = request.POST['stock_symbol']
    context = {}
    for stock in Stock.objects.all():
        if stock_symbol.lower() == stock.symbol.lower():
            context['company'] = stock.company
            context['symbol'] = stock.symbol
            load_data = Stock.objects.get(symbol=stock.symbol)
            recent_data = load_data.info_set.order_by('-date')
            try:
                if recent_data.count() <= 10:
                    load_data = download(stock.symbol)
                elif datetime.now() - datetime.strptime(str(recent_data[0].date)[:10], '%Y-%m-%d') >= timedelta(days=5):
                    load_data.info_set.all().delete()
                    load_data = download(stock.symbol)
            except (django.db.utils.IntegrityError,requests.exceptions.SSLError):
                recent_data = load_data.info_set.order_by('-date')
            recent_data = load_data.info_set.order_by('-date')
            for data in recent_data:
                data.date = datetime.strptime(str(data.date)[:10], '%Y-%m-%d').date()
                data.volume = int(data.volume)
            context['data'] = recent_data[:10]
            context['30day_data'] = recent_data[:30]
            break
    else:
        messages.info(request, 'Invaild stock symbol')
        return redirect('stock:mainpage')
    return render(request, 'stock_info.html', context)


def stock_click_data(request, stock_symbol):
    stock_symbol = stock_symbol
    context = {}
    for stock in Stock.objects.all():
        if stock_symbol.lower() == stock.symbol.lower():
            context['company'] = stock.company
            context['symbol'] = stock.symbol
            load_data = Stock.objects.get(symbol=stock.symbol)
            recent_data = load_data.info_set.order_by('-date')
            for data in recent_data:
                data.date = datetime.strptime(str(data.date)[:10], '%Y-%m-%d').date()
                data.volume = int(data.volume)
            context['data'] = recent_data[:10]
            context['30day_data'] = recent_data[:30]
            break
    else:
        messages.info(request, 'Invaild stock symbol')
        return redirect('stock:mainpage')
    return render(request, 'stock_info.html', context)


def stock_detail_info(request):
    stock_symbol = request.POST['stock_symbol'][9: -18]
    context = {}
    data_load = Stock.objects.get(symbol=stock_symbol)
    detail_data = data_load.info_set.order_by('date')
    for data in detail_data:
        data.date = datetime.strptime(str(data.date)[:10], '%Y-%m-%d').date()
        data.volume = int(data.volume)
    context['symbol'] = data_load.symbol
    context['company'] = data_load.company
    context['all_data'] = detail_data
    return render(request, 'stock_detail.html', context)


def download(stock_symbol):
    load_data = Stock.objects.get(symbol=stock_symbol)
    data_his = yf.download(stock_symbol, period="1y", interval="1d",rounding=True,end=datetime.now()-timedelta(days=1))
    for pdtimestamp, price_dict in data_his.to_dict('index').items():
        timestamp = Timestamp(pdtimestamp, tz='UTC')
        load_data.info_set.update_or_create(
            id="{}_{}".format(timestamp, stock_symbol),
            open_price=price_dict['Open'],
            high_price=price_dict['High'],
            low_price=price_dict['Low'],
            close_price=price_dict['Close'],
            adj_close_price=price_dict['Adj Close'],
            volume=price_dict['Volume'],
            date=timestamp,
            change=price_dict['Close']-price_dict['Open'],
            change_percent=((price_dict['Close']-price_dict['Open'])/price_dict['Open'])*100,
            )
        load_data.save()
    return load_data


def world_indices(request):
    names = WorldIndices.objects.all()
    context = {}
    con = []
    for index, name in enumerate(names):
        load_data = name.info_set.all()
        index1 = 'tab' + str(index)
        full_info = (index1, name, load_data)
        con.append(full_info)
    context['info'] = con
    return render(request, 'world_indices.html', context)


def world_indices_trend(request):
    load_file_name = WorldIndices.objects.all()
    file_name = []
    con = {}
    for data in load_file_name:
        name = data.file
        file_name.append(name)
    pre_data = []
    pre_name = []
    for index, name in enumerate(file_name):
        result = RNN_torch.main(name)
        pre_data.append(result.values)
        pre_name.append(load_file_name[index])
        pre_date = result.index
    con['name'] = pre_name
    con['date'] = pre_date
    for i in range(4):
        name = 'result'+ str(i)
        con[name] = np.reshape(pre_data[i], (5))
    return render(request, 'trend.html',con)

