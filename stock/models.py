from django.db import models

# Create your models here.


class Stock(models.Model):
    id = models.AutoField(primary_key=True)
    symbol = models.CharField('Symbol', max_length=50, default='')
    company = models.CharField('Company', max_length=50, default='Other company')
    def __str__(self):
        return self.symbol


class StockInfo(models.Model):
    id = models.CharField('timestamp_symbol', max_length=50, primary_key=True)
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE, related_name='info_set')
    open_price = models.DecimalField('Open', max_digits=11, decimal_places=2, default=None)
    high_price = models.DecimalField('High', max_digits=11, decimal_places=2, default=None)
    low_price = models.DecimalField('Low', max_digits=11, decimal_places=2, default=None)
    close_price = models.DecimalField('Close', max_digits=11, decimal_places=2, default=None)
    adj_close_price = models.DecimalField('Adj Close', max_digits=11, decimal_places=2, default=None)
    volume = models.IntegerField('Volume', default=None)
    date = models.DateTimeField('Date', auto_now=False, auto_now_add=False, default=None)
    change = models.DecimalField('change', max_digits=11, decimal_places=2, default=None)
    change_percent = models.DecimalField('change_percent', max_digits=11, decimal_places=2, default=None)

    def __str__(self):
        return "date:{};stock:{};close price:{}".format(self.date, self.stock, self.close_price)


class WorldIndices(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField('Index', max_length=50)
    file = models.CharField('File', max_length=50)

    def __str__(self):
        return self.name

class WorldIndicesInfo(models.Model):
    id = models.CharField('timestamp_symbol', max_length=50, primary_key=True)
    index = models.ForeignKey(WorldIndices, on_delete=models.CASCADE, related_name='info_set')
    open_price = models.DecimalField('Open', max_digits=11, decimal_places=2, default=None)
    high_price = models.DecimalField('High', max_digits=11, decimal_places=2, default=None)
    low_price = models.DecimalField('Low', max_digits=11, decimal_places=2, default=None)
    close_price = models.DecimalField('Close', max_digits=11, decimal_places=2, default=None)
    adj_close_price = models.DecimalField('Adj Close', max_digits=11, decimal_places=2, default=None)
    volume = models.IntegerField('Volume', default=None)
    date = models.DateTimeField('Date', auto_now=False, auto_now_add=False, default=None)
    def __str__(self):
        return "date:{};Index:{};close price:{}".format(self.date, self.index, self.close_price)
