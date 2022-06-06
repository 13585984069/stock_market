from django.contrib import admin
from .models import StockInfo, Stock, WorldIndices, WorldIndicesInfo
# Register your models here.

admin.site.register(StockInfo)
admin.site.register(Stock)
admin.site.register(WorldIndices)
admin.site.register(WorldIndicesInfo)
