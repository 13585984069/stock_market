from django.urls import path
from . import views


app_name = 'stock'
urlpatterns = [
    path('stock/', views.data_info, name='mainpage'),
    path('stock/data/', views.stock_search_data, name='search_data'),
    path('stock/indices/', views.world_indices, name='world_indices'),
    path('stock/data/detail', views.stock_detail_info, name='detail'),
    path('stock/<str:stock_symbol>/', views.stock_click_data, name='click_data'),
    path('stock/indices/trend', views.world_indices_trend, name='trend'),
]