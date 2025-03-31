from django.urls import path
from . import views

urlpatterns = [
    # Dashboard
    path('', views.index, name='index'),
    
    # Prediction
    path('predict/', views.make_prediction, name='predict'),
    path('predict/results/<int:pk>/', views.prediction_results, name='prediction-results'),
    # Devices
    path('devices/', views.DeviceListView.as_view(), name='device-list'),
    path('devices/<str:pk>/', views.DeviceDetailView.as_view(), name='device-detail'),
    path('devices/create/', views.DeviceCreateView.as_view(), name='device-create'),
    
    # Device readings
    path('readings/create/', views.DeviceReadingCreateView.as_view(), name='reading-create'),
    
    # Maintenance events
    path('maintenance/create/', views.MaintenanceEventCreateView.as_view(), name='maintenance-create'),
    
    # API endpoints for charts
    path('api/device-health/', views.device_health_data, name='device-health-data'),
    path('api/maintenance-history/', views.maintenance_history_data, name='maintenance-history-data'),
    
     # Model training
    path('train-model/', views.train_model_view, name='train-model'),
]
