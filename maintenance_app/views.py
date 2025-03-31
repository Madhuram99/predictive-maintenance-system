
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.views.generic import ListView, DetailView, CreateView, UpdateView
from django.urls import reverse_lazy
from django.contrib import messages
from django.db.models import Avg, Sum, Count,Max,Min
from django.utils import timezone
from datetime import timedelta
import os
from django.contrib.auth.decorators import login_required, permission_required
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.conf import settings
import logging
# Correct import (in models.py, views.py, or wherever the error occurs)
from django.utils import timezone
from django.core.cache import cache
from django.contrib import messages
from maintenance_app.models import ModelVersion
from .models import Device, DeviceReading, MaintenanceEvent, PredictionResult
from .forms import DeviceForm, DeviceReadingForm, MaintenanceEventForm, PredictionForm
from .ml_models.predictive_model import PredictiveMaintenanceModel
from maintenance_app.ml_models.predictive_model import PredictiveMaintenanceModel
logger = logging.getLogger(__name__)

def index(request):
    # Dashboard data
    model = PredictiveMaintenanceModel()
    active_version = model.get_active_version()
    versions = ModelVersion.objects.all()[:5]
    
    # Correct counting
    total_devices = Device.objects.count()
    readings_last_day = DeviceReading.objects.filter(
        timestamp__gte=timezone.now() - timedelta(days=1)
    ).count()  # count() is on QuerySet
    
    recent_predictions = PredictionResult.objects.order_by('-prediction_date')[:5]
    high_risk_devices = PredictionResult.objects.filter(
        failure_probability__gte=0.7,
        prediction_date__gte=timezone.now() - timedelta(days=7)
    ).values('device').annotate(
        max_prob=Max('failure_probability')
    ).order_by('-max_prob')[:5]
    
    context = {
        'model_info': model.get_model_info(),
        'active_version': active_version,
        'model_versions': versions,
        'total_devices': total_devices,
        'readings_last_day': readings_last_day,
        'recent_predictions': recent_predictions,
        'high_risk_devices': high_risk_devices,
    }
    return render(request, 'maintenance_app/index.html', context)

def make_prediction(request):
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            device = form.cleaned_data['device']
            
            try:
                # Get the latest reading for this device
                latest_reading = DeviceReading.objects.filter(
                    device=device
                ).latest('timestamp')
                
                # Initialize and use the model
                # Initialize model with logging
                model = PredictiveMaintenanceModel()
                logger.info(f"Making prediction for device {device.device_id}")
                logger.info(f"Model type: {type(model.model)}")
                if hasattr(model.model, 'feature_importances_'):
                    logger.info(f"Model has {len(model.model.feature_importances_)} features")
                
                prediction, probability = model.predict_from_device_reading(latest_reading)
                
                # Determine recommended action
                if probability >= 0.7:
                    recommended_action = (
                        "Immediate maintenance recommended. "
                        "Failure probability is high."
                    )
                    alert_level = 'danger'
                elif probability >= 0.4:
                    recommended_action = (
                        "Monitor device closely. "
                        "Consider scheduling maintenance soon."
                    )
                    alert_level = 'warning'
                else:
                    recommended_action = (
                        "No immediate action needed. "
                        "Device operating within normal parameters."
                    )
                    alert_level = 'success'
                
                # Save prediction result
                result = PredictionResult.objects.create(
                    device=device,
                    failure_predicted=prediction,
                    failure_probability=probability,
                    recommended_action=recommended_action
                )
                
                # Add message for the user
                messages.add_message(
                    request,
                    messages.INFO if alert_level == 'success' else messages.WARNING,
                    f"Prediction complete for {device.device_id}. "
                    f"Failure probability: {probability:.1%}"
                )
                
                return redirect('prediction-results', pk=result.pk)
            
            except DeviceReading.DoesNotExist:
                messages.error(request, "No readings available for this device")
    else:
        # Check if device_id was passed in GET parameters
        device_id = request.GET.get('device')
        initial = {'device': device_id} if device_id else {}
        form = PredictionForm(initial=initial)
    
    return render(request, 'maintenance_app/predict.html', {'form': form})


# Device views
class DeviceListView(ListView):
    model = Device
    context_object_name = 'devices'
    
class DeviceDetailView(DetailView):
    model = Device
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        device = self.get_object()
        context['readings'] = device.readings.all()[:10]
        context['maintenance_events'] = device.maintenance_events.all()[:10]
        context['predictions'] = device.predictions.all()[:10]
        return context

class DeviceCreateView(CreateView):
    model = Device
    form_class = DeviceForm
    template_name = 'maintenance_app/device_form.html'
    success_url = reverse_lazy('device-list')  # Make sure this URL name exists
# Device Reading views
class DeviceReadingCreateView(CreateView):
    model = DeviceReading
    form_class = DeviceReadingForm
    
    def get_success_url(self):
        return reverse_lazy('device-detail', kwargs={'pk': self.object.device.pk})
    
# Maintenance Event views
class MaintenanceEventCreateView(CreateView):
    model = MaintenanceEvent
    form_class = MaintenanceEventForm
    
    def get_success_url(self):
        return reverse_lazy('device-detail', kwargs={'pk': self.object.device.pk})

# API endpoints for dashboard charts
def device_health_data(request):
    # Get the latest prediction for each device
    devices = Device.objects.all()
    data = []
    
    for device in devices:
        try:
            latest_prediction = device.predictions.latest('prediction_date')
            data.append({
                'device_id': device.device_id,
                'probability': latest_prediction.failure_probability,
                'health_score': 100 - (latest_prediction.failure_probability * 100)
            })
        except PredictionResult.DoesNotExist:
            # No predictions yet for this device
            data.append({
                'device_id': device.device_id,
                'probability': 0,
                'health_score': 100
            })
    
    return JsonResponse(data, safe=False)

# maintenance_app/views.py
from django.db.models.functions import TruncMonth


def maintenance_history_data(request):
    """Get maintenance events grouped by month for the last 12 months"""
    last_12_months = timezone.now() - timedelta(days=365)
    
    # Use Django's TruncMonth for SQLite compatibility
    events = MaintenanceEvent.objects.filter(
        event_date__gte=last_12_months
    ).annotate(
        month=TruncMonth('event_date')
    ).values('month', 'event_type').annotate(
        count=Count('id')
    ).order_by('month', 'event_type')
    
    # Convert the QuerySet to a list of dictionaries
    result = []
    for event in events:
        result.append({
            'month': event['month'].strftime('%Y-%m'),
            'event_type': event['event_type'],
            'count': event['count']
        })
    
    return JsonResponse(result, safe=False)
def train_model_view(request):
    if request.method == 'POST':
        # Path to your training data
        data_file = os.path.join(settings.BASE_DIR, 'data/training_data.csv')
        
        # Check if file exists
        if not os.path.exists(data_file):
            messages.error(request, "Training data file not found")
            return HttpResponseRedirect(reverse('train-model'))
        
        try:
            # Run the training command
            from django.core.management import call_command
            call_command('train_model', data_file)
            messages.success(request, "Model trained successfully!")
        except Exception as e:
            messages.error(request, f"Error training model: {str(e)}")
        
        return HttpResponseRedirect(reverse('train-model'))
    
    # Check if model exists
    model_exists = os.path.exists(os.path.join(
        settings.BASE_DIR, 
        'maintenance_app/ml_models/predictive_maintenance_model.pkl'
    ))
    
    return render(request, 'maintenance_app/train_model.html', {
        'model_exists': model_exists
    })
    
def prediction_results(request, pk):
    result = PredictionResult.objects.get(pk=pk)
    return render(request, 'maintenance_app/results.html', {
        'result': result,
        'device': result.device
    })    

def model_status(request):
    """API endpoint to check model status"""
    model = PredictiveMaintenanceModel()
    return JsonResponse(model.get_model_info())