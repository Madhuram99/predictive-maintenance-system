from django import forms
from .models import Device, DeviceReading, MaintenanceEvent
from django.utils import timezone
class DeviceForm(forms.ModelForm):
    class Meta:
        model = Device
        fields = ['device_id', 'device_type', 'installation_date', 'location']
        widgets = {
            'installation_date': forms.DateInput(attrs={'type': 'date'}),
        }

class DeviceReadingForm(forms.ModelForm):
    class Meta:
        model = DeviceReading
        fields = ['device', 'metric1', 'metric2', 'metric3', 'metric4', 
                  'metric5', 'metric6', 'metric7', 'metric8', 'metric9']
        
class MaintenanceEventForm(forms.ModelForm):
    class Meta:
        model = MaintenanceEvent
        fields = ['device', 'event_type', 'description', 'cost', 'downtime_hours']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 3}),
        }

class PredictionForm(forms.Form):
    device = forms.ModelChoiceField(queryset=Device.objects.all())