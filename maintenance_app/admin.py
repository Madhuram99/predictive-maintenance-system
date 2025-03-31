from django.contrib import messages
from django.contrib import admin
from .models import Device, DeviceReading, MaintenanceEvent, PredictionResult,ModelVersion, ModelTrainingLog
from django.db.models import Count
from django.utils import timezone
from .forms import DeviceForm, DeviceReadingForm, MaintenanceEventForm, PredictionForm
class DeviceReadingInline(admin.TabularInline):
    model = DeviceReading
    extra = 1
    fields = ('timestamp', 'metric1', 'metric2', 'metric3', 'metric4', 'metric5')
    readonly_fields = ('timestamp',)

class MaintenanceEventInline(admin.TabularInline):
    model = MaintenanceEvent
    extra = 1
    fields = ('event_date', 'event_type', 'description', 'cost', 'downtime_hours')
    readonly_fields = ('event_date',)

class PredictionResultInline(admin.TabularInline):
    model = PredictionResult
    extra = 1
    fields = ('prediction_date', 'failure_predicted', 'failure_probability', 'recommended_action')
    readonly_fields = ('prediction_date',)

@admin.register(Device)
class DeviceAdmin(admin.ModelAdmin):
    form = DeviceForm
    list_display = ('device_id', 'device_type', 'installation_date', 'location')
    search_fields = ('device_id', 'location')
    list_filter = ('device_type', 'installation_date')
    
    def save_model(self, request, obj, form, change):
        # First save the device
        super().save_model(request, obj, form, change)
        
        # Create initial reading if this is a new device
        if not change:
            DeviceReading.objects.create(
                device=obj,
                metric1=0,
                metric2=0,
                metric3=0,
                metric4=0,
                metric5=0,
                metric6=0,
                metric7=0,
                metric8=0,
                metric9=0
            )
    
    # ... existing code ...
    
    actions = ['recalculate_predictions']  # Add this line
    
    @admin.action(description='Recalculate predictions for selected devices')
    def recalculate_predictions(self, request, queryset):
        from maintenance_app.ml_models.predictive_model import PredictiveMaintenanceModel
        model = PredictiveMaintenanceModel()
        created_count = 0
        
        for device in queryset:
            latest_reading = device.readings.order_by('-timestamp').first()
            if latest_reading:
                prediction, probability = model.predict_from_device_reading(latest_reading)
                
                # Determine recommended action
                if probability >= 0.7:
                    action = "Schedule immediate maintenance"
                elif probability >= 0.4:
                    action = "Monitor closely"
                else:
                    action = "No action needed"
                
                # Create or update prediction
                PredictionResult.objects.update_or_create(
                    device=device,
                    defaults={
                        'failure_predicted': prediction,
                        'failure_probability': probability,
                        'recommended_action': action
                    }
                )
                created_count += 1
        
        self.message_user(
            request,
            f"Successfully recalculated predictions for {created_count} devices",
            messages.SUCCESS
        )

@admin.register(DeviceReading)
class DeviceReadingAdmin(admin.ModelAdmin):
    list_display = ('device', 'timestamp', 'metric1', 'metric2', 'metric3')
    list_filter = ('device', 'timestamp')
    search_fields = ('device__device_id',)
    date_hierarchy = 'timestamp'

@admin.register(MaintenanceEvent)
class MaintenanceEventAdmin(admin.ModelAdmin):
    list_display = ('device', 'event_date', 'event_type', 'cost', 'downtime_hours')
    list_filter = ('event_type', 'event_date', 'device')
    search_fields = ('device__device_id', 'description')
    date_hierarchy = 'event_date'

@admin.register(PredictionResult)
class PredictionResultAdmin(admin.ModelAdmin):
    list_display = ('device', 'prediction_date', 'failure_predicted', 'failure_probability')
    list_filter = ('failure_predicted', 'prediction_date', 'device')
    search_fields = ('device__device_id', 'recommended_action')
    date_hierarchy = 'prediction_date'
    
@admin.register(ModelVersion)
class ModelVersionAdmin(admin.ModelAdmin):
    list_display = ('version', 'trained_at', 'is_active')
    list_filter = ('is_active',)
    actions = ['activate_version']
    
    @admin.action(description='Activate selected versions')
    def activate_version(self, request, queryset):
        for version in queryset:
            version.activate()
        self.message_user(request, f"Activated {queryset.count()} versions")

@admin.register(ModelTrainingLog)
class ModelTrainingLogAdmin(admin.ModelAdmin):
    list_display = ('timestamp', 'success', 'version', 'data_file')
    list_filter = ('success', 'version')
    search_fields = ('notes', 'data_file')
    readonly_fields = ('timestamp',)    