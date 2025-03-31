from django.db import models
from django.utils import timezone

class Device(models.Model):
    device_id = models.CharField(max_length=50, primary_key=True)
    device_type = models.CharField(max_length=10)
    installation_date = models.DateField()
    location = models.CharField(max_length=100)
    
    def __str__(self):
        return self.device_id
    def save(self, *args, **kwargs):
        # First save the device
        super().save(*args, **kwargs)
        
        # Then create an initial reading with default values
        DeviceReading.objects.create(
            device=self,
            metric1=0,
            metric2=0,
            metric3=0,
            metric4=0,
            metric5=0,
            metric6=0,  # This was missing
            metric7=0,
            metric8=0,
            metric9=0
        )
    @classmethod
    def sync_from_training_data(cls, device_ids):
        """Sync devices from a list of IDs"""
        for device_id in device_ids:
            cls.objects.get_or_create(
                device_id=device_id,
                defaults={
                    'device_type': device_id[0],
                    'installation_date': timezone.now().date(),
                    'location': 'Auto-created'
                }
            )    
class DeviceReading(models.Model):
    device = models.ForeignKey(Device, on_delete=models.CASCADE, related_name='readings')
    timestamp = models.DateTimeField(default=timezone.now)
    metric1 = models.FloatField()
    metric2 = models.FloatField()
    metric3 = models.FloatField()
    metric4 = models.FloatField()
    metric5 = models.FloatField()
    metric6 = models.FloatField()
    metric7 = models.FloatField()
    metric8 = models.FloatField()
    metric9 = models.FloatField()
    
    class Meta:
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"{self.device_id} reading at {self.timestamp}"

class MaintenanceEvent(models.Model):
    device = models.ForeignKey(Device, on_delete=models.CASCADE, related_name='maintenance_events')
    event_date = models.DateTimeField(default=timezone.now)
    event_type = models.CharField(max_length=50, choices=[
        ('scheduled', 'Scheduled Maintenance'),
        ('predictive', 'Predictive Maintenance'),
        ('failure', 'Failure Event'),
    ])
    description = models.TextField()
    cost = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    downtime_hours = models.FloatField(null=True, blank=True)
    
    class Meta:
        ordering = ['-event_date']
    
    def __str__(self):
        return f"{self.device_id} - {self.event_type} - {self.event_date}"

class PredictionResult(models.Model):
    device = models.ForeignKey(Device, on_delete=models.CASCADE, related_name='predictions')
    prediction_date = models.DateTimeField(default=timezone.now)
    failure_predicted = models.BooleanField()
    failure_probability = models.FloatField()
    recommended_action = models.TextField()
    
    class Meta:
        ordering = ['-prediction_date']
    
    def __str__(self):
        return f"{self.device_id} - Prediction: {self.failure_predicted} ({self.failure_probability:.2f})"
 
class ModelTrainingLog(models.Model):
    """Tracks model training sessions"""
    timestamp = models.DateTimeField(auto_now_add=True)
    success = models.BooleanField(default=False)
    metrics = models.JSONField(default=dict)
    data_file = models.CharField(max_length=255)
    notes = models.TextField(blank=True)
    version = models.ForeignKey('ModelVersion', on_delete=models.SET_NULL, null=True, blank=True)

    class Meta:
        ordering = ['-timestamp']
        verbose_name = "Training Log"
        verbose_name_plural = "Training Logs"

    def __str__(self):
        return f"Training {'success' if self.success else 'fail'} at {self.timestamp}"

class ModelVersion(models.Model):
    """Tracks different versions of the predictive model"""
    version = models.CharField(max_length=50, unique=True)
    trained_at = models.DateTimeField(auto_now_add=True)
    performance = models.JSONField(default=dict)
    is_active = models.BooleanField(default=False)
    model_path = models.CharField(max_length=255)
    
    class Meta:
        ordering = ['-trained_at']
        verbose_name = "Model Version"
        verbose_name_plural = "Model Versions"
    
    def activate(self):
        """Activate this version and deactivate all others"""
        ModelVersion.objects.exclude(pk=self.pk).update(is_active=False)
        self.is_active = True
        self.save()
    
    def __str__(self):
        return f"v{self.version} ({'active' if self.is_active else 'inactive'})"