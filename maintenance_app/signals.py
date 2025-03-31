# signals.py
from datetime import timedelta, timezone
from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from django.core.cache import cache
import pandas as pd
from sklearn import logger
from .models import Device, DeviceReading, ModelTrainingLog
from django.utils import timezone
@receiver([post_save, post_delete], sender=Device)
def update_device_count(sender, **kwargs):
    cache.set('total_devices', Device.objects.count(), 3600)

@receiver(post_save, sender=DeviceReading)
def update_reading_counts(sender, **kwargs):
    # Make sure count() is called on the QuerySet, not timedelta
    count = DeviceReading.objects.filter(
        timestamp__gte=timezone.now() - timedelta(days=1)
    ).count()  # Correct
    
    cache.set('daily_readings', count, 3600)
    
@receiver(post_save, sender=ModelTrainingLog)
def sync_devices_on_train(sender, instance, **kwargs):
    if instance.success and instance.data_file:
        try:
            df = pd.read_csv(instance.data_file)
            Device.sync_from_training_data(df['device'].unique())
        except Exception as e:
            logger.error(f"Failed to sync devices from training log: {str(e)}")    