# maintenance_app/utils/data_preparation.py
import pandas as pd
from maintenance_app.models import Device, DeviceReading
from datetime import timedelta
from django.utils import timezone

def export_device_readings_to_csv(output_path, days_back=365):
    """
    Export device readings from the database to a CSV file in the format
    expected by the predictive model.
    """
    # Calculate the date range
    end_date = timezone.now()
    start_date = end_date - timedelta(days=days_back)
    
    # Get all readings in the date range
    readings = DeviceReading.objects.filter(
        timestamp__gte=start_date,
        timestamp__lte=end_date
    ).select_related('device').order_by('timestamp')
    
    # Prepare data in the expected format
    data = []
    for reading in readings:
        data.append({
            'date': reading.timestamp.strftime('%m/%d/%Y'),
            'device': reading.device.device_id,
            'metric1': reading.metric1,
            'metric2': reading.metric2,
            'metric3': reading.metric3,
            'metric4': reading.metric4,
            'metric5': reading.metric5,
            'metric6': reading.metric6,
            'metric7': reading.metric7,
            'metric8': reading.metric8,
            'metric9': reading.metric9,
            'failure': 0  # Default to no failure - needs to be updated based on maintenance records
        })
    
    # Convert to DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    return df