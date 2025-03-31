from django.core.management.base import BaseCommand
from maintenance_app.ml_models.predictive_model import PredictiveMaintenanceModel
from maintenance_app.models import ModelVersion, ModelTrainingLog, PredictionResult, Device, DeviceReading
import os
from django.conf import settings
from django.utils import timezone
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import uuid
import numpy as np
import logging
import pickle
from scipy.stats import truncnorm  # For generating bounded normal distributions

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Train and activate new predictive maintenance model version'
    
    def add_arguments(self, parser):
        parser.add_argument('data_file', type=str, help='Path to training data CSV')
        parser.add_argument('--keep-predictions', action='store_true', 
                          help='Keep existing predictions')
    
    def _get_realistic_metrics(self, df, device_id):
        """Generate realistic initial metrics based on training data"""
        device_data = df[df['device'] == device_id]
        metrics = {}
        
        for i in range(1, 10):
            col = f'metric{i}'
            if col in device_data.columns:
                # Get stats for this device's metrics
                mean = device_data[col].mean()
                std = device_data[col].std()
                
                # Generate value within 2 standard deviations of mean
                lower, upper = max(0, mean - 2*std), mean + 2*std
                metrics[col] = truncnorm(
                    (lower - mean) / std, 
                    (upper - mean) / std, 
                    loc=mean, 
                    scale=std
                ).rvs(1)[0]
            else:
                metrics[col] = 0.0
                
        return metrics
    
    def handle(self, *args, **options):
        data_file = options['data_file']
        keep_predictions = options['keep_predictions']
        
        # Generate version ID and paths
        version_id = str(uuid.uuid4())[:8]
        model_path = os.path.join(
            PredictiveMaintenanceModel.MODEL_DIR, 
            f'model_v{version_id}.pkl'
        )
        os.makedirs(PredictiveMaintenanceModel.MODEL_DIR, exist_ok=True)
        
        try:
            # Initialize and train model
            model = PredictiveMaintenanceModel()
            
            # Load and validate data
            self.stdout.write(self.style.SUCCESS(f"Loading data from {data_file}"))
            df = model.load_data(data_file)
            
            # SYNC DEVICES FROM TRAINING DATA WITH REALISTIC METRICS
            device_ids = df['device'].unique()
            new_devices = 0
            
            for device_id in device_ids:
                device, created = Device.objects.get_or_create(
                    device_id=device_id,
                    defaults={
                        'device_type': device_id[0],
                        'installation_date': timezone.now().date(),
                        'location': 'Training Import'
                    }
                )
                
                if created:
                    new_devices += 1
                    # Create initial reading with realistic values
                    metrics = self._get_realistic_metrics(df, device_id)
                    DeviceReading.objects.create(
                        device=device,
                        timestamp=timezone.now(),
                        **metrics
                    )
                    self.stdout.write(f"Created new device: {device_id} with realistic metrics")
            
            self.stdout.write(self.style.SUCCESS(
                f"Created {new_devices} new devices with realistic initial metrics"
            ))
            
            # Verify class distribution
            class_counts = df['failure'].value_counts()
            if len(class_counts) < 2:
                msg = f"Training data needs both classes. Current distribution:\n{class_counts}"
                ModelTrainingLog.objects.create(
                    success=False,
                    metrics={},
                    data_file=data_file,
                    notes=msg
                )
                raise ValueError(msg)
            
            # Data processing pipeline
            self.stdout.write(self.style.SUCCESS("Preprocessing data..."))
            df = model.preprocess_data()
            
            self.stdout.write(self.style.SUCCESS("Feature engineering..."))
            df = model.feature_engineering()
            
            self.stdout.write(self.style.SUCCESS("Preparing train/test data..."))
            X_train, X_test, y_train, y_test = model.prepare_data()
            
            self.stdout.write(self.style.SUCCESS("Training model..."))
            model.train_model(X_train, y_train)
            
            # Evaluate model
            metrics = model.evaluate_model(X_test, y_test)
            
            # Save model
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': model.model,
                    'scaler': model.scaler,
                    'feature_selector': model.feature_selector
                }, f)
            
            # Create and activate version
            version = ModelVersion.objects.create(
                version=version_id,
                performance=metrics,
                model_path=model_path,
                is_active=True
            )
            
            # Deactivate other versions
            ModelVersion.objects.exclude(pk=version.pk).update(is_active=False)
            
            # Log training session
            ModelTrainingLog.objects.create(
                success=True,
                metrics=metrics,
                data_file=data_file,
                version=version,
                notes=f"Successfully trained v{version_id}"
            )
            
            # Clear old predictions unless --keep-predictions
            if not keep_predictions:
                deleted_count = PredictionResult.objects.all().delete()[0]
                self.stdout.write(self.style.SUCCESS(f"Cleared {deleted_count} old predictions"))
            
            # Output results
            self.stdout.write(self.style.SUCCESS(
                f"\nSuccessfully trained and activated model v{version_id}\n"
                f"Accuracy: {metrics['accuracy']:.4f}\n"
                f"ROC AUC: {metrics.get('roc_auc', 'N/A')}\n"
                f"Model saved to: {model_path}"
            ))
            
        except Exception as e:
            # Log failed training attempt
            ModelTrainingLog.objects.create(
                success=False,
                metrics={},
                data_file=data_file,
                notes=f"Training failed: {str(e)}"
            )
            
            # Clean up failed model file
            if os.path.exists(model_path):
                os.remove(model_path)
                
            logger.error(f"Training failed: {str(e)}")
            self.stdout.write(self.style.ERROR(f"Training failed: {str(e)}"))
            raise