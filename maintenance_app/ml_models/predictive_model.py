from uuid import UUID
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score,accuracy_score
import pickle
from django.utils import timezone
import os
from django.conf import settings
import logging
from django.core.cache import cache
from maintenance_app.models import ModelVersion, DeviceReading

# Constants
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Setup logging
logger = logging.getLogger(__name__)

class PredictiveMaintenanceModel:
    MODEL_DIR = os.path.join(settings.BASE_DIR, 'maintenance_app/ml_models')
    MODEL_PATH = os.path.join(MODEL_DIR, 'predictive_maintenance_model.pkl')
    SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
    
    def __init__(self):
     self.model = None
     self.scaler = None
     self.feature_selector = None
     self.df = None
     self.selected_features = None
     self.current_version = None
    
     # Ensure model directory exists
     os.makedirs(self.MODEL_DIR, exist_ok=True)
    
     # Try to load the active model
     try:
        self.load_active_model()
     except Exception as e:
        logger.warning(f"Could not load model on initialization: {e}")
        # Initialize a simple model for testing if no saved model
        self.model = GradientBoostingClassifier(random_state=RANDOM_STATE)
        self.scaler = StandardScaler()
    
    def load_data(self, file_path):
        """Load and preprocess the data"""
        self.df = pd.read_csv(file_path)
        
        # Validate required columns
        required_columns = ['date', 'device', 'failure'] + [f'metric{i}' for i in range(1, 10)]
        missing_cols = [col for col in required_columns if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        # SYNC DEVICES FROM DATA
        from maintenance_app.models import Device
        for device_id in self.df['device'].unique():
         Device.objects.get_or_create(
            device_id=device_id,
            defaults={
                'device_type': device_id[0],
                'installation_date': timezone.now().date(),
                'location': 'Auto-created'
            }
        )    
        return self.df

    def preprocess_data(self):
        """Convert date to datetime and extract features"""
        self.df['date'] = pd.to_datetime(self.df['date'], format='%m/%d/%Y', errors='coerce')
        self.df['date'] = self.df['date'].fillna(pd.NaT)
        
        # Extract temporal features
        self.df['day_of_week'] = self.df['date'].dt.dayofweek
        self.df['day_of_month'] = self.df['date'].dt.day
        self.df['month'] = self.df['date'].dt.month
        self.df['year'] = self.df['date'].dt.year
        
        # Extract device type from device ID (first character)
        self.df['device_type'] = self.df['device'].str[0]
        
        # One-hot encode device type
        self.df = pd.get_dummies(self.df, columns=['device_type'], prefix='device')
        
        # Drop original date and device columns
        self.df = self.df.drop(['date', 'device'], axis=1)
        
        return self.df

    def feature_engineering(self):
      """Create additional features from existing ones"""
      # Create rolling features for each metric
      metric_cols = [f'metric{i}' for i in range(1, 10)]
    
      for col in metric_cols:
        # Skip if column has no variance
        if self.df[col].nunique() == 1:
            continue
            
        self.df[f'{col}_rolling_mean_7'] = self.df[col].rolling(window=7, min_periods=1).mean()
        self.df[f'{col}_rolling_std_7'] = self.df[col].rolling(window=7, min_periods=1).std()
        self.df[f'{col}_diff_1'] = self.df[col].diff(1)
    
      # Drop rows with NaN values and columns with no variance
      self.df = self.df.dropna()
      self.df = self.df.loc[:, self.df.nunique() > 1]
    
      return self.df

    def prepare_data(self):
      """Prepare features and target variable"""
      # Separate features and target
      X = self.df.drop('failure', axis=1)
      y = self.df['failure']
    
      # Handle class imbalance
      smote = SMOTE(random_state=RANDOM_STATE)
      X_res, y_res = smote.fit_resample(X, y)
    
      # Feature selection with variance threshold first
      from sklearn.feature_selection import VarianceThreshold
      selector = VarianceThreshold(threshold=0.01)  # Remove low-variance features
      X_filtered = selector.fit_transform(X_res)
    
      # Then apply SelectKBest
      selector = SelectKBest(f_classif, k=min(20, X_filtered.shape[1]))
      X_selected = selector.fit_transform(X_filtered, y_res)
      self.selected_features = X.columns[selector.get_support()]
    
      # Split data ensuring both classes are represented
      from sklearn.model_selection import train_test_split
      X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_res, 
        test_size=TEST_SIZE,
        stratify=y_res,
        random_state=RANDOM_STATE
    )
    
      # Scale features
      self.scaler = StandardScaler()
      X_train_scaled = self.scaler.fit_transform(X_train)
      X_test_scaled = self.scaler.transform(X_test)
    
      return X_train_scaled, X_test_scaled, y_train, y_test
    def train_model(self, X_train, y_train):
        """Train the predictive model"""
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=RANDOM_STATE,
            subsample=0.8
        )
        
        self.model.fit(X_train, y_train)
        return self.model
    
    def evaluate_model(self, X_test, y_test):
      """Evaluate model performance with safe metrics"""
      if not self.model:
        raise ValueError("Model has not been trained yet")
        
      y_pred = self.model.predict(X_test)
    
      metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
      }
    
      # Only calculate metrics that make sense for the data
      if len(np.unique(y_test)) > 1:
        metrics.update({
            'classification_report': classification_report(
                y_test, y_pred, 
                output_dict=True,
                zero_division=0
            )
        })
        
        if hasattr(self.model, "predict_proba"):
            y_prob = self.model.predict_proba(X_test)[:, 1]
            metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
    
      return metrics
    
    def save_model(self, version_id=None):
     """Save model and scaler to disk with version tracking"""
     if not self.model or not self.scaler:
        raise ValueError("Model or scaler not initialized")
    
     # Generate version ID if not provided
     if not version_id:
        version_id = str(UUID.uuid4())[:8]
    
     model_path = os.path.join(self.MODEL_DIR, f'model_v{version_id}.pkl')
    
     try:
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_selector': self.feature_selector,
                'selected_features': self.selected_features
            }, f)
        
        logger.info(f"Model saved to {model_path}")
        return model_path
     except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

    def load_saved_model(self):
        """Load saved model and scaler from disk"""
        try:
            with open(self.MODEL_PATH, 'rb') as f:
                saved_data = pickle.load(f)
                self.model = saved_data['model']
                self.scaler = saved_data['scaler']
                self.selected_features = saved_data['selected_features']
                
            logger.info("Model loaded successfully")
            return True
        except FileNotFoundError:
            logger.warning("Saved model not found")
            return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def reload_model(self):
        """Force reload the model from disk"""
        cache_key = 'predictive_model_instance'
        cache.delete(cache_key)
        return self.load_saved_model()

    def predict_from_device_reading(self, device_reading):
        """Make prediction from a DeviceReading model instance"""
        if not self.model and not self.load_saved_model():
            logger.warning("No model available, using default prediction")
            return False, 0.1  # Default prediction
        
        # Extract features from device reading
        data = self._prepare_prediction_data(device_reading)
        
        # Make prediction
        return self.predict_failure(data)

    def _prepare_prediction_data(self, device_reading):
        """Prepare prediction data from device reading"""
        data = {
            'metric1': device_reading.metric1,
            'metric2': device_reading.metric2,
            'metric3': device_reading.metric3,
            'metric4': device_reading.metric4,
            'metric5': device_reading.metric5,
            'metric6': device_reading.metric6,
            'metric7': device_reading.metric7,
            'metric8': device_reading.metric8,
            'metric9': device_reading.metric9,
            'day_of_week': device_reading.timestamp.weekday(),
            'day_of_month': device_reading.timestamp.day,
            'month': device_reading.timestamp.month,
            'year': device_reading.timestamp.year,
        }
        
        # Add one-hot encoding for device type
        device_type = device_reading.device.device_type[0]
        for d_type in ['S', 'W', 'Z']:
            data[f'device_{d_type}'] = 1 if device_type == d_type else 0
        
        # Calculate rolling features from recent readings
        recent_readings = DeviceReading.objects.filter(
            device=device_reading.device
        ).order_by('-timestamp')[:7]
        
        self._calculate_rolling_features(data, recent_readings)
        
        return data

    def _calculate_rolling_features(self, data, recent_readings):
        """Calculate rolling features for prediction"""
        metric_cols = [f'metric{i}' for i in range(1, 10)]
        for col in metric_cols:
            values = [getattr(reading, col) for reading in recent_readings]
            
            if values:
                data[f'{col}_rolling_mean_7'] = sum(values) / len(values)
                data[f'{col}_rolling_std_7'] = np.std(values) if len(values) > 1 else 0
                data[f'{col}_diff_1'] = values[0] - values[1] if len(values) > 1 else 0
            else:
                data[f'{col}_rolling_mean_7'] = data[col]
                data[f'{col}_rolling_std_7'] = 0
                data[f'{col}_diff_1'] = 0

    def predict_failure(self, device_data):
        """Make predictions on new device data"""
        if not self.model or not self.scaler:
            if not self.load_saved_model():
                logger.warning("No model available, returning default prediction")
                return False, 0.1
            
        # Create DataFrame from input data
        df = pd.DataFrame([device_data])
        
        # Handle feature selection if available
        if self.selected_features is not None:
            for feature in self.selected_features:
                if feature not in df.columns:
                    df[feature] = 0
            df = df[self.selected_features]
        
        # Scale features
        try:
            X = self.scaler.transform(df)
        except Exception as e:
            logger.error(f"Scaling error: {e}")
            X = df.values
        
        # Make prediction
        try:
            failure_probability = self.model.predict_proba(X)[0, 1]
            failure_predicted = failure_probability > 0.5
            return failure_predicted, failure_probability
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return False, 0.1
    
    def get_active_version(self):
     """Get the currently active model version from database"""
     try:
        return ModelVersion.objects.filter(is_active=True).first()
     except Exception as e:
        logger.error(f"Error getting active version: {e}")
        return None

    def load_active_model(self):
     """Load the currently active model version"""
     active_version = self.get_active_version()
     if not active_version:
        logger.warning("No active model version found")
        return False
    
     try:
        with open(active_version.model_path, 'rb') as f:
            saved_data = pickle.load(f)
            self.model = saved_data['model']
            self.scaler = saved_data['scaler']
            self.feature_selector = saved_data.get('feature_selector')
            self.current_version = active_version.version
            logger.info(f"Loaded active model version: {active_version.version}")
            return True
     except Exception as e:
        logger.error(f"Error loading active model: {e}")
        return False    
    
    def get_model_info(self):
     """Return information about the currently loaded model"""
     if not self.model:
        return {
            'status': 'no_model_loaded',
            'message': 'No predictive model is currently loaded'
        }
    
     info = {
        'model_type': type(self.model).__name__,
        'features_count': len(self.selected_features) if self.selected_features else 0,
        'current_version': self.current_version,
        'is_active': False
     }
    
     # Add version info if available
     active_version = self.get_active_version()
     if active_version:
        info.update({
            'is_active': True,
            'version_info': {
                'version': active_version.version,
                'trained_at': active_version.trained_at,
                'performance': active_version.performance
            }
        })
    
     return info