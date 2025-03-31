from django import template
from maintenance_app.ml_models.predictive_model import PredictiveMaintenanceModel

register = template.Library()

@register.simple_tag
def get_model_info():
    model = PredictiveMaintenanceModel()
    return model.get_model_info()