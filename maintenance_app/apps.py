from django.apps import AppConfig


class MaintenanceAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'maintenance_app'
    def ready(self):
        import maintenance_app.signals