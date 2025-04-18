# Generated by Django 5.1.7 on 2025-03-31 13:34

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('maintenance_app', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='ModelTrainingLog',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('training_date', models.DateTimeField(auto_now_add=True)),
                ('success', models.BooleanField()),
                ('metrics', models.JSONField()),
                ('data_file', models.CharField(max_length=255)),
                ('notes', models.TextField(blank=True)),
            ],
            options={
                'ordering': ['-training_date'],
            },
        ),
    ]
