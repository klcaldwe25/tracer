# Generated by Django 4.2.7 on 2023-11-12 17:13

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('tracer_main', '0004_canvas_delete_tuple'),
    ]

    operations = [
        migrations.AlterField(
            model_name='canvas',
            name='image',
            field=models.ImageField(upload_to='canvas'),
        ),
    ]
