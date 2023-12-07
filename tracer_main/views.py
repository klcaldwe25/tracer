from django.shortcuts import render
from tracer_main.services import *
from tracer_main.models import Canvas
import base64

def main(request):
    context = {"img" : ""}
    if request.method == "POST":
        name = request.POST.get("name")
        selection = request.POST.get("shape_type")
        if selection and name:
            match selection:
                case "projectile":                   
                    context = {"graphic" : base64.b64encode(ProjectileService().fire(name, 550, 900)).decode('utf-8')}
                case "clock":
                    context = {"graphic" : base64.b64encode(ClockService().run(name, 500, 500)).decode('utf-8')}
                
            
    return render(request, "tracer_main/main.html", context)