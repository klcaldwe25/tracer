from django.shortcuts import render
from tracer_main.services import *
import base64

def main(request):
    context = {"graphic" : base64.b64encode(InitialService().run()).decode('utf-8')}
    if request.method == "POST":
        context = {"graphic" : base64.b64encode(CirclesService().run()).decode('utf-8')}   

    return render(request, "tracer_main/main.html", context)

"""
        if selection:
            match selection:
                case "circle":
                    payload = { 'scale' : False, 
                                'vals' : {
                                    'x' : 1,
                                    'y' : 1,
                                    'z' : 1
                                },
                                'rotate_x' : False, 'x_val' : 0,
                                'rotate_y' : False, 'y_val' : 0,
                                'rotate_z' : False, 'z_val' :0}
                    if request.POST.get("scale_x") != "":
                        payload["scale"] = True
                        payload["vals"]["x"] = float(request.POST.get("scale_x"))
                    if request.POST.get("scale_y") != "":
                        payload["scale"] = True
                        payload["vals"]["y"] = float(request.POST.get("scale_y"))
                    if request.POST.get("scale_z") != "":
                        payload["scale"] = True
                        payload["vals"]["z"] = float(request.POST.get("scale_z"))
                    if request.POST.get("rotate_x") != "":
                        payload["rotate_x"] = True
                        payload["x_val"] = float(request.POST.get("rotate_x"))                        
                    if request.POST.get("rotate_y") != "":
                        payload["rotate_y"] = True
                        payload["y_val"] = float(request.POST.get("rotate_y"))   
                    if request.POST.get("rotate_z") != "":
                        payload["rotate_z"] = True
                        payload["z_val"] = float(request.POST.get("rotate_z"))    

                    context = {"graphic" : base64.b64encode(CircleService().run(payload)).decode('utf-8')}
                
                case "circles":
                    context = {"graphic" : base64.b64encode(CirclesService().run()).decode('utf-8')}       
"""