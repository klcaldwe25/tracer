from dataclasses import dataclass   
from tracer_main.daos import *
import numpy as np
from math import pi

@dataclass  
class ProjectileDTO:
    position : PointDAO
    velocity : VectorDAO

@dataclass
class EnviromentDTO:
    gravity : VectorDAO
    wind : VectorDAO

class ProjectileService:
    def fire(self, name, height, width):
        env = EnviromentDTO(VectorDAO(0, -0.1, 0), VectorDAO(-0.01, 0, 0))
        proj = ProjectileDTO(PointDAO(0, 1, 0), VectorDAO(1, 1.8, 0).norm().multiply(10))
        canvas = CanvasDAO(name, height, width)

        while proj.position.tuple[1] >= 0:
            x = int(proj.position.tuple[0])
            y = int(proj.position.tuple[1])
            if x <= width and y <= height:
                canvas.set_pixel(x, canvas.height - y, [1, 0, 0])    
                proj = self.tick(env, proj)
            else:
                break
            
        return canvas.create_img()
    
    def tick(self, env, proj):
        position = proj.position.add(proj.velocity)
        velocity = proj.velocity.add(env.gravity)
        return ProjectileDTO(position, velocity)
    
class ClockService:
    def run(self, name, height, width):
        canvas = CanvasDAO(name, height, width)
        center = PointDAO(0, 0, 0)
        twelve = PointDAO(0, 1, 0)
        scale_val = height * 3/8

        for hour in range(1,13):
            point = twelve.rotate_z(hour * pi / 6)
            scaled_point = point.scale(scale_val, scale_val, 0).translate(round(width/2), round(height/2), 0)
            canvas.set_pixel(round(scaled_point.tuple[0]), round(scaled_point.tuple[1]), [1, 0, 0])
            canvas.set_pixel(round(scaled_point.tuple[0])+1, round(scaled_point.tuple[1]), [1, 0, 0])
            canvas.set_pixel(round(scaled_point.tuple[0]), round(scaled_point.tuple[1])+1, [1, 0, 0])
            canvas.set_pixel(round(scaled_point.tuple[0])-1, round(scaled_point.tuple[1]), [1, 0, 0])
            canvas.set_pixel(round(scaled_point.tuple[0]), round(scaled_point.tuple[1])-1, [1, 0, 0])

        return canvas.create_img()
