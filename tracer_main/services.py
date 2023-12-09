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
    
class CircleService:
    def run(self, name):
        ray_origin = PointDAO(0, 0, -5)
        wall_z = 10
        wall_size = 7
        canvas_pixels = 100
        pixel_size = wall_size / canvas_pixels
        half = wall_size / 2
        color = ColorDAO(1, 0, 0)

        canvas = CanvasDAO(name, canvas_pixels, canvas_pixels)
        shape = SphereDAO()

        for y in range(0, canvas_pixels-1):
            world_y = half - pixel_size * y
            for x in range(0, canvas_pixels-1):
                world_x = -half + pixel_size * x
                position = PointDAO(world_x, world_y, wall_z)
                r = RayDAO(ray_origin, position.subtract(ray_origin).norm())
                xs = shape.intersect(r)

                if xs:
                    canvas.set_pixel(x, y, color.tuple)

        return canvas.create_img()





                

