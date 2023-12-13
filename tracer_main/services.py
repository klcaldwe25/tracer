from dataclasses import dataclass   
from tracer_main.daos import *
import numpy as np
from math import pi

class InitialService:
    def run(self):
        return CanvasDAO().create_img()
    
class CircleService:
    ray_origin : PointDAO
    canvas_pixels : int
    canvas : CanvasDAO
    transform_matrix : MatrixDAO
    color : ColorDAO

    def __init__(self):
        self.ray_origin = PointDAO(0, 0, -50)
        self.canvas_pixels = 500
        self.canvas = CanvasDAO()
        self.color = ColorDAO(1, 0, 0)
        self.transform_matrix = IdentityMatrix()

    def run(self, payload):
        wall_z = 10
        wall_size = 7
        pixel_size = wall_size / self.canvas_pixels
        half = wall_size / 2

        if payload["scale"] == True:
            self.transform_matrix = self.transform_matrix.scale(payload["vals"]["x"], payload["vals"]["y"], payload["vals"]["z"])
        if payload["rotate_x"] == True:
            self.transform_matrix = self.transform_matrix.rotate_x(pi/payload["x_val"])
        if payload["rotate_y"] == True:
            self.transform_matrix = self.transform_matrix.rotate_y(pi/payload["y_val"])
        if payload["rotate_z"] == True:
            self.transform_matrix = self.transform_matrix.rotate_z(pi/payload["z_val"])                    

        shape = SphereDAO(self.transform_matrix)
        shape.material.color = ColorDAO(1, 0.2, 1)

        light = LightDAO(PointDAO(-10, 10, -10), ColorDAO(1, 1, 1))

        for y in range(0, self.canvas_pixels-1):
            world_y = half - pixel_size * y
            for x in range(0, self.canvas_pixels-1):
                world_x = -half + pixel_size * x
                position = PointDAO(world_x, world_y, wall_z)
                r = RayDAO(self.ray_origin, position.subtract(self.ray_origin).norm())
                xs = Intersections(shape.intersect(r))

                if xs.hit():
                    hit = xs.hit()
                    point = r.position(hit.t)
                    normal = hit.obj.normal_at(point)
                    eye = -r.direction
                    color = hit.obj.material.lighting(light, point, eye, normal)
                    self.canvas.set_pixel(x, y, color.tuple)

        return self.canvas.create_img()





                

