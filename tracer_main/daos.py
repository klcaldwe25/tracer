from io import BytesIO
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import cos, sin, sqrt


class MatrixDAO:
    matrix : np.ndarray

    def __init__(self, matrix):
        self.matrix = np.array(matrix)

    def get_rows(self):
        return len(self.matrix[:])
    
    def get_cols(self):
        return len(self.matrix[:][0])

    def get_element(self, row, col):
        return self.matrix[row][col]

    def set_element(self, row, col, val):
        self.matrix[row][col] = val

    def __eq__(self, other):
        return np.allclose(self.matrix, other.matrix, 0.00001, 0.00001)    

    def __mul__(self, other):
        if isinstance(other, MatrixDAO):
            return MatrixDAO(self.matrix @ other.matrix)
        elif isinstance(other, TupleDAO):
            return TupleDAO(np.ravel((self.matrix @ other.tuple.reshape(-1,1)).reshape(1, -1))) 

    def transpose(self):
        return MatrixDAO(self.matrix.transpose())

    def __invert__(self):
        try:
            return MatrixDAO(np.linalg.inv(self.matrix))
        except:
            return "Not invertible" 
        
    def translate(self, x, y, z, inverse=False):
        if inverse:
            return self * ~TranslationMatrix(x, y, z)

        return self * TranslationMatrix(x, y, z)   

    def scale(self, x, y, z, inverse=False):
        if inverse:
            return self * ~ScalingMatrix(x, y, z)
        
        return self * ScalingMatrix(x, y, z)
           
    def rotate_x(self, r, inverse=False):
        if inverse:
            return self * ~RotateX(r)

        return self * RotateX(r)
    
    def rotate_y(self, r, inverse=False):
        if inverse:
            return self * ~RotateY(r)

        return RotateY(r) * self

    def rotate_z(self, r, inverse=False):
        if inverse:
            return self * ~RotateZ(r)

        return self * RotateZ(r)
    
    def shear(self, xy, xz, yx, yz, zx, zy):
        return self * ShearingMatrix(xy, xz, yx, yz, zx, zy)
        
class IdentityMatrix(MatrixDAO):
    def __init__(self):
        self.matrix = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])

class TranslationMatrix(MatrixDAO):
    def __init__(self, x, y, z):
        self.matrix = np.array([[1, 0, 0, x],
                                [0, 1, 0, y],
                                [0, 0, 1, z],
                                [0, 0, 0, 1]])
        
class ScalingMatrix(MatrixDAO):
    def __init__(self, x, y, z):
        self.matrix = np.array([[x, 0, 0, 0],
                                [0, y, 0, 0],
                                [0, 0, z, 0],
                                [0, 0, 0, 1]])

class RotateX(MatrixDAO):
    def __init__(self, r):
        self.matrix = np.array([[1, 0, 0, 0],
                                [0, cos(r), -sin(r), 0],
                                [0, sin(r), cos(r), 0],
                                [0, 0, 0, 1]])      

class RotateY(MatrixDAO):
    def __init__(self, r):
        self.matrix = np.array([[cos(r), 0, sin(r), 0],
                                [0, 1, 0, 0],
                                [-sin(r), 0, cos(r), 0],
                                [0, 0, 0, 1]])    

class RotateZ(MatrixDAO):
    def __init__(self, r):
        self.matrix = np.array([[cos(r), -sin(r), 0, 0],
                                [sin(r), cos(r), 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])                  

class ShearingMatrix(MatrixDAO):
    def __init__(self, xy, xz, yx, yz, zx, zy):
        self.matrix = np.array([[1, xy, xz, 0],
                                [yx, 1, yz, 0],
                                [zx, zy, 1, 0],
                                [0, 0, 0, 1]])

class TupleDAO:
    tuple : np.array

    def __init__(self, tuple):
        self.tuple = np.array(tuple)

    def add(self, other):
        return TupleDAO(np.add(self.tuple, other.tuple)) 
    
    def subtract(self, other):
        return TupleDAO(np.subtract(self.tuple, other.tuple))

    def negate(self):
        return TupleDAO(np.negative(self.tuple))

    def __mul__(self, other):
        if isinstance(other, TupleDAO):
            return TupleDAO(np.multiply(self.tuple, other.tuple))
        elif isinstance(other, MatrixDAO):
            return TupleDAO(np.ravel((other.matrix @ self.tuple.reshape(-1,1)).reshape(1, -1))) 
        else:
            return TupleDAO(self.tuple * other)  

    def divide(self, other):
        if isinstance(other, TupleDAO):
            return TupleDAO(np.divide(self.tuple, other.tuple))
        else:
            return TupleDAO(self.tuple / other)

    def mag(self):
        return np.linalg.norm(self.tuple)                   

    def norm(self):
        return TupleDAO(self.tuple / self.mag())

    def dot(self, other):
        return np.dot(self.tuple, other.tuple)
    
    def cross(self, other):
        return TupleDAO(np.append(np.cross(self.tuple[:3], other.tuple[:3]), 0))      
    
    def __eq__(self, other):
        return np.allclose(self.tuple, other.tuple, 0.00001, 0.00001)
    
    def reflect(self, other):
        return TupleDAO(self.subtract(other * 2 * self.dot(other)).tuple)

    def translate(self, x, y, z, inverse=False):
        if inverse:
            return self * ~TranslationMatrix(x, y, z)
        
        return self * TranslationMatrix(x, y, z)
    
    def scale(self, x, y, z, inverse=False):
        if inverse:
            return self * ~ScalingMatrix(x, y, z)
        
        return self * ScalingMatrix(x, y, z)
    
    def rotate_x(self, r, inverse=False):
        if inverse:
            return self * ~RotateX(r)

        return self * RotateX(r)
    
    def rotate_y(self, r, inverse=False):
        if inverse:
            return self * ~RotateY(r)

        return self * RotateY(r)

    def rotate_z(self, r, inverse=False):
        if inverse:
            return self * ~RotateZ(r)

        return self * RotateZ(r)

    def shear(self, xy, xz, yx, yz, zx, zy):
        return self * ShearingMatrix(xy, xz, yx, yz, zx, zy)
        
class PointDAO(TupleDAO):
    def __init__(self, x, y, z):
        self.tuple = np.array([x, y, z, 1])

class VectorDAO(TupleDAO):
    def __init__(self, x, y, z):
        self.tuple = np.array([x, y, z, 0])

class ColorDAO(TupleDAO):
    def __init__(self, r, g, b):
        self.tuple = np.array([r, g, b])

class CanvasDAO:
    height : int = 500
    width : int = 500
    canvas : np.ndarray

    def __init__(self):
        self.canvas = np.zeros((self.height, self.width, 3))

    def get_pixel(self, x, y):
        return self.canvas[y][x]
    
    def set_pixel(self, x, y, pixel):
        if pixel[0] > 1:
            pixel[0] = 1
        if pixel[1] > 1: 
            pixel[1] = 1
        if pixel[2] > 1:
            pixel[2] = 1

        self.canvas[y][x] = np.array(pixel)

    def create_img(self):
        buffer = BytesIO()
        plt.imsave(buffer, self.canvas)
        buffer.seek(0)
        return buffer.getvalue()
    
class RayDAO:
    origin : PointDAO
    direction : VectorDAO

    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    def position(self, t):
        return self.origin.add(self.direction * t)
    
    def transform(self, matrix : MatrixDAO):
        return RayDAO(matrix * self.origin, matrix * self.direction)
    
class LightDAO:
    intensity : ColorDAO
    position : PointDAO

    def __init__(self, position : PointDAO, intensity : ColorDAO):
        self.position = position      
        self.intensity = intensity

class MaterialDAO:
    color : ColorDAO
    ambient : float
    diffuse : float
    specular : float
    shininess : float

    def __init__(self, color=ColorDAO(1, 1, 1), ambient=0.1, diffuse=0.9, specular=0.9, shininess=200.0):
        self.color = color
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess

    def lighting(self, light : LightDAO, point : PointDAO, eyev : VectorDAO, normalv : VectorDAO):
        effective_color = self.color * light.intensity
        lightv = light.position.subtract(point).norm()
        ambient = effective_color * self.ambient
        light_dot_normal = lightv.dot(normalv)

        if light_dot_normal >= 0:
            diffuse = effective_color * self.diffuse * light_dot_normal
            reflectv = lightv.negate().reflect(normalv)
            reflect_dot_eye = reflectv.dot(eyev)

            if reflect_dot_eye > 0:
                factor = reflect_dot_eye ** self.shininess
                specular = light.intensity * self.specular * factor
            else:
                specular = ColorDAO(0, 0, 0)

        else:
            diffuse = ColorDAO(0, 0, 0)
            specular = ColorDAO(0, 0, 0)

        return ambient.add(diffuse).add(specular)

class SphereDAO:
    center : PointDAO = PointDAO(0, 0, 0)
    transform : MatrixDAO
    material : MaterialDAO

    def __init__(self, transform = IdentityMatrix(), material = MaterialDAO()):
        self.transform = transform
        self.material = material

    def intersect(self, ray : RayDAO):
        r2 = ray.transform(~self.transform)

        sphere_to_ray = r2.origin.subtract(self.center)

        a = r2.direction.dot(r2.direction)
        b = r2.direction.dot(sphere_to_ray) * 2
        c = sphere_to_ray.dot(sphere_to_ray) - 1
        d = b**2 - 4 * a * c
        
        if d < 0:
            return []

        else:
            t1 = (-b - sqrt(d)) / (2 * a)
            t2 = (-b + sqrt(d)) / (2 * a)
            return [Intersection(t1, self), Intersection(t2, self)]
        
    def normal_at(self, world_point : PointDAO):
        object_point = ~self.transform * world_point
        object_normal = object_point.subtract(self.center)
        world_normal = ~self.transform.transpose() * object_normal
        world_normal.tuple[3] = 0
        return world_normal.norm()

class IntersectionPreCompute:
    t : float
    obj : SphereDAO
    point : PointDAO
    eyev : VectorDAO
    normalv : VectorDAO
    inside : bool

    def __init__(self, t : float, obj : SphereDAO, point : PointDAO, eyev : VectorDAO, normalv : VectorDAO, inside : bool):
        self.t = t
        self.obj = obj
        self.point = point
        self.eyev = eyev
        self.normalv = normalv
        self.inside = inside

class Intersection:
    obj : SphereDAO
    t : float

    def __init__(self, t, obj):
        self.obj = obj
        self.t = t

    def pre_compute(self, r : RayDAO):
        t = self.t
        obj = self.obj
        point = r.position(self.t)
        eyev = r.direction.negate()
        normalv = self.obj.normal_at(point)
        inside = False

        if normalv.dot(eyev) < 0:
            inside = True
            normalv = normalv.negate()

        return IntersectionPreCompute(t, obj, point, eyev, normalv, inside)

class Intersections:
    intersections : np.array([Intersection])

    def __init__(self, intersections : [Intersection]):
        self.intersections = np.array(intersections)

    def hit(self):
        return min(filter(lambda x: x.t > 0, self.intersections), key=lambda x: x.t, default=None)

class WorldDAO:
    light : LightDAO
    spheres : [SphereDAO]

    def __init__(self, 
                 light=LightDAO(PointDAO(-10, 10, -10), ColorDAO(1, 1, 1)), 
                 spheres=[SphereDAO(material = MaterialDAO(color=ColorDAO(0.8, 1.0, 0.6), diffuse=0.7, specular=0.2)), SphereDAO(IdentityMatrix().scale(0.5, 0.5, 0.5))]):
        self.light = light
        self.spheres = spheres

    def intersect_world(self, r : RayDAO):
        intersections = []

        for s in self.spheres:
            intersections.append(s.intersect(r))

        return Intersections(sorted(list(np.array(intersections).ravel()), key=lambda x: x.t))
    
    def shade_hit(self, pc : IntersectionPreCompute):
        return pc.obj.material.lighting(self.light, pc.point, pc.eyev, pc.normalv)
    
    def color_at(self, r : RayDAO):
        hit = self.intersect_world(r).hit()
        if hit:
            return self.shade_hit(hit.pre_compute(r))
        
        return ColorDAO(0, 0, 0)

class PointOfView:
    frm : PointDAO
    to : PointDAO
    up : VectorDAO

    def __init__(self, frm : VectorDAO, to, up):
        self.frm = frm
        self.to = to
        self.up = up

    def transform(self, msg=""):
        forward = self.to.subtract(self.frm).norm()
        upn = self.up.norm()
        left = forward.cross(upn)
        true_up = left.cross(forward)

        return MatrixDAO([[left.tuple[0], left.tuple[1], left.tuple[2], 0],
                            [true_up.tuple[0], true_up.tuple[1], true_up.tuple[2], 0],
                            [-forward.tuple[0], -forward.tuple[1], -forward.tuple[2], 0],
                            [0, 0, 0, 1]]
                        ).translate(-self.frm.tuple[0], -self.frm.tuple[1], -self.frm.tuple[2])
        

    


