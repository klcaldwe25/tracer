from io import BytesIO
from .models import Canvas
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

    def equal(self, other):
        return np.allclose(self.matrix, other.matrix)    

    def multiply(self, other):       
        if isinstance(other, MatrixDAO):
            return MatrixDAO(np.matmul(self.matrix, other.matrix)) 
        elif isinstance(other, TupleDAO):
            return TupleDAO(np.ravel(np.matmul(self.matrix, other.tuple.reshape(-1,1)).reshape(1, -1))) 

    def transpose(self):
        return MatrixDAO(self.matrix.transpose())
    
    def inverse(self):
        try:    
            return MatrixDAO(np.linalg.inv(self.matrix))
        except:
            return "Not invertible"           

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

    def multiply(self, other):
        if isinstance(other, TupleDAO):
            return TupleDAO(np.multiply(self.tuple, other.tuple))
        elif isinstance(other, MatrixDAO):
            return TupleDAO(np.ravel(np.matmul(other.matrix, self.tuple.reshape(-1,1)).reshape(1, -1))) 
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
        if isinstance(other, VectorDAO):
            return TupleDAO(np.append(np.cross(self.tuple[:3], other.tuple[:3]), 0))      
    
    def equal(self, other):
        return np.allclose(self.tuple, other.tuple)

    def translate(self, x, y, z, inverse=False):
        if inverse:
            return TranslationMatrix(x, y, z).inverse().multiply(self)
        
        return TranslationMatrix(x, y, z).multiply(self)
    
    def scale(self, x, y, z, inverse=False):
        if inverse:
            return ScalingMatrix(x, y, z).inverse().multiply(self)
        
        return ScalingMatrix(x, y, z).multiply(self)
    
    def rotate_x(self, r, inverse=False):
        if inverse:
            return RotateX(r).inverse().multiply(self)

        return RotateX(r).multiply(self)
    
    def rotate_y(self, r, inverse=False):
        if inverse:
            return RotateY(r).inverse().multiply(self)

        return RotateY(r).multiply(self)

    def rotate_z(self, r, inverse=False):
        if inverse:
            return RotateZ(r).inverse().multiply(self)

        return RotateZ(r).multiply(self)

    def shear(self, xy, xz, yx, yz, zx, zy):
        return ShearingMatrix(xy, xz, yx, yz, zx, zy).multiply(self) 
        
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
    name : str
    height : int
    width : int
    canvas : np.ndarray

    def __init__(self, name, height, width):
        self.name = name
        self.height = height
        self.width = width
        self.canvas = np.zeros((self.height, self.width, 3))

    def get_pixel(self, x, y):
        return self.canvas[y][x]
    
    def set_pixel(self, x, y, pixel):
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
        return self.origin.add(self.direction.multiply(t))
    
    def transform(self, matrix : MatrixDAO):
        return RayDAO(matrix.multiply(self.origin), matrix.multiply(self.direction))
    
class SphereDAO:
    center : PointDAO
    transform : MatrixDAO

    def __init__(self):
        self.center = PointDAO(0, 0, 0)
        self.transform = IdentityMatrix()

    def set_transform(self, transform : MatrixDAO):
        self.transform = transform

    def intersect(self, ray : RayDAO):
        r2 = ray.transform(self.transform.inverse())

        sphere_to_ray = r2.origin.subtract(self.center)

        a = r2.direction.dot(r2.direction)
        b = r2.direction.dot(sphere_to_ray) * 2
        c = sphere_to_ray.dot(sphere_to_ray) - 1
        d = b**2 - 4 * a * c
        if d < 0:
            return np.array([])

        else:
            t1 = (-b - sqrt(d)) / (2 * a)
            t2 = (-b + sqrt(d)) / (2 * a)
            return Intersections([Intersection(t1, self), Intersection(t2, self)])

class Intersection:
    obj : SphereDAO
    t : float

    def __init__(self, t, obj):
        self.obj = obj
        self.t = t

class Intersections:
    intersections : np.array([Intersection])

    def __init__(self, intersections : [Intersection]):
        self.intersections = np.array(intersections)

    def hit(self):
        return min(filter(lambda x: x.t > 0, self.intersections), key=lambda x: x.t, default=None)

    


