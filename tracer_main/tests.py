#from django.test import TestCase
from unittest import TestCase
from tracer_main.daos import *

from math import sqrt, pi

class TupleTestCase(TestCase):
    def testEqualTuple(self):
        self.assertTrue(PointDAO(1.00001, 2, 3) == PointDAO(1, 2, 3))
        self.assertFalse(PointDAO(1.00001, 2, 3) == PointDAO(1.1, 2, 3))

    def testTwo(self):
        self.assertTrue(VectorDAO(3, -2, 5) + VectorDAO(-2, 3, 1) == VectorDAO(1, 1, 6))
        self.assertFalse(VectorDAO(3, -2, 5) + VectorDAO(-2, 3, 1) == VectorDAO(1.1, 1, 6))

    def testThree(self):
        self.assertTrue(PointDAO(3, 2, 1) - PointDAO(5, 6, 7) == VectorDAO(-2, -4, -6))
        self.assertFalse(PointDAO(3, 2, 1) - PointDAO(5, 6, 7) == VectorDAO(-2.1, -4, -6))

    def testFour(self):
        self.assertTrue(-VectorDAO(1, -2, 3) == VectorDAO(-1, 2, -3))
        self.assertFalse(-VectorDAO(1, -2, 3) == VectorDAO(-1.1, 2, -3))

    def testFive(self):
        self.assertTrue((VectorDAO(1, -2, 3) * 3.5) == VectorDAO(3.5, -7, 10.5))
        self.assertFalse((VectorDAO(1, -2, 3) * 3.5) == VectorDAO(3.51, -7, 10.5))

    def testSix(self):
        self.assertTrue((PointDAO(1.0, 0.2, 0.4) * PointDAO(0.9, 1.0, 0.1)) == PointDAO(0.9, 0.2, 0.04))
        self.assertFalse((PointDAO(1.0, 0.2, 0.4) * PointDAO(0.9, 1.0, 0.1)) == PointDAO(0.91, 0.2, 0.04))

    def testSeven(self):
        self.assertTrue(VectorDAO(1, -2, 3).divide(2) == VectorDAO(0.5, -1, 1.5))
        self.assertFalse(VectorDAO(1, -2, 3).divide(2) == VectorDAO(0.51, -1, 1.5))

    def testEight(self):
        self.assertTrue(VectorDAO(1, 2, 3).mag() == sqrt(14))
        self.assertFalse(VectorDAO(1, 2, 3).mag() == sqrt(15))

    def testNine(self):
        self.assertTrue(VectorDAO(1, 2, 3).norm() == VectorDAO(1/sqrt(14), 2/sqrt(14), 3/sqrt(14)))
        self.assertFalse(VectorDAO(1, 2, 3).norm() == VectorDAO(1/sqrt(14), 2/sqrt(15), 3/sqrt(14)))

    def testTen(self):
        self.assertTrue(VectorDAO(1, 2, 3).dot(VectorDAO(2, 3, 4)) == 20)
        self.assertFalse(VectorDAO(1, 2, 3).dot(VectorDAO(2, 3, 4)) == 21)

    def testEleven(self):
        self.assertTrue(VectorDAO(1, 2, 3).cross(VectorDAO(2, 3, 4)) == VectorDAO(-1, 2, -1))
        self.assertFalse(VectorDAO(1, 2, 3).cross(VectorDAO(2, 3, 4)) == VectorDAO(-1, 2, 1))

    def testReflect(self):
        self.assertTrue(VectorDAO(1, -1, 0).reflect(VectorDAO(0, 1, 0)) == VectorDAO(1, 1, 0))

        self.assertTrue(VectorDAO(0, -1, 0).reflect(VectorDAO(sqrt(2)/2, sqrt(2)/2, 0)) == VectorDAO(1, 0, 0))

class MatrixTestCase(TestCase):
    def testOne(self):
        A = MatrixDAO([[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 8, 7, 6],
                        [5, 4, 3, 2]])
            
        self.assertTrue(A.get_element(0, 0) == 1)

    def testTwo(self):
        A = MatrixDAO([[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 8, 7, 6],
                        [5, 4, 3, 2]])
        
        B = MatrixDAO([[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 8, 7, 6],
                        [5, 4, 3, 2]])

        C = MatrixDAO([[-2, 1, 2, 3],
                        [3, 2, 1, -1],
                        [4, 3, 6, 5],
                        [1, 2, 7, 8]])        

        self.assertTrue(A == B)
        self.assertFalse(A == C)     

    def testThree(self):
        A = MatrixDAO([[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 8, 7, 6],
                        [5, 4, 3, 2]])
        
        B = MatrixDAO([[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 8, 7, 6],
                        [5, 4, 3, 2]])

        C = MatrixDAO([[-2, 1, 2, 3],
                        [3, 2, 1, -1],
                        [4, 3, 6, 5],
                        [1, 2, 7, 8]])

        D = MatrixDAO([[20, 22, 50, 48],
                        [44, 54, 114, 108],
                        [40, 58, 110, 102],
                        [16, 26, 46, 42]])           
                
        self.assertTrue((A * C) == D)
        self.assertFalse((A * B) == D)

    def testFive(self):
        m1 = MatrixDAO([[1, 2, 3, 4],
                        [2, 4, 4, 2],
                        [8, 6, 4, 1],
                        [0, 0, 0, 1]])

        t1 = PointDAO(1, 2, 3)
        t2 = PointDAO(18, 24, 33)

        self.assertTrue((m1 * t1) == t2)
 
    def testSix(self):
        A = MatrixDAO([[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 8, 7, 6],
                        [5, 4, 3, 2]])
                
        self.assertTrue((A * IdentityMatrix()) == A)

    def testSeven(self):
        t1 = PointDAO(1, 2, 3)   

        self.assertTrue((IdentityMatrix() * t1) == t1)

    def testEight(self):
        A = MatrixDAO([[0, 9, 3, 0],
                        [9, 8, 0, 8],
                        [1, 8, 5, 3],
                        [0, 0, 5, 8]])
        
        B = MatrixDAO([[0, 9, 1, 0],
                        [9, 8, 8, 0],
                        [3, 0, 5, 5],
                        [0, 8, 3, 8]])
                
        self.assertTrue(A.transpose() == B)
        self.assertTrue(IdentityMatrix().transpose() == IdentityMatrix())

    def testInverse(self):
        A = MatrixDAO([[3, -9, 7, 3],
                        [3, -8, 2, -9],
                        [-4, 4, 4, 1],
                        [-6, 5, -1, 1]])
        
        B = MatrixDAO([[8, 2, 2, 2],
                        [3, -1, 7, 0],
                        [7, 0, 5, 4],
                        [6, -2, 0, 5]])  

        C = A * B

        self.assertTrue((C * ~B) == A)


    def testTranslation(self):
        self.assertTrue(PointDAO(-3, 4, 5).translate(5, -3, 2) == PointDAO(2, 1, 7))

        self.assertTrue(PointDAO(-3, 4, 5).translate(5, -3, 2, True) == PointDAO(-8, 7, 3))

        self.assertTrue(VectorDAO(-3, 4, 5).translate(5, -3, 2) == VectorDAO(-3, 4, 5))
       
    def testScaling(self):
        self.assertTrue(PointDAO(-4, 6, 8).scale(2, 3, 4) == PointDAO(-8, 18, 32))
        
        self.assertTrue(VectorDAO(-4, 6, 8).scale(2, 3, 4) == VectorDAO(-8, 18, 32))

        self.assertTrue(VectorDAO(-4, 6, 8).scale(2, 3, 4, True) == VectorDAO(-2, 2, 2))
     
    def testRotation(self):
        p1 = PointDAO(0, 1, 0)
        p2 = PointDAO(0, 0, 1)

        self.assertTrue(p1.rotate_x(pi/4) == PointDAO(0, sqrt(2)/2, sqrt(2)/2))

        self.assertTrue(p1.rotate_x(pi/2) == PointDAO(0, 0, 1))

        self.assertTrue(p1.rotate_x(pi/4, True) == PointDAO(0, sqrt(2)/2, -sqrt(2)/2))

        self.assertTrue(p2.rotate_y(pi/4) == PointDAO(sqrt(2)/2, 0, sqrt(2)/2))

        self.assertTrue(p2.rotate_y(pi/2) == PointDAO(1, 0, 0))

        self.assertTrue(p1.rotate_z(pi/4) == PointDAO(-sqrt(2)/2, sqrt(2)/2, 0))

        self.assertTrue(p1.rotate_z(pi/2) == PointDAO(-1, 0, 0))
 
    def testShearing(self):
        p3 = PointDAO(2, 3, 4)

        self.assertTrue(p3.shear(1, 0, 0, 0, 0, 0) == PointDAO(5, 3, 4))     

        self.assertTrue(p3.shear(0, 1, 0, 0, 0, 0) == PointDAO(6, 3, 4))   

        self.assertTrue(p3.shear(0, 0, 1, 0, 0, 0) == PointDAO(2, 5, 4))         

        self.assertTrue(p3.shear(0, 0, 0, 1, 0, 0) == PointDAO(2, 7, 4))         

        self.assertTrue(p3.shear(0, 0, 0, 0, 1, 0) == PointDAO(2, 3, 6))         

        self.assertTrue(p3.shear(0, 0, 0, 0, 0, 1) == PointDAO(2, 3, 7))         
    
class RayTestCase(TestCase):
    def testPosition(self):
        r = RayDAO(PointDAO(2, 3, 4), VectorDAO(1, 0, 0))

        self.assertTrue(r.position(0) == PointDAO(2, 3, 4))

        self.assertTrue(r.position(1) == PointDAO(3, 3, 4))

        self.assertTrue(r.position(-1) == PointDAO(1, 3, 4))

        self.assertTrue(r.position(2.5) == PointDAO(4.5, 3, 4))   

class SphereTestCase(TestCase):
    # These are broken now
    def testIntersect(self):
        xs1 = Intersections(SphereDAO().intersect(RayDAO(PointDAO(0, 0, -5), VectorDAO(0, 0, 1))))

        self.assertTrue(xs1.intersections[0].t == 4.0)
        self.assertTrue(xs1.intersections[1].t == 6.0)

        xs2 = Intersections(SphereDAO().intersect(RayDAO(PointDAO(0, 1, -5), VectorDAO(0, 0, 1))))

        self.assertTrue(xs2.intersections[0].t == 5.0)
        self.assertTrue(xs2.intersections[1].t == 5.0)

        xs3 = Intersections(SphereDAO().intersect(RayDAO(PointDAO(0, 2, -5), VectorDAO(0, 0, 1))))

        self.assertTrue(xs3.intersections.size == 0)             

        xs4 = Intersections(SphereDAO().intersect(RayDAO(PointDAO(0, 0, 0), VectorDAO(0, 0, 1))))

        self.assertTrue(xs4.intersections[0].t == -1)
        self.assertTrue(xs4.intersections[1].t == 1)

        xs5 = Intersections(SphereDAO().intersect(RayDAO(PointDAO(0, 0, 5), VectorDAO(0, 0, 1))))

        self.assertTrue(xs5.intersections[0].t == -6)
        self.assertTrue(xs5.intersections[1].t == -4)

    def testNormalAt(self):
        self.assertTrue(SphereDAO().normal_at(PointDAO(1, 0, 0)) == VectorDAO(1, 0, 0))

        self.assertTrue(SphereDAO().normal_at(PointDAO(0, 1, 0)) == VectorDAO(0, 1, 0))

        self.assertTrue(SphereDAO().normal_at(PointDAO(0, 0, 1)) == VectorDAO(0, 0, 1)) 

        self.assertTrue(SphereDAO().normal_at(PointDAO(sqrt(3)/3, sqrt(3)/3, sqrt(3)/3)) == VectorDAO(sqrt(3)/3, sqrt(3)/3, sqrt(3)/3))     

        self.assertTrue(SphereDAO(IdentityMatrix().translate(0, 1, 0)).normal_at(PointDAO(0, 1.70711, -0.70711)) == VectorDAO(0, 0.70711, -0.70711)) 

        self.assertTrue(SphereDAO(IdentityMatrix().scale(1, 0.5, 1).rotate_z(pi/5)).normal_at(PointDAO(0, sqrt(2)/2, -sqrt(2)/2)) == VectorDAO(0, 0.97014, -0.24254))     

class IntersectionTestCase(TestCase):
    def testIntersections(self):
        i1 = Intersection(1, SphereDAO())
        i2 = Intersection(2, SphereDAO())

        xs = Intersections([i1, i2])

        self.assertTrue(xs.intersections[0].t == 1)
        self.assertTrue(xs.intersections[1].t == 2)

    def testHit(self):
        i1_1 = Intersection(1, SphereDAO())
        i2_1 = Intersection(2, SphereDAO())
        xs1 = Intersections([i1_1, i2_1])

        self.assertTrue(xs1.hit() == i1_1)    

        i1_2 = Intersection(-1, SphereDAO())
        i2_2 = Intersection(1, SphereDAO())
        xs2 = Intersections([i1_2, i2_2])  

        self.assertTrue(xs2.hit() == i2_2)

        i1_3 = Intersection(-1, SphereDAO())
        i2_3 = Intersection(-2, SphereDAO())
        xs3 = Intersections([i1_3, i2_3])  

        self.assertTrue(xs3.hit() == None)   

        i1_4 = Intersection(5, SphereDAO())
        i2_4 = Intersection(7, SphereDAO())
        i3_4 = Intersection(-3, SphereDAO())
        i4_4 = Intersection(2, SphereDAO())
        xs4 = Intersections([i1_4, i2_4, i3_4, i4_4])  

        self.assertTrue(xs4.hit() == i4_4)                 

    def testPreCompute(self):
        pc1 = Intersection(4, SphereDAO()).pre_compute(RayDAO(PointDAO(0, 0, -5), VectorDAO(0, 0, 1)))
        self.assertTrue(pc1.t == 4)
        self.assertTrue(pc1.point == PointDAO(0, 0, -1))
        self.assertTrue(pc1.eyev == VectorDAO(0, 0, -1))
        self.assertTrue(pc1.normalv == VectorDAO(0, 0, -1))
        self.assertTrue(pc1.inside == False)

        pc2 = Intersection(1, SphereDAO()).pre_compute(RayDAO(PointDAO(0, 0, 0), VectorDAO(0, 0, 1)))
        self.assertTrue(pc2.point == PointDAO(0, 0, 1))
        self.assertTrue(pc2.eyev == VectorDAO(0, 0, -1))
        self.assertTrue(pc2.inside == True)
        self.assertTrue(pc2.normalv == VectorDAO(0, 0, -1))

class MaterialTestCase(TestCase):
    def testLighting(self):
        self.assertTrue(MaterialDAO()
                        .lighting(LightDAO(PointDAO(0, 0, -10), ColorDAO(1, 1, 1)), PointDAO(0, 0, 0), VectorDAO(0, 0, -1), VectorDAO(0, 0, -1)) == ColorDAO(1.9, 1.9, 1.9)
        )
        
        self.assertTrue(MaterialDAO()
                        .lighting(LightDAO(PointDAO(0, 0, -10), ColorDAO(1, 1, 1)), PointDAO(0, 0, 0), VectorDAO(0, sqrt(2)/2, -sqrt(2)/2), VectorDAO(0, 0, -1)) == ColorDAO(1, 1, 1)
        )

        self.assertTrue(MaterialDAO()
                        .lighting(LightDAO(PointDAO(0, 10, -10), ColorDAO(1, 1, 1)), PointDAO(0, 0, 0), VectorDAO(0, 0, -1), VectorDAO(0, 0, -1)) == ColorDAO(0.7364, 0.7364, 0.7364)
        )

        self.assertTrue(MaterialDAO()
                        .lighting(LightDAO(PointDAO(0, 10, -10), ColorDAO(1, 1, 1)), PointDAO(0, 0, 0), VectorDAO(0, -sqrt(2)/2, -sqrt(2)/2), VectorDAO(0, 0, -1)) == ColorDAO(1.6364, 1.6364, 1.6364)
        )                

        self.assertTrue(MaterialDAO()
                        .lighting(LightDAO(PointDAO(0, 0, 10), ColorDAO(1, 1, 1)), PointDAO(0, 0, 0), VectorDAO(0, 0, -1), VectorDAO(0, 0, -1)) == ColorDAO(0.1, 0.1, 0.1)
        )

class WorldTestCase(TestCase):
    def testIntersect(self):
        w = WorldDAO()
        xs = w.intersect_world(RayDAO(PointDAO(0, 0, -5), VectorDAO(0, 0, 1)))

        self.assertTrue(xs.intersections[0].t == 4)
        self.assertTrue(xs.intersections[1].t == 4.5)
        self.assertTrue(xs.intersections[2].t == 5.5)
        self.assertTrue(xs.intersections[3].t == 6)                


    def testShadeHit(self):
        shade_w1 = WorldDAO()
        shade_c1 = shade_w1.shade_hit(Intersection(4, shade_w1.spheres[0]).pre_compute(RayDAO(PointDAO(0, 0, -5), VectorDAO(0, 0, 1))))
        self.assertTrue(shade_c1 == ColorDAO(0.38066, 0.47583, 0.2855))

        w2 = WorldDAO()
        w2.light = LightDAO(PointDAO(0, 0.25, 0), ColorDAO(1, 1, 1))
        c2 = w2.shade_hit(Intersection(0.5, w2.spheres[1]).pre_compute(RayDAO(PointDAO(0, 0, 0), VectorDAO(0, 0, 1))))        
        self.assertTrue(c2 == ColorDAO(0.90498, 0.90498, 0.90498))

    def testColorAt(self):
        w1 = WorldDAO()
        c1 = w1.color_at(RayDAO(PointDAO(0, 0, -5), VectorDAO(0, 1, 0)))
        self.assertTrue(c1 == ColorDAO(0, 0, 0))


        w2 = WorldDAO()
        c2 = w2.color_at(RayDAO(PointDAO(0, 0, -5), VectorDAO(0, 0, 1)))
        self.assertTrue(c2 == ColorDAO(0.38066, 0.47583, 0.2855))

        w3 = WorldDAO(spheres=[SphereDAO(material=MaterialDAO(ambient=1)), SphereDAO(material=MaterialDAO(ambient=1))])
        c3 = w3.color_at(RayDAO(PointDAO(0, 0, 0.75), VectorDAO(0, 0, -1)))
        self.assertTrue(c3 == w3.spheres[1].material.color)

class PointOfViewTestCase(TestCase):
    def testTransform(self):
        self.assertTrue(PointOfView(PointDAO(0, 0, 0), PointDAO(0, 0, -1), VectorDAO(0, 1, 0)).transform() == IdentityMatrix())

        self.assertTrue(PointOfView(PointDAO(0, 0, 0), PointDAO(0, 0, 1), VectorDAO(0, 1, 0)).transform() == IdentityMatrix().scale(-1, 1, -1))

        self.assertTrue(PointOfView(PointDAO(0, 0, 8), PointDAO(0, 0, 0), VectorDAO(0, 1, 0)).transform() == IdentityMatrix().translate(0, 0, -8))

        result = MatrixDAO([[-0.50709, 0.50709, 0.67612, -2.36643],
                              [0.76772, 0.60609, 0.12122, -2.82843],
                              [-0.35857, 0.59761, -0.71714, 0],
                              [0, 0, 0, 1]])

        self.assertTrue(PointOfView(PointDAO(1, 3, 2), PointDAO(4, -2, 8), VectorDAO(1, 1, 0)).transform() == result)
