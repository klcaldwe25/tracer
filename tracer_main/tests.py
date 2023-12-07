#from django.test import TestCase
from unittest import TestCase
from tracer_main.daos import *
from math import sqrt, pi

class TupleTestCase(TestCase):
    def testEqualTuple(self):
        self.assertTrue(PointDAO(1.00001, 2, 3).equal(PointDAO(1, 2, 3)))
        self.assertFalse(PointDAO(1.00001, 2, 3).equal(PointDAO(1.1, 2, 3)))

    def testTwo(self):
        self.assertTrue(VectorDAO(3, -2, 5).add(VectorDAO(-2, 3, 1)).equal(VectorDAO(1, 1, 6)))
        self.assertFalse(VectorDAO(3, -2, 5).add(VectorDAO(-2, 3, 1)).equal(VectorDAO(1.1, 1, 6)))

    def testThree(self):
        self.assertTrue(PointDAO(3, 2, 1).subtract(PointDAO(5, 6, 7)).equal(VectorDAO(-2, -4, -6)))
        self.assertFalse(PointDAO(3, 2, 1).subtract(PointDAO(5, 6, 7)).equal(VectorDAO(-2.1, -4, -6)))

    def testFour(self):
        self.assertTrue(VectorDAO(1, -2, 3).negate().equal(VectorDAO(-1, 2, -3)))
        self.assertFalse(VectorDAO(1, -2, 3).negate().equal(VectorDAO(-1.1, 2, -3)))

    def testFive(self):
        self.assertTrue(VectorDAO(1, -2, 3).multiply(3.5).equal(VectorDAO(3.5, -7, 10.5)))
        self.assertFalse(VectorDAO(1, -2, 3).multiply(3.5).equal(VectorDAO(3.51, -7, 10.5)))

    def testSix(self):
        self.assertTrue(PointDAO(1.0, 0.2, 0.4).multiply(PointDAO(0.9, 1.0, 0.1)).equal(PointDAO(0.9, 0.2, 0.04)))
        self.assertFalse(PointDAO(1.0, 0.2, 0.4).multiply(PointDAO(0.9, 1.0, 0.1)).equal(PointDAO(0.91, 0.2, 0.04)))

    def testSeven(self):
        self.assertTrue(VectorDAO(1, -2, 3).divide(2).equal(VectorDAO(0.5, -1, 1.5)))
        self.assertFalse(VectorDAO(1, -2, 3).divide(2).equal(VectorDAO(0.51, -1, 1.5)))

    def testEight(self):
        self.assertTrue(VectorDAO(1, 2, 3).mag() == sqrt(14))
        self.assertFalse(VectorDAO(1, 2, 3).mag() == sqrt(15))

    def testNine(self):
        self.assertTrue(VectorDAO(1, 2, 3).norm().equal(VectorDAO(1/sqrt(14), 2/sqrt(14), 3/sqrt(14))))
        self.assertFalse(VectorDAO(1, 2, 3).norm().equal(VectorDAO(1/sqrt(14), 2/sqrt(15), 3/sqrt(14))))

    def testTen(self):
        self.assertTrue(VectorDAO(1, 2, 3).dot(VectorDAO(2, 3, 4)) == 20)
        self.assertFalse(VectorDAO(1, 2, 3).dot(VectorDAO(2, 3, 4)) == 21)

    def testEleven(self):
        self.assertTrue(VectorDAO(1, 2, 3).cross(VectorDAO(2, 3, 4)).equal(VectorDAO(-1, 2, -1)))
        self.assertFalse(VectorDAO(1, 2, 3).cross(VectorDAO(2, 3, 4)).equal(VectorDAO(-1, 2, 1)))

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

        self.assertTrue(A.equal(B))
        self.assertFalse(A.equal(C))      

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
                
        self.assertTrue(A.multiply(C).equal(D))
        self.assertFalse(A.multiply(B).equal(D))

    def testFive(self):
        m1 = MatrixDAO([[1, 2, 3, 4],
                        [2, 4, 4, 2],
                        [8, 6, 4, 1],
                        [0, 0, 0, 1]])

        t1 = PointDAO(1, 2, 3)
        t2 = PointDAO(18, 24, 33)

        self.assertTrue(m1.multiply(t1).equal(t2))  
 
    def testSix(self):
        A = MatrixDAO([[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 8, 7, 6],
                        [5, 4, 3, 2]])
                
        self.assertTrue(A.multiply(IdentityMatrix()).equal(A))

    def testSeven(self):
        t1 = PointDAO(1, 2, 3)   

        self.assertTrue(IdentityMatrix().multiply(t1).equal(t1))  

    def testEight(self):
        A = MatrixDAO([[0, 9, 3, 0],
                        [9, 8, 0, 8],
                        [1, 8, 5, 3],
                        [0, 0, 5, 8]])
        
        B = MatrixDAO([[0, 9, 1, 0],
                        [9, 8, 8, 0],
                        [3, 0, 5, 5],
                        [0, 8, 3, 8]])
                
        self.assertTrue(A.transpose().equal(B))
        self.assertTrue(IdentityMatrix().transpose().equal(IdentityMatrix()))

    def testInverse(self):
        A = MatrixDAO([[3, -9, 7, 3],
                        [3, -8, 2, -9],
                        [-4, 4, 4, 1],
                        [-6, 5, -1, 1]])
        
        B = MatrixDAO([[8, 2, 2, 2],
                        [3, -1, 7, 0],
                        [7, 0, 5, 4],
                        [6, -2, 0, 5]])  

        C = A.multiply(B)

        self.assertTrue(C.multiply(B.inverse()).equal(A))


    def testTranslation(self):
        self.assertTrue(PointDAO(-3, 4, 5).translate(5, -3, 2).equal(PointDAO(2, 1, 7)))

        self.assertTrue(PointDAO(-3, 4, 5).translate(5, -3, 2, True).equal(PointDAO(-8, 7, 3)))

        self.assertTrue(VectorDAO(-3, 4, 5).translate(5, -3, 2).equal(VectorDAO(-3, 4, 5)))
       
    def testScaling(self):
        self.assertTrue(PointDAO(-4, 6, 8).scale(2, 3, 4).equal(PointDAO(-8, 18, 32)))
        
        self.assertTrue(VectorDAO(-4, 6, 8).scale(2, 3, 4).equal(VectorDAO(-8, 18, 32)))

        self.assertTrue(VectorDAO(-4, 6, 8).scale(2, 3, 4, True).equal(VectorDAO(-2, 2, 2)))
     
    def testRotation(self):
        p1 = PointDAO(0, 1, 0)
        p2 = PointDAO(0, 0, 1)

        self.assertTrue(p1.rotate_x(pi/4).equal(PointDAO(0, sqrt(2)/2, sqrt(2)/2)))

        self.assertTrue(p1.rotate_x(pi/2).equal(PointDAO(0, 0, 1)))

        self.assertTrue(p1.rotate_x(pi/4, True).equal(PointDAO(0, sqrt(2)/2, -sqrt(2)/2)))

        self.assertTrue(p2.rotate_y(pi/4).equal(PointDAO(sqrt(2)/2, 0, sqrt(2)/2)))

        self.assertTrue(p2.rotate_y(pi/2).equal(PointDAO(1, 0, 0)))

        self.assertTrue(p1.rotate_z(pi/4).equal(PointDAO(-sqrt(2)/2, sqrt(2)/2, 0)))

        self.assertTrue(p1.rotate_z(pi/2).equal(PointDAO(-1, 0, 0)))
 
    def testShearing(self):
        p3 = PointDAO(2, 3, 4)

        self.assertTrue(p3.shear(1, 0, 0, 0, 0, 0).equal(PointDAO(5, 3, 4)))      

        self.assertTrue(p3.shear(0, 1, 0, 0, 0, 0).equal(PointDAO(6, 3, 4)))     

        self.assertTrue(p3.shear(0, 0, 1, 0, 0, 0).equal(PointDAO(2, 5, 4)))         

        self.assertTrue(p3.shear(0, 0, 0, 1, 0, 0).equal(PointDAO(2, 7, 4)))         

        self.assertTrue(p3.shear(0, 0, 0, 0, 1, 0).equal(PointDAO(2, 3, 6)))         

        self.assertTrue(p3.shear(0, 0, 0, 0, 0, 1).equal(PointDAO(2, 3, 7)))         
    
class RayTestCase(TestCase):
    def testPosition(self):
        r = RayDAO(PointDAO(2, 3, 4), VectorDAO(1, 0, 0))

        self.assertTrue(r.position(0).equal(PointDAO(2, 3, 4)))

        self.assertTrue(r.position(1).equal(PointDAO(3, 3, 4)))

        self.assertTrue(r.position(-1).equal(PointDAO(1, 3, 4)))  

        self.assertTrue(r.position(2.5).equal(PointDAO(4.5, 3, 4)))   

class SphereTestCase(TestCase):
    def testIntersect(self):
        r1 = RayDAO(PointDAO(0, 0, -5), VectorDAO(0, 0, 1))
        s1 = SphereDAO()     
        xs1 = s1.intersect(r1)

        self.assertTrue(xs1[0] == 4.0)
        self.assertTrue(xs1[1] == 6.0)

        r2 = RayDAO(PointDAO(0, 1, -5), VectorDAO(0, 0, 1))
        s2 = SphereDAO()
        xs2 = s2.intersect(r2)

        self.assertTrue(xs2[0] == 5.0)
        self.assertTrue(xs2[1] == 5.0)

        r3 = RayDAO(PointDAO(0, 2, -5), VectorDAO(0, 0, 1))
        s3 = SphereDAO()
        xs3 = s3.intersect(r3)

        self.assertTrue(xs3.size == 0)             

        r4 = RayDAO(PointDAO(0, 0, 0), VectorDAO(0, 0, 1))
        s4 = SphereDAO()
        xs4 = s4.intersect(r4)

        self.assertTrue(xs4[0] == -1)
        self.assertTrue(xs4[1] == 1)

        r5 = RayDAO(PointDAO(0, 0, 5), VectorDAO(0, 0, 1))
        s5 = SphereDAO()
        xs5 = s5.intersect(r5)

        self.assertTrue(xs5[0] == -6)
        self.assertTrue(xs5[1] == -4)

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
