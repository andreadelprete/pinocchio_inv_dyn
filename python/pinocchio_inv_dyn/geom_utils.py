#from polytope_conversion_utils import *
from numpy import zeros, sqrt, array, vstack
import numpy as np
#from math import cos, sin, tan, atan, pi
import matplotlib.pyplot as plt
import cdd
import plot_utils as plut

NUMBER_TYPE = 'float'  # 'float' or 'fraction'

''' Compute the projection matrix of the cross product.
'''
def crossMatrix( v ):
    VP = np.array( [[  0,  -v[2], v[1] ],
                    [ v[2],  0,  -v[0] ],
                    [-v[1], v[0],  0   ]] );
    return VP;
    
''' Check whether v is inside a 3d cone with the specified normal direction
    and friction coefficient. 
'''
def is_vector_inside_cone(v, mu, n):
    P = np.eye(3) - np.outer(n, n);
    return (np.linalg.norm(np.dot(P,v)) - mu*np.dot(n,v)<=0.0);

    
''' Find the intersection between two lines:
        a1^T x = b1
        a2^T x = b2
'''
def find_intersection(a1, b1, a2, b2):
    x = np.zeros(2);
    den = (a1[0]*a2[1] - a2[0]*a1[1]);
    if(abs(den)<1e-6):
        print "ERROR: Impossible to find intersection between two lines that are parallel";
        return x;
        
    if(np.abs(a1[0])>np.abs(a2[0])):
        x[1] = (-a2[0]*b1 + a1[0]*b2)/den;
        x[0] = (b1-a1[1]*x[1])/a1[0];
    else:
        x[1] = (-a2[0]*b1 + a1[0]*b2)/den;
        x[0] = (b2-a2[1]*x[1])/a2[0];        
    return x;
    
''' Find the line passing through two points:
        a^T x1 + b = 0
        a^T x2 + b = 0
'''
def find_line(x1, x2):
    den = (x1[0]*x2[1] - x2[0]*x1[1]);
    if(abs(den)<1e-4):
#        print "ERROR: x1 and x2 are too close, x1=(%f,%f), x2=(%f,%f)" % (x1[0],x1[1],x2[0],x2[1]);
        return (zeros(2),-1);
#    a = np.array([-(x1[1] - x2[1])/den, -(x2[0] - x1[0])/den]);
#    a_norm = np.linalg.norm(a);
#    a /= a_norm;
#    b = -1.0/a_norm;

    a = np.array([x2[1]-x1[1], x1[0]-x2[0]]);
    a /= np.linalg.norm(a);
    b = -a[0]*x1[0] - a[1]*x1[1];
#    print "a=(%f,%f), a2=(%f,%f), b=%f, b2=%f" % (a[0],a[1],a2[0],a2[1],b,b2);
    return (a,b);

    
''' Compute the area of a 2d triangle with vertices a, b and c. 
'''
def compute_triangle_area(a, b, c):
    la = np.linalg.norm(a-b);
    lb = np.linalg.norm(b-c);
    lc = np.linalg.norm(c-a);
    s = 0.5*(la+lb+lc);
    return sqrt(s*(s-la)*(s-lb)*(s-lc));

    
''' Plot inequalities A*x<=b on x-y plane.
'''
def plot_inequalities(A, b, x_bounds, y_bounds, ls='--', color='k', ax=None, lw=8):
    if(A.shape[1]!=2):
        print "[ERROR in plot_inequalities] matrix does not have 2 columns";
        return;
#    if(A.shape[0]!=len(b)):
#        print "[ERROR in plot_inequalities] matrix and vector does not have the same number of rows";
#        return;

    if(ax==None):
        f, ax = plut.create_empty_figure();
    p = np.zeros(2);     # p height
    p_x = np.zeros(2);
    p_y = np.zeros(2);
    for i in range(A.shape[0]):
        if(np.abs(A[i,1])>1e-13):
            p_x[0] = x_bounds[0];   # p x coordinate
            p_x[1] = x_bounds[1];   # p x coordinate
            p[0] = p_x[0];
            p[1] = 0;
            p_y[0] = (b[i] - np.dot(A[i,:],p) )/A[i,1];
            
            p[0] = p_x[1];
            p_y[1] = (b[i] - np.dot(A[i,:],p) )/A[i,1];
    
            ax.plot(p_x, p_y, ls=ls, color=color, linewidth=lw);
        elif(np.abs(A[i,0])>1e-13):
            p_y[0] = y_bounds[0];
            p_y[1] = y_bounds[1];
            p[0] = 0;
            p[1] = p_y[0];
            p_x[0] = (b[i] - np.dot(A[i,:],p) )/A[i,0];
    
            p[1] = p_y[1];
            p_x[1] = (b[i] - np.dot(A[i,:],p) )/A[i,0];
            ax.plot(p_x, p_y, ls=ls, color=color, linewidth=lw);
        else:
            pass;
#            print "[WARNING] Could not print one inequality as all coefficients are 0: A[%d,:]=[%f,%f]" % (i,A[i,0],A[i,1]);
    return ax;

''' Plot the polytope A*x<=b with vectices V '''
def plot_polytope(A, b, V=None, color='b', ax=None, plotLines=True, lw=4):
    if(ax==None):
        f, ax = plut.create_empty_figure();
    
    if(plotLines):
        plot_inequalities(A, b, [-1,1], [-1,1], color=color, ls='--', ax=ax, lw=lw);
    n = b.shape[0];    
    if(n<2):
        return (ax,None);
        
    if(V==None):
        V = np.zeros((n,2));
        for i in range(n):
            V[i,:] = find_intersection(A[i,:], b[i], A[(i+1)%n,:], b[(i+1)%n]);
                                       
    xx = np.zeros(2);
    yy = np.zeros(2);
    for i in range(n):
        xx[0] = V[i,0];
        xx[1] = V[(i+1)%n,0];
        yy[0] = V[i,1];
        yy[1] = V[(i+1)%n,1];
        line, = ax.plot(xx, yy, color=color, ls='-', lw=2*lw);
    
    return (ax, line);
    
def compute_convex_hull(S):
    """
    Returns the matrix A and the vector b such that:
        {x = S z, sum z = 1, z>=0} if and only if {A x + b >= 0}.
    """
    V = np.hstack([np.ones((S.shape[1], 1)), S.T])
    # V-representation: first column is 0 for rays, 1 for vertices
    V_cdd = cdd.Matrix(V, number_type=NUMBER_TYPE)
    V_cdd.rep_type = cdd.RepType.GENERATOR
    P = cdd.Polyhedron(V_cdd)
    H = np.array(P.get_inequalities())
    b, A = H[:, 0], H[:, 1:]
    return (A,b)