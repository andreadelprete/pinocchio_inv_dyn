# -*- coding: utf-8 -*-
"""
Function to compute the convex hull of a set of points (using the cdd library).

Created on Fri Jul  3 17:52:35 2015

@author: adelpret
"""
import cdd
import numpy as np
import matplotlib.pyplot as plt

NUMBER_TYPE = 'float'  # 'float' or 'fraction'

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
    
def plot_convex_hull(A, b, points=None):
    X_MIN = np.min(points[:,0]);
    X_MAX = np.max(points[:,0]);
    X_MIN -= 0.1*(X_MAX-X_MIN);
    X_MAX += 0.1*(X_MAX-X_MIN);

    Y_MIN = np.min(points[:,1]);
    Y_MAX = np.max(points[:,1]);
    Y_MIN -= 0.1*(Y_MAX-Y_MIN);
    Y_MAX += 0.1*(Y_MAX-Y_MIN);
    
    f, ax = plt.subplots();
    
    ''' plot inequalities on x-y plane '''
    com_x = np.zeros(2);
    com_y = np.zeros(2);
    com   = np.zeros(2);
    for i in range(A.shape[0]):
        if(np.abs(A[i,1])>1e-5):
            com_x[0] = X_MIN;   # com x coordinate
            com_x[1] = X_MAX;   # com x coordinate
            com[0] = com_x[0];
            com[1] = 0;
            com_y[0] = (-b[i] - np.dot(A[i,:],com) )/A[i,1];
            
            com[0] = com_x[1];
            com_y[1] = (-b[i] - np.dot(A[i,:],com) )/A[i,1];
    
            ax.plot(com_x, com_y, 'k-');
        else:
            com_y[0] = Y_MIN;
            com_y[1] = Y_MAX;
            com[0] = 0;
            com[1] = com_y[0];
            com_x[0] = (-b[i] - np.dot(A[i,:],com) )/A[i,0];
    
            com[1] = com_y[1];
            com_x[1] = (-b[i] - np.dot(A[i,:],com) )/A[i,0];
    
            ax.plot(com_x, com_y, 'k-');        

    if(points!=None):
        ax.plot(points[:,0], points[:,1], 'o', markersize=30);
        
    ax.set_xlim([X_MIN, X_MAX]);
    ax.set_ylim([Y_MIN, Y_MAX]);
    
    plt.show()
    

if __name__ == "__main__":
    points = np.random.rand(30, 2)   # 30 random points in 2-D
    (A,b) = compute_convex_hull(points.T);
    plot_convex_hull(A,b,points);
    

