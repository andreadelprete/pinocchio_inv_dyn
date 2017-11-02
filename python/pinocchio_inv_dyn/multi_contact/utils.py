# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 16:54:39 2016

@author: adelpret
"""

from pinocchio_inv_dyn.polytope_conversion_utils import crossMatrix, cone_span_to_face, eliminate_redundant_inequalities
from pinocchio_inv_dyn.geom_utils import is_vector_inside_cone, plot_inequalities
from pinocchio_inv_dyn.transformations import euler_matrix

import pinocchio as se3
import pinocchio_inv_dyn.plot_utils as plut
import matplotlib.pyplot as plt
from math import atan, pi

import numpy as np
from numpy.linalg import norm
from numpy.random import uniform
from numpy import sqrt
import cProfile

EPS = 1e-5;

def compute_centroidal_cone_generators(contact_points, contact_normals, mu, contact_tangents=None):
    ''' Compute two matrices. The first one contains the 6d generators of the
        centroidal cone. The second one contains the 3d generators of the 
        contact cones (4 generators for each contact point).
    '''
    assert contact_points.shape[1]==3, "Wrong size of contact_points"
    assert contact_normals.shape[1]==3, "Wrong size of contact_normals"
    assert contact_points.shape[0]==contact_normals.shape[0], "Size of contact_points and contact_normals do not match"
    contact_points = np.asarray(contact_points);
    contact_normals = np.asarray(contact_normals);
    if(contact_tangents is not None):
        assert contact_tangents.shape[1]==3, "Wrong size of contact_tangents"
        assert contact_points.shape[0]==contact_tangents.shape[0], "Size of contact_points and contact_tangents do not match"
        contact_tangents = np.asarray(contact_tangents);
    
    nContacts = contact_points.shape[0];
    cg = 4;                         # number of generators for each friction cone
    nGen = nContacts*cg;            # total number of generators
    if(isinstance(mu, (list, tuple, np.ndarray))):
        mu = np.asarray(mu).squeeze();
    else:
        mu = mu*np.ones(nContacts);
    G = np.zeros((3,nGen));
    A = np.zeros((6,nGen));
    P = np.zeros((6,3));
    P[:3,:] = -np.identity(3);
    muu = mu/sqrt(2.0);
    for i in range(nContacts):
        assert norm(contact_normals[i,:])!=0.0, "Length of contact normals cannot be zero"
        contact_normals[i,:]  = contact_normals[i,:]/norm(contact_normals[i,:]);

        # compute tangent directions
        if(contact_tangents is None):
            T1 = np.cross(contact_normals[i,:], [1.,0.,0.]);
            if(norm(T1)<EPS):
                T1 = np.cross(contact_normals[i,:], [0.,1.,0.]);
        else:
            assert norm(contact_tangents[i,:])!=0.0, "Length of contact tangents cannot be zero"
            T1 = contact_tangents[i,:];
        T1 = T1/norm(T1);
        T2 = np.cross(contact_normals[i,:], T1);
        
        G[:,cg*i+0] =  muu[i]*T1 + muu[i]*T2 + contact_normals[i,:];
        G[:,cg*i+1] =  muu[i]*T1 - muu[i]*T2 + contact_normals[i,:];
        G[:,cg*i+2] = -muu[i]*T1 + muu[i]*T2 + contact_normals[i,:];
        G[:,cg*i+3] = -muu[i]*T1 - muu[i]*T2 + contact_normals[i,:];
        # compute matrix mapping contact forces to gravito-inertial wrench
        P[3:,:] = -crossMatrix(contact_points[i,:]);
        # project generators in 6d centroidal space
        A[:,cg*i:cg*i+cg] = np.dot(P, G[:,cg*i:cg*i+cg]);

    # normalize generators
    for i in range(nGen):
        G[:,i] /= norm(G[:,i]);
        A[:,i] /= norm(A[:,i]);
    
    return (A, G);


def check_static_eq(H, h, mass, c0, g_vector):
    w = np.zeros(6);
    w[2] = -mass*9.81;
    w[3:] = mass*np.cross(c0, g_vector);
    return np.max(np.dot(H, w) - h) <= 1e-12;

def find_static_equilibrium_com(mass, com_lb, com_ub, H, h, MAX_ITER=1000):
    ''' Find a position of the center of mass that is in static equilibrium.'''
    FOUND_STATIC_COM = False;
    g_vector = np.array([0,0,-9.81]);
    w = np.zeros(6);
    w[2] = -mass*9.81;
    i = 0;
    while(not FOUND_STATIC_COM):
        #STEVE: consider actual bounds on contact points
        c0 = np.array([np.random.uniform(com_lb[j], com_ub[j]) for j in range(3)]);
        w[3:] = mass*np.cross(c0, g_vector);
        FOUND_STATIC_COM = np.max(np.dot(H, w) - h) <= 1e-12;
        i += 1;
        if(i>=MAX_ITER):
#            print "ERROR: Could not find com position in static equilibrium in %d iterations."%MAX_ITER;
            return (False,c0);
    return (True,c0);


def generate_rectangle_contacts(lx, ly, pos, rpy):
    ''' Generate the 4 contact points and the associated normal directions
        for a rectangular contact.
    '''    
    # contact points in local frame
    p = np.array([[ lx,  ly, 0],
               [ lx, -ly, 0],
               [-lx, -ly, 0],
               [-lx,  ly, 0]]);
    # normal direction in local frame
    n  = np.array([0, 0, 1]);
    # compute rotation matrix
    R = euler_matrix(rpy[0], rpy[1], rpy[2], 'sxyz');
    R = R[:3,:3];
    # contact points in world frame
    p[0,:] = pos.reshape(3) + np.dot(R,p[0,:]);
    p[1,:] = pos.reshape(3) + np.dot(R,p[1,:]);
    p[2,:] = pos.reshape(3) + np.dot(R,p[2,:]);
    p[3,:] = pos.reshape(3) + np.dot(R,p[3,:]);
    # normal directions in world frame
    n  = np.dot(R,n);
    N = np.vstack([n, n, n, n]);
    return (p,N);



def generate_contacts(N_CONTACTS, lx, ly, mu, CONTACT_POINT_LOWER_BOUNDS, CONTACT_POINT_UPPER_BOUNDS, 
                      RPY_LOWER_BOUNDS, RPY_UPPER_BOUNDS, MIN_CONTACT_DISTANCE, GENERATE_QUASI_FLAT_CONTACTS=False):
    ''' Generate the contact points and the contact normals associated to randomly 
        generated rectangular contact surfaces.
    '''
    contact_pos = np.zeros((N_CONTACTS, 3));
    contact_rpy = np.zeros((N_CONTACTS, 3));
    p = np.zeros((4*N_CONTACTS,3)); # contact points
    N = np.zeros((4*N_CONTACTS,3)); # contact normals
    g_vector = np.array([0, 0, -9.81]);
    
    # Generate contact positions and orientations 
    for i in range(N_CONTACTS):
        while True:
            contact_pos[i,:] = uniform(CONTACT_POINT_LOWER_BOUNDS, CONTACT_POINT_UPPER_BOUNDS);      # contact position
            collision = False;
            for j in range(i-1):
                if(np.linalg.norm(contact_pos[i,:]-contact_pos[j,:])<MIN_CONTACT_DISTANCE):
                    collision = True;
            if(not collision):
                break;
        
        while True:
            contact_rpy[i,:] = uniform(RPY_LOWER_BOUNDS, RPY_UPPER_BOUNDS);      # contact orientation
            (p[i*4:i*4+4,:],N[i*4:i*4+4,:]) = generate_rectangle_contacts(lx, ly, contact_pos[i,:], contact_rpy[i,:].T);
            if(GENERATE_QUASI_FLAT_CONTACTS==False or is_vector_inside_cone(-g_vector, mu, N[i*4,:].T)):
                break;
    return (p, N);



def compute_GIWC(contact_points, contact_normals, mu, contact_tangents=None, eliminate_redundancies=False):
    ''' Compute the gravito-inertial wrench cone (i.e. the centroidal cone).
        @param contact_points Nx3 matrix containing the contact points
        @param contact_normals Nx3 matrix containing the contact normals
        @param mu A scalar friction coefficient, or an array of friction coefficients (one for each contact point)
        @param contact_tangents Nx3 matrix containing the contact tangents
        @return (H,h) Matrix and vector defining the GIWC as H*w <= h
        @note If the first tangent direction is specified through the optional parameter contact_tangents,
              the second tangent direction is computed as the cross product between normal and first tangent
              direction. The reason for specifying the contact_tangents is to get a specific linearization
              of the friction cones.
    '''
    (S_centr, S) = compute_centroidal_cone_generators(contact_points, contact_normals, mu, contact_tangents);
    # convert generators to inequalities
    H = cone_span_to_face(S_centr,eliminate_redundancies);
    h = np.zeros(H.shape[0]);

    # normalize inequalities
    for i in range(H.shape[0]):
        norm_Hi = norm(H[i,:])
        if(norm_Hi>1e-9):
            H[i,:] /= norm_Hi;
    return (H,h);

    
def compute_support_polygon(H, h, mass, g_vector, eliminate_redundancies=False):
    ''' Compute the 2d support polygon A*c<=b given the gravito-inertial wrench 
        cone (GIWC) as H*w <= h.
        Project inequalities from 6d to 2d x-y com space.
        The com wrench to maintain static equilibrium is an affine function
        of the com position c:
            w = D*c+d
        Substituting this into the GIWC inequalities we get:
            H*w <= h
            H*D*c + H*d <= h
            H*D*c <= h - H*d
    '''
    D = np.zeros((6,3));
    d = np.zeros(6);
    D[3:,:] = -mass*crossMatrix(g_vector);
    d[:3]   = mass*g_vector;
    A = np.dot(H, D[:,:2]);
    b = h - np.dot(H, d);
    if(eliminate_redundancies):
        (A,b) = eliminate_redundant_inequalities(A,b);
    return (A,b);
    
    
def compute_com_acceleration_polytope(com_pos, H, h, mass, g_vector, eliminate_redundancies=False):
    ''' Compute the inequalities A*x<=b defining the polytope of feasible CoM accelerations
        assuming zero rate of change of angular momentum.
        @param c0 Current com position
        @param H Matrix of GIWC, which can be computed by compute_GIWC
        @param h Vector of GIWC, which can be computed by compute_GIWC
        @param mass Mass of the system in Kg
        @param g_vector Gravity vector
        @return (A,b)
        The com wrench to generate a com acceleration ddc is an affine function
        of ddc:
            w = K*ddc+d
    '''
    K = np.zeros((6,3));
    K[:3,:] = mass*np.identity(3);
    K[3:,:] = mass*crossMatrix(com_pos);
    b = h - np.dot(H, np.dot(K, g_vector)); #constant term of F
    A = np.dot(-H,K); #matrix multiplying com acceleration 
    if(eliminate_redundancies):
        (A,b) = eliminate_redundant_inequalities(A,b);
        
    # normalize inequalities
    for i in range(A.shape[0]):
        norm_Ai = norm(A[i,:])
        if(norm_Ai>EPS):
            A[i,:] /= norm_Ai;
            b[i]   /= norm_Ai;
    return A,b;
    

def can_I_stop(c0, dc0, contact_points, contact_normals, mu, mass, T_0, MAX_ITER=1000, DO_PLOTS=False, verb=0, 
               eliminate_redundancies=False):
    ''' Determine whether the system can come to a stop without changing contacts.
        Keyword arguments:
        c0 -- initial CoM position 
        dc0 -- initial CoM velocity 
        contact points -- a matrix containing the contact points
        contact normals -- a matrix containing the contact normals
        mu -- friction coefficient (either a scalar or an array)
        mass -- the robot mass
        T_0 -- an initial guess for the time to stop
        Output: (is_stable, c_final, dc_final), where:
        is_stable -- boolean value
        c_final -- final com position
        dc_final -- final com velocity
    '''
#        Steps:
#            - Compute GIWC: H*w <= h, where w=(m*(g-ddc), m*cx(g-ddc))
#            - Project GIWC in (alpha,DDalpha) space, where c=c0+alpha*v, ddc=-DDalpha*v, v=dc0/||dc0||: A*(alpha,DDalpha)<=b
#            - Find ordered (left-most first, clockwise) vertices of 2d polytope A*(alpha,DDalpha)<=b: V
#            - Initialize: alpha=0, Dalpha=||dc0||
#            - LOOP:
#                - Find current active inequality: a*alpha + b*DDalpha <= d (i.e. DDalpha upper bound for current alpha value)
#                - If DDalpha_upper_bound<0: return False
#                - Find alpha_max (i.e. value of alpha corresponding to right vertex of active inequality)
#                - Initialize: t=T_0, t_ub=10, t_lb=0
#                LOOP:
#                    - Integrate LDS until t: DDalpha = d/b - (a/d)*alpha
#                    - if(Dalpha(t)==0 && alpha(t)<=alpha_max):  return (True, alpha(t)*v)
#                    - if(alpha(t)==alpha_max && Dalpha(t)>0):   alpha=alpha(t), Dalpha=Dalpha(t), break
#                    - if(alpha(t)<alpha_max && Dalpha>0):       t_lb=t, t=(t_ub+t_lb)/2
#                    - else                                      t_ub=t, t=(t_ub+t_lb)/2
    assert mass>0.0, "Mass is not positive"
    assert T_0>0.0, "Time is not positive"
    assert mu>0.0, "Friction coefficient is not positive"
    c0 = np.asarray(c0).squeeze();
    dc0 = np.asarray(dc0).squeeze();
    contact_points = np.asarray(contact_points);
    contact_normals = np.asarray(contact_normals);
    assert c0.shape[0]==3, "Com position has not size 3"
    assert dc0.shape[0]==3, "Com velocity has not size 3"
    assert contact_points.shape[1]==3, "Contact points have not size 3"
    assert contact_normals.shape[1]==3, "Contact normals have not size 3"
    assert contact_points.shape[0]==contact_normals.shape[0], "Number of contact points and contact normals do not match"
    
    g_vector = np.array([0,0,-9.81]);
    (H,h) = compute_GIWC(contact_points, contact_normals, mu, eliminate_redundancies=eliminate_redundancies);
    
    # If initial com velocity is zero then test static equilibrium
    if(np.linalg.norm(dc0) < EPS):
        w = np.zeros(6);
        w[2] = -mass*9.81;
        w[3:] = mass*np.cross(c0, g_vector);
        if(np.max(np.dot(H, w) - h) < EPS):
            return (True, c0, dc0);
        return (False, c0, dc0);

    # Project GIWC in (alpha,DDalpha) space, where c=c0+alpha*v, ddc=DDalpha*v, v=dc0/||dc0||: a*alpha + b*DDalpha <= d
    v = dc0/np.linalg.norm(dc0);
    K = np.zeros((6,3));
    K[:3,:] = mass*np.identity(3);
    K[3:,:] = mass*crossMatrix(c0);
    d = h - np.dot(H, np.dot(K,g_vector));
    b = np.dot(np.dot(-H,K),v);             # vector multiplying com acceleration
    tmp = np.array([[0,0,0,0,1,0],          #temp times the variation dc will result in 
                    [0,0,0,-1,0,0],         # [ 0 0 0 dc_y -dc_x 0]^T
                    [0,0,0,0,0,0]]).T;   
    a = mass*9.81*np.dot(np.dot(H,tmp),v);
    
    if(DO_PLOTS):
        range_plot = 10;
        ax = plot_inequalities(np.vstack([a,b]).T, d, [-range_plot,range_plot], [-range_plot,range_plot]);
        plt.axis([0,range_plot,-range_plot,0])
        plt.title('Feasible com pos-acc');
        plut.movePlotSpines(ax, [0, 0]);
        ax.set_xlabel('com pos');
        ax.set_ylabel('com acc');
        plt.show();
        
    # Eliminate redundant inequalities
    if(eliminate_redundancies):
        A_red, d = eliminate_redundant_inequalities(np.vstack([a,b]).T, d);
        a = A_red[:,0];
        b = A_red[:,1];
    
    # Normalize inequalities to have unitary coefficients for DDalpha: b*DDalpha <= d - a*alpha
    for i in range(a.shape[0]):
        if(abs(b[i]) > EPS):
            a[i] /= abs(b[i]);
            d[i] /= abs(b[i]);
            b[i] /= abs(b[i]);
        elif(verb>0):
            print "WARNING: cannot normalize %d-th inequality because coefficient of DDalpha is almost zero"%i, b[i];    
    
    # Initialize: alpha=0, Dalpha=||dc0||
    alpha = 0;
    Dalpha = np.linalg.norm(dc0);

    #sort b indices to only keep negative values
    negative_ids = np.where(b<0)[0];
    if(negative_ids.shape[0]==0):
        # CoM acceleration is unbounded
        return (True, c0, 0.0*v);
    
    for iiii in range(MAX_ITER):
        # Find current active inequality: b*DDalpha <= d - a*alpha (i.e. DDalpha lower bound for current alpha value) 
        a_alpha_d = a*alpha-d;
        a_alpha_d_negative_bs = a_alpha_d[negative_ids];
        (i_DDalpha_min, DDalpha_min) = [(i,a_min) for (i, a_min) in [(j, a_alpha_d[j]) for j in negative_ids] if (a_min >= a_alpha_d_negative_bs).all()][0];
            
        # If DDalpha_lower_bound>0: return False 
        if(DDalpha_min >= -EPS):
            if(verb>0):
                print "Algorithm converged because DDalpha_min is positive", DDalpha_min;
            return (False, c0+alpha*v, Dalpha*v);
        
        # Find alpha_max (i.e. value of alpha corresponding to right vertex of active inequality)
        den = b*a[i_DDalpha_min] + a;
        i_pos = np.where(den>0)[0];
        if(i_pos.shape[0]==0):
#            print "WARNING b*a_i0+a is never positive, that means that alpha_max is unbounded";
            alpha_max = 10.0;
        else:
            alpha_max = np.min((d[i_pos] + b[i_pos]*d[i_DDalpha_min])/den[i_pos]);
        
        if(verb>0):
            print "[can_I_stop] DDalpha_min=%.3f, alpha=%.3f, Dalpha=%.3f, alpha_max=%.3f, a=%.3f" % (DDalpha_min, alpha, Dalpha, alpha_max, - d[i_DDalpha_min]);
            
        if(alpha_max<alpha):
            # We reach the right limit of the polytope of feasible com pos-acc.
            # This means there is no feasible com acc for farther com position (with zero angular momentum derivative)
            return (False, c0+alpha*v, Dalpha*v);
            
        # If DDalpha is not always negative on the current segment then update alpha_max to the point 
        # where the current segment intersects the x axis 
        if( a[i_DDalpha_min]*alpha_max - d[i_DDalpha_min] > 0.0):
            alpha_max = d[i_DDalpha_min] / a[i_DDalpha_min];
            if(verb>0):
                print "Updated alpha_max", alpha_max;
        
        # Initialize: t=T_0, t_ub=10, t_lb=0
        t = T_0;
        t_ub = 10;
        t_lb = 0;
        if(abs(a[i_DDalpha_min])>EPS):
            omega = sqrt(a[i_DDalpha_min]+0j);
            if(verb>0):
                print "omega", omega;
        else:
            # if the acceleration is constant over time I can compute directly the time needed
            # to bring the velocity to zero:
            #     Dalpha(t) = Dalpha(0) + t*DDalpha = 0
            #     t = -Dalpha(0)/DDalpha
            t = Dalpha / d[i_DDalpha_min];
            alpha_t  = alpha + t*Dalpha - 0.5*t*t*d[i_DDalpha_min];
            if(alpha_t <= alpha_max+EPS):
                if(verb>0):
                    print "DDalpha_min is independent from alpha, algorithm converged to Dalpha=0";
                return (True, c0+alpha_t*v, 0.0*v);
            # if alpha reaches alpha_max before the velocity is zero, then compute the time needed to reach alpha_max
            #     alpha(t) = alpha(0) + t*Dalpha(0) + 0.5*t*t*DDalpha = alpha_max
            #     t = (- Dalpha(0) +/- sqrt(Dalpha(0)^2 - 2*DDalpha(alpha(0)-alpha_max))) / DDalpha;
            # where DDalpha = -d[i_DDalpha_min]
            # Having two solutions, we take the smallest one because we want to find the first time
            # at which alpha reaches alpha_max
            delta = sqrt(Dalpha**2 + 2*d[i_DDalpha_min]*(alpha-alpha_max))
            t = ( Dalpha - delta) / d[i_DDalpha_min];
            if(t<0.0):
                # If the smallest time at which alpha reaches alpha_max is negative print a WARNING because this should not happen
                print "WARNING: Time is less than zero:", t, alpha, Dalpha, d[i_DDalpha_min], alpha_max; 
                t = (Dalpha + delta) / d[i_DDalpha_min];
                if(t<0.0):
                    # If also the largest time is negative print an ERROR and return
                    print "ERROR: Time is still less than zero:", t, alpha, Dalpha, d[i_DDalpha_min], alpha_max; 
                    return (False, c0+alpha*v, Dalpha*v);
            
            
        bisection_converged = False;
        for jjjj in range(MAX_ITER):
            # Integrate LDS until t: DDalpha = a*alpha - d
            if(abs(a[i_DDalpha_min])>EPS):
                # if a=0 then the acceleration is a linear function of the position and I need to use this formula to integrate
                sh = np.sinh(omega*t);
                ch = np.cosh(omega*t);
                alpha_t  = ch*alpha + sh*Dalpha/omega + (1-ch)*(d[i_DDalpha_min]/a[i_DDalpha_min]);
                Dalpha_t = omega*sh*alpha + ch*Dalpha - omega*sh*(d[i_DDalpha_min]/a[i_DDalpha_min]);
            else:
                # if a=0 then the acceleration is constant and I need to use this formula to integrate
                alpha_t  = alpha + t*Dalpha - 0.5*t*t*d[i_DDalpha_min];
                Dalpha_t = Dalpha - t*d[i_DDalpha_min];
            
            if(np.imag(alpha_t) != 0.0):
                print "ERROR alpha is imaginary", alpha_t;
                return (False, c0+alpha*v, Dalpha*v);
            if(np.imag(Dalpha_t) != 0.0):
                print "ERROR Dalpha is imaginary", Dalpha_t
                return (False, c0+alpha*v, Dalpha*v);
                
            alpha_t = np.real(alpha_t);
            Dalpha_t = np.real(Dalpha_t);
            if(verb>0):
                print "Bisection iter",jjjj,"alpha",alpha_t,"Dalpha",Dalpha_t,"t", t
            
            if(abs(Dalpha_t)<EPS and alpha_t <= alpha_max+EPS):
                if(verb>0):
                    print "Algorithm converged to Dalpha=0";
                return (True, c0+alpha_t*v, Dalpha_t*v);
            if(abs(alpha_t-alpha_max)<EPS and Dalpha_t>0):
                alpha = alpha_max+EPS;
                Dalpha = Dalpha_t;
                bisection_converged = True;
                break;
            if(alpha_t<alpha_max and Dalpha_t>0):       
                t_lb=t;
                t=(t_ub+t_lb)/2;
            else:                                      
                t_ub=t; 
                t=(t_ub+t_lb)/2;
        
        if(not bisection_converged):
            print "ERROR: Bisection search did not converge in %d iterations"%MAX_ITER;
            return (False, c0+alpha*v, Dalpha*v);

    print "ERROR: Numerical integration did not converge in %d iterations"%MAX_ITER;
    if(DO_PLOTS):
        plt.show();
        
    return (False, c0+alpha*v, Dalpha*v);
    
    
    
''' PROBLEM 2: MULTI-CONTACT CAPTURE POINT (what's the max vel in a given direction such that I can stop)
    Input: initial CoM position c0, initial CoM velocity direction v, contact points CP, friction coefficient mu, T_0=1
    Output: The (norm of the) max initial CoM velocity such that I can stop without changing contacts is 
            a piece-wise linear function of the CoM position (on the given line), which is returned as a
            list P of 2D points in the space where:
            - x is the CoM offset relatively to c0 along v (i.e. c=c0+x*v)
            - y is the maximum CoM velocity along v
    Steps:
        - Compute GIWC: H*w <= h, where w=(m*(g-ddc), m*cx(g-ddc))
        - Project GIWC in (alpha,DDalpha) space, where c=c0+alpha*v, ddc=-DDalpha*v: A*(alpha,DDalpha)<=b
        - Find ordered (right-most first, counter clockwise) vertices of 2d polytope A*(alpha,DDalpha)<=b: V
        - if(line passing through c0 in direction v does not intersect static-equilibrium polytope): return []
        - Find extremum of static-equilibrium polytope in direction v: c1
        - Initialize: alpha=||c1-c0||, Dalpha=0
        - Initialize: P=[(alpha,0)]
        - LOOP:
            - Find current active inequality: a*alpha + b*DDalpha <= d (i.e. DDalpha upper bound for current alpha value)
            - Find alpha_min (i.e. value of alpha corresponding to left vertex of active inequality)
            - Initialize: t=-T_0, t_ub=-10, t_lb=0
            LOOP:
                - Integrate backward in time LDS until t:   DDalpha = d/b - (a/d)*alpha
                - if(alpha(t)==alpha_min):                  alpha=alpha(t), Dalpha=Dalpha(t)
                                                            P = [(alpha_min,Dalpha)]+P                   
                                                            break
                - if(alpha(t)<alpha_min):                   t_ub=t, t=(t_ub+t_lb)/2
                - else:                                     t_lb=t, t=(t_ub+t_lb)/2
            - if(alpha_min<=0): return P
            - if(left vertex of active inequality is left-most vertex): return P


    
    PROBLEM 3: VELOCITY PROPAGATION WITH CONTACT TRANSITION
    Input: initial CoM pos c0, initial CoM vel dc0, final CoM pos c2, initial contacts CP0, final contacts CP2, 
           friction coefficient mu, min CoM distance d_min to travel in CP1
    Assumptions: 
        - CoM path is a straight line from c0 to c2. 
        - CP0 and CP2 contains the same number of contacts N, out of which N-1 contacts must be exactly the same.
    Output: Min and max feasible final CoM velocity (in norm) c2_min, c2_max
    Steps:
        - Define CP1 as the intersection of CP0 and CP2
        - Define alpha as: c=c0+alpha*v, ddc=DDalpha*v
        - Define v=(c2-c0)/||c2-c0||
        - Compute the GIWCs associated to CP0, CP1, CP2: H^i*w <= h^i, where w=(m*(g-ddc), m*cx(g-ddc)), i=0,1,2
        - Project GIWCs Ci in (alpha,DDalpha) space: A^i*(alpha,DDalpha)<=b^i
        - Compute intersection between C0 and C1 (C01), and C1 and C2 (C12)
        - If C01 contains zero
'''

def test():
    DO_PLOTS = False;
    PLOT_3D = False;
    mass = 75;             # mass of the robot
    mu = 0.5;           # friction coefficient
    lx = 0.1;           # half foot size in x direction
    ly = 0.07;          # half foot size in y direction
    USE_DIAGONAL_GENERATORS = True;
    GENERATE_QUASI_FLAT_CONTACTS = True;
    #First, generate a contact configuration
    CONTACT_POINT_UPPER_BOUNDS = [ 0.5,  0.5,  0.0];
    CONTACT_POINT_LOWER_BOUNDS = [-0.5, -0.5,  0.0];
    gamma = atan(mu);   # half friction cone angle
    RPY_LOWER_BOUNDS = [-0*gamma, -0*gamma, -pi];
    RPY_UPPER_BOUNDS = [+0*gamma, +0*gamma, +pi];
    MIN_CONTACT_DISTANCE = 0.3;
    N_CONTACTS = 2
    READ_CONTACTS_FROM_FILE = True;
    X_MARG = 0.07;
    Y_MARG = 0.07;
    
    if(READ_CONTACTS_FROM_FILE):
        import pickle
        f = open("./data.pkl", 'rb');
        res = pickle.load(f);
        f.close();
#        (p, N) = generate_contacts(N_CONTACTS, lx, ly, mu, CONTACT_POINT_LOWER_BOUNDS, CONTACT_POINT_UPPER_BOUNDS, RPY_LOWER_BOUNDS, RPY_UPPER_BOUNDS, MIN_CONTACT_DISTANCE, GENERATE_QUASI_FLAT_CONTACTS);
        p = res['contact_points'].T;
        N = res['contact_normals'].T;
        print "Contact points\n", p;
        print "Contact normals\n", 1e3*N
        X_LB = np.min(p[:,0]-X_MARG);
        X_UB = np.max(p[:,0]+X_MARG);
        Y_LB = np.min(p[:,1]-Y_MARG);
        Y_UB = np.max(p[:,1]+Y_MARG);
        Z_LB = np.min(p[:,2]-0.05);
        Z_UB = np.max(p[:,2]+1.5);
        (H,h) = compute_GIWC(p, N, mu, False, USE_DIAGONAL_GENERATORS);
        (succeeded, c0) = find_static_equilibrium_com(mass, [X_LB, Y_LB, Z_LB], [X_UB, Y_UB, Z_UB], H, h);
        if(not succeeded):
            print "Impossible to find a static equilibrium CoM position with the contacts read from file";
            return
    else:
        succeeded = False;
        while(succeeded == False):
            (p, N) = generate_contacts(N_CONTACTS, lx, ly, mu, CONTACT_POINT_LOWER_BOUNDS, CONTACT_POINT_UPPER_BOUNDS, RPY_LOWER_BOUNDS, RPY_UPPER_BOUNDS, MIN_CONTACT_DISTANCE, GENERATE_QUASI_FLAT_CONTACTS);        
            X_LB = np.min(p[:,0]-X_MARG);
            X_UB = np.max(p[:,0]+X_MARG);
            Y_LB = np.min(p[:,1]-Y_MARG);
            Y_UB = np.max(p[:,1]+Y_MARG);
            Z_LB = np.min(p[:,2]-0.05);
            Z_UB = np.max(p[:,2]+1.5);
            (H,h) = compute_GIWC(p, N, mu, False, USE_DIAGONAL_GENERATORS);
            (succeeded, c0) = find_static_equilibrium_com(mass, [X_LB, Y_LB, Z_LB], [X_UB, Y_UB, Z_UB], H, h);
        
    dc0 = np.random.uniform(-1, 1, size=3); 
    dc0[2] = 0;
    
    if(DO_PLOTS):
        f, ax = plut.create_empty_figure();
        for j in range(p.shape[0]):
            ax.scatter(p[j,0], p[j,1], c='k', s=100);
        ax.scatter(c0[0], c0[1], c='r', s=100);
        com_x = np.zeros(2);
        com_y = np.zeros(2);
        com_x[0] = c0[0]; 
        com_y[0] = c0[1];
        com_x[1] = c0[0]+dc0[0]; 
        com_y[1] = c0[1]+dc0[1];
        ax.plot(com_x, com_y, color='b');
        plt.axis([X_LB,X_UB,Y_LB,Y_UB]);           
        plt.title('Contact Points and CoM position'); 
        
    if(PLOT_3D):
        fig = plt.figure(figsize=plt.figaspect(0.5)*1.5)
        ax = fig.gca(projection='3d')
        line_styles =["b", "r", "c", "g"];
        ss = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3];
        ax.scatter(c0[0],c0[1],c0[2], c='k', marker='o');
        for i in range(p.shape[0]):
            ax.scatter(p[i,0],p[i,1],p[i,2], c=line_styles[i%len(line_styles)], marker='o');
            for s in ss:
                ax.scatter(p[i,0]+s*N[i,0],p[i,1]+s*N[i,1],p[i,2]+s*N[i,2], c=line_styles[i%len(line_styles)], marker='x');
        for s in ss:
            ax.scatter(c0[0]+s*dc0[0],c0[1]+s*dc0[1],c0[2]+s*dc0[2], c='k', marker='x');
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z');
   
#    return can_I_stop(c0, dc0, p, N, mu, mass, 1.0, 100, DO_PLOTS=DO_PLOTS);
    (has_stopped, c_final, dc_final) = can_I_stop(c0, dc0, p, N, mu, mass, 1.0, 100, DO_PLOTS=DO_PLOTS);
    print "Contact points\n", p;
    print "Contact normals\n", N
    print "Initial com position", c0
    print "Initial com velocity", dc0, "norm %.3f"%norm(dc0)
    print "Final com position", c_final
    print "Final com velocity", dc_final, "norm %.3f"%norm(dc_final)
    if(has_stopped):
        print "The system is stable"
    else:
        print "The system is unstable"
    
    return True;
        

if __name__=="__main__":
    np.set_printoptions(precision=2, suppress=True, linewidth=100);
    np.random.seed(0);
    for i in range(1):
        try:
            test();
#            ret = cProfile.run("test()");
        except Exception as e:
            print e;
            continue;
