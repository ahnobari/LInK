import numpy as np
from .Solver import solve_rev_vectorized_batch_CPU


def onSegment(p, q, r): 
    if ( (q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and 
           (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))): 
        return True
    return False
  
def orientation(p, q, r): 
    val = (float(q[1] - p[1]) * (r[0] - q[0])) - (float(q[0] - p[0]) * (r[1] - q[1])) 
    if (val > 0): 
        # Clockwise orientation 
        return 1
    elif (val < 0): 
        # Counterclockwise orientation 
        return 2
    else:  
        # Collinear orientation 
        return 0
    
def line_line_collision(p1,q1,p2,q2): 
    o1 = orientation(p1, q1, p2) 
    o2 = orientation(p1, q1, q2) 
    o3 = orientation(p2, q2, p1) 
    o4 = orientation(p2, q2, q1) 
    # General case 
    if ((o1 != o2) and (o3 != o4)): 
        return True

    # Special Cases 
    # p1 , q1 and p2 are collinear and p2 lies on segment p1q1 
    if ((o1 == 0) and onSegment(p1, p2, q1)): 
        return True
    # p1 , q1 and q2 are collinear and q2 lies on segment p1q1 
    if ((o2 == 0) and onSegment(p1, q2, q1)): 
        return True
    # p2 , q2 and p1 are collinear and p1 lies on segment p2q2 
    if ((o3 == 0) and onSegment(p2, p1, q2)): 
        return True
    # p2 , q2 and q1 are collinear and q1 lies on segment p2q2 
    if ((o4 == 0) and onSegment(p2, q1, q2)): 
        return True
    # If none of the cases 
    return False

def point_line_distance(p1, p2, p3):
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
    v = p2 - p1
    w = p3 - p1
    s = p3 - p2
    u = p1 - p2
    
    if np.dot(v,w) * np.dot(u,s) < 0:
        d = np.min([np.linalg.norm(p1-p3),np.linalg.norm(p2-p3)])
    else:
        d = np.cross(v,w)/np.linalg.norm(v)
    
    return d

def linkage_collisions(A,x0,nt,sol=None,start=0,end=2*np.pi,steps=200):
    n_joints = (A.sum(-1)>0).sum()
    A = A[:n_joints,:][:,:n_joints]
    x0 = x0[:n_joints]
    nt = nt[:n_joints]
    n_links = int(A.sum()/2)
    collision_matrix = np.zeros((n_links,n_links),dtype=bool)
    if sol is None:
        sol = solve_rev_vectorized_batch_CPU(np.expand_dims(A,0),np.expand_dims(x0,0),np.expand_dims(nt,0),np.linspace(start,end,steps))[0]
    l1,l2 = np.where(np.triu(A))
    for i in range(sol.shape[1]):
        s = sol[:,i,:]
        for j in range(n_links):
            for k in range(n_links):
                if line_line_collision(s[l1[j]],s[l2[j]],s[l1[k]],s[l2[k]]):
                    collision_matrix[j,k] = True
                    collision_matrix[k,j] = True
    return collision_matrix

def linkage_joint_collisions(A,x0,nt,tolerance=0.08, sol=None, start=0,end=2*np.pi,steps=200):
    
    n_joints = (A.sum(-1)>0).sum()
    A = A[:n_joints,:][:,:n_joints]
    x0 = x0[:n_joints]
    nt = nt[:n_joints]

    n_links = int(A.sum()/2)
    collision_matrix = np.zeros((n_links,A.shape[0]),dtype=bool)
    if sol is None:
        sol = solve_rev_vectorized_batch_CPU(np.expand_dims(A,0),np.expand_dims(x0,0),np.expand_dims(nt,0),np.linspace(start,end,steps))[0]
    l1,l2 = np.where(np.triu(A))
    for i in range(sol.shape[1]):
        s = sol[:,i,:]
        for j in range(n_links):
            for k in range(A.shape[0]):
                if point_line_distance(s[l1[j]],s[l2[j]],x0[k]) < tolerance:
                    collision_matrix[j,k] = True
    
    for i in range(n_links):
        collision_matrix[i,l1[i]] = False
        collision_matrix[i,l2[i]] = False
    
    return collision_matrix