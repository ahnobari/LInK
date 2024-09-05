import numpy as np
from .Solver import solve_rev_vectorized_batch_CPU
from tqdm import trange
import gurobipy as gp

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
        d = abs(np.cross(v,w)/np.linalg.norm(v))
    
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
                if point_line_distance(s[l1[j]],s[l2[j]],s[k]) < tolerance:
                    collision_matrix[j,k] = True
    
    for i in range(n_links):
        collision_matrix[i,l1[i]] = False
        collision_matrix[i,l2[i]] = False
    
    return collision_matrix

def get_layers(A, x0, node_types, sol, zero_act = False, relax_groud = True):
    
    O = linkage_collisions(A, x0, node_types, sol=sol)
    O &= ~np.eye(O.shape[0],dtype=bool)
    C = linkage_joint_collisions(A, x0, node_types, sol=sol).T
    l1,l2 = np.where(np.triu(A))
    AJ = np.zeros([A.shape[0],len(l1)])
    AJ[l1,np.arange(len(l1))] = 1
    AJ[l2,np.arange(len(l1))] = 1
    AJ = AJ.astype(bool)
    
    lt = 1.0
    N = lt*10000

    m = gp.Model("LinkageCAD")
    m.setParam('LogToConsole', 0)

    z = m.addVars(O.shape[0], lb=0, ub=O.shape[0]*lt, vtype=gp.GRB.INTEGER, name="z")
    u = m.addVars(C.shape[0], lb=0, ub=O.shape[0]*lt, vtype=gp.GRB.INTEGER, name="u")
    v = m.addVars(C.shape[0], lb=0, ub=O.shape[0]*lt, vtype=gp.GRB.INTEGER, name="v")
    y = m.addVars(int(C.sum()), vtype=gp.GRB.BINARY, name="y")
    x = m.addVars(int(O.sum()//2), vtype=gp.GRB.BINARY, name="x")
    ml = m.addVar(lb=0, ub=O.shape[0]*lt + lt, vtype=gp.GRB.INTEGER, name="ml")
    
    if zero_act:
        act_con = m.addConstr(z[0]==0., name="z0")

    counter = 0
    for i in range(O.shape[0]):
        col_i = np.where(O[i])[0]
        col_i = col_i[col_i>i]
        for j in col_i:
            m.addConstr(z[i]-z[j] + N * x[counter] >= lt, name=f"z{i}-{j}+")
            m.addConstr(z[j]-z[i] + N * (1 - x[counter]) >= lt, name=f"z{i}-{j}-")
            counter += 1

    for j in range(C.shape[0]):
        col_i = np.where(AJ[j])[0]
        for i in col_i:
            m.addConstr(v[j] >= z[i], name=f"v{j}>z{i}")
            m.addConstr(u[j] <= z[i], name=f"u{j}<z{i}")

    counter = 0
    for j in range(C.shape[0]):
        if node_types[j] == 0 or not relax_groud:
            col_i = np.where(C[j])[0]
            for i in col_i:
                m.addConstr(z[i] >= v[j] + lt - N * y[counter], name=f"z{i}>v{j}")
                m.addConstr(z[i] <= u[j] - lt + N * (1-y[counter]), name=f"z{i}<u{j}")
                counter += 1

    m.addConstrs((ml >= z[i] for i in range(O.shape[0])), name="ml>z")

    m.setObjective(ml, gp.GRB.MINIMIZE)
    m.setParam('TimeLimit', 20)
    m.setParam('Threads', 24)

    m.optimize()
    
    if m.status == gp.GRB.INFEASIBLE:
        print("Not Manufacturable!")
        return get_layers_relaxed(A, x0, node_types, sol), False
        
    else:
        if m.status == gp.GRB.TIME_LIMIT:
            print("Time limit reached")
            print("Objective value: ", m.objVal)
        else:
            print("Optimal")
            print("Objective value: ", m.objVal)

        z = np.array([z[i].X for i in range(O.shape[0])]).astype(int)
        return z, True

def get_layers_relaxed(A_, x0_, node_types_, sol_, verbose=True, relax_groud = False):
    A = A_
    x0 = x0_
    nt = node_types_
    col_mat = linkage_collisions(A,x0,nt,sol=sol_)
    
    n_links = int(A.sum()/2)
    z_index = np.zeros(n_links).astype(int) - 1
    
    current_z = 0
    
    while np.any(z_index == -1):
        available_inds = col_mat[z_index == current_z,:].sum(0) == 0
        if np.any(np.logical_and(available_inds,z_index == -1)):
            z_index[np.where(np.logical_and(available_inds,z_index == -1))[0][0]] = current_z
        else:
            z_index[np.where(z_index==-1)[0][0]] = current_z + 1
            current_z += 1
        
    return z_index

def get_3d_config(A,x0,nt,z_index):
    
    A, x0, nt, z_index = np.array(A), np.array(x0), np.array(nt), np.array(z_index)
    
    n_joints = (A.sum(-1)>0).sum()
    A = A[:n_joints,:][:,:n_joints]
    x0 = x0[:n_joints]
    nt = nt[:n_joints]
    n_links = int(A.sum()/2)

    l1,l2 = np.where(np.triu(A))
    
    linkages = []
    
    max_len = 0
    min_len = float(np.inf)

    for j in range(n_links):
        length= np.linalg.norm(x0[l1[j]]-x0[l2[j]])
        if length>max_len:
            max_len = float(length)
        if length<min_len:
            min_len = float(length)
    
    scale_min = 0.25/min_len
    scale_target = 1.0/max_len
    scale = max(scale_min,scale_target)
    
    x0 = x0*scale

    for j in range(n_links):
        length= np.linalg.norm(x0[l1[j]]-x0[l2[j]])
        angle = np.arctan2(x0[l2[j]][1]-x0[l1[j]][1],x0[l2[j]][0]-x0[l1[j]][0])
        linkages.append([length,0.1,0.05,0.03, angle, x0[l1[j]].tolist()+[float(z_index[j])*0.05]])
    
    joints_max_z = np.zeros(x0.shape[0])
    joints_min_z = np.zeros(x0.shape[0]) + np.inf
    
    for i in range(n_links):
        joints_max_z[l1[i]] = max(joints_max_z[l1[i]],z_index[i])
        joints_max_z[l2[i]] = max(joints_max_z[l2[i]],z_index[i])
        joints_min_z[l1[i]] = min(joints_min_z[l1[i]],z_index[i])
        joints_min_z[l2[i]] = min(joints_min_z[l2[i]],z_index[i])

    joints_max_z = joints_max_z*0.05
    joints_min_z = joints_min_z*0.05

    joints = []
    for i in range(x0.shape[0]):
        if nt[i] == 1:
            for j in np.where(l1==i)[0]:
                joints.append(x0[i].tolist()+[0.05,z_index[j]*0.05,1])
            for j in np.where(l2==i)[0]:
                joints.append(x0[i].tolist()+[0.05,z_index[j]*0.05,1])
        else:
            joints.append(x0[i].tolist()+[float(joints_max_z[i]-joints_min_z[i])+0.05,float(joints_min_z[i]+joints_max_z[i])/2,0])

    return [linkages, joints], joints_max_z, scale

def get_animated_3d_config(A,x0,nt,z_index,sol, highlights = [-1]):
    A, x0, nt, z_index, sol = np.array(A), np.array(x0), np.array(nt), np.array(z_index), np.array(sol)
    configs = []
    for i in range(sol.shape[1]):
        c,z,s = get_3d_config(A,sol[:,i,:],nt,z_index)
        configs.append(c)
    
    if len(highlights) > 1:
        highligh_curve = []
        for i in highlights:
            highligh_curve.append(np.pad(sol[i,:,:]*s,[[0,0],[0,1]],constant_values=z[i]+0.025))
        highligh_curve = np.array(highligh_curve)
    else:
        highligh_curve = np.pad(sol[highlights[0],:,:]*s,[[0,0],[0,1]],constant_values=z[-1]+0.025)

    return configs, highligh_curve.tolist()

def create_3d_html(A,x0,nt,z_index,sol, template_path='./static/animation.htm', save_path='./static/animated.html', highlights = [-1]):
    
    res,hc = get_animated_3d_config(A,x0,nt,z_index,sol,highlights=highlights)
    js_var = 'window.res = ' + str(res) + ';\n'
    js_var += 'window.hc = ' + str(hc) + ';'
    js_var += 'window.multi_high = ' + str(int(len(highlights)>1)) + ';'
    
    with open(template_path, 'r') as file:
        filedata = file.read()
        
    filedata = filedata.replace('{res}',js_var)
    
    with open(save_path, 'w') as file:
        file.write(filedata)