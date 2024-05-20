from .Solver import solve_mechanism
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import xml.etree.ElementTree as etree
from svgpath2mpl import parse_path

def draw_mechanism(A,x0,fixed_nodes=None,motor=[0,1],node_types=None, ax=None, highlight=None, solve=True, thetas = np.linspace(0,np.pi*2,200), def_alpha = 1.0, def_c = "#0078a7", h_alfa =1.0, h_c = "#f15a24"):
    
    if fixed_nodes is None and node_types is None:
        raise ValueError("Either fixed_nodes or node_types should be provided")

    if fixed_nodes is None:
        fixed_nodes = np.where(node_types)[0]
    
    def fetch_path():
    #     r = requests.get(svg_url)
        root = etree.parse(StringIO('<svg id="Layer_1" data-name="Layer 1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 620 338"><defs><style>.cls-1{fill:#1a1a1a;stroke:#1a1a1a;stroke-linecap:round;stroke-miterlimit:10;stroke-width:20px;}</style></defs><path class="cls-1" d="M45.5,358.5l70.71-70.71M46,287.5H644m-507.61,71,70.72-70.71M223,358.5l70.71-70.71m20.18,70.72,70.71-70.71m13.67,70.7,70.71-70.71m20.19,70.72,70.71-70.71m15.84,70.71,70.71-70.71M345,39.62A121.38,121.38,0,1,1,223.62,161,121.38,121.38,0,0,1,345,39.62Z" transform="translate(-35.5 -29.62)"/></svg>')).getroot()
        view_box = root.attrib.get('viewBox')
        if view_box is not None:
            view_box = [int(x) for x in view_box.split()]
            xlim = (view_box[0], view_box[0] + view_box[2])
            ylim = (view_box[1] + view_box[3], view_box[1])
        else:
            xlim = (0, 500)
            ylim = (500, 0)
        path_elem = root.findall('.//{http://www.w3.org/2000/svg}path')[0]
        return xlim, ylim, parse_path(path_elem.attrib['d'])
    _,_,p = fetch_path()
    p.vertices -= p.vertices.mean(axis=0)
    p.vertices = (np.array([[np.cos(np.pi), -np.sin(np.pi)],[np.sin(np.pi), np.cos(np.pi)]])@p.vertices.T).T
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,10))
    
    A,x0,fixed_nodes,motor = np.array(A),np.array(x0),np.array(fixed_nodes),np.array(motor)
    
    x = x0
    
    N = A.shape[0]
    for i in range(N):
        if i in fixed_nodes:
            if i == highlight:
                ax.scatter(x[i,0],x[i,1],color=h_c,s=700,zorder=10,marker=p)
            else:
                ax.scatter(x[i,0],x[i,1],color="#1a1a1a",s=700,zorder=10,marker=p)
        else:
            if i == highlight:
                ax.scatter(x[i,0],x[i,1],color=h_c,s=100,zorder=10,facecolors=h_c,alpha=0.7)
            else:
                ax.scatter(x[i,0],x[i,1],color="#1a1a1a",s=100,zorder=10,facecolors='#ffffff',alpha=0.7)
        
        for j in range(i+1,N):
            if A[i,j]:
                if (motor[0] == i and motor[1] == j) or(motor[0] == j and motor[1] == i):
                    ax.plot([x[i,0],x[j,0]],[x[i,1],x[j,1]],color="#ffc800",linewidth=4.5)
                else:
                    ax.plot([x[i,0],x[j,0]],[x[i,1],x[j,1]],color="#1a1a1a",linewidth=4.5,alpha=0.6)
                
    if solve:
        sol,ord = solve_mechanism(A,x0,motor,fixed_nodes,thetas)
        x = sol
        
        for i in range(A.shape[0]):
            if not ord[i] in fixed_nodes:
                if ord[i] == highlight:
                    ax.plot(x[i,:,0],x[i,:,1],'-',color=h_c,linewidth=4.5,alpha=h_alfa)
                else:
                    ax.plot(x[i,:,0],x[i,:,1],'--',color=def_c,linewidth=1.5, alpha=def_alpha)
    ax.axis('equal')
    ax.axis('off')
    
    return ax