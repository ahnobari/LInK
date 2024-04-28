import torch
import numpy as np
from .CurveUtils import uniformize

def rnd_fourier_transform(curves_in,n):
    
    # equidistant sampling (Remove Timing)
    curves = uniformize(curves_in,n)

    # center curves
    curves = curves - curves.mean(1).unsqueeze(1)
    
    # apply uniform scaling
    s = torch.sqrt(torch.square(curves).sum(-1).sum(-1)/n).unsqueeze(-1).unsqueeze(-1)
    curves = curves/s

    # reduce with fft
    n_freq = np.random.randint(5,11)
    curves = torch.cat([torch.real(torch.fft.ifft(torch.fft.fft(curves[:,:,0],dim=1)[:,0:n_freq],n=n,dim=1)).unsqueeze(2),
                        torch.real(torch.fft.ifft(torch.fft.fft(curves[:,:,1],dim=1)[:,0:n_freq],n=n,dim=1)).unsqueeze(2)],dim=2)
    
    # randomly pick some curves to not have fft
    no_fft = torch.rand([curves.shape[0]]) > 0.5
    curves[no_fft] = curves_in[no_fft]

    return curves

def sample_partials(curves,n):

    # get random partials with atlease 30% of the curve
    l = torch.cumsum(torch.nn.functional.pad(torch.norm(curves[:,1:,:] - curves[:,:-1,:],dim=-1),[1,0,0,0]),-1)
    l = l/l[:,-1].unsqueeze(-1)
    
    sampling = torch.linspace(0,1,n).to(l.device).unsqueeze(0).tile([l.shape[0],1])
    sampling = sampling * (0.7 * torch.rand(curves.shape[0],1) + 0.3).to(curves.device)
    end_is = torch.searchsorted(l,sampling)[:,1:]
    end_ids = end_is.unsqueeze(-1).tile([1,1,2])

    l_end = torch.gather(l,1,end_is)
    l_start = torch.gather(l,1,end_is-1)
    ws = (l_end - sampling[:,1:])/(l_end-l_start)

    end_gather = torch.gather(curves,1,end_ids)
    start_gather = torch.gather(curves,1,end_ids-1)

    partial_curves = torch.cat([curves[:,0:1,:],(end_gather - (end_gather-start_gather)*ws.unsqueeze(-1))],1)

    # center curves
    partial_curves = partial_curves - partial_curves.mean(1).unsqueeze(1)

    # apply uniform scaling
    s = torch.sqrt(torch.square(partial_curves).sum(-1).sum(-1)/n).unsqueeze(-1).unsqueeze(-1)
    partial_curves = partial_curves/s

    # apply random rotation'
    R = torch.eye(2).unsqueeze(0).to(partial_curves.device)
    R = R.repeat([partial_curves.shape[0],1,1])

    theta = torch.rand([partial_curves.shape[0]]).to(partial_curves.device) * 2 * np.pi
    R[:,0,0] = torch.cos(theta)
    R[:,0,1] = -torch.sin(theta)
    R[:,1,0] = torch.sin(theta)
    R[:,1,1] = torch.cos(theta)

    partial_curves = torch.bmm(R,partial_curves.transpose(1,2)).transpose(1,2)

    return partial_curves

def preprocess_curves(curves: torch.tensor, n: int = 200) -> torch.tensor:
    
    # equidistant sampling (Remove Timing)
    curves = uniformize(curves,n)

    # center curves
    curves = curves - curves.mean(1).unsqueeze(1)
    
    # apply uniform scaling
    s = torch.sqrt(torch.square(curves).sum(-1).sum(-1)/n).unsqueeze(-1).unsqueeze(-1)
    curves = curves/s

    # find the furthest point on the curve
    max_idx = torch.square(curves).sum(-1).argmax(dim=1)

    # rotate curves so that the furthest point is horizontal
    theta = -torch.atan2(curves[torch.arange(curves.shape[0]),max_idx,1],curves[torch.arange(curves.shape[0]),max_idx,0])

    # normalize the rotation
    R = torch.eye(2).unsqueeze(0).to(curves.device)
    R = R.repeat([curves.shape[0],1,1])

    # theta = torch.rand([curves.shape[0]]).to(curves.device) * 2 * np.pi
    R[:,0,0] = torch.cos(theta)
    R[:,0,1] = -torch.sin(theta)
    R[:,1,0] = torch.sin(theta)
    R[:,1,1] = torch.cos(theta)

    curves = torch.bmm(R,curves.transpose(1,2)).transpose(1,2)

    return curves

def prep_curves(curves, n, fourier=False):
    curves = torch.tensor(curves).float()
    if fourier:
        curves_1 = rnd_fourier_transform(curves,n)
    else:
        curves_1 = curves
    curves_2 = curves_1 + 0.0
    partials = sample_partials(curves_1,n)
    curves = preprocess_curves(curves_2,n)
    return curves, partials

def download_data(data_folder='./Data/', test_folder='./TestData/'):
    import gdown
    x0_id = '1vwEwUnLD76OnkORhhKyGsjG43T5Zm95F'
    connectivity_id = '1ejU9Rkn-WVGUn7ZfUYs9MYko2uaivGvP'
    graphs_id = '1Arr36z0VbLEHbPImXUOs3QEvjl_7weGe'
    node_types_id = '1OEVlqH1x_pThFnUZ4W3jmO4EtNrmUF4v'
    target_curves_id = '1MUHgeQTIxclozb3Xnmz_O4aZnreVMNoT'
    
    random_mechanisms_test_id = '1xmr-UwvxSR3PpjvEALhWoY_x9kyT4jMe'
    alphabet_test_id = '1Y9-FSQU0YHd1_F0rPUT6WiPcV590jbYJ'
    gcp_micp_test_id = '1Hk_IIKYsQQKS83fr60m35OwxOe-YUbq_'
    
    gdown.download(id=x0_id, output=data_folder+'x0.npy', quiet=False)
    gdown.download(id=connectivity_id, output=data_folder+'connectivity.npy', quiet=False)
    gdown.download(id=graphs_id, output=data_folder+'graphs.npy', quiet=False)
    gdown.download(id=node_types_id, output=data_folder+'node_types.npy', quiet=False)
    gdown.download(id=target_curves_id, output=data_folder+'target_curves.npy', quiet=False)
    
    gdown.download(id=random_mechanisms_test_id, output=test_folder+'random_mechanisms.npy', quiet=False)
    gdown.download(id=alphabet_test_id, output=test_folder+'alphabet.npy', quiet=False)
    gdown.download(id=gcp_micp_test_id, output=test_folder+'gcp_micp.npy', quiet=False)
    
    
    