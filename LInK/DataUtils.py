import torch
import numpy as np
from .CurveUtils import uniformize

from torch.utils.data import Dataset, DataLoader

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
    x0_id = '1HEHhSGQS029LJTMPbeKzduuh1qVns4a2'
    connectivity_id = '1Tc4l8288oa_RfOWfnAmNQYAN9R2FZBpc'
    graphs_id = '1Arr36z0VbLEHbPImXUOs3QEvjl_7weGe'
    node_types_id = '15-2MTlm26l2xO6ZMai3zlplSkjwWSSDC'
    target_curves_id = '1BbE8FTWqSUUCmuff6iofgrP3vwPEN90n'
    
    random_mechanisms_test_id = '1n0PFrQazIaKyALoWFVyQ4TwlS_Q11W4n'
    alphabet_test_id = '14iT6lk78O_VY7up86sO_4sdxZ7CPEJXc'
    gcp_micp_test_id = '1pURHa_ztAISMjRusR6HSHlEEa8RK_Au8'
    
    gdown.download(id=x0_id, output=data_folder+'x0.npy', quiet=False)
    gdown.download(id=connectivity_id, output=data_folder+'connectivity.npy', quiet=False)
    gdown.download(id=graphs_id, output=data_folder+'graphs.npy', quiet=False)
    gdown.download(id=node_types_id, output=data_folder+'node_types.npy', quiet=False)
    gdown.download(id=target_curves_id, output=data_folder+'target_curves.npy', quiet=False)
    
    gdown.download(id=random_mechanisms_test_id, output=test_folder+'random_mechanisms.npy', quiet=False)
    gdown.download(id=alphabet_test_id, output=test_folder+'alphabet.npy', quiet=False)
    gdown.download(id=gcp_micp_test_id, output=test_folder+'gcp_micp.npy', quiet=False)
    
class LInKDataset(Dataset):
    def __init__(self, mechanism_graphs, curves, mechanism_sequences, x0s, masks, curve_size=200):
        self.mechanism_graphs = mechanism_graphs
        self.curves = curves
        self.mechanism_sequences = mechanism_sequences
        self.x0s = x0s
        self.masks = masks

        self.curve_size = curve_size
        
    def __len__(self):
        return self.x0s.shape[0]
    
    def load(self, idx):
        return None
    
    def batch_load(self, idxs, device=None):
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        batch_mech = self.mechanism_graphs[idxs]
        batch_curves = torch.tensor(self.curves[idxs]).float().to(device)

        x, edge_index, size = zip(*batch_mech)
        size = np.array(size)
        x = torch.tensor(np.concatenate(x)).float().to(device)
        num_edges = np.array([e.shape[1] for e in edge_index])
        edge_index = torch.tensor(np.concatenate(edge_index,-1) + np.repeat(np.cumsum(np.pad(size,[1,0],constant_values=0))[:-1],num_edges).reshape(1,-1)).long().to(device)
        b = torch.tensor(np.repeat(np.arange(size.shape[0]),size)).long().to(device)
        # ext_idx = torch.tensor(np.cumsum(size)-1).long().to(device)
        
        base, inp = prep_curves(batch_curves, self.curve_size)

        batch_sequence = torch.tensor(self.mechanism_sequences[idxs]).long().to(device)
        batch_x0 = torch.tensor(self.x0s[idxs]).float().to(device)
        batch_mask = torch.tensor(self.masks[idxs]).bool().to(device)
        
        return {'inp':inp, 'base':base}, {'x':x, 'edge_index':edge_index, 'batch':b}, [batch_sequence, batch_x0, batch_mask]