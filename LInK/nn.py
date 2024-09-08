import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torchvision import models
import numpy as np
from .DataUtils import prep_curves
from tqdm import tqdm, trange

import transformers

def download_checkpoint(checkpoint_folder='./Checkpoints', checkpoint_name='checkpoint.LInK', id='1lNFz8jNrKWeJNqYPuz9LUDF0cb4fl08d', id_cpu="1ZpOMiH-Z6oyD1Sw97PK0qb3wybfA6fW9"):
    import gdown
    gdown.download(id=id, output=f'{checkpoint_folder}/{checkpoint_name}', quiet=False)
    cpu_name = checkpoint_name.replace('.','CPU.')
    gdown.download(id=id_cpu, output=f'{checkpoint_folder}/{cpu_name}', quiet=False)
    
def download_emdedding(embeddings_folder = './Embeddings', id='10o3lwi0RUg3QCsDeRJp9DJg_PB4XCETa'):
    import gdown
    gdown.download(id=id, output=f'{embeddings_folder}/embeddings.npy', quiet=False)
    
class GraphHop(nn.Module):
    def __init__(self, num_node_features = 5, hidden_dim = 768, embedding_dim = 512, num_layers= 16, num_attn_heads=8):
        super(GraphHop, self).__init__()
        self.num_layers = num_layers
        self.projection = nn.Linear(num_node_features, hidden_dim)
        self.convs = nn.ModuleList([
            gnn.GINConv(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )) for _ in range(num_layers)
        ])
        self.GAT = gnn.GATConv(hidden_dim * (num_layers+1), hidden_dim//4, heads=4)
        # self.residuals = nn.ModuleList([
        #     nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)
        # ])
        
        self.HopAttn = nn.MultiheadAttention(hidden_dim, num_attn_heads, batch_first=True)
        
        self.fc = nn.Linear(hidden_dim, embedding_dim)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, batch, target_idx = None):
        x = self.projection(x) # x: [num_nodes, hidden_dim]
        x_list = [x]

        for i in range(self.num_layers):
            h = self.convs[i](x_list[-1], edge_index) # h: [num_nodes, hidden_dim]
            # if i > 0:
            #     residual = self.residuals[i - 1](x_list[-1]) + h # residual: [num_nodes, hidden_dim]
            #     x_list.append(self.relu(residual)) # x_list[i]: [num_nodes, hidden_dim]
            # else:
            x_list.append(h) # x_list[i]: [num_nodes, hidden_dim]

        x = torch.cat(x_list, dim=1) # x: [num_nodes, hidden_dim * num_layers]
        x = self.GAT(x, edge_index) # x: [num_nodes, hidden_dim]
        x = self.relu(x) # x: [num_nodes, hidden_dim]
        
        Q = x.unsqueeze(1) # Q: [num_graphs, 1, hidden_dim]
        V = torch.cat([r.unsqueeze(1) for r in x_list], dim=1) # V: [num_graphs, num_layers+1, hidden_dim]
        K = V # K: [num_graphs, num_layers+1, hidden_dim]
        
        x = self.HopAttn(Q, K, V)[0].squeeze() # x: [num_graphs, hidden_dim]
        
        if target_idx is not None:
            x = x[target_idx] # x: [num_graphs, hidden_dim]
        else:
            x = gnn.global_mean_pool(x, batch) # x: [num_graphs, hidden_dim]
            
        x = self.fc(x) # x: [num_graphs, embedding_dim]
        return x

class LInKDecoder(nn.Module):
    def __init__(self, latent_dim = 512, vocab_size = 26, hidden_size = 256, intemediate_size = 1024, num_hidden_layers = 8, num_attention_heads = 16, max_position_embeddings = 128):
        super(LInKDecoder, self).__init__()
        # llama Decoder With fixed length latent input before SOS
        self.latent_map = nn.Linear(latent_dim, hidden_size)
        self.continious_map = nn.Linear(2, hidden_size)

        model_config = transformers.LlamaConfig(
            vocab_size = vocab_size,
            hidden_size = hidden_size,
            intermediate_size = intemediate_size,
            num_hidden_layers = num_hidden_layers,
            num_attention_heads = num_attention_heads,
            max_position_embeddings = max_position_embeddings
        )

        self.decoder = transformers.LlamaModel(model_config)

        self.SOS = nn.Parameter(torch.randn(1,1,hidden_size))

        self.node_type_classifier = nn.Linear(hidden_size, 3)
        self.continious_mlp = nn.Linear(hidden_size, 2)
        self.connectivity_classifier = nn.Linear(hidden_size, 21)
    
    def forward(self, latent, inputs):
        input_ids, positions, mask = inputs

        latent = self.latent_map(latent)
        latent = latent.unsqueeze(1)
        sos = self.SOS.repeat(latent.shape[0],1,1)
        pose_embeddings = self.continious_map(positions)

        sequence_emb = self.decoder.embed_tokens(input_ids)
        sequence_emb[:,1::4] = pose_embeddings

        x = torch.cat([latent, sos, sequence_emb], dim=1)
        mask = torch.cat([torch.ones(mask.shape[0],2).bool().to(mask.device), mask], dim=1).float()
        x = self.decoder(inputs_embeds=x, attention_mask=mask).last_hidden_state

        node_types = self.node_type_classifier(x[:,1::4,:])[:,:-1,:]
        continious = self.continious_mlp(x[:,2::4,:])
        connectivity_1 = self.connectivity_classifier(x[:,3::4,:])
        connectivity_2 = self.connectivity_classifier(x[:,4::4,:])

        return x, node_types, continious, connectivity_1, connectivity_2

class GATBaseline(nn.Module):
    def __init__(self, num_node_features = 5, hidden_dim = 768, embedding_dim = 512, num_layers= 16, num_attn_heads=8):
        super(GATBaseline, self).__init__()
        self.projection = nn.Linear(num_node_features, hidden_dim)
        self.convs = nn.ModuleList([
            gnn.GATConv(hidden_dim, hidden_dim//num_attn_heads, heads=num_attn_heads) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_dim, embedding_dim)
        self.relu = nn.ReLU()
        
        self.num_layers = num_layers

    def forward(self, x, edge_index, batch, target_idx = None):
        x = self.projection(x) # x: [num_nodes, hidden_dim]
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index) # x: [num_nodes, hidden_dim]
            x = self.relu(x) # x: [num_nodes, hidden_dim]
        if target_idx is not None:
            x = x[target_idx] # x: [num_graphs, hidden_dim]
        else:
            x = gnn.global_mean_pool(x, batch) # x: [num_graphs, hidden_dim]
        x = self.fc(x) # x: [num_graphs, embedding_dim]
        return x

class GCNBaseline(nn.Module):
    def __init__(self, num_node_features = 5, hidden_dim = 768, embedding_dim = 512, num_layers= 16, num_attn_heads=None):
        super(GCNBaseline, self).__init__()
        self.projection = nn.Linear(num_node_features, hidden_dim)
        self.convs = nn.ModuleList([
            gnn.GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_dim, embedding_dim)
        self.relu = nn.ReLU()
        
        self.num_layers = num_layers

    def forward(self, x, edge_index, batch, target_idx = None):
        x = self.projection(x) # x: [num_nodes, hidden_dim]
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index) # x: [num_nodes, hidden_dim]
            x = self.relu(x) # x: [num_nodes, hidden_dim]
        if target_idx is not None:
            x = x[target_idx] # x: [num_graphs, hidden_dim]
        else:
            x = gnn.global_mean_pool(x, batch) # x: [num_graphs, hidden_dim]
        x = self.fc(x) # x: [num_graphs, embedding_dim]
        return x

class GINBaseline(nn.Module):
    def __init__(self, num_node_features = 5, hidden_dim = 768, embedding_dim = 512, num_layers= 16, num_attn_heads=None):
        super(GINBaseline, self).__init__()
        self.projection = nn.Linear(num_node_features, hidden_dim)
        self.convs = nn.ModuleList([
            gnn.GINConv(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_dim, embedding_dim)
        self.relu = nn.ReLU()
        
        self.num_layers = num_layers

    def forward(self, x, edge_index, batch, target_idx = None):
        x = self.projection(x) # x: [num_nodes, hidden_dim]
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index) # x: [num_nodes, hidden_dim]
            x = self.relu(x) # x: [num_nodes, hidden_dim]
        if target_idx is not None:
            x = x[target_idx] # x: [num_graphs, hidden_dim]
        else:
            x = gnn.global_mean_pool(x, batch) # x: [num_graphs, hidden_dim]
        x = self.fc(x) # x: [num_graphs, embedding_dim]
        return x

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class LinearLayer(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 use_bias = True,
                 use_bn = False,
                 **kwargs):
        super(LinearLayer, self).__init__(**kwargs)

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.use_bn = use_bn
        
        self.linear = nn.Linear(self.in_features, 
                                self.out_features, 
                                bias = self.use_bias and not self.use_bn)
        if self.use_bn:
             self.bn = nn.BatchNorm1d(self.out_features)

    def forward(self,x):
        x = self.linear(x)
        if self.use_bn:
            x = self.bn(x)
        return x

class ProjectionHead(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 out_features,
                 head_type = 'nonlinear',
                 **kwargs):
        super(ProjectionHead,self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.head_type = head_type

        if self.head_type == 'linear':
            self.layers = LinearLayer(self.in_features,self.out_features,True, False)
        elif self.head_type == 'nonlinear':
            self.layers = nn.Sequential(
                LinearLayer(self.in_features,self.hidden_features,True, True),
                nn.ReLU(),
                LinearLayer(self.hidden_features,self.out_features,True,False))
        
    def forward(self,x):
        x = self.layers(x)
        return x

class Contrastive_Curve(nn.Module):
    def __init__(self, in_channels=1, emb_size=128):
        super().__init__()
        self.convnet =  models.resnet50(pretrained=True)
        self.convnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(1, 1), bias=False, padding="same")
        # self.convnet.maxpool = Identity()
        self.convnet.fc = Identity()
        
        for p in self.convnet.parameters():
            p.requires_grad = True
            
        self.projector = ProjectionHead(2048, 2048, emb_size)

    def forward(self,x):
        
        x = torch.unsqueeze(x,1)

        out = self.convnet(x)
        
        xp = self.projector(torch.squeeze(out))
        
        return xp
    
def Clip_loss(z_i,z_j, temperature = 0.07):
    
    z_i = nn.functional.normalize(z_i)
    z_j = nn.functional.normalize(z_j)
    sim = torch.einsum('i d, j d -> i j', z_i, z_j)/temperature 
    labels = torch.arange(z_i.shape[0], device = z_i.device) 
    loss_t = nn.functional.cross_entropy(sim, labels)
    loss_i = nn.functional.cross_entropy(sim.T, labels)
    
    return (loss_t + loss_i)/2

class ContrastiveTrainLoop:
    def __init__(self, emb_size = 512, curve_size=200, lr = 1e-4, weight_decay=1e-4, cosine_schedule = True, lr_final=1e-5,
                 schedule_max_steps = 100, num_node_features = 5, hidden_dim=768, num_layers=16, num_attn_heads=8, device = None, baseline = None):

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model_input = Contrastive_Curve(emb_size = emb_size).to(self.device)
        self.model_base = Contrastive_Curve(emb_size = emb_size).to(self.device)
        if baseline == "GAT":
            self.model_mechanism = GATBaseline(num_node_features=num_node_features, hidden_dim=hidden_dim * 2, num_layers=num_layers, num_attn_heads=num_attn_heads, embedding_dim=emb_size).to(self.device)
        elif baseline == "GCN":
            self.model_mechanism = GCNBaseline(num_node_features=num_node_features, hidden_dim=hidden_dim * 2, num_layers=num_layers, num_attn_heads=num_attn_heads, embedding_dim=emb_size).to(self.device)
        elif baseline == "GIN":
            self.model_mechanism = GINBaseline(num_node_features=num_node_features, hidden_dim=int(hidden_dim * 1.5) , num_layers=num_layers, num_attn_heads=num_attn_heads, embedding_dim=emb_size).to(self.device)
        else:
            self.model_mechanism = GraphHop(num_node_features=num_node_features, hidden_dim=hidden_dim, num_layers=num_layers, num_attn_heads=num_attn_heads, embedding_dim=emb_size).to(self.device)
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.AdamW(list(self.model_input.parameters()) + list(self.model_base.parameters()) + list(self.model_mechanism.parameters()), lr = lr, weight_decay=weight_decay)

        self.cosine_schedule = cosine_schedule
        self.lr_final = lr_final
        self.schedule_max_steps = schedule_max_steps

        if self.cosine_schedule:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max = schedule_max_steps, eta_min = lr_final)
        
        self.current_epoch = 0
        self.curve_size = curve_size
        

    def reset_optimizer(self):
        self.optimizer = torch.optim.AdamW(list(self.model_input.parameters()) + list(self.model_base.parameters()), lr = self.lr, weight_decay=self.weight_decay)
        if self.cosine_schedule:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max = self.schedule_max_steps, eta_min = self.lr_final)

    def train(self, data, data_mech, batch_size, epochs = 100, temperature = 0.07, continue_loop=True, verbose = True):
        if continue_loop:
            self.model_input.train()
            self.model_base.train()
        else:
            self.model_input.train()
            self.model_base.train()
            self.current_epoch = 0
            self.reset_optimizer()
        self.model_input.train()
        self.model_base.train()

        steps_per_epoch = int(np.ceil(len(data)/batch_size))
        hist = []
        for epoch in range(epochs):
            if verbose:
                prog = tqdm(range(steps_per_epoch))
            else:
                prog = range(steps_per_epoch)

            epoch_loss = 0
            
            shuffle_idx = np.random.permutation(len(data))
            
            for i in prog:
                self.optimizer.zero_grad()
                batch = torch.tensor(data[shuffle_idx[i*batch_size:(i+1)*batch_size]]).float().to(self.device)
                batch_mech = data_mech[shuffle_idx[i*batch_size:(i+1)*batch_size]]
                
                x, edge_index, size = zip(*batch_mech)
                size = np.array(size)
                x = torch.tensor(np.concatenate(x)).float().to(self.device)
                num_edges = np.array([e.shape[1] for e in edge_index])
                edge_index = torch.tensor(np.concatenate(edge_index,-1) + np.repeat(np.cumsum(np.pad(size,[1,0],constant_values=0))[:-1],num_edges).reshape(1,-1)).long().to(self.device)
                b = torch.tensor(np.repeat(np.arange(size.shape[0]),size)).long().to(self.device)
                ext_idx = torch.tensor(np.cumsum(size)-1).long().to(self.device)
                
                base, inp = prep_curves(batch, self.curve_size)

                z_i = self.model_input(inp)
                z_j = self.model_base(base)
                z_k = self.model_mechanism(x, edge_index, b)

                loss_c = Clip_loss(z_i,z_j, temperature)
                loss_m = Clip_loss(z_j,z_k, temperature)
                loss = loss_c + loss_m
                loss.backward()
                self.optimizer.step()
                
                hist.append([loss.item(), loss_c.item(), loss_m.item()])

                epoch_loss += loss.item()

                if verbose:
                    prog.set_postfix_str(f'Loss: {loss.item():.7f}, Epoch Loss: {epoch_loss/(i+1):.7f} || C: {loss_c.item():.7f}, M: {loss_m.item():.7f}')

            self.current_epoch += 1
            if self.cosine_schedule and self.current_epoch <= self.schedule_max_steps:
                self.scheduler.step()

            if verbose:
                print(f'Epoch {self.current_epoch}, Loss: {epoch_loss/steps_per_epoch}')
                
        return hist

    def compute_embeddings_base(self, data, batch_size, verbose = True):
        with torch.no_grad():
            self.model_base.eval()
            embeddings = []
            steps_per_epoch = int(np.ceil(len(data)/batch_size))
            if verbose:
                prog = tqdm(range(steps_per_epoch))
            else:
                prog = range(steps_per_epoch)
            for i in prog:
                batch = torch.tensor(data[i*batch_size:(i+1)*batch_size]).float().to(self.device)
                base, inp = prep_curves(batch, self.curve_size)
                emb = self.model_base(base)
                embeddings.append(emb.detach().cpu().numpy())
            return np.concatenate(embeddings)
        
    def compute_embeddings_input(self, data, batch_size, verbose = False):
        with torch.no_grad():
            self.model_input.eval()
            embeddings = []
            steps_per_epoch = int(np.ceil(len(data)/batch_size))
            if verbose:
                prog = tqdm(range(steps_per_epoch))
            else:
                prog = range(steps_per_epoch)
            for i in prog:
                batch = torch.tensor(data[i*batch_size:(i+1)*batch_size]).float().to(self.device)
                base, inp = prep_curves(batch, self.curve_size)
                emb = self.model_input(inp)
                embeddings.append(emb.detach().cpu().numpy())
            return np.concatenate(embeddings)
        
