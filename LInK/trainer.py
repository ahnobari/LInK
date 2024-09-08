import torch
from torch import nn
import numpy as np
from tqdm.autonotebook import tqdm, trange
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.profiler import profile, record_function, ProfilerActivity
import bitsandbytes as bnb
import torch_optimizer as topt
import os
from .DataUtils import prep_curves
from .nn import Clip_loss

def recon_loss(decode_out, decoder_inps):
    input_ids, positions, mask = decoder_inps
    raw_out, node_type, continious, connectivity_1, connectivity_2 = decode_out

    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    node_type_target = input_ids[:,::4] * mask[:,::4]
    node_type_pred = node_type.transpose(1,2)
    node_type_loss = torch.sum(loss_fn(node_type_pred, node_type_target) * mask[:,::4])/mask[:,::4].sum()

    continious_target = positions * mask[:,1::4].unsqueeze(-1)
    continious_pred = continious * mask[:,1::4].unsqueeze(-1)
    continious_loss = torch.sum((continious_pred - continious_target)**2)/mask[:,1::4].sum()/2

    connectivity_1_target = (input_ids[:,2::4]-4) * mask[:,2::4]
    connectivity_1_pred = connectivity_1.transpose(1,2)
    connectivity_1_loss = torch.sum(loss_fn(connectivity_1_pred, connectivity_1_target) * mask[:,2::4])/mask[:,2::4].sum()

    connectivity_2_target = (input_ids[:,3::4]-4) * mask[:,3::4]
    connectivity_2_pred = connectivity_2.transpose(1,2)
    connectivity_2_loss = torch.sum(loss_fn(connectivity_2_pred, connectivity_2_target) * mask[:,3::4])/mask[:,3::4].sum()

    recon_loss = node_type_loss + continious_loss + connectivity_1_loss + connectivity_2_loss

    return recon_loss, node_type_loss, continious_loss, connectivity_1_loss, connectivity_2_loss


class Trainer:
    def __init__(self, model_input, model_base, model_mechanism, decoder, curve_size=200, lr=1e-4, weight_decay=1e-4, cosine_schedule=True, lr_final=1e-5,
                 schedule_max_steps=100, device=None, multi_gpu=False, mixed_precision=True, DDP_train=True, 
                 Compile=True, checkpoint_path=None, enable_profiling=False, optimizer='AdamW'):
        
        self.multi_gpu = multi_gpu
        self.DDP = DDP_train if multi_gpu else False
        self.mixed_precision = mixed_precision
        self.enable_profiling = enable_profiling
        self.optimizer_name = optimizer

        self.curve_size = curve_size
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.device = device
        self.model_input = model_input
        self.model_base = model_base
        self.model_mechanism = model_mechanism
        self.decoder = decoder

        if Compile:
            if hasattr(self.model_input, 'compile'):
                self.model_input = torch.compile(self.model_input)
            if hasattr(self.model_base, 'compile'):
                self.model_base = torch.compile(self.model_base)
            if hasattr(self.model_mechanism, 'compile'):
                self.model_mechanism = torch.compile(self.model_mechanism)
            # if hasattr(self.decoder, 'compile'):
            #     self.decoder = torch.compile(self.decoder)
        
        if self.DDP:
            if self.enable_profiling:
                with record_function("DDP setup"):
                    self.setup_ddp()
            else:
                self.setup_ddp()
        elif self.multi_gpu and type(self.multi_gpu) is list:
            self.model_input = self.model_input.to(self.device)
            self.model_base = self.model_base.to(self.device)
            self.model_mechanism = self.model_mechanism.to(self.device)
            self.decoder = self.decoder.to(self.device)
            self.model_input = nn.DataParallel(self.model_input, device_ids=multi_gpu)
            self.model_base = nn.DataParallel(self.model_base, device_ids=multi_gpu)
            self.model_mechanism = nn.DataParallel(self.model_mechanism, device_ids=multi_gpu)
            self.decoder = nn.DataParallel(self.decoder, device_ids=multi_gpu)
        elif self.multi_gpu:
            self.model_input = self.model_input.to(self.device)
            self.model_base = self.model_base.to(self.device)
            self.model_mechanism = self.model_mechanism.to(self.device)
            self.decoder = self.decoder.to(self.device)
            self.model_input = nn.DataParallel(self.model_input)
            self.model_base = nn.DataParallel(self.model_base)
            self.model_mechanism = nn.DataParallel(self.model_mechanism)
            self.decoder = nn.DataParallel(self.decoder)
        else:
            self.model_input = self.model_input.to(self.device)
            self.model_base = self.model_base.to(self.device)
            self.model_mechanism = self.model_mechanism.to(self.device)
            self.decoder = self.decoder.to(self.device)
        
        self.lr = lr
        self.weight_decay = weight_decay
        
        if self.enable_profiling:
            with record_function("Optimizer setup"):
                self.setup_optimizer()
        else:
            self.setup_optimizer()
        
        self.cosine_schedule = cosine_schedule
        self.lr_final = lr_final
        self.schedule_max_steps = schedule_max_steps
        
        if self.cosine_schedule:
            if self.enable_profiling:
                with record_function("Cosine Annealing LR Scheduler setup"):
                    self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=schedule_max_steps, eta_min=lr_final)
            else:
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=schedule_max_steps, eta_min=lr_final)
        else:
            self.scheduler = None
        
        self.current_epoch = 0
        
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

        torch.cuda.empty_cache()

    def setup_optimizer(self):
        params = list(self.model_input.parameters()) + list(self.model_base.parameters()) + list(self.model_mechanism.parameters()) + list(self.decoder.parameters())
        if self.optimizer_name == 'Adam':
            self.optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'AdamW':
            self.optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'SGD':
            self.optimizer = torch.optim.SGD(params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'Adam8':
            self.optimizer = bnb.optim.Adam8bit(params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'Adafactor':
            self.optimizer = topt.Adafactor(params, lr=self.lr, weight_decay=self.weight_decay)

    def setup_ddp(self):
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '12355'
        
        self.rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        if self.enable_profiling:
            with record_function("DDP init"):
                dist.init_process_group(backend='nccl')
        else:
            dist.init_process_group(backend='nccl')

        torch.cuda.set_device(self.rank)

        if self.enable_profiling:
            with record_function("DDP model to rank"):
                self.model_input = self.model_input.to(self.rank)
                self.model_base = self.model_base.to(self.rank)
                self.model_mechanism = self.model_mechanism.to(self.rank)
                self.decoder = self.decoder.to(self.rank)
        else:
            self.model_input = self.model_input.to(self.rank)
            self.model_base = self.model_base.to(self.rank)
            self.model_mechanism = self.model_mechanism.to(self.rank)
            self.decoder = self.decoder.to(self.rank)
        
        if self.enable_profiling:
            with record_function("DDP model setup"):
                self.model_input = DDP(self.model_input, device_ids=[self.rank])
                self.model_base = DDP(self.model_base, device_ids=[self.rank])
                self.model_mechanism = DDP(self.model_mechanism, device_ids=[self.rank])
                self.decoder = DDP(self.decoder, device_ids=[self.rank])
        else:
            self.model_input = DDP(self.model_input, device_ids=[self.rank])
            self.model_base = DDP(self.model_base, device_ids=[self.rank])
            self.model_mechanism = DDP(self.model_mechanism, device_ids=[self.rank])
            self.decoder = DDP(self.decoder, device_ids=[self.rank])

    def cleanup_ddp(self):
        if self.DDP:
            dist.destroy_process_group()

    def is_main_process(self):
        return self.rank == 0 if self.DDP else True

    def save_checkpoint(self, path):
        if self.is_main_process():
            checkpoint = {
                'model_input_state_dict': self.model_input.module.state_dict() if isinstance(self.model_input, (nn.DataParallel, DDP)) else self.model_input.state_dict(),
                'model_base_state_dict': self.model_base.module.state_dict() if isinstance(self.model_base, (nn.DataParallel, DDP)) else self.model_base.state_dict(),
                'model_mechanism_state_dict': self.model_mechanism.module.state_dict() if isinstance(self.model_mechanism, (nn.DataParallel, DDP)) else self.model_mechanism.state_dict(),
                'decoder_state_dict': self.decoder.module.state_dict() if isinstance(self.decoder, (nn.DataParallel, DDP)) else self.decoder.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'current_epoch': self.current_epoch,
            }
            if self.scheduler:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(self.model_input, (nn.DataParallel, DDP)):
            self.model_input.module.load_state_dict(checkpoint['model_input_state_dict'])
            self.model_base.module.load_state_dict(checkpoint['model_base_state_dict'])
            self.model_mechanism.module.load_state_dict(checkpoint['model_mechanism_state_dict'])
            self.decoder.module.load_state_dict(checkpoint['decoder_state_dict'])
        else:
            self.model_input.load_state_dict(checkpoint['model_input_state_dict'])
            self.model_base.load_state_dict(checkpoint['model_base_state_dict'])
            self.model_mechanism.load_state_dict(checkpoint['model_mechanism_state_dict'])
            self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:
            if self.is_main_process():
                print("Optimizer state dict not found in checkpoint or incompatible with current optimizer.")

        self.current_epoch = checkpoint['current_epoch']

        try:
            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except:
            if self.is_main_process():
                print("Scheduler state dict not found in checkpoint or incompatible with current scheduler.")

    def reset_optimizer(self):
        self.setup_optimizer()
        
        if self.cosine_schedule:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.schedule_max_steps, eta_min=self.lr_final)
        else:
            self.scheduler = None

    def train(self, batch_load_fn, data_idx, batch_size, epochs=100, temperature=0.07, continue_loop=True, verbose=True, checkpoint_interval=10, checkpoint_dir='Checkpoints'):
        
        if not continue_loop:
            self.model_input.train()
            self.model_base.train()
            self.model_mechanism.train()
            self.decoder.train()
            self.current_epoch = 0
            self.reset_optimizer()
        else:
            self.model_input.train()
            self.model_base.train()
            self.model_mechanism.train()
            self.decoder.train()
        
        if self.mixed_precision:
            scaler = torch.cuda.amp.GradScaler()
        
        torch.cuda.empty_cache()

        # split data for DDP
        if self.DDP:
            data_idx = np.array_split(data_idx, self.world_size)[self.rank]

        steps_per_epoch = int(np.ceil(len(data_idx) / batch_size))
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        hist = []
        for epoch in range(epochs):
            if verbose and self.is_main_process():
                prog = tqdm(range(steps_per_epoch))
            else:
                prog = range(steps_per_epoch)

            epoch_loss = 0
            
            shuffle_idx = np.random.permutation(len(data_idx))
            
            for i in prog:
                self.optimizer.zero_grad()
                
                # Use the batch_load_fn to load the data
                batch, batch_mech, decoder_inps = batch_load_fn(data_idx[shuffle_idx[i*batch_size:(i+1)*batch_size]], self.device)
                
                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        z_i = self.model_input(batch['inp'])
                        z_j = self.model_base(batch['base'])
                        z_k = self.model_mechanism(batch_mech['x'], batch_mech['edge_index'], batch_mech['batch'])
                        decode_out = self.decoder(z_k, decoder_inps)

                        loss_c = Clip_loss(z_i, z_j, temperature)
                        loss_m = Clip_loss(z_j, z_k, temperature)
                        loss_recon, loss_node_type, loss_continious, loss_connectivity_1, loss_connectivity_2 = recon_loss(decode_out, decoder_inps)
                        loss = loss_c + loss_m + loss_recon
                else:
                    z_i = self.model_input(batch['inp'])
                    z_j = self.model_base(batch['base'])
                    z_k = self.model_mechanism(batch_mech['x'], batch_mech['edge_index'], batch_mech['batch'])
                    decode_out = self.decoder(z_k, decoder_inps)

                    loss_c = Clip_loss(z_i, z_j, temperature)
                    loss_m = Clip_loss(z_j, z_k, temperature)
                    loss_recon, loss_node_type, loss_continious, loss_connectivity_1, loss_connectivity_2 = recon_loss(decode_out, decoder_inps)
                    loss = loss_c + loss_m + loss_recon
                
                if self.mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                
                hist.append([loss.item(), loss_c.item(), loss_m.item(), loss_recon.item(), loss_node_type.item(), loss_continious.item(), loss_connectivity_1.item(), loss_connectivity_2.item()])
                epoch_loss += loss.item()
                
                if self.is_main_process() and verbose:
                    prog.set_postfix_str(f'Loss: {loss.item():.7f}, Epoch Loss: {epoch_loss/(i+1):.7f} || C: {loss_c.item():.7f}, M: {loss_m.item():.7f} || R: {loss_recon.item():.7f}, NT: {loss_node_type.item():.7f}, P: {loss_continious.item():.7f}, C1: {loss_connectivity_1.item():.7f}, C2: {loss_connectivity_2.item():.7f}')
            
            self.current_epoch += 1
            if self.cosine_schedule and self.current_epoch <= self.schedule_max_steps:
                self.scheduler.step()
                
            if verbose and self.is_main_process():
                print(f'Epoch {self.current_epoch}, Loss: {epoch_loss/steps_per_epoch}')

            self.save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_epoch_{self.current_epoch}.pth'))

            if (self.current_epoch-1) % checkpoint_interval == 0:
                pass
            elif self.is_main_process():
                os.remove(os.path.join(checkpoint_dir, f'checkpoint_epoch_{self.current_epoch-1}.pth'))

        if self.DDP:
            dist.barrier()

        return hist

    def profile(self, batch_load_fn, data_idx, batch_size, temperature=0.07):
        if not self.enable_profiling:
            if self.is_main_process():
                print("Profiling is not enabled. Set enable_profiling=True in the Trainer initialization to use this feature.")
            return

        self.model_input.train()
        self.model_base.train()
        self.model_mechanism.train()
        self.decoder.train()

        if self.mixed_precision:
            with record_function("Mixed Precision setup"):
                scaler = torch.cuda.amp.GradScaler()
        
        torch.cuda.empty_cache()

        if self.DDP:
            data_idx = np.array_split(data_idx, self.world_size)[self.rank]

        shuffle_idx = np.random.permutation(len(data_idx))
        
        for i in range(5):  # Profile only 5 steps
            with record_function(f"Training step {i} Data Preparation"):
                self.optimizer.zero_grad()
                batch, batch_mech = batch_load_fn(data_idx[shuffle_idx[i*batch_size:(i+1)*batch_size]], self.device)
            
            with record_function(f"Training step {i} Forward"):
                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        z_i = self.model_input(batch['inp'])
                        z_j = self.model_base(batch['base'])
                        z_k = self.model_mechanism(batch_mech['x'], batch_mech['edge_index'], batch_mech['batch'])
                        decode_out = self.decoder(z_k, decoder_inps)

                        loss_c = Clip_loss(z_i, z_j, temperature)
                        loss_m = Clip_loss(z_j, z_k, temperature)
                        loss_recon, loss_node_type, loss_continious, loss_connectivity_1, loss_connectivity_2 = recon_loss(decode_out, decoder_inps)
                        loss = loss_c + loss_m + loss_recon
                else:
                    z_i = self.model_input(batch['inp'])
                    z_j = self.model_base(batch['base'])
                    z_k = self.model_mechanism(batch_mech['x'], batch_mech['edge_index'], batch_mech['batch'])
                    decode_out = self.decoder(z_k, decoder_inps)

                    loss_c = Clip_loss(z_i, z_j, temperature)
                    loss_m = Clip_loss(z_j, z_k, temperature)
                    loss_recon, loss_node_type, loss_continious, loss_connectivity_1, loss_connectivity_2 = recon_loss(decode_out, decoder_inps)
                    loss = loss_c + loss_m + loss_recon
            
            if self.mixed_precision:
                with record_function(f"Training step {i} Backward"):
                    scaler.scale(loss).backward()
                with record_function(f"Training step {i} Optimizer Step"):
                    scaler.step(self.optimizer)
                with record_function(f"Training step {i} Scaler Update"):
                    scaler.update()
            else:
                with record_function(f"Training step {i} Backward"):
                    loss.backward()
                with record_function(f"Training step {i} Optimizer Step"):    
                    self.optimizer.step()

        if self.DDP:
            dist.barrier()