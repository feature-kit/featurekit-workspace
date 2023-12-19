import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dists

import transformers
from interp_utils import get_scheduler

from tqdm import tqdm
import numpy as np

import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'



def get_sparse_weights(n_atoms=2000, l1_min=4e-1, l1_max=2, sharpness=5.0):
    if l1_min == l1_max:
        return l1_min*torch.ones(n_atoms)
    N = n_atoms-1
    beta = N/((l1_max/l1_min)**(sharpness)-1)
    alpha = l1_min/beta**(1/sharpness)
    weights = alpha*(torch.arange(n_atoms)+beta)**(1/sharpness)
    return weights

def inv_perm(perm):
    inv_perm = torch.zeros_like(perm)
    inv_perm[perm] = torch.arange(len(perm)).to(perm.device)
    return inv_perm


class Autoencoder(nn.Module):
    def __init__(self, n_features, d_model):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.encoder = nn.Linear(n_features, d_model)
        self.decoder = nn.Linear(d_model, n_features)
    def forward(self, x):
        hidden_state = self.encoder(x)
        return F.gelu(self.decoder(hidden_state))


class NeuronFiringRateObserver(nn.Module):
    def __init__(self, n_neurons):
        super().__init__()
        self.n_neurons = n_neurons
        self.fire_counts = nn.Parameter(torch.zeros(n_neurons, dtype=torch.long), requires_grad=False)
        self.total_observations = 0
    def observe(self, acts):
        assert len(acts.shape) == 2
        self.fire_counts += (acts.abs() > 2e-2).long().sum(dim=0)
        self.total_observations += acts.shape[0]
    @property
    def firing_rates(self):
        return self.fire_counts.float()/self.total_observations
    @property
    def dead_neurons(self):
        print('min max ratio', self.fire_counts.float().min()/self.fire_counts.max())
        return (self.fire_counts == 0)
    def reset(self):
        self.fire_counts.zero_()
        self.total_observations = 0

from inspect import getframeinfo, stack

# class Timer:
#     def __init__(self, silent=False):
#         self.last_event = torch.cuda.Event(enable_timing=True)
#         self.last_event.record()
#         self.info_to_moving_avg = {}
#         self.silent = silent
#     def print(self, s):
#         if not self.silent:
#             print(s)
#     def log(self, info=None):
#         caller = getframeinfo(stack()[1][0])

#         # torch.cuda.synchronize()
#         now = torch.cuda.Event(enable_timing=True)
#         now.record()
#         now.synchronize()
#         # torch.cuda.synchronize()

#         if info is None:
#             info = f'Line {caller.lineno}'
        
        
#         delta = self.last_event.elapsed_time(now)/1000
#         if info is not None:
#             if info not in self.info_to_moving_avg:
#                 self.info_to_moving_avg[info] = delta
#                 self.print(f'{info} >> current: {delta:.2E}')
#             else:
#                 moving_avg = 0.9*self.info_to_moving_avg[info]+0.1*delta
#                 self.info_to_moving_avg[info] = moving_avg
#                 self.print(f'{info} >> current: {delta:.2e}, avg: {moving_avg:.2E}')
#         else:
#             self.print(f'current: {delta:.2E}')
#         self.last_event = now
#     def reset(self):
#         self.last_event = torch.cuda.Event(enable_timing=True)
#         self.last_event.record()
#         if not self.silent:
#             print()
#     def readout(self):
#         for info, moving_avg in self.info_to_moving_avg.items():
#             print(f'{info}: {moving_avg:.2E}')


from interp_utils import see
class LossDatapointGatherer:
    def __init__(self):
        self.losses = []
        self.examples = []
        self.total_observations = 0
    def observe(self, batch_losses, batch_examples):
        batch_losses = batch_losses.squeeze()
        assert batch_losses.shape[0] == batch_examples.shape[0]
        assert len(batch_examples.shape) == 2
        batch_losses = batch_losses.detach().cpu().tolist()
        batch_examples = list(batch_examples.detach().cpu())
        self.losses.extend(batch_losses)
        self.examples.extend(batch_examples)
        self.total_observations += len(batch_losses)
    def get_examples(self):
        return torch.cat(self.examples, dim=0).to(device)
    def sample(self, k=10):
        unnormed_probs = torch.tensor(self.losses)**2
        probs = unnormed_probs/unnormed_probs.sum()
        # see(probs.shape)
        # see(k)
        idxs = torch.multinomial(probs, num_samples=k, replacement=False)
        return torch.stack([self.examples[idx] for idx in idxs])
    def reset(self):
        self.losses = []
        self.examples = []
        self.total_observations = 0

def get_warmup_scheduler(optimizer, warmup_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            return 1

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

def get_l1_weights(l1_min, l1_max, n_features):
    return (l1_max-l1_min)*torch.arange(n_features)/(n_features-1)+l1_min


# from comet_ml import Experiment

class SparseAutoencoder(nn.Module):
    def __init__(self, n_features, d_model, act=nn.ReLU, lr=1e-4, l1_coef=3e-1, steps_before_dead_reinit=4000, observations_for_reinit=100, accum_iters=1, l1_ratio=1.0, disable_comet=False):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.encoder = nn.Linear(d_model, n_features)
        self.decoder = nn.Linear(n_features, d_model, bias=False)
        self.norm_writeout()
        self.encoder.weight.data = self.decoder.weight.data.T.detach().clone()
        self.encoder.bias.data = -0.01*torch.ones_like(self.encoder.bias.data)

        self.input_bias = nn.Parameter(torch.zeros(d_model))
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.neuron_observer = NeuronFiringRateObserver(n_features)
        self.loss_datapoint_gatherer = LossDatapointGatherer()
        self.act = act()
        self.steps_before_dead_reinit = steps_before_dead_reinit
        self.observations_for_reinit = observations_for_reinit
        self.cycle_count = 0
        self.l1_coef = l1_coef
        self.scheduler = get_warmup_scheduler(self.optimizer, 1000)
        self.accum_iters = accum_iters
        self.device = 'cuda'
        self.first_batch_seen = False

        assert l1_ratio >= 1.0, 'l1_ratio must be >= 1.0'

        l1_weights = get_l1_weights(1, l1_ratio, self.n_features).to(self.device)
        l1_weights = l1_weights/l1_weights.mean()
        self.l1_weights = nn.Parameter(l1_weights, requires_grad=False)

        # if not disable_comet:
        #     self.experiment = Experiment(
        #         api_key="SM1kql4BgAke3LsOprlsxafDm",
        #         project_name="sparse-autoencoder",
        #         workspace="noanabeshima"
        #     )
        #     self.experiment.log_parameters({
        #         'n_features': n_features,
        #         'd_model': d_model,
        #         'act': act,
        #         'lr': lr,
        #         'l1_coef': l1_coef,
        #         'steps_before_dead_reinit': steps_before_dead_reinit,
        #         'observations_for_reinit': observations_for_reinit,
        #         'accum_iters': accum_iters,
        #         'l1_ratio': l1_ratio,
        #     })
        
        self.disable_comet = disable_comet

    def get_preacts(self, x):
        return self.encoder(x - self.input_bias[None])
    def get_acts(self, x):
        return self.act(self.encoder(x - self.input_bias[None]))
    def run(self, x):
        if not self.first_batch_seen:
            self.first_batch_seen = True
            with torch.no_grad():
                self.input_bias.data = x.mean(dim=0).data.to(self.input_bias.dtype)
        acts = self.get_acts(x)
        pred = self.decoder(acts) + self.input_bias[None]
        return pred, acts
    def forward(self, x):
        pred, acts = self.run(x)
        per_eg_mse = ((pred - x)**2).mean(dim=1)
        
        raw_l1 = acts.abs().mean(dim=0)
        
        
        with torch.no_grad():
            perm = acts.mean(dim=0).sort(descending=False).indices
        
        print('Stub! make sure perm is correctly being used.')
            
        l1, base_l1 = (self.l1_weights[None]*raw_l1[perm]).mean(), raw_l1.mean()

        self.cycle_count += 1

        flag = ''

        if self.cycle_count > self.steps_before_dead_reinit//2:
            if self.cycle_count == self.steps_before_dead_reinit//2 + 1:
                print("Halfway to reinit")
            flag += 'halfway'
            self.neuron_observer.observe(acts)
        if self.cycle_count > self.steps_before_dead_reinit:
            flag += ' logging'
            self.loss_datapoint_gatherer.observe(per_eg_mse, x)
        if self.cycle_count > self.steps_before_dead_reinit + self.observations_for_reinit:
            flag += ' reinit'
            print('trying to reinit dead neurons')
            self.reinit_dead_neurons()
            self.cycle_count = 0
            self.neuron_observer.reset()
            self.loss_datapoint_gatherer.reset()

        mse = per_eg_mse.mean()
        loss = mse + self.l1_coef*l1
        loss.backward()

        if self.cycle_count % self.accum_iters == 0:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            self.norm_writeout()

            # if not self.disable_comet:
            #     self.experiment.log_metric('loss', loss.item())
            #     self.experiment.log_metric('mse', mse.item())
            #     self.experiment.log_metric('l1', base_l1.item())            


        
        return loss, per_eg_mse.mean(), base_l1, flag
        
    def reinit_dead_neurons(self):
        dead_neurons = self.neuron_observer.dead_neurons
        if dead_neurons.sum() <= 1:
            print("No dead neurons to reinit")
            return
        else:
            print(f"Reinitializing {dead_neurons.sum()} neurons")
        reinit_vecs = self.loss_datapoint_gatherer.sample(k=int(dead_neurons.sum())).to(self.device) - self.input_bias[None]
        
        normed_reinit_vecs = reinit_vecs/torch.norm(reinit_vecs, dim=1, keepdim=True)

        self.decoder.weight.data[:,dead_neurons] = normed_reinit_vecs.T
        avg_encoder_norm = self.encoder.weight.data[~dead_neurons].norm(dim=1).mean()
        self.encoder.weight.data[dead_neurons,:] = normed_reinit_vecs*0.2*avg_encoder_norm
        self.encoder.bias.data[dead_neurons] = 0.0

        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)
        self.scheduler = get_warmup_scheduler(self.optimizer, 400)
    
    def norm_writeout(self):
        with torch.no_grad():
            self.decoder.weight.div_(torch.norm(self.decoder.weight, dim=1, keepdim=True))

class SparseMLP(nn.Module):
    def __init__(self, n_features, d_model, act=nn.ReLU, lr=1e-4, l1_coef=3e-1, steps_before_dead_reinit=4000, observations_for_reinit=100, accum_iters=1, l1_ratio=1.0, disable_comet=False):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.encoder = nn.Linear(d_model, n_features)
        self.decoder = nn.Linear(n_features, d_model, bias=False)
        # self.norm_writeout()
        # self.encoder.weight.data = self.decoder.weight.data.T.detach().clone()
        # self.encoder.bias.data = -0.01*torch.ones_like(self.encoder.bias.data)

        self.input_bias = nn.Parameter(torch.zeros(d_model))
        self.output_bias = nn.Parameter(torch.zeros(d_model))
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.neuron_observer = NeuronFiringRateObserver(n_features)
        self.loss_datapoint_gatherer = LossDatapointGatherer()
        self.act = act()
        self.steps_before_dead_reinit = steps_before_dead_reinit
        self.observations_for_reinit = observations_for_reinit
        self.cycle_count = 0
        self.l1_coef = l1_coef
        self.scheduler = get_warmup_scheduler(self.optimizer, 1000)
        self.accum_iters = accum_iters
        self.device = device
        self.first_batch_seen = False
        # self.timer = Timer(silent=True)

        assert l1_ratio >= 1.0, 'l1_ratio must be >= 1.0'

        l1_weights = get_l1_weights(1, l1_ratio, self.n_features).to(self.device)
        l1_weights = l1_weights/l1_weights.mean()
        self.l1_weights = nn.Parameter(l1_weights, requires_grad=False)

        # if not disable_comet:
        #     self.experiment = Experiment(
        #         api_key="SM1kql4BgAke3LsOprlsxafDm",
        #         project_name="sparse-autoencoder",
        #         workspace="noanabeshima"
        #     )
        #     self.experiment.log_parameters({
        #         'n_features': n_features,
        #         'd_model': d_model,
        #         'act': act,
        #         'lr': lr,
        #         'l1_coef': l1_coef,
        #         'steps_before_dead_reinit': steps_before_dead_reinit,
        #         'observations_for_reinit': observations_for_reinit,
        #         'accum_iters': accum_iters,
        #         'l1_ratio': l1_ratio,
        #     })
        
        self.disable_comet = disable_comet

    def get_preacts(self, x):
        return self.encoder(x - self.input_bias[None])
    def get_acts(self, x):
        return self.act(self.get_preacts(x))
    def run(self, x, y):
        if not self.first_batch_seen:
            self.first_batch_seen = True
            with torch.no_grad():
                self.input_bias.data = x.mean(dim=0).data.to(self.input_bias.dtype)
                self.output_bias.data = y.mean(dim=0).data.to(self.output_bias.dtype)
        acts = self.get_acts(x)
        pred = self.decoder(acts) + self.output_bias[None]
        return pred, acts
    def forward(self, x, y):
        # self.timer.reset()
        
        pred, acts = self.run(x, y)
        # self.timer.log()
        per_eg_mse = ((pred - y)**2).mean(dim=1)
        # self.timer.log()
        raw_l1 = acts.abs().mean(dim=0)
        # self.timer.log()
        
        with torch.no_grad():
            perm = acts.mean(dim=0).sort(descending=False).indices            
        # self.timer.log()
        l1, base_l1 = (self.l1_weights[None]*raw_l1[perm]).mean(), raw_l1.mean()
        # self.timer.log()

        self.cycle_count += 1

        flag = ''

        if self.cycle_count > self.steps_before_dead_reinit//2:
            if self.cycle_count == self.steps_before_dead_reinit//2 + 1:
                print("Halfway to reinit")
            flag += 'halfway'
            self.neuron_observer.observe(acts)
        if self.cycle_count > self.steps_before_dead_reinit:
            flag += ' logging'
            self.loss_datapoint_gatherer.observe(per_eg_mse, x)
        if self.cycle_count > self.steps_before_dead_reinit + self.observations_for_reinit:
            flag += ' reinit'
            print('trying to reinit dead neurons')
            self.reinit_dead_neurons()
            self.cycle_count = 0
            self.neuron_observer.reset()
            self.loss_datapoint_gatherer.reset()

        mse = per_eg_mse.mean()
        loss = mse + self.l1_coef*l1
        loss.backward()

        if self.cycle_count % self.accum_iters == 0:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            self.norm_writeout()

            # if not self.disable_comet:
            #     self.experiment.log_metric('loss', loss.item())
            #     self.experiment.log_metric('mse', mse.item())
            #     self.experiment.log_metric('l1', base_l1.item())            

        return loss, per_eg_mse.mean(), base_l1, flag


    def reinit_dead_neurons(self):
        dead_neurons = self.neuron_observer.dead_neurons
        if dead_neurons.sum() == 0:
            print("No dead neurons to reinit")
            return
        else:
            print(f"Reinitializing {dead_neurons.sum()} neurons")
        reinit_vecs = self.loss_datapoint_gatherer.sample(k=int(dead_neurons.sum())).to(self.device) - self.input_bias[None]
        
        normed_reinit_vecs = reinit_vecs/torch.norm(reinit_vecs, dim=1, keepdim=True)

        self.decoder.weight.data[:,dead_neurons] = normed_reinit_vecs.T
        avg_encoder_norm = self.encoder.weight.data[~dead_neurons].norm(dim=1).mean()
        self.encoder.weight.data[dead_neurons,:] = normed_reinit_vecs*0.2*avg_encoder_norm
        self.encoder.bias.data[dead_neurons] = 0.0

        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.scheduler = get_warmup_scheduler(self.optimizer, 400)
    
    def norm_writeout(self):
        with torch.no_grad():
            self.decoder.weight.div_(torch.norm(self.decoder.weight, dim=0, keepdim=True))



class SimpleSparseMLP(nn.Module):
    def __init__(self, n_features, d_model, act=nn.ReLU, l1_ratio=1.0, l1_coef=3e-1):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.encoder = nn.Linear(d_model, n_features)
        self.decoder = nn.Linear(n_features, d_model, bias=False)
        self.norm_writeout()

        self.input_bias = nn.Parameter(torch.zeros(d_model))
        self.output_bias = nn.Parameter(torch.zeros(d_model))
        self.act = act()
        self.l1_coef = l1_coef
        self.first_batch_seen = False

        assert l1_ratio >= 1.0, 'l1_ratio must be >= 1.0'

        l1_weights = get_l1_weights(1, l1_ratio, self.n_features)
        l1_weights = l1_weights/l1_weights.mean()
        self.l1_weights = nn.Parameter(l1_weights, requires_grad=False)

        self.l1_ratio = l1_ratio

    def get_preacts(self, x, indices=None):
        if indices is not None:
            if isinstance(indices, int):
                indices = slice(indices, indices+1)
            if isinstance(indices, slice):
                indices = [indices]
            return (x - self.input_bias[None]) @ self.encoder.weight[*indices].T + self.encoder.bias[*indices][None]
        else:
            return self.encoder(x - self.input_bias[None])
    def get_acts(self, x, indices=None):
        return self.act(self.get_preacts(x, indices=indices))
    def forward(self, x, return_acts=False):
        acts = self.get_acts(x)
        pred = self.decoder(acts) + self.output_bias[None]
        if return_acts:
            return pred, acts
        return pred
        
    def get_losses(self, x, y):
        if not self.first_batch_seen:
            self.first_batch_seen = True
            with torch.no_grad():
                self.input_bias.data = x.mean(dim=0).data.to(self.input_bias.dtype)
        pred, acts = self.forward(x, return_acts=True)

        per_eg_mse = ((pred - y)**2).mean(dim=1)
        raw_l1 = acts.abs().mean(dim=0)
        
        with torch.no_grad():
            perm = acts.mean(dim=0).sort(descending=False).indices            
        l1, base_l1 = (self.l1_weights[None]*raw_l1[perm]).mean(), raw_l1.mean()

        mse = per_eg_mse.mean()
        loss = mse + self.l1_coef*l1

        return loss, mse, base_l1


    def reinit_dead_neurons(self):
        dead_neurons = self.neuron_observer.dead_neurons
        if dead_neurons.sum() == 0:
            print("No dead neurons to reinit")
            return
        else:
            print(f"Reinitializing {dead_neurons.sum()} neurons")
        reinit_vecs = self.loss_datapoint_gatherer.sample(k=int(dead_neurons.sum())).to(self.device) - self.input_bias[None]
        
        normed_reinit_vecs = reinit_vecs/torch.norm(reinit_vecs, dim=1, keepdim=True)

        self.decoder.weight.data[:,dead_neurons] = normed_reinit_vecs.T
        avg_encoder_norm = self.encoder.weight.data[~dead_neurons].norm(dim=1).mean()
        self.encoder.weight.data[dead_neurons,:] = normed_reinit_vecs*0.2*avg_encoder_norm
        self.encoder.bias.data[dead_neurons] = 0.0

        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.scheduler = get_warmup_scheduler(self.optimizer, 400)
    
    def norm_writeout(self):
        with torch.no_grad():
            self.decoder.weight.div_(torch.norm(self.decoder.weight, dim=0, keepdim=True))
    

class SparseNNMF(nn.Module):
    def __init__(self, n_features, d_model, orthog_k=0, bias=False, disable_tqdm=False, l1_ratio=1.0, l1_sharpness=5.0):
        super().__init__()
        assert isinstance(orthog_k, int) or orthog_k is False
        if orthog_k != 0:
            assert orthog_k > 1, 'orthog_k must be > 1'
            self.orthog_mask = nn.Parameter(1-torch.eye(orthog_k), requires_grad=False)
        self.n_features = n_features
        self.d_model = d_model
        self.unsigned_codes = None
        self.atoms = nn.Parameter(torch.randn(n_features, d_model)/np.sqrt(d_model))
        self.orthog_k = orthog_k

        self.bias = nn.Parameter(torch.zeros(d_model)) if bias else None

        self.disable_tqdm = disable_tqdm


        assert l1_ratio >= 1.0
        sparse_weights = get_sparse_weights(n_atoms=n_features, l1_min=1, l1_max=l1_ratio, sharpness=l1_sharpness).to(self.atoms.device)
        sparse_weights = sparse_weights/sparse_weights.mean()
        self.sparse_weights = nn.Parameter(sparse_weights, requires_grad=False)

        # self.timer = Timer(silent=True)

        self.norm_atoms()

    def codes(self, codes_subset=None):
        if codes_subset is not None:
            return self.unsigned_codes[codes_subset].abs()
        else:
            return self.unsigned_codes.abs()

    def forward(self, frozen_codes=False, frozen_atoms=False, codes_subset=None):
        if codes_subset is not None:
            codes = self.codes(codes_subset).detach() if frozen_codes else self.codes(codes_subset)
        else:
            codes = self.codes().detach() if frozen_codes else self.codes()
        atoms = self.atoms.detach() if frozen_atoms else self.atoms

        pred = codes @ atoms

        if self.bias is not None:
            pred += self.bias[None]

        return pred, codes

    def norm_atoms(self):
        # self.timer.reset()
        with torch.no_grad():
            self.atoms.data = F.normalize(self.atoms.data, dim=1)
        # self.timer.log('atom normalization')
    
    def train(self, train_data, n_epochs=1000, lr=1e-2, sparse_coef = 1, frozen_codes=False, frozen_atoms=False, reinit_codes=False, orthog_coef=0., mean_init=False, alt_sparse_loss=False):
        if self.bias is not None and mean_init is True:
            self.bias.data = train_data.mean(dim=0)
        
        if self.orthog_k != 0 and not frozen_atoms:
            assert orthog_coef > 0, 'orthog_coef must be > 0'
        if reinit_codes or self.unsigned_codes is None or self.unsigned_codes.shape[0] != train_data.shape[0]:

            if self.unsigned_codes is not None and self.unsigned_codes.shape[0] != train_data.shape[0]:
                print('reinitializing codes because train_data size changed')
            self.unsigned_codes = nn.Parameter(torch.randn(train_data.shape[0], self.n_features, device=self.atoms.device, dtype=self.atoms.dtype)/np.sqrt(self.n_features))
        
        self.alt_sparse_loss = alt_sparse_loss
        
        
        optimizer = optim.Adam(self.parameters(), lr=lr)

        scheduler = get_scheduler(optimizer, n_epochs)
    
        pbar = tqdm(range(n_epochs)) if not self.disable_tqdm else range(n_epochs)
        for i in pbar:
            # self.timer.reset()
            pred, codes = self(frozen_codes=frozen_codes, frozen_atoms=frozen_atoms)
            # self.timer.log('get pred, codes')

            mse_loss = F.mse_loss(pred, train_data.data)

            
            loss = mse_loss

            # self.timer.reset()
            if not frozen_codes:
                # self.timer.reset()
                code_mean_across_batch = codes.mean(dim=0)
                with torch.no_grad():
                    perm = code_mean_across_batch.sort(descending=True).indices
                    raw_sparse = code_mean_across_batch.mean()

                # self.timer.log('calculate perm')

                sparse_loss = (code_mean_across_batch[perm] * self.sparse_weights).mean()
                # self.timer.log('final sparse loss calculation')
                loss += sparse_coef*sparse_loss
                # self.timer.log('add sparse loss to total loss')

            
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            scheduler.step()
            # self.timer.log('optimizer step')

            if i % 100 == 0:
                loss_string = f'loss: {loss.item():.1E}, mse: {mse_loss.item():.1E}'
                if not frozen_codes:
                    loss_string += f', sparse: {raw_sparse.item():.1E}'
                # if self.orthog_k != 0 and not frozen_atoms:
                #     loss_string += f', orthog: {orthog_loss.item():.3f}'
                
                if not self.disable_tqdm:
                    pbar.set_description(loss_string)

            if not frozen_atoms:
                self.norm_atoms()
            
        return mse_loss.item(), sparse_loss.item()