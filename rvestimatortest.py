import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
from sklearn.model_selection import ParameterGrid
import pandas as pd
import time
import os
import argparse
import matplotlib.pyplot as plt
import pickle

def torch_interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor, dim: int=-1, extrapolate: str='constant') -> torch.Tensor:
    """One-dimensional linear interpolation between monotonically increasing sample
    points, with extrapolation beyond sample points.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    Args:
        x: The :math:`x`-from torchcubicspline import natural_cubic_spline_coeffs values, same size as `x`.
    """
    # Move the interpolation dimension to the last axis
    x = x.movedim(dim, -1).contiguous()
    xp = xp.movedim(dim, -1).contiguous()
    fp = fp.movedim(dim, -1).contiguous()
    
    m = torch.diff(fp) / torch.diff(xp) # slope
    b = fp[..., :-1] - m * xp[..., :-1] # offset
    indices = torch.searchsorted(xp, x, right=False)
    
    if extrapolate == 'constant':
        # Pad m and b to get constant values outside of xp range
        m = torch.cat([torch.zeros_like(m)[..., :1], m, torch.zeros_like(m)[..., :1]], dim=-1)
        b = torch.cat([fp[..., :1], b, fp[..., -1:]], dim=-1)
    else: # extrapolate == 'linear'
        indices = torch.clamp(indices - 1, 0, m.shape[-1] - 1)

    values = m.gather(-1, indices) * x + b.gather(-1, indices)
    
    return values.movedim(-1, dim)


class SpectrumDataset(Dataset):

    """
        Charge les spectres du dataset du rv-datachallenge Plato.

        ----- Paramètres d'initialisation -----

        * n_pixel = correspond à L dans AESTRA -> Nb d'échantillons
                                                  Dans AESTRA : 2000

        * lambda_min, lambda_max               -> Bornes inf et sup du domaine du spectre, si None on prend ceux du spectre d'origine
                                                  Dans AESTRA : 5000 - 5050 Å

        
        ----- Variables internes accessibles ----- 

        * self.n_spec, self.n_pixel

        * self.all_specs_numpy         -> ndarray de taille (n_spec, n_pixel)        dtype=float32

        * self.wave_numpy              -> ndarray de taille (n_pixel, )              dtype=float32

        * self.template_spec_numpy     -> ndarray de taille (n_pixel, )              dtype=float32
        
        * self.all_specs_torch         -> tensor de taille [n_spec, n_pixel]         dtype=float32

        * self.wave_torch              -> tensor de taille [n_pixel]                 dtype=float32

        * self.template_spec_torch     -> tensor de taille [n_pixel]                 dtype=float32

        * self.c

        * self.wavelength_step         -> Pas de la grille de longueur d'onde        dtype=float


               
    """
    def __init__(self, n_pixel = 2000, lambda_min=5000., lambda_max=5050.):
        
        # On charge le fichier Analyse_material, celui-ci contient le spectre template et la grille de longueur d'ondes
        with open('STAR1134_HPN_Analyse_material.p', 'rb') as f:
            analyse_material_data = pickle.load(f)

        wave_numpy = analyse_material_data['wave'].to_numpy(dtype='float64') # -> dtype=float64
        specs_numpy = np.load('STAR1134_HPN_flux_YVA.npy').astype('float64') # -> De taille (n_spec, n_pixel) / dtype=float64

        # On peut choisir de crop ou non le spectre sur certaines zones de longueur d'ondes
        if lambda_min is None:
            lambda_min = wave_numpy.min()
        if lambda_max is None:
            lambda_max = wave_numpy.max()

        # Mask pour crop
        wave_mask = (wave_numpy >= lambda_min) & (wave_numpy <= lambda_max)

        # On crop
        wave_numpy = wave_numpy[wave_mask]
        specs_numpy = specs_numpy[:, wave_mask]

        # On récupère le spectre template que l'on crop aussi
        template_spec_numpy = analyse_material_data['stellar_template'].to_numpy(dtype='float64') # -> dtype=float64
        template_spec_numpy = template_spec_numpy[wave_mask] 

        # Réechantillonage des spectres sur une grille régulière si n_pixel est non nul ou non None
        if n_pixel:
            resampled_specs_numpy = []

            wave_new_numpy = np.linspace(lambda_min, lambda_max, n_pixel, dtype='float64') # Grille régulière
            for spec in specs_numpy:
                spec_resampled = np.interp(wave_new_numpy, wave_numpy, spec)
                resampled_specs_numpy.append(spec_resampled)
            
            resampled_specs_numpy = np.array(resampled_specs_numpy, dtype='float64')

            self.all_specs_numpy = resampled_specs_numpy
            self.wave_numpy = wave_new_numpy
            template_spec_numpy = np.interp(wave_new_numpy, wave_numpy, template_spec_numpy) # Réechantillonage du spectre template aussi
            self.template_spec_numpy = template_spec_numpy.astype('float64')

        else:
            # Sinon on garde les données initiales
            self.all_specs_numpy = specs_numpy
            self.wave_numpy = wave_numpy
            self.template_spec_numpy = template_spec_numpy
        
        # Conversion en torch (dtype = float64)
        self.all_specs_torch = torch.from_numpy(self.all_specs_numpy)
        self.wave_torch = torch.from_numpy(self.wave_numpy)
        self.template_spec_torch = torch.from_numpy(self.template_spec_numpy)

        # Données supplémentaires utiles sur le dataset:
        self.n_spec, self.n_pixel = self.all_specs_numpy.shape

        self.c = 299_792_458.0
        self.c_tensor = torch.tensor([299_792_458.0], dtype=torch.float64)

        self.wavelength_step = (lambda_max - lambda_min) / (self.n_pixel)

    # Renvoie la longueur du dataset (n_spec)
    def __len__(self):
        return self.n_spec
    
    # Affiche les infos du dataset avec print()
    def __str__(self):
        return (f"-- Dataset de {self.n_spec} spectres de {self.n_pixel} pixels --\n-- λmin = {self.wave_numpy.min()}, λmax = {self.wave_numpy.max()} --\n-- Pas : {self.wavelength_step} λ --")
    
    def __getitem__(self, index):
        batch_yobs = self.all_specs_torch[index, :] # Renvoie un batch [B, n_pixel] dtype = torch.float64

        return batch_yobs, index

    # Shift un batch de spectre avec un batch de vitesses d'offset données
    def doppler_shift_batch(self, batch_yobs, batch_voffset, interp_method='np.interp'):
        """
            Simule un shift doppler pour un batch de spectre de dim [B, n_pixel] et un batch de vitesse d'offset de taille [B]
            Retourne un batch de spectres augmentés de taille [B, n_pixel]

            On dispose de différentes d'interpolation : 

             * np.interp -> Méthode linéaire classique mais plus lente que torch_interp qui fait pareil
             * scipy.interpolate.interp1d -> Interpolation cubique
             * torch_interp -> Méthode linéaire parrallélisée -> gain de perf
             * cubic_transform -> Le plus lent, Méthode utilisée par AESTRA

        """

        # Méthode numpy classique -> interpolation linéaire et valeurs de frontières fixées constantes -> Meilleure option en terme de perf pour de petits batch
        if interp_method == 'np.interp':
            # On va stocker les spectres shiftés dans batch_yaug de taille [B, n_pixel]
            batch_yaug = np.zeros(shape=batch_yobs.shape, dtype='float32')

            for i, yobs_i in enumerate(batch_yobs):                 # yobs_i est un tenseur de taille [n_pixel] de dtype=float32
                
                yobs_i_numpy = yobs_i.numpy()                       # ndarray de taille (n_pixel, ) 
                voffset_i = batch_voffset[i].numpy()                # float32

                doppler_factor = np.sqrt((1 - voffset_i/self.c) / (1 + voffset_i/self.c), dtype='float32') # float32

                wave_shifted_numpy = self.wave_numpy * doppler_factor


                yaug_i_numpy = np.interp(self.wave_numpy, wave_shifted_numpy, yobs_i_numpy)

                
                batch_yaug[i, :] = yaug_i_numpy
            
            batch_yaug = torch.from_numpy(batch_yaug) # De taille [B, n_pixel] de type float32 -> Ok
    
            return batch_yaug

        # Méthode scipy -> interpolation cubique et valeurs de frontières extrapolées
        elif interp_method == 'scipy.interpolate.interp1d':
            # On va stocker les spectres shiftés dans batch_yaug de taille [B, n_pixel]
            batch_yaug = np.zeros(shape=batch_yobs.shape, dtype='float32')

            for i, yobs_i in enumerate(batch_yobs):                 # yobs_i est un tenseur de taille [n_pixel] de dtype=float32
                
                yobs_i_numpy = yobs_i.numpy()                       # ndarray de taille (n_pixel, ) 
                voffset_i = batch_voffset[i].numpy()                # float32

                doppler_factor = np.sqrt((1 - voffset_i/self.c) / (1 + voffset_i/self.c), dtype='float32') # float32

                wave_shifted_numpy = self.wave_numpy * doppler_factor

                interpolator = interp1d(wave_shifted_numpy, yobs_i, kind='cubic', bounds_error=False, fill_value="extrapolate" ,assume_sorted=True)

                yaug_i_numpy = interpolator(self.wave_numpy)

                batch_yaug[i, :] = yaug_i_numpy
            
            batch_yaug = torch.from_numpy(batch_yaug) # De taille [B, n_pixel] de type float32 -> Ok
    
            return batch_yaug

        # Méthode parallélisée -> interpolation linéaire et valeurs de frontières fixées constantes
        elif interp_method == 'torch_interp':

            # On travaille par batch
            B = batch_yobs.shape[0]
            
            batch_wave = self.wave_torch.unsqueeze(0) # [n_pixel] -> [1, n_pixel]
            batch_wave = batch_wave.expand(B, -1) # [B, n_pixel]

            batch_doppler_factor = torch.sqrt( (1 - batch_voffset/self.c_tensor) /  (1 + batch_voffset/self.c_tensor)) # [B]
            batch_doppler_factor = batch_doppler_factor.unsqueeze(-1) # [B, 1] pour pouvoir faire du [B, n_pixel] * [B, 1] par la suite 
            
            batch_wave_shifted = batch_wave * batch_doppler_factor # Ressort en double

            batch_yaug = torch_interp(batch_wave, batch_wave_shifted, batch_yobs)

            return batch_yaug
        
        # Méthode utilisée dans AESTRA -> un peu plus rapide que scipy.interpolate.interp1d() sur de gros batch > 1000 mais sinon très lent
        elif interp_method == 'cubic_transform':
           
            # Cubic transform attend 3 arguments : 
            #   - x_rest = tenseur de la grille de longueurs d'onde de taille    [n_pixel]
            #   - y_rest = batch de spectres de taille                           [B, n_pixel]
            #   - wave_shifted = batch des grilles de longueurs d'onde shiftée   [B, n_pixel] 
            

            # On travaille par batch
            B = batch_yobs.shape[0]

            batch_wave = self.wave_torch.unsqueeze(0) # [n_pixel] -> [1, n_pixel]
            batch_wave = batch_wave.expand(B, -1) # [B, n_pixel]

            # Conversion en double car besoin d'une grande précision
            batch_voffset_double = batch_voffset.double() 

            batch_doppler_factor = torch.sqrt( (1 - batch_voffset_double/self.c_tensor) /  (1 + batch_voffset_double/self.c_tensor)) # [B]
            batch_doppler_factor = batch_doppler_factor.unsqueeze(-1) # [B, 1] pour pouvoir faire du [B, n_pixel] * [B, 1] par la suite 
            
            batch_wave_shifted = batch_wave * batch_doppler_factor

            batch_wave_shifted = batch_wave_shifted.float() 


            batch_yaug = cubic_transform(
                xrest=self.wave_torch,
                yrest=batch_yobs,
                wave_shifted=batch_wave_shifted
            )

            return batch_yaug
            
    # Plot un spec du dataset
    def plot_spec(self, index = None, with_template=False):
        
        # Si on ne précise pas quel spectre plot on en prend un au hasard
        if index is None:
            index = np.random.randint(0, self.n_spec)
        
        spec_to_plot = self.all_specs_numpy[index, :]

        plt.figure(figsize=(18, 6))
        plt.title('Exemple de spectre')
        if with_template:
            plt.plot(self.wave_numpy, self.template_spec_numpy, linestyle='dashed', color='grey', label='Template')
        plt.plot(self.wave_numpy, spec_to_plot, label=f'Spectre n°{index}')
        plt.legend()
        plt.grid(alpha=0.5)
        plt.show()

        plt.figure(figsize=(18, 6))
        plt.title('(Zoom) Exemple de spectre')
        if with_template:
            plt.plot(self.wave_numpy, self.template_spec_numpy, linestyle='dashed', color='grey', label='Template')
        plt.plot(self.wave_numpy, spec_to_plot, label=f'Spectre n°{index}')
        plt.xlim(5012, 5013)
        plt.legend()
        plt.grid(alpha=0.5)
        plt.show()

    # Plot un spec du dataset ainsi qu'un shift doppler pur de celui-ci
    def plot_doppler(self, index = None, v_offset=950, plot_edges=False, interp_method='np.interp'):
        if index is None:
            index = np.random.randint(0, self.n_spec)
        
        spec_to_plot = self.all_specs_numpy[index, :]
        
        # Il faut le mettre sous forme de batch pour le rentrer dans la fonction doppler_shift_batch
        batched_spec_to_plot = self.all_specs_torch[index, :].unsqueeze(0) # Le unsqueeze rajoute la dimension du batch au début 
        batched_voffset = torch.tensor([v_offset])

        batch_yaug = self.doppler_shift_batch(batched_spec_to_plot, batched_voffset, interp_method)

        shifted_spec_to_plot = batch_yaug.cpu().squeeze().numpy()

        plt.figure(figsize=(18, 6))
        plt.title(f'Exemple de spectre shifté v={v_offset} m/s')
        plt.plot(self.wave_numpy, spec_to_plot, label='Spectre original')
        plt.plot(self.wave_numpy, shifted_spec_to_plot, label=f'Spectre Shifté voffset = {v_offset}')
        plt.legend()
        plt.grid(alpha=0.5)
        plt.show()

        # L'interpolation peut causer des problèmes au niveau des bords : plot_edges permet de visualiser les 10 premiers/derniers points
        if plot_edges:
            plt.figure(figsize=(18, 6))
            plt.title(f'(Zoom) Bord Gauche v={v_offset} m/s')
            plt.plot(self.wave_numpy, spec_to_plot, label='Spectre original', marker="x")
            plt.plot(self.wave_numpy, shifted_spec_to_plot, label=f'Spectre Shifté voffset = {v_offset}', marker="o")
            plt.xlim(self.wave_numpy.min() - 1 * self.wavelength_step, self.wave_numpy.min() + 10 * self.wavelength_step)
            plt.legend()
            plt.grid(alpha=0.5)
            plt.show()

            plt.figure(figsize=(18, 6))
            plt.title(f'(Zoom) Bord Droit v={v_offset} m/s')
            plt.plot(self.wave_numpy, spec_to_plot, label='Spectre original', marker="x")
            plt.plot(self.wave_numpy, shifted_spec_to_plot, label=f'Spectre Shifté voffset = {v_offset}', marker="o")
            plt.xlim(self.wave_numpy.max() - 10 * self.wavelength_step, self.wave_numpy.max() + 1 * self.wavelength_step)
            plt.legend()
            plt.grid(alpha=0.5)
            plt.show()
    
class MLP(nn.Module):
    def __init__(self,
                 n_in,
                 n_out,
                 n_hidden=(16, 16, 16),
                 act=(nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU()),
                 dropout=0):
        super(MLP, self).__init__()

        layer = []
        n_ = [n_in, *n_hidden, n_out]
        for i in range(len(n_)-1):
                layer.append(nn.Linear(n_[i], n_[i+1]))
                layer.append(act[i])
                layer.append(nn.Dropout(p=dropout))
        self.mlp = nn.Sequential(*layer)

    def forward(self, x):
        return self.mlp(x)

class RVEstimator(nn.Module):
    def __init__(self,
                 n_in,
                 sizes = [5,10],
                 n_hidden=(128, 64, 32),
                 act=(nn.PReLU(128),nn.PReLU(64),nn.PReLU(32), nn.Identity()),
                 dropout=0):
        super(RVEstimator, self).__init__()

        filters = [128,64]
        self.conv1,self.conv2 = self._conv_blocks(filters, sizes, dropout=dropout)
        self.n_feature = filters[-1] * ((n_in //sizes[0])//sizes[1])

        self.pool1, self.pool2 = tuple(nn.MaxPool1d(s) for s in sizes[:2])
        print("self.n_feature:",self.n_feature)
        self.mlp = MLP(self.n_feature, 1, n_hidden=n_hidden, act=act, dropout=dropout)
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=-1)

    def _conv_blocks(self, filters, sizes, dropout=0):
        convs = []
        for i in range(len(filters)):
            f_in = 1 if i == 0 else filters[i-1]
            f = filters[i]
            s = sizes[i]
            p = s // 2
            conv = nn.Conv1d(in_channels=f_in,
                             out_channels=f,
                             kernel_size=s,
                             padding=p,
                            )
            norm = nn.InstanceNorm1d(f)
            act = nn.PReLU(num_parameters=f)
            drop = nn.Dropout(p=dropout)
            convs.append(nn.Sequential(conv, norm, act, drop))
        return tuple(convs)

    def forward(self, x):
        # compression
        x = x.unsqueeze(1)
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.softmax(x)
        x = self.flatten(x)
        x = self.mlp(x)
        return x
    
def train_one_epoch(model, dataloader, dataset, optimizer, sigmav, interp_method, device):
    model.train()
    total_loss = 0.0
    for yobs, _ in dataloader:
        B = yobs.size(0)
        # random offsets
        voff = torch.zeros(B).uniform_(-3, 3)
        yobs = yobs.to(device)
        voff = voff.to(device)
        yaug = dataset.doppler_shift_batch(yobs, voff, interp_method)
        noise = torch.randn_like(yaug) * 1e-3
        mask = (torch.rand_like(yaug) > 0.2)
        noise[mask] = 0
        yaug = yaug + noise

        b_obs = dataset.template_spec_torch.unsqueeze(0).to(device)
        robs = (yobs - b_obs).float()
        raug = (yaug - b_obs).float()

        vobs_pred = model(robs)
        vaug_pred = model(raug)
        voff_pred = vaug_pred - vobs_pred
        voff = voff.unsqueeze(-1)
        loss = (1/sigmav)**2 * F.mse_loss(voff_pred, voff)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * B
    return total_loss / len(dataloader.dataset)


def eval_model(model, dataloader, dataset, sigmav, interp_method, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for yobs, _ in dataloader:
            B = yobs.size(0)
            voff = torch.zeros(B).uniform_(-3, 3)
            yobs = yobs.to(device)
            voff = voff.to(device)
            yaug = dataset.doppler_shift_batch(yobs, voff, interp_method)
            noise = torch.randn_like(yaug) * 1e-3
            mask = (torch.rand_like(yaug) > 0.2)
            noise[mask] = 0
            yaug = yaug + noise

            b_obs = dataset.template_spec_torch.unsqueeze(0).to(device)
            robs = (yobs - b_obs).float()
            raug = (yaug - b_obs).float()

            vobs_pred = model(robs)
            vaug_pred = model(raug)
            voff_pred = vaug_pred - vobs_pred
            voff = voff.unsqueeze(-1)
            loss = (1/sigmav)**2 * F.mse_loss(voff_pred, voff)
            total_loss += loss.item() * B
    return total_loss / len(dataloader.dataset)


def run_experiment(config, device):
    # Unpack config
    n_pixel = config['n_pixel']
    lambda_min = config['lambda_min']
    lambda_max = config['lambda_max']
    batch_size = config['batch_size']
    lr = config['lr']
    n_epochs = config['n_epochs']
    dropout = config['dropout']
    hidden = tuple(config['mlp_hidden'])
    conv_sizes = config['conv_sizes']
    sigmav = config['sigmav']

    # Dataset
    dataset = SpectrumDataset(n_pixel=n_pixel, lambda_min=lambda_min, lambda_max=lambda_max)
    train_len = int(0.8 * len(dataset))
    test_len = len(dataset) - train_len
    train_set, test_set = random_split(dataset, [train_len, test_len])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    # Model
    model = RVEstimator(n_in=dataset.n_pixel,
                        sizes=conv_sizes,
                        n_hidden=hidden,
                        dropout=dropout).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    interp_method = config.get('interp_method', 'torch_interp')

    # Training loop
    history = {'epoch': [], 'train_loss': [], 'test_loss': []}
    start_time = time.time()

    # Pre-move static tensors
    dataset.wave_torch = dataset.wave_torch.to(device)
    dataset.template_spec_torch = dataset.template_spec_torch.to(device)
    dataset.c_tensor = dataset.c_tensor.to(device)

    for epoch in range(1, n_epochs+1):
        train_loss = train_one_epoch(model, train_loader, dataset, optimizer, sigmav, interp_method, device)
        test_loss = eval_model(model, test_loader, dataset, sigmav, interp_method, device)
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        print(f"Config {config['name']} | Epoch {epoch}/{n_epochs} -> train_loss: {train_loss:.4f}, test_loss: {test_loss:.4f}")

    duration = time.time() - start_time
    # Results summary
    result = {
        'name': config['name'],
        'duration_s': duration,
        'best_test_loss': min(history['test_loss']),
        'final_train_loss': history['train_loss'][-1],
        'final_test_loss': history['test_loss'][-1]
    }
    # Merge config and result
    return {**config, **result}


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Hyperparameter grid
    grid = [
        {
            'name': 'base',
            'n_pixel': 2000,
            'lambda_min': 5000,
            'lambda_max': 5050,
            'batch_size': 64,
            'lr': 1e-4,
            'n_epochs': 500,
            'dropout': 0.0,
            'mlp_hidden': [128, 64, 32],
            'conv_sizes': [5, 10],
            'sigmav': 0.3,
            'interp_method': 'torch_interp'
        }
    ]
    # Extend with different LR
    more = []
    for lr in [1e-3, 5e-4]:
        cfg = grid[0].copy()
        cfg['name'] = f"lr{lr}"
        cfg['lr'] = lr
        more.append(cfg)
    grid.extend(more)

    results = []
    for cfg in grid:
        res = run_experiment(cfg, device)
        results.append(res)
        # save intermediate
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)

    print("All experiments done. Results saved to", args.output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameter search for RVEstimator')
    parser.add_argument('--output', type=str, default='rvestimator_results.csv', help='Path to save results CSV')
    args = parser.parse_args()
    main(args)
