#!/usr/bin/env python
# coding: utf-8
import io, os, sys, time, random , pickle, humanize, psutil, GPUtil, argparse
import numpy as np
import pickle
from scipy.special import gamma
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, Dataset
from itertools import chain
from torchinterp1d import Interp1d
from torchcubicspline import natural_cubic_spline_coeffs
from astropy.timeseries import LombScargle
import matplotlib.pyplot as plt

def cubic_evaluate(coeffs, tnew):
    t = coeffs[0]
    a,b,c,d = [item.squeeze(-1) for item in coeffs[1:]]
    maxlen = b.size(-1) - 1
    index = torch.bucketize(tnew, t) - 1
    index = index.clamp(0, maxlen)  # clamp because t may go outside of [t[0], t[-1]]; this is fine
    # will never access the last element of self._t; this is correct behaviour
    fractional_part = tnew - t[index]

    batch_size, spec_size = tnew.shape
    batch_ind = torch.arange(batch_size,device=tnew.device)
    batch_ind = batch_ind.repeat((spec_size,1)).T

    inner = c[batch_ind, index] + d[batch_ind, index] * fractional_part
    inner = b[batch_ind, index] + inner * fractional_part
    return a[batch_ind, index] + inner * fractional_part

def cubic_transform(xrest, yrest, wave_shifted):
    #wave_shifted = - xobs * z + xobs
    #print("xrest:",xrest.shape,"yrest:",yrest.shape)
    coeffs = natural_cubic_spline_coeffs(xrest, yrest.unsqueeze(-1))
    out = cubic_evaluate(coeffs, wave_shifted)
    #print("out:",out.shape)
    return out

def moving_mean(x,y,w=None,n=20,skip_weight=True):
    dx = (x.max()-x.min())/n
    xgrid = np.linspace(x.min(),x.max(),n+2)
    xgrid = xgrid[1:-1]
    ygrid = np.zeros_like(xgrid)
    delta_y = np.zeros_like(xgrid)
    for i,xmid in enumerate(xgrid):
        mask = x>(xmid-dx)
        mask *= x<(xmid+dx)
        if skip_weight:
            ygrid[i] = np.mean(y[mask])
            delta_y[i] = y[mask].std()/np.sqrt(mask.sum())
        else:
            ygrid[i] = np.average(y[mask],weights=w[mask])
            delta_y[i] = np.sqrt(np.cov(y[mask], aweights=w[mask]))/np.sqrt(mask.sum())
    return xgrid,ygrid,delta_y

'''
def calculate_fft(time,signal):
    time_interval = time[1]-time[0]
    # Perform the FFT
    fft = np.fft.fft(signal)
    # Calculate the frequency axis
    freq_axis = np.fft.fftfreq(len(signal), time_interval)
    real  = freq_axis>0
    p_axis = 1.0/freq_axis[real]
    # Only show the real part of the power spectrum
    power_spectrum = np.real(fft * np.conj(fft))
    power_spectrum /= max(power_spectrum[real])
    return p_axis,power_spectrum[real]
'''
def plot_fft(timestamp,signals,fname,labels,period=100,fs=14):
    cs = ["grey","k","b","r"]
    alphas = [1,1,1,0.7]
    lw = [2,2,2,2]
    fig,ax = plt.subplots(figsize=(4,2.5),constrained_layout=True)
    pmax=0
    for i,ts in enumerate(signals[:len(cs)]):
        if "encode" in labels[i]:continue
        if "doppler" in labels[i]:continue
        frequency, power = LombScargle(timestamp, ts).autopower()
        p_axis = 1.0/frequency
        # Plot the result
        ax.plot(p_axis,power, c=cs[i],lw=lw[i],label="%s"%(labels[i]), alpha=alphas[i])
        if power.max()>pmax: pmax = power.max()
    ax.set_xlim(1,299)
    ax.set_ylim(0,1.1*pmax)
    ax.set_xlabel('Period [days]');ax.set_ylabel('Power')
    ax.axvline(period,ls="--",c="grey",zorder=-10,label="$P_{true}$")
    if "uniform" in fname:
        ax.set_yticks([0.01,0.02,0.03])
        title = r"$\mathbf{Case\ I \ (N=1000)}$"
    elif "dynamic" in fname:
        ax.set_yticks([0.05,0.10,0.15,0.20])
        title = r"$\mathbf{Case\ II \ (N=200)}$"
    else:title="test"
    ax.legend(fontsize=fs,title=title)
    plt.savefig("[%s]periodogram.png"%fname,dpi=300)
    #with open("results-%s.pkl"%fname,"wb")  as f:
    #    pickle.dump(signals,f)
    #    pickle.dump(labels,f)
    return

def plot_sphere(pos,radius,ax,c="grey",alpha=0.5,zorder=0):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radius * np.outer(np.cos(u), np.sin(v)) + pos[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + pos[1]
    z = radius* np.outer(np.ones(np.size(u)), np.cos(v)) + pos[2]
    # Plot the surface
    ax.plot_surface(x, y, z, alpha=alpha, zorder=zorder,color=c)
    return

def density_plot(points,bins=30):
    x,y,z = points
    fig, ax = plt.subplots()
    density,X,Y,_ = ax.hist2d(x, y, bins=bins)
    #print("X,Y",X,Y)
    X, Y = np.meshgrid(X[1:],Y[1:])
    mesh_dict = {"XY":[X,Y,density]}
    return mesh_dict

def visualize_encoding(points,points_aug,RV_encode,radius=0,tag=None):

    axis_mean = points.mean(axis=1,keepdims=True)
    axis_std = points.std(axis=1,keepdims=True)
    points -= axis_mean
    points /= axis_std

    points_aug -= axis_mean
    points_aug /= axis_std

    rand = np.random.randint(points.shape[1],size=(points.shape[1]))
    print("rand:",rand.shape)
    N = len(rand)
    dist = ((points-points[:,rand])**2).sum(axis=0)
    dist_aug = ((points-points_aug)**2).sum(axis=0)

    print("random pairs: %.5f"%dist.mean(),dist.shape)
    print("augment pairs: %.5f"%dist_aug.mean(),dist_aug.shape)

    bins = np.logspace(-4,1,20)
    fig,ax = plt.subplots(figsize=(4,2.5),constrained_layout=True)
    _=ax.hist(dist,label=r"$\langle \Delta s_{rand} \rangle $: %.3f"%dist.mean(),
              color="b",bins=bins,log=False,histtype="stepfilled",alpha=0.7)
    _=ax.hist(dist_aug,label=r"$\langle \Delta s_{aug} \rangle$: %.3f"%dist_aug.mean(),
              color="r",bins=bins,log=False,histtype="stepfilled",alpha=0.7)
    ax.legend(loc=2);ax.set_xlabel("latent distance $\Delta s$");ax.set_ylabel("N")
    ax.set_xscale('log')
    plt.savefig("[%s]histogram.png"%tag,dpi=300)

    import matplotlib.colors

    elev=20;azim=150; dtr = np.pi/180.0
    viewpoint = np.array([np.cos(elev*dtr)*np.cos(azim*dtr),
                          np.cos(elev*dtr)*np.sin(azim*dtr),
                          np.sin(elev*dtr)])
    dist = 8
    viewpoint *= dist
    print("viewpoint:",viewpoint.shape,"points:",points.shape)
    depth = ((points-viewpoint[:,None])**2).sum(axis=0)**0.5
    depth /= depth.min()
    size = 40/depth**2+5
    #colors = points[0] 
    colors = RV_encode
    print("colors:",colors.min(),colors.max())
    print("RV_encode:",RV_encode.min(),RV_encode.max())
    #print("depth:",depth.shape)
    #print(size.min(),size.mean(),size.max())
    # 3D rendering

    b_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["b","skyblue"])
    r_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["r","salmon"])

    fig = plt.figure(figsize = (10, 8))
    ax = plt.axes(projection ="3d")
    # Add x, y gridlines
    #ax.grid(b = True, color ='grey',linestyle ='-.', linewidth = 0.3,alpha = 0.2)
    pic = ax.scatter(points[0], points[1], points[2], s=size, marker="o",
                     alpha=1,c=colors,cmap="viridis")
    #ax.scatter(points_aug[0], points_aug[1], points_aug[2], s=size, marker="o",alpha=1,c=colors,cmap=r_cmap)
    #for i in range(N):plt.plot([points[0,i],points_aug[0,i]],[points[1,i],points_aug[1,i]],[points[2,i],points_aug[2,i]],c="grey",lw=0.5)

    xlim=(-4, 5)
    ylim=(-3, 5)
    zlim=(-4, 7)

    ms=5;c="darkgrey"
    ax.scatter(points[0], points[1],[zlim[0]]*N,s=ms,c=c,alpha=1)
    ax.scatter(points_aug[0], points_aug[1],[zlim[0]]*N,
               s=ms,c=c,alpha=1)
    ms=5;c="grey"
    ax.scatter(points[0],[ylim[0]]*N, points[2],s=ms,c=c,alpha=1)
    ax.scatter(points_aug[0],[ylim[0]]*N, points_aug[2],
               s=ms,c=c,alpha=1)

    pos = [0,4,0]
    fs = 20
    # plot a sphere
    #if radius > 0:plot_sphere(pos,radius,ax,alpha=0.5,zorder=0)
    ax.set_proj_type('persp', focal_length=0.5)
    ax.set_xlabel("$s_1$",fontsize=fs)
    ax.set_ylabel("$s_2$",fontsize=fs)
    ax.set_zlabel("$s_3$",fontsize=fs)
    ax.xaxis.labelpad=-10
    ax.yaxis.labelpad=-10
    ax.zaxis.labelpad=-10

    ax.set_xticklabels([]);ax.set_yticklabels([]);ax.set_zticklabels([])
    ax.view_init(elev=elev,azim=azim,roll=0)
    ax.dist=dist
    ax.set(xlim=xlim, ylim=ylim, zlim=zlim)
    cbar = fig.colorbar(pic, ax=ax,location = 'top', pad=0.0, shrink=0.4)
    #cbar.ax.set_xticks([])
    #cbar.ax.set_xticklabels([-2,-1,0,1],fontsize=12)
    cbar.set_label("$v_{encode}$[m/s]",fontsize=16,labelpad=10)
    #ax.set_aspect('equal')
    #plt.subplots_adjust(left=0.08, bottom=0.08, right=0.95, top=0.98)
    plt.savefig("[%s]R1-3D.png"%tag,dpi=300)
    exit()
    return

############ Functions for creating batched files ###############
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def load_batch(batch_name, subset=None):
    with open(batch_name, 'rb') as f:
        if torch.cuda.is_available():
            batch = pickle.load(f)
        else:
            batch = CPU_Unpickler(f).load()

    if subset is not None:
        return batch[subset]
    return batch

# based on https://medium.com/speechmatics/how-to-build-a-streaming-dataloader-with-pytorch-a66dd891d9dd
class BatchedFilesDataset(IterableDataset):

    def __init__(self, file_list, load_fct, shuffle=False, shuffle_instance=False):
        assert len(file_list), "File list cannot be empty"
        self.file_list = file_list
        self.shuffle = shuffle
        self.shuffle_instance = shuffle_instance
        self.load_fct = load_fct

    def process_data(self, idx):
        if self.shuffle:
            idx = random.randint(0, len(self.file_list) -1)
        batch_name = self.file_list[idx]
        data = self.load_fct(batch_name)
        data = list(zip(*data))
        if self.shuffle_instance:
            random.shuffle(data)
        for x in data:
            yield x

    def get_stream(self):
        return chain.from_iterable(map(self.process_data, range(len(self.file_list))))

    def __iter__(self):
        return self.get_stream()

    def __len__(self):
        return len(self.file_list)


def mem_report():
    print("CPU RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ))

    if torch.cuda.device_count() ==0: return

    GPUs = GPUtil.getGPUs()
    for i, gpu in enumerate(GPUs):
        print('GPU {:d} ... Mem Free: {:.0f}MB / {:.0f}MB | Utilization {:3.0f}%'.format(i, gpu.memoryFree, gpu.memoryTotal, gpu.memoryUtil*100))
    return


def resample_to_restframe(wave_obs,wave_rest,y,w,z):
    wave_z = (wave_rest.unsqueeze(1)*(1 + z)).T
    wave_obs = wave_obs.repeat(y.shape[0],1)
    # resample observed spectra to restframe
    yrest = Interp1d()(wave_obs, y, wave_z)
    wrest =  Interp1d()(wave_obs, w, wave_z)

    # interpolation = extrapolation outside of observed region, need to mask
    msk = (wave_z<=wave_obs.min())|(wave_z>=wave_obs.max())
    # yrest[msk]=0 # not needed because all spectral elements are weighted
    wrest[msk]=0
    return yrest,wrest

def generate_lines(xrange,max_amp=0.7,width=0.3,n_lines=100):
    amps = np.random.uniform(low=0.01,high=max_amp,size=n_lines)
    sigmas = np.random.normal(loc=width,scale=0.1*width,size=n_lines)
    line_loc = np.random.uniform(low=(xrange[0]+width),high=(xrange[1]-width),size=n_lines)
    sigmas = np.maximum(sigmas,0.01)
    lines = {"loc":line_loc,"amp":amps,"sigma":sigmas}
    return lines

def evaluate_lines(wave,lines,z=0,depth=1,skew=0,broaden=1,window=5):
    abs_lines = np.ones_like(wave)
    line_location = lines["loc"]+lines["loc"]*z
    for i,loc in enumerate(line_location):
        amp,sigma = lines["amp"][i],broaden*lines["sigma"][i]
        mask = (wave>(loc-window*sigma))*(wave<(loc+window*sigma))
        if skew>0:signal = gamma_profile(wave[mask],amp,loc,sigma, skew)
        else:signal = amp*np.exp(-0.5*((wave[mask]-loc)/sigma)**2)
        abs_lines[mask] *= (1-depth*signal)
    return abs_lines

def gauss(x, *p):
    amp, mu, sigma, b = p
    return amp*np.exp(-(x-mu)**2/(2.*sigma**2))+b


def gamma_profile(x, amp, mu, sigma, skew):
    a = 4/skew**2; b=2*a; sigma_0 = a**0.5/b
    mu0 = (a-1)/b
    y = np.zeros_like(x)
    xloc = ((x-mu)/sigma)*sigma_0 + mu0
    mask = xloc>0
    y[mask] = ((xloc[mask])**(a-1))*np.exp(-b*(xloc[mask]))
    y/=y.max()
    return amp*y


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
    