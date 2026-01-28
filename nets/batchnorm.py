from builtins import NotImplementedError
from enum import Enum
from typing import Tuple
import torch
from torch.functional import Tensor
import torch.nn as nn
from torch.types import Number

from geoopt.tensor import ManifoldParameter, ManifoldTensor
from Geometry.hyperbolic import Hyperboloid

class BatchNormTestStatsMode(Enum):
    BUFFER = 'buffer'
    REFIT = 'refit'
    ADAPT = 'adapt'


class BatchNormDispersion(Enum):
    NONE = 'mean'
    SCALAR = 'scalar'
    VECTOR = 'vector'


class BatchNormTestStatsInterface:
    def set_test_stats_mode(self, mode : BatchNormTestStatsMode):
        pass

# %% base classes

class BaseBatchNorm(nn.Module, BatchNormTestStatsInterface):
    def __init__(self, eta = 1.0, eta_test = 0.1, test_stats_mode : BatchNormTestStatsMode = BatchNormTestStatsMode.BUFFER):
        super().__init__()
        self.eta = eta
        self.eta_test = eta_test
        self.test_stats_mode = test_stats_mode

    def set_test_stats_mode(self, mode : BatchNormTestStatsMode):
        self.test_stats_mode = mode


class SchedulableBatchNorm(BaseBatchNorm):
    def set_eta(self, eta = None, eta_test = None):
        if eta is not None:
            self.eta = min(eta, 1.0)  # clamp eta to ensure it doesn't exceed 1
        if eta_test is not None:
            self.eta_test = min(eta_test, 1.0)  # clamp eta_test to ensure it doesn't exceed 1


class BaseDomainBatchNorm(nn.Module, BatchNormTestStatsInterface):
    def __init__(self):
        super().__init__()
        self.batchnorm = torch.nn.ModuleDict()

    def set_test_stats_mode(self, mode : BatchNormTestStatsMode):
        for bn in self.batchnorm.values():
            if isinstance(bn, BatchNormTestStatsInterface):
                bn.set_test_stats_mode(mode)

    def add_domain_(self, layer : BaseBatchNorm, domain : Tensor):
        self.batchnorm[str(domain.item())] = layer

    def get_domain_obj(self, domain : Tensor):
        return self.batchnorm[domain.item()]

    @torch.no_grad()
    def initrunningstats(self, X, domain):
        self.batchnorm[str(domain.item())].initrunningstats(X)

    def forward_domain_(self, X, domain):
        res = self.batchnorm[str(domain.item())](X)
        return res

    def forward(self, X, d):
        du = d.unique()

        X_normalized = torch.empty_like(X)
        res = [(self.forward_domain_(X[d==domain], domain),torch.nonzero(d==domain))
                for domain in du]
        X_out, ixs = zip(*res)
        X_out, ixs = torch.cat(X_out), torch.cat(ixs).flatten()
        X_normalized[ixs] = X_out
        
        return X_normalized


# %% SPD manifold implementation

class SPDBatchNormImpl(BaseBatchNorm):
    def __init__(self, shape : Tuple[int,...] or torch.Size, batchdim : int, 
                 eta = 1., eta_test = 0.1,
                 karcher_steps : int = 1, learn_mean = True, learn_std = True, 
                 dispersion : BatchNormDispersion = BatchNormDispersion.SCALAR, 
                 eps = 1e-5, mean = None, std = None, manifold = None, **kwargs):
        super().__init__(eta, eta_test)
        # the last two dimensions are used for SPD manifold


        if dispersion == BatchNormDispersion.VECTOR:
            raise NotImplementedError()

        self.dispersion = dispersion
        self.learn_mean = learn_mean
        self.learn_std = learn_std
        self.batchdim = batchdim
        self.karcher_steps = karcher_steps
        self.eps = eps
        
        from lib.lorentz.manifold import CustomLorentz
        self._custom_lorentz_manifold = manifold
        if isinstance(manifold.k, torch.Tensor):
            k_value = manifold.k.item() if manifold.k.numel() == 1 else float(manifold.k.detach().cpu().item())
        else:
            k_value = float(manifold.k)
        K_value = -1.0 / k_value
        self.manifold = Hyperboloid(K=K_value)
        
        init_mean = self.manifold.zero(shape[-1])
        init_var = torch.ones((1), dtype=torch.float64)

        self.register_buffer('running_mean', init_mean)
        self.register_buffer('running_var', init_var)
        self.register_buffer('running_mean_test', init_mean)
        self.register_buffer('running_var_test', init_var)

        self.mean = mean
        
        if self.dispersion is not BatchNormDispersion.NONE:
            if std is not None:
                self.std = std
            else:
                if self.learn_std:
                    self.std = nn.parameter.Parameter(init_var.clone())
                else:
                    self.std = init_var.clone()


    @torch.no_grad()
    def initrunningstats(self, X):
        self.running_mean = self.manifold.frechet_mean(X,max_iter=100)
        self.running_mean_test = self.running_mean.clone()

        if self.dispersion is BatchNormDispersion.SCALAR:
            self.running_var = self.manifold.frechet_variance(X, self.running_mean_test)
            self.running_var_test = self.running_var.clone()

    def _transform_mean_for_curvature(self, mean, old_K, new_K):
        """Transform running_mean when curvature K changes: log → scale → exp"""
        if torch.allclose(old_K, new_K, rtol=1e-6):
            return mean
        
        # Step 1: log0 with old K to get tangent vector
        # v = log_o(x), ||v|| = d(o, x)
        v = self.manifold.log0(mean)  # manifold still has old_K here
        
        # Step 2: scale tangent vector to preserve geodesic distance
        # d_old = ||v|| / sqrt(-K_old), d_new = ||v_new|| / sqrt(-K_new)
        # For d_old = d_new: ||v_new|| = ||v|| * sqrt(-K_new / -K_old) = ||v|| * sqrt(K_old / K_new)
        scale_factor = (old_K / new_K).sqrt()
        v_scaled = v * scale_factor
        
        # Step 3: Update K and exp0 with new K
        self.manifold.K.data.fill_(new_K.item())
        mean_new = self.manifold.exp0(v_scaled, project=True)
        
        return mean_new
    
    def _transform_var_for_curvature(self, var, old_K, new_K):
        """Transform running_var: var_new = var_old * (K_old / K_new)"""
        if torch.allclose(old_K, new_K, rtol=1e-6):
            return var
        
        scale_factor = old_K / new_K
        return var * scale_factor

    def forward(self, X):
        # Update Hyperboloid K from learnable CustomLorentz k
        # K = -1/k to ensure geometric consistency
        if isinstance(self._custom_lorentz_manifold.k, torch.Tensor):
            k_value = self._custom_lorentz_manifold.k.item() if self._custom_lorentz_manifold.k.numel() == 1 else float(self._custom_lorentz_manifold.k.detach().cpu().item())
        else:
            k_value = float(self._custom_lorentz_manifold.k)
        K_new = torch.tensor(-1.0 / k_value, dtype=self.manifold.K.dtype, device=self.manifold.K.device)
        
        with torch.no_grad():
            K_old = self.manifold.K.clone()
            
            if not torch.allclose(K_old, K_new, rtol=1e-6):
                self.running_mean = self._transform_mean_for_curvature(
                    self.running_mean, K_old, K_new)
                self.manifold.K.data.fill_(K_old.item())
                self.running_mean_test = self._transform_mean_for_curvature(
                    self.running_mean_test, K_old, K_new)
                
                if self.dispersion is BatchNormDispersion.SCALAR:
                    self.running_var = self._transform_var_for_curvature(
                        self.running_var, K_old, K_new)
                    self.running_var_test = self._transform_var_for_curvature(
                        self.running_var_test, K_old, K_new)
            else:
                self.manifold.K.data.fill_(K_new.item())
        
        bs, h, w, c = X.shape
        X = X.view(-1, c)   
        if self.training:
            batch_mean = self.manifold.frechet_mean(X, max_iter=100)
            rm = self.manifold.geodesic(self.running_mean, batch_mean,self.eta)
            if self.dispersion is BatchNormDispersion.SCALAR:
                batch_var = self.manifold.frechet_variance(X,batch_mean)
                rv = (1. - self.eta) * self.running_var + self.eta * batch_var

        else:
            if self.test_stats_mode == BatchNormTestStatsMode.BUFFER:
                pass
            elif self.test_stats_mode == BatchNormTestStatsMode.REFIT:
                self.initrunningstats(X)
            elif self.test_stats_mode == BatchNormTestStatsMode.ADAPT:              
                pass
            rm = self.running_mean_test
            if self.dispersion is BatchNormDispersion.SCALAR:
                rv = self.running_var_test

        if self.dispersion is BatchNormDispersion.SCALAR:
            inv_input_mean = self.manifold.gyroinv(rm)
            Xn = self.manifold.gyrotrans(inv_input_mean,X)
            factor = 1 / (rv + self.eps).sqrt()
            Xn = self.manifold.gyroscalarprod(Xn,factor)

        else:
            inv_input_mean = self.manifold.gyroinv(rm)
            Xn = self.manifold.gyrotrans(inv_input_mean,X)

        if self.training:
            with torch.no_grad():
                self.running_mean = rm.clone()
                self.running_mean_test = self.manifold.geodesic(self.running_mean_test, batch_mean,self.eta_test)
                if self.dispersion is not BatchNormDispersion.NONE:
                    self.running_var = rv.clone()
                    batch_var_test = self.manifold.frechet_variance(X,batch_mean)
                    self.running_var_test = (1. - self.eta_test) * self.running_var_test + self.eta_test * batch_var_test
        
        Xn = self.manifold.projx(Xn)
        
        Xn = Xn.view(bs, h, w, c)
        return Xn


class SPDBatchNorm(SPDBatchNormImpl):
    """
    Batch normalization on the SPD manifold.
    
    Implements [Brooks et al. 2019, NIPS] (dispersion=NONE) 
    and [Kobler et al. 2022, ICASSP] (dispersion=SCALAR).
    Default: dispersion=SCALAR.
    """
    def __init__(self, shape: Tuple[int, ...] or torch.Size, 
                 batchdim: int,
                 eta=0.1, **kwargs):
        if 'dispersion' not in kwargs.keys():
            kwargs['dispersion'] = BatchNormDispersion.SCALAR
        if 'eta_test' in kwargs.keys():
            raise RuntimeError('This parameter is ignored in this subclass. Use another batch normailzation variant.')
        super().__init__(shape=shape, batchdim=batchdim, 
                         eta=1.0, eta_test=eta, **kwargs)


class SPDBatchReNorm(SPDBatchNormImpl):
    """Batch re normalization on the SPD manifold [Kobler et al. 2022, ICASSP]"""
    def __init__(self, shape: Tuple[int, ...] or torch.Size, 
                 batchdim: int,
                 eta=0.1, **kwargs):
        if 'dispersion' not in kwargs.keys():
            kwargs['dispersion'] = BatchNormDispersion.SCALAR
        if 'eta_test' in kwargs.keys():
            raise RuntimeError('This parameter is ignored in this subclass.')
        super().__init__(shape=shape, batchdim=batchdim, 
                         eta=eta, eta_test=eta, **kwargs)


class AdaMomSPDBatchNorm(SPDBatchNormImpl,SchedulableBatchNorm):
    """
    Adaptive momentum batch normalization on the SPD manifold [proposed].

    The momentum terms can be controlled via a momentum scheduler.
    """
    def __init__(self, shape: Tuple[int, ...] or torch.Size, 
                 batchdim: int,
                 eta=1.0, eta_test=0.1, **kwargs):
        super().__init__(shape=shape, batchdim=batchdim, 
                         eta=eta, eta_test=eta_test, **kwargs)


class DomainSPDBatchNormImpl(BaseDomainBatchNorm):
    """Domain-specific batch normalization on the SPD manifold [proposed]"""

    domain_bn_cls = None # needs to be overwritten by subclasses

    def __init__(self, shape : Tuple[int,...] or torch.Size, batchdim :int,
                 learn_mean : bool = True, learn_std : bool = True,
                 dispersion : BatchNormDispersion = BatchNormDispersion.NONE,
                 test_stats_mode : BatchNormTestStatsMode = BatchNormTestStatsMode.BUFFER,
                 eta = 1., eta_test = 0.1, domains : Tensor = Tensor([]), manifold = None, **kwargs):
        super().__init__()

        if dispersion == BatchNormDispersion.VECTOR:
            raise NotImplementedError()

        self.dispersion = dispersion
        self.learn_mean = learn_mean
        self.learn_std = learn_std

        from lib.lorentz.manifold import CustomLorentz
        self._custom_lorentz_manifold = manifold
        if isinstance(manifold.k, torch.Tensor):
            k_value = manifold.k.item() if manifold.k.numel() == 1 else float(manifold.k.detach().cpu().item())
        else:
            k_value = float(manifold.k)
        K_value = -1.0 / k_value
        manifold_obj = Hyperboloid(K=K_value)  
        init_mean = manifold_obj.zero(shape[-1])  
        self.mean= init_mean
        
        if self.dispersion is BatchNormDispersion.SCALAR:
            init_var = torch.ones((1), dtype=torch.float64)
            if self.learn_std:
                self.std = nn.parameter.Parameter(init_var.clone())
            else:
                self.std = init_var.clone()
        else:
            self.std = None
        
        cls = type(self).domain_bn_cls
        for domain in domains:
            self.add_domain_(cls(shape=shape, batchdim=batchdim, 
                                learn_mean=learn_mean,learn_std=learn_std, dispersion=dispersion,
                                mean=self.mean, std=self.std, eta=eta, eta_test=eta_test, 
                                manifold=manifold, **kwargs),
                            domain)

        self.set_test_stats_mode(test_stats_mode)

class AdaMomDomainSPDBatchNorm(DomainSPDBatchNormImpl):
    """
    Domain-specific + adaptive momentum batch normalization [Yong et al. 2020, ECCV]
    """

    domain_bn_cls = AdaMomSPDBatchNorm
