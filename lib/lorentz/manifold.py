import torch
from typing import Tuple, Union, Dict

from lib.geoopt import Lorentz
from lib.geoopt.manifolds.lorentz import math
import torch.nn.functional as F


class CustomLorentz(Lorentz):
    def __init__(self, k=1, learnable=False, min_k=1e-5):
        # Ensure initial k > 0 (using eps=1e-5 as minimum)
        k = max(k, 1e-5)
        super(CustomLorentz, self).__init__(k=k, learnable=learnable)
        self.min_k = min_k
        # If curvature is learnable, register forward hook to ensure k stays positive
        if learnable:
            self.register_forward_pre_hook(self._clamp_curvature)
    
    def _clamp_curvature(self, module, input):
        """Ensure curvature parameter k stays > 0 (numerical stability, minimum eps=1e-5)"""
        if self.k.requires_grad:
            with torch.no_grad():
                self.k.clamp_(min=self.min_k)

    def sqdist(self, x, y, dim=-1):
        """ Squared Lorentzian distance, as defined in the paper 'Lorentzian Distance Learning for Hyperbolic Representation'"""
        return -2*self.k - 2 * math.inner(x, y, keepdim=False, dim=dim)

    def add_time(self, space):
        """ Concatenates time component to given space component. """
        time = self.calc_time(space)
        return torch.cat([time, space], dim=-1)

    def calc_time(self, space):
        """ Calculates time component from given space component. """
        return torch.sqrt(torch.norm(space, dim=-1, keepdim=True)**2+self.k)

    def centroid(self, x, w=None, eps=1e-8):
        """ Centroid implementation. Adapted the code from Chen et al. (2022) """
        if w is not None:
            avg = w.matmul(x)
        else:
            avg = x.mean(dim=-2)

        denom = (-self.inner(avg, avg, keepdim=True))
        denom = denom.abs().clamp_min(eps).sqrt()

        centroid = torch.sqrt(self.k) * avg / denom

        return centroid
    
    def mid_point(self, x, w=None, eps=1e-8):
        """
        Compute the weighted Frechet mean (midpoint) of points on the manifold.
        Compatible with Hypformer's mid_point interface.
        
        Parameters
        ----------
        x : torch.Tensor
            Points on the manifold (..., N, D+1)
        w : torch.Tensor, optional
            Weights for each point (..., M, N). Default is None (equal weights).
        eps : float
            Small value for numerical stability
        
        Returns
        -------
        torch.Tensor
            Midpoint on the manifold (..., D+1) or (..., M, D+1) if w is provided
        """
        if w is not None:
            # w: (..., M, N), x: (..., N, D+1) -> avg: (..., M, D+1)
            avg = w.matmul(x)
        else:
            # Average over second-to-last dimension
            avg = x.mean(dim=-2)
        
        denom = (-self.inner(avg, avg, keepdim=True))
        denom = denom.abs().clamp_min(eps).sqrt()
        
        return torch.sqrt(self.k) * avg / denom
    
    def cinner(self, x, y):
        """
        Compute the cross-inner product (Minkowski inner product for batches).
        
        Computes <x, y>_L = -x0*y0 + sum(xi*yi) for all pairs.
        
        Parameters
        ----------
        x : torch.Tensor
            First set of points (..., N, D+1)
        y : torch.Tensor
            Second set of points (..., M, D+1)
        
        Returns
        -------
        torch.Tensor
            Cross inner products (..., N, M)
        """
        x = x.clone()
        x.narrow(-1, 0, 1).mul_(-1)
        return x @ y.transpose(-1, -2)

    def switch_man(self, x, manifold_in: Lorentz):
        """ Projection between Lorentz manifolds (e.g. change curvature) """
        x = manifold_in.logmap0(x)
        return self.expmap0(x)
    
    def pt_addition(self, x, y):
        """ Parallel transport addition proposed by Chami et al. (2019) """
        z = self.logmap0(y)
        z = self.transp0(x, z)

        return self.expmap(x, z)

    #################################################
    #       Reshaping operations
    #################################################
    def lorentz_flatten(self, x: torch.Tensor) -> torch.Tensor:
        """ Implements flattening operation directly on the manifold. Based on Lorentz Direct Concatenation (Qu et al., 2022) """
        # 1. Check if input is already on Lorentz manifold
        # Check if Minkowski inner product equals -k
        dn = x.size(-1) - 1
        x_sq = x**2
        quad_form = -x_sq.narrow(-1, 0, 1) + x_sq.narrow(-1, 1, dn).sum(dim=-1, keepdim=True)
        expected = -self.k
        assert torch.allclose(quad_form, expected, atol=1e-5, rtol=1e-5), f"Input not on manifold! Expected quad_form={expected.item():.6f}, got mean={quad_form.mean().item():.6f}, min={quad_form.min().item():.6f}, max={quad_form.max().item():.6f}"
        
        bs,h,w,c = x.shape
        # bs x H x W x C
        time = x.narrow(-1, 0, 1).view(-1, h*w)
        space = x.narrow(-1, 1, x.shape[-1] - 1).flatten(start_dim=1) # concatenate all x_s

        time_rescaled = torch.sqrt(torch.sum(time**2, dim=-1, keepdim=True)+(((h*w)-1)*(-self.k)))
        x = torch.cat([time_rescaled, space], dim=-1) 
        
        # 2. Check if still on manifold after concatenation
        is_on_manifold, reason = self.check_on_manifold(x, atol=1e-5, rtol=1e-5, dim=-1)
        if not is_on_manifold:
            raise RuntimeError(f"lorentz_flatten not on manifold: {reason}")

        return x

    def lorentz_reshape_img(self, x: torch.Tensor, img_dim) -> torch.Tensor:
        """ Implements reshaping a flat tensor to an image directly on the manifold. Based on Lorentz Direct Split (Qu et al., 2022) """
        space = x.narrow(-1, 1, x.shape[-1] - 1)
        space = space.view((-1, img_dim[0], img_dim[1], img_dim[2]-1))
        img = self.add_time(space)

        return img


    #################################################
    #       Activation functions
    #################################################
    def lorentz_relu(self, x: torch.Tensor, add_time: bool=True) -> torch.Tensor:
        """ Implements ReLU activation directly on the manifold. """
        return self.lorentz_activation(x, torch.relu, add_time)
    
    def lorentz_elu(self, x: torch.Tensor, add_time: bool=True) -> torch.Tensor:
        """ Implements ReLU activation directly on the manifold. """
        return self.lorentz_activation(x, F.elu, add_time)    

    def lorentz_activation(self, x: torch.Tensor, activation, add_time: bool=True) -> torch.Tensor:
        """ Implements activation directly on the manifold. """
        x = activation(x.narrow(-1, 1, x.shape[-1] - 1))
        if add_time:
            x = self.add_time(x)
        return x
    
    def tangent_relu(self, x: torch.Tensor) -> torch.Tensor:
        """ Implements ReLU activation in tangent space. """
        return self.expmap0(torch.relu(self.logmap0(x)))
    
    #################################################
    #       Manifold validation functions
    #################################################
    def check_on_manifold(self, x: torch.Tensor, atol=1e-5, rtol=1e-5, dim=-1) -> Tuple[bool, Union[str, None]]:
        """
        Check if point x is on the Lorentz manifold
        
        Parameters
        ----------
        x : torch.Tensor
            Point to check
        atol : float
            Absolute tolerance
        rtol : float
            Relative tolerance
        dim : int
            Manifold dimension
        
        Returns
        -------
        bool
            Whether on manifold
        str
            If not on manifold, returns reason; otherwise returns None
        """
        # Check if Minkowski inner product equals -k
        dn = x.size(dim) - 1
        x_sq = x**2
        quad_form = -x_sq.narrow(dim, 0, 1) + x_sq.narrow(dim, 1, dn).sum(dim=dim, keepdim=True)
        
        ok1 = torch.allclose(quad_form, -self.k, atol=atol, rtol=rtol)
        if not ok1:
            expected = -self.k.item()
            actual = quad_form.mean().item()
            return False, f"Minkowski quadratic form mismatch: expected {expected:.6f}, got {actual:.6f}"
        
        # Check if time component is positive
        ok2 = (x.narrow(dim, 0, 1) > 0).all()
        if not ok2:
            min_time = x.narrow(dim, 0, 1).min().item()
            return False, f"Time component not positive: min value = {min_time:.6f}"
        
        return True, None
    
    def batch_check_on_manifold(self, x: torch.Tensor, atol=1e-5, rtol=1e-5, dim=-1, return_details=False) -> Union[bool, Dict]:
        """
        Batch check if points are on the manifold
        
        Parameters
        ----------
        x : torch.Tensor
            Points to check (batch_size, ..., feature_dim)
        atol : float
            Absolute tolerance
        rtol : float
            Relative tolerance
        dim : int
            Manifold dimension
        return_details : bool
            Whether to return detailed information
        
        Returns
        -------
        bool or dict
            If return_details=False, returns whether all points are on manifold
            If return_details=True, returns detailed information dict
        """
        # Check Minkowski inner product
        dn = x.size(dim) - 1
        x_sq = x**2
        quad_form = -x_sq.narrow(dim, 0, 1) + x_sq.narrow(dim, 1, dn).sum(dim=dim, keepdim=True)
        
        # Flatten all dimensions except manifold dimension
        quad_form_flat = quad_form.view(-1)
        k_flat = -self.k.expand_as(quad_form_flat)
        
        # Calculate error for each point
        errors = torch.abs(quad_form_flat - k_flat)
        max_error = errors.max().item()
        mean_error = errors.mean().item()
        
        # Check time components
        time_components = x.narrow(dim, 0, 1)
        min_time = time_components.min().item()
        all_positive = (time_components > 0).all().item()
        
        # Determine if on manifold
        quad_form_ok = torch.allclose(quad_form, -self.k, atol=atol, rtol=rtol)
        all_ok = quad_form_ok and all_positive
        
        if return_details:
            return {
                'all_on_manifold': all_ok,
                'quad_form_ok': quad_form_ok,
                'time_positive': all_positive,
                'max_error': max_error,
                'mean_error': mean_error,
                'min_time': min_time,
                'expected_k': -self.k.item(),
                'quad_form_mean': quad_form_flat.mean().item(),
                'quad_form_std': quad_form_flat.std().item()
            }
        else:
            return all_ok
