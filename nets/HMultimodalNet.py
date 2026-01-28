"""
EEG-MoCE for EAV dataset

Architecture:
1. Single modality encoders (EEG, Audio, Vision) with individual curvatures
2. Curvature projection to unified space (log-exp map with scaling)
3. Hyperbolic cross-attention fusion (each modality attends to others only)
4. Hyperbolic classification (LorentzMLR)

Output: (B, num_classes) emotion class logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from geoopt.optim import RiemannianAdam
from geoopt.tensor import ManifoldParameter
from lib.lorentz.layers import LorentzMLR, LorentzeLU, LorentzAvgPool2d
from lib.lorentz.manifold import CustomLorentz
import nets.batchnorm as bn


# Single Modality Encoders

class EEGEncoder(nn.Module):
    """EEG Encoder - outputs (B, F2, 1, target_seq_len)"""
    
    def __init__(self, 
                 chunk_size: int = 151,
                 num_electrodes: int = 60,
                 F1: int = 8,
                 F2: int = 16,
                 D: int = 2,
                 kernel_1: int = 64,
                 kernel_2: int = 16,
                 dropout: float = 0.25,
                 target_seq_len: int = 64):
        super(EEGEncoder, self).__init__()
        
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.target_seq_len = target_seq_len
        
        self.block1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, kernel_1), stride=1, padding=(0, kernel_1 // 2), bias=False),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            nn.Conv2d(self.F1, self.F1 * self.D, (num_electrodes, 1),
                      stride=1, padding=(0, 0), groups=self.F1, bias=False),
            nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=4),
            nn.Dropout(p=dropout)
        )
        
        self.ec1 = nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, kernel_2),
                             stride=1, padding=(0, kernel_2 // 2), bias=False, groups=self.F1 * self.D)
        self.ec2 = nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), groups=1, bias=False, stride=1)
        self.ec2_bn = nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, target_seq_len))
    
    def forward(self, x):
        """x: (B, num_electrodes, chunk_size) -> (B, 1, target_seq_len, F2)"""
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.ec1(x)
        x = self.ec2(x)
        x = self.ec2_bn(x)
        x = self.adaptive_pool(x)
        x = x.permute(0, 2, 3, 1)
        return x


class TemporalTransformer(nn.Module):
    """Temporal Transformer for sequence modeling"""
    
    def __init__(self, hidden_dim, num_heads=2, num_layers=2, dim_feedforward=None, dropout=0.1):
        super(TemporalTransformer, self).__init__()
        
        if dim_feedforward is None:
            dim_feedforward = hidden_dim * 4
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = x + self.transformer_encoder(x)
        x = self.norm(x)
        x = x.transpose(1, 2)
        return x


class AudioEncoder(nn.Module):
    """Audio Encoder - outputs (B, 1, seq_len, hidden_dim)"""
    
    def __init__(self, freq_bins=128, hidden_dim=16, seq_len=64, dropout=0.5,
                 use_temporal_transformer=True, transformer_heads=2, transformer_layers=2):
        super(AudioEncoder, self).__init__()
        
        self.freq_bins = freq_bins
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.use_temporal_transformer = use_temporal_transformer
        
        self.freq_conv = nn.Sequential(
            nn.Conv1d(freq_bins, hidden_dim * 4, kernel_size=1),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Conv1d(hidden_dim * 4, hidden_dim * 2, kernel_size=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Temporal Convolution
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.AdaptiveAvgPool1d(seq_len)
        )
        
        if use_temporal_transformer:
            self.temporal_transformer = TemporalTransformer(
                hidden_dim=hidden_dim,
                num_heads=transformer_heads,
                num_layers=transformer_layers,
                dropout=dropout * 0.2
            )
    
    def forward(self, x):
        """x: (B, freq_bins, time_frames) -> (B, 1, seq_len, hidden_dim)"""
        if x.shape[1] != self.freq_bins and x.shape[2] == self.freq_bins:
            x = x.transpose(1, 2)
        
        x = self.freq_conv(x)
        x = self.temporal_conv(x)
        
        if self.use_temporal_transformer:
            x = self.temporal_transformer(x)
        
        x = x.unsqueeze(2).permute(0, 2, 3, 1)
        return x


class VisionEncoder(nn.Module):
    """Vision AU Encoder - outputs (B, 1, seq_len, hidden_dim). Input: (B, T, num_au)"""
    
    def __init__(self, num_au=35, hidden_dim=16, seq_len=64, dropout=0.5,
                 use_temporal_transformer=True, transformer_heads=2, transformer_layers=2):
        super(VisionEncoder, self).__init__()
        
        self.num_au = num_au
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.use_temporal_transformer = use_temporal_transformer
        
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(num_au, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Temporal Convolution
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(seq_len)
        )
        
        if use_temporal_transformer:
            self.temporal_transformer = TemporalTransformer(
                hidden_dim=hidden_dim,
                num_heads=transformer_heads,
                num_layers=transformer_layers,
                dropout=dropout * 0.2
            )
        
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        """x: (B, T, num_au) -> (B, 1, seq_len, hidden_dim)"""
        x = x.transpose(1, 2)
        x = self.spatial_conv(x)
        x = self.temporal_conv(x)
        
        if self.use_temporal_transformer:
            x = self.temporal_transformer(x)
        
        x = self.dropout(x)
        x = x.unsqueeze(2).permute(0, 2, 3, 1)
        return x


# Hyperbolic Layers for Fusion

class HypLinear(nn.Module):
    """Hyperbolic Linear Layer, Implements [Yang et al. 2024, KDD] Hypformer"""
    
    def __init__(self, manifold, in_features, out_features, bias=True):
        super().__init__()
        self.manifold = manifold
        self.in_features = in_features + 1  # +1 for time dimension
        self.out_features = out_features
        self.linear = nn.Linear(self.in_features, out_features, bias=bias)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear.weight, gain=math.sqrt(2))
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0)
    
    def forward(self, x):
        x_space = self.linear(x)
        x_time = ((x_space ** 2).sum(dim=-1, keepdim=True) + self.manifold.k).sqrt()
        return torch.cat([x_time, x_space], dim=-1)


class HypLayerNorm(nn.Module):
    """Hyperbolic Layer Normalization, Implements [Yang et al. 2024, KDD] Hypformer"""
    
    def __init__(self, manifold, in_features):
        super().__init__()
        self.manifold = manifold
        self.layer = nn.LayerNorm(in_features)
    
    def forward(self, x):
        x_space = x[..., 1:]
        x_space = self.layer(x_space)
        x_time = ((x_space ** 2).sum(dim=-1, keepdim=True) + self.manifold.k).sqrt()
        return torch.cat([x_time, x_space], dim=-1)


class CurvatureAwareCrossAttention(nn.Module):
    """
    Curvature-Aware Hyperbolic Cross Attention
    
    Features:
    1. Curvature-Scaled Temperature: τ^(m) = τ_0 / sqrt(|K^(m)|)
    2. Curvature Prior Bias: λ * log(|K^(j)| + ε)
    
    Reference: attention mechanism in [Yang et al. 2024, KDD] Hypformer
    """
    
    def __init__(self, manifold, in_channels, out_channels, num_heads=4, 
                 use_weight=True, curvature_prior_init=0.1, eps=1e-6,
                 use_curvature_prior=True):
        super().__init__()
        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight
        self.eps = eps
        self.use_curvature_prior = use_curvature_prior
        
        self.Wq = nn.ModuleList([HypLinear(manifold, in_channels, out_channels) for _ in range(num_heads)])
        self.Wk = nn.ModuleList([HypLinear(manifold, in_channels, out_channels) for _ in range(num_heads)])
        if use_weight:
            self.Wv = nn.ModuleList([HypLinear(manifold, in_channels, out_channels) for _ in range(num_heads)])
        
        self.base_temperature = nn.Parameter(torch.tensor(math.sqrt(out_channels)))
        
        if use_curvature_prior:
            lambda_target = max(curvature_prior_init, 1e-6)
            lambda_raw_init = math.log(math.exp(lambda_target) - 1 + 1e-8)
            self.curvature_prior_weight = nn.Parameter(torch.tensor(lambda_raw_init))
        else:
            self.curvature_prior_weight = None
    
    def cinner(self, x, y):
        """Compute Lorentzian inner product: <p,q>_L = <p_s, q_s> - p_t * q_t"""
        x = x.clone()
        x.narrow(-1, 0, 1).mul_(-1)
        return x @ y.transpose(-1, -2)
    
    def curvature_scaled_temperature(self, curvatures):
        """Compute τ^(m) = τ_0 / sqrt(|K^(m)|) where |K| = 1/k"""
        k_tensor = torch.stack([c.detach() if isinstance(c, torch.Tensor) else torch.tensor(c) 
                                for c in curvatures])
        K_magnitude = 1.0 / (k_tensor + self.eps)
        temperatures = self.base_temperature / torch.sqrt(K_magnitude + self.eps)
        return temperatures
    
    def curvature_prior_bias(self, curvatures):
        """Compute bias^(j) = λ * log(|K^(j)| + ε) where λ > 0 via softplus"""
        if not self.use_curvature_prior or self.curvature_prior_weight is None:
            return None
        
        k_tensor = torch.stack([c.detach() if isinstance(c, torch.Tensor) else torch.tensor(c) 
                                for c in curvatures])
        K_magnitude = 1.0 / (k_tensor + self.eps)
        log_curvatures = torch.log(K_magnitude + self.eps)
        lambda_positive = F.softplus(self.curvature_prior_weight)
        biases = lambda_positive * log_curvatures
        return biases
    
    def forward(self, modalities, curvatures, return_attn=False, verify_manifold=False):
        """modalities: (B, num_modalities, D+1), curvatures: list of k values"""
        B, num_mod, D_plus_1 = modalities.shape
        device = modalities.device
        
        if verify_manifold:
            t_flat = modalities.reshape(-1, D_plus_1)
            is_ok, reason = self.manifold.check_on_manifold(t_flat, atol=1e-5, rtol=1e-5)
            if not is_ok:
                raise ValueError(f"CurvatureAwareCrossAttention input not on manifold: {reason}")
        
        temperatures = self.curvature_scaled_temperature(curvatures).to(device)
        prior_biases = self.curvature_prior_bias(curvatures)
        if prior_biases is not None:
            prior_biases = prior_biases.to(device)
        
        K_f = self.manifold.k
        
        all_head_outputs = []
        all_head_attn = []
        
        for h in range(self.num_heads):
            Q = self.Wq[h](modalities)
            K = self.Wk[h](modalities)
            if self.use_weight:
                V = self.Wv[h](modalities)
            else:
                V = modalities
            
            inner_products = self.cinner(Q, K)
            base_scores = 2 * K_f + 2 * inner_products
            temp_scaled_scores = base_scores / temperatures.view(1, num_mod, 1)
            
            if prior_biases is not None:
                scores_with_prior = temp_scaled_scores + prior_biases.view(1, 1, num_mod)
            else:
                scores_with_prior = temp_scaled_scores
            
            diag_mask = torch.eye(num_mod, device=device, dtype=modalities.dtype)
            all_scores_masked = scores_with_prior - diag_mask.unsqueeze(0) * 1e9
            attn_weights = F.softmax(all_scores_masked, dim=-1)
            
            head_outputs = []
            for query_idx in range(num_mod):
                other_indices = [i for i in range(num_mod) if i != query_idx]
                V_others = V[:, other_indices, :]
                weights_others = attn_weights[:, query_idx:query_idx+1, other_indices]
                weights_others = weights_others / (weights_others.sum(dim=-1, keepdim=True) + 1e-8)
                output = self.manifold.mid_point(V_others, weights_others)
                head_outputs.append(output.squeeze(1))
            
            head_output = torch.stack(head_outputs, dim=1)
            all_head_outputs.append(head_output)
            attn_weights_clean = attn_weights * (1 - diag_mask.unsqueeze(0))
            all_head_attn.append(attn_weights_clean)
        
        head_stack = torch.stack(all_head_outputs, dim=1)
        outputs = []
        for mod_idx in range(num_mod):
            mod_output = self.manifold.mid_point(head_stack[:, :, mod_idx, :])
            outputs.append(mod_output)
        output = torch.stack(outputs, dim=1)  # (B, num_mod, D'+1)
        
        if verify_manifold:
            is_ok, reason = self.manifold.check_on_manifold(
                output.reshape(-1, output.shape[-1]), atol=1e-5, rtol=1e-5
            )
            if not is_ok:
                raise ValueError(f"CurvatureAwareCrossAttention output not on manifold: {reason}")
        
        if return_attn:
            attn_matrix = torch.stack(all_head_attn, dim=1)  # (B, H, num_mod, num_mod)
            return output, attn_matrix
        return output


class HyperbolicFusionModule(nn.Module):
    """
    Hyperbolic Multimodal Fusion Module with Curvature-Oriented Fusion
    
    1. Project to unified curvature space
    2. Curvature-aware cross attention
    3. Aggregate via Fréchet mean
    """
    
    def __init__(self, 
                 fusion_manifold,
                 feature_dim,
                 hidden_dim,
                 num_heads=4,
                 num_layers=2,
                 curvature_prior_init=0.1,
                 modality_names=None):
        super().__init__()
        
        if modality_names is None:
            modality_names = ['eeg', 'audio', 'vision']
        self.modality_names = modality_names
        self.num_modalities = len(modality_names)
        
        self.fusion_manifold = fusion_manifold
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.cross_attention_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = feature_dim if i == 0 else hidden_dim
            out_dim = hidden_dim
            
            self.cross_attention_layers.append(
                CurvatureAwareCrossAttention(
                    fusion_manifold, in_dim, out_dim, 
                    num_heads=num_heads,
                    curvature_prior_init=curvature_prior_init,
                    use_curvature_prior=(i == 0)
                )
            )
            self.layer_norms.append(HypLayerNorm(fusion_manifold, out_dim))
        
        self.output_proj = HypLinear(fusion_manifold, hidden_dim, hidden_dim)
    
    def project_to_unified_curvature(self, x, source_manifold):
        """Project to unified curvature: log → scale by sqrt(k_target/k_source) → exp"""
        if source_manifold.k == self.fusion_manifold.k:
            return x
        
        v = source_manifold.logmap0(x)
        scale = torch.sqrt(self.fusion_manifold.k / source_manifold.k)
        v_scaled = v * scale
        return self.fusion_manifold.expmap0(v_scaled)
    
    def forward(self, eeg_features, audio_features, vision_features, 
                eeg_manifold, audio_manifold, vision_manifold, 
                return_contributions=False, verify_manifold=False):
        """Curvature-Oriented Multimodal Fusion"""
        curvatures = [eeg_manifold.k, audio_manifold.k, vision_manifold.k]
        
        eeg_proj = self.project_to_unified_curvature(eeg_features, eeg_manifold)
        audio_proj = self.project_to_unified_curvature(audio_features, audio_manifold)
        vision_proj = self.project_to_unified_curvature(vision_features, vision_manifold)
        
        if verify_manifold:
            for name, feat in [('eeg_proj', eeg_proj), ('audio_proj', audio_proj), ('vision_proj', vision_proj)]:
                is_ok, reason = self.fusion_manifold.check_on_manifold(feat, atol=1e-5, rtol=1e-5)
                if not is_ok:
                    raise ValueError(f"Fusion {name} not on manifold after curvature projection: {reason}")
        
        modality_features = torch.stack([eeg_proj, audio_proj, vision_proj], dim=1)
        all_attn_weights = []
        
        x = modality_features
        for i, (attn_layer, ln) in enumerate(zip(self.cross_attention_layers, self.layer_norms)):
            if return_contributions:
                attn_out, attn_weights = attn_layer(x, curvatures, return_attn=True, verify_manifold=verify_manifold)
                all_attn_weights.append(attn_weights)
            else:
                attn_out = attn_layer(x, curvatures, verify_manifold=verify_manifold)
            
            attn_out = ln(attn_out)
            
            if verify_manifold:
                is_ok, reason = self.fusion_manifold.check_on_manifold(
                    attn_out.reshape(-1, attn_out.shape[-1]), atol=1e-5, rtol=1e-5
                )
                if not is_ok:
                    raise ValueError(f"Fusion layer {i} output not on manifold after LayerNorm: {reason}")
            
            x = attn_out
        
        fused = self.fusion_manifold.mid_point(x)
        
        if verify_manifold:
            is_ok, reason = self.fusion_manifold.check_on_manifold(fused, atol=1e-5, rtol=1e-5)
            if not is_ok:
                raise ValueError(f"Fused features not on manifold after mid_point: {reason}")
        
        fused = self.output_proj(fused)
        
        if verify_manifold:
            is_ok, reason = self.fusion_manifold.check_on_manifold(fused, atol=1e-5, rtol=1e-5)
            if not is_ok:
                raise ValueError(f"Final fused features not on manifold: {reason}")
        
        if return_contributions:
            contributions = self._compute_modality_contributions(all_attn_weights)
            return fused, contributions
        
        return fused
    
    def _compute_modality_contributions(self, all_attn_weights):
        """Compute modality contributions from attention weights"""
        num_mod = self.num_modalities
        stacked = torch.stack(all_attn_weights, dim=0)
        avg_attn = stacked.mean(dim=(0, 2))
        
        mask = 1 - torch.eye(num_mod, device=avg_attn.device, dtype=avg_attn.dtype)
        cross_attn = avg_attn * mask.unsqueeze(0)
        attention_received = cross_attn.sum(dim=1)
        
        total_attention = attention_received.sum(dim=-1, keepdim=True)
        contribution_ratio = attention_received / (total_attention + 1e-8)
        avg_contribution = contribution_ratio.mean(dim=0)
        
        attn_matrix = avg_attn.mean(dim=0)
        
        result = {
            f'{name}_contribution': avg_contribution[i].item()
            for i, name in enumerate(self.modality_names)
        }
        result['attention_matrix'] = attn_matrix.detach().cpu().numpy()
        
        return result


# Main Multimodal Model

class HMultimodalNet(nn.Module):
    """Hyperbolic Multimodal Fusion Network for Emotion Recognition"""
    
    def __init__(self,
                 # EEG params
                 eeg_chunk_size: int = 151,
                 eeg_num_electrodes: int = 30,
                 eeg_F1: int = 8,
                 eeg_F2: int = 16,
                 eeg_D: int = 2,
                 # Audio params
                 audio_freq_bins: int = 128,
                 audio_time_frames: int = 1024,
                 audio_hidden_dim: int = 16,
                 # Vision params
                 vision_num_au: int = 35,
                 vision_time_length: int = 603,
                 vision_hidden_dim: int = 16,
                 # Common params
                 unified_hidden_dim: int = 16,  # Unified dimension for all modalities
                 seq_len: int = 64,
                 num_classes: int = 5,
                 dropout: float = 0.25,
                 pool_kernel: int = 8,
                 # Transformer params
                 use_temporal_transformer: bool = True,
                 transformer_heads: int = 2,
                 transformer_layers: int = 2,
                 # Fusion params
                 fusion_num_heads: int = 4,
                 fusion_num_layers: int = 2,
                 fusion_hidden_dim: int = 64,
                 curvature_prior_init: float = 0.1,
                 # Hyperbolic params
                 bnorm_dispersion=bn.BatchNormDispersion.SCALAR,
                 domains=[],
                 domain_adaptation=True,
                 device='cuda',
                 dtype=torch.float64,
                 lr=0.01,
                 weight_decay=1e-3,
                 learnable_curvature=True,
                 curvature_lr_scale=0.1,
                 curvature_init=1.0,
                 unified_curvature: float = 0.5,  # Unified curvature for fusion
                 curvature_mode: str = 'mean', 
                 **kwargs):
        
        super(HMultimodalNet, self).__init__()
        
        self.unified_hidden_dim = unified_hidden_dim
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.pool_kernel = pool_kernel
        self.bnorm_dispersion = bnorm_dispersion
        self.domain_adaptation = domain_adaptation
        self.learnable_curvature = learnable_curvature
        self.curvature_lr_scale = curvature_lr_scale
        self.curvature_init = curvature_init
        self.curvature_mode = curvature_mode
        self.domains = domains
        self.device_ = device
        self.dtype_ = dtype
        self.lr = lr
        self.weight_decay = weight_decay
        self.total_forward_calls = 0
        self.curvature_prior_init = curvature_prior_init
        
        self.eeg_manifold = CustomLorentz(k=curvature_init, learnable=learnable_curvature)
        self.audio_manifold = CustomLorentz(k=curvature_init, learnable=learnable_curvature)
        self.vision_manifold = CustomLorentz(k=curvature_init, learnable=learnable_curvature)
        self.fusion_manifold = CustomLorentz(k=unified_curvature, learnable=False)
        
        self.eeg_encoder = EEGEncoder(
            chunk_size=eeg_chunk_size,
            num_electrodes=eeg_num_electrodes,
            F1=eeg_F1,
            F2=unified_hidden_dim,  # Use unified_hidden_dim to match
            D=eeg_D,
            dropout=dropout,
            target_seq_len=seq_len
        )
        
        # Audio Encoder
        self.audio_encoder = AudioEncoder(
            freq_bins=audio_freq_bins,
            hidden_dim=unified_hidden_dim,
            seq_len=seq_len,
            dropout=dropout,
            use_temporal_transformer=use_temporal_transformer,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers
        )
        
        self.vision_encoder = VisionEncoder(
            num_au=vision_num_au,
            hidden_dim=unified_hidden_dim,
            seq_len=seq_len,
            dropout=dropout,
            use_temporal_transformer=use_temporal_transformer,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers
        )
        
        self._final_time_dim = (seq_len - pool_kernel) // pool_kernel + 1
        self._feature_dim = 1 + self._final_time_dim * unified_hidden_dim
        bn_shape = (1, 1, seq_len, unified_hidden_dim + 1)
        if domain_adaptation:
            self.eeg_bn = bn.AdaMomDomainSPDBatchNorm(
                bn_shape, batchdim=0, domains=domains,
                learn_mean=False, learn_std=True, dispersion=bnorm_dispersion,
                eta=1., eta_test=.1, dtype=dtype, device=device, manifold=self.eeg_manifold
            )
        else:
            self.eeg_bn = bn.AdaMomSPDBatchNorm(
                bn_shape, batchdim=0, dispersion=bnorm_dispersion,
                learn_mean=False, learn_std=True, eta=1., eta_test=.1,
                dtype=dtype, device=device, manifold=self.eeg_manifold
            )
        self.eeg_elu = LorentzeLU(self.eeg_manifold)
        self.eeg_avpool = LorentzAvgPool2d(self.eeg_manifold, (1, pool_kernel), stride=pool_kernel)
        
        if domain_adaptation:
            self.audio_bn = bn.AdaMomDomainSPDBatchNorm(
                bn_shape, batchdim=0, domains=domains,
                learn_mean=False, learn_std=True, dispersion=bnorm_dispersion,
                eta=1., eta_test=.1, dtype=dtype, device=device, manifold=self.audio_manifold
            )
        else:
            self.audio_bn = bn.AdaMomSPDBatchNorm(
                bn_shape, batchdim=0, dispersion=bnorm_dispersion,
                learn_mean=False, learn_std=True, eta=1., eta_test=.1,
                dtype=dtype, device=device, manifold=self.audio_manifold
            )
        self.audio_elu = LorentzeLU(self.audio_manifold)
        self.audio_avpool = LorentzAvgPool2d(self.audio_manifold, (1, pool_kernel), stride=pool_kernel)
        
        if domain_adaptation:
            self.vision_bn = bn.AdaMomDomainSPDBatchNorm(
                bn_shape, batchdim=0, domains=domains,
                learn_mean=False, learn_std=True, dispersion=bnorm_dispersion,
                eta=1., eta_test=.1, dtype=dtype, device=device, manifold=self.vision_manifold
            )
        else:
            self.vision_bn = bn.AdaMomSPDBatchNorm(
                bn_shape, batchdim=0, dispersion=bnorm_dispersion,
                learn_mean=False, learn_std=True, eta=1., eta_test=.1,
                dtype=dtype, device=device, manifold=self.vision_manifold
            )
        self.vision_elu = LorentzeLU(self.vision_manifold)
        self.vision_avpool = LorentzAvgPool2d(self.vision_manifold, (1, pool_kernel), stride=pool_kernel)
        
        self.fusion_module = HyperbolicFusionModule(
            fusion_manifold=self.fusion_manifold,
            feature_dim=self._feature_dim - 1,
            hidden_dim=fusion_hidden_dim,
            num_heads=fusion_num_heads,
            num_layers=fusion_num_layers,
            curvature_prior_init=curvature_prior_init,
            modality_names=['eeg', 'audio', 'vision']
        )
        
        self.classifier = LorentzMLR(self.fusion_manifold, fusion_hidden_dim + 1, num_classes)
        
        if self.curvature_mode == 'mean':
            with torch.no_grad():
                K_eeg = -1.0 / self.eeg_manifold.k
                K_audio = -1.0 / self.audio_manifold.k
                K_vision = -1.0 / self.vision_manifold.k
                mean_K = (K_eeg + K_audio + K_vision) / 3
                fusion_k = -1.0 / mean_K
                self.fusion_manifold.k.copy_(fusion_k)
    
    def _encode_modality(self, x, encoder, manifold, hyp_bn, elu, avpool, domains):
        """Encode single modality: Euclidean encoder → Hyperbolic layers"""
        x = encoder(x)
        x = x / (x.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        x = manifold.projx(F.pad(x, pad=(1, 0)))
        x = hyp_bn(x, domains)
        x = elu(x)
        x = avpool(x)
        x = manifold.lorentz_flatten(x)
        return x
    
    def forward(self, eeg_input, audio_input, vision_input, domains, 
                verify_manifold=False, return_contributions=False):
        """
        Forward pass
        
        Args:
            eeg_input: (B, num_electrodes, chunk_size)
            audio_input: (B, freq_bins, time_frames)
            vision_input: (B, time_length, num_au)
            domains: domain labels
            verify_manifold: whether to verify manifold constraints
            return_contributions: whether to return modality fusion contributions
        
        Returns:
            logits: (B, num_classes)
            features: dict with modality features and fused features
                      includes 'contributions' if return_contributions=True
        """
        self.total_forward_calls += 1
        
        if self.curvature_mode == 'mean':
            K_eeg = -1.0 / self.eeg_manifold.k
            K_audio = -1.0 / self.audio_manifold.k
            K_vision = -1.0 / self.vision_manifold.k
            mean_K = (K_eeg + K_audio + K_vision) / 3
            fusion_k = -1.0 / mean_K
            with torch.no_grad():
                self.fusion_manifold.k.copy_(fusion_k)
        
        eeg_features = self._encode_modality(
            eeg_input, self.eeg_encoder, self.eeg_manifold,
            self.eeg_bn, self.eeg_elu, self.eeg_avpool, domains
        )
        
        audio_features = self._encode_modality(
            audio_input, self.audio_encoder, self.audio_manifold,
            self.audio_bn, self.audio_elu, self.audio_avpool, domains
        )
        
        vision_features = self._encode_modality(
            vision_input, self.vision_encoder, self.vision_manifold,
            self.vision_bn, self.vision_elu, self.vision_avpool, domains
        )
        
        if verify_manifold:
            for name, feat, man in [
                ('EEG', eeg_features, self.eeg_manifold),
                ('Audio', audio_features, self.audio_manifold),
                ('Vision', vision_features, self.vision_manifold)
            ]:
                is_ok, reason = man.check_on_manifold(feat, atol=1e-5, rtol=1e-5)
                if not is_ok:
                    raise ValueError(f"{name} features not on manifold: {reason}")
        
        if return_contributions:
            fused_features, contributions = self.fusion_module(
                eeg_features, audio_features, vision_features,
                self.eeg_manifold, self.audio_manifold, self.vision_manifold,
                return_contributions=True, verify_manifold=verify_manifold
            )
        else:
            fused_features = self.fusion_module(
                eeg_features, audio_features, vision_features,
                self.eeg_manifold, self.audio_manifold, self.vision_manifold,
                verify_manifold=verify_manifold
            )
            contributions = None
        
        logits = self.classifier(fused_features)
        
        features = {
            'eeg': eeg_features,
            'audio': audio_features,
            'vision': vision_features,
            'fused': fused_features
        }
        
        if contributions is not None:
            features['contributions'] = contributions
        
        return logits, features
    
    def feature_dim(self):
        """Get single modality feature dimension"""
        return self._feature_dim
    
    def get_curvatures(self):
        """Get current curvature values for all manifolds"""
        return {
            'eeg': self.eeg_manifold.k.item(),
            'audio': self.audio_manifold.k.item(),
            'vision': self.vision_manifold.k.item(),
            'fusion': self.fusion_manifold.k.item()
        }
    
    def get_curvature_prior_weights(self):
        """Get learned curvature prior weight (λ) from first cross-attention layer"""
        first_layer = self.fusion_module.cross_attention_layers[0]
        if hasattr(first_layer, 'curvature_prior_weight') and first_layer.curvature_prior_weight is not None:
            lambda_positive = F.softplus(first_layer.curvature_prior_weight).item()
            return {'lambda': lambda_positive}
        
        return None
    
    def configure_optimizers(self, lr=None, weight_decay=None):
        """Configure optimizers: Euclidean/curvature -> Adam, ManifoldParameter -> RiemannianAdam"""
        euclidean_params = []
        curvature_params = []
        manifold_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'manifold.k' in name:
                # Curvature is a scalar parameter, NOT a point on manifold
                # Must use regular Adam, not RiemannianAdam
                curvature_params.append(param)
            elif isinstance(param, ManifoldParameter):
                # ManifoldParameter lives on the manifold, use RiemannianAdam
                manifold_params.append(param)
            else:
                # Regular Euclidean params (Conv, Linear, BN, LorentzMLR.a, LorentzMLR.z)
                euclidean_params.append(param)
        
        # Build optimizer groups
        optimizers = []
        
        # 1. Adam for Euclidean parameters
        euclidean_groups = [dict(params=euclidean_params, weight_decay=weight_decay)]
        
        # Add curvature params with scaled learning rate (if learnable)
        if curvature_params and self.learnable_curvature:
            curvature_lr = lr * self.curvature_lr_scale if lr is not None else lr
            euclidean_groups.append(dict(params=curvature_params, lr=curvature_lr, weight_decay=0.))
        
        euclidean_optimizer = torch.optim.Adam(euclidean_groups, lr=lr)
        optimizers.append(euclidean_optimizer)
        
        # 2. RiemannianAdam for ManifoldParameter
        if manifold_params:
            riemannian_groups = [dict(params=manifold_params, weight_decay=0., stabilize=1)]
            riemannian_optimizer = RiemannianAdam(riemannian_groups, lr=lr, stabilize=1)
            optimizers.append(riemannian_optimizer)
        
        return optimizers if len(optimizers) > 1 else optimizers[0]
    
    def domainadapt_finetune(self, eeg_x, audio_x, vision_x, d, target_domains):
        """Source-Free Domain Adaptation"""
        if self.domain_adaptation:
            self.eeg_bn.set_test_stats_mode(bn.BatchNormTestStatsMode.REFIT)
            self.audio_bn.set_test_stats_mode(bn.BatchNormTestStatsMode.REFIT)
            self.vision_bn.set_test_stats_mode(bn.BatchNormTestStatsMode.REFIT)
            
            with torch.no_grad():
                for du in d.unique():
                    mask = d == du
                    self.forward(eeg_x[mask], audio_x[mask], vision_x[mask], d[mask])
            
            self.eeg_bn.set_test_stats_mode(bn.BatchNormTestStatsMode.BUFFER)
            self.audio_bn.set_test_stats_mode(bn.BatchNormTestStatsMode.BUFFER)
            self.vision_bn.set_test_stats_mode(bn.BatchNormTestStatsMode.BUFFER)

