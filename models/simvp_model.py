from tokenize import group
import torch
from torch import nn
#from timm.models.layers import trunc_normal_
from modules import GASubBlock
from API.utils import MinkowskiDropPath, MinkowskiGRN, MinkowskiLayerNorm, MinkowskiGroupNorm
from typing import Union
import numpy as np
import math
from timm.models.layers import DropPath,trunc_normal_
from MinkowskiEngine import (
    MinkowskiConvolution,
    MinkowskiDepthwiseConvolution,
    MinkowskiLinear,
    MinkowskiGELU,
    MinkowskiSiLU,
    MinkowskiConvolutionTranspose,
    SparseTensor,
)
from MinkowskiOps import (
    to_sparse,
    to_sparse_all,
)


#from timm.models.layers import DropPath
def dense_coordinates(shape: Union[list, torch.Size]):
    """
    coordinates = dense_coordinates(tensor.shape)
    """
    r"""
    Assume the input to have BxCxD1xD2x....xDN format.

    If the shape of the tensor do not change, use 
    """
    spatial_dim = len(shape) - 2
    assert (
        spatial_dim > 0
    ), "Invalid shape. Shape must be batch x channel x spatial dimensions."

    # Generate coordinates
    size = [i for i in shape]
    B = size[0]
    coordinates = torch.from_numpy(
        np.stack(
            [   
                s.reshape(-1)
                for s in np.meshgrid(
                    np.linspace(0, B - 1, B),
                    *(np.linspace(0, s - 1, s) for s in size[2:]),
                    indexing="ij",
                )
            ],
            1,
        )
    ).int()
    return coordinates

def to_sparse_stride4(dense_tensor: torch.Tensor, coordinate_map_key=None,coordinate_manager=None ,coordinates: torch.Tensor = None):
    r"""Converts a (differentiable) dense tensor to a sparse tensor with all coordinates.

    Assume the input to have BxCxD1xD2x....xDN format.

    If the shape of the tensor do not change, use `dense_coordinates` to cache the coordinates.
    Please refer to tests/python/dense.py for usage

    Example::

       >>> dense_tensor = torch.rand(3, 4, 5, 6, 7, 8)  # BxCxD1xD2xD3xD4
       >>> dense_tensor.requires_grad = True
       >>> stensor = to_sparse(dense_tensor)

    """
    spatial_dim = dense_tensor.ndim - 2
    assert (
        spatial_dim > 0
    ), "Invalid shape. Shape must be batch x channel x spatial dimensions."

    if coordinates is None:
        coordinates = dense_coordinates(dense_tensor.shape)

    feat_tensor = dense_tensor.permute(0, *(2 + i for i in range(spatial_dim)), 1)
    return SparseTensor(
        feat_tensor.reshape(-1, dense_tensor.size(1)),
        #coordinates,
        device=dense_tensor.device,
        tensor_stride=4,
        coordinate_map_key=coordinate_map_key,
        coordinate_manager=coordinate_manager
    )



def sampling_generator(N, reverse=False):
    samplings = [False, True] * (N // 2)
    if N==1:
        samplings = [False]
    if reverse: return list(reversed(samplings[:N]))
    else: return samplings[:N]

def upsample_mask(mask, scale):
        assert len(mask.shape) == 2
        p = int(mask.shape[1] ** .5)
        return mask.reshape(-1, p, p).\
                    repeat_interleave(scale, axis=1).\
                    repeat_interleave(scale, axis=2)


def gen_random_mask(x, mask_ratio):
        N = x.shape[0]
        L = (x.shape[2] // 4) ** 2
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.randn(N, L, device=x.device)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return mask


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, upsampling=False, act_norm=False):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        if upsampling is True:
            stride = 2
            self.conv = MinkowskiConvolutionTranspose(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dimension=2)
        else:
            self.conv = MinkowskiConvolution(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dimension=2,  dilation=dilation)
        

        self.norm = MinkowskiGroupNorm(out_channels, 1e-5)
        self.act = MinkowskiSiLU(inplace=True)
        #self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            #trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y

class BasicConv2d_D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, upsampling=False, act_norm=False):
        super(BasicConv2d_D, self).__init__()
        self.act_norm = act_norm
        if upsampling is True:
            self.conv = nn.Sequential(*[
                nn.Conv2d(in_channels, 4 * out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.PixelShuffle(2)
            ])
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.SiLU(inplace=True) 
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y

class ConvSC(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, downsampling=False, upsampling=False, act_norm=True, is_3d=False):
        super(ConvSC, self).__init__()

        stride = 2 if downsampling is True else 1
        padding = (kernel_size - stride + 1) // 2

        self.conv = BasicConv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, upsampling=upsampling,
                            padding=padding, act_norm=act_norm)

    def forward(self, x):
        y = self.conv(x)
        return y

class ConvSC_D(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, downsampling=False, upsampling=False, act_norm=True, is_3d=False):
        super(ConvSC_D, self).__init__()

        stride = 2 if downsampling is True else 1
        padding = (kernel_size - stride + 1) // 2

        self.conv = BasicConv2d_D(C_in, C_out, kernel_size=kernel_size, stride=stride, upsampling=upsampling,
                            padding=padding, act_norm=act_norm)

    def forward(self, x):
        y = self.conv(x)
        return y



class Encoder(nn.Module):
    def __init__(self, C_in, C_hid, N_S, spatio_kernel):
        samplings = sampling_generator(N_S)
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, spatio_kernel, downsampling=samplings[0]),
            *[ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s) for s in samplings[1:]]
        )

    def forward(self, x):  # B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
            #print(latent.dense()[0].shape)
        return latent, enc1.dense()[0]


class Decoder(nn.Module):
    def __init__(self, C_hid, C_out, N_S, spatio_kernel):
        samplings = sampling_generator(N_S, reverse=True)
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            *[ConvSC_D(C_hid, C_hid, spatio_kernel, upsampling=s) for s in samplings[:-1]],
            ConvSC_D(C_hid, C_hid, spatio_kernel, upsampling=samplings[-1])
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)
        #self.readout = MinkowskiConvolution(in_channels=C_hid, out_channels=C_out, kernel_size=1,dimension=2)
        
    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec)-1):
            hid = self.dec[i](hid)
            #print(hid.dense()[0].shape)
        Y = self.dec[-1](hid + enc1)
        Y = self.readout(Y)

        return Y

class mrla_light_layer(nn.Module):
    """
    multi-head layer attention module: MRLA-light
    when heads = channels, channelwise (Q(K)' is then pointwise(channelwise) multiplication)
    
    Args:
        input_dim: input channel c (output channel is the same)
        heads: number of heads
        dim_perhead: channels per head
        k_size: kernel size of conv1d
        input : [b, c, h, w]
        output: [b, c, h, w]
        
        Wq, Wk: conv1d
        Wv: conv2d
        Q: [b, 1, c]
        K: [b, 1, c]
        V: [b, c, h, w]
    """
    def __init__(self, input_dim, heads=None, dim_perhead=None, k_size=3):
        super(mrla_light_layer, self).__init__()
        self.input_dim = input_dim
        
        if (heads == None) and (dim_perhead == None):
            raise ValueError("arguments heads and dim_perhead cannot be None at the same time !")
        elif dim_perhead != None:
            heads = int(input_dim / dim_perhead)
        else:
            heads = heads
        self.heads = heads
        self.k_size = k_size
    
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.Wq = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.Wk = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.Wv = nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1, groups=input_dim, bias=False) 
        self._norm_fact = 1 / math.sqrt(input_dim / heads)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()
        # feature descriptor on the global spatial information
        y = self.avg_pool(x) # [b, c, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2) # [b, 1, c]
        
        Q = self.Wq(y) # Q: [b, 1, c] 
        K = self.Wk(y) # K: [b, 1, c]
        V = self.Wv(x) # V: [b, c, h, w]
        Q = Q.view(b, self.heads, 1, int(c/self.heads)) # [b, g, 1, c/g]
        K = K.view(b, self.heads, 1, int(c/self.heads)) # [b, g, 1, c/g]
        V = V.view(b, self.heads, int(c/self.heads), h, w) # [b, g, c/g, h, w]
        # Q.is_contiguous()
        
        attn = torch.einsum('... i d, ... j d -> ... i j', Q, K) * self._norm_fact
        # attn.size() # [b, g, 1, 1]
    
        attn = self.sigmoid(attn.view(b, self.heads, 1, 1, 1)) # [b, g, 1, 1, 1]
        output = V * attn.expand_as(V) # [b, g, c/g, h, w]
        output = output.view(b, c, h, w)
        
        return output    

class sla_layer(nn.Module):
    """
    multi-head layer attention module: MRLA-light
    when heads = channels, channelwise (Q(K)' is then pointwise(channelwise) multiplication)
    
    Args:
        input_dim: input channel c (output channel is the same)
        heads: number of heads
        dim_perhead: channels per head
        k_size: kernel size of conv1d
        input : [b, c, h, w]
        output: [b, c, h, w]
        
        Wq, Wk: conv1d
        Wv: conv2d
        Q: [b, 1, c]
        K: [b, 1, c]
        V: [b, c, h, w]
    """
    def __init__(self, input_dim, heads=None, dim_perhead=None, k_size=3):
        super(sla_layer, self).__init__()
        self.input_dim = input_dim
        
        if (heads == None) and (dim_perhead == None):
            raise ValueError("arguments heads and dim_perhead cannot be None at the same time !")
        elif dim_perhead != None:
            heads = int(input_dim / dim_perhead)
        else:
            heads = heads
        self.heads = heads
        self.k_size = k_size

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.Wq = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.Wk = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.Wv = nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1, groups=input_dim, bias=False) 
        self._norm_fact = 1 / math.sqrt(input_dim / heads)
        self.sigmoid = nn.Sigmoid()
        #self.lambda_t = nn.Parameter(torch.randn(input_dim, 1, 1))  
        
    def forward(self, x, en_x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()
        # feature descriptor on the global spatial information
        y = self.avg_pool(x) # [b, c, 1, 1]
        en_y = self.avg_pool(en_x)
        en_y = en_y.squeeze(-1).transpose(-1,-2)
        Q = self.Wq(en_y) # Q: [b, 1, c] 
        K = self.Wk(en_y) # K: [b, 1, c]
        V = self.Wv(x) # V: [b, c, h, w]
        Q = Q.view(b, self.heads, 1, int(c/self.heads)) # [b, g, 1, c/g]
        K = K.view(b, self.heads, 1, int(c/self.heads)) # [b, g, 1, c/g]
        V = V.view(b, self.heads, int(c/self.heads), h, w) # [b, g, c/g, h, w]
        # Q.is_contiguous()
        
        attn = torch.einsum('... i d, ... j d -> ... i j', Q, K) * self._norm_fact
        # attn.size() # [b, g, 1, 1]
        
        attn = self.sigmoid(attn.view(b, self.heads, 1, 1, 1)) # [b, g, 1, 1, 1]
        output = V * attn.expand_as(V) # [b, g, c/g, h, w]
        output = output.view(b, c, h, w)
        return output + x
        

class mrla_module(nn.Module):
    dim_perhead = 32
    def __init__(self, input_dim):
        super(mrla_module, self).__init__()
        self.mrla = mrla_light_layer(input_dim=input_dim, dim_perhead=self.dim_perhead)
        self.lambda_t = nn.Parameter(torch.randn(input_dim, 1, 1))  
        
    def forward(self, xt, ot_1):
        attn_t = self.mrla(xt)
        out = attn_t + self.lambda_t.expand_as(ot_1) * ot_1 
        return out
        

class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, drop_path=0.1, layer_scale_init_value=1e-6,ls_init_value=1e-6 ):
        super(ConvBlock,self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dwconv = nn.Conv2d(in_dim,in_dim,kernel_size=7,padding=3,groups=in_dim) #depthwise conv
        self.norm = nn.LayerNorm(in_dim, 1e-6) #nn.GroupNorm(8,in_dim) 
        self.pwconv1 = nn.Linear(in_dim, 8 * in_dim) 
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(8 * in_dim, in_dim) 
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gamma = nn.Parameter(ls_init_value * torch.ones(in_dim)) if ls_init_value > 0 else None
        if in_dim != out_dim:
            self.reduction = nn.Conv2d(
                in_dim, out_dim, kernel_size=1, stride=1, padding=0)
        self.apply(self._init_weights)

    
    def _init_weights(self, m):
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self,x):
        input = x 
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        x = self.gamma.reshape(1, -1, 1, 1) * x 
        x = input + self.drop_path(x)
        return x if self.in_dim == self.out_dim else self.reduction(x)

class Mid_ConvNeXt(nn.Module):
    def __init__(self, channel_in, channel_hid, N2,  drop_path=0.):
        super(Mid_ConvNeXt, self).__init__()
        self.N2 = N2
        self.channel = channel_hid
        dpr = [  # stochastic depth decay rule
            x.item() for x in torch.linspace(1e-2, drop_path, self.N2 * 2)]
        enc_layers = [ConvBlock(channel_in, channel_hid, drop_path=dpr[0])]
        for i in range(1, N2-1):
            enc_layers.append(ConvBlock(channel_hid, channel_hid, drop_path=dpr[i]))
        enc_layers.append(ConvBlock(channel_hid, channel_in, drop_path=dpr[-1]))
        self.enc = nn.Sequential(*enc_layers)
        #self.mrla = mrla_module(channel_hid)
        #self.bn_mrla = nn.GroupNorm(8,self.channel)
    def forward(self, x):
        z = x
        for i in range(self.N2):
            z = self.enc[i](z)
        return z
        # z = x
        # z = self.enc[0](z)
        # identity = z
        # for i in range(1, self.N2-1):
        #     z = self.enc[i](z)
        #     z = z + self.bn_mrla(self.mrla(z,identity))
        #     identity = z 
        # z = self.enc[-1](z)
        #return z


class GABlock(nn.Module):
    def __init__(self, in_channels, out_channels, mlp_ratio=8., drop=0.0, drop_path=0.0):
        super(GABlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.block = GASubBlock(in_channels, kernel_size=21, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path, act_layer=nn.GELU)

        if in_channels != out_channels:
            self.reduction = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            
    def forward(self, x):
        z = self.block(x)
        return z if self.in_channels == self.out_channels else self.reduction(z)



class Mid_GANet(nn.Module):
    def __init__(self, channel_in, channel_hid, N2, mlp_ratio=4., drop=0.0, drop_path=0.1):
        super(Mid_GANet, self).__init__()

        self.N2 = N2
        enc_layers = [GABlock(channel_in, channel_hid, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)]
        for i in range(1, N2-1):
            enc_layers.append(GABlock(channel_hid, channel_hid, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path))
        enc_layers.append(GABlock(channel_hid, channel_in, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path))
        self.enc = nn.Sequential(*enc_layers)

    def forward(self, x):
        z = x
        for i in range(self.N2):
            z = self.enc[i](z)
        B,T,C,H,W = z.shape()
        y = z.reshape(B, T, C, H, W)
        return y


class Mid_IncepNet(nn.Module):
    def __init__(self, channel_in, channel_hid, N2, incep_ker=[3,5,7,11], groups=8, **kwargs):
        super(Mid_IncepNet, self).__init__()
        
        self.N2 = N2
        enc_layers = [gInception_ST(channel_in, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1,N2-1):
            enc_layers.append(gInception_ST(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
        enc_layers.append(gInception_ST(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))


        dec_layers = [gInception_ST(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1,N2-1):
            dec_layers.append(gInception_ST(2*channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
        dec_layers.append(gInception_ST(2*channel_hid, channel_hid//2, channel_in, incep_ker= incep_ker, groups=groups))


        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B,T,C,H,W = x.shape
        x = x.reshape(B,T*C,H,W)

        # encoder
        skips = []
        z = x
        for i in range(self.N2):
            z = self.enc[i](z)
            if i < self.N2-1:
                skips.append(z)

        # decoder
        z = self.dec[0](z)
        for i in range(1,self.N2):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1) )

        y = z.reshape(B,T,C,H,W)
        return y


class Mid_SlaNet(nn.Module):
    def __init__(self, channel_in, channel_hid, N2, drop_path=0.1,**kwargs):
        super(Mid_SlaNet, self).__init__()
        self.channel = channel_hid
        self.N2 = N2
        dpr = [  # stochastic depth decay rule
            x.item() for x in torch.linspace(1e-2, drop_path, self.N2 * 2)]
        enc_layers = [ConvBlock(channel_in, channel_hid, drop_path=dpr[0])]
        for i in range(1,N2-1):
            enc_layers.append(ConvBlock(channel_hid,channel_hid, drop_path=dpr[i]))
        enc_layers.append(ConvBlock(channel_hid,channel_hid,drop_path=dpr[N2-1]))

        
        dec_layers = [ConvBlock(channel_hid,channel_hid,drop_path=dpr[N2])]
        for i in range(1,N2-1):
            dec_layers.append(ConvBlock(channel_hid,channel_hid,drop_path=dpr[N2+i]))
        dec_layers.append(ConvBlock(channel_hid,channel_in,drop_path=dpr[2*N2-1]))
        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)
        self.sla = sla_layer(channel_hid, dim_perhead=16)
        self.bn_sla = nn.GroupNorm(8,self.channel)
        #self.conv_last = nn.Conv2d(channel_hid,channel_in,kernel_size=1,padding=0)
    def forward(self, x):
        # encoder
        skips = []
        z = x
        for i in range(self.N2):
            z = self.enc[i](z)
            if i < self.N2:
                skips.append(z)
        
        # decoder
        for i in range(self.N2 - 1):
            z = self.dec[i](z)
            z = z + self.bn_sla(self.sla(z,skips[-i+self.N2-1]))
        z = self.dec[-1](z)
        return z


class SimVP_Model(nn.Module):
    def __init__(self, in_shape, hid_S=16, hid_T=256, N_S=4, N_T=4, model_type='convnext',
        mlp_ratio=8., drop=0.0, drop_path=0.0, spatio_kernel_enc=3, spatio_kernel_dec=3, pre_seq_length=10, aft_seq_length=10, **kwargs):
        super(SimVP_Model, self).__init__()
        T, C, H, W = in_shape
        
        self.enc = Encoder(C, hid_S, N_S, spatio_kernel_enc)
        self.dec = Decoder(hid_S, C, N_S, spatio_kernel_dec)
        
        if model_type == 'IncepU':
            self.hid = Mid_IncepNet(T*hid_S, hid_T, N_T)
        elif model_type == 'convnext':
            self.hid = Mid_ConvNeXt(T*hid_S, hid_T, N_T, drop_path=0.)
        elif model_type == 'ganet':
            self.hid = Mid_GANet(T*hid_S, hid_T, N_T, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type == 'slanet':
            self.hid = Mid_SlaNet(T*hid_S, hid_T, N_T, drop_path=0.)
    
    def forward(self, x_raw, maskratio=0):
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B*T, C, H, W)
        mask = gen_random_mask(x, maskratio)
        mask = upsample_mask(mask, 4)
        mask = mask.unsqueeze(1).type_as(x)
        x *= (1.-mask)
        x = to_sparse_all(x)
        embed, skip = self.enc(x)
        embed = embed.dense()[0]
        _, C_, H_, W_ = embed.shape
        z = embed.view(B, T, C_, H_, W_).reshape(B, T*C_, H_, W_)
        hid = self.hid(z)
        hid = hid.reshape(B*T, C_, H_, W_)
        Y = self.dec(hid,skip)
        Y = Y.reshape(B, T, C, H, W)
        return Y

if __name__ == '__main__':
    
    model =  SimVP_Model(in_shape=[10,1,64,64]).cuda()
    tensor = torch.randn(16,10,1,64,64).cuda()
    #tensor = to_sparse(tensor)
    model(tensor)
    print('ok')