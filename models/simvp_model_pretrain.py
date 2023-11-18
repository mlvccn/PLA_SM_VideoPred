import torch
from torch import nn
#from timm.models.layers import trunc_normal_
from modules import GASubBlock
from API.utils import MinkowskiDropPath, MinkowskiGRN, MinkowskiLayerNorm, MinkowskiGroupNorm
from typing import Union
import numpy as np
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
            #self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,output_padding=stride //2 )
            self.conv = MinkowskiConvolutionTranspose(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dimension=2)
        else:
            #self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
            self.conv = MinkowskiConvolution(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dimension=2,  dilation=dilation)
        
        #self.norm = nn.GroupNorm(2, out_channels)
        #self.act = nn.LeakyReLU(0.2, inplace=True)
        #self.norm = MinkowskiLayerNorm(out_channels, 1e-6)
        self.norm = MinkowskiGroupNorm(out_channels, 1e-6)
        self.act = MinkowskiSiLU()
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
            stride = 2
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,output_padding=stride //2 )
            #self.conv = MinkowskiConvolutionTranspose(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dimension=2)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
            # self.conv = MinkowskiConvolution(
            #     in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dimension=2,  dilation=dilation)

        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.SiLU() #nn.LeakyReLU(0.2, inplace=True)  
        # self.norm = MinkowskiLayerNorm(out_channels, 1e-6)
        # self.act = MinkowskiGELU()
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

class GroupConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,groups,act_norm=False):
        super(GroupConv2d, self).__init__()
        self.act_norm=act_norm
        if in_channels%groups != 0:
            groups=1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,groups=groups)
        self.norm = nn.GroupNorm(groups,out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y


class gInception_ST(nn.Module):
    def __init__(self, C_in, C_hid, C_out, incep_ker = [3,5,7,11], groups = 8):        
        super(gInception_ST, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)
        
        layers = []
        for ker in incep_ker:
            layers.append(GroupConv2d(C_hid, C_out, kernel_size=ker, stride=1, padding=ker//2, groups=groups, act_norm=True))
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y += layer(x)
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
        return latent.dense()[0], enc1.dense()[0]


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


class Block(nn.Module):
    """ Sparse ConvNeXtV2 Block. 

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, in_dim, out_dim, drop_path=0., D=2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dwconv = MinkowskiDepthwiseConvolution(in_dim, kernel_size=7, bias=True, dimension=D)
        self.norm = MinkowskiLayerNorm(in_dim, 1e-6)
        self.pwconv1 = MinkowskiLinear(in_dim, 4 * out_dim)   
        self.act = MinkowskiGELU()
        self.pwconv2 = MinkowskiLinear(4 * out_dim, out_dim)
        self.grn = MinkowskiGRN(4  * out_dim)
        self.drop_path = MinkowskiDropPath(drop_path)
    
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        if self.in_dim == self.out_dim:
            x = input + self.drop_path(x)
        return x


class Mid_ConvNeXt(nn.Module):
    def __init__(self, channel_in, channel_hid, N2,  drop_path=0.):
        super(Mid_ConvNeXt, self).__init__()

        self.N2 = N2
        enc_layers = [Block(channel_in, channel_hid, drop_path=drop_path)]
        for i in range(1, N2-1):
            enc_layers.append(Block(channel_hid, channel_hid, drop_path=drop_path))
        enc_layers.append(Block(channel_hid, channel_in, drop_path=drop_path))
        self.enc = nn.Sequential(*enc_layers)

    def forward(self, x):
        
        # B, T, C, H, W = x.shape
        # x = x.reshape(B, T*C, H, W)
        # mask = gen_random_mask(x, maskratio)
        # mask = upsample_mask(mask, 8)
        # mask = mask.unsqueeze(1).type_as(x)
        # x *= (1.-mask)
        #x = to_sparse(x)  #1,640,16,16
        z = x
        for i in range(self.N2):
            z = self.enc[i](z)
            #print(z.dense()[0].shape)
        #y = z.reshape(B, T, C, H, W)
        return z


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
        B,T,C,H,W = x.shape
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


class SimVP_Model_Pretrain(nn.Module):
    def __init__(self, in_shape, hid_S=16, hid_T=256, N_S=4, N_T=4, model_type='',
        mlp_ratio=8., drop=0.0, drop_path=0.0, spatio_kernel_enc=3, spatio_kernel_dec=3, pre_seq_length=10, aft_seq_length=10, **kwargs):
        super(SimVP_Model_Pretrain, self).__init__()
        T, C, H, W = in_shape
        
        self.enc = Encoder(C, hid_S, N_S, spatio_kernel_enc)
        self.dec = Decoder(hid_S, C, N_S, spatio_kernel_dec)
        
        if model_type == 'IncepU':
            self.hid = Mid_IncepNet(T*hid_S, hid_T, N_T)
        elif model_type == 'convnext':
            self.hid = Mid_ConvNeXt(T*hid_S, hid_T, N_T, drop_path=0.)
        elif model_type == 'ganet':
            self.hid = Mid_GANet(T*hid_S, hid_T, N_T, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
    
    def forward(self, x_raw, maskratio=0.5):
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B*T, C, H, W)
        mask = gen_random_mask(x, maskratio)
        mask = upsample_mask(mask, 4)
        mask = mask.unsqueeze(1).type_as(x)
        x *= (1.-mask)

        x = to_sparse_all(x)
        embed, skip = self.enc(x)
        
        Y = self.dec(embed,skip)
        Y = Y.reshape(B, T, C, H, W)
        return Y

if __name__ == '__main__':
    # tensor = torch.randn(16,1,64,64).cuda()
    # mask = gen_random_mask(tensor, 0.9)
    # mask = upsample_mask(mask, 8)
    # mask = mask.unsqueeze(1).type_as(tensor).cuda()
    # tensor *= (1.-mask)
    # x = to_sparse(tensor)  #1,640,16,16
    # dense = x.dense()[0]
    # dwconv = MinkowskiDepthwiseConvolution(320, kernel_size=7, bias=True, dimension=2).cuda()
    # x = dwconv(x)
    # dense_final = x.dense()[0]
    # print('ok') 
    model =  SimVP_Model(in_shape=[10,1,64,64])
    tensor = torch.randn(16,1,64,64).cuda()
    tensor = to_sparse(tensor)
    model(tensor)
    print('ok')