import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.densenet import _DenseBlock
from torchvision.models.densenet import _DenseLayer
from torchvision.models.densenet import *
from collections import OrderedDict
from fastai2.callback.hook import *
from fastai2.layers import *
from fastai2.imports import *
from scipy import io as scio
from pathlib import Path
Path.ls = lambda x: list(x.iterdir())
from functools import partial
cuda = torch.cuda.is_available()

class ConvBlock(nn.Sequential):
    """ Creates the conv2D + BatchNorm + LeakyReLu sequence for down or upsampling """
    def __init__(self, ni, nf, n_convs=2, ks=5, stride=1, padding=None, bias=False, 
                 norm_type=nn.BatchNorm2d, bn_first=True, act_fn=nn.LeakyReLU, 
                 init_fn= partial(nn.init.kaiming_uniform_, a=0.01), **kwargs):
        if padding is None:
            padding = ((ks-1)//2)
        layers = []
        nh = ni//2 if nf==1 else nf
        for i in range(n_convs):
            ni = ni if i==0 else nh
            if  i==n_convs-1: nh = nf 
            conv = nn.Conv2d(ni, nh, kernel_size=ks, stride=stride, padding=padding, bias=bias, **kwargs)
            init_fn(conv.weight)
            
            if act_fn==nn.Sigmoid and i==n_convs-1:
                l = [act_fn()]
            else:
                bn = norm_type(nf) if nf!=1 else norm_type(nh)
                act = act_fn() if act_fn!=nn.Sigmoid else nn.LeakyReLU()
                l = [act, bn]
            
            if bn_first: l.reverse()
            layers+= [conv]+l
        
        super().__init__(*layers) 

class TransitionBlock(nn.Sequential):
    def __init__(self, ni, nf, upsample=False, pxshuff=False, pool_fn=nn.AvgPool2d):
        super(TransitionBlock, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(ni))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(ni, nf,
                                          kernel_size=1, stride=1, bias=False))
        if upsample:
            if pxshuff:
                self.add_module('PixShuff', PixelShuffle_ICNR(nf))
            else:
                self.add_module('deconv', nn.ConvTranspose2d(nf,nf, kernel_size=2, stride=2, padding=0, bias=True))
        else: 
            self.add_module('pool', pool_fn(kernel_size=2, stride=2))

class PixelShuffle_ICNR(nn.Sequential):
    "Upsample by `scale` from `ni` filters to `nf` (default `ni`), using `nn.PixelShuffle`."
    def __init__(self, ni, nf=None, scale=2, blur=False, norm_type=NormType.Weight, act_cls=defaults.activation):
        super().__init__()
        nf = ifnone(nf, ni)
        layers = [ConvLayer(ni, nf*(scale**2), ks=1, norm_type=norm_type, act_cls=act_cls, bias_std=.01),
                  nn.PixelShuffle(scale)]
        layers[0][0].weight.data.copy_(icnr_init(layers[0][0].weight.data))
        if blur: layers += [nn.ReplicationPad2d((1,0,1,0)), nn.AvgPool2d(2, stride=1)]
        super().__init__(*layers)

class RedNet(nn.Module):
    
    def __init__(self, blocks=[3,3,4], stem_szs=[64,64], c_in=3, loops=1, bn_size=4, 
                 pxshuff=False, pool='AvgPool2d', noise=.05):
        
        super(RedNet, self).__init__()
        
        self.loops = loops
        self.noise = noise
        pool_fn = getattr(nn, pool)
        stem_szs = [c_in+1, *stem_szs]
        block_szs = [stem_szs[-1]*2**(i+1) if i<3 else 64*2**2 for i in range(len(blocks))]

        ### Build Encoder ##
        encoder = nn.Sequential()
        
        ## Stem
        for i in range(len(stem_szs)-1):
            encoder.add_module('ConvBlock-'+str(i+1), ConvBlock(stem_szs[i],stem_szs[i+1], ks=5, stride=1))
            encoder.add_module('Pool-'+str(i+1), nn.MaxPool2d(3,2,1))
        ## DenseBlocks Down
        for i in range(len(blocks)):
            ni = stem_szs[-1] if i==0 else block_szs[i]
            up = True if i==(len(blocks)-1) else False
            nf = block_szs[i-1] if up else block_szs[i+1]
            encoder.add_module('DenseBlock-'+str(i+1), _DenseBlock(num_layers= blocks[i], num_input_features= ni, 
                                                                   bn_size= bn_size, growth_rate= block_szs[i],
                                                                   drop_rate=0))
            encoder.add_module('Transition-'+str(i+1), TransitionBlock(ni= (ni+block_szs[i]*blocks[i]),
                                                                            nf= nf, upsample=up, pxshuff=pxshuff, 
                                                                            pool_fn=pool_fn))
        self.encoder = encoder
        self.hooks = hook_outputs([l for l in self.encoder.modules() 
                                            if isinstance(l, _DenseBlock) or isinstance(l, ConvBlock)])
        
        # Run an input through the encoder to hook the outputs and later get their shapes
        encoder(torch.randn(1,c_in+1,128,128))
        
        ### Build Decoder ##
        decoder = nn.Sequential()
        
        ## Denseblocks up
        for i in reversed(range(len(blocks)-1)):
            layers = []
            ni = block_szs[i] + self.hooks[len(stem_szs)+i-1].stored.shape[1]
            nf = stem_szs[-1] if i==0 else block_szs[i-1]

            layers += [_DenseBlock(num_layers= blocks[i], num_input_features= ni, 
                                            bn_size= bn_size, growth_rate= block_szs[i],
                                            drop_rate=0)]
            layers += [TransitionBlock(ni= (ni+block_szs[i]*blocks[i]), nf= nf, upsample=True, pxshuff=pxshuff)]
            
            decoder.add_module('DenseBlock+Up_'+str(len(blocks)-1-i), nn.Sequential(*layers))
        ## Head
        for i in reversed(range(1, len(stem_szs))):
            layers = []
            ni = stem_szs[i]+self.hooks[i-1].stored.shape[1]
            nf = 1 if i==1 else stem_szs[i]
            act = nn.Sigmoid if i==1 else nn.LeakyReLU
            layers+= [ConvBlock(ni=ni, nf=nf, ks=5, stride=1, act_fn=act)]
            if i!=1:
                if pxshuff:
                    layers += [PixelShuffle_ICNR(nf)]
                else:
                    layers += [nn.ConvTranspose2d(nf,nf, kernel_size=2, stride=2, padding=0, bias=True)]
            decoder.add_module('ConvBlock+Up-'+str(len(stem_szs)-i), nn.Sequential(*layers))
        
        self.decoder = decoder
        self.hooks2 = hook_outputs([l for l in self.decoder.modules() 
                                            if isinstance(l, _DenseBlock) or isinstance(l, ConvBlock)])
        
    def _do_forward(self, catx):

        """ Performs one forward pass in model """

        # DownSampling
        y = self.encoder(catx)
        skips = list(reversed([h.stored for h in self.hooks][:-1]))
        #UpSampling
        for skip, n in zip(skips , self.decoder._modules.keys()):
            ss = skip.shape[-2:]
            if ss != y.shape[-2:]:
                y = F.interpolate(y, ss, mode='nearest')
            y = torch.cat((y,skip), 1)
            m = self.decoder._modules[n]
            y = m(y)
        return y
    
    def forward(self,x):
        
        """ Performs the looped forward passes with catenated recurent outputs"""
        
        x = torch.clamp( (x + torch.randn_like(x) * self.noise), 0., 1.)
        init_edge = torch.zeros_like(x)[:,0,:,:].unsqueeze(1)
        self.edges = [init_edge]
        for i in range(self.loops):
            catx = torch.cat((x, self.edges[-1]), 1)
            self.edges.append( self._do_forward(catx) )
        return self.edges[1:]