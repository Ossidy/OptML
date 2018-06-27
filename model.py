import torch
from torch import nn, optim 
from torch.autograd import Variable
from torch.nn import functional as F 
import numpy as np
import os
from collections import OrderedDict


class ConvSet(nn.Module):
    def __init__(self, in_channels, out_channels, kernersize, stride, padding, bias=False, 
                 batchNorm=True, activation='ReLU', name='Default_net'):
        
        super(ConvSet, self).__init__()
        
        assert activation in ['ReLU', 'LeakyReLU'], "Activation methods not implemented!!!"
        self.convSet = nn.Sequential()
        self.nameList = []
        
        conv_name = name + '.{}-{}.conv'.format(in_channels, out_channels)
        activ_name = name + '.{}.' + activation
        activ_name = activ_name.format(out_channels)
        if batchNorm:
            batch_name = name + '.{}.batchnorm'.format(out_channels)
            
        self.convSet.add_module(conv_name,
                               nn.Conv2d(in_channels, out_channels, kernersize, stride, padding, bias=bias))
        self.nameList.append(conv_name)
        
        if batchNorm:
            self.convSet.add_module(batch_name,
                                   nn.BatchNorm2d(out_channels))
            self.nameList.append(batch_name)
        
        if activation == 'ReLU':
            self.convSet.add_module(activ_name,
                                   nn.ReLU(inplace=True))
        elif activation == 'LeakyReLU':
            self.convSet.add_module(activ_name,
                                   nn.LeakyReLU(0.2, inplace=True))
        self.nameList.append(activ_name)

    def forward(self, x):
        x = self.convSet(x)
        return x
    
    def getEntriesAndNames(self):
        moduleList = list(self.convSet)
        return moduleList, self.nameList


class TConvSet(nn.Module):
    def __init__(self, in_channels, out_channels, kernersize, stride, padding, bias=False, 
                 batchNorm=True, activation='ReLU', name='Default_net'):
        
        super(TConvSet, self).__init__()
        
        assert activation in ['ReLU', 'LeakyReLU'], "Activation methods not implemented!!!"
        self.convSet = nn.Sequential()
        self.nameList = []
        
        conv_name = name + '.{}-{}.transconv'.format(in_channels, out_channels)
        activ_name = name + '.{}.' + activation
        activ_name = activ_name.format(out_channels)
        if batchNorm:
            batch_name = name + '.{}.batchnorm'.format(out_channels)
            
        self.convSet.add_module(conv_name,
                               nn.ConvTranspose2d(in_channels, out_channels, kernersize, stride, padding, bias=bias))
        self.nameList.append(conv_name)
        
        if batchNorm:
            self.convSet.add_module(batch_name,
                                   nn.BatchNorm2d(out_channels))
            self.nameList.append(batch_name)
        
        if activation == 'ReLU':
            self.convSet.add_module(activ_name,
                                   nn.ReLU(inplace=True))
        elif activation == 'LeakyReLU':
            self.convSet.add_module(activ_name,
                                   nn.LeakyReLU(0.2, inplace=True))
        self.nameList.append(activ_name)

    def forward(self, x):
        x = self.convSet(x)
        return x
    
    def getEntriesAndNames(self):
        moduleList = list(self.convSet)
        return moduleList, self.nameList
    

class GeomEncoder(nn.Module):
    def __init__(self, encode_dim=20, in_channels=1, imsize=64, nfeat=64,  extra_layer=0, batchNorm=True, activation='ReLU'):
        
        super(GeomEncoder, self).__init__()  
        # initial input
        self.initial = ConvSet(in_channels, nfeat, 4, 2, 1, False, 
                               batchNorm=batchNorm, activation=activation, name='initial')
        
        # pyramid structure
        encoder_list = []
        name_list = []
        c_imsize, c_feat = imsize / 2, nfeat
        
        ind = 0       
        while c_imsize >= 4:
            in_feat = c_feat
            out_feat = c_feat * 2
            
            ind += 1
            layer_name = 'pyramid_' + str(ind)
            convnet = ConvSet(in_feat, out_feat, 4, 2, 1, bias=False, 
                                            batchNorm=batchNorm, activation=activation, name=layer_name)
            
            entries, names = convnet.getEntriesAndNames()
            encoder_list.extend(entries)
            name_list.extend(names)
        
            c_feat *= 2
            c_imsize = c_imsize / 4
        # Tensor[None, 256, 8, 8]
 
        # final convolutional layer, out_feat=20 is to be changed
        final_conv = ConvSet(c_feat, 20, 4, 2, 1, bias=False, 
                                            batchNorm=batchNorm, activation=activation, name='final_conv')
        entries, names = final_conv.getEntriesAndNames()
        encoder_list.extend(entries)
        name_list.extend(names)
        
        encoder_dict = OrderedDict(zip(name_list, encoder_list))
        self.encoder =  nn.Sequential(encoder_dict)
        # Tensor[None, 20, 4, 4] -> 320
        
        self.final_layer = nn.Sequential()
        self.final_layer.add_module('encoded',
                                    nn.Linear(320, encode_dim))
        
    def forward(self, x):
        x = self.initial(x)
        x = self.encoder(x)
        x = self.final_layer(x.view(2, 320))
        return x
    
    def get_num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params

    
class GeomDecoder(nn.Module):
    def __init__(self, encode_dim=20, noise_dim=20, out_channels=1, imsize=64, nfeat=64, extra_layer=0, batchNorm=True, activation='ReLU'):
        
        super(GeomDecoder, self).__init__() 
        
        self.noise_dim = noise_dim
        self.encode_dim = encode_dim
        
        cngf, tisize = nfeat//2, 4
        while tisize != imsize:
            cngf = cngf * 2
            tisize = tisize * 2
            
        nz = noise_dim + encode_dim
        
        # initial input
        self.initial = TConvSet(nz, cngf, 4, 1, 0, False, 
                               batchNorm=batchNorm, activation=activation, name='initial')
        
        # pyramid structure
        decoder_list = []
        name_list = []        
        c_imsize, c_feat = 4, cngf
        
        ind = 0
        while c_imsize < imsize // 2:
            ind += 1
            layer_name = 'pyramid_' + str(ind)
            tconvnet = TConvSet(c_feat, c_feat//2, 4, 2, 1, bias=False,
                               batchNorm=batchNorm, activation=activation, name=layer_name)
            
            entries, names = tconvnet.getEntriesAndNames()
            decoder_list.extend(entries)
            name_list.extend(names)
            
            c_feat = c_feat // 2
            c_imsize = c_imsize * 2
        
        # extra layer
        ind = 0
        for i in range(extra_layer):
            ind += 1
            layer_name = 'extra_layer_' + str(ind)
            extra_tconvnet = ConvSet(c_feat, c_feat, 3, 1, 1, bias=False, name=layer_name)
            
            entries, names = extra_tconvnet.getEntriesAndNames()
            decoder_list.extend(entries)
            name_list.extend(names)       
        
        # final output layer
        final_tconvnet = TConvSet(c_feat, out_channels, 4, 2, 1, bias=False,
                   batchNorm=batchNorm, activation=activation, name='final_layer')

        entries, names = final_tconvnet.getEntriesAndNames()
        decoder_list.extend(entries)
        name_list.extend(names)
        
        decoder_dict = OrderedDict(zip(name_list, decoder_list))
        self.decoder =  nn.Sequential(decoder_dict)   
        
        
    def forward(self, *args):
        if len(args) == 2:
            x, noise = args[0], args[1]
            assert noise.shape[1] == self.noise_dim and x.shape[1] == self.encode_dim, "Dimension of noise or encoded vector does not match"
            x = torch.cat((x, noise), dim=1)
        else:
            x = args[0]
            assert self.noise_dim == 0, "Model needs noise with dimension dim={} as input".format(self.noise_dim)
            assert x.shape[1] == self.encode_dim, "Dimension of encoded vector does not match"
        
        x = x.view((x.shape[0], x.shape[1], 1, 1))
        x = self.initial(x)        
        x = self.decoder(x)
        return x
    
    def get_num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params