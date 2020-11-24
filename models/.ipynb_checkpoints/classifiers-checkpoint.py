# +
import torch
import torch.nn as nn
import torchvision.models as models

import os
import numpy as np


# -

class BaseClassifier(nn.Module):
    def __init__(self):
        super(BaseClassifier, self).__init__()
        self.name = 'BaseClassifier'
        
    def forward(self):
        raise NotImplementedError("Forward function to be implemented.")
    
    def save(self, root_path):
        file_path = os.path.join(root_path, self.name+'_best.pt')
        torch.save(self.state_dict(), file_path)
        
    def load(self, ck_path):
        self.load_state_dict(torch.load(ck_path))


class EvalCompoundResNet(nn.Module):
    def __init__(self, weight_path, module_names=None):
        super(EvalCompoundResNet, self).__init__()
        self.weight_path = weight_path
        if module_names is None:
            self.module_names = ['model_eyeglasses_male_smiling_young.pt', 'model_bags_under_eyes_bald_big_lips_narrow_eyes.pt',
                                'model_big_nose_black_hair_blond_hair_brown_hair.pt', 'model_high_cheekbones_pale_skin_pointy_nose_sideburns.pt']
            #self.module_names = ['model_0_1_2_3.pt', 'model_4_5_6_7.pt']
        else:
            self.module_names = module_names
        
        # append the module name to the weight_path
        self.module_names = [os.path.join(weight_path, module_name) for module_name in self.module_names]
        self.num_module = len(self.module_names)
        
        self.module_list = nn.ModuleList()
        for i in range(self.num_module):
            m = TrainCompoundResNet(['dummy_key'] * 4)
            
            state_dict = torch.load(self.module_names[i])
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove 'module.' of dataparallel
                new_state_dict[name] = v
    
            m.load_state_dict(new_state_dict)
            m.requires_grad = False
            self.module_list.append(m)
            
    def forward(self, x):
        out = []
        for i in range(self.num_module):
            ot = self.module_list[i](x)
            out.append(ot)
        return torch.cat(out, 1)
    
    def predict(self, x):
        pred_attr = self.forward(x)
        preds = torch.sigmoid(pred_attr).detach().cpu().numpy()
        return preds
    
    def predict_quantize(self, x):
        pred_attr = self.forward(x)
        preds = (torch.sigmoid(pred_attr) > 0.5).long().detach().cpu().numpy()
        return preds


class TrainCompoundResNet(BaseClassifier):
    def __init__(self, selected_attr):
        super(TrainCompoundResNet, self).__init__()
        self.selected_attr = selected_attr
        self.num_attr = len(selected_attr)
        self.resnet_list = nn.ModuleList()
        
        for i in range(self.num_attr):
            m = models.resnet50()
            m.fc = nn.Linear(2048, 1)
            self.resnet_list.append(m)
        
        '''
        m = models.resnet50()
        m.fc = nn.Linear(2048, 10)
        self.resnet_list.append(m)
        '''
        
    def forward(self, x):
        attr_pred = []
        for i in range(self.num_attr):
            out_i = self.resnet_list[i](x)
            attr_pred.append(out_i)
        #landmark_reg = self.resnet_list[-1](x)
        
        return torch.cat(attr_pred, 1)#, landmark_reg


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, act=None, norm=None, stride=1, dilation=1):
        super(ConvBlock, self).__init__()
        final_padding = ((kernel_size-1) * dilation ) // 2
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size,
                              padding=final_padding, stride=stride, dilation=dilation, bias=True)
        
        if act == 'relu':
            self.act_fn = nn.ReLU()
        elif act == 'lrelu':
            self.act_fn = nn.LeakyReLU(0.1, True)
        elif act == 'none':
            self.act_fn = None
            
        if norm == 'bn':
            self.norm = nn.BatchNorm2d()
        elif norm == 'none':
            self.norm = None
    
    def forward(self, x):
        out = self.conv(x)
        if self.act_fn is not None:
            out = self.act_fn(out)
        if self.norm is not None:
            out = self.norm(out)
        return out


class Classifier(nn.Module):
    def __init__(self, input_size, channel_size, out_dim):
        super(Classifier, self).__init__()
        self.input_size = input_size
        self.channel_size = channel_size
        self.out_dim = out_dim
        
        self.avg_pool = nn.AvgPool2d(2, stride=2)
        self.nets = [nn.Conv2d(3, channel_size, 7, stride=1, padding=3, bias=True)]
    
        for level in range(int(np.log2(input_size//4))):
            self.nets.append(nn.Sequential(ConvBlock(channel_size, channel_size, 3, act='relu', norm='none')))
            self.nets.append(self.avg_pool)
        self.nets = nn.Sequential(*self.nets)
        
        self.fc = nn.Sequential(nn.Linear(channel_size*4*4, 100),
                               nn.ReLU(),
                               nn.Linear(100, out_dim))
        
    def forward(self, x):
        feat = self.nets(x).view(x.size(0), -1)
        out = self.fc(feat)
        return out


class TrainResNet(BaseClassifier):
    def __init__(self, selected_attr):
        super(TrainResNet, self).__init__()
        self.selected_attr = selected_attr
        self.num_attr = len(selected_attr)
        
        self.m = models.resnet50()
        self.m.fc = nn.Linear(2048, self.num_attr)
        
        '''
        m = models.resnet50()
        m.fc = nn.Linear(2048, 10)
        self.resnet_list.append(m)
        '''
        
    def forward(self, x):
        return self.m(x)
    
    def predict(self, x):
        pred_attr = self.forward(x)
        preds = torch.sigmoid(pred_attr).detach().cpu().numpy()
        return preds
