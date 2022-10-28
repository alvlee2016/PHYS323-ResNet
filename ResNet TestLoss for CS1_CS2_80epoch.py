#!/usr/bin/env python
# coding: utf-8

# In[1]:


#imports
import pandas as pd     # gives access to DataFrame functions
import numpy as np     #bunch of math functions very commonly used
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
import torch.optim as optim
import time
from torch.utils.data import DataLoader


from functools import partial
from dataclasses import dataclass
from collections import OrderedDict

import matplotlib # plotting libraries
import matplotlib.pyplot as plt
import matplotlib.cm as cm
get_ipython().run_line_magic('matplotlib', 'inline')
import scipy # useful things, such as some fitting routines

import matplotlib.colors as colors


# In[2]:


params = {
    'font.family': 'serif',
    'font.size' : 24, 'axes.titlesize' : 42, 'axes.labelsize' : 32, 'axes.linewidth' : 2,
    # ticks
    'xtick.labelsize' : 24, 'ytick.labelsize' : 24, 'xtick.major.size' : 16, 'xtick.minor.size' : 8,
    'ytick.major.size' : 16, 'ytick.minor.size' : 8, 'xtick.major.width' : 2, 'xtick.minor.width' : 2,
    'ytick.major.width' : 2, 'ytick.minor.width' : 2, 'xtick.direction' : 'in', 'ytick.direction' : 'in',
    # markers
    'lines.markersize' : 8, 'lines.markeredgewidth' : 2, 'errorbar.capsize' : 5, 'lines.linewidth' : 2,
    #'lines.linestyle' : None, 'lines.marker' : None,
    'savefig.bbox' : 'tight', 'legend.fontsize' : 24,
    'axes.labelsize': 24, 'axes.titlesize':24, 'xtick.labelsize':18, 'ytick.labelsize':18,
    'backend': 'Agg', 'mathtext.fontset': 'dejavuserif',
    'figure.facecolor':'w',
    #pad
    'axes.labelpad':12,
    # ticks
    'xtick.major.pad': 6,   'xtick.minor.pad': 6,   
    'ytick.major.pad': 3.5, 'ytick.minor.pad': 3.5,
}
plt.rcParams.update(params)


# In[3]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# In[4]:


#load AmBe data
filename='/depot/darkmatter/data/xenonnt/AmBe/_ambe_bottom_cw11d2m-yqoqgyttzr.npy'
df=pd.DataFrame(np.load(filename))


# In[5]:


df1=pd.read_hdf('/depot/darkmatter/data/xenonnt/rn220/nt_sr0_rn220_runs_cmtv7_tag20220305_picklev4.hdf', 'table')


# In[6]:


df.head()


# In[7]:


for key in df:
    print(key)


# In[8]:


ER_events=df1[(df1['cut_cs2_area_fraction_top'])
                &(df1['cut_daq_veto'])
                &(df1['cut_main_is_valid_triggering_peak'])
                &(df1['cut_interaction_exists'])
                &(df1['cut_run_boundaries'])
                &(df1['cut_s1_area_fraction_top'])
                &(df1['cut_s1_max_pmt'])
                &(df1['cut_s1_pattern_bottom'])
                &(df1['cut_s1_pattern_top'])
                &(df1['cut_s1_single_scatter'])
                &(df1['cut_s1_width'])
                &(df1['cut_s2_pattern'])
                &(df1['cut_s2_recon_pos_diff'])
                &(df1['cut_s2_single_scatter'])
                &(df1['cut_s1_tightcoin_3fold'])
                &(df1['cut_pres2_junk'])
                &(df1['cuts_rn220'])
       &(df1['cs1']< 200)&(df1['cs1']>0)&(df1['cs2']< 2e5)&(df1['cs2']>0)]


# In[9]:


mask=((df['cut_cs2_area_fraction_top'])
                &(df['cut_daq_veto'])
                &(df['cut_main_is_valid_triggering_peak'])
                &(df['cut_interaction_exists'])
                &(df['cut_run_boundaries'])
                &(df['cut_s1_area_fraction_top'])
                &(df['cut_s1_max_pmt'])
                &(df['cut_s1_pattern_bottom'])
                &(df['cut_s1_pattern_top'])
                &(df['cut_s1_single_scatter'])
                &(df['cut_s1_width'])
                &(df['cut_s2_pattern'])
                &(df['cut_s2_recon_pos_diff'])
                &(df['cut_s2_single_scatter'])
               
                &(df['cut_nv_tpc_coincidence_ambe'])
                &(df['cut_s1_tightcoin_3fold'])
                &(df['cut_fiducial_volume_ambe']) )
               #&(df['cut_fiducial_volume'])
               #&(df['cut_s2_naive_bayes'])
               #&(df['cut_s1_naive_bayes'])
NR_events=df.loc[(df['cs1']< 200)&(df['cs1']>0)&(df['cs2']< 2e5)&(df['cs2']>0)& mask]


# In[10]:


# a panda dataframe, mask()
mask=((df['cut_cs2_area_fraction_top'])
                &(df['cut_daq_veto'])
                &(df['cut_main_is_valid_triggering_peak'])
                &(df['cut_interaction_exists'])
                &(df['cut_run_boundaries'])
                &(df['cut_s1_area_fraction_top'])
                &(df['cut_s1_max_pmt'])
                &(df['cut_s1_pattern_bottom'])
                &(df['cut_s1_pattern_top'])
                &(df['cut_s1_single_scatter'])
                &(df['cut_s1_width'])
                &(df['cut_s2_pattern'])
                &(df['cut_s2_recon_pos_diff'])
                &(df['cut_s2_single_scatter'])
              # &(df['cut_s2_width_wire_modeled_wimps'])
                # &(df['cut_nv_tpc_coincidence_ambe'])
                &(df['cut_s1_tightcoin_3fold'])
                &(df['cut_fiducial_volume_ambe'])
                # &(df['cut_fiducial_volume'])
     )
               
               #&(df['cut_s2_naive_bayes'])
              # &(df['cut_s1_naive_bayes'])
unlabelled_events=df.loc[(df['cs1']< 200)&(df['cs1']>0)&(df['cs2']< 2e5)&(df['cs2']>0)& mask]


# In[11]:


dataset = np.concatenate([ER_events[['cs1','cs2']].values, NR_events[['cs1','cs2']].values])
unlabelled_dataset = unlabelled_events[['cs1','cs2']].values


# In[12]:


median_s2nhits = np.median(dataset[:,0])
median_s1nhits= np.median(dataset[:,1])
norm_tensor = Tensor([median_s2nhits,median_s1nhits]) 


# In[13]:


N_ER = len(ER_events)
N_NR = len(NR_events)
weights = np.concatenate([np.repeat([(N_ER + N_NR)/N_ER], N_ER), np.repeat([(N_ER + N_NR)/N_NR], N_NR)])
classes = ('ER', 'NR')
labels = np.zeros([N_ER+N_NR], dtype='long')
labels[:N_ER] = np.repeat([0], N_ER, axis=0)
labels[N_ER:] = np.repeat([1], N_NR, axis=0)


# In[14]:


shuffle_indices = np.random.permutation(N_ER + N_NR) # random numbers

shuffled_data = dataset[shuffle_indices] #shuffled data at different permutaion numbers 
shuffled_weights = weights[shuffle_indices] # using random nos, find the wights of the NR,ER 
shuffled_labels = labels[shuffle_indices]    

unlabelled_weights = np.zeros(unlabelled_dataset.shape[0]) + 1
unlabelled_labels = np.zeros(unlabelled_dataset.shape[0]) - 1


# In[15]:


training_frac = 0.8 # 80% training data size for training
training_size = int((N_ER + N_NR)*training_frac)
training_data = np.concatenate([shuffled_data[:training_size], unlabelled_dataset])
training_weights = np.concatenate([shuffled_weights[:training_size], unlabelled_weights])
training_labels = np.concatenate([shuffled_labels[:training_size], unlabelled_labels])
validation_data = shuffled_data[training_size:]
validation_weights = shuffled_weights[training_size:]
validation_labels = shuffled_labels[training_size:]


# In[16]:


batch_size = 128
train_dset = torch.utils.data.TensorDataset(Tensor(training_data)/norm_tensor, Tensor(training_labels).type(torch.LongTensor))
train_sampler = torch.utils.data.WeightedRandomSampler(training_weights, training_size)
trainloader = torch.utils.data.DataLoader(
    train_dset,
    batch_size=batch_size,
    sampler=train_sampler,
    num_workers=1
)


# In[17]:


fig = plt.figure(figsize=(6,4), dpi=200)
ax = fig.add_subplot(111)
cmap_integer = cm.get_cmap('Set1')

ax.scatter(training_data[:, 0][training_labels==-1],training_data[:,0][training_labels==-1], color=cmap_integer(0), s=1,label='unlabelled')
ax.scatter(training_data[:, 0][training_labels==0],training_data[:,1][training_labels==0], color=cmap_integer(1), s=1, label='ER')
ax.scatter(training_data[:, 0][training_labels==1],training_data[:,1][training_labels==1], color=cmap_integer(2), s=1, label='NR')

ax.legend()
ax.set(xlabel='cs1', ylabel='cs2', xlim = (0,5000), ylim = (0,200))
plt.show()


# In[18]:


validation_data[:2]


# In[19]:


#ModuleDict to create a dictionary with different activation functions, 


# In[20]:


def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]


# In[21]:


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='selu'):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Identity()
        self.activate = activation_func(activation)
        self.shortcut = nn.Identity()   
    
    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


# In[22]:


class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.expansion = expansion
        self.shortcut = nn.Sequential(
            nn.Linear(self.in_channels, self.expanded_channels), )       
        
    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels


# In[ ]:





# In[23]:


def lin_bn(in_channels, out_channels):
    return nn.Sequential(nn.Linear(in_channels, out_channels))


# In[24]:


class ResNetBasicBlock(ResNetResidualBlock):
    """
    Basic ResNet block composed by two layers of 3x3conv/batchnorm/activation
    """
    expansion = 1
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            lin_bn(self.in_channels, self.out_channels),
            activation_func(self.activation),
            lin_bn(self.out_channels, self.expanded_channels),
        )


# In[25]:


class ResNetBottleNeckBlock(ResNetResidualBlock):
    expansion = 4
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
        self.blocks = nn.Sequential(
           lin_bn(self.in_channels, self.out_channels),
             activation_func(self.activation),
             lin_bn(self.out_channels, self.out_channels),
             activation_func(self.activation),
             lin_bn(self.out_channels, self.expanded_channels),
        )
    


# In[26]:


class ResNetLayer(nn.Module):
    """
    A ResNet layer composed by `n` blocks stacked one after the other
    """
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        self.blocks = nn.Sequential(
            block(in_channels , out_channels, *args, **kwargs),
            *[block(out_channels * block.expansion, 
                    out_channels, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


# In[27]:


class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by layers with increasing features.
    """
    def __init__(self, in_channels=3, blocks_sizes=[64, 128, 256, 512], deepths=[2,2,2,2], 
                 activation='selu', block=ResNetBasicBlock, *args, **kwargs):
        super().__init__()
        self.blocks_sizes = blocks_sizes
        
        self.gate = nn.Sequential(
            nn.Linear(in_channels, self.blocks_sizes[0]),
            activation_func(activation),
        )
        
        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([ 
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=deepths[0], activation=activation, 
                        block=block,*args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion, 
                          out_channels, n=n, activation=activation, 
                          block=block, *args, **kwargs) 
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths[1:])]       
        ])
        
        
    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x


# In[28]:


class ResnetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.decoder = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x


# In[29]:


class ResNet(nn.Module):
    
    def __init__(self, in_channels, n_classes, blocks_sizes, deepths, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, blocks_sizes, deepths, *args, **kwargs)
        self.decoder = ResnetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# In[30]:


def resnet34(blocks_sizes, in_channels = 2, n_classes = 2, block=ResNetBasicBlock, *args, **kwargs):
    return ResNet(in_channels, n_classes, blocks_sizes, deepths=[3, 4, 6, 3], *args, **kwargs)


# In[31]:


net = resnet34(blocks_sizes=[64, 128, 256, 512])
net


# In[32]:


#Training the data
criterion = nn.CrossEntropyLoss(
    weight=Tensor([(N_ER + N_NR)/N_ER, (N_ER + N_NR)/N_NR])
)


# In[33]:


def train_criterion(outputs, labels, epoch, T1=50, T2=100, a_f=0.2):
    '''Based on https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.664.3543&rep=rep1&type=pdf'''
    labelled_bool = (labels!=-1)
    unlabelled_bool = (labels==-1)
    pseudolabels = Tensor.argmax(outputs, axis=1)
    labelled_loss = criterion(outputs[labelled_bool], labels[labelled_bool])
    unlabelled_loss = criterion(outputs[unlabelled_bool], pseudolabels[unlabelled_bool])
    if epoch<T1:
        a = 0
    elif epoch<T2:
        a = (epoch-T1)/(T2-T1)*a_f
    else:
        a = a_f
    return labelled_loss + unlabelled_loss*a


# In[34]:


# the learning rate lr, if doubled, we take a step size twice the size of each iteration.
#too small lr - network takes long to train(parameters dont change much each iteration)
#too larg lr - param change a lot,likely in different directions, might overshoot

optimizer = torch.optim.Adam(net.parameters(), lr=2.5e-4, weight_decay=4e-2)


# In[35]:


training_loss = [] #save training loss for plotting
test_loss = [] #save test loss for plotting
epoch_training_loss = []


for epoch in range(80):  # loop over the dataset multiple times(120)

    running_loss = 0.0
    net.train()
    for i ,data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels_loop = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = train_criterion(outputs, labels_loop, epoch)
        loss.backward()
        optimizer.step()
        
        training_loss.append(loss.detach())

        # print statistics
        running_loss += loss.item()
        if i % 500 == 499:    # print every 1000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
    net.eval()
    
    # we save the current training information
    epoch_training_loss.append(train_criterion(net(Tensor(training_data)/norm_tensor), Tensor(training_labels).type(torch.long), epoch).detach())
    test_loss.append(criterion(net(Tensor(validation_data)/norm_tensor), Tensor(validation_labels).type(torch.long)).detach())
    if epoch%10 == 9:
        print(epoch, test_loss[-1])
    
print('Finished Training')


# In[36]:


net.eval()


# In[37]:


sm = nn.Softmax(dim=-1)


# In[38]:


NR_data_NN_labels = sm(net(Tensor(NR_events[['s1_n_hits','s2_n_hits']].values)/norm_tensor)).detach().numpy()

#NR_data_NN_labels = sm(net(Tensor(NR_events[['cs1', 'cs2']].values)/norm_tensor)).detach().numpy()

thresholds = 1-np.logspace(-2,-4,100)
i = 0
acceptance = 1
while acceptance >= 0.5:
    n_s = sum(NR_data_NN_labels[:,1] > thresholds[i])
    n = len(NR_data_NN_labels)
    acceptance = n_s/n
    i = i+1
    
cut_off = thresholds[i-1]


# In[39]:


NN_labels = sm(net(Tensor(validation_data)/norm_tensor)).detach().numpy()
NN_ER_cs1 = []
NN_NR_cs1 = []
NN_ER_cs2 = []
NN_NR_cs2 = []
for i in range (0,np.size(NN_labels[:,1])):
    if NN_labels[i,1] < cut_off:
        NN_ER_cs1.append(validation_data[i][0])
        NN_ER_cs2.append(validation_data[i][1])
    else:
        NN_NR_cs1.append(validation_data[i][0])
        NN_NR_cs2.append(validation_data[i][1])


# In[40]:


fig = plt.figure(figsize=(6,4), dpi=200)
ax = fig.add_subplot(111)
cmap_integer = cm.get_cmap('Set1')

#ax.scatter(training_data[:, 0][training_labels==-1],training_data[:,0][training_labels==-1], color=cmap_integer(0), s=1,label='unlabelled')
ax.scatter(NN_ER_cs1,NN_ER_cs2, color=cmap_integer(1), s=1, label='ER')
ax.scatter(NN_NR_cs1,NN_NR_cs2, color=cmap_integer(2), s=1, label='NR')

ax.legend()
ax.set(xlabel='cs1', ylabel='cs2', xlim=[0,4200], ylim=[0,150])
plt.show()


# In[41]:


x_bin_edges = np.linspace(min(training_data[:,0]),max(training_data[:, 0]))
y_bin_edges = np.linspace(min(training_data[:, 1]),max(training_data[:, 1]))

x_bin_centers = x_bin_edges[:-1] + np.diff(x_bin_edges)
y_bin_centers = np.sqrt(y_bin_edges[:-1]*y_bin_edges[1:])

X, Y = np.meshgrid(x_bin_centers, y_bin_centers)
X_edges, Y_edges = np.meshgrid(x_bin_edges, y_bin_edges)


# In[42]:


C = np.zeros_like(X)
X_flat = X.reshape(1,-1)
Y_flat = Y.reshape(1,-1)
NN_output = np.array(sm(net(Tensor(np.concatenate([[X_flat], [Y_flat]], axis=0)[0].T)/norm_tensor)).detach())[:,1].reshape(X.shape)


# In[44]:


x_bin_edges = np.linspace(0,500,100)
y_bin_edges = np.linspace(0,150,100)

x_bin_centers = x_bin_edges[:-1] + np.diff(x_bin_edges)
y_bin_centers = np.sqrt(y_bin_edges[:-1]*y_bin_edges[1:])

X, Y = np.meshgrid(x_bin_centers, y_bin_centers)
X_edges, Y_edges = np.meshgrid(x_bin_edges, y_bin_edges)


# In[45]:


fig = plt.figure(figsize=(6, 4), dpi=200)
ax = fig.add_subplot(111)
ax.plot(range(len(training_loss)), training_loss)
ax.set(xlabel='loop', ylabel='training loss')
plt.show()


# In[46]:


ER_data_NN_labels = sm(net(Tensor(ER_events[['cs1','cs2']].values)/norm_tensor)).detach().numpy()
NR_data_NN_labels = sm(net(Tensor(NR_events[['cs1','cs2']].values)/norm_tensor)).detach().numpy()


# In[47]:


def binomial_interval(n, n_s, z=1):
    n_f = n-n_s
    p = (n_s+(z**2)/2)/(n+z**2)
    err = z/(n+z**2)*np.sqrt(n_s*n_f/n+(z**2)/4)
    return np.array([
        n_s/n - (p - err), (p + err)- n_s/n
    ])


# In[48]:


thresholds = 1-np.logspace(-2,-4,100)
leakage = np.zeros_like(thresholds)
leakage_up = np.zeros_like(thresholds)
leakage_down = np.zeros_like(thresholds)
acceptance = np.zeros_like(thresholds)
acceptance_up = np.zeros_like(thresholds)
acceptance_down = np.zeros_like(thresholds)
for i, threshold in enumerate(thresholds):
    n_s = sum(ER_data_NN_labels[:,1] > threshold)
    n = len(ER_data_NN_labels)
    leakage[i] = n_s/n
    leakage_down[i], leakage_up[i] = binomial_interval(n, n_s, z=1)
    n_s = sum(NR_data_NN_labels[:,1] > threshold)
    n = len(NR_data_NN_labels)
    acceptance[i] = n_s/n
    acceptance_down[i], acceptance_up[i] = binomial_interval(n, n_s, z=1)


# In[49]:


fig = plt.figure(figsize=(6, 4), dpi=200)
ax = fig.add_subplot(111)
ax.plot(thresholds, leakage)
ax.fill_between(thresholds, leakage-leakage_down, leakage_up+leakage, alpha=0.5)
ax.plot(thresholds, acceptance)
ax.fill_between(thresholds, acceptance-acceptance_down, acceptance_up+acceptance, alpha=0.5)
ax.set(ylabel='fraction', xlabel='threshold')
#yscale = 'log', ylim=[0.001, 1]
plt.show()


# In[50]:


leakage


# In[51]:


acceptance


# In[52]:


fig = plt.figure(figsize=(6, 4), dpi=200)
ax = fig.add_subplot(111)
ax.plot(acceptance, leakage)
ax.set(xlabel='acceptance', ylabel='leakage')
plt.show()


# In[53]:


print(f'{100*leakage[::-1][np.searchsorted(acceptance[::-1], 0.5)]}% leakage at 50% acceptance')


# In[54]:


print(f'{100*leakage[::-1][np.searchsorted(acceptance[::-1], .9)]}% leakage at 90% acceptance')


# In[55]:


fig = plt.figure(figsize=(6, 4), dpi=200)
ax = fig.add_subplot(111)
ax.plot(range(len(epoch_training_loss)), epoch_training_loss, label='training loss')
ax.plot(range(len(test_loss)), test_loss, label='test loss')
ax.set(xlabel='epoch', ylabel='loss', ylim=[0.15, 2])
ax.legend()
plt.show()


# In[ ]:




