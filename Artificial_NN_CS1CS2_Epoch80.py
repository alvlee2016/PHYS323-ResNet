#!/usr/bin/env python
# coding: utf-8

# In[1]:


#imports
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
import torch.optim as optim

import matplotlib.pyplot as plt
import matplotlib.cm as cm


# In[2]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# ## Data
# 
# Then, we load data and introduce some quality cuts. The quality cuts primary remove events that are have various issues or artifacts, such as multiple scatters. We don't care about multiple scatters that we can cut away because WIMP dark matter would not multiple-scatter.

# In[3]:


#load AmBe data
filename='/depot/darkmatter/data/xenonnt/AmBe/_ambe_bottom_cw11d2m-yqoqgyttzr.npy'
df=pd.DataFrame(np.load(filename))


# In[4]:


df1=pd.read_hdf('/depot/darkmatter/data/xenonnt/rn220/nt_sr0_rn220_runs_cmtv7_tag20220305_picklev4.hdf', 'table')


# In[5]:


df.head()


# In[6]:


for key in df:
    print(key)


# In[7]:


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


# In[8]:


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
                &(df['cut_nv_tpc_coincidence_ambe'])
                &(df['cut_s1_tightcoin_3fold'])
                # &(df['cut_fiducial_volume_ambe'])
     )
               #&(df['cut_fiducial_volume'])
               #&(df['cut_s2_naive_bayes'])
              # &(df['cut_s1_naive_bayes'])
NR_events=df.loc[(df['cs1']< 200)&(df['cs1']>0)&(df['cs2']< 2e5)&(df['cs2']>0)& mask]


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
              # &(df['cut_s2_width_wire_modeled_wimps'])
                # &(df['cut_nv_tpc_coincidence_ambe'])
                &(df['cut_s1_tightcoin_3fold'])
                &(df['cut_fiducial_volume_ambe'])
                # &(df['cut_fiducial_volume'])
     )
               
               #&(df['cut_s2_naive_bayes'])
              # &(df['cut_s1_naive_bayes'])
unlabelled_events=df.loc[(df['cs1']< 200)&(df['cs1']>0)&(df['cs2']< 2e5)&(df['cs2']>0)& mask]


# We can see that we have very uneven classes, with way more ER events than NR events. This is something we need to deal with later.

# In[10]:


len(unlabelled_events)


# In[11]:


len(ER_events)


# In[12]:


len(NR_events)


# In[14]:


dataset = np.concatenate([ER_events[['cs1','cs2']].values, NR_events[['cs1','cs2']].values])
unlabelled_dataset = unlabelled_events[['cs1','cs2']].values


# I normalise every dimension by the median values of that dimension. This is helpful for our neural net, as it is difficult for many ML algorithms to deal with very different orders of magnitude. It also messes with weight decay regularisation, which we will use later.

# In[15]:


median_cs1 = np.median(dataset[:,0])
median_e_charge = np.median(dataset[:,1])
norm_tensor = Tensor([median_cs1,median_e_charge])


# In[16]:


N_ER = len(ER_events)
N_NR = len(NR_events)
weights = np.concatenate([np.repeat([(N_ER + N_NR)/N_ER], N_ER), np.repeat([(N_ER + N_NR)/N_NR], N_NR)])
classes = ('ER', 'NR')
labels = np.zeros([N_ER+N_NR], dtype='long')
labels[:N_ER] = np.repeat([0], N_ER, axis=0)
labels[N_ER:] = np.repeat([1], N_NR, axis=0)


# In[17]:


shuffle_indices = np.random.permutation(N_ER + N_NR)

shuffled_data = dataset[shuffle_indices]
shuffled_weights = weights[shuffle_indices]
shuffled_labels = labels[shuffle_indices]

unlabelled_weights = np.zeros(unlabelled_dataset.shape[0]) + 1
unlabelled_labels = np.zeros(unlabelled_dataset.shape[0]) - 1


# In[18]:


training_frac = 0.8
training_size = int((N_ER + N_NR)*training_frac)
training_data = np.concatenate([shuffled_data[:training_size], unlabelled_dataset])
training_weights = np.concatenate([shuffled_weights[:training_size], unlabelled_weights])
training_labels = np.concatenate([shuffled_labels[:training_size], unlabelled_labels])
validation_data = shuffled_data[training_size:]
validation_weights = shuffled_weights[training_size:]
validation_labels = shuffled_labels[training_size:]


# I use a minibatch size of 128. This is somewhat on the high side, but as we can see later things seem to work ok.

# In[19]:


batch_size = 128
train_dset = torch.utils.data.TensorDataset(Tensor(training_data)/norm_tensor, Tensor(training_labels).type(torch.LongTensor)
)
train_sampler = torch.utils.data.WeightedRandomSampler(training_weights, training_size)
trainloader = torch.utils.data.DataLoader(
    train_dset,
    batch_size=batch_size,
    sampler=train_sampler,
    num_workers=1
)


# Below I plot ER, NR, and unlabelled datasets.

# In[21]:


fig = plt.figure(figsize=(6, 4), dpi=200)
ax = fig.add_subplot(111)
cmap_integer = cm.get_cmap('Set1')

ax.scatter(training_data[:, 0][training_labels==-1], training_data[:, 1][training_labels==-1], color=cmap_integer(0), s=1, label='unlabelled')
ax.scatter(training_data[:, 0][training_labels==0], training_data[:, 1][training_labels==0], color=cmap_integer(1), s=1, label='ER')
ax.scatter(training_data[:, 0][training_labels==1], training_data[:, 1][training_labels==1], color=cmap_integer(2), s=1, label='NR')

ax.legend()
ax.set(xlabel='cS1', ylabel='cS2', xlim=[0, 200], ylim=[100, 20000], yscale='log')
plt.show()


# ## Neural net architecture
# 
# Here I define the NN. It is a very simple dense neural net, with only 4 hidden layers. The [SiLU activation function](https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html#torch.nn.SiLU) is used here. For the hidden layers, [dropout](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html) is used both for regularisation, and to improve generalisation.

# In[22]:


class Net(nn.Module):

    def __init__(self, dropout_p):
        super(Net, self).__init__()
        self.DNN1 = nn.Sequential(
            nn.Linear(2,50), nn.SiLU(),
            nn.Linear(50,500), nn.Dropout(dropout_p), nn.SiLU(),
            nn.Linear(500,500), nn.Dropout(dropout_p), nn.SiLU(),
            nn.Linear(500,200), nn.Dropout(dropout_p), nn.SiLU(),
            nn.Linear(200,50), nn.Dropout(dropout_p), nn.SiLU(),
            nn.Linear(50,2),
        )

    def forward(self, x):
        x = self.DNN1(x)
        return x


# In[23]:


dropout_p = 0.35
net = Net(dropout_p)
print(net)


# In[22]:


Tensor(validation_data[:1])


# In[25]:


net(Tensor(validation_data[:2]))


# ## Training
# 
# [Cross entropy loss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss) is used as the loss function. A loss function needs to be defined because neural nets are (stochastically) optimised according to some criteria; in this case, the cross entropy loss is a criteria that is optimised (minimised) when the predicted class agrees with the labelled class.
# 
# Where there is no known label, the neural net is used to generate a 'pseudolabel', and then the cross entropy loss is computed based on that. We don't want to treat pseudolabels quite as seriously as real labels, so we give them a lower weight, $a_f$. This pseudolabel method is essentially coding in some amount of confirmation bias. This is not always a good thing, and [this paper](https://arxiv.org/pdf/1908.02983.pdf) discusses this in detail and goes over a strategy that mitigates the issue, though that is not implemented here.

# In[26]:


criterion = nn.CrossEntropyLoss(
    weight=Tensor([(N_ER + N_NR)/N_ER, (N_ER + N_NR)/N_NR])
)
# train_criterion = nn.CrossEntropyLoss(weight=Tensor([(N_ER + N_NR)/N_ER, (N_ER + N_NR)/N_NR]))


# In[27]:


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


# The [AdamW optimizer](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW) is used. There are many choices, but this is a simple one that is adaptive with few hyperparameters. We use AdamW and the standard [backpropagation](https://en.wikipedia.org/wiki/Backpropagation) method for training. I make the printing loop print the test loss as well as a diagnostic.

# In[28]:


optimizer = torch.optim.AdamW(net.parameters(), lr=2.5e-4, weight_decay=4e-2)


# In[29]:


training_loss = []
test_loss = []
epoch_training_loss = []

for epoch in range(80):  # loop over the dataset multiple times

    running_loss = 0.0
    net.train()
    for i, data in enumerate(trainloader, 0):
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
    epoch_training_loss.append(train_criterion(net(Tensor(training_data)/norm_tensor), Tensor(training_labels).type(torch.long), epoch).detach())
    test_loss.append(criterion(net(Tensor(validation_data)/norm_tensor), Tensor(validation_labels).type(torch.long)).detach())
    if epoch%10 == 9:
        print(epoch, test_loss[-1])
    
print('Finished Training')


# ## Using our neural net
# 
# First, I just classify a grid of points to be able to make a plot that shows how the classifier neural net behaves. This is a nice visualisation, but it no longer becomes possible for higher dimensions. There are other plotting options such as making 2D slices of a higher dimensional space in that situation.

# In[32]:


x_bin_edges = np.linspace(0,200,100)
y_bin_edges = np.logspace(2,4.2,100)

x_bin_centers = x_bin_edges[:-1] + np.diff(x_bin_edges)
y_bin_centers = np.sqrt(y_bin_edges[:-1]*y_bin_edges[1:])

X, Y = np.meshgrid(x_bin_centers, y_bin_centers)
X_edges, Y_edges = np.meshgrid(x_bin_edges, y_bin_edges)


# In[34]:


net.eval()


# In[35]:


sm = nn.Softmax(dim=-1)


# In[36]:


C = np.zeros_like(X)
X_flat = X.reshape(1,-1)
Y_flat = Y.reshape(1,-1)
NN_output = np.array(sm(net(Tensor(np.concatenate([[X_flat], [Y_flat]], axis=1)[0].T)/norm_tensor)).detach())[:,1].reshape(X.shape)


# I also plot training and test loss as a function of the minibatch loop and the epoch. This is a diagnostic plot; if the loss goes up after a while, that suggests that our model is overfitting.

# In[37]:


fig = plt.figure(figsize=(6, 4), dpi=200)
ax = fig.add_subplot(111)
ax.plot(range(len(training_loss)), training_loss)
ax.set(xlabel='loop', ylabel='training loss')
plt.show()


# In[38]:


fig = plt.figure(figsize=(6, 4), dpi=200)
ax = fig.add_subplot(111)
ax.plot(range(len(epoch_training_loss)), epoch_training_loss, label='training loss')
ax.plot(range(len(test_loss)), test_loss, label='test loss')
ax.set(xlabel='epoch', ylabel='loss', ylim=[0.15, 2])
ax.legend()
plt.show()


# Finally, some metrics. We want to maximise the fraction of NR events we are keeping (acceptance), while minimising the fraction of ER events we are letting through (leakage). These two parameters can be plotted against each other to determine the trade-off, and a numerical metric defined as the leakage fraction with 50% acceptance is printed at the end.

# In[47]:


ER_data_NN_labels = sm(net(Tensor(ER_events[['cs1','cs2']].values)/norm_tensor)).detach().numpy()
NR_data_NN_labels = sm(net(Tensor(NR_events[['cs1','cs2']].values)/norm_tensor)).detach().numpy()


# In[48]:


def binomial_interval(n, n_s, z=1):
    n_f = n-n_s
    p = (n_s+(z**2)/2)/(n+z**2)
    err = z/(n+z**2)*np.sqrt(n_s*n_f/n+(z**2)/4)
    return np.array([
        n_s/n - (p - err), (p + err)- n_s/n
    ])


# In[49]:


binomial_interval(100, 20, z=1)


# In[50]:


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


# In[51]:


fig = plt.figure(figsize=(6, 4), dpi=200)
ax = fig.add_subplot(111)
ax.plot(thresholds, leakage)
ax.fill_between(thresholds, leakage-leakage_down, leakage_up+leakage, alpha=0.5)
ax.plot(thresholds, acceptance)
ax.fill_between(thresholds, acceptance-acceptance_down, acceptance_up+acceptance, alpha=0.5)
ax.set(yscale='log', ylabel='fraction', xlabel='threshold')
plt.show()


# In[52]:


fig = plt.figure(figsize=(6, 4), dpi=200)
ax = fig.add_subplot(111)
ax.plot(acceptance, leakage)
ax.set(yscale='log', xlabel='acceptance', ylabel='leakage')
plt.show()


# In[53]:


print(f'{100*leakage[::-1][np.searchsorted(acceptance[::-1], 0.5)]}% leakage at 50% acceptance')


# In[ ]:




