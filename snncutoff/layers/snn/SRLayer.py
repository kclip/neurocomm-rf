from torch import nn
from typing import Type
from snncutoff.neuron import LIF

class SRLayer(nn.Module):
    def __init__(self, 
                 neuron: Type[nn.Module]=LIF,
                 hook_enable: bool=True,
                 regularizer: Type[nn.Module]=None, 
                 neuron_params: dict = {'vthr': 1.0, 
                                        'T': 4,
                                        'delta': 0.5,
                                        'mem_init': 0.,
                                        'multistep': True,
                                        'reset_mode': 'hard',
                                        },
                 **kwargs):

        super(SRLayer, self).__init__()
        
        self.T = neuron_params['T'] 
        self.vthr = neuron_params['vthr']
        self.multistep = neuron_params['multistep']
        self.neuron=neuron
        self._neuron=self.neuron(**neuron_params)
        self.regularizer = regularizer
        self.neuron_params = neuron_params
        self.hook_enable = hook_enable
        
    def forward(self, x):
        x = self.reshape(x)
        spike_post, mem_post = self._neuron(x)
        if self.regularizer is not None:
            loss = self.regularizer(spike_post.clone(), mem_post.clone()/self.vthr)
        return spike_post
         
    
    def reshape(self,x):
        if self.multistep:
            batch_size = int(x.shape[0]/self.T)
            new_dim = [self.T, batch_size]
            new_dim.extend(x.shape[1:])
            return x.reshape(new_dim)
        else:
            return x.unsqueeze(0)
        


