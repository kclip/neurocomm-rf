import torch
import torch.nn as nn

class Conv2dLayer(nn.Module):
    def __init__(self,in_plane,out_plane,kernel_size,stride,padding,droprate=0.0,bias=True,batch_norm=True):
        super(Conv2dLayer, self).__init__()
        if batch_norm:
            self.fwd = nn.Sequential(
                nn.Conv2d(in_plane,out_plane,kernel_size,stride,padding,bias=bias),
                nn.BatchNorm2d(out_plane),
                nn.Dropout(p=droprate)
            )
        else:
            self.fwd = nn.Sequential(
                nn.Conv2d(in_plane,out_plane,kernel_size,stride,padding,bias=bias),
                nn.Dropout(p=droprate)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.fwd(x)
        x = self.relu(x)
        return x

class Conv1dLayer(nn.Module):
    def __init__(self,in_plane,out_plane,kernel_size,stride,padding,droprate=0.35,bias=True,batch_norm=True):
        super(Conv1dLayer, self).__init__()
        if batch_norm:
            self.fwd = nn.Sequential(
                nn.Conv1d(in_plane,out_plane,kernel_size,stride,padding,bias=bias),
                nn.BatchNorm1d(out_plane),
                nn.Dropout(p=droprate)
            )
        else:
            self.fwd = nn.Sequential(
                nn.Conv1d(in_plane,out_plane,kernel_size,stride,padding,bias=bias),
                nn.Dropout(p=droprate)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.fwd(x)
        x = self.relu(x)
        return x


class LinearLayer(nn.Module):
    def __init__(self,in_plane,out_plane,droprate=0.0,bias=False,batch_norm=False):
        super(LinearLayer, self).__init__()
        if batch_norm:
            self.fwd = nn.Sequential(
                nn.Linear(in_plane,out_plane),
                nn.BatchNorm1d(out_plane),
                nn.Dropout(p=droprate)
            )
        else:
            self.fwd = nn.Sequential(
                nn.Linear(in_plane,out_plane,bias=bias),
                nn.Dropout(p=droprate)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.fwd(x)
        x = self.relu(x)
        return x



class _FCSNN(nn.Module):
    def __init__(self,output_dim = 24):
        super(_FCSNN, self).__init__()
        self.features = nn.Sequential(
            LinearLayer(256,256,droprate=0.0),
            LinearLayer(256,256,droprate=0.0),
        )
        self.classifier = nn.Linear(256,output_dim)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        x = self.features(input)
        x = self.classifier(x)
        return x



import numpy as np
from snncutoff.neuron import BRF, LIF, ALIF, RF
from snncutoff.layers.snn import SRLayer

import torch.nn.functional as F
class RF_encoding(nn.Module):
    def __init__(self,T, num_neurons,droprate=0.35, neuron='brf'):
        super(RF_encoding, self).__init__()
        self.T = T
        self.dropout = nn.Dropout(p=droprate)
        self.neuron = neuron 
        self.droprate = droprate
        self.num_neurons = num_neurons
        self._neuron = None
        self.hook_enable = True
        
    def params_init(self, T, device):
        if self.neuron == 'brf':
            weight1 = nn.Parameter(torch.randn(self.num_neurons,1 ).to(device))
            self.register_parameter('weight1',weight1)
            nn.init.xavier_uniform_(self.weight1)
            # nn.init.kaiming_normal_(self.weight1, nonlinearity='relu')
            weight2 = nn.Parameter(torch.randn(self.num_neurons,1 ).to(device))
            self.register_parameter('weight2',weight2)
            nn.init.xavier_uniform_(self.weight2)
            # nn.init.kaiming_normal_(self.weight2, nonlinearity='relu')
            neuron_params = SRLayer().neuron_params
            neuron_params['T'] = T
            self._neuron = SRLayer(neuron= BRF,
                                   hook_enable = False,
                                   neuron_params=neuron_params,
            )
        elif self.neuron == 'rf':
            weight1 = nn.Parameter(torch.randn(self.num_neurons,1 ).to(device))
            self.register_parameter('weight1',weight1)
            nn.init.xavier_uniform_(self.weight1)
            # nn.init.kaiming_normal_(self.weight1, nonlinearity='relu')
            weight2 = nn.Parameter(torch.randn(self.num_neurons,1 ).to(device))
            self.register_parameter('weight2',weight2)
            nn.init.xavier_uniform_(self.weight2)
            # nn.init.kaiming_normal_(self.weight2, nonlinearity='relu')
            neuron_params = SRLayer().neuron_params
            neuron_params['T'] = T
            self._neuron = SRLayer(neuron= RF,
                                   hook_enable = False,
                                   neuron_params=neuron_params,
            )
        elif self.neuron == 'lif':
            weight1 = nn.Parameter(torch.randn(self.num_neurons//2,1 ).to(device))
            self.register_parameter('weight1',weight1)
            nn.init.xavier_uniform_(self.weight1)
            # nn.init.kaiming_normal_(self.weight1, nonlinearity='relu')
            bias1 = nn.Parameter(torch.randn(self.num_neurons//2).to(device))
            self.register_parameter('bias1',bias1)
            torch.nn.init.constant_(self.bias1,0)

            weight2 = nn.Parameter(torch.randn(self.num_neurons//2,1 ).to(device))
            self.register_parameter('weight2',weight2)
            nn.init.xavier_uniform_(self.weight2)
            # nn.init.kaiming_normal_(self.weight2, nonlinearity='relu')
            bias2 = nn.Parameter(torch.randn(self.num_neurons//2).to(device))
            self.register_parameter('bias2',bias2)
            torch.nn.init.constant_(self.bias2,0)

            neuron_params = SRLayer().neuron_params
            neuron_params['T'] = T
            self._neuron = SRLayer(neuron= LIF,
                                   hook_enable = False,
                                   neuron_params=neuron_params,
            )

        elif self.neuron == 'alif':
            weight1 = nn.Parameter(torch.randn(self.num_neurons//2,1 ).to(device))
            self.register_parameter('weight1',weight1)
            nn.init.xavier_uniform_(self.weight1)
            # nn.init.kaiming_normal_(self.weight1, nonlinearity='relu')
            bias1 = nn.Parameter(torch.randn(self.num_neurons//2).to(device))
            self.register_parameter('bias1',bias1)
            torch.nn.init.constant_(self.bias1,0)

            weight2 = nn.Parameter(torch.randn(self.num_neurons//2,1).to(device))
            self.register_parameter('weight2',weight2)
            nn.init.xavier_uniform_(self.weight2)
            # nn.init.kaiming_normal_(self.weight2, nonlinearity='relu')
            bias2 = nn.Parameter(torch.randn(self.num_neurons//2).to(device))
            self.register_parameter('bias2',bias2)
            torch.nn.init.constant_(self.bias2,0)


            neuron_params = SRLayer().neuron_params
            neuron_params['T'] = T
            self._neuron = SRLayer(neuron= ALIF,
                                   hook_enable = False,
                                   neuron_params=neuron_params,
            )

    def forward(self, x):
        if not hasattr(self, 'weight1'):
            self.params_init(x.shape[0],x.device)
        if 'rf' in self.neuron:
           return self.forward_brf(x)
        elif 'lif' in self.neuron:
           return self.forward_lif(x)

    def forward_lif(self, x):
        new_dim = [int(x.shape[1]*x.shape[0]),]
        new_dim.extend(x.shape[2:])
        x = x.reshape(new_dim)    
        real = F.linear(x[...,0:1], self.weight1, self.bias1)
        imag = F.linear(x[...,1:2], self.weight2, self.bias2)
        x = torch.cat([real,imag],dim=-1)

        if self.training and self.droprate > 0.0:
            dropout_prob = self.droprate
            # Generate a dropout mask
            mask = (torch.rand_like(x) >= dropout_prob).float() / (1 - dropout_prob)
            # Apply the same mask to real and imaginary parts
            x = x * mask

        x = self._neuron(x)
        new_dim = [self.T, x.shape[0]//self.T,*x.shape[1:]]
        x = x.reshape(new_dim).sum(1)
        return x

    def forward_brf(self, x):
        new_dim = [int(x.shape[1]*x.shape[0]),]
        new_dim.extend(x.shape[2:])
        x = x.reshape(new_dim)    
        _real = F.linear(x[...,0:1], self.weight1)
        _imag = F.linear(x[...,1:2], self.weight2)

        if self.training and self.droprate > 0.0:
            dropout_prob = self.droprate
            # Generate a dropout mask
            mask = (torch.rand_like(_real) >= dropout_prob).float() / (1 - dropout_prob)
            # Apply the same mask to real and imaginary parts
            _real = _real * mask
            mask = (torch.rand_like(_imag) >= dropout_prob).float() / (1 - dropout_prob)
            _imag = _imag * mask
        real = _real - _imag
        imag = _real + _imag
        x = torch.complex(real,imag)
        x = self._neuron(x)
        new_dim = [self.T, x.shape[0]//self.T,*x.shape[1:]]
        x = x.reshape(new_dim).sum(1)
        # return x.unsqueeze(-2)
        return x
    
DEFAULT_LI_ADAPTIVE_TAU_M_MEAN = 20.
DEFAULT_LI_ADAPTIVE_TAU_M_STD = 5.
DEFAULT_LI_TAU_M = 20.


class LICell(torch.nn.Module):
    def __init__(
            self,
            output_dim,
            T,
            tau_mem: float = DEFAULT_LI_TAU_M,
            adaptive_tau_mem: bool = True,
            adaptive_tau_mem_mean: float = DEFAULT_LI_ADAPTIVE_TAU_M_MEAN,
            adaptive_tau_mem_std: float = DEFAULT_LI_ADAPTIVE_TAU_M_STD,
    ) -> None:
        super(LICell, self).__init__()


        self.adaptive_tau_mem = adaptive_tau_mem
        self.adaptive_tau_mem_mean = adaptive_tau_mem_mean
        self.adaptive_tau_mem_std = adaptive_tau_mem_std
        self.T = T
        self.init = False
        self.tau_mem = tau_mem
        self.tau_mem = tau_mem * torch.ones(output_dim)
        if self.adaptive_tau_mem:
            self.tau_mem = torch.nn.Parameter(self.tau_mem).to('cuda')
            torch.nn.init.normal_(self.tau_mem, mean=self.adaptive_tau_mem_mean, std=self.adaptive_tau_mem_std)

    def reshape(self,x):
        batch_size = int(x.shape[0]/self.T)
        new_dim = [self.T, batch_size]
        new_dim.extend(x.shape[1:])
        return x.reshape(new_dim)

    def post_reshape(self,x):
        new_dim = [int(x.shape[1]*self.T),]
        new_dim.extend(x.shape[2:])
        return x.reshape(new_dim)

    def forward(self, x) -> torch.Tensor:
        x = self.reshape(x)
        # if not self.init:
        #     tau_mem = self.tau_mem * torch.ones_like(x[0,0])

        #     if self.adaptive_tau_mem:
        #         self.tau_mem = torch.nn.Parameter(tau_mem)
        #         torch.nn.init.normal_(self.tau_mem, mean=self.adaptive_tau_mem_mean, std=self.adaptive_tau_mem_std)
        #     self.init = True
        tau_mem = torch.abs(self.tau_mem)
        alpha = torch.exp(-1 * 1 / tau_mem)
        # alpha = torch.sigmoid(self.tau_mem)
        u = 0.0
        out = []
        for t in range(x.shape[0]):
            u = u*alpha + x[t]*(1.0 - alpha)
            out.append(u)
        return self.post_reshape(torch.stack(out,dim=0))


class FCITS(nn.Module):
    def __init__(self,output_dim = 6):
        super(FCITS, self).__init__()
        pool = nn.MaxPool1d(2)
        T = 110
        input_T = 220
        num_neurons = 64
        self.rf_encoding  = RF_encoding(T,input_T,num_neurons,droprate=0.35)        # w = 1*num_neurons//44
        self.features = nn.Sequential(
            Conv1dLayer(1,64,3,1,1,droprate=0.35),
            pool,
            Conv1dLayer(64,32,3,1,1,droprate=0.35),
            pool,
            Conv1dLayer(32,16,3,1,1,droprate=0.3),
            pool,
            nn.Flatten(1,-1)
        )
        w = int(num_neurons/2/2/2)
        # w = 1*num_neurons//44
        self.fc =  LinearLayer(16*w,100,droprate=0.5) #4 for m=3 12 for m=4
        self.classifier = nn.Linear(100,output_dim)
        self.li_update = LICell(output_dim,T)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        x = self.rf_encoding(input)
        x = self.features(x)
        x = self.fc(x)
        x = self.classifier(x)
        x = self.li_update(x)
        return x


class FCSNN_ITS(nn.Module):
    def __init__(self,args, output_dim = 6):
        super(FCSNN_ITS, self).__init__()
        self.bias = True if 'lif' in args.neuron['name']  else False

        T = args.neuron['T']
        neuron = args.neuron['name']
        num_neurons = args.architecture['input_dims'] # if neuron=='brf' else int(args.architecture['input_dims']*2)
        hidden_features = args.architecture['hidden_features'] 

        self.rf_encoding  = RF_encoding(T,num_neurons,droprate=0.4,neuron=neuron)        # w = 1*num_neurons//44
        # self.fc0 =  LinearLayer(1,num_neurons,droprate=0.0,bias=self.bias) #4 for m=3 12 for m=4
        self.fc1 =  LinearLayer(num_neurons,hidden_features,droprate=0.4,bias=self.bias) #4 for m=3 12 for m=4
        self.fc2 =  LinearLayer(hidden_features,hidden_features,droprate=0.4,bias=self.bias) #4 for m=3 12 for m=4
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if self.bias:
                    torch.nn.init.constant_(m.bias,0)
        self.classifier = nn.Linear(hidden_features,output_dim,bias=self.bias)
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        self.li_update = LICell(output_dim,T)

    def forward(self, input):
        x = self.rf_encoding(input)
        # x = self.fc0(input)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.classifier(x)
        x = self.li_update(x)
        return x

class FCSNN(nn.Module):
    def __init__(self,args, output_dim = 24):
        super(FCSNN, self).__init__()
        self.bias = True if 'lif' in args.neuron['name']  else False

        T = args.neuron['T']
        neuron = args.neuron['name']
        num_neurons = args.architecture['input_dims'] # if neuron=='brf' else int(args.architecture['input_dims']*2)
        hidden_features = args.architecture['hidden_features'] 

        self.rf_encoding  = RF_encoding(T,num_neurons,droprate=0.4,neuron=neuron)        # w = 1*num_neurons//44
        # self.fc0 =  LinearLayer(1,num_neurons,droprate=0.0,bias=self.bias) #4 for m=3 12 for m=4
        self.fc1 =  LinearLayer(num_neurons,hidden_features,droprate=0.4,bias=self.bias) #4 for m=3 12 for m=4
        self.fc2 =  LinearLayer(hidden_features,hidden_features,droprate=0.4,bias=self.bias) #4 for m=3 12 for m=4
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if self.bias:
                    torch.nn.init.constant_(m.bias,0)
        self.classifier = nn.Linear(hidden_features,output_dim,bias=self.bias)
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        self.li_update = LICell(output_dim,T)

    def forward(self, input):
        x = self.rf_encoding(input)
        # x = self.fc0(input)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.classifier(x)
        x = self.li_update(x)
        return x

# ## RF neuron
class FCSHD(nn.Module):
    def __init__(self,args, output_dim = 20):
        super(FCSHD, self).__init__()
        self.bias = True if 'lif' in args.neuron['name']  else False
        self.input_size = 700
        self.feature_size = 128
        self.fc1 =  LinearLayer(self.input_size,self.feature_size,droprate=0.0,bias=self.bias) #4 for m=3 12 for m=4
        self.fc2 =  LinearLayer(self.feature_size,self.feature_size,droprate=0.0,bias=self.bias) #4 for m=3 12 for m=4

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if self.bias:
                    torch.nn.init.constant_(m.bias,0)
        self.classifier = nn.Linear(self.feature_size,output_dim,bias=self.bias)
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        self.li_update = LICell(output_dim,250)

    def forward(self, input):
        x = self.fc1(input)
        x = self.fc2(x)
        x = self.classifier(x)
        x = self.li_update(x)
        return x


class FirstOutputOnly(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        out = self.module(x)
        return out[0]  # Only return the first element

class FCWrapped(nn.Module):
    def __init__(self, base_model: nn.Sequential, args, insert_index=1):
        super().__init__()

        # Get the wireless channel module
        wireless_ch_module = wirelss_ch_odfm(args.wireless_ch)

        # Wrap to only return the first element of the tuple
        wireless_ch = FirstOutputOnly(wireless_ch_module)


        # Build the new layers list with insertion
        layers = list(base_model.children())
        num_layers = len(layers)

        assert 0 < insert_index < num_layers, (
            f"insert_index must be between 1 and {num_layers - 1} (exclusive), "
            f"but got {insert_index}"
        )

        layers.insert(insert_index, wireless_ch)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

import math

class wirelss_ch_odfm(nn.Module):
    def __init__(self, wireless_ch: dict = {'modulation': 'analog'}):
        super(wirelss_ch_odfm, self).__init__()  
        self.noise = None
        self.weights = None
        self.h = None
        self.n = None
        self.group_size = 8
        self.num_path = 5
        self.E = wireless_ch.E  # average energy consumption of per data subcarrier, in dB
        self.num_bit = 0.0
        self.first_epoch = True
        self.modulation = wireless_ch.modulation
        self.pilot_interval = 8
        self.pilotValue = math.sqrt(10 ** (self.E  / 10))
        self.code_rate = 0.5
        # self.B = wireless_ch.B  # for digital case, number of bits per subcarrier. Set to be an even number to be supported by QAM 
        self.num_ofdma = wireless_ch.num_ofdma
        self.rec_rate = []
        self.power_constrs = wireless_ch.power_constrs
        self.b = wireless_ch.b
        self.dist = wireless_ch.distance
        self.rx_snr_db = wireless_ch.rx_snr_db
        self.adaptive_power = wireless_ch.adaptive_power
        self.wireless = True
        
    def forward(self, x):
        if self.modulation=='analog':
            return self._forward_analog(x)
        elif self.modulation=='noiseless':
            return self._forward_noiseless(x), 0.0

    def _forward_noiseless(self, x):
        return x

    def _normalized_power(self,T, E_max, alpha):
        # Generate weights w_i = alpha^(T-i)
        weights = np.array([alpha**(T-i) for i in range(1, T+1)])
        
        # Normalize weights to ensure average equals E_max
        scaling_factor = E_max * T / np.sum(weights)
        P_t = scaling_factor * weights
        
        return P_t

    def normalized_power(self,T, E_max, alpha):
        P_t = self._normalized_power(T, E_max, alpha)
        return P_t

    def _forward_analog(self, input):
        pam_points = self._generate_mbit_pam(self.num_bit,device=input.device)
        #input shape (time_step, batch_size, num_neurons)
        input = input.transpose(0, 1)*(2**self.num_bit)  # (batch_size, time_step, num_neurons)
        # The number of data subcarriers is the number of output neurons
        batch_size = input.shape[0] 
        num_data_carriers = input.shape[2]
        num_pilot = int(torch.ceil(torch.tensor(num_data_carriers / (self.pilot_interval - 1))))
        # print(num_pilot)
        num_allcarriers = num_data_carriers + num_pilot
        allCarriers = torch.arange(num_allcarriers,device=input.device)
        pilotCarriers = torch.arange(0, num_allcarriers, self.pilot_interval,device=input.device)
        if pilotCarriers[-1] != allCarriers[-1]:
            allCarriers = torch.cat([allCarriers, torch.tensor([num_allcarriers]).to(input.device) ])
            num_allcarriers += 1
            pilotCarriers = torch.cat([pilotCarriers, torch.tensor([allCarriers[-1]]).to(input.device) ])
            num_pilot += 1
        dataCarriers = allCarriers[~torch.isin(allCarriers, pilotCarriers)]

        # prepare for interpolation
        lower_indices = torch.searchsorted(pilotCarriers, dataCarriers, right=True) - 1
        upper_indices = lower_indices + 1
        lower_indices = torch.clamp(lower_indices, 0, num_pilot - 1).to(input.device) 
        upper_indices = torch.clamp(upper_indices, 0, num_pilot - 1).to(input.device) 
        weights = (dataCarriers.float() - pilotCarriers[lower_indices].float()) / (pilotCarriers[upper_indices].float() - pilotCarriers[lower_indices].float())

        temperature = 0.0001  # You can adjust this value
        input_decoding_snn = torch.zeros_like(input).transpose(0,1) # (time_step, batch_size, num_neurons)
        rec_rate = 0

        # E_max = 10 ** (self.E / 10)  # transform db to Watt
        E_max = 10 ** ((self.E - 30) / 10)   # transform dbm to Watt
        # Power allocation over time
        T =  input.size(1)
        E_t = self.normalized_power(T,E_max,self.b)*0.0+E_max
        tx_power = []
        tot_spike = 0.0
        for t in range(input.size(1)):
            OFDM_data = torch.zeros((batch_size, self.num_ofdma, num_allcarriers), dtype=torch.complex64,device=input.device)
 
            OFDM_data[:, :, pilotCarriers] = self.pilotValue * torch.ones_like(OFDM_data[:, :, pilotCarriers],device=input.device)  # assign a known pilot symbol to all pilot subcarriers in an OFDM symbol

            # calculate the power scaling 
            channel_power = (1/(2*self.num_path))**0.5 # ensures unit power per tap in Rayleigh fading
            channelResponse = torch.normal(mean=0, std=channel_power, size=(batch_size, self.num_ofdma, self.num_path),device=input.device) +1j * torch.normal(mean=0, std=channel_power, size=(batch_size, self.num_ofdma, self.num_path),device=input.device) # the impulse response of the wireless channel


            H_exact, H_power_gain = self._simulated_H_dist(self.dist, channelResponse, num_allcarriers)
            # noise_power = 1e-3
            noise_std = self._simulated_noise_std()
            noise_power = noise_std**2
            noise_rx = torch.normal(mean=0, std=noise_std, size=(batch_size, self.num_ofdma, num_allcarriers),device=input.device)
            pam_values = pam_points[input[:, t, :].long()] + 0.j  # (batch_size, num_neurons)
            if self.adaptive_power:
                E_watt = self._compute_adaptive_power(
                    rx_snr_db=self.rx_snr_db,
                    H_power_gain=H_power_gain,
                    noise_power=noise_power,
                    num_data_carriers=num_data_carriers,
                    pam_values=pam_values.clone(),
                    mode=self.power_constrs
                )
            else:
                E_watt = E_t[t]


            if self.power_constrs == 'block':
                gamma_raw = num_data_carriers * E_watt/(torch.sum(pam_values.abs() ** 2, dim=1,keepdim=True)) 
                gamma = torch.where(torch.isfinite(gamma_raw), gamma_raw, torch.zeros_like(gamma_raw))
                _tx_power = E_watt * 38.09*1e-6 * num_data_carriers
            elif self.power_constrs == 'peak':
                gamma_raw = E_watt / torch.abs(pam_values).pow(2).max(dim=-1,keepdim=True)[0]
                gamma = torch.where(torch.isfinite(gamma_raw), gamma_raw, torch.zeros_like(gamma_raw))
                _tx_power = E_watt * 38.09*1e-6 * torch.abs(pam_values).sum(1)
            _tx_power = torch.tensor(_tx_power, device=input.device).mean()
            tx_power.append(_tx_power)
            OFDM_data[:, :, dataCarriers] = (pam_values*torch.sqrt(gamma)).to(torch.complex64).unsqueeze(1).expand(-1, self.num_ofdma, -1).to(input.device)# assign all the data symbols to their corresponding data subcarriers in an OFDM symbol
            
            signal_rx = OFDM_data * H_exact
            OFDM_demod = signal_rx + noise_rx
            #compute received energy

            # Only use data subcarriers to compute SNR
            # print(rx_snr_db)
            # print(10 * torch.log10(E_watt)+30)
            pilots = OFDM_demod[:, :, pilotCarriers] # (batch_size, num_ofdm, num_pilots)
            # print(OFDM_data[:, :, pilotCarriers] .abs().mean(dim=[-2,-1],keepdim=True))
            Hest_at_pilots = pilots / self.pilotValue  # (batch_size, num_ofdm, num_pilots)

            Hest_abs = torch.lerp(
                torch.abs(Hest_at_pilots[torch.arange(batch_size,device=input.device).unsqueeze(1).unsqueeze(2), torch.arange(self.num_ofdma,device=input.device).unsqueeze(0).unsqueeze(2), lower_indices]),
                torch.abs(Hest_at_pilots[torch.arange(batch_size,device=input.device).unsqueeze(1).unsqueeze(2), torch.arange(self.num_ofdma,device=input.device).unsqueeze(0).unsqueeze(2), upper_indices]),
                weights.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, num_data_carriers)
            )
            # Interpolation for the phases of the channel estimates
            Hest_phase = torch.lerp(
                torch.angle(Hest_at_pilots[torch.arange(batch_size,device=input.device).unsqueeze(1).unsqueeze(2), torch.arange(self.num_ofdma,device=input.device).unsqueeze(0).unsqueeze(2), lower_indices]),
                torch.angle(Hest_at_pilots[torch.arange(batch_size,device=input.device).unsqueeze(1).unsqueeze(2), torch.arange(self.num_ofdma,device=input.device).unsqueeze(0).unsqueeze(2), upper_indices]),
                weights.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, num_data_carriers)
            )
            Hest = Hest_abs * torch.exp(1j * Hest_phase)  # (batch_size, num_ofdm, num_data_carriers)

            equalized_data = OFDM_demod[:, :, dataCarriers] / Hest /torch.sqrt(gamma).unsqueeze(1) # equalized data symbols (batch_size, num_ofdm, num_data_carriers)
            # print(equalized_data.abs().max(dim=-1,keepdim=True)[0].max(dim=-2,keepdim=True)[0])
            # print(equalized_data.abs().mean(dim=[-2,-1],keepdim=True))
            # equalized_data = equalized_data/equalized_data.abs().max(dim=-1,keepdim=True)[0].max(dim=-2,keepdim=True)[0]# normalization
            # equalized_data = equalized_data/equalized_data.abs().mean(dim=[-2,-1],keepdim=True)# normalization
            # MRC for the equalized data
            combined_OFDM = equalized_data.sum(dim=1)  # Shape: (batch_size, num_carriers)
            mrc_output = combined_OFDM / self.num_ofdma  # Shape: (batch_size, num_carriers)
            distances = torch.abs(mrc_output.unsqueeze(-1) - pam_points)  # (batch_size, num_data_carriers, 2^num_bits+1)
            if self.training:
                soft_probs = torch.softmax(-distances / temperature, dim=-1)  # Shape: (batch_size, num_data_carriers, 2^num_bits+1)
                symbol_indices = torch.arange(2 ** self.num_bit, device=distances.device).expand(distances.size(0), distances.size(1), -1)  # Shape: (batch_size, num_data_carriers, 2^num_bits+1)
                closest_symbols = torch.sum(soft_probs * symbol_indices, dim=-1)  # Shape: (batch_size, num_data_carriers)
            else:
                closest_symbols = torch.argmin(distances, dim=-1)  # (batch_size, num_data_carriers)
            rec_rate +=  torch.sum(torch.eq(closest_symbols, input[:,t,:]))/(batch_size*num_data_carriers)
            # rec = torch.eq(closest_symbols, input[:,t,:]).sum(-1)/(num_data_carriers)
            # print(rec.min())
            # print ("Recovery rate: ", torch.sum(torch.eq(closest_symbols, input[:,t,:]))/(batch_size*num_data_carriers), torch.sum(pam_values.abs()))
            tot_spike += torch.sum(pam_values.abs())
            input_decoding_snn[t] = closest_symbols
        
        tx_power = torch.stack(tx_power,dim=0).sum()
        # self.rec_rate.append(rec_rate/(t+1))
        # print(rec_rate/(t+1))
        # save_pickle(self.rec_rate,name='E_'+str(self.E)+'G'+str(self.num_bit)+'_'+self.modulation, path=os.path.dirname('/LOCAL2/dengyu/project/neurocomm/snr_false_rate/'))
        # print(tot_spike,tx_power,tx_power_spike)
        # print(tx_power/(t+1))
        return input_decoding_snn/(2**self.num_bit), tx_power
    
    def _simulated_H_dist(self, dist, channelResponse, num_allcarriers, antenna_gain=0):
        # distance, 1-150m
        f_c = 3.5    # carrier frequency in GHz
        PL_dB = 32.4 + 17.3 * torch.log10(torch.tensor(dist)) + 20 * torch.log10(torch.tensor(f_c))
        G_tx_dB = antenna_gain  # Transmit antenna gain in dB
        G_rx_dB = antenna_gain  # Receive antenna gain in dB
        PL_dB = PL_dB - G_tx_dB - G_rx_dB
        PL_linear = 10 ** (-PL_dB / 10)
        channelResponse *= PL_linear.sqrt()
        H_exact = torch.fft.fft(channelResponse, n=num_allcarriers, dim=2)  # (batch_size, num_ofdm, num_allcarriers)
        return H_exact, PL_linear


    def _simulated_noise_std(self):
        k = 1.38e-23  # Boltzmann constant [J/K]
        TEMP = 290       # Temperature in Kelvin
        BW = 20e6     # Bandwidth in Hz (20 MHz)
        NF_dB = 9     # UT noise figure in dB
        NF_linear = 10 ** (NF_dB / 10)
        thermal_noise_power = k * TEMP * BW * NF_linear  # in watts
        noise_std = (thermal_noise_power / 2) ** 0.5  # divide by 2 for real/imag separately
        return noise_std
    
    def _compute_adaptive_power(self, rx_snr_db, H_power_gain, noise_power, num_data_carriers, pam_values, mode='block'):
        """
        Compute transmit energy E_watt needed to achieve target rx_snr_db based on the channel.

        Args:
            rx_snr_db (float): Target receive SNR in dB.
            H_exact (Tensor): Channel frequency response (batch_size, num_ofdm, num_carriers), complex.
            pam_values (Tensor): Modulated symbols (batch_size, num_symbols), complex.
            mode (str): 'block' or 'peak' power constraint mode.

        Returns:
            gamma (Tensor): Scaling factor (batch_size, 1).
            E_watt (Tensor): Transmit power per batch (batch_size,)
        """
        # Convert dB SNR to linear
        rx_snr_linear = 10 ** (rx_snr_db / 10)
        # Compute mean channel gain per batch
        # Get noise power
        if mode == 'block':
            E_watt = (rx_snr_linear * noise_power) / (H_power_gain)  # shape: (batch_size,)
        elif mode == 'peak':
            subcarrier_power = torch.abs(pam_values).pow(2).max(dim=-1,keepdim=True)[0]
            E_watt_raw = (rx_snr_linear * noise_power) / ((H_power_gain)*subcarrier_power)  # shape: (batch_size,)
            E_watt = torch.where(torch.isfinite(E_watt_raw), E_watt_raw, torch.zeros_like(E_watt_raw))
        else:
            raise ValueError(f"Unsupported power mode: {mode}. Choose 'block' or 'peak'.")

        return E_watt

    def _generate_mbit_pam(self, m, device):
        if m == 0:
            # For m = 0, we only need two points: 0 and 1
            constellation_points = torch.tensor([0, 1], dtype=torch.float32, device=device)
        else:
            # For m >= 2, we need 2^m points with zero at the first index
                # Determine the number of levels
            L = 2 ** m
            
            # Generate unnormalized PAM levels (symmetrical around 0)
            levels = torch.arange(-L + 1, L, 2, device=device, dtype=torch.float32)
            
            # Calculate the mean square value of unnormalized levels
            mean_square = torch.mean(levels ** 2)
            
            # Normalize the levels so that the mean square equals 1
            constellation_points = levels / torch.sqrt(mean_square)
            constellation_points = torch.cat([torch.tensor([0.0], device=device), constellation_points])

        return constellation_points
        
        

# ## RF neuron
class FCSHD_NeuroComm(nn.Module):
    def __init__(self,E,neuron_params, output_dim = 20):
        super(FCSHD_NeuroComm, self).__init__()
        pool = nn.AvgPool1d(2)
        self.input_size = 700
        self.feature_size = 128
        self.fc =  LinearLayer(self.input_size,self.feature_size,droprate=0.0) #4 for m=3 12 for m=4
        # self.fc =  LinearLayer(self.feature_size,1024,droprate=0.0) #4 for m=3 12 for m=4
        self.wireless_ch = wirelss_ch_odfm(E,neuron_params)
        self.classifier = nn.Linear(self.feature_size,output_dim, bias=False)
        self.li_update = LICell(output_dim,250)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
    def forward(self, input):
        x = self.fc(input)
        x = self.wireless_ch(x)
        x = self.classifier(x)
        x = self.li_update(x)
        return x

