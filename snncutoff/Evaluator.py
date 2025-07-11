import torch
import torch.nn as nn
from typing import Type
from snncutoff.cutoff import BaseCutoff
from snncutoff.API import get_cutoff
from snncutoff.utils import OutputHook, sethook
from torch.utils.data import DataLoader
from tqdm import tqdm

class Evaluator:
    def __init__(
        self,
        net: nn.Module,
        args: None,
        cutoff: Type[BaseCutoff] = None,
    ) -> None:
        """A unified, easy-to-use API for evaluating (most) SNN models.

        Args:
            net (nn.Module):
                The base classifier.
            cutoff (Type[BaseCutoff], optional):
                An actual cutoff instance which inherits
                SNNCutoff's BaseCutoff. Defaults to None.
        Raises:
            ValueError:
                If both postprocessor_name and postprocessor are None.
            ValueError:
                If the specified ID dataset {id_name} is not supported.
            TypeError:
                If the passed postprocessor does not inherit BasePostprocessor.
        """
        self.net = net
        self.args = args
        self.neuron = args.neuron['name']
        self.net.eval()
        self.T = args.neuron['T']
        cutoff=get_cutoff(args.cutoff.name)
        self.cutoff = cutoff(T=self.T, snn_settings=args.snn_settings, cutoff_settings=args.cutoff)
        self.add_time_dim = args.snn_settings.add_time_dim

    def evaluation(self,data_loader):
        outputs_list, label_list = self.cutoff.inference(net=self.net, data_loader=data_loader)
        new_label = label_list.unsqueeze(0)
        outputs_list = torch.softmax(outputs_list,dim=-1)
        acc =(outputs_list.max(-1)[1] == new_label).float().sum(1)/label_list.size()[0]
        return acc.cpu().numpy().tolist(), 0.0
    
    def oct_evaluation(self,data_loader):
        outputs_list, label_list = self.cutoff.inference(net=self.net, data_loader=data_loader)
        new_label = label_list.unsqueeze(0)
        index = (outputs_list.max(-1)[1] == new_label).float()
        for t in range(self.T-1,0,-1):
            index[t-1] = index[t]*index[t-1]
        index[-1] = 1.0
        index = torch.argmax(index,dim=0)
        mask = torch.nn.functional.one_hot(index, num_classes=self.T)
        outputs_list = outputs_list*mask.transpose(0,1).unsqueeze(-1)
        outputs_list = outputs_list.sum(0)
        acc = (outputs_list.max(-1)[1]  == new_label[0]).float().sum()/label_list.size()[0]
        return acc.cpu().numpy().item(), (index+1).cpu().numpy()

    def cutoff_evaluation(self,data_loader,train_loader,epsilon=0.0):
        acc, timestep, conf = self.cutoff.cutoff_evaluation(net=self.net, 
                                                            data_loader=data_loader,
                                                            train_loader=train_loader,
                                                            epsilon=epsilon)
        return acc, timestep, conf

    @torch.no_grad()
    def inference(self,input):
        predition = self.net(input)
        return predition.max()
    
    @torch.no_grad()
    def ANN_OPS(self,input_size):
            net = self.net
            print('ANN MOPS.......')
            output_hook = OutputHook(output_type='connection')
            net = sethook(output_hook,output_type='connection')(net)
            inputs = torch.randn(input_size).unsqueeze(0).to(net.device)
            outputs = net(inputs)
            connections = list(output_hook)
            net = sethook(output_hook)(net,remove=True)
            tot_fp = 0
            for name,w,output in connections:
                fin = torch.prod(torch.tensor(w))
                N_neuron = torch.prod(torch.tensor(output))
                tot_fp += (fin*2+1)*N_neuron
            print(tot_fp)
            return tot_fp
    
    def preprocess(self,x):
        if self.add_time_dim:
            x = x.unsqueeze(1)
            x = x.repeat(1,self.T,1,1,1)
        return x.transpose(0,1)[0:self.T]
    
    @torch.no_grad()
    def SNN_SOP_Count(self,
                    data_loader: DataLoader,
                    progress: bool = True):
            net = self.net
            connections = []
            print('get connection......')
            for data, label in tqdm(data_loader,
                            disable=not progress):
                data = data.cuda()
                data = self.preprocess(data)
                output_hook = OutputHook(output_type='connection')
                net = sethook(output_hook,output_type='connection')(net)
                outputs = net(data)
                connections = list(output_hook)
                net = sethook(output_hook)(net,remove=True)
                fin_tot = []
                for name,w,output in connections:
                    fin = torch.prod(torch.tensor(w))
                    fin_tot.append(fin)
                break

            print('SNN SOP.......')
            tot_sop = 0
            i = 0
            for data, label in tqdm(data_loader,
                            disable=not progress):
                data = data.cuda()
                data = self.preprocess(data)
                label = label.cuda()
                output_hook = OutputHook(output_type='activation')
                net = sethook(output_hook,output_type='activation')(net)
                outputs = net(data)
                connections = list(output_hook)
                net = sethook(output_hook)(net,remove=True)
                s_avg = fin_tot[0]*data.sum()/torch.prod(torch.tensor(data.size()[2:])) # average spike at input layer
                n = 0
                # s_avg = 0.*s_avg
                tot_fp = 0.0
                for output_size, output_spike in connections:
                    fin = fin_tot[n]
                    N_neuron = torch.prod(torch.tensor(output_size)[2:])
                    tot_fp += fin*s_avg*N_neuron 
                    s_avg = ((output_spike>0).float()).sum()/N_neuron
                    print(tot_fp)
                    # print(s_avg/output_spike.shape[0]/output_spike.shape[1])
                    # s_avg = output_spike.sum()/N_neuron
                    n += 1
                    print(n)
                tot_sop += tot_fp
                print(tot_fp)
                i += data.size()[1]
            tot_sop = tot_sop/i
            return tot_sop.cpu().numpy().item()
       
    @torch.no_grad()
    def Energy_Estimate(self,
                  data_loader: DataLoader,
                  progress: bool = True):
        if self.neuron == 'brf':
            N_som_add = 6 
            N_som_mul = 5 
            N_p_add = 1 
            N_p_mul = 0 
        elif self.neuron == 'rf':
            N_som_add = 4 
            N_som_mul = 4 
            N_p_add = 1 
            N_p_mul = 0 
        elif self.neuron == 'alif':
            N_som_add = 2 
            N_som_mul = 3 
            N_p_add = 2 
            N_p_mul = 0 
        elif self.neuron == 'lif':
            N_som_add = 1 
            N_som_mul = 2 
            N_p_add = 1 
            N_p_mul = 0 



        E_add = 0.1
        E_mul = 3.2
        net = self.net
        connections = []
        print('get connection......')
        for data, label in tqdm(data_loader,
                        disable=not progress):
            data = data.cuda()
            data = self.preprocess(data)
            output_hook = OutputHook(output_type='connection')
            net = sethook(output_hook,output_type='connection')(net)
            outputs = net(data)
            connections = list(output_hook)
            net = sethook(output_hook)(net,remove=True)
            fin_tot = []
            for name,w,output in connections:
                fin = torch.prod(torch.tensor(w))
                fin_tot.append(fin)
            break

        print('SNN SOP.......')
        tot_sop = 0
        E_som = 0
        E_synapse = 0
        i = 0
        for data, label in tqdm(data_loader,
                        disable=not progress):
            data = data.cuda()
            data = self.preprocess(data)
            label = label.cuda()
            output_hook = OutputHook(output_type='activation')
            net = sethook(output_hook,output_type='activation')(net)
            outputs = net(data)
            connections = list(output_hook)
            net = sethook(output_hook)(net,remove=True)
            r = fin_tot[0]*data.sum()/torch.prod(torch.tensor(data.size()[2:])) # average spike at input layer
            n = 0
            r = 0.*r                                            #the input synaptic operations is ignored
            tot_fp = 0.0
            T = data.shape[0] #get the total number of time steps 
            E_som_l = 0
            E_synapse_l = 0
            for output_size, output_spike in connections:
                fin = fin_tot[n]                                #get the number of pre-synaptic neurons
                K = torch.prod(torch.tensor(output_size)[2:])   #get the number of neurons K in layer l
                syn_ops = fin*r*K                               #get the synaptic operations of layer l-1 per sample 
                tot_fp += syn_ops                               #get the synaptic operations of layer l-1
                r = ((output_spike>0).float()).sum()/K/output_size[1]          #get the averge spiking rate of layer l per sample

                _E_som_l = T*K*(N_som_add*E_add+N_som_mul*E_mul) + r*(N_p_add*E_add+N_p_mul*E_mul) #get the energy per smaple 
                E_som_l += _E_som_l*output_size[1]                                                    #get the energy per batch
                E_synapse_l += syn_ops*E_add*output_size[1]                                        #get the energy per batch

                # print(s_avg/output_spike.shape[0]/output_spike.shape[1])
                # s_avg = output_spike.sum()/N_neuron
                n += 1

            tot_sop += tot_fp
            E_som += E_som_l
            E_synapse += E_synapse_l
            i += data.size()[1]
        
        tot_sop = tot_sop/i
        E_som = E_som/i
        E_synapse = E_synapse/i
        E_total = E_som + E_synapse
        return E_total.cpu().numpy().item(),E_som.cpu().numpy().item(),E_synapse.cpu().numpy().item(), tot_sop.cpu().numpy().item()

    @torch.no_grad()
    def energy_estimate_layer_its(self,
                  data_loader: DataLoader,
                  progress: bool = True):
        if self.neuron == 'brf':
            N_som_add = 6 
            N_som_mul = 5 
            N_p_add = 1 
            N_p_mul = 0 
        elif self.neuron == 'rf':
            N_som_add = 4 
            N_som_mul = 4 
            N_p_add = 1 
            N_p_mul = 0 
        elif self.neuron == 'alif':
            N_som_add = 2 
            N_som_mul = 3 
            N_p_add = 2 
            N_p_mul = 0 
        elif self.neuron == 'lif':
            N_som_add = 1 
            N_som_mul = 2 
            N_p_add = 1 
            N_p_mul = 0 



        E_add = 0.1
        E_mul = 3.2
        net = self.net
        connections = []
        print('get connection......')
        for data, label in tqdm(data_loader,
                        disable=not progress):
            data = data.cuda()
            data = self.preprocess(data)
            output_hook = OutputHook(output_type='connection')
            net = sethook(output_hook,output_type='connection')(net)
            outputs = net(data)
            connections = list(output_hook)
            net = sethook(output_hook)(net,remove=True)
            fin_tot = [0.0,]
            for name,w,output in connections:
                fin = torch.prod(torch.tensor(w))
                fin_tot.append(fin)
            break

        print('SNN SOP.......')
        tot_sop = 0
        E_som = 0
        E_synapse = 0
        i = 0
        for data, label in tqdm(data_loader,
                        disable=not progress):
            data = data.cuda()
            data = self.preprocess(data)
            label = label.cuda()
            output_hook = OutputHook(output_type='activation')
            net = sethook(output_hook,output_type='activation')(net)
            outputs = net(data)
            connections = list(output_hook)
            net = sethook(output_hook)(net,remove=True)
            T = data.shape[0] #get the total number of time steps 
            E_som_l = []
            E_synapse_l = [] 
            spike_l = []
            r = fin_tot[0]*data.sum()/torch.prod(torch.tensor(data.size()[2:])) # average spike at input layer
            n=0
            r=0.*r
            for output_size, output_spike in connections:
                fin = fin_tot[n]                                #get the number of pre-synaptic neurons
                K = torch.prod(torch.tensor(output_size)[2:])   #get the number of neurons K in layer l
                syn_ops = fin*r*K                               #get the synaptic operations of layer l-1 per sample 
                spike_l.append(syn_ops)                                #get the synaptic operations of layer l-1
                r = ((output_spike>0).float()).sum()/K/output_size[1]          #get the averge spiking rate of layer l per sample
                # spike_per_t = ((output_spike>0).float()).sum(dim=-1)
                # print(spike_per_t.max(),spike_per_t.min())
                _E_som_l = T*K*(N_som_add*E_add+N_som_mul*E_mul) + r*(N_p_add*E_add+N_p_mul*E_mul) #get the energy per smaple 
                E_som_l.append(_E_som_l*output_size[1])                                                 #get the energy per batch
                E_synapse_l.append(syn_ops*E_add*output_size[1])                                        #get the energy per batch

                # print(s_avg/output_spike.shape[0]/output_spike.shape[1])
                # s_avg = output_spike.sum()/N_neuron
                n += 1
            E_som_l = torch.stack(E_som_l,dim=0)
            E_synapse_l = torch.stack(E_synapse_l,dim=0)
            spike_l = torch.stack(spike_l,dim=0)
            E_som += E_som_l
            E_synapse += E_synapse_l
            tot_sop += spike_l
            i += data.size()[1]
        
        tot_sop = tot_sop/i
        E_som = E_som/i
        E_synapse = E_synapse/i
        E_total = E_som + E_synapse
        return E_total.cpu().numpy(),E_som.cpu().numpy(),E_synapse.cpu().numpy(), tot_sop.cpu().numpy()

    @torch.no_grad()
    def energy_estimate_layer(self,
                  data_loader: DataLoader,
                  progress: bool = True):
        if self.neuron == 'brf':
            N_som_add = 6 
            N_som_mul = 5 
            N_p_add = 1 
            N_p_mul = 0 
        elif self.neuron == 'rf':
            N_som_add = 4 
            N_som_mul = 4 
            N_p_add = 1 
            N_p_mul = 0 
        elif self.neuron == 'alif':
            N_som_add = 2 
            N_som_mul = 3 
            N_p_add = 2 
            N_p_mul = 0 
        elif self.neuron == 'lif':
            N_som_add = 1 
            N_som_mul = 2 
            N_p_add = 1 
            N_p_mul = 0 



        E_add = 0.1
        E_mul = 3.2
        net = self.net
        connections = []
        print('get connection......')
        for data, label in tqdm(data_loader,
                        disable=not progress):
            data = data.cuda()
            data = self.preprocess(data)
            output_hook = OutputHook(output_type='connection')
            net = sethook(output_hook,output_type='connection')(net)
            outputs = net(data)
            connections = list(output_hook)
            net = sethook(output_hook)(net,remove=True)
            fin_tot = []
            for name,w,output in connections:
                fin = torch.prod(torch.tensor(w))
                fin_tot.append(fin)
            break

        print('SNN SOP.......')
        tot_sop = 0
        E_som = 0
        E_synapse = 0
        i = 0
        for data, label in tqdm(data_loader,
                        disable=not progress):
            data = data.cuda()
            data = self.preprocess(data)
            label = label.cuda()
            output_hook = OutputHook(output_type='activation')
            net = sethook(output_hook,output_type='activation')(net)
            outputs = net(data)
            connections = list(output_hook)
            net = sethook(output_hook)(net,remove=True)
            r = fin_tot[0]*data.sum()/torch.prod(torch.tensor(data.size()[2:])) # average spike at input layer
            n = 0
            r = 0.*r                                            #the input synaptic operations is ignored
            T = data.shape[0] #get the total number of time steps 
            E_som_l = []
            E_synapse_l = [] 
            spike_l = []
            for output_size, output_spike in connections:
                fin = fin_tot[n]                                #get the number of pre-synaptic neurons
                K = torch.prod(torch.tensor(output_size)[2:])   #get the number of neurons K in layer l
                syn_ops = fin*r*K                               #get the synaptic operations of layer l-1 per sample 
                spike_l.append(syn_ops)                                #get the synaptic operations of layer l-1
                r = ((output_spike>0).float()).sum()/K/output_size[1]          #get the averge spiking rate of layer l per sample
                # spike_per_t = ((output_spike>0).float()).sum(dim=-1)
                # print(spike_per_t.max(),spike_per_t.min())
                _E_som_l = T*K*(N_som_add*E_add+N_som_mul*E_mul) + r*(N_p_add*E_add+N_p_mul*E_mul) #get the energy per smaple 
                E_som_l.append(_E_som_l*output_size[1])                                                 #get the energy per batch
                E_synapse_l.append(syn_ops*E_add*output_size[1])                                        #get the energy per batch

                # print(s_avg/output_spike.shape[0]/output_spike.shape[1])
                # s_avg = output_spike.sum()/N_neuron
                n += 1
            E_som_l = torch.stack(E_som_l,dim=0)
            E_synapse_l = torch.stack(E_synapse_l,dim=0)
            spike_l = torch.stack(spike_l,dim=0)
            E_som += E_som_l
            E_synapse += E_synapse_l
            tot_sop += spike_l
            i += data.size()[1]
        
        tot_sop = tot_sop/i
        E_som = E_som/i
        E_synapse = E_synapse/i
        E_total = E_som + E_synapse
        return E_total.cpu().numpy(),E_som.cpu().numpy(),E_synapse.cpu().numpy(), tot_sop.cpu().numpy()

    @torch.no_grad()
    def energy_estimate_layer_per_step(self,
                  data_loader: DataLoader,
                  progress: bool = True):
        if self.neuron == 'brf':
            N_som_add = 6 
            N_som_mul = 5 
            N_p_add = 1 
            N_p_mul = 0 
        elif self.neuron == 'rf':
            N_som_add = 4 
            N_som_mul = 4 
            N_p_add = 1 
            N_p_mul = 0 
        elif self.neuron == 'alif':
            N_som_add = 2 
            N_som_mul = 3 
            N_p_add = 2 
            N_p_mul = 0 
        elif self.neuron == 'lif':
            N_som_add = 1 
            N_som_mul = 2 
            N_p_add = 1 
            N_p_mul = 0 



        E_add = 0.1
        E_mul = 3.2
        net = self.net
        connections = []
        print('get connection......')
        for data, label in tqdm(data_loader,
                        disable=not progress):
            data = data.cuda()
            data = self.preprocess(data)
            output_hook = OutputHook(output_type='connection')
            net = sethook(output_hook,output_type='connection')(net)
            outputs = net(data)
            connections = list(output_hook)
            net = sethook(output_hook)(net,remove=True)
            fin_tot = []
            for name,w,output in connections:
                fin = torch.prod(torch.tensor(w))
                fin_tot.append(fin)
            break

        print('SNN SOP.......')
        tot_sop = 0
        E_som = 0
        E_synapse = 0
        i = 0
        for data, label in tqdm(data_loader,
                        disable=not progress):
            data = data.cuda()
            data = self.preprocess(data)
            label = label.cuda()
            output_hook = OutputHook(output_type='activation')
            net = sethook(output_hook,output_type='activation')(net)
            outputs = net(data)
            connections = list(output_hook)
            net = sethook(output_hook)(net,remove=True)
            r = fin_tot[0]*data[:,0:1].sum(dim=[1,2])/torch.prod(torch.tensor(data.size()[2:])) # average spike at input layer
            n = 0
            r = 0.*r                                            #the input synaptic operations is ignored
            T = data.shape[0] #get the total number of time steps 
            E_som_l = []
            E_synapse_l = [] 
            spike_l = []
            spike_l_all = []
            for output_size, output_spike in connections:
                output_spike = output_spike[:,0:1]
                fin = fin_tot[n]                                #get the number of pre-synaptic neurons
                K = torch.prod(torch.tensor(output_size)[2:])   #get the number of neurons K in layer l
                syn_ops = fin*r*K                               #get the synaptic operations of layer l-1 per sample 
                # spike_l.append(syn_ops)                                #get the synaptic operations of layer l-1
                # output_spike [T, B, N]
                spike_l.append(output_spike)
                r = ((output_spike>0).float()).sum(dim=[1,2])/K/output_size[1]          #get the averge spiking rate of layer l per sample
                # spike_per_t = ((output_spike>0).float()).sum(dim=-1)
                # print(spike_per_t.max(),spike_per_t.min())
                _E_som_l = T*K*(N_som_add*E_add+N_som_mul*E_mul) + r*(N_p_add*E_add+N_p_mul*E_mul) #get the energy per smaple 
                E_som_l.append(_E_som_l*output_size[1])                                                 #get the energy per batch
                E_synapse_l.append(syn_ops*E_add*output_size[1])                                        #get the energy per batch

                # print(s_avg/output_spike.shape[0]/output_spike.shape[1])
                # s_avg = output_spike.sum()/N_neuron
                n += 1
            E_som_l = torch.stack(E_som_l,dim=0)
            E_synapse_l = torch.stack(E_synapse_l,dim=0)
            spike_l = torch.stack(spike_l,dim=0)
            E_som += E_som_l
            E_synapse += E_synapse_l
            tot_sop += spike_l
            i += data.size()[1]
            break
        # tot_sop = tot_sop/i
        # E_som = E_som/i
        # E_synapse = E_synapse/i
        E_total = E_som + E_synapse
        return E_total.cpu().numpy(),E_som.cpu().numpy(),E_synapse.cpu().numpy(), tot_sop.cpu().numpy(), data




    @torch.no_grad()
    def tx_power_estimate_layer(self,
                  data_loader: DataLoader,
                  progress: bool = True):

        net = self.net
        connections = []
        print('get connection......')
        tx_energy_tot = 0.0
        i = 0.0
        for data, label in tqdm(data_loader,
                        disable=not progress):
            data = data.cuda()
            data = self.preprocess(data)
            output_hook = OutputHook(output_type='wireless')
            net = sethook(output_hook,output_type='wireless')(net)
            outputs = net(data)
            tx_energy = list(output_hook)[0]
            tx_energy_tot += tx_energy
            net = sethook(output_hook)(net,remove=True)
            i += data.size()[1]
        tx_energy_tot = tx_energy_tot/i
        return tx_energy_tot.cpu().numpy()


    @torch.no_grad()
    def tx_power_estimate_layer_per_step(self,data):
        net = self.net
        connections = []
        print('get connection......')
        tx_energy_tot = 0.0
        i = 0.0
        output_hook = OutputHook(output_type='wireless')
        net = sethook(output_hook,output_type='wireless')(net)
        outputs = net(data)
        tx_energy = list(output_hook)[0]
        net = sethook(output_hook)(net,remove=True)
        return tx_energy.cpu().numpy()

    @torch.no_grad()
    def SNN_Spike_Count(self,
                  data_loader: DataLoader,
                  progress: bool = True):
            net = self.net
            connections = []
            print('SNN SOP.......')
            i = 0
            tot_spike = 0
            for data, label in tqdm(data_loader,
                            disable=not progress):
                data = data.cuda()
                # data = self.preprocess(data)
                label = label.cuda()
                output_hook = OutputHook(output_type='activation')
                net = sethook(output_hook,output_type='activation')(net)
                outputs = net(data)
                connections = list(output_hook)
                net = sethook(output_hook)(net,remove=True)
                tot_fp = 0
                # tot_fp = data.sum()
                n = 1
                # for output_size, output_spike in connections:
                #     tot_fp += output_spike.sum()
                #     n += 1
                
                output_size, output_spike = connections[-1]
                tot_fp += output_spike.sum()

                tot_spike += tot_fp
                i += data.size()[1]
            tot_spike = tot_spike/i
            return tot_spike.cpu().numpy().item()
