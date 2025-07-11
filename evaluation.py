import os
import sys
import torch
import warnings
import torch.optim
from datetime import datetime
from argparse import ArgumentParser
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from snncutoff import data_loaders
import numpy as np
from snncutoff.Evaluator import Evaluator
from snncutoff.utils import multi_to_single_step, preprocess_ann_arch
from snncutoff.API import get_model
from snncutoff.utils import set_seed, save_pickle, load_config, dict_to_namespace, save_config
import torch.backends.cudnn as cudnn

def main(args):
    if args.training.seed is not None:
        set_seed(args.training.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.training.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, test_dataset = data_loaders.get_data_loaders(path=args.dataset.dir, data=args.dataset.name, transform=False,resize=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.training.batch_size,
                                              shuffle=False, num_workers=args.training.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.training.batch_size,
                                              shuffle=False, num_workers=args.training.workers, pin_memory=True)
    models = get_model(args)

    inputs = test_dataset[0][0].unsqueeze(1)
    inputs = torch.ones_like(inputs)
    models.to(device)
    inputs = inputs.to(device)
    output = models(inputs)
    i= 0
    path = args.evaluation.model_path

    state_dict = torch.load(path, map_location=torch.device('cpu'))

    missing_keys, unexpected_keys = models.load_state_dict(state_dict, strict=False)
    if not args.snn_settings.multistep:
        if not args.snn_settings.multistep_ann:
            models = preprocess_ann_arch(models)
        models = multi_to_single_step(models, args.multistep_ann, reset_mode=args.reset_mode)

    from snncutoff.models.fc_snn import FCWrapped
    args.wireless_ch.modulation = 'analog'

    E_set=[0] # final 100 digital 2 B
    distance_set = [100]
    insert_index = [2]
    power_constrs = ['peak']
    modulation_set = ['noiseless','analog']  #you can just use noiseless for none wirless chnanel
    # modulation_set = ['analog']

    # rx_snr_set = [-2000, 0,2,4]
    rx_snr_set = [20]


    noiseless_iterations = (
        len(insert_index)       # 1
        * 1                     # only 1 power_constr due to break
        * 1                     # only 1 rx_snr due to break
        * 1                     # only 1 E due to break
        * 1                     # only 1 distance due to break
    )  # = 1

    # For 'analog' modulation:
    # Case 1: rx_snr == -2000 (fully executed)
    analog_rxsnr_full = (
        len(insert_index)       # 1
        * len(power_constrs)    # 2
        * 1                     # only rx_snr = -2000
        * len(E_set)            # 4
        * len(distance_set)     # 5
    )  # = 1 * 2 * 1 * 4 * 5 = 40

    # Case 2: rx_snr > -1000 (adaptive_power=True, breaks after 1 E and 1 dist)
    analog_rxsnr_partial = (
        len(insert_index)       # 1
        * len(power_constrs)    # 2
        * (len(rx_snr_set)-1)   # rx_snr = 0, 2, 4
        * 1                     # only 1 E due to break
        * 1                     # only 1 distance due to break
    )  # = 1 * 2 * 3 * 1 * 1 = 6

    # Total effective iterations:
    total_iterations = noiseless_iterations + analog_rxsnr_full + analog_rxsnr_partial  # 1 + 40 + 6 = 47
    iterations = 0
    neuron = args.neuron['name']
    alpha = float(args.regularizer.alpha)
    results = {
    'meta': {
        'iteration': 0,
        'total_iterations': total_iterations
    },
    'data': {}
    }
    # allowed_configs = {
    #     'lif': [0.0],
    #     'brf': [0.0],
    # }

    # if (neuron in allowed_configs and alpha in allowed_configs[neuron]):
    if True:
        for modulation in modulation_set: 
            args.wireless_ch.modulation = modulation
            for i in insert_index:
                for power_constr in power_constrs:
                    args.wireless_ch.power_constrs = power_constr
                    for rx_snr in rx_snr_set:
                        args.wireless_ch.adaptive_power = False
                        args.wireless_ch.rx_snr_db = rx_snr
                        if rx_snr > -1000:
                            args.wireless_ch.adaptive_power = True
                        for E in E_set:
                            for dist in distance_set: 
                                args.wireless_ch.E = E
                                args.wireless_ch.distance = dist
                                model = build_model(args,device,test_dataset)
                                model = FCWrapped(model, args,insert_index=i).to('cuda')
                                evaluator = Evaluator(model, args)


                                mod =  args.wireless_ch.modulation
                                pcon = args.wireless_ch.power_constrs
                                E = args.wireless_ch.E
                                dist = args.wireless_ch.distance
                                adaptive_power = args.wireless_ch.adaptive_power 
                                rx_snr = args.wireless_ch.rx_snr_db

                                if mod == 'noiseless':
                                    if 'its' in args.dataset.name:
                                        tot_energy_l, soma_energy_l, syn_energy_l, tot_spike_l = evaluator.energy_estimate_layer_its(test_loader)
                                    else:
                                        tot_energy_l, soma_energy_l, syn_energy_l, tot_spike_l = evaluator.energy_estimate_layer_per_step(test_loader)
                                    tx_energy_l = 0.0
                                else: 
                                    tot_energy_l, soma_energy_l, syn_energy_l, tot_spike_l, data = evaluator.energy_estimate_layer_per_step(test_loader)
                                    tx_energy_l = evaluator.tx_power_estimate_layer_per_step(data)

                                acc, loss = evaluator.evaluation(test_loader)

                                iterations += 1
                                key = (neuron, alpha)
                                print(key, acc[-1],'spike:')#, tot_spike_l,'tx_energy:', tx_energy_l,'syn_energy_l:',syn_energy_l)
                                
                                results['meta']['iteration'] = iterations  # update current progress

                                results['data'][key] = {
                                    'metrics': {
                                        'accuracy': acc,
                                        'total_energy': tot_energy_l,
                                        'soma_energy': soma_energy_l,
                                        'syn_energy': syn_energy_l,
                                        'tx_energy': tx_energy_l,
                                        'total_spike': tot_spike_l,
                                    },
                                }
                                path = '/LOCAL2/dengyu/project/isac_snn/Experiment'
                                save_pickle(results,name=neuron+'_results_intro',path=path)
                                if mod == 'noiseless':
                                    break
                            if mod == 'noiseless'  or adaptive_power:
                                break
                        if mod == 'noiseless':
                            break
                    if mod == 'noiseless':
                        break
                if mod == 'noiseless':
                    break
    else:
        print('Simulation skipped.')


def build_model(args,device,test_dataset):
    models = get_model(args)

    inputs = test_dataset[0][0].unsqueeze(1)
    inputs = torch.ones_like(inputs)
    models.to(device)
    inputs = inputs.to(device)
    output = models(inputs)
    path = args.evaluation.model_path

    state_dict = torch.load(path, map_location=torch.device('cpu'))

    missing_keys, unexpected_keys = models.load_state_dict(state_dict, strict=False)
    if not args.snn_settings.multistep:
        if not args.snn_settings.multistep_ann:
            models = preprocess_ann_arch(models)
        models = multi_to_single_step(models, args.multistep_ann, reset_mode=args.reset_mode)
    return models 



def update_nested_config(config, key, value):
    """Update nested config dict with dotted key like 'neuron.T'."""
    keys = key.split('.')
    d = config
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    # Auto-cast value (int, float, bool)
    if value.lower() in ['true', 'false']:
        value = value.lower() == 'true'
    else:
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass  # Keep as string
    d[keys[-1]] = value

if __name__ == '__main__':
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str)
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    config = load_config(args.config)
    args = dict_to_namespace(config)
    args.neuron=config['neuron']
    args.architecture=config['architecture']
    main(args)