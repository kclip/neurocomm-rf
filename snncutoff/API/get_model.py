
from snncutoff.models.vgglike import *
from snncutoff.models.vggsnn import *
from snncutoff.layers.ann import *
from snncutoff.layers.snn import *
from snncutoff.utils import add_ann_constraints, add_snn_layers
from snncutoff.models.VGG import VGG
from snncutoff.models.ResNet import get_resnet
from snncutoff.models.sew_resnet import get_sewresnet
from snncutoff.models.fc_snn import FCSNN,FCSHD, FCITS, FCSHD_NeuroComm, FCSNN_ITS
from snncutoff.models import sew_resnet
from .get_layer import get_layer
from .get_regularizer import get_regularizer
from snncutoff.utils import pre_config


def get_model(args):
    input_size  = InputSize(args.dataset.name.lower())
    num_classes  = OuputSize(args.dataset.name.lower())
    if args.snn_settings.method !='ann' and args.snn_settings.method !='snn':
        AssertionError('Training method is wrong!')


    model = ann_models(args, args.training.model,input_size,num_classes,multistep=args.neuron['multistep']) if args.snn_settings.from_relu else snn_models(args.model, args.neuron.T,input_size, num_classes) 
    pre_config_layer = get_layer(args.neuron, args.regularizer, args.snn_settings.method)
    model = pre_config(model, pre_config_layer, method=args.snn_settings.method)
    return model
   

def get_basemodel(name):
    if name.lower() in ['vgg11','vgg13','vgg16','vgg19',]:
        return 'vgg'
    elif name.lower() in ['resnet18','resnet20','resnet34','resnet50','resnet101','resnet152']:
        return 'resnet'
    elif name.lower() in ['sew_resnet18','sew_resnet20','sew_resnet34','sew_resnet50','sew_resnet101','sew_resnet152']:
        return 'sew_resnet'
    else:
        pass

def ann_models(args, model_name, input_size, num_classes,multistep):
    base_model = get_basemodel(model_name)
    if base_model == 'vgg':
        return VGG(model_name.upper(), num_classes, dropout=0)
    elif base_model == 'resnet':
        return get_resnet(model_name, input_size=input_size, num_classes=num_classes,multistep=multistep)
    elif model_name == 'vggann':
        return VGGANN(num_classes=num_classes)
    elif model_name == 'vgg-gesture':
        return VGG_Gesture()
    elif model_name == 'vgg-ncaltech101':
        return VGGANN_NCaltech101()
    elif model_name == 'fcsnn':
        return FCSNN(args)
    elif model_name == 'fcits':
        return FCITS()
    elif model_name == 'fcshd':
        return FCSHD(args)
    elif model_name == 'fcsnn_its':
        return FCSNN_ITS(args)
    elif model_name == 'fcshd_neurocomm':
        return FCSHD_NeuroComm(num_classes=num_classes,E=args.E,neuron_params=args.neuron_params)
    else:
        AssertionError('The network is not suported!')
        exit(0)

def snn_models(model_name, T, input_size, num_classes):
    base_model = get_basemodel(model_name)
    if model_name.lower() == 'vggsnn':
        return VGGSNN(num_classes=num_classes)
    elif base_model=='sew_resnet':
        model = get_sewresnet(model_name, input_size=input_size, num_classes=num_classes,T=T)
        return model
    else:
        AssertionError('This architecture is not suported yet!')

def InputSize(name):
    if 'cifar10-dvs' in name.lower() or 'dvs128-gesture' in name.lower():
        return 128 #'2-128-128'
    elif 'cifar10' in name.lower() or 'cifar100' in name.lower():
        return 32 #'3-32-32'
    elif 'imagenet' in name.lower():
        if 'tiny-imagenet' == name.lower():
            return 64
        else:
            return 224 #'3-224-224'
    elif  'ncaltech101' in name.lower():
        return 240 #'2-240-180'
    else:
        NameError('This dataset name is not supported!')

def OuputSize(name):
    if 'cifar10-dvs' == name.lower() or 'cifar10' == name.lower() :
        return 10
    elif 'dvs128-gesture' == name.lower():
        return 11
    elif 'cifar100' == name.lower():
        return 100
    elif 'ncaltech101' == name.lower():
        return 101
    elif 'imagenet-' in name.lower():
        output_size = name.lower().split("-")[-1]
        return int(output_size)
    elif 'tiny-imagenet' == name.lower():
        return 200
    elif 'imagenet' == name.lower():
        return 1000
    elif 'radioml2018' == name.lower():
        return 24
    else:
        NameError('This dataset name is not supported!')
