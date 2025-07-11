from snncutoff.layers.ann import ReLU, QCFS, Clip
from snncutoff.layers.snn import SRLayer, SimpleBaseLayer
from snncutoff.neuron import LIF, ALIF, MLIF, BRF, RF
from typing import Type, Union
from .get_regularizer import get_regularizer

# Dictionary mapping for ANN constructors
ann_layers: dict[str, Type] = {
    'relu': ReLU,
    'qcfs': QCFS,
    'clip': Clip,
}

# Dictionary mapping for SNN layers
snn_layers: dict[str, Type] = {
    'srlayer': SRLayer,
    'simplebaselayer': SimpleBaseLayer,
}

# Dictionary mapping for SNN layers
snn_neurons: dict[str, Type] = {
    'lif': LIF,
    'alif': ALIF,
    'mlif': MLIF,
    'brf': BRF,
    'rf': RF,
}


# Function to get the constructor or layer based on name and method
def get_layer(args: dict, regularizer: dict, method: str) -> Union[Type, None]:
    """
    Retrieve the constructor or layer class based on the provided name and method.
    
    Args:
        name (str): The name of the constructor or layer.
        method (str): The method type, either 'ann' or 'snn'.

    Returns:
        Type: The corresponding constructor or layer class.
        None: If the name or method is invalid.
    """
    if method == 'ann':
        layer =  ann_layers.get(args['name'].lower())
        layer = layer(neuron_params=args,
                  regularizer=get_regularizer(regularizer.name.lower(), method),)
        return layer
    elif method == 'snn':
        # act =  ann_activations.get(args.name.lower())
        layer = SRLayer
        layer = layer(neuron=snn_neurons[args['name'].lower()],
                  neuron_params=args,
                  regularizer=get_regularizer(regularizer.name.lower(),method),
        )
        return layer
    else:
        raise ValueError(f"Invalid method: {method}. Expected 'ann' or 'snn'.")