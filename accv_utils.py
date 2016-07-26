import hickle
import os
import numpy as np
from keras import backend as K
from read_config import read_config

def cache_data(data, path):
    if os.path.isdir(os.path.dirname(path)):
        file = open(path, 'w')
        hickle.dump(data, file)
        file.close()
    else:
        print('Directory doesn\'t exists')        
        
def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        file = open(path, 'r')
        data = hickle.load(file)
    return data
    
def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()


def merge_several_folds_geom(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a *= np.array(data[i])
    a = np.power(a, 1 / nfolds)
    return a.tolist()    
    
    
def load_partial_weights(model, file_path):
    """Load partial layer weights from a HDF5 save file.
        """
    import h5py
    f = h5py.File(file_path, mode='r')

    if hasattr(model, 'flattened_layers'):
        # support for legacy Sequential/Merge behavior
        flattened_layers = model.flattened_layers
    else:
        flattened_layers = model.layers

    if 'nb_layers' in f.attrs:

        for k in range(len(flattened_layers)):
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            flattened_layers[k].set_weights(weights)

    else:
        print('nb_layers attribute missing in given file')
        # new file format
        layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
        if len(layer_names) != len(flattened_layers):
            print('You are trying to load a weight file '
                  'containing ' + str(len(layer_names)) +
                  ' layers into a model with ' +
                  str(len(flattened_layers)) + ' layers.')

        # we batch weight value assignments in a single backend call
        # which provides a speedup in TensorFlow.
        weight_value_tuples = []
        layer_count = 0
        print layer_names
        print len(flattened_layers)
        model_k = 0
        for k, name in enumerate(layer_names):
            # Suriya debug
            print k
            layer_count += 1

            if layer_count > (len(flattened_layers)-2):
                continue
            g = f[name]
            weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
            if len(weight_names):
                weight_values = [g[weight_name] for weight_name in weight_names]
                if read_config('mode'):
                    for weight_value in weight_values:
                        print('Weight value name: {}'.format(weight_value))
                        print('Weight value shape: {}'.format(weight_value.shape))
                if model_k > len(flattened_layers):
                    continue
                layer = flattened_layers[model_k]
                print('model_layer: {}, saved_layer: {}'.format(layer.name, name))
                symbolic_weights = layer.trainable_weights + layer.non_trainable_weights
                if (len(weight_values) != len(symbolic_weights)) and (layer.name != name):
                    print('Layer #' + str(k) +
                                    ' (named "' + layer.name +
                                    '" in the current model) was found to '
                                    'correspond to layer ' + name +
                                    ' in the save file. '
                                    'However the new layer ' + layer.name +
                                    ' expects ' + str(len(symbolic_weights)) +
                                    ' weights, but the saved weights have ' +
                                    str(len(weight_values)) +
                                    ' elements.')
                    model_k += 1
                    layer = flattened_layers[model_k]
                    print('Layers after forwarding through the model')
                    print('model_layer: {}, saved_layer: {}'.format(layer.name, name))
                    symbolic_weights = layer.trainable_weights + layer.non_trainable_weights
                    if (len(weight_values) != len(symbolic_weights)) and (layer.name != name):
                        print('Layer #' + str(k) +
                              ' (named "' + layer.name +
                              '" in the current model) was found to '
                              'correspond to layer ' + name +
                              ' in the save file. '
                              'However the new layer ' + layer.name +
                              ' expects ' + str(len(symbolic_weights)) +
                              ' weights, but the saved weights have ' +
                              str(len(weight_values)) +
                              ' elements.')

                weight_value_tuples += zip(symbolic_weights, weight_values)

            model_k += 1

        K.batch_set_value(weight_value_tuples)

    f.close()