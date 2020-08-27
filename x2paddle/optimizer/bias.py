import copy


class BiasOpt:
    def __init__(self):
        self.conv_layers = [
            'fluid.layers.conv2d', 'fluid.layers.conv2d_transpose'
        ]
        self.act_layers = [
            'fluid.layers.relu', 'fluid.layers.relu6', 'fluid.layers.sigmoid',
            'fluid.layers.exp', 'fluid.layers.tanh', 'fluid.layers.softplus',
            'fluid.layers.leaky_relu'
        ]

    def run(self, graph):
        layers = copy.deepcopy(graph.layers)
        for layer_id, layer in layers.items():
            if layer.kernel in self.conv_layers or layer.kernel == "fluid.layers.transpose":
                if len(graph.edges_out[layer_id]) != 1:
                    continue

                out_layer_id = graph.edges_out[layer_id][0]
                if graph.layers[
                        out_layer_id].kernel != "fluid.layers.elementwise_add":
                    continue
                if graph.layers[out_layer_id].attrs.get('axis', -1) != -1:
                    continue

                in_layer_id = graph.edges_in[out_layer_id]
                bias_layer_id = in_layer_id[1 - in_layer_id.index(layer_id)]
                if graph.layers[
                        bias_layer_id].kernel != "fluid.layers.create_parameter":
                    continue

                bias_layer = graph.layers[bias_layer_id]
                if len(bias_layer.attrs['shape']) != 1:
                    continue
                if len(graph.edges_out[bias_layer_id]) != 1:
                    continue
                if bias_layer.outputs[0] in graph.outputs:
                    continue

                if layer.kernel == "fluid.layers.transpose":
                    if layer.attrs['perm'] != [0, 2, 3, 1]:
                        continue
                    in_layer_id = graph.edges_in[layer_id][0]
                    if graph.layers[in_layer_id].kernel not in self.conv_layers:
                        continue
                    if graph.layers[in_layer_id].attrs['bias_attr'] != False:
                        continue
                    if len(graph.edges_out[in_layer_id]) != 1:
                        continue
                    graph.layers[in_layer_id].attrs[
                        'bias_attr'] = bias_layer.attrs['name']
                    graph.del_layer(bias_layer_id)
                    graph.del_layer(out_layer_id)
                else:
                    graph.layers[layer_id].attrs[
                        'bias_attr'] = bias_layer.attrs['name']
                    graph.del_layer(bias_layer_id)
                    graph.del_layer(out_layer_id)
