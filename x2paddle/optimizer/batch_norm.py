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
            can_be_optimized = True
            if layer.kernel != "fluid.layers.elemenwise_mul":
                can_be_optimized = False
                continue
            input_ids = graph.edges_in[layer_id]
