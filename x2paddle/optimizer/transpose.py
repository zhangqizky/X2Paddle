import copy
import sys


class TransposeOpt:
    def __init__(self):
        self.image_layers = [
            'fluid.layers.conv2d', 'fluid.layers.batch_norm',
            'fluid.layers.conv2d_transpose', 'fluid.layers.resize_nearest',
            'fluid.layers.resize_bilinear', 'fluid.layers.pool2d',
            'fluid.layers.pad2d'
        ]
        self.direct_layers = [
            'fluid.layers.relu', 'fluid.layers.relu6', 'fluid.layers.abs',
            'fluid.layers.sigmoid', 'fluid.layers.exp', 'fluid.layers.rsqrt',
            'fluid.layers.swish_f32', 'fluid.layers.tanh',
            'fluid.layers.softplus', 'fluid.layers.leaky_relu',
            'fluid.layers.floor', 'fluid.layers.erf'
        ]
        self.elementwise_layers = [
            'fluid.layers.elementwise_add', 'fluid.layers.elementwise_sub',
            'fluid.layers.elementwise_mul', 'fluid.layers.elementwise_div'
        ]

    def get_transpose_num(self, graph):
        count = 0
        for layer_id, layer in graph.layers.items():
            if layer.kernel == "fluid.layers.transpose":
                count += 1
        return count

    def strip_direct_layers(self, graph):
        # 构建opt_graph
        # 删除所有direct_layers， 便于对transpose进行优化
        opt_graph = copy.deepcopy(graph)

        remove_layer_ids = set()
        for layer_id, layer in opt_graph.layers.items():
            if layer.kernel in self.direct_layers:
                layer_out = opt_graph.edges_out[layer_id]
                layer_in = opt_graph.edges_in[layer_id]
                if len(layer_out) == 0 or len(layer_in) == 0:
                    continue

                assert len(
                    layer_in
                ) == 1, "There should be only 1 input for direct layers."

                remove_layer_ids.add(layer_id)

        for layer_id in remove_layer_ids:
            opt_graph.del_layer(layer_id)
        return opt_graph

    def run(self, graph):
        optimized_transpose_layers = list()
        modified_layer_attrs = dict()
        modified_parameters = dict()
        scanned_layers = set()
        total_layer_num = len(graph.layers)

        def strip_transpose(_graph):
            layers = copy.deepcopy(_graph.layers)
            for layer_id, layer in layers.items():
                if layer_id in scanned_layers:
                    continue
                scanned_layers.add(layer_id)
                percent = round(len(scanned_layers) / total_layer_num * 100, 2)
                sys.stderr.write("\rOptimize Transpose Layers...{}%".format(
                    percent))

                if layer.kernel != "fluid.layers.transpose":
                    continue
                if layer.attrs["perm"] != [0, 2, 3, 1]:
                    continue

                transpose_layer_ids = list()
                elementwise_layer_ids = list()
                concat_layer_ids = list()
                can_be_optimized = True
                modified_attrs = dict()
                parameter_layers = list()
                parameters = dict()

                for out in _graph.edges_out[layer_id]:
                    if _graph.layers[out].kernel == "fluid.layers.transpose":
                        if _graph.layers[out].attrs["perm"] != [0, 3, 1, 2]:
                            can_be_optimized = False
                            continue
                        transpose_layer_ids.append(out)
                    elif _graph.layers[out].kernel in self.elementwise_layers:
                        elementwise_layer_ids.append(out)
                    elif _graph.layers[out].kernel == "fluid.layers.concat":
                        elementwise_layer_ids.append(out)
                        concat_layer_ids.append(out)
                    else:
                        can_be_optimized = False
                        break

                visited_layers = set()
                while len(elementwise_layer_ids) > 0 and can_be_optimized:
                    current_id = elementwise_layer_ids.pop(0)
                    visited_layers.add(current_id)
                    for out in _graph.edges_out[current_id]:
                        if _graph.layers[
                                out].kernel == "fluid.layers.transpose":
                            if _graph.layers[out].attrs["perm"] != [0, 3, 1, 2]:
                                can_be_optimized = False
                                break
                            if out not in visited_layers:
                                transpose_layer_ids.append(out)
                        elif _graph.layers[
                                out].kernel in self.elementwise_layers:
                            if out not in visited_layers:
                                elementwise_layer_ids.append(out)
                        elif _graph.layers[out].kernel == "fluid.layers.concat":
                            if out not in visited_layers:
                                elementwise_layer_ids.append(out)
                                concat_layer_ids.append(out)
                        else:
                            can_be_optimized = False
                            break

                    all_create_parameter = True
                    for ipt in _graph.edges_in.get(current_id, []):
                        if _graph.layers[
                                ipt].kernel == "fluid.layers.transpose":
                            all_creater_parameter = False
                            if _graph.layers[ipt].attrs["perm"] != [0, 2, 3, 1]:
                                can_be_optimized = False
                                break
                            if ipt not in visited_layers:
                                transpose_layer_ids.append(ipt)
                        elif _graph.layers[
                                ipt].kernel in self.elementwise_layers:
                            all_creater_parameter = False
                            if ipt not in visited_layers:
                                elementwise_layer_ids.append(ipt)
                        elif _graph.layers[ipt].kernel == "fluid.layers.concat":
                            all_creater_parameter = False
                            if ipt not in visited_layers:
                                elementwise_layer_ids.append(ipt)
                                concat_layer_ids.append(ipt)
                        elif _graph.layers[
                                ipt].kernel == "fluid.layers.create_parameter":
                            if ipt not in visited_layers:
                                elementwise_layer_ids.append(ipt)
                                parameter_layers.append(ipt)
                        else:
                            can_be_optimized = False
                            break
                        if all_create_parameter:
                            can_be_optimized = False
                            break

                    if not can_be_optimized:
                        break
                if not can_be_optimized:
                    continue

                concat_layer_ids = list(set(concat_layer_ids))
                for l in concat_layer_ids:
                    axis = _graph.layers[l].attrs.get('axis', 0)
                    _graph.layers[l].attrs['axis'] = [0, 2, 3, 1][axis]
                    modified_attrs[l] = _graph.layers[l].attrs

                parameter_layers = list(set(parameter_layers))
                for l in parameter_layers:
                    for o in _graph.edges_out[l]:
                        if _graph.layers[o].kernel in self.elementwise_layers:
                            axis = _graph.layers[o].attrs.get('axis', -1)
                            _graph.layers[o].attrs['axis'] = [0, 3, 1, 2][axis]
                            modified_attrs[o] = _graph.layers[o].attrs
                        else:
                            can_be_optimized = False
                            break
                        if not can_be_optimized:
                            break
                    s = _graph.layers[l].attrs['shape']
                    p = _graph.parameters[_graph.layers[l].outputs[0]]
                    if len(s) == 4:
                        _graph.layers[l].attrs[
                            'shape'] = [s[0], s[3], s[1], s[2]]
                        modified_attrs[l] = _graph.layers[l].attrs
                        parameters[_graph.layers[l].outputs[0]] = np.transpose(
                            p, (0, 3, 1, 2))
                    elif len(s) == 3:
                        _graph.layers[l].attrs['shape'] = [s[2], s[0], s[1]]
                        modified_attrs[l] = _graph.layers[l].attrs
                        parameters[_graph.layers[l].outputs[0]] = np.transpose(
                            p, (2, 0, 1))

                if not can_be_optimized:
                    continue

                transpose_layer_ids.append(layer_id)
                transpose_layer_ids = list(set(transpose_layer_ids))
                for transpose_layer_id in transpose_layer_ids:
                    _graph.del_layer(transpose_layer_id)
                optimized_transpose_layers.extend(transpose_layer_ids)
                modified_layer_attrs.update(modified_attrs)
                modified_parameters.update(parameters)
                return True
            return False

        before_transpose_num = self.get_transpose_num(graph)

        opt_graph = self.strip_direct_layers(graph)
        total_layer_num = len(opt_graph.layers)
        while strip_transpose(opt_graph):
            pass

        for layer_id in optimized_transpose_layers:
            graph.del_layer(layer_id)

        for layer_id, attrs in modified_layer_attrs.items():
            graph.layers[layer_id].attrs = attrs

        for name, parameter in modified_parameters.items():
            graph.parameters[name] = parameter

        current_transpose_num = self.get_transpose_num(graph)
        print(
            "\nTranspose layers optimized, before: transpose_num={}, after: transpose_num={}".
            format(before_transpose_num, current_transpose_num))
