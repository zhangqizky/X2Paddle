#   Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
from __future__ import division
import paddle.fluid as fluid
from paddle.fluid.initializer import Constant
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.proto import framework_pb2
from collections import OrderedDict
import copy
import numpy
import time
import collections
import sys
import os
import six


class PaddleLayer(object):
    def __init__(self, kernel, inputs, outputs, **kwargs):
        assert isinstance(
            inputs,
            dict), "parameter 'inputs' for PaddleLayer should be type of dict"
        assert isinstance(
            outputs,
            list), "parameter 'outputs' for PaddleLayer should be type of list"
        for k, v in inputs.items():
            if isinstance(v, list):
                for i in v:
                    assert isinstance(
                        i, six.string_types
                    ), "value in inputs should be type of string or list of string"
            else:
                assert isinstance(v, six.string_types) or isinstance(
                    v, list
                ), "value in inputs should be type of string or list of string"
        for v in outputs:
            assert isinstance(
                v, six.
                string_types), "elements in outputs should be type of string"
        self.kernel = kernel
        self.inputs = inputs
        self.outputs = outputs
        self.attrs = kwargs
        self.id = str(time.time())

    def add_block(self, block):
        block.father_layer = self
        self.blocks.append(block)

    def get_code(self, with_outputs=True):
        code = ""

        #        if len(self.outputs) == 1:
        #            code = self.outputs[0]
        #        else:
        #            for output in self.outputs:
        #                code += "{}, ".format(output)
        #            code = code.strip(", ")
        #        code += " = "

        code += "{}(".format(self.kernel)
        for k, v in self.inputs.items():
            if isinstance(v, list):
                code += "{}=[{}], ".format(k, ", ".join(v))
            else:
                code += "{}={}, ".format(k, v)
        for k, v in self.attrs.items():
            code += "{}={}, ".format(k, v)
        code = code.strip(", ")
        code += ")"
        return code


class PaddleProgram(object):
    def __init__(self):
        self.layers = OrderedDict()
        self.edges_out = dict()
        self.edges_in = dict()
        self.inputs = list()
        self.outputs = list()
        self.parameters = dict()
        self.father_layer = None

    def clear(self):
        self.layers = OrderedDict()
        self.edges_out = dict()
        self.edges_in = dict()
        self.inputs = list()
        self.outputs = list()
        self.parameters = dict()

    def add_layer(self, kernel, inputs, outputs, **kwargs):
        layer = PaddleLayer(kernel, inputs, outputs, **kwargs)
        layer_id = str(len(self.layers))
        if self.father_layer is not None:
            layer_id = "{}.{}.{}".format(layer_id,
                                         len(self.father_layer.blocks()),
                                         self.father_layer.id)
        self.layers[layer_id] = layer
        return layer_id

    def del_layer(self, layer_id):
        layer = self.layers[layer_id]
        outputs = self.edges_out.get(layer_id, [])
        inputs = self.edges_in.get(layer_id, [])

        assert len(
            inputs) <= 1, "There should be 0 or 1 input for deleted layer."

        if len(inputs) == 0:
            for out in outputs:
                while layer_id in self.edges_in[out]:
                    index = self.edges_in[out].index(layer_id)
                    del self.edges_in[out][index]

                input_keys = list(self.layers[out].inputs.keys())
                for k in input_keys:
                    if self.layers[out].inputs[k] == layer.outputs[0]:
                        del self.layers[out].inputs[k]

            del self.layers[layer_id]
            if layer_id in self.edges_in:
                del self.edges_in[layer_id]
            if layer_id in self.edges_out:
                del self.edges_out[layer_id]
            return

        # 将所有输出layer的输入layer进行替换
        for out in outputs:
            for i in range(len(self.edges_in[out])):
                if self.edges_in[out][i] == layer_id:
                    self.edges_in[out][i] = inputs[0]

        # 将输出layer赋给输入layer的输出
        replace_index = self.edges_out[inputs[0]].index(layer_id)
        del self.edges_out[inputs[0]][replace_index]
        for i, out in enumerate(outputs):
            self.edges_out[inputs[0]].insert(replace_index + i, out)
            for k, v in self.layers[out].inputs.items():
                if v == layer.outputs[0]:
                    self.layers[out].inputs[k] = list(layer.inputs.values())[0]

        del self.layers[layer_id]
        if layer_id in self.edges_out:
            del self.edges_out[layer_id]
        if layer_id in self.edges_in:
            del self.edges_in[layer_id]

    def build(self):
        outputs_from_nodes = dict()
        for layer_id, layer in self.layers.items():
            for input_key, input_var in layer.inputs.items():
                vs = input_var
                if not isinstance(vs, list):
                    vs = [vs]
                for v in vs:
                    assert v in outputs_from_nodes, "Couldn't find {} in previous layers, the layers should be make by topological sort".format(
                        v)
                    in_layer_id = outputs_from_nodes[v]
                    if in_layer_id not in self.edges_out:
                        self.edges_out[in_layer_id] = list()
                    self.edges_out[in_layer_id].append(layer_id)

                    if layer_id not in self.edges_in:
                        self.edges_in[layer_id] = list()
                    self.edges_in[layer_id].append(in_layer_id)
            for output in layer.outputs:
                outputs_from_nodes[output] = layer_id

        layer_ids = copy.deepcopy(list(self.layers.keys()))
        for layer_id in layer_ids:
            if len(self.edges_in.get(layer_id, [])) == 0 and len(
                    self.edges_out.get(layer_id, [])) == 0:
                del self.layers[layer_id]

    def gen_code(self, code_dir):
        def write_code(f, code_list, indent=0):
            indent_blank = "    " * indent
            for code_line in code_list:
                if code_line.strip() == "":
                    f.write('\n')
                else:
                    f.write(indent_blank + code_line + '\n')

        if not os.path.exists(code_dir):
            os.makedirs(code_dir)
        f = open(os.path.join(code_dir, 'x2paddle_model.py'), 'w')

        write_code(
            f, [
                "from paddle.fluid.initializer import Constant",
                "from paddle.fluid.param_attr import ParamAttr",
                "import paddle.fluid as fluid"
                "", "def x2paddle_net():"
            ],
            indent=0)
        for layer_id, layer in self.layers.items():
            edges_in = self.edges_in.get(layer_id, [])
            edges_out = self.edges_out.get(layer_id, [])
            if len(edges_in) == 0 and len(edges_out) == 0:
                continue

            line = ""

            if len(layer.outputs) == 1:
                line = layer.outputs[0]
            else:
                for output in layer.outputs:
                    line += "{}, ".format(output)
                line = line.strip(", ")

            line += " = {}(".format(layer.kernel)
            for k, v in layer.inputs.items():
                if isinstance(v, list):
                    line += "{}=[{}], ".format(k, ", ".join(v))
                else:
                    line += "{}={}, ".format(k, v)
            for k, v in layer.attrs.items():
                line += "{}={}, ".format(k, v)
            line = line.strip(", ")
            line += ")"
            write_code(f, [line], indent=1)

        write_code(
            f, [
                "return [{}], [{}]".format(", ".join(self.inputs),
                                           ", ".join(self.outputs))
            ],
            indent=1)
        f.close()

    def gen_model(self, save_dir):
        code_dir = os.path.join(save_dir, 'model_with_code')
        infer_dir = os.path.join(save_dir, 'inference_model')
        self.gen_code(code_dir)
        sys.path.append(code_dir)
        import x2paddle_model
        scope = fluid.Scope()
        startup_program = fluid.Program()
        main_program = fluid.Program()
        with fluid.scope_guard(scope):
            with fluid.program_guard(main_program, startup_program):
                inputs, outputs = x2paddle_model.x2paddle_net()
                exe = fluid.Executor(fluid.CPUPlace())
                exe.run(startup_program)

                param_dir = os.path.join(code_dir, 'weights')
                for k, v in self.parameters.items():
                    if scope.find_var(k):
                        self.dump_parameter(k, v, param_dir)

                def if_exist(var):
                    b = os.path.exists(
                        os.path.join(os.path.join(param_dir, var.name)))
                    return b

                fluid.io.load_vars(
                    exe, param_dir, main_program, predicate=if_exist)
                fluid.io.save_inference_model(
                    dirname=infer_dir,
                    feeded_var_names=[i.name for i in inputs],
                    target_vars=outputs,
                    executor=exe)
        print("Model has been converted, saved in {}".format(save_dir))
        print("=====Model inputs info=====")
        for ipt in self.inputs:
            print("Tensor: {}".format(ipt))
        print("=====Model outputs info====")
        for out in self.outputs:
            print("Tensor: {}".format(out))

    def dump_parameter(self, param_name, param, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        dtype_map = {
            "int16": [framework_pb2.VarType.INT16, 'h'],
            "int32": [framework_pb2.VarType.INT32, 'i'],
            "int64": [framework_pb2.VarType.INT64, 'q'],
            "float16": [framework_pb2.VarType.FP16, 'e'],
            "float32": [framework_pb2.VarType.FP32, 'f'],
            "float64": [framework_pb2.VarType.FP64, 'd'],
            "bool": [framework_pb2.VarType.BOOL, None]
        }
        shape = param.shape
        if str(param.dtype) in ['uint8', 'uint_8', 'bool']:
            param = param.astype('int64')
        if len(shape) == 0:
            assert param.size == 1, "Unexpected situation happend!"
            shape = [1]
        assert str(
            param.dtype) in dtype_map, "Unknown dtype {} of params: {}.".format(
                str(param.dtype), param_name)
        fp = open(os.path.join(save_dir, param_name), 'wb')
        numpy.array([0], dtype='int32').tofile(fp)
        numpy.array([0], dtype='int64').tofile(fp)
        numpy.array([0], dtype='int32').tofile(fp)
        tensor_desc = framework_pb2.VarType.TensorDesc()
        tensor_desc.data_type = dtype_map[str(param.dtype)][0]
        tensor_desc.dims.extend(shape)
        desc_size = tensor_desc.ByteSize()
        numpy.array([desc_size], dtype='int32').tofile(fp)
        fp.write(tensor_desc.SerializeToString())
        param.tofile(fp)
        fp.close()

    def visualize(self, save_dir):
        from graphviz import Digraph
        dot = Digraph("PaddleGraph", "Generated by X2Paddle")
        for layer_id, layer in self.layers.items():
            dot.node(layer_id, layer.kernel)

        for layer_id, outputs in self.edges_out.items():
            for out in outputs:
                dot.edge(layer_id, out)

        with open(os.path.join(save_dir, 'graph.dot'), 'w') as f:
            f.write(dot.source)

        dot.format = 'svg'
        dot.render(filename='graph', directory=save_dir)
