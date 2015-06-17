from ctypes import c_int
from ctree.c.nodes import *
from ctree.jit import LazySpecializedFunction
from ctree.visitors import NodeTransformer
from ctree.frontend import get_ast
import numpy as np

np_functional_functions = ["np_map", "np_reduce", "np_filter"]


def np_map(function, array):
    return np.array(map(function, array))


def np_reduce(function, array):
    return reduce(function, array)


def np_filter(function, array):
    return np.array(filter(function, array))


class SpecializeNPFunctional(LazySpecializedFunction):
    def args_to_subconfig(self, args):
        func = args[0]
        A = args[1]

        return {
            'ptr': np.ctypeslib.ndpointer(A.dtype, A.ndim, A.shape),
            'func': func,
        }

    def transform(self, tree, program_config):
        pass


class TransformFunctionalNP(NodeTransformer):
    def __init__(self, array_type):
        self.array_type = array_type
        self.inner_type = self.array_type._dtype_.type()
        self.specialized_functions = []

    def visit_FunctionCall(self, node):
        func_name = node.func.name
        print "Node: ", node, ", name: ", func_name
        if func_name not in np_functional_functions:
            return node

        args = [self.visit(arg) for arg in node.args]

        specialized_func_name = "specialized_" + func_name
        inner_func_name = args[0]
        # for param in args[0].params:
        #     param.type = self.inner_type
        # args[0].return_type = self.inner_type

        number_items = np.prod(self.array_type._shape_)

        if func_name == "np_map":
            return_type = self.array_type()
            defn = [
                For(Assign(SymbolRef("i", c_int()), Constant(0)),
                    Lt(SymbolRef("i"), Constant(number_items)),
                    PreInc(SymbolRef("i")),
                    [
                        Assign(ArrayRef(SymbolRef("A"), SymbolRef("i")),
                               FunctionCall(inner_func_name,
                                            [ArrayRef(SymbolRef("A"),
                                                      SymbolRef("i"))])),
                    ]),
                Return(SymbolRef("A")),
            ]
        elif func_name == "np_reduce":
            return_type = self.inner_type
            for_iteration = [Assign(
                SymbolRef("accumulator"),
                FunctionCall(inner_func_name,
                             [
                                 SymbolRef("accumulator"),
                                 ArrayRef(SymbolRef("A"), SymbolRef("i"))
                             ]
                             )
            )]

            defn = [
                Assign(SymbolRef("accumulator", self.inner_type),
                       ArrayRef(SymbolRef("A"), Constant(0))),
                For(Assign(SymbolRef("i", c_int()), Constant(1)),
                    Lt(SymbolRef("i"), Constant(number_items)),
                    PreInc(SymbolRef("i")),
                    for_iteration
                    ),
                Return(SymbolRef("accumulator")),
            ]

        elif func_name == "np_filter":
            #args[0].return_type = bool
            return_type = self.array_type()
            # TODO implement filter

        # tree = [args[0],
        #         FunctionDecl(return_type, specialized_func_name,
        #                      params=[SymbolRef("A", self.array_type())],
        #                      defn=defn),
        #         FunctionCall(SymbolRef(specialized_func_name), args[1]),
        #         ]
        self.specialized_functions.append(FunctionDecl(return_type, specialized_func_name,
                             params=[SymbolRef("A", self.array_type())],
                             defn=defn))
        tree = FunctionCall(SymbolRef(specialized_func_name), [args[1]])
        # tree = CFile(specialized_func_name, [tree])
        # SpecializeNPFunctional.from_function(node.func, "NP")
        return tree

    def map_transform(self, node):
        pass

    def visit_FunctionDef(self, node):
        print "FunctionDef"
        return node

    # def visit_FunctionDecl(self, node):
    #     print "FunctionDecl: "
    #     try:
    #         node_str = String(node)
    #         print node_str
    #     except:
    #         print "<<not printable>>"
    #     node.params = [self.visit(p) for p in node.params]
    #     node.defn = [self.visit(d) for d in node.defn]
    #     return node
