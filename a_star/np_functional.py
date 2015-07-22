from _ast import Attribute
from ctypes import c_int
from ctree.c.nodes import *
from ctree.visitors import NodeTransformer
import numpy as np

np_functional_functions = ["np_map", "np_reduce", "np_filter",
                           "np_elementwise"]


def np_map(function, array):
    return np.array(map(function, array))


def np_reduce(function, array):
    return reduce(function, array)


def np_filter(function, array):
    return np.array(filter(function, array))


def np_elementwise(function, array1, array2):
    return function(array1, array2)


class TransformFunctionalNP(NodeTransformer):
    def __init__(self, array_type, recursive_specializer=None):
        self.array_type = array_type
        self.inner_type = self.array_type._dtype_.type()
        self.specialized_functions = []
        self.recursive_specializer = recursive_specializer

    def visit_Call(self, node):
        #self.generic_visit(node)
        if isinstance(node.func, Attribute):
            return node

        func_name = node.func.id
        if func_name not in np_functional_functions:
            return node

        args = [self.visit(arg) for arg in node.args]

        specialized_func_name = "specialized_" + func_name
        inner_func_name = args[0]

        number_items = np.prod(self.array_type._shape_)

        params = [SymbolRef("A", self.array_type())]
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
            return_type = self.array_type()
            defn = None  # change that
            # TODO implement filter

        elif func_name == "np_elementwise":
            params.append(SymbolRef("B", self.array_type()))
            return_type = self.array_type()
            defn = [

                For(Assign(SymbolRef("i", c_int()), Constant(0)),
                    Lt(SymbolRef("i"), Constant(number_items)),
                    PreInc(SymbolRef("i")),
                    [
                        Assign(ArrayRef(SymbolRef("A"), SymbolRef("i")),
                               FunctionCall(inner_func_name,
                                            [ArrayRef(SymbolRef("A"),
                                                      SymbolRef("i")),
                                             ArrayRef(SymbolRef("B"),
                                                      SymbolRef("i"))])),
                    ]),
                Return(SymbolRef("A")),
            ]
        else:
            return_type = None
            defn = None

        func_decl = FunctionDecl(
            return_type, specialized_func_name, params=params, defn=defn)

        self.specialized_functions.append(func_decl)
        if self.recursive_specializer is not None and specialized_func_name not\
                in self.recursive_specializer.func_def_names:
            self.recursive_specializer.func_defs.append(func_decl)
            self.recursive_specializer.func_def_names.append(
                specialized_func_name)
        tree = FunctionCall(SymbolRef(specialized_func_name), args[1:])
        return tree
