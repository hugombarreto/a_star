from _ast import FunctionDef
import ast
import ctypes
from ctree.c.nodes import FunctionCall, SymbolRef, FunctionDecl
from ctree.cpp.nodes import CppDefine

from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.nodes import Project
from ctree.transformations import PyBasicConversions
from ctree.transformations import CFile
from ctree.types import get_ctype
from ctree.visitors import NodeTransformer
import numpy as np

from AStarSpecializer.a_star import GridAsArray

import logging
from AStarSpecializer.np_functional import TransformFunctionalNP

logging.basicConfig(level=20)


# class SimpleMathTransform(NodeTransformer):
#     def visit_FunctionCall(self, node):
#         func_name = node.func.name
#         print "Node: ", node, ", name: ", func_name
#         if func_name not in np_functional_functions:
#             return node

class LiftFunctions(NodeTransformer):
    lambda_counter = 0
    def __init__(self):
        self._lifted_functions = []

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        self._lifted_functions.append(PyBasicConversions().visit(node))
        return None

    def visit_Lambda(self, node):
        if isinstance(node, ast.Lambda):
            self.generic_visit(node)
            macro_name = "LAMBDA_"+ str(LiftFunctions.lambda_counter)
            LiftFunctions.lambda_counter += 1
            node = PyBasicConversions().visit(node)
            node.name = macro_name
            macro = CppDefine(macro_name, node.params, node.defn[0].value)
            self._lifted_functions.append(macro)

            return SymbolRef(macro_name)
        else:
            return node

    def get_lifted_functions(self):
        return self._lifted_functions

class AStarSpecializer(LazySpecializedFunction):

    def args_to_subconfig(self, args):
        A = args[0]
        arg_type = np.ctypeslib.ndpointer(A.dtype, A.ndim, A.shape)
        return_type = arg_type._dtype_.type
        return {'arg_type': arg_type,'return_type': return_type}

    def transform(self, py_ast, program_config):
        """ This method converts the python AST to C AST
        :param py_ast:
        :param program_config:
        :return:
        """
        print "Transform"
        arg_type = program_config.args_subconfig['arg_type']
        return_type = program_config.args_subconfig['return_type']

        lift_functions = LiftFunctions()
        tree_fn = py_ast.find(FunctionDef, name="apply")
        for statement in tree_fn.body:
            lift_functions.visit(statement)
        lift_functions_list = lift_functions.get_lifted_functions()

        tree = PyBasicConversions().visit(py_ast)

        tree_fn = tree.find(FunctionDecl, name="apply")
        tree_fn.return_type = return_type()
        tree_fn.params[0].type = arg_type()

        functional_np = TransformFunctionalNP(arg_type)
        tree = functional_np.visit(tree)
        lift_functions_list += functional_np.specialized_functions

        tree = lift_functions_list + [tree.body[0]]

        tree = CFile("generated", tree)
        tree = PyBasicConversions().visit(tree)

        generated_c_file = CFile("generated", tree)
        return [generated_c_file]

    def finalize(self, transform_result, program_config):
        # creates a project, transform_result is the return of the
        # transform method we created above, since we made it a single object,
        # not a list or a tuple we may use it without indexing
        project = Project(transform_result) # Project holds a list of files

        # same as in the above method
        arg_type = program_config.args_subconfig['arg_type']

        return_type = arg_type._dtype_.type

        # this creates a C function prototype that returns something of type
        # return_type and take one argument of type arg_type, here they are
        # both the same
        entry_type = ctypes.CFUNCTYPE(return_type, arg_type)

        return ConcreteFunction("apply", project, entry_type)


class ConcreteFunction(ConcreteSpecializedFunction):
    def __init__(self, entry_name, project_node, entry_typesig):
        self._c_function = self._compile(entry_name, project_node,
                                         entry_typesig)

    def __call__(self, A):
        return self._c_function(A)


def get_specialized_a_star_grid(grid_class):
    assert(issubclass(grid_class, GridAsArray))

    class SpecializedAStar(grid_class):
        def __init__(self, grid):
            """
            Args:
              grid (numpy.array): The any dimensions grid to be used for the
                A* algorithm, the barriers are represented by `numpy.inf`
            """
            super(SpecializedAStar, self).__init__(grid)
            print "SpecializedAStar"
            py_heuristic = super(SpecializedAStar,
                                self)._calculate_1_norm
            c_heuristic = AStarSpecializer.from_function(py_heuristic,
                                                        'Heuristic')

            self._calculate_1_norm = c_heuristic


    return SpecializedAStar