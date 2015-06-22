import ast
import ctypes
from ctree.c.nodes import FunctionCall, SymbolRef, FunctionDecl, Struct, \
    BinaryOp, Constant, Op, StructDef
from ctree.cpp.nodes import CppDefine

from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.nodes import Project
from ctree.transformations import PyBasicConversions
from ctree.transformations import CFile
from ctree.visitors import NodeTransformer, NodeVisitor
import numpy as np

from AStarSpecializer.a_star import GridAsArray

import logging
from AStarSpecializer.np_functional import TransformFunctionalNP

logging.basicConfig(level=20)


class FunctionsLifter(NodeTransformer):
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
            macro_name = "LAMBDA_" + str(self.lambda_counter)
            self.lambda_counter += 1
            node = PyBasicConversions().visit(node)
            node.name = macro_name
            macro = CppDefine(macro_name, node.params, node.defn[0].value)
            self._lifted_functions.append(macro)

            return SymbolRef(macro_name)
        else:
            return node

    def get_lifted_functions(self):
        return self._lifted_functions


class ClassToStructureTransformer(NodeTransformer):
    """Intercepts PriorityQueue declaration and substitute by a `struct`
       the struct definitions are stored in the lift list
    """

    def __init__(self, class_name, defn, self_defined=False):
        self.class_name = class_name
        self.defn = defn
        self.self_defined = self_defined
        self.lift = []

    def visit_Assign(self, node):
        """Will look for class_name constructors"""
        if isinstance(node.value, ast.Call):
            is_self_attribute = isinstance(node.value.func, ast.Attribute) and\
                                node.value.func.value.id == 'self' and \
                                node.value.func.attr == self.class_name
            has_id = hasattr(node.value.func, 'id') and \
                     node.value.func.id == self.class_name

            if self.self_defined and is_self_attribute or \
                            not self.self_defined and has_id:
                self.lift.append(StructDef(self.class_name, self.defn))
                node = Struct(node.targets[0].id, self.class_name)
                return node

        self.generic_visit(node)
        return node

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            pass
        else:
            self.generic_visit(node)
        return node


class CompleteAttrName(NodeVisitor):
    def __init__(self):
        self.complete_name = ""

    def get_complete_name(self, node):
        self.visit(node)
        return self.complete_name

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name):
            self.complete_name += node.value.id
        self.complete_name += "." + node.attr


class MethodCallsTransformer(NodeTransformer):
    def __init__(self):
        self.variable_types = {}

    def visit_Assign(self, node):
        """Keeps track of the variable types"""
        self.generic_visit(node)
        if not hasattr(node, 'value'):
            return node
        node_type = node.value
        if isinstance(node_type, ast.Call):
            node_type = node_type.func.id

        for target in node.targets:
            attr_name = None
            if hasattr(target, 'id'):
                attr_name = target.id
            elif isinstance(target, ast.Attribute):
                attr_name = CompleteAttrName().get_complete_name(node)
            if attr_name is not None:
                if not isinstance(node_type, str):
                    node_type = str(node_type)
                self.variable_types[attr_name] = node_type
        return node

    def visit_Struct(self, node):
        """Keep track of variables initialized as structures"""
        self.generic_visit(node)
        self.variable_types[node.name] = node.struct_type
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        func = node.func
        if isinstance(func, ast.Attribute):
            object_name = func.value.id
            if object_name in self.variable_types:
                fc = FunctionCall(SymbolRef(self.variable_types[object_name]
                                              + "_" + func.attr),
                                  [SymbolRef(object_name)] + node.args)
                return fc

                self.specialized_functions.append(spec_func)
            elif object_name == 'self':
                return FunctionCall(SymbolRef("self_" + func.attr), node.args)
            return FunctionCall(SymbolRef(object_name + "_" + func.attr),
                                node.args)
        return node

    def convert_method_to_c_function(self, name, function):
        pass


class DictToArrayTransformer(NodeTransformer):
    def __init__(self, grid_type):
        self.grid_type = grid_type
        self.number_items = np.prod(grid_type._shape_)
        "NodeInfo nodes_info[][][]"

    def visit_Assign(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Dict):
            node = SymbolRef(node.targets[0], self.grid_type())
        return node


class PySpecificAStarConversions(NodeTransformer):
    def __init__(self, grid_type):
        self.grid_type = grid_type
        self.specialized_functions = []

    def visit_For(self, node):
        self.generic_visit()

        return node


class HeuristicSpecializer(LazySpecializedFunction):
    def args_to_subconfig(self, args):
        A = args[0]
        arg_type = np.ctypeslib.ndpointer(A.dtype, A.ndim, A.shape)
        return_type = arg_type._dtype_.type
        return {'arg_type': arg_type, 'return_type': return_type}

    def transform(self, py_ast, program_config):
        """ This method converts the python AST to C AST
        :param py_ast:
        :param program_config:
        :return:
        """
        arg_type = program_config.args_subconfig['arg_type']
        return_type = program_config.args_subconfig['return_type']

        lift_functions = FunctionsLifter()
        tree_fn = py_ast.find(ast.FunctionDef, name="apply")
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
        project = Project(transform_result)  # Project holds a list of files

        # same as in the above method
        arg_type = program_config.args_subconfig['arg_type']
        return_type = program_config.args_subconfig['return_type']

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


class ConcreteSpecializedAStar(ConcreteSpecializedFunction):
    def __init__(self, entry_name, project_node, entry_typesig):
        self._c_function = self._compile(entry_name, project_node,
                                         entry_typesig)

    def __call__(self, grid, start, target):
        return self._c_function(grid, start, target)


class AStarSpecializer(LazySpecializedFunction):
    def args_to_subconfig(self, args):
        grid = args[0]
        start = args[1]
        arg_type_nodes = np.ctypeslib.ndpointer(start.dtype, start.ndim,
                                                start.shape)
        grid_type = np.ctypeslib.ndpointer(grid.dtype, grid.ndim, grid.shape)
        return {'arg_type_nodes': arg_type_nodes, 'grid_type': grid_type}

    def transform(self, py_ast, program_config):
        lift_list = []

        arg_type_nodes = program_config.args_subconfig['arg_type_nodes']
        grid_type = program_config.args_subconfig['grid_type']

        tree = DictToArrayTransformer(grid_type).visit(py_ast)

        substitute_priority_queue = ClassToStructureTransformer(
            "PriorityQueue",
            [
                BinaryOp(SymbolRef('_heap', ctypes.c_double()), Op.ArrayRef(),
                         Constant(5)),
                SymbolRef('i', ctypes.c_int())
            ]
        )
        tree = substitute_priority_queue.visit(tree)
        lift_list += substitute_priority_queue.lift

        substitute_node_info = ClassToStructureTransformer(
            "NodeInfo",
            [
                SymbolRef('f', ctypes.c_double()),
                SymbolRef('g', ctypes.c_double()),
                SymbolRef('parent', arg_type_nodes()),
                SymbolRef('closed', ctypes.c_bool())
            ],
            self_defined=True
        )
        tree = substitute_node_info.visit(tree)
        lift_list += substitute_node_info.lift

        tree = MethodCallsTransformer().visit(tree)

        py_specific = PySpecificAStarConversions(grid_type=grid_type)
        tree = py_specific.visit(tree)
        tree = PyBasicConversions().visit(tree)

        tree.body[0].return_type = ctypes.c_int()

        tree = lift_list + [tree.body[0]]
        tree = CFile("generated", tree)
        # tree_fn = tree.find(FunctionDecl, name="apply")
        # tree_fn.return_type = return_type()
        # tree_fn.params[0].type = arg_type()

        return [tree]

    def finalize(self, transform_result, program_config):
        project = Project(transform_result)

        arg_type = program_config.args_subconfig['arg_type_nodes']

        entry_type = ctypes.CFUNCTYPE(ctypes.c_int, arg_type, arg_type,
                                      arg_type)

        return ConcreteSpecializedAStar("apply", project, entry_type)


def get_specialized_a_star_grid(grid_class):
    assert (issubclass(grid_class, GridAsArray))

    c_a_star = AStarSpecializer.from_function(grid_class.a_star, "Test")
    grid_class.a_star = c_a_star

    setattr(grid_class, 'c_a_star', c_a_star)

    # This is being made so that new types of grids will create new
    # specializations. Passing the grid as a parameter to the function allows
    # ctree to notice the difference and JIT a new function.
    def a_star(self, start, target):
        return c_a_star(self._grid, start, target)

    grid_class.a_star = a_star

    # py_heuristic = grid_class._calculate_1_norm
    # c_heuristic = HeuristicSpecializer.from_function(py_heuristic, 'Heuristic')
    # grid_class._calculate_1_norm = c_heuristic

    return grid_class

