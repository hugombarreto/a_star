import ast
import numpy as np
from ctree.c.nodes import *
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.nodes import Project
from ctree.templates.nodes import StringTemplate
from ctree.transformations import PyBasicConversions
from ctree.transformations import CFile
from ctree.visitors import NodeTransformer
from AStarSpecializer.a_star import GridAsArray

import logging
from AStarSpecializer.generic_transformers import FunctionsLifter, \
    ClassToStructureTransformer, MethodCallsTransformer
from AStarSpecializer.np_functional import TransformFunctionalNP
from AStarSpecializer.priority_queue_interface import substitute_priority_queue, \
    PriorityQueueInterface, substitute_node_info

logging.basicConfig(level=logging.DEBUG)


class DictToArrayTransformer(NodeTransformer):
    def __init__(self, grid_type, defines=None):
        self._grid_type = grid_type
        self._number_items = np.prod(grid_type._shape_)
        self.defines = defines or []

    def visit_Assign(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Call) and hasattr(node.value.func, 'id')\
                and node.value.func.id == 'defaultdict':
            struct_name = "struct NodeInfo"
            initialization = None
            for definition in self.defines:
                if isinstance(definition, StructDef) and \
                                definition.struct_type == struct_name:
                    initialization = definition.initializer
            template_entries = {
                'num_items': Constant(self._number_items),
                'struct_body': initialization or Constant(0)
            }
            node = MultiNode(body=[
                StringTemplate("""\
                    const int number_items = $num_items;
                    const struct NodeInfo node_info_initializer = $struct_body;
                    struct NodeInfo nodes_info_temp[number_items];
                    for(int i = 0; i < number_items; ++i) {
                        nodes_info_temp[i] = node_info_initializer;
                    }""", template_entries),
                Assign(SymbolRef("nodes_info", Struct("NodeInfo", ptr=True)),
                       SymbolRef("nodes_info_temp"))
            ])
        return node


class PySpecificAStarConversions(NodeTransformer):
    def __init__(self, grid_type):
        self._grid_type = grid_type
        self.lift = []

    def visit_For(self, node):
        self.generic_visit(node)

        call_neighbors = Assign(SymbolRef("neighbors",
                                          ctypes.POINTER(ctypes.c_int)()),
                                node.iter)
        target_name = node.target.elts[0].id
        target_ref = SymbolRef(target_name)
        index_name = "i"
        index_ref = SymbolRef(index_name)

        weight = SymbolRef(node.target.elts[1].id,
                           self._grid_type._dtype_.type())
        for_loop = For(
            Assign(SymbolRef(index_name, ctypes.c_int()), Constant(0)),
            # TODO this assumes max num of neighbors is 2 * (num of dimensions)
            And(Lt(index_ref, Constant(2 * self._grid_type._ndim_)),
                GtE(ArrayRef(SymbolRef("neighbors"), index_ref), Constant(0))),
            PreInc(index_ref),
            [
                Assign(SymbolRef(target_name, ctypes.c_int()),
                    ArrayRef(SymbolRef("neighbors"), index_ref)),
                Assign(weight, ArrayRef(SymbolRef("self"), target_ref))
            ] + node.body
        )
        node = MultiNode(body=[call_neighbors, for_loop])
        return node


def get_func_declarations():
    dec = PriorityQueueInterface.functions
    dec.update({
        # TODO remove the temporary self functions below
        "self__get_neighbor_weight_list": FunctionDecl(
            return_type=ctypes.POINTER(ctypes.c_int)(),
            name="self__get_neighbor_weight_list",
            params=[SymbolRef("node", ctypes.c_int())],
            defn=[StringTemplate("""return NULL;""")]
        ),
        "self__calculate_heuristic_cost": FunctionDecl(
            return_type=ctypes.c_double(),
            name="self__calculate_heuristic_cost",
            params=[SymbolRef("current_node_id", ctypes.c_int()),
                    SymbolRef("target_node_id", ctypes.c_int())],
            defn=[StringTemplate("""return 0.0;""")]
        ),
    })
    return dec


# TODO find out why g is not getting the type properly
class fixGType(NodeTransformer):
    def visit_Assign(self, node):
        if hasattr(node.targets[0], 'id'):
            if node.targets[0].id == "g":
                node.targets[0] = SymbolRef("g", ctypes.c_double())
        return node

    # FIXME return type should be answer
    def visit_Return(self, node):
        return Return(Constant(0))


class ConcreteHeuristicFunction(ConcreteSpecializedFunction):
    def __init__(self, entry_name, project_node, entry_typesig):
        self._c_function = self._compile(entry_name, project_node,
                                         entry_typesig)

    def __call__(self, A):
        return self._c_function(A)


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

        return ConcreteHeuristicFunction("apply", project, entry_type)


class RecursiveSpecializer(object):
    """Specializes a body and all the eventual methods called from it"""
    def __init__(self, tree, self_object, grid_type):
        self.includes = []
        self.defines = []
        self.typedef = []
        self.local_include = []
        self.func_defs = []
        self.func_def_dict = {}
        self.tree = tree
        self.body = None

        # specific functions parameters
        # TODO may remove this when creating a specializers list
        self.grid_type = grid_type
        self.self_object = self_object

        # self.specializers = []  # not yet

    def visit(self, body=None):
        if body is None:
            body = self.tree.body[0]

        body = substitute_node_info.visit(body)
        self.typedef += substitute_node_info.lift

        body = substitute_priority_queue.visit(body)
        self.typedef += substitute_priority_queue.lift

        dict_to_array_transformer = DictToArrayTransformer(self.grid_type,
                                                           self.typedef)
        body = dict_to_array_transformer.visit(body)

        method_calls_transformer = MethodCallsTransformer(self.func_def_dict,
                                                          self.self_object,
                                                          self)
        body = method_calls_transformer.visit(body)
        self.func_defs = self.func_def_dict.values() + \
                         method_calls_transformer.lift

        py_specific = PySpecificAStarConversions(self.grid_type)
        body = py_specific.visit(body)

        body = fixGType().visit(body)

        body = PyBasicConversions().visit(body)

        self.body = body

        return body

    def get_c_file(self, return_type, param_types):
        self.body.return_type = return_type
        for param, param_type in zip(self.body.params, param_types):
            param.type = param_type

        complete_list = self.includes + self.defines + self.typedef + \
                        self.local_include + self.func_defs + \
                        [self.body]
        return CFile("generated", complete_list, path="/Users/hugo/Desktop")


class AStarSpecializer(LazySpecializedFunction):
    def args_to_subconfig(self, args):
        grid = args[0]
        grid_type = np.ctypeslib.ndpointer(grid.dtype, grid.ndim, grid.shape)
        self.self_object = args[3]
        return {'grid_type': grid_type}

    def transform(self, py_ast, program_config):
        #ctree.browser_show_ast(py_ast)

        grid_type = program_config.args_subconfig['grid_type']
        specializer = RecursiveSpecializer(py_ast, self.self_object, grid_type)
        specializer.typedef += PriorityQueueInterface.definitions
        specializer.func_def_dict = get_func_declarations()  # PriorityQueueInterface.functions
        specializer.includes = [StringTemplate("""\
            #include <stdlib.h>
            #include <math.h>
        """)]
        specializer.defines = [StringTemplate("""
            #define False 0
            #define True 1
            #define inf INFINITY

            //#define NUM_DIMENSIONS 3
            #define GRID_DIMENSIONS {3,4,5}
        """)]
        specializer.local_include = [StringTemplate("""\
            #include "/Users/hugo/Dropbox/Estudos/Berkeley/SEJITS/A*/scratch_dev/priority_queue_interface.h"
        """)]
        specializer.visit()

        #TODO use the following string as a C template
        priority_queue_functions = StringTemplate("""

        unsigned get_parent(unsigned const element_index);
        unsigned get_first_child(unsigned const element_index);

        inline unsigned get_parent(unsigned const element_index)
        {
            return (element_index - 1)/2;
        }

        inline unsigned get_first_child(unsigned const element_index)
        {
            return 2 * element_index + 1;
        }

        struct PriorityQueue* new_heap(unsigned const heap_size)
        {
            struct PriorityQueue* heap;
            if ((heap = malloc(sizeof(*heap))) == NULL) {
                return NULL;
            }
            if ((heap->array = malloc(heap_size * sizeof(*(heap->array)))) == NULL) {
                free(heap);
                return NULL;
            }
            heap->size = 0;
            heap->max_size = heap_size;
            return heap;
        }

        int heap_insert(struct PriorityQueue* const heap, const struct heap_element element)
        {
            if (heap->size >= heap->max_size) {
                return 1;
            }
            unsigned element_index = heap->size;
            for (unsigned parent_index; element_index > 0; element_index = parent_index) {
                parent_index = get_parent(element_index);
                struct heap_element parent = heap->array[parent_index];
                if (element.priority >= parent.priority) {
                    break;
                }
                heap->array[element_index] = parent;
            }
            heap->array[element_index] = element;
            heap->size++;
            return 0;
        }

        struct heap_element* find_heap_min(const struct PriorityQueue* heap)
        {
            if (heap->size == 0) {
                return NULL;
            }
            return &(heap->array[0]);
        }

        int delete_heap_min(struct PriorityQueue* const heap)
        {
            if (heap->size == 0) {
                return 1;
            }
            struct heap_element element = heap->array[--(heap->size)];
            unsigned element_index = 0;
            for (unsigned first_child = get_first_child(element_index); first_child < heap->size; first_child = get_first_child(element_index)){
                unsigned lowest_child;
                unsigned second_child = first_child + 1;
                if ((second_child < heap->size) && (heap->array[first_child].priority > heap->array[second_child].priority)){
                    lowest_child = second_child;
                } else {
                    lowest_child = first_child;
                }
                if (element.priority <= heap->array[lowest_child].priority) {
                    break;
                }
                heap->array[element_index] = heap->array[lowest_child];
                element_index = lowest_child;
            }
            heap->array[element_index] = element;

            return 0;
        }

        void free_heap(struct PriorityQueue* const heap)
        {
            if (heap == NULL) {
                return;
            }
            free(heap->array);
            free(heap);
        }
        """)

        specializer.func_defs.append(priority_queue_functions)

        return_type = ctypes.c_int() # FIXME ctypes.POINTER(ctypes.c_int)()
        param_types = [grid_type(), ctypes.c_int(), ctypes.c_int()]
        c_file = specializer.get_c_file(return_type, param_types)

        return c_file

    def finalize(self, transform_result, program_config):
        project = Project(transform_result)
        grid_type = program_config.args_subconfig['grid_type']
        entry_type = ctypes.CFUNCTYPE(ctypes.c_int,
                                      grid_type, ctypes.c_int,ctypes.c_int)
        return ConcreteSpecializedAStar("apply", project, entry_type)


class ConcreteSpecializedAStar(ConcreteSpecializedFunction):
    def __init__(self, entry_name, project_node, entry_typesig):
        self._c_function = self._compile(entry_name, project_node,
                                         entry_typesig)

    def __call__(self, grid, start, target, self_object):
        return self._c_function(grid, start, target)


def combine_coordinates(coordinates, grid_dimensions):
    dimension_multiplier = 1
    combined_coordinates = 0
    for coordinate, dimension_size in zip(reversed(coordinates),
                                          reversed(grid_dimensions)):
        combined_coordinates += coordinate * dimension_multiplier
        dimension_multiplier *= dimension_size

    return combined_coordinates


def decompose_coordinates(combined_coordinates, grid_dimensions):
    dimension_multiplier = 1
    decomposed_coordinates = []
    for dimension_size in reversed(grid_dimensions):
        new_dimension_multiplier = dimension_multiplier * dimension_size
        multiplied_term = combined_coordinates % new_dimension_multiplier
        decomposed_coordinates.append(multiplied_term / dimension_multiplier)
        dimension_multiplier = new_dimension_multiplier
        combined_coordinates -= multiplied_term

    return decomposed_coordinates[::-1]


def get_specialized_a_star_grid(grid_class):
    assert (issubclass(grid_class, GridAsArray))

    c_a_star = AStarSpecializer.from_function(grid_class.a_star, "a_star")
    setattr(grid_class, 'c_a_star', c_a_star)

    # This is being made so that new types of grids will create new
    # specializations. Passing the grid as a parameter to the function allows
    # ctree to notice the difference and JIT a new function.
    def a_star(self, start, target):
        start = combine_coordinates(start, self.grid_shape)
        target = combine_coordinates(target, self.grid_shape)
        return c_a_star(self._grid, start, target, self)

    grid_class.a_star = a_star

    # py_heuristic = grid_class._calculate_1_norm
    # c_heuristic = HeuristicSpecializer.from_function(py_heuristic, 'Heuristic')
    # grid_class._calculate_1_norm = c_heuristic

    return grid_class
