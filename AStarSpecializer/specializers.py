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
from AStarSpecializer.generic_transformers import LambdaLifter,\
    MethodCallsTransformer, ReturnTypeFinder
from AStarSpecializer.np_functional import TransformFunctionalNP
from AStarSpecializer.priority_queue_interface import transform_priority_queue,\
    transform_node_info, PriorityQueueInterface

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
        # FIXME this assumes max num of neighbors is 2 * (num of dimensions)
        num_neighbors = 2 * self._grid_type._ndim_
        # call_neighbors = Assign(SymbolRef("neighbors",
        #                                   ctypes.POINTER(ctypes.c_int)()),
        #                         node.iter)
        call_neighbors = [ArrayDef(SymbolRef("neighbors", ctypes.c_int()),
                                   num_neighbors,
                                   Array(ctypes.c_int(),
                                         body=[Constant(-1)]*num_neighbors)),
                          FunctionCall(SymbolRef(node.iter.func.name),
                                       [SymbolRef("neighbors"),
                                        SymbolRef(node.iter.args[0].id)])]

        target_name = node.target.elts[0].id
        target_ref = SymbolRef(target_name)
        index_name = "i"
        index_ref = SymbolRef(index_name)

        weight = SymbolRef(node.target.elts[1].id,
                           self._grid_type._dtype_.type())
        for_loop = For(
            Assign(SymbolRef(index_name, ctypes.c_int()), Constant(0)),
            And(Lt(index_ref, Constant(num_neighbors)),
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
            params=[SymbolRef("neighbors", ctypes.POINTER(ctypes.c_int)()),
                    SymbolRef("node", ctypes.c_int())],
            defn=[StringTemplate("""return NULL;""")]
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
        if isinstance(node.value, ast.Name):
            return Return(Constant(np.int32()))
        return node

class RecursiveSpecializer(object):
    """Specializes a body and all the eventual methods called from it"""
    def __init__(self, tree, self_object, grid_type):
        self.includes = []
        self.defines = []
        self.typedef = []
        self.local_include = []
        self.func_def_names = []
        self.func_defs = []
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
        #ctree.browser_show_ast(body)

        body = transform_node_info.visit(body)
        body = transform_priority_queue.visit(body)
        for tdef in transform_node_info.lift + transform_priority_queue.lift:
            if tdef not in self.typedef:
                self.typedef.append(tdef)

        LambdaLifter(self).visit(body)

        coord_type = np.ctypeslib.ndpointer(ctypes.c_int, 1, (self.grid_type._ndim_))
        TransformFunctionalNP(coord_type, self).visit(body)

        dict_to_array_transformer = DictToArrayTransformer(self.grid_type,
                                                           self.typedef)
        body = dict_to_array_transformer.visit(body)

        MethodCallsTransformer(self).visit(body)

        py_specific = PySpecificAStarConversions(self.grid_type)
        body = py_specific.visit(body)

        body = fixGType().visit(body)

        return_type_finder = ReturnTypeFinder(self)
        return_type_finder.visit(body)
        return_type = return_type_finder.return_type

        body = PyBasicConversions().visit(body)

        self.body = body

        self.body.return_type = return_type

        return body

    def get_c_file(self, return_type, param_types):
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
        grid_type = program_config.args_subconfig['grid_type']
        specializer = RecursiveSpecializer(py_ast, self.self_object, grid_type)
        specializer.typedef += PriorityQueueInterface.definitions
        specializer.func_defs += get_func_declarations().values()
        specializer.func_def_names += get_func_declarations().keys()
        specializer.includes = [StringTemplate("""\
            #include <stdlib.h>
            #include <math.h>
        """)]
        specializer.defines = [StringTemplate("""
            #define False 0
            #define True 1
            #define inf INFINITY

            #define NUM_DIMENSIONS 3
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


def decompose_coordinates(combined_coordinates, grid_dimensions, offset=0):
    dimension_multiplier = 1
    decomposed_coordinates = []
    for dimension_size in reversed(grid_dimensions):
        new_dimension_multiplier = dimension_multiplier * dimension_size
        multiplied_term = combined_coordinates % new_dimension_multiplier
        decomposed_coordinates.append(multiplied_term / dimension_multiplier + offset)
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

    return grid_class
