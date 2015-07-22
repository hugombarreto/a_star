import ast
import numpy as np
from ctree.c.nodes import *
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.nodes import Project
from ctree.templates.nodes import StringTemplate
from ctree.transformations import PyBasicConversions
from ctree.transformations import CFile
from ctree.visitors import NodeTransformer
from a_star.astar import GridAsArray

import logging
from a_star.generic_transformers import LambdaLifter,\
    MethodCallsTransformer, ReturnTypeFinder
from a_star.np_functional import TransformFunctionalNP
from a_star.priority_queue_interface import transform_priority_queue
from a_star.priority_queue_interface import transform_node_info, \
    PriorityQueueInterface

#logging.basicConfig(level=logging.DEBUG)


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


class NeighborWeightListConverter(object):
    def __init__(self, rec_specializer):
        self.rec_specializer = rec_specializer
        self.self_object = rec_specializer.self_object
        self.grid_shape = rec_specializer.grid_type._shape_
        self.grid_dim = rec_specializer.grid_type._ndim_
        self.max_index = reduce(lambda x,y: x*y, self.grid_shape) - 1
        self.num_neighbors = self._get_number_neighbors()
        self.func_name = "self__get_neighbor_weight_list"

    def visit(self, tree):
        if self.func_name not in self.rec_specializer.func_def_names:
            self._add_function()
        return tree

    def _add_function(self):
        neighbor_differences = self._get_neighbor_differences()
        body = [ArrayDef(SymbolRef("neighbor_differences", ctypes.c_int()),
                         self.num_neighbors,
                         Array(ctypes.c_int(), None, neighbor_differences)),
                Assign(SymbolRef("neighbor_index", ctypes.c_int()),
                       Constant(0)),
                For(Assign(SymbolRef("i", ctypes.c_int()), Constant(0)),
                    Lt(SymbolRef("i"), Constant(self.num_neighbors)),
                    PreInc(SymbolRef("i")),
                    [
                        Assign(SymbolRef("neighbor", ctypes.c_int()),
                               Add(SymbolRef("node"),
                                   ArrayRef(SymbolRef("neighbor_differences"),
                                            SymbolRef("i")))),
                        If(Or(Lt(SymbolRef("neighbor"), Constant(0)),
                              Gt(SymbolRef("neighbor"),
                                 Constant(self.max_index))),
                           Continue()),
                        Assign(ArrayRef(SymbolRef("neighbors"),
                                        PostInc(SymbolRef("neighbor_index"))),
                               SymbolRef("neighbor"))
                    ]),
                Return(SymbolRef("neighbors"))
                ]
        func_decl = FunctionDecl(return_type=ctypes.POINTER(ctypes.c_int)(),
                                 name=self.func_name,
                                 params=[
                                     SymbolRef("neighbors",
                                               ctypes.POINTER(ctypes.c_int)()),
                                     SymbolRef("node", ctypes.c_int())
                                 ],
                                 defn=body)

        self.rec_specializer.func_defs.append(func_decl)
        self.rec_specializer.func_def_names.append(self.func_name)

    def _get_neighbor_differences(self):
        neighborhood_matrix = self.self_object.neighborhood_matrix
        grid_multiplier = self._get_grid_multiplier()
        neighbor_differences = list(neighborhood_matrix.dot(grid_multiplier))
        return map(lambda x: Constant(x), neighbor_differences)

    def _get_number_neighbors(self):
        neighborhood_matrix = self.self_object.neighborhood_matrix
        return neighborhood_matrix.shape[0]

    def _get_grid_multiplier(self):
        grid_multiplier = [1]
        mult_accumulator = 1
        for m in reversed(self.grid_shape[1:]):
            mult_accumulator *= m
            grid_multiplier.append(mult_accumulator)

        return np.array(grid_multiplier[::-1])


class NodesInfoTransformer(NodeTransformer):
    def __init__(self, recursive_specializer):
        super(NodesInfoTransformer, self).__init__()
        self.recursive_specializer = recursive_specializer
        self.grid_type = recursive_specializer.grid_type

    def visit_Return(self, node):
        if isinstance(node.value, ast.Name):
            if node.value.id == "nodes_info":
                node = self._get_new_return()
        return node

    def _get_new_return(self):
        return MultiNode([StringTemplate("""\
            if (nodes_info[target_id].parent > 0) {
                for(int i = target_id; i != start_id; i = nodes_parent[i]) {
                    nodes_parent[i] = nodes_info[i].parent;
                }
            }
        """), Assign(SymbolRef("nodes_parent_temp", self.grid_type()),
                     SymbolRef("nodes_parent")),
              Return(SymbolRef("nodes_parent_temp"))])


def get_func_declarations():
    return PriorityQueueInterface.functions


# TODO find out why g is not getting the type properly
class fixGType(NodeTransformer):
    def visit_Assign(self, node):
        if hasattr(node.targets[0], 'id'):
            if node.targets[0].id == "g":
                node.targets[0] = SymbolRef("g", ctypes.c_double())
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

        coord_type = np.ctypeslib.ndpointer(ctypes.c_int, 1,
                                            self.grid_type._ndim_)
        TransformFunctionalNP(coord_type, self).visit(body)

        DictToArrayTransformer(self.grid_type, self.typedef).visit(body)

        NeighborWeightListConverter(self).visit(body)

        MethodCallsTransformer(self).visit(body)

        py_specific = PySpecificAStarConversions(self.grid_type)
        body = py_specific.visit(body)

        NodesInfoTransformer(self).visit(body)

        body = fixGType().visit(body)

        return_type_finder = ReturnTypeFinder(self)
        return_type_finder.visit(body)
        return_type = return_type_finder.return_type

        body = PyBasicConversions().visit(body)

        self.body = body

        self.body.return_type = return_type

        return body

    def get_c_file(self, param_types, return_type=None):
        for param, param_type in zip(self.body.params, param_types):
            param.type = param_type

        if return_type is not None:
            self.body.return_type = return_type

        complete_list = self.includes + self.defines + self.typedef + \
                        self.local_include + self.func_defs + \
                        [self.body]
        return CFile("generated", complete_list)


class AStarSpecializer(LazySpecializedFunction):
    def args_to_subconfig(self, args):
        grid = args[0]
        grid_type = np.ctypeslib.ndpointer(grid.dtype, grid.ndim, grid.shape)
        self.self_object = args[4]
        return {'grid_type': grid_type}

    def transform(self, py_ast, program_config):
        grid_type = program_config.args_subconfig['grid_type']

        # Adding the nodes_parent parameter to the apply function
        py_ast.body[0].args.args.append(ast.Name(id="nodes_parent"))

        specializer = RecursiveSpecializer(py_ast, self.self_object, grid_type)
        specializer.typedef += PriorityQueueInterface.definitions
        specializer.func_defs += get_func_declarations().values()
        specializer.func_def_names += get_func_declarations().keys()
        specializer.includes = [StringTemplate("""\
            #include <stdlib.h>
            #include <math.h>
        """)]
        # TODO change NUM_DIMENSIONS and GRID_DIMENSIONS accordingly
        specializer.defines = [StringTemplate("""
            #define False 0
            #define True 1
            #define inf INFINITY

            #define NUM_DIMENSIONS 2
            #define GRID_DIMENSIONS {100,100}
        """)]
        specializer.local_include = [StringTemplate("""\
            #include "/Users/hugo/Dropbox/Estudos/Berkeley/SEJITS/Astar/scratch_dev/priority_queue_interface.h"
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

        param_types = [grid_type(), ctypes.c_int(), ctypes.c_int(),
                       grid_type()]
        c_file = specializer.get_c_file(param_types)

        return c_file

    def finalize(self, transform_result, program_config):
        project = Project(transform_result)
        grid_type = program_config.args_subconfig['grid_type']
        entry_type = ctypes.CFUNCTYPE(grid_type, grid_type, ctypes.c_int,
                                      ctypes.c_int, grid_type)
        return ConcreteSpecializedAStar("apply", project, entry_type)


class ConcreteSpecializedAStar(ConcreteSpecializedFunction):
    def __init__(self, entry_name, project_node, entry_typesig):
        self._c_function = self._compile(entry_name, project_node,
                                         entry_typesig)

    def __call__(self, grid, start, target, nodes_parent, self_object):
        return self._c_function(grid, start, target, nodes_parent)


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
        decomposed_coordinates.append(multiplied_term / dimension_multiplier +
                                      offset)
        dimension_multiplier = new_dimension_multiplier
        combined_coordinates -= multiplied_term

    return decomposed_coordinates[::-1]


class SpecializedGrid(GridAsArray):
    def __init__(self, grid):
        super(SpecializedGrid, self).__init__(grid)
        self._c_a_star = AStarSpecializer.from_function(self.a_star, "a_star")

    def specialized_a_star(self, start_id, target_id):
        start = combine_coordinates(start_id, self.grid_shape)
        target = combine_coordinates(target_id, self.grid_shape)
        nodes_parent = np.ones(self.grid_shape) * -1
        return self._c_a_star(self._grid, start, target, nodes_parent, self)