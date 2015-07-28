from ctypes import POINTER
import numpy as np
from ctree.c.nodes import *
from ctree.templates.nodes import StringTemplate
from a_star.generic_transformers import ClassToStructureTransformer
from a_star.structure import StructDef, CodegenableStruct


# TODO may change heap_element to HeapElement
class heap_element(CodegenableStruct):
    _fields_ = [("id", ctypes.c_int), ("priority", ctypes.c_long)]


class PriorityQueue(CodegenableStruct):
    _fields_ = [
        ("array", POINTER(heap_element)),
        ("size", ctypes.c_uint),
        ("max_size", ctypes.c_uint)
    ]


class NodeInfo(CodegenableStruct):
    _fields_ = [
        ('f', ctypes.c_double),
        ('g', ctypes.c_double),
        ('parent', ctypes.c_int),
        ('closed', ctypes.c_int)
    ]


transform_priority_queue = ClassToStructureTransformer(
    PriorityQueue,
    constructor=FunctionCall(SymbolRef("PriorityQueue_init")),
    pointer=True
)

transform_node_info = ClassToStructureTransformer(
    NodeInfo, initial_values=[0.0, np.inf, -1, 0], self_defined=True)


class PriorityQueueInterface(object):
    def __init__(self, grid_dimensions):
        self.heap_size = reduce(lambda x, y: x*y, grid_dimensions)/2
        self.grid_dimensions = grid_dimensions
        self.num_dimensions = len(grid_dimensions)

        self.definitions = [StructDef(heap_element)]

        self.functions = {
            "PriorityQueue_init": FunctionDecl(
                return_type=POINTER(PriorityQueue)(),
                name="PriorityQueue_init",
                params=None,
                defn=[Return(FunctionCall(SymbolRef("new_heap"),
                                          [Constant(self.heap_size)]))]
            ),
            "PriorityQueue_push": FunctionDecl(
                return_type=None,
                name="PriorityQueue_push",
                params=[SymbolRef("queue", POINTER(PriorityQueue)()),
                        SymbolRef("element", heap_element())],
                defn=[StringTemplate("""heap_insert(queue, element);""")]
            ),
            "PriorityQueue_pop": FunctionDecl(
                return_type=heap_element(),
                name="PriorityQueue_pop",
                params=[SymbolRef("queue", POINTER(PriorityQueue)())],
                defn=[StringTemplate("""\
                    struct heap_element element = *find_heap_min(queue);
                    delete_heap_min(queue);
                    return element;""")
                      ]
            ),
            "Node": FunctionDecl(
                return_type=heap_element(),
                name="Node",
                params=[SymbolRef("id", ctypes.c_int()),
                        SymbolRef("priority", ctypes.c_long())],
                defn=[StringTemplate("""\
                    struct heap_element element;
                    element.id = id;
                    element.priority = priority;
                    return element;""")
                      ]
            ),
            "np_array": FunctionDecl(  # FIXME memory leak by returning pointer
                return_type=ctypes.POINTER(ctypes.c_int)(),
                name="np_array",
                params=[SymbolRef("coordinate", ctypes.c_int())],
                defn=[StringTemplate("""\
                    int dimension_multiplier = 1;
                    int grid_dimensions[] = $GRID_DIMENSIONS;
                    int* multiple_coordinate;
                    if ((multiple_coordinate = malloc($NUM_DIMENSIONS * sizeof(*multiple_coordinate))) == NULL) {
                        return NULL;
                    }
                    for (int i = $NUM_DIMENSIONS - 1; i >= 0; --i) {
                        int new_dimension_multiplier = dimension_multiplier * grid_dimensions[i];
                        int multiplied_term = coordinate % new_dimension_multiplier;
                        multiple_coordinate[i] = multiplied_term / dimension_multiplier;
                        dimension_multiplier = new_dimension_multiplier;
                        coordinate -= multiplied_term;
                    }
                    return multiple_coordinate;
                """,{'GRID_DIMENSIONS': Array(body=[Constant(i) for i in
                                                    self.grid_dimensions]),
                     'NUM_DIMENSIONS': Constant(self.num_dimensions)})]
            ),
            "len": FunctionDecl(
                return_type=ctypes.c_uint(),
                name="len",
                params=[SymbolRef("queue", POINTER(PriorityQueue)())],
                defn=[StringTemplate("""return queue->size;""")]
            ),
        }
