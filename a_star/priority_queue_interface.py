import numpy as np
from ctree.c.nodes import *
from ctree.templates.nodes import StringTemplate
from a_star.generic_transformers import ClassToStructureTransformer

transform_priority_queue = ClassToStructureTransformer(
    "PriorityQueue",
    [
        SymbolRef("array", Struct("heap_element", ptr=True)),
        SymbolRef("size", np.uintc()),
        SymbolRef("max_size", np.uintc()),
    ],
    constructor=FunctionCall(SymbolRef("PriorityQueue_init")),
    pointer=True
)

transform_node_info = ClassToStructureTransformer(
    "NodeInfo",
    [
        SymbolRef('f', ctypes.c_double()),
        SymbolRef('g', ctypes.c_double()),
        SymbolRef('parent', ctypes.c_int()),
        SymbolRef('closed', ctypes.c_int())
    ],
    initial_values=[0, np.inf, -1, 0],
    self_defined=True
)


class PriorityQueueInterface(object):
    def __init__(self, grid_dimensions):
        self.heap_size = reduce(lambda x, y: x*y, grid_dimensions)/2
        self.grid_dimensions = grid_dimensions
        self.num_dimensions = len(grid_dimensions)

        self.definitions = [StructDef("heap_element", [
            SymbolRef("id", ctypes.c_int()),
            SymbolRef("priority", ctypes.c_long())
        ])]

        self.functions = {
            "PriorityQueue_init": FunctionDecl(
                return_type=Struct("PriorityQueue", ptr=True),
                name="PriorityQueue_init",
                params=None,
                defn=[Return(FunctionCall(SymbolRef("new_heap"),
                                          [Constant(self.heap_size)]))]
            ),
            "PriorityQueue_push": FunctionDecl(
                return_type=None,
                name="PriorityQueue_push",
                params=[SymbolRef("queue", Struct("PriorityQueue", ptr=True)),
                        SymbolRef("element", Struct("heap_element"))],
                defn=[StringTemplate("""heap_insert(queue, element);""")]
            ),
            "PriorityQueue_pop": FunctionDecl(
                return_type=Struct("heap_element"),
                name="PriorityQueue_pop",
                params=[SymbolRef("queue", Struct("PriorityQueue", ptr=True))],
                defn=[StringTemplate("""\
                    struct heap_element element = *find_heap_min(queue);
                    delete_heap_min(queue);
                    return element;""")
                      ]
            ),
            "Node": FunctionDecl(
                return_type=Struct("heap_element"),
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
                return_type=np.uintc(),
                name="len",
                params=[SymbolRef("queue", Struct("PriorityQueue", ptr=True))],
                defn=[StringTemplate("""return queue->size;""")]
            ),
        }
