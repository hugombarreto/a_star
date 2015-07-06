import numpy as np
from ctree.c.nodes import *
from ctree.templates.nodes import StringTemplate
from AStarSpecializer.generic_transformers import ClassToStructureTransformer

substitute_priority_queue = ClassToStructureTransformer(
    "PriorityQueue",
    [
        SymbolRef("array", Struct("heap_element", ptr=True)),
        SymbolRef("size", np.uintc()),
        SymbolRef("max_size", np.uintc()),
    ],
    constructor=FunctionCall(SymbolRef("PriorityQueue_init")),
    pointer=True
)

substitute_node_info = ClassToStructureTransformer(
    "NodeInfo",
    [
        SymbolRef('f', ctypes.c_double()),
        SymbolRef('g', ctypes.c_double()),
        SymbolRef('parent', ctypes.c_int()),
        SymbolRef('closed', ctypes.c_int())
    ],
    initial_values=[0, np.inf, 0, 0],
    self_defined=True
)


class PriorityQueueInterface(object):
    definitions = [StructDef("heap_element", [
        SymbolRef("id", ctypes.c_int()),
        SymbolRef("priority", ctypes.c_long())
    ])]

    functions = {
        "PriorityQueue_init": FunctionDecl(
            return_type=Struct("PriorityQueue", ptr=True),
            name="PriorityQueue_init",
            params=None,
            # TODO properly determine the heap size
            defn=[Return(FunctionCall(SymbolRef("new_heap"), [Constant(10)]))]
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
        "len": FunctionDecl(
            return_type=np.uintc(),
            name="len",
            params=[SymbolRef("queue", Struct("PriorityQueue", ptr=True))],
            defn=[StringTemplate("""return queue->size;""")]
        ),
    }