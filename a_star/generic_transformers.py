from ctree import get_ast
from ctree.visitors import NodeTransformer, NodeVisitor
import ast
from ctree.c.nodes import *
from ctree.cpp.nodes import CppDefine
from ctree.transformations import PyBasicConversions


class LambdaLifter(NodeTransformer):
    lambda_counter = 0

    def __init__(self, recursive_specializer=None):
        self.lifted_functions = []
        self.recursive_specializer = recursive_specializer

    def visit_Lambda(self, node):
        if isinstance(node, ast.Lambda):
            self.generic_visit(node)
            macro_name = "LAMBDA_" + str(self.lambda_counter)
            LambdaLifter.lambda_counter += 1
            node = PyBasicConversions().visit(node)
            node.name = macro_name
            macro = CppDefine(macro_name, node.params, node.defn[0].value)
            self.lifted_functions.append(macro)
            if self.recursive_specializer is not None:
                self.recursive_specializer.defines.append(macro)

            return SymbolRef(macro_name)
        else:
            return node


class ClassToStructureTransformer(NodeTransformer):
    """Intercepts class declaration and substitute by a `struct`
       the struct definitions are stored in the lift list
    """

    def __init__(self, class_name, defn, initial_values=None,
                 self_defined=False, constructor=None, pointer=False):
        self.class_name = class_name
        self.defn = defn
        self.self_defined = self_defined
        self.initial_values = initial_values
        self.constructor = constructor
        self.pointer = pointer
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
                struct_init = None
                struct = Struct(self.class_name, ptr=self.pointer)
                node = SymbolRef(node.targets[0].id, struct)
                if self.constructor is not None:
                    node = Assign(node, self.constructor)
                elif self.initial_values is not None:
                    values_ast = [Constant(i) for i in self.initial_values]
                    struct_init = ast.parse(ast.List(elts=values_ast))
                    struct_init = PyBasicConversions().visit(struct_init)
                    node = Assign(node, struct_init)
                self.lift.append(StructDef(self.class_name, self.defn,
                                           initializer=struct_init))
                return node

        self.generic_visit(node)
        return node

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Attribute):
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


class TypeTrackingTransformer(NodeTransformer):
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
                self.variable_types[attr_name] = node_type
        return node

    def visit_SymbolRef(self, node):
        if node.type is not None:
            node_type = node.type
            self.variable_types[node.name] = node_type
        return node

    def get_type_str(self, object_name):
        type_str = str(self.variable_types[object_name])
        if ' ' in type_str:
            type_str = type_str.split(' ')[-1]
        if type_str[-1] == "*":
            type_str = type_str[0:-1]

        return type_str


class MethodCallsTransformer(TypeTrackingTransformer):
    def __init__(self, recursive_specializer=None):
        super(MethodCallsTransformer, self).__init__()
        self.object_methods = []
        self_object = recursive_specializer.self_object
        if self_object is not None:
            for attr in dir(self_object):
                if hasattr(getattr(self_object, attr), '__call__'):
                    self.object_methods.append(attr)
        self.self_object = self_object
        self.recursive_specializer = recursive_specializer

    def visit_Call(self, node):
        self.generic_visit(node)
        func = node.func
        if isinstance(func, ast.Attribute):
            object_name = func.value.id
            if object_name in self.variable_types:
                func_name = str(self.get_type_str(object_name)) + "_" + \
                            func.attr
                node = FunctionCall(SymbolRef(func_name),
                                    [SymbolRef(object_name)] + node.args)
            elif object_name == 'self':
                if func.attr not in self.object_methods:
                    raise NotImplementedError("Method from self object was not"
                                              " found (%s)" % func.attr)
                func_name = "self_" + func.attr
                if func_name not in self.recursive_specializer.func_def_names:
                    func_def = get_ast(getattr(self.self_object, func.attr))
                    func_def = func_def.body[0]
                    func_def.name = func_name

                    func_def = self.recursive_specializer.visit(func_def)
                    self.recursive_specializer.func_defs.append(func_def)
                    self.recursive_specializer.func_def_names.append(func_name)

                node = FunctionCall(SymbolRef(func_name), node.args)
            else:
                func_name = object_name + "_" + func.attr
                node = FunctionCall(SymbolRef(func_name), node.args)

            if func_name not in self.recursive_specializer.func_def_names:
                params = []
                generic_att_cntr = 0
                for param in node.args:
                    if not hasattr(param, 'name'):
                        param = SymbolRef('attr' + str(generic_att_cntr))
                    params.append(param)
                    generic_att_cntr += 1

                self.recursive_specializer.func_defs.append(
                    FunctionDecl(None, func_name, params=params, defn=None))
                self.recursive_specializer.func_def_names.append(func_name)
        return node


class ReturnTypeFinder(TypeTrackingTransformer):
    def __init__(self, recursive_specializer=None):
        super(ReturnTypeFinder, self).__init__()
        self.recursive_specializer = recursive_specializer
        self.return_type = None

    def visit_Return(self, node):
        if node.value is None:
            return node

        attr_name = None

        if isinstance(node.value, FunctionCall):
            name = node.value.func.name
            funcs = self.recursive_specializer.func_defs
            ret = next((f.return_type for f in funcs if f.name == name), None)
            self.return_type = ret
        elif hasattr(node.value, 'id'):
            attr_name = node.value.id
        elif hasattr(node.value, 'name'):
            attr_name = node.value.name
        elif isinstance(node.value, ast.Attribute):
            attr_name = CompleteAttrName().get_complete_name(node)
        elif isinstance(node.value, ast.Num):
            self.return_type = get_ctype(node.value.n)
        else:
            self.return_type = node.value

        if attr_name is not None:
            self.return_type = self.variable_types[attr_name]

        return node

