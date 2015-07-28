from ctypes import Structure
from ctree.c.nodes import Statement, Block, SymbolRef
from ctree.types import register_type_codegenerators

register_type_codegenerators({
    Structure: lambda t: "struct " + type(t).__name__})


class CodegenableStruct(Structure):
    def codegen(self):
        return "struct " + type(self).__name__

    def __str__(self):
        return self.codegen()


class StructDef(Statement):
    def __init__(self, struct_type, initializer=None):
        super(StructDef, self).__init__()
        self.struct_type = struct_type
        fields = map(lambda (n, t): SymbolRef(n, t()), struct_type._fields_)
        self.block = Block(fields)
        self.struct_name = struct_type.__name__
        self.initializer = initializer

    def codegen(self, indent=0):
        return "struct " + self.struct_name + self.block.codegen(indent)
