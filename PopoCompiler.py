from llvmlite import ir
from llvmlite import binding
from llvmlite.ir import Argument, GlobalVariable
from llvmlite.ir.instructions import AllocaInstr
from PopoParser import *
from ctypes.util import find_library
from copy import deepcopy
import enum
import os
import platform
import argparse
import ctypes
import codecs
import uuid
import re
import subprocess


class Types(enum.Enum):
    INT = 1
    FLOAT = 2
    BOOL = 3
    STRING = 4
    VOID = 5
    CHAR = 6
    ENUM = 7
    STRUCT = 8


class Scope:
    def __init__(
        self,
        parent=None,
        is_loop_scope=False,
        loop_cond_block=None,
        loop_end_block=None,
    ):
        self.parent = parent
        self.symbols = {}
        self.type_parameters = set()

        self.current_type_bindings = {}
        self.is_loop_scope = is_loop_scope
        self.loop_cond_block = loop_cond_block
        self.loop_end_block = loop_end_block

    def define_type_parameter(self, name: str):

        if name in self.type_parameters:
            raise Exception(
                f"Type parameter '{name}' already declared in this immediate scope."
            )

        self.type_parameters.add(name)

    def is_type_parameter(self, name: str) -> bool:

        if name in self.type_parameters:
            return True
        if self.parent:
            return self.parent.is_type_parameter(name)
        return False

    def get_bound_type(self, type_param_name: str) -> ir.Type:

        if type_param_name in self.current_type_bindings:
            return self.current_type_bindings[type_param_name]
        if self.parent:

            return self.parent.get_bound_type(type_param_name)
        return None

    def define(self, name, value):
        """Defines a value symbol (variable pointer, function, etc.) in this scope."""
        if name in self.symbols:
            raise Exception(f"Symbol '{name}' already defined in this scope.")
        self.symbols[name] = value

    def resolve(self, name):
        """Resolves a value symbol by searching from current scope up to global."""
        if name in self.symbols:
            return self.symbols[name]
        if self.parent:
            return self.parent.resolve(name)

        return None

    def find_loop_scope(self):
        """Finds the innermost enclosing loop scope for break/continue."""
        if self.is_loop_scope:
            return self
        if self.parent:
            return self.parent.find_loop_scope()
        return None


class Macro:
    def __init__(self, name, args, body):
        self.name = name
        self.args = args
        self.body = body

    def expand(self, arg_values):
        if self.args is None:
            return self.body
        if len(arg_values) != len(self.args):
            raise Exception(
                f"Macro '{self.name}' expects {len(self.args)} args, got {len(arg_values)}."
            )
        result = self.body
        for param, val in zip(self.args, arg_values):
            result = re.sub(rf"\b{re.escape(param)}\b", val, result)
        return result


_macro_def_fn = re.compile(r"^\s*#cdef\s+([A-Za-z_]\w*)\s*\((.*?)\)\s+(.*)$")
_macro_def_obj = re.compile(r"^\s*#cdef\s+([A-Za-z_]\w*)\s+(.*)$")
_macro_call = re.compile(r"\b([A-Za-z_]\w*)\s*\(([^()]*)\)")


def preprocess_macros(text: str) -> str:
    macros = {}
    lines = text.splitlines()
    output = []

    for line in lines:
        fn_match = _macro_def_fn.match(line)
        obj_match = _macro_def_obj.match(line)

        if fn_match:
            name, args_str, body = fn_match.groups()
            args = [arg.strip() for arg in args_str.split(",") if arg.strip()]
            macros[name] = Macro(name, args, body)
        elif obj_match:
            name, body = obj_match.groups()
            macros[name] = Macro(name, None, body)
        else:

            def replacer(match):
                name = match.group(1)
                arg_str = match.group(2)
                if name in macros and macros[name].args is not None:
                    args = [a.strip() for a in arg_str.split(",")]
                    return macros[name].expand(args)
                return match.group(0)

            line = _macro_call.sub(replacer, line)

            for name, macro in macros.items():
                if macro.args is None:
                    line = re.sub(rf"\b{re.escape(name)}\b", macro.body, line)

            output.append(line)

    return "\n".join(output)


def _substitute(node, bindings):
    from copy import deepcopy

    if isinstance(node, str):
        return bindings.get(node, node)

    if isinstance(node, Literal):
        return deepcopy(node)

    if isinstance(node, list):
        return [_substitute(n, bindings) for n in node]

    if isinstance(node, str):
        return bindings.get(node, node)

    if isinstance(node, MacroCall):
        return MacroCall(
            _substitute(node.name, bindings),
            _substitute(node.args, bindings),
        )

    if isinstance(node, FunctionCall):
        return FunctionCall(
            _substitute(node.call_name, bindings),
            _substitute(node.params, bindings),
        )

    if isinstance(node, StructMethodCall):
        return StructMethodCall(
            _substitute(node.struct_name, bindings),
            _substitute(node.method_name, bindings),
            _substitute(node.params, bindings),
        )

    if isinstance(node, VariableDeclaration):
        return VariableDeclaration(
            node.is_mutable,
            _substitute(node.identifier, bindings),
            _substitute(node.type, bindings),
            _substitute(node.value, bindings),
        )

    if isinstance(node, Assignment):
        return Assignment(
            _substitute(node.identifier, bindings),
            node.operator,
            _substitute(node.value, bindings),
        )

    if isinstance(node, AdditiveOperator):
        return AdditiveOperator(
            node.operator,
            _substitute(node.left, bindings),
            _substitute(node.right, bindings),
        )

    if isinstance(node, MultiplicativeOperator):
        return MultiplicativeOperator(
            node.operator,
            _substitute(node.left, bindings),
            _substitute(node.right, bindings),
        )

    if isinstance(node, ComparisonOperator):
        return ComparisonOperator(
            node.operator,
            _substitute(node.left, bindings),
            _substitute(node.right, bindings),
        )

    if isinstance(node, LogicalOperator):
        return LogicalOperator(
            node.operator,
            _substitute(node.left, bindings),
            _substitute(node.right, bindings),
        )

    if isinstance(node, UnaryOperator):
        return UnaryOperator(node.operator, _substitute(node.operand, bindings))

    if isinstance(node, TypeConv):

        new_target = _substitute(node.target_type, bindings)
        new_expr = _substitute(node.expr, bindings)
        return TypeConv(new_target, new_expr)

    if isinstance(node, PostfixOperator):
        return PostfixOperator(node.operator, _substitute(node.operand, bindings))

    if isinstance(node, ReturnStatement):
        return ReturnStatement(_substitute(node.value, bindings))

    if isinstance(node, MemberAccess):
        return MemberAccess(
            _substitute(node.struct_name, bindings),
            _substitute(node.member_name, bindings),
        )

    if isinstance(node, StructInstantiation):
        return StructInstantiation(
            _substitute(node.struct_name, bindings),
            _substitute(node.field_assignments, bindings),
        )

    if isinstance(node, FieldAssignment):
        return FieldAssignment(
            _substitute(node.identifier, bindings), _substitute(node.value, bindings)
        )

    if isinstance(node, TypeOf):
        return TypeOf(_substitute(node.expr, bindings))

    if isinstance(node, IfStatement):
        return IfStatement(
            _substitute(node.condition, bindings),
            _substitute(node.body, bindings),
            _substitute(node.elifs, bindings),
            _substitute(node.else_body, bindings),
        )

    if isinstance(node, WhileLoop):
        return WhileLoop(
            _substitute(node.condition, bindings), _substitute(node.body, bindings)
        )

    if isinstance(node, ForLoop):
        return ForLoop(
            _substitute(node.init, bindings),
            _substitute(node.condition, bindings),
            _substitute(node.increment, bindings),
            _substitute(node.body, bindings),
        )

    if isinstance(node, ForeachLoop):
        return ForeachLoop(
            _substitute(node.identifier, bindings),
            _substitute(node.var_type, bindings),
            _substitute(node.iterable, bindings),
            _substitute(node.body, bindings),
        )

    if isinstance(node, ControlStatement):
        return deepcopy(node)

    if isinstance(node, Program):
        return Program(_substitute(node.statements, bindings))

    if isinstance(node, ModuleAccess):
        return ModuleAccess(
            _substitute(node.alias, bindings), _substitute(node.name, bindings)
        )

    if isinstance(node, QualifiedAccess):
        return QualifiedAccess(
            _substitute(node.left, bindings), _substitute(node.name, bindings)
        )

    if isinstance(node, EnumAccess):
        return EnumAccess(
            _substitute(node.enum_name, bindings), _substitute(node.value, bindings)
        )

    if isinstance(node, EnumDeclaration):
        return deepcopy(node)

    if isinstance(node, StructMember):
        return StructMember(
            _substitute(node.identifier, bindings), _substitute(node.var_type, bindings)
        )

    if isinstance(node, Parameter):
        return Parameter(
            _substitute(node.identifier, bindings), _substitute(node.var_type, bindings)
        )

    if isinstance(node, SpecialDeclaration):
        return deepcopy(node)

    if isinstance(node, MacroDeclaration):
        return deepcopy(node)

    if isinstance(node, FunctionDeclaration):
        return deepcopy(node)

    if isinstance(node, StructDeclaration):
        return deepcopy(node)

    if isinstance(node, Comment):
        return deepcopy(node)

    if (
        isinstance(node, ImportModule)
        or isinstance(node, ImportC)
        or isinstance(node, Extern)
    ):
        return deepcopy(node)

    raise Exception(f"Substitution not implemented for node: {node}")


def parse_code(code):

    code = preprocess_macros(code)

    lexer.input(code)
    ast = parser.parse(code)
    return ast


def parse_file(path):
    with open(path, "r") as f:
        code = f.read()
    return parse_code(code)


def resolve_c_library(name):
    """
    Map a bare import like "stdio" to the right runtime libc name for this platform.
    """

    if os.path.isabs(name) or name.endswith((".so", ".dll", ".dylib")):
        return name

    plat = platform.system().lower()
    if name.lower() in ("c", "stdio"):
        if plat == "windows":
            return find_library("msvcrt") or "msvcrt.dll"
        elif plat == "darwin":
            return find_library("c") or "libc.dylib"
        else:
            return find_library("c") or "libc.so.6"

    return find_library(name) or name


class Compiler:
    def __init__(self, opt=None, codemodel=None, is_jit=False):
        binding.initialize()
        binding.initialize_native_target()
        binding.initialize_native_asmprinter()

        self.module = ir.Module(name="popo_module")
        try:
            self.target_triple = binding.get_default_triple()
            self.target = binding.Target.from_triple(self.target_triple)
            if is_jit:
                self.target_machine = self.target.create_target_machine()
            else:
                self.target_machine = self.target.create_target_machine(
                    reloc="static",
                    codemodel="default" if not codemodel else codemodel,
                    opt=0 if not opt else opt,
                )

            self.data_layout_obj = self.target_machine.target_data

            self.module.triple = self.target_triple
            self.module.data_layout = str(self.data_layout_obj)

        except RuntimeError as e:
            print(f"Fatal Error: Failed to initialize LLVM target information: {e}")
            print(
                "Please ensure LLVM is correctly installed and configured for your system."
            )

            self.target_machine = None
            self.data_layout_obj = None
            raise
        self.builder = None
        self.function = None

        self.global_scope = Scope(parent=None)
        self.current_scope = self.global_scope

        self.block_count = 0

        printf_ty = ir.FunctionType(
            ir.IntType(32), [ir.IntType(8).as_pointer()], var_arg=True
        )
        printf = ir.Function(self.module, printf_ty, name="printf")
        self.global_scope.define("printf", printf)

        self.struct_types = {}
        self.struct_field_indices = {}
        self.struct_methods = {}
        self.enum_types = {}
        self.enum_members = {}
        self.global_strings = {}
        self.imported_libs = []
        self.loaded_modules = {}
        self.module_aliases = {}
        self.module_enum_types = {}
        self.module_enum_members = {}
        self.module_struct_types = {}
        self.module_struct_fields = {}

        self.generic_function_templates = {}
        self.instantiated_functions = {}

        self.macros = {}
        self.type_codes = {
            "int": 0,
            "float": 1,
            "bool": 2,
            "string": 3,
            "void": 4,
            "char": 5,
        }
        self._next_type_code = 6
        self.identified_types = {"char": ir.IntType(8)}

        self.main_function = None

    def enter_scope(
        self, is_loop_scope=False, loop_cond_block=None, loop_end_block=None
    ):

        self.current_scope = Scope(
            self.current_scope, is_loop_scope, loop_cond_block, loop_end_block
        )

    def exit_scope(self):

        if self.current_scope.parent is None:

            print(
                "Warning: Attempting to exit global scope or program compilation finished."
            )

            if self.current_scope is not self.global_scope:
                self.current_scope = self.global_scope
            return
        self.current_scope = self.current_scope.parent

    def _get_scope_depth(self):
        depth = 0
        s = self.current_scope
        while s:
            depth += 1
            s = s.parent
        return depth

    def create_function(self, name, ret_type, arg_types):
        conv_ret_type = self.convert_type(ret_type)
        arg_types = [self.convert_type(arg) for arg in arg_types]
        func_type = ir.FunctionType(conv_ret_type, arg_types)

        self.function = ir.Function(self.module, func_type, name)
        self.current_block = self.function.append_basic_block(name + "_entry")
        self.builder = ir.IRBuilder(self.current_block)
        self.current_scope.define(arg.name, [arg for arg in self.function.args])
        for i, arg in enumerate(self.function.args):
            arg.name = f"arg{i}"
            self.current_scope.define(arg.name, arg)
        return self.function

    def compile_enum_access(self, ast):
        """
        ast.left  is the enum name (a string, e.g. "Status"),
        ast.name  is the variant (e.g. "Ok").
        """
        enum_name = ast.left if isinstance(ast.left, str) else None
        if enum_name is None:
            raise Exception(f"Invalid enum qualifier: {ast}")

        if enum_name not in self.enum_members:
            raise Exception(f"Enum '{enum_name}' is not defined.")

        members = self.enum_members[enum_name]
        if ast.name not in members:
            raise Exception(f"Enum '{enum_name}' has no member '{ast.name}'.")

        return members[ast.name]

    def compile_module_access(self, ast: ModuleAccess):

        if isinstance(ast.alias, ModuleAccess):
            inner = ast.alias
            enum_name = inner.name
            variant = ast.name

            if enum_name not in self.enum_members:
                raise Exception(f"Enum '{enum_name}' not defined.")
            members = self.enum_members[enum_name]
            if variant not in members:
                raise Exception(f"Enum '{enum_name}' has no member '{variant}'.")
            return members[variant]

        alias = ast.alias
        name = ast.name

        if alias not in self.module_aliases:
            raise Exception(f"Module '{alias}' not imported.")
        path = self.module_aliases[alias]
        namespace = self.loaded_modules[path]

        if name not in namespace:
            raise Exception(f"Module '{alias}' has no symbol '{name}'.")
        return namespace[name]

    def compile_struct(self, ast: StructDeclaration):
        name = ast.name
        if name in self.struct_types:
            raise Exception(f"Name Error: struct '{name}' already declared.")

        struct_ty = ir.global_context.get_identified_type(name)

        llvm_field_tys = [self.convert_type(member.var_type) for member in ast.members]
        struct_ty.set_body(*llvm_field_tys)

        self.struct_types[name] = struct_ty
        self.struct_field_indices[name] = {
            member.identifier: idx for idx, member in enumerate(ast.members)
        }

        self.struct_methods[name] = ast.methods

        if name not in self.type_codes:
            self.type_codes[name] = self._next_type_code
            self._next_type_code += 1

        for method in ast.methods:
            self._compile_struct_method(name, struct_ty, method)

    def _compile_struct_method(
        self,
        struct_name: str,
        struct_llvm_type: ir.IdentifiedStructType,
        method_ast: FunctionDeclaration,
    ):

        mangled_fn_name = f"{struct_name}_{method_ast.name}"
        if method_ast.is_static:
            mangled_fn_name = f"{struct_name}_static_{method_ast.name}"

        llvm_ret_type = self.convert_type(method_ast.return_type)

        llvm_param_types = []
        ast_params_for_llvm_mapping = []

        if not method_ast.is_static:

            llvm_param_types.append(struct_llvm_type.as_pointer())

            if not method_ast.params:
                raise Exception(
                    f"Instance method '{method_ast.name}' in struct '{struct_name}' must declare 'self' as its first parameter (e.g., 'self: <{struct_name}>')."
                )

            ast_params_for_llvm_mapping = method_ast.params
            for param_node in method_ast.params[1:]:
                llvm_param_types.append(self.convert_type(param_node.var_type))
        else:

            ast_params_for_llvm_mapping = method_ast.params
            for param_node in method_ast.params:
                llvm_param_types.append(self.convert_type(param_node.var_type))

        llvm_fn_type = ir.FunctionType(llvm_ret_type, llvm_param_types)
        method_llvm_func = ir.Function(self.module, llvm_fn_type, name=mangled_fn_name)

        prev_function = self.function
        prev_builder = self.builder

        self.function = method_llvm_func

        self.enter_scope()

        entry_block = method_llvm_func.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(entry_block)

        if not method_ast.is_static:

            self_param_ast = ast_params_for_llvm_mapping[0]
            self_llvm_arg = method_llvm_func.args[0]
            self_llvm_arg.name = self_param_ast.identifier

            self_alloca = self.builder.alloca(
                self_llvm_arg.type, name=f"{self_param_ast.identifier}_ptr"
            )
            self.builder.store(self_llvm_arg, self_alloca)
            self.current_scope.define(self_param_ast.identifier, self_alloca)

            for i, param_ast_node in enumerate(ast_params_for_llvm_mapping[1:]):
                param_llvm_arg = method_llvm_func.args[i + 1]
                param_name = param_ast_node.identifier
                param_llvm_arg.name = param_name

                param_alloca = self.builder.alloca(
                    param_llvm_arg.type, name=f"{param_name}_ptr"
                )
                self.builder.store(param_llvm_arg, param_alloca)
                self.current_scope.define(param_name, param_alloca)
        else:

            for i, param_ast_node in enumerate(ast_params_for_llvm_mapping):
                param_llvm_arg = method_llvm_func.args[i]
                param_name = param_ast_node.identifier
                param_llvm_arg.name = param_name

                param_alloca = self.builder.alloca(
                    param_llvm_arg.type, name=f"{param_name}_ptr"
                )
                self.builder.store(param_llvm_arg, param_alloca)
                self.current_scope.define(param_name, param_alloca)

        if method_ast.body:
            for stmt_node in method_ast.body:
                self.compile(stmt_node)

        if not self.builder.block.is_terminated:
            if isinstance(llvm_ret_type, ir.VoidType):
                self.builder.ret_void()
            elif (
                not method_ast.is_static
                and llvm_ret_type == struct_llvm_type.as_pointer()
            ):

                self_llvm_arg_for_return = method_llvm_func.args[0]
                self.builder.ret(self_llvm_arg_for_return)

        self.exit_scope()

        self.function = prev_function
        self.builder = prev_builder

    def compile_struct_instantiation(self, node: StructInstantiation):

        if isinstance(node.struct_name, ModuleAccess):

            alias, struct_name = node.struct_name.alias, node.struct_name.name
            if alias not in self.module_aliases:
                raise Exception(f"Module '{alias}' not imported.")
            path = self.module_aliases[alias]

            if struct_name not in self.module_struct_types.get(path, {}):
                raise Exception(f"Module '{alias}' has no struct '{struct_name}'.")
            struct_ty = self.module_struct_types[path][struct_name]
            field_indices = self.module_struct_fields[path][struct_name]

        else:

            struct_name = node.struct_name
            if struct_name not in self.struct_types:
                raise Exception(f"Struct '{struct_name}' not defined.")
            struct_ty = self.struct_types[struct_name]
            field_indices = self.struct_field_indices[struct_name]

        struct_ptr = self.builder.alloca(struct_ty, name=f"{struct_name}_inst")

        if node.field_assignments == [None]:

            for idx in range(len(struct_ty.elements)):
                zero = ir.Constant(struct_ty.elements[idx], 0)
                self.builder.store(zero, struct_ptr)

            return struct_ptr
        for assignment in node.field_assignments:
            fld = assignment.identifier
            if fld not in field_indices:
                raise Exception(f"Struct '{struct_name}' has no field '{fld}'.")
            idx = field_indices[fld]

            zero = ir.Constant(ir.IntType(32), 0)
            fld_ptr = self.builder.gep(
                struct_ptr, [zero, ir.Constant(ir.IntType(32), idx)], inbounds=True
            )

            val = self.compile(assignment.value)

            expected = struct_ty.elements[idx]

            if isinstance(expected, ir.IntType) and expected.width == 8:
                if isinstance(
                    val.type, ir.PointerType
                ) and val.type.pointee == ir.IntType(8):
                    val = self.builder.load(val, name=f"{fld}_byte")

            if val.type != expected:
                raise Exception(
                    f"Type mismatch storing to {struct_name}.{fld}: "
                    f"expected {expected}, got {val.type}"
                )

            self.builder.store(val, fld_ptr)

        return struct_ptr

    def assign_struct_field(self, struct_ptr, struct_name, field_name, value):
        """
        Emit code to store `value` (an LLVM Value) into `struct_ptr->field_name`.
        """

        indices = self.struct_field_indices.get(struct_name)
        if indices is None:
            raise Exception(f"Struct '{struct_name}' is not defined.")
        if field_name not in indices:
            raise Exception(f"Struct '{struct_name}' has no field '{field_name}'.")

        idx = indices[field_name]

        zero = ir.Constant(ir.IntType(32), 0)
        field_ptr = self.builder.gep(
            struct_ptr, [zero, ir.Constant(ir.IntType(32), idx)], inbounds=True
        )

        self.builder.store(value, field_ptr)

    def compile_struct_method_call(self, ast):

        struct_ptr = self.get_variable(ast.struct_name)
        if struct_ptr is None:
            raise Exception(f"Variable '{ast.struct_name}' not declared.")

        struct_type_name = struct_ptr.type.pointee.name

        method_full = f"{struct_type_name}_{ast.method_name}"

        try:
            method_func = self.module.get_global(method_full)
        except KeyError:
            raise Exception(
                f"Method '{ast.method_name}' not found on struct '{struct_type_name}'"
            )

        fn_ty = method_func.function_type
        num_fixed = len(fn_ty.args)

        args = [struct_ptr]
        for i, param_node in enumerate(ast.params, start=1):

            if isinstance(param_node, str):
                val = self.get_variable(param_node)
            else:
                val = self.compile(param_node)

            if i < num_fixed:
                expected = fn_ty.args[i]
                if (
                    isinstance(expected, ir.IntType)
                    and expected.width == 8
                    and isinstance(val.type, ir.PointerType)
                    and val.type.pointee == ir.IntType(8)
                ):
                    val = self.builder.load(val)
                if val.type != expected:
                    raise Exception(
                        f"Method '{method_full}' arg {i} type mismatch: "
                        f"expected {expected}, got {val.type}"
                    )
            args.append(val)

        return self.builder.call(
            method_func, args, name=f"{ast.struct_name}_{ast.method_name}_call"
        )

    def compile_member_access(self, node: MemberAccess):

        struct_ptr = self.get_variable(node.struct_name)
        if struct_ptr is None:
            raise Exception(f"Variable '{node.struct_name}' not found.")

        struct_ty = struct_ptr.type.pointee
        struct_name = struct_ty.name

        if struct_name not in self.struct_field_indices:
            raise Exception(f"Struct type '{struct_name}' not defined.")
        field_indices = self.struct_field_indices[struct_name]

        member_name = node.member_name
        if member_name not in field_indices:
            raise Exception(f"Struct '{struct_name}' has no member '{member_name}'.")

        idx = field_indices[member_name]

        zero = ir.Constant(ir.IntType(32), 0)
        member_ptr = self.builder.gep(
            struct_ptr, [zero, ir.Constant(ir.IntType(32), idx)], inbounds=True
        )

        return self.builder.load(member_ptr, name=f"{node.struct_name}_{member_name}")

    def _instantiate_and_compile_generic(
        self,
        func_name_str,
        generic_func_ast: FunctionDeclaration,
        inferred_bindings: dict,
        concrete_types_tuple: tuple,
    ):
        """
        Helper to monomorphize a generic function.
        """

        type_names_for_mangling = "_".join(
            str(t)
            .replace(" ", "_")
            .replace("*", "p")
            .replace("[", "arr")
            .replace("]", "")
            .replace("%", "")
            for t in concrete_types_tuple
        )

        type_names_for_mangling = re.sub(r"[^a-zA-Z0-9_p]", "", type_names_for_mangling)

        mangled_name = f"{generic_func_ast.name}__{type_names_for_mangling}"
        if len(mangled_name) > 200:
            mangled_name = f"{generic_func_ast.name}__{uuid.uuid4().hex[:8]}"

        try:
            return self.module.get_global(mangled_name)
        except KeyError:
            pass

        original_scope_bindings = self.current_scope.current_type_bindings
        self.current_scope.current_type_bindings = inferred_bindings

        original_ast_name = generic_func_ast.name
        generic_func_ast.name = mangled_name

        instantiated_llvm_func = self.compile(generic_func_ast)

        generic_func_ast.name = original_ast_name

        self.current_scope.current_type_bindings = original_scope_bindings

        if not isinstance(instantiated_llvm_func, ir.Function):
            raise Exception(
                f"Instantiation of '{original_ast_name}' to '{mangled_name}' did not result in an LLVM function."
            )

        self.instantiated_functions[
            (func_name_str, concrete_types_tuple)
        ] = instantiated_llvm_func
        return instantiated_llvm_func

    def convert_type(self, type_name_or_node):
        if isinstance(type_name_or_node, ir.Type):
            return type_name_or_node

        type_name_str_for_param_check = None
        if isinstance(type_name_or_node, str):
            type_name_str_for_param_check = type_name_or_node
        elif isinstance(type_name_or_node, TypeParameterNode):
            type_name_str_for_param_check = type_name_or_node.name

        if type_name_str_for_param_check:

            if hasattr(self.current_scope, "get_bound_type"):
                bound_llvm_type = self.current_scope.get_bound_type(
                    type_name_str_for_param_check
                )
                if bound_llvm_type:

                    return bound_llvm_type

            if hasattr(
                self.current_scope, "is_type_parameter"
            ) and self.current_scope.is_type_parameter(type_name_str_for_param_check):
                raise Exception(
                    f"Internal Error: convert_type called for unbound type parameter '{type_name_str_for_param_check}'. "
                    "Generic templates should not be fully type-converted until instantiation."
                )

        if isinstance(type_name_or_node, ArrayTypeNode):
            ast_node = type_name_or_node
            llvm_element_type = self.convert_type(ast_node.element_type)

            if not isinstance(llvm_element_type, ir.Type):
                raise Exception(
                    f"Element type '{ast_node.element_type}' of array did not resolve to LLVM type. Got: {llvm_element_type}"
                )

            if ast_node.size_expr:
                if not isinstance(ast_node.size_expr, Literal) or not isinstance(
                    ast_node.size_expr.value, int
                ):
                    raise Exception(
                        f"Array size in type annotation must be a constant integer literal, got {ast_node.size_expr}"
                    )
                size = ast_node.size_expr.value
                if size < 0:
                    raise Exception(f"Array size must be non-negative, got {size}")
                return ir.ArrayType(llvm_element_type, size)
            else:

                raise Exception(
                    f"Array type annotation '{ast_node}' within a larger type (or used directly without an initializer context) "
                    "must have an explicit size. Unsized arrays like <[int]> are only allowed as the outermost type "
                    "of a variable being initialized, e.g., 'let x <[int]> = [1,2,3];'."
                )

        if isinstance(type_name_or_node, PointerTypeNode):
            ast_node = type_name_or_node
            llvm_pointee_type = self.convert_type(ast_node.pointee_type)
            return llvm_pointee_type.as_pointer()

        type_name_str = None
        if isinstance(type_name_or_node, str):
            type_name_str = type_name_or_node
        elif isinstance(type_name_or_node, TypeAnnotation):
            base, bits = type_name_or_node.base, type_name_or_node.bits
            if base == "int":
                if bits in (8, 16, 32, 64, 128):
                    return ir.IntType(bits)
                else:
                    raise Exception(f"Unsupported integer width: {bits}")
            if base == "float":
                if bits == 32:
                    return ir.FloatType()
                if bits == 64:
                    return ir.DoubleType()
                else:
                    raise Exception(f"Unsupported float width: {bits}")

            raise Exception(f"Unknown parameterized type: {base}({bits})")

        elif isinstance(type_name_or_node, ModuleAccess):
            alias, name = type_name_or_node.alias, type_name_or_node.name
            if alias not in self.module_aliases:
                raise Exception(f"Module '{alias}' not imported.")
            path = self.module_aliases[alias]
            m_enums = self.module_enum_types.get(path, {})
            if name in m_enums:
                return m_enums[name]
            m_structs = self.module_struct_types.get(path, {})
            if name in m_structs:
                return m_structs[name].as_pointer()
            raise Exception(f"Module '{alias}' has no type '{name}'.")
        elif isinstance(type_name_or_node, TypeAnnotation):
            base, bits = type_name_or_node.base, type_name_or_node.bits
            if base == "int":
                if bits in (8, 16, 32, 64):
                    return ir.IntType(bits)
                else:
                    raise Exception(f"Unsupported integer width: {bits}")
            if base == "float":
                if bits == 32:
                    return ir.FloatType()
                if bits == 64:
                    return ir.DoubleType()
                else:
                    raise Exception(f"Unsupported float width: {bits}")
            raise Exception(f"Unknown base type '{base}' in TypeAnnotation")
        if type_name_str is None:
            raise Exception(
                f"Unknown type representation passed to convert_type: {type_name_or_node} (type: {type(type_name_or_node)})"
            )

        if type_name_str == "int":
            return ir.IntType(32)
        if type_name_str == "float":
            return ir.FloatType()
        if type_name_str == "double":
            return ir.DoubleType()
        if type_name_str == "bool":
            return ir.IntType(1)
        if type_name_str == "string":
            return ir.IntType(8).as_pointer()
        if type_name_str == "noret":
            return ir.VoidType()
        if type_name_str == "auto":
            raise Exception("'auto' type must be inferred, not passed to convert_type.")
        if type_name_str in self.identified_types:
            return self.identified_types[type_name_str]
        if type_name_str in self.enum_types:
            return self.enum_types[type_name_str]
        if type_name_str in self.struct_types:
            return self.struct_types[type_name_str]

        raise Exception(f"Unknown concrete type name: '{type_name_str}'")

    def create_block(self, name):
        block = self.function.append_basic_block(name)
        self.builder.position_at_end(block)
        return block

    def create_variable_mut(
        self,
        name: str,
        var_type_ast_or_llvm: object,
        initial_value_llvm: ir.Value = None,
    ):
        """
        Helper to declare and optionally initialize a mutable local variable.
        This assumes we are already inside a function context (self.function and self.builder are set).
        All allocas will be placed in the function's entry block by self.builder.alloca().

        Args:
            name (str): The name of the variable.
            var_type_ast_or_llvm (object): AST node for the type, or a pre-converted LLVM type.
            initial_value_llvm (ir.Value, optional): Pre-compiled LLVM value for initialization.
        """
        if self.function is None or self.builder is None:
            raise Exception(
                "create_variable_mut can only be called within a function compilation context."
            )

        llvm_type = None
        if isinstance(var_type_ast_or_llvm, ir.Type):
            llvm_type = var_type_ast_or_llvm
        else:
            llvm_type = self.convert_type(var_type_ast_or_llvm)

        if llvm_type is None:
            raise Exception(
                f"Could not determine LLVM type for variable '{name}' from '{var_type_ast_or_llvm}'."
            )

        var_ptr = self.builder.alloca(llvm_type, name=f"{name}_ptr")

        try:
            self.current_scope.define(name, var_ptr)
        except Exception as e:

            raise Exception(f"Error defining variable '{name}' in scope: {e}") from e

        if initial_value_llvm is not None:

            if initial_value_llvm.type != llvm_type:
                if isinstance(llvm_type, ir.FloatType) and isinstance(
                    initial_value_llvm.type, ir.IntType
                ):
                    coerced_value = self.builder.sitofp(
                        initial_value_llvm, llvm_type, name=f"{name}_init_conv"
                    )
                    self.builder.store(coerced_value, var_ptr)

                else:
                    raise Exception(
                        f"Type mismatch for variable '{name}' initializer. Expected {llvm_type}, got {initial_val_llvm.type}."
                    )
            else:
                self.builder.store(initial_val_llvm, var_ptr)

        return var_ptr

    def create_variable_immut(
        self,
        name: str,
        var_type_ast_or_llvm: object,
        initial_value_ast_or_llvm_const: object,
    ) -> ir.GlobalVariable:
        """
        Declare an immutable (constant) global variable in the module.
        The initial_value_ast_or_llvm_const MUST evaluate to a compile-time constant.
        """

        llvm_type = None
        if isinstance(var_type_ast_or_llvm, ir.Type):
            llvm_type = var_type_ast_or_llvm
        else:
            llvm_type = self.convert_type(var_type_ast_or_llvm)

        if llvm_type is None:
            raise Exception(
                f"Could not determine LLVM type for global constant '{name}'."
            )

        llvm_initializer_const = None
        if initial_value_ast_or_llvm_const is None:

            raise Exception(f"Global immutable variable '{name}' must be initialized.")

        if isinstance(initial_value_ast_or_llvm_const, ir.Constant):
            llvm_initializer_const = initial_value_ast_or_llvm_const
        else:

            current_builder = self.builder
            self.builder = None

            compiled_init_val = self.compile(initial_value_ast_or_llvm_const)

            self.builder = current_builder

            if not isinstance(compiled_init_val, ir.Constant):
                raise Exception(
                    f"Initializer for global constant '{name}' must be a compile-time constant expression. Got {type(compiled_init_val)}."
                )
            llvm_initializer_const = compiled_init_val

        if llvm_initializer_const.type != llvm_type:

            if isinstance(llvm_type, ir.IntType) and isinstance(
                llvm_initializer_const.type, ir.IntType
            ):

                raise Exception(
                    f"Type mismatch for global constant '{name}'. Expected {llvm_type}, "
                    f"got initializer of type {llvm_initializer_const.type}."
                )
            elif isinstance(llvm_type, ir.PointerType) and isinstance(
                llvm_initializer_const.type, ir.PointerType
            ):
                if llvm_type != llvm_initializer_const.type:
                    raise Exception(
                        f"Pointer type mismatch for global constant '{name}'. Expected {llvm_type}, got {llvm_initializer_const.type}."
                    )
            else:
                raise Exception(
                    f"Type mismatch for global constant '{name}'. Expected {llvm_type}, "
                    f"got initializer of type {llvm_initializer_const.type}."
                )

        try:
            gvar = self.module.get_global(name)
            if gvar.type.pointee != llvm_type:
                raise Exception(
                    f"Global variable '{name}' already exists with a different type."
                )
            if not gvar.global_constant:
                raise Exception(
                    f"Global variable '{name}' already exists but is not constant."
                )

        except KeyError:
            gvar = ir.GlobalVariable(self.module, llvm_type, name=name)

        gvar.linkage = "internal"
        gvar.global_constant = True
        gvar.initializer = llvm_initializer_const

        existing_in_scope = self.global_scope.resolve(name)
        if existing_in_scope is None:
            self.global_scope.define(name, gvar)
        elif existing_in_scope is not gvar:

            raise Exception(
                f"Symbol '{name}' already defined in global scope with a different value."
            )

        return gvar

    def guess_type(self, value):
        if isinstance(value, int):

            if -(2**31) <= value < 2**31:
                return ir.IntType(32)
            elif -(2**63) <= value < 2**63:
                return ir.IntType(64)

            elif -(2**127) <= value < 2**127:
                return ir.IntType(128)
            else:

                raise Exception(
                    f"Integer literal {value} is too large for supported integer types (e.g., i128)."
                )

        elif isinstance(value, float):
            return ir.FloatType()

        elif isinstance(value, str):

            return ir.IntType(8).as_pointer()

        elif isinstance(value, bool):
            return ir.IntType(1)

        else:
            raise Exception(
                f"Cannot guess LLVM type for Python value '{value}' of type {type(value)}."
            )

    def create_global_string(self, val: str) -> ir.Value:

        if val in self.global_strings:
            return self.global_strings[val]

        bytes_ = bytearray(val.encode("utf8")) + b"\00"
        str_ty = ir.ArrayType(ir.IntType(8), len(bytes_))

        uniq = uuid.uuid4().hex[:8]
        name = f".str_{uniq}"

        gvar = ir.GlobalVariable(self.module, str_ty, name=name)
        gvar.linkage = "internal"
        gvar.global_constant = True
        gvar.initializer = ir.Constant(str_ty, bytes_)

        zero = ir.Constant(ir.IntType(32), 0)
        str_ptr = self.builder.gep(gvar, [zero, zero], inbounds=True)

        self.global_strings[val] = str_ptr
        return str_ptr

    def set_variable(self, name: str, value_llvm: ir.Value):
        """
        Stores a new LLVM value into an existing variable's memory location.
        Assumes 'name' refers to a variable (AllocaInstr or mutable GlobalVariable).
        """
        if self.builder is None:
            raise Exception(
                "set_variable called outside of a function/block context where a builder is active."
            )

        var_ptr = self.current_scope.resolve(name)

        if var_ptr is None:
            raise Exception(
                f"Variable '{name}' not declared before assignment in set_variable."
            )

        if not isinstance(var_ptr, (ir.AllocaInstr, ir.GlobalVariable)):
            raise Exception(
                f"Symbol '{name}' is not a variable pointer (AllocaInstr or GlobalVariable), cannot set its value. Got: {type(var_ptr)}"
            )

        if isinstance(var_ptr, ir.GlobalVariable) and var_ptr.global_constant:
            raise Exception(f"Cannot assign to global constant '{name}'.")

        expected_value_type = var_ptr.type.pointee
        actual_value_type = value_llvm.type

        coerced_value_llvm = value_llvm
        if actual_value_type != expected_value_type:

            if isinstance(expected_value_type, ir.FloatType) and isinstance(
                actual_value_type, ir.IntType
            ):
                coerced_value_llvm = self.builder.sitofp(
                    value_llvm, expected_value_type, f"{name}_set_conv"
                )
            elif isinstance(expected_value_type, ir.PointerType) and isinstance(
                actual_value_type, ir.PointerType
            ):
                if expected_value_type != actual_value_type:
                    print(
                        f"Warning (set_variable): Pointer type mismatch for '{name}'. Expected {expected_value_type}, got {actual_value_type}. Bitcasting."
                    )
                    coerced_value_llvm = self.builder.bitcast(
                        value_llvm, expected_value_type
                    )
            else:
                raise Exception(
                    f"Type mismatch in set_variable for '{name}'. Expected value of type "
                    f"{expected_value_type}, got {actual_value_type}."
                )

        self.builder.store(coerced_value_llvm, var_ptr)

        return var_ptr

    def get_variable(self, name_or_node_ast: object):
        """
        Retrieves the LLVM value of a symbol or materializes a literal.
        If 'name_or_node_ast' is a string (identifier):
            - Resolves the identifier using the current scope.
            - If it's a pointer to memory (AllocaInstr, GlobalVariable), loads the value.
            - If it's a direct LLVM value (Function, Constant), returns it.
        If 'name_or_node_ast' is a Literal AST node, creates and returns an ir.Constant.
        Otherwise, it compiles the AST node.
        """

        if isinstance(name_or_node_ast, str):
            identifier_name = name_or_node_ast
            resolved_symbol = self.current_scope.resolve(identifier_name)

            if resolved_symbol is None:

                try:
                    resolved_symbol = self.module.get_global(identifier_name)
                except KeyError:
                    raise Exception(
                        f"Variable or symbol '{identifier_name}' not declared."
                    )

            if isinstance(resolved_symbol, (ir.AllocaInstr, ir.GlobalVariable)):

                if self.builder is None:
                    raise Exception(
                        "get_variable trying to load from memory, but no builder is active (not in function context?)."
                    )
                return self.builder.load(resolved_symbol, name=identifier_name + "_val")
            elif isinstance(resolved_symbol, (ir.Function, ir.Constant)):

                return resolved_symbol
            elif isinstance(resolved_symbol, ir.Argument):

                return resolved_symbol
            else:
                raise Exception(
                    f"Resolved symbol '{identifier_name}' is of an unexpected type: {type(resolved_symbol)}"
                )

        elif isinstance(name_or_node_ast, Literal):
            py_value = name_or_node_ast.value

            if isinstance(py_value, str):

                unescaped_str = codecs.decode(py_value, "unicode_escape")

                return self.create_global_string(unescaped_str)
            else:

                llvm_type = self.guess_type(py_value)
                try:
                    return ir.Constant(llvm_type, py_value)
                except OverflowError:
                    raise Exception(
                        f"Cannot create LLVM constant for literal '{py_value}' of type {llvm_type}."
                    )

        else:
            if (
                self.builder is None
                and self.function is None
                and hasattr(ast, "lineno")
            ):
                print(
                    f"Warning: Compiling complex AST node {type(name_or_node_ast)} outside function context. Ensure it produces a constant or global."
                )
            return self.compile(name_or_node_ast)

    def create_if(self, condition):

        true_block = self.create_block("if_true")
        false_block = self.create_block("if_false")
        end_block = self.create_block("if_end")

        self.builder.cbranch(self.compile(condition), true_block, false_block)

        self.if_stack.append((true_block, false_block, end_block))
        return true_block, false_block, end_block

    def end_if(self):
        true_block, false_block, end_block = self.if_stack.pop()

        self.builder.position_at_end(true_block)
        if not self.builder.block.is_terminated:
            self.builder.branch(end_block)

        self.builder.position_at_end(false_block)
        if not self.builder.block.is_terminated:
            self.builder.branch(end_block)

        self.builder.position_at_end(end_block)

    def create_while(self, condition):

        loop_block = self.create_block("while_loop")
        end_block = self.create_block("while_end")

        self.builder.cbranch(self.compile(condition), loop_block, end_block)

        self.loop_stack.append((loop_block, end_block))
        return loop_block, end_block

    def end_while(self):
        loop_block, end_block = self.loop_stack.pop()

        self.builder.position_at_end(loop_block)

        if not self.builder.block.is_terminated:
            self.builder.branch(end_block)

        self.builder.position_at_end(end_block)
        if not self.builder.block.is_terminated:

            self.builder.ret_void()

    def _compile_array_literal(
        self, ast_node: ArrayLiteralNode, target_array_type: ir.ArrayType
    ):
        if not isinstance(target_array_type, ir.ArrayType):
            raise Exception(
                "Internal compiler error: _compile_array_literal called with non-array target type."
            )

        llvm_element_type = target_array_type.element
        expected_len = target_array_type.count

        if len(ast_node.elements) != expected_len:

            raise Exception(
                f"Array literal length mismatch. Expected {expected_len}, got {len(ast_node.elements)}."
            )

        llvm_elements = []
        for i, elem_ast in enumerate(ast_node.elements):
            elem_llvm_val = self.compile(elem_ast)

            if elem_llvm_val.type != llvm_element_type:
                if isinstance(llvm_element_type, ir.FloatType) and isinstance(
                    elem_llvm_val.type, ir.IntType
                ):
                    elem_llvm_val = self.builder.sitofp(
                        elem_llvm_val, llvm_element_type, name=f"arr_elem_{i}_conv"
                    )

                else:
                    raise Exception(
                        f"Type mismatch for array element {i}. Expected {llvm_element_type}, "
                        f"got {elem_llvm_val.type}."
                    )
            llvm_elements.append(elem_llvm_val)

        return ir.Constant(target_array_type, llvm_elements)

    def compile(self, ast: Node):
        if ast is None:
            return
        else:
            try:
                print(
                    f"[DEBUG] Current block: {self.builder.block.name}, Terminated: {self.builder.block.is_terminated}"
                )
                print(f"[DEBUG] Current Scope.symbols : {self.current_scope.symbols}")
            except:
                ...

            if isinstance(ast, VariableDeclaration):
                name = ast.identifier
                declared_type_ast = ast.type
                init_expr_ast = ast.value

                if self.function is None:
                    if ast.is_mutable:
                        raise Exception(
                            f"Mutable variable '{name}' can only be declared inside functions for now."
                        )

                    llvm_global_var_type = None
                    initial_val_llvm_const = None

                    if declared_type_ast == "auto":
                        if init_expr_ast is None:
                            raise Exception(
                                f"Global 'auto' const '{name}' requires an initializer."
                            )

                        if not isinstance(init_expr_ast, (Literal, ArrayLiteralNode)):
                            raise Exception(
                                f"Initializer for global 'auto' const '{name}' must be a literal or array literal."
                            )

                        initial_val_llvm_const = self.compile(init_expr_ast)
                        if not isinstance(initial_val_llvm_const, ir.Constant):
                            raise Exception(
                                f"Initializer for global 'auto' const '{name}' did not compile to a constant."
                            )
                        llvm_global_var_type = initial_val_llvm_const.type

                    elif isinstance(declared_type_ast, ArrayTypeNode):
                        element_llvm_type = self.convert_type(
                            declared_type_ast.element_type
                        )
                        explicit_size = None
                        if declared_type_ast.size_expr:
                            if not isinstance(
                                declared_type_ast.size_expr, Literal
                            ) or not isinstance(declared_type_ast.size_expr.value, int):
                                raise Exception(
                                    f"Global array '{name}' size must be a constant integer literal."
                                )
                            explicit_size = declared_type_ast.size_expr.value
                            if explicit_size <= 0:
                                raise Exception(
                                    f"Global array '{name}' size must be positive."
                                )

                        inferred_size = None
                        if init_expr_ast:
                            if not isinstance(init_expr_ast, ArrayLiteralNode):
                                raise Exception(
                                    f"Initializer for global array '{name}' must be an array literal."
                                )
                            if not init_expr_ast.elements and not explicit_size:
                                raise Exception(
                                    f"Global array '{name}' with empty initializer needs explicit size."
                                )
                            inferred_size = (
                                len(init_expr_ast.elements)
                                if init_expr_ast.elements
                                else 0
                            )

                        final_size = None
                        if explicit_size is not None:
                            final_size = explicit_size
                            if (
                                inferred_size is not None
                                and explicit_size != inferred_size
                            ):
                                raise Exception(
                                    f"Global array '{name}' size mismatch: declared {explicit_size}, init has {inferred_size}."
                                )
                        elif inferred_size is not None:
                            final_size = inferred_size
                        else:
                            raise Exception(
                                f"Global array '{name}' size cannot be determined."
                            )

                        llvm_global_var_type = ir.ArrayType(
                            element_llvm_type, final_size
                        )
                        if init_expr_ast:
                            initial_val_llvm_const = self._compile_array_literal(
                                init_expr_ast, llvm_global_var_type
                            )
                        elif not ast.is_mutable:
                            raise Exception(
                                f"Global constant array '{name}' must be initialized."
                            )
                        else:
                            initial_val_llvm_const = ir.Constant(
                                llvm_global_var_type, None
                            )

                    else:
                        llvm_global_var_type = self.convert_type(declared_type_ast)
                        if init_expr_ast:
                            initial_val_llvm_const = self.compile(init_expr_ast)
                            if not isinstance(initial_val_llvm_const, ir.Constant):
                                raise Exception(
                                    f"Initializer for global const '{name}' must be a constant."
                                )
                            if initial_val_llvm_const.type != llvm_global_var_type:

                                raise Exception(
                                    f"Type mismatch for global const '{name}'. Expected {llvm_global_var_type}, got {initial_val_llvm_const.type}."
                                )
                        elif not ast.is_mutable:
                            raise Exception(
                                f"Global constant '{name}' must be initialized."
                            )
                        else:
                            initial_val_llvm_const = ir.Constant(
                                llvm_global_var_type, None
                            )

                    global_var = ir.GlobalVariable(
                        self.module, llvm_global_var_type, name=name
                    )
                    global_var.linkage = "internal"
                    global_var.global_constant = not ast.is_mutable
                    if initial_val_llvm_const is not None:
                        global_var.initializer = initial_val_llvm_const
                    elif not ast.is_mutable:
                        raise Exception(
                            f"Global constant '{name}' must have an initializer value resolved."
                        )

                    self.global_scope.define(name, global_var)
                    return global_var

                llvm_var_type = None
                initial_val_llvm = None

                if declared_type_ast == "auto":
                    if init_expr_ast is None:
                        raise Exception(
                            f"Local 'auto' variable '{name}' requires an initializer."
                        )
                    initial_val_llvm = self.compile(init_expr_ast)
                    llvm_var_type = initial_val_llvm.type

                elif isinstance(declared_type_ast, ArrayTypeNode):
                    element_llvm_type = self.convert_type(
                        declared_type_ast.element_type
                    )
                    explicit_size = None
                    if declared_type_ast.size_expr:
                        if not isinstance(
                            declared_type_ast.size_expr, Literal
                        ) or not isinstance(declared_type_ast.size_expr.value, int):
                            raise Exception(
                                f"Array '{name}' size must be a constant integer literal."
                            )
                        explicit_size = declared_type_ast.size_expr.value
                        if explicit_size <= 0:
                            raise Exception(f"Array '{name}' size must be positive.")

                    inferred_size_from_init = None
                    if init_expr_ast:
                        if not isinstance(init_expr_ast, ArrayLiteralNode):
                            raise Exception(
                                f"Initializer for array '{name}' must be an array literal."
                            )
                        if not init_expr_ast.elements and not explicit_size:
                            raise Exception(
                                f"Array '{name}' with empty initializer needs an explicit size."
                            )

                        inferred_size_from_init = (
                            len(init_expr_ast.elements) if init_expr_ast.elements else 0
                        )

                    final_size = None
                    if explicit_size is not None:
                        final_size = explicit_size
                        if (
                            inferred_size_from_init is not None
                            and explicit_size != inferred_size_from_init
                        ):
                            raise Exception(
                                f"Array '{name}' size mismatch: declared as {explicit_size}, "
                                f"but initializer has {inferred_size_from_init} elements."
                            )
                    elif inferred_size_from_init is not None:
                        final_size = inferred_size_from_init
                    else:
                        raise Exception(
                            f"Array '{name}' size cannot be determined. "
                            "Provide an explicit size or a non-empty initializer."
                        )

                    llvm_var_type = ir.ArrayType(element_llvm_type, final_size)

                    if init_expr_ast:
                        initial_val_llvm = self._compile_array_literal(
                            init_expr_ast, llvm_var_type
                        )

                else:
                    llvm_var_type = self.convert_type(declared_type_ast)

                    if init_expr_ast is not None:
                        compiled_rhs = self.compile(init_expr_ast)

                        if compiled_rhs.type == llvm_var_type:
                            initial_val_llvm = compiled_rhs
                        else:

                            coerced = False
                            if isinstance(llvm_var_type, ir.IntType) and isinstance(
                                compiled_rhs.type, ir.IntType
                            ):
                                target_w, source_w = (
                                    llvm_var_type.width,
                                    compiled_rhs.type.width,
                                )
                                if target_w < source_w:
                                    print(
                                        f"Warning: Initializer for '{name}': Truncating i{source_w} to i{target_w}."
                                    )
                                    initial_val_llvm = self.builder.trunc(
                                        compiled_rhs,
                                        llvm_var_type,
                                        name=f"{name}_init_trunc",
                                    )
                                    coerced = True
                                elif target_w > source_w:
                                    print(
                                        f"DEBUG: Initializer for '{name}': Extending i{source_w} to i{target_w}."
                                    )
                                    initial_val_llvm = self.builder.sext(
                                        compiled_rhs,
                                        llvm_var_type,
                                        name=f"{name}_init_sext",
                                    )
                                    coerced = True

                                elif target_w == source_w:
                                    initial_val_llvm = compiled_rhs
                                    coerced = True

                            elif isinstance(llvm_var_type, ir.FloatType) and isinstance(
                                compiled_rhs.type, ir.IntType
                            ):
                                initial_val_llvm = self.builder.sitofp(
                                    compiled_rhs,
                                    llvm_var_type,
                                    name=f"{name}_init_sitofp",
                                )
                                coerced = True
                            elif isinstance(llvm_var_type, ir.IntType) and isinstance(
                                compiled_rhs.type, ir.FloatType
                            ):
                                initial_val_llvm = self.builder.fptosi(
                                    compiled_rhs,
                                    llvm_var_type,
                                    name=f"{name}_init_fptosi",
                                )
                                coerced = True
                            elif isinstance(
                                llvm_var_type, ir.PointerType
                            ) and isinstance(compiled_rhs.type, ir.PointerType):
                                if llvm_var_type != compiled_rhs.type:
                                    print(
                                        f"Warning: Pointer type mismatch for '{name}'. Expected {llvm_var_type}, got {compiled_rhs.type}. Bitcasting."
                                    )
                                    initial_val_llvm = self.builder.bitcast(
                                        compiled_rhs, llvm_var_type
                                    )
                                else:
                                    initial_val_llvm = compiled_rhs
                                coerced = True
                            elif (
                                isinstance(llvm_var_type, ir.IntType)
                                and llvm_var_type.width == 8
                                and isinstance(compiled_rhs.type, ir.PointerType)
                                and isinstance(compiled_rhs.type.pointee, ir.IntType)
                                and compiled_rhs.type.pointee.width == 8
                            ):
                                print(
                                    f"Warning: Assigning string literal (i8*) to char (i8) for '{name}'. Taking first character."
                                )
                                initial_val_llvm = self.builder.load(
                                    compiled_rhs, name=f"{name}_init_char_from_str"
                                )
                                coerced = True

                            if not coerced:
                                raise Exception(
                                    f"Type mismatch for variable '{name}'. Expected {llvm_var_type}, "
                                    f"got {compiled_rhs.type} from initializer. No coercion rule applied."
                                )

                            if initial_val_llvm.type != llvm_var_type:
                                raise Exception(
                                    f"Internal Compiler Error: Coercion failed for '{name}'. Expected {llvm_var_type}, "
                                    f"still have {initial_val_llvm.type} after coercion attempt."
                                )

                if llvm_var_type is None:
                    raise Exception(
                        f"Internal: llvm_var_type not determined for '{name}'"
                    )

                var_ptr = self.builder.alloca(llvm_var_type, name=name + "_ptr")
                self.current_scope.define(name, var_ptr)

                if initial_val_llvm is not None:
                    self.builder.store(initial_val_llvm, var_ptr)
                elif not ast.is_mutable and declared_type_ast != "auto":
                    raise Exception(
                        f"Immutable local variable '{name}' must be initialized."
                    )
                return

            elif isinstance(ast, GenericTypeParameterDeclarationNode):

                self.current_scope.define_type_parameter(ast.name)

                return

            elif isinstance(ast, FunctionDeclaration):
                func_name_in_ast = ast.name

                is_defining_new_template = False
                if hasattr(ast, "type_parameters") and ast.type_parameters:

                    all_params_bound = True
                    for tp_name in ast.type_parameters:
                        if not self.current_scope.get_bound_type(tp_name):
                            all_params_bound = False
                            break
                    if not all_params_bound:
                        is_defining_new_template = True

                if is_defining_new_template:

                    if func_name_in_ast in self.generic_function_templates:

                        pass
                    else:

                        self.generic_function_templates[func_name_in_ast] = (
                            ast.type_parameters,
                            ast,
                        )

                    existing_symbol = self.current_scope.resolve(func_name_in_ast)
                    if not (
                        existing_symbol
                        and isinstance(existing_symbol, dict)
                        and existing_symbol.get("_is_generic_template")
                    ):
                        self.current_scope.define(
                            func_name_in_ast,
                            {"_is_generic_template": True, "name": func_name_in_ast},
                        )

                    return

                prev_function = self.function
                prev_builder = self.builder

                llvm_ret_type = self.convert_type(ast.return_type)
                llvm_param_types = []
                for p_ast in ast.params:

                    llvm_param_types.append(self.convert_type(p_ast.var_type))

                if not isinstance(llvm_ret_type, ir.Type):
                    raise Exception(
                        f"Return type '{ast.return_type}' for function '{func_name_in_ast}' did not resolve to an LLVM type. Resolved to: {llvm_ret_type} (type: {type(llvm_ret_type)})"
                    )
                for i, lp_type in enumerate(llvm_param_types):
                    if not isinstance(lp_type, ir.Type):
                        raise Exception(
                            f"Parameter type '{ast.params[i].var_type}' for function '{func_name_in_ast}' did not resolve to an LLVM type. Resolved to: {lp_type} (type: {type(lp_type)})"
                        )

                llvm_func_type = ir.FunctionType(
                    llvm_ret_type,
                    llvm_param_types,
                    var_arg=ast.is_static if hasattr(ast, "is_vararg") else False,
                )

                try:
                    llvm_function = self.module.get_global(func_name_in_ast)
                    if (
                        not isinstance(llvm_function, ir.Function)
                        or llvm_function.function_type != llvm_func_type
                    ):
                        raise Exception(
                            f"Function '{func_name_in_ast}' signature mismatch or non-function symbol conflict."
                        )
                    if not llvm_function.is_declaration and ast.body:
                        raise Exception(
                            f"Function '{func_name_in_ast}' body redefined."
                        )
                except KeyError:
                    llvm_function = ir.Function(
                        self.module, llvm_func_type, name=func_name_in_ast
                    )

                self.function = llvm_function

                if not (hasattr(ast, "type_parameters") and ast.type_parameters):

                    if (
                        self.current_scope.resolve(func_name_in_ast)
                        is not llvm_function
                    ):
                        self.current_scope.define(func_name_in_ast, llvm_function)

                if ast.body:
                    self.enter_scope()

                    entry_block = llvm_function.append_basic_block(name="entry")
                    self.builder = ir.IRBuilder(entry_block)

                    for i, param_ast_node in enumerate(ast.params):
                        llvm_arg = llvm_function.args[i]
                        llvm_arg.name = param_ast_node.identifier
                        param_alloca = self.builder.alloca(
                            llvm_param_types[i], name=f"{param_ast_node.identifier}_ptr"
                        )
                        self.builder.store(llvm_arg, param_alloca)
                        self.current_scope.define(
                            param_ast_node.identifier, param_alloca
                        )

                    for stmt_node in ast.body:
                        self.compile(stmt_node)

                    if not self.builder.block.is_terminated and isinstance(
                        llvm_ret_type, ir.VoidType
                    ):
                        self.builder.ret_void()

                    self.exit_scope()

                self.function = prev_function
                self.builder = prev_builder
                if func_name_in_ast == "main":
                    self.main_function = llvm_function
                return llvm_function

            elif isinstance(ast, AddressOfNode):

                target_expr_ast = ast.expression

                if isinstance(target_expr_ast, str):
                    var_ptr = self.current_scope.resolve(target_expr_ast)
                    if var_ptr is None:
                        raise Exception(
                            f"Cannot take address of unknown variable '{target_expr_ast}'"
                        )
                    return var_ptr

                elif isinstance(target_expr_ast, ArrayIndexNode):

                    array_expr_for_gep = None
                    if isinstance(target_expr_ast.array_expr, str):
                        array_expr_for_gep = self.current_scope.resolve(
                            target_expr_ast.array_expr
                        )
                    else:

                        array_expr_for_gep = self.compile(target_expr_ast.array_expr)

                    if not isinstance(array_expr_for_gep.type, ir.PointerType):
                        raise Exception(
                            f"Cannot GEP from non-pointer type for &array[idx]: {array_expr_for_gep.type}"
                        )

                    index_val = self.compile(target_expr_ast.index_expr)
                    if not isinstance(index_val.type, ir.IntType):
                        raise Exception(
                            f"Array index for address-of must be an integer, got {index_val.type}"
                        )

                    zero = ir.Constant(ir.IntType(32), 0)
                    element_ptr = self.builder.gep(
                        array_expr_for_gep,
                        [zero, index_val],
                        inbounds=True,
                        name="addr_of_elem_ptr",
                    )
                    return element_ptr

                elif isinstance(target_expr_ast, MemberAccess):

                    struct_expr_for_gep = None

                    if isinstance(target_expr_ast.struct_name, str):
                        struct_expr_for_gep = self.current_scope.resolve(
                            target_expr_ast.struct_name
                        )
                    else:
                        struct_expr_for_gep = self.compile(target_expr_ast.struct_name)

                    if not isinstance(
                        struct_expr_for_gep.type, ir.PointerType
                    ) or not isinstance(
                        struct_expr_for_gep.type.pointee, ir.IdentifiedStructType
                    ):
                        raise Exception(
                            f"Cannot take address of field from non-struct pointer for '{target_expr_ast.struct_name}'"
                        )

                    struct_llvm_type = struct_expr_for_gep.type.pointee
                    struct_name_str = struct_llvm_type.name
                    field_name_str = target_expr_ast.member_name

                    field_indices = self.struct_field_indices.get(struct_name_str)
                    if field_indices is None or field_name_str not in field_indices:
                        raise Exception(
                            f"Struct '{struct_name_str}' has no field '{field_name_str}' for address-of."
                        )

                    idx = field_indices[field_name_str]
                    zero = ir.Constant(ir.IntType(32), 0)
                    llvm_idx = ir.Constant(ir.IntType(32), idx)

                    field_ptr = self.builder.gep(
                        struct_expr_for_gep,
                        [zero, llvm_idx],
                        inbounds=True,
                        name="addr_of_field_ptr",
                    )
                    return field_ptr
                else:

                    raise Exception(
                        f"Cannot take address of expression type: {type(target_expr_ast)}"
                    )

            elif isinstance(ast, DereferenceNode):
                ptr_val = self.compile(ast.expression)
                if not isinstance(ptr_val.type, ir.PointerType):
                    raise Exception(
                        f"Cannot dereference non-pointer type: {ptr_val.type}"
                    )

                return self.builder.load(ptr_val, name="deref_val")

            elif isinstance(ast, ArrayIndexNode):
                index_llvm_val = self.compile(ast.index_expr)
                if not (isinstance(index_llvm_val.type, ir.IntType)):
                    raise Exception(
                        f"Array index must be an integer, got {index_llvm_val.type}"
                    )

                array_base_for_gep = None
                array_expr_str_representation = str(ast.array_expr)

                if isinstance(ast.array_expr, str):

                    array_base_for_gep = self.current_scope.resolve(ast.array_expr)
                    if array_base_for_gep is None:
                        raise Exception(
                            f"Array variable '{ast.array_expr}' not found for indexing."
                        )
                    if not isinstance(array_base_for_gep.type, ir.PointerType):
                        raise Exception(
                            f"Symbol '{ast.array_expr}' is not a pointer, cannot index. Type: {array_base_for_gep.type}"
                        )

                elif (
                    isinstance(ast.array_expr, ArrayIndexNode)
                    or isinstance(ast.array_expr, MemberAccess)
                    or isinstance(ast.array_expr, AsPtrNode)
                    or isinstance(ast.array_expr, DereferenceNode)
                ):

                    array_base_for_gep = self.compile(ast.array_expr)
                    if not isinstance(array_base_for_gep.type, ir.PointerType):
                        raise Exception(
                            f"Base of chained index/access '{ast.array_expr}' "
                            f"did not result in a pointer. Got: {array_base_for_gep.type}"
                        )
                else:

                    temp_val = self.compile(ast.array_expr)
                    if not isinstance(temp_val.type, ir.PointerType):
                        raise Exception(
                            f"Array expression '{ast.array_expr}' did not evaluate to a pointer "
                            f"for indexing. Got: {temp_val.type}"
                        )
                    array_base_for_gep = temp_val

                zero = ir.Constant(ir.IntType(32), 0)
                element_ptr = None
                pointee_type = array_base_for_gep.type.pointee

                if isinstance(pointee_type, ir.ArrayType):

                    element_ptr = self.builder.gep(
                        array_base_for_gep,
                        [zero, index_llvm_val],
                        inbounds=True,
                        name=f"{array_expr_str_representation}_elem_ptr",
                    )
                else:

                    element_ptr = self.builder.gep(
                        array_base_for_gep,
                        [index_llvm_val],
                        inbounds=True,
                        name=f"{array_expr_str_representation}_elem_ptr",
                    )

                return element_ptr
            elif isinstance(ast, ImportModule):
                path = ast.path
                alias = ast.alias or os.path.splitext(os.path.basename(path))[0]

                module_asts = parse_file(path)

                before = set(self.module.globals)

                helper = Compiler()
                helper.module = self.module
                helper.imported_libs = list(self.imported_libs)
                helper.enum_types = dict(self.enum_types)
                helper.enum_members = {k: dict(v) for k, v in self.enum_members.items()}
                helper.struct_types = dict(self.struct_types)
                helper.struct_field_indices = dict(self.struct_field_indices)
                helper.struct_methods = dict(self.struct_methods)

                helper.module_aliases = dict(self.module_aliases)
                helper.loaded_modules = dict(self.loaded_modules)

                for node in module_asts.statements:
                    if isinstance(node, FunctionDeclaration) and node.name == "main":
                        continue
                    if isinstance(
                        node,
                        (
                            FunctionDeclaration,
                            StructDeclaration,
                            EnumDeclaration,
                            Extern,
                            ImportModule,
                        ),
                    ):
                        helper.compile(node)

                after = set(self.module.globals)
                new = after - before
                namespace = {name: self.module.globals[name] for name in new}

                self.module_enum_types[path] = dict(helper.enum_types)
                self.module_enum_members[path] = {
                    k: dict(v) for k, v in helper.enum_members.items()
                }
                self.module_struct_types[path] = dict(helper.struct_types)
                self.module_struct_fields[path] = dict(helper.struct_field_indices)

                self.loaded_modules[path] = namespace
                self.module_aliases[alias] = path
                return

            elif isinstance(ast, ModuleAccess):

                if isinstance(ast.alias, ModuleAccess):
                    inner = ast.alias
                    mod_alias = inner.alias
                    enum_name = inner.name
                    variant = ast.name

                    if mod_alias not in self.module_aliases:
                        raise Exception(f"Module '{mod_alias}' not imported.")
                    path = self.module_aliases[mod_alias]

                    m_enums = self.module_enum_members.get(path, {})
                    if enum_name not in m_enums:
                        raise Exception(
                            f"Enum '{enum_name}' not defined in module '{mod_alias}'."
                        )
                    members = m_enums[enum_name]

                    if variant not in members:
                        raise Exception(
                            f"Enum '{enum_name}' has no member '{variant}' in module '{mod_alias}'."
                        )
                    return members[variant]

                if isinstance(ast.alias, str) and ast.alias in self.enum_members:
                    enum_name = ast.alias
                    variant = ast.name
                    members = self.enum_members[enum_name]
                    if variant not in members:
                        raise Exception(
                            f"Enum '{enum_name}' has no member '{variant}'."
                        )
                    return members[variant]

                alias = ast.alias
                name = ast.name

                if alias not in self.module_aliases:
                    raise Exception(f"Module '{alias}' not imported.")

                path = self.module_aliases[alias]
                namespace = self.loaded_modules[path]
                if name not in namespace:
                    raise Exception(f"Module '{alias}' has no symbol '{name}'.")
                return namespace[name]

            elif isinstance(ast, TypeConv):

                val = self.compile(ast.expr)
                src_ty = val.type

                tgt_ty = self.convert_type(ast.target_type)

                if src_ty == tgt_ty:
                    return val

                if isinstance(src_ty, ir.IntType) and isinstance(tgt_ty, ir.IntType):
                    sw = src_ty.width
                    tw = tgt_ty.width
                    if sw < tw:

                        return self.builder.sext(val, tgt_ty, name="conv_sext")
                    else:

                        return self.builder.trunc(val, tgt_ty, name="conv_trunc")

                if isinstance(src_ty, ir.IntType) and isinstance(tgt_ty, ir.FloatType):
                    return self.builder.sitofp(val, tgt_ty, name="conv_sitofp")

                if isinstance(src_ty, ir.FloatType) and isinstance(tgt_ty, ir.IntType):
                    return self.builder.fptosi(val, tgt_ty, name="conv_fptosi")

                if isinstance(src_ty, ir.FloatType) and isinstance(
                    tgt_ty, ir.FloatType
                ):

                    return (
                        self.builder.fpext(val, tgt_ty, name="conv_fpext")
                        if src_ty.width < tgt_ty.width
                        else self.builder.fptrunc(val, tgt_ty, name="conv_fptrunc")
                    )

                if isinstance(src_ty, ir.PointerType) and isinstance(
                    tgt_ty, ir.PointerType
                ):
                    return self.builder.bitcast(val, tgt_ty, name="conv_bitcast")

                raise Exception(f"Cannot convert from {src_ty} to {tgt_ty}")

            elif isinstance(ast, IfStatement):
                func = self.function
                if_id_suffix = f".{self.block_count}"
                self.block_count += 1

                cond_val = self.compile(ast.condition)
                if not (
                    isinstance(cond_val.type, ir.IntType) and cond_val.type.width == 1
                ):
                    raise Exception(
                        f"If condition must be a boolean (i1), got {cond_val.type}"
                    )

                then_bb = func.append_basic_block(f"if_then{if_id_suffix}")
                merge_bb = func.append_basic_block(f"if_merge{if_id_suffix}")

                current_false_target_bb = merge_bb
                if ast.elifs or ast.else_body:
                    current_false_target_bb = func.append_basic_block(
                        f"if_cond_false{if_id_suffix}"
                    )

                self.builder.cbranch(cond_val, then_bb, current_false_target_bb)

                self.builder.position_at_end(then_bb)
                self.compile(ast.body)
                if not self.builder.block.is_terminated:
                    self.builder.branch(merge_bb)

                if ast.elifs:
                    for i, (elif_cond_ast, elif_body_ast) in enumerate(ast.elifs):
                        self.builder.position_at_end(current_false_target_bb)

                        elif_then_bb = func.append_basic_block(
                            f"elif{i}_then{if_id_suffix}"
                        )

                        next_false_target_for_elif = merge_bb
                        if i < len(ast.elifs) - 1 or ast.else_body:
                            next_false_target_for_elif = func.append_basic_block(
                                f"elif{i}_false_path{if_id_suffix}"
                            )

                        elif_cond_val = self.compile(elif_cond_ast)

                        self.builder.cbranch(
                            elif_cond_val, elif_then_bb, next_false_target_for_elif
                        )

                        self.builder.position_at_end(elif_then_bb)
                        self.compile(elif_body_ast)
                        if not self.builder.block.is_terminated:
                            self.builder.branch(merge_bb)

                        current_false_target_bb = next_false_target_for_elif

                self.builder.position_at_end(current_false_target_bb)
                if ast.else_body:
                    self.compile(ast.else_body)
                    if not self.builder.block.is_terminated:
                        self.builder.branch(merge_bb)
                elif current_false_target_bb != merge_bb:

                    if not self.builder.block.is_terminated:
                        self.builder.branch(merge_bb)

                self.builder.position_at_end(merge_bb)

                if not merge_bb.is_terminated and not merge_bb.instructions:

                    is_void_function = isinstance(func.return_value.type, ir.VoidType)
                    if not is_void_function:

                        if not merge_bb.instructions:
                            print(
                                f"Warning: Adding 'unreachable' to empty merge block '{merge_bb.name}' in non-void function '{func.name}'. This implies all paths before it returned."
                            )
                            self.builder.unreachable()

                return

            elif isinstance(ast, WhileLoop):

                cond_block = self.function.append_basic_block("while_cond")
                body_block = self.function.append_basic_block("while_body")
                end_block = self.function.append_basic_block("while_end")

                self.builder.branch(cond_block)

                self.builder.position_at_end(cond_block)
                cond_val = self.compile(ast.condition)
                if (
                    not isinstance(cond_val.type, ir.IntType)
                    or cond_val.type.width != 1
                ):
                    raise Exception(
                        f"While loop condition must evaluate to a boolean (i1), got {cond_val.type}"
                    )
                self.builder.cbranch(cond_val, body_block, end_block)

                self.builder.position_at_end(body_block)

                self.enter_scope(
                    is_loop_scope=True,
                    loop_cond_block=cond_block,
                    loop_end_block=end_block,
                )

                self.compile(ast.body)

                if not self.builder.block.is_terminated:
                    self.builder.branch(cond_block)

                self.exit_scope()

                self.builder.position_at_end(end_block)
                return
            elif isinstance(ast, SizeofNode):
                target_ast = ast.target_ast_node
                llvm_type_to_get_size_of = None

                if isinstance(
                    target_ast,
                    (
                        str,
                        ArrayTypeNode,
                        PointerTypeNode,
                        TypeParameterNode,
                        TypeAnnotation,
                        ModuleAccess,
                    ),
                ):

                    is_type_target = False
                    if isinstance(
                        target_ast, (ArrayTypeNode, PointerTypeNode, TypeAnnotation)
                    ):
                        is_type_target = True
                    elif isinstance(target_ast, str):

                        try:

                            llvm_type_to_get_size_of = self.convert_type(target_ast)
                            if isinstance(llvm_type_to_get_size_of, ir.Type):
                                is_type_target = True
                            else:

                                is_type_target = False
                                if self.current_scope.is_type_parameter(
                                    target_ast
                                ) and not self.current_scope.get_bound_type(target_ast):
                                    raise Exception(
                                        f"sizeof(<{target_ast}>) : Type parameter '{target_ast}' is not bound to a concrete type."
                                    )

                        except Exception:

                            is_type_target = False

                    if is_type_target:
                        if not isinstance(llvm_type_to_get_size_of, ir.Type):
                            llvm_type_to_get_size_of = self.convert_type(target_ast)
                    else:
                        compiled_expr_val = self.compile(target_ast)
                        llvm_type_to_get_size_of = compiled_expr_val.type

                else:
                    compiled_expr_val = self.compile(target_ast)
                    llvm_type_to_get_size_of = compiled_expr_val.type

                if not isinstance(llvm_type_to_get_size_of, ir.Type):
                    raise Exception(
                        f"Could not determine a valid LLVM type for sizeof target: {target_ast}. Got: {llvm_type_to_get_size_of}"
                    )

                if self.data_layout_obj is None:
                    raise Exception(
                        "Compiler's DataLayout object not initialized. Cannot compute sizeof."
                    )

                try:
                    size_in_bytes = llvm_type_to_get_size_of.get_abi_size(
                        self.data_layout_obj
                    )
                except Exception as e:
                    raise Exception(
                        f"Error getting ABI size for type '{llvm_type_to_get_size_of}' in sizeof: {e}"
                    )

                if size_in_bytes == 0 and not llvm_type_to_get_size_of.is_zero_sized:
                    opaque_info = (
                        f", is_opaque={llvm_type_to_get_size_of.is_opaque}"
                        if hasattr(llvm_type_to_get_size_of, "is_opaque")
                        else ""
                    )
                    name_info = (
                        f", name='{llvm_type_to_get_size_of.name}'"
                        if hasattr(llvm_type_to_get_size_of, "name")
                        else ""
                    )
                    raise Exception(
                        f"sizeof target '{target_ast}' (type '{llvm_type_to_get_size_of}'{name_info}{opaque_info}) "
                        "resulted in size 0, but type is not zero-sized. Type may be incomplete."
                    )

                return ir.Constant(ir.IntType(64), size_in_bytes)
            elif isinstance(ast, AsPtrNode):
                target_expr_ast = ast.expression_ast

                if isinstance(target_expr_ast, str):
                    var_name = target_expr_ast
                    var_mem_location_ptr = self.current_scope.resolve(var_name)

                    if var_mem_location_ptr is None:
                        raise Exception(f"as_ptr: Unknown variable '{var_name}'")
                    if not isinstance(
                        var_mem_location_ptr, (ir.AllocaInstr, ir.GlobalVariable)
                    ):
                        raise Exception(
                            f"as_ptr: Symbol '{var_name}' is not a variable location (not Alloca or Global). Got {type(var_mem_location_ptr)}"
                        )

                    return var_mem_location_ptr

                elif isinstance(target_expr_ast, ArrayIndexNode):

                    array_base_ptr = None
                    array_expr_str_repr = str(target_expr_ast.array_expr)
                    if isinstance(target_expr_ast.array_expr, str):
                        array_base_ptr = self.current_scope.resolve(
                            target_expr_ast.array_expr
                        )
                    else:
                        array_base_ptr = self.compile(target_expr_ast.array_expr)

                    if array_base_ptr is None or not isinstance(
                        array_base_ptr.type, ir.PointerType
                    ):
                        raise Exception(
                            f"as_ptr: Base of array index '{array_expr_str_repr}' is not a valid pointer."
                        )

                    index_llvm_val = self.compile(target_expr_ast.index_expr)
                    if not (isinstance(index_llvm_val.type, ir.IntType)):
                        raise Exception("as_ptr: Array index must be an integer.")

                    zero = ir.Constant(ir.IntType(32), 0)
                    element_ptr = None
                    pointee_type = array_base_ptr.type.pointee
                    if isinstance(pointee_type, ir.ArrayType):
                        element_ptr = self.builder.gep(
                            array_base_ptr,
                            [zero, index_llvm_val],
                            inbounds=True,
                            name=f"asptr_arr_elem_ptr",
                        )
                    else:
                        element_ptr = self.builder.gep(
                            array_base_ptr,
                            [index_llvm_val],
                            inbounds=True,
                            name=f"asptr_ptr_elem_ptr",
                        )
                    return element_ptr

                elif isinstance(target_expr_ast, MemberAccess):

                    struct_instance_ptr = None
                    struct_expr_str_repr = str(target_expr_ast.struct_name)

                    if isinstance(target_expr_ast.struct_name, str):
                        alloca_for_struct_ptr = self.current_scope.resolve(
                            target_expr_ast.struct_name
                        )
                        if alloca_for_struct_ptr is None or not isinstance(
                            alloca_for_struct_ptr.type, ir.PointerType
                        ):
                            raise Exception(
                                f"as_ptr: Struct instance '{target_expr_ast.struct_name}' not found or not a pointer."
                            )

                        if isinstance(
                            alloca_for_struct_ptr.type.pointee, ir.PointerType
                        ) and isinstance(
                            alloca_for_struct_ptr.type.pointee.pointee,
                            ir.IdentifiedStructType,
                        ):
                            struct_instance_ptr = self.builder.load(
                                alloca_for_struct_ptr,
                                name=f"{target_expr_ast.struct_name}_val_ptr",
                            )
                        elif isinstance(
                            alloca_for_struct_ptr.type.pointee, ir.IdentifiedStructType
                        ):
                            struct_instance_ptr = alloca_for_struct_ptr
                        else:
                            raise Exception(
                                f"as_ptr: Variable '{target_expr_ast.struct_name}' is not a pointer to a struct instance."
                            )
                    else:
                        struct_instance_ptr = self.compile(target_expr_ast.struct_name)

                    if not isinstance(
                        struct_instance_ptr.type, ir.PointerType
                    ) or not isinstance(
                        struct_instance_ptr.type.pointee, ir.IdentifiedStructType
                    ):
                        raise Exception(
                            f"as_ptr: Expression for struct instance '{struct_expr_str_repr}' did not yield a pointer to a known struct type."
                        )

                    struct_llvm_type = struct_instance_ptr.type.pointee
                    struct_name_str = struct_llvm_type.name
                    field_name_str = target_expr_ast.member_name

                    field_indices = self.struct_field_indices.get(struct_name_str)
                    if field_indices is None or field_name_str not in field_indices:
                        raise Exception(
                            f"as_ptr: Struct '{struct_name_str}' has no field '{field_name_str}'."
                        )

                    idx = field_indices[field_name_str]
                    zero = ir.Constant(ir.IntType(32), 0)
                    llvm_idx = ir.Constant(ir.IntType(32), idx)

                    field_ptr = self.builder.gep(
                        struct_instance_ptr,
                        [zero, llvm_idx],
                        inbounds=True,
                        name=f"asptr_field_ptr",
                    )
                    return field_ptr
                else:

                    raise Exception(
                        f"as_ptr: Cannot take pointer of expression type '{type(target_expr_ast)}'. Target must be an l-value (variable, array element, or field)."
                    )
            elif type(ast) == ReturnStatement:

                ret_value = ast.value

                if ret_value is not None:
                    value = self.get_variable(ret_value)
                    self.builder.ret(value)
                else:
                    self.builder.ret_void()
            elif isinstance(ast, Literal):
                py_value = ast.value

                if isinstance(py_value, str):

                    unescaped_str = codecs.decode(py_value, "unicode_escape")

                    return self.create_global_string(unescaped_str)

                elif isinstance(py_value, (int, float, bool)):

                    llvm_type = self.guess_type(py_value)

                    try:
                        return ir.Constant(llvm_type, py_value)
                    except OverflowError:
                        raise Exception(
                            f"Cannot create LLVM constant for literal '{py_value}' of inferred type {llvm_type}."
                        )
                    except Exception as e:
                        raise Exception(
                            f"Error creating LLVM constant for '{py_value}' (type {llvm_type}): {e}"
                        )

                else:
                    raise Exception(
                        f"Unsupported Python value type in Literal AST node: {type(py_value)}"
                    )

            elif isinstance(ast, Assignment):
                lhs_ast_node = ast.identifier
                rhs_ast_node = ast.value
                assign_op = ast.operator

                rhs_llvm_val = self.compile(rhs_ast_node)

                target_ptr = None

                if isinstance(lhs_ast_node, str):
                    var_name = lhs_ast_node

                    var_mem_location = self.current_scope.resolve(var_name)

                    if var_mem_location is None or not isinstance(
                        var_mem_location.type, ir.PointerType
                    ):
                        raise Exception(
                            f"Cannot assign to undefined or non-assignable symbol '{var_name}'."
                        )

                    if (
                        isinstance(var_mem_location, ir.GlobalVariable)
                        and var_mem_location.global_constant
                    ):
                        raise Exception(
                            f"Cannot assign to global constant '{var_name}'."
                        )

                    target_ptr = var_mem_location

                elif isinstance(lhs_ast_node, DereferenceNode):

                    ptr_expr_llvm = self.compile(lhs_ast_node.expression)
                    if not isinstance(ptr_expr_llvm.type, ir.PointerType):
                        raise Exception(
                            f"Cannot assign to dereferenced non-pointer: {lhs_ast_node.expression}"
                        )
                    target_ptr = ptr_expr_llvm

                elif isinstance(lhs_ast_node, ArrayIndexNode):
                    array_base_ptr = None

                    if isinstance(lhs_ast_node.array_expr, str):
                        array_name = lhs_ast_node.array_expr
                        array_base_ptr = self.current_scope.resolve(array_name)
                        if array_base_ptr is None or not isinstance(
                            array_base_ptr.type, ir.PointerType
                        ):
                            raise Exception(
                                f"Array '{array_name}' not found or not a pointer for LHS of assignment."
                            )
                    else:
                        array_base_ptr = self.compile(lhs_ast_node.array_expr)
                        if not isinstance(array_base_ptr.type, ir.PointerType):
                            raise Exception(
                                f"LHS array expression in assignment did not evaluate to a pointer: {lhs_ast_node.array_expr}"
                            )

                    array_expr_for_gep = None
                    raw_array_expr = lhs_ast_node.array_expr
                    if isinstance(raw_array_expr, str):
                        resolved_array_sym = self.current_scope.resolve(raw_array_expr)
                        if resolved_array_sym is None:
                            raise Exception(
                                f"Array '{raw_array_expr}' not found for LHS assignment."
                            )
                        if not isinstance(resolved_array_sym.type, ir.PointerType):
                            raise Exception(
                                f"Symbol '{raw_array_expr}' is not a pointer for array assignment."
                            )

                        if isinstance(
                            resolved_array_sym.type.pointee, ir.PointerType
                        ) and isinstance(
                            resolved_array_sym.type.pointee.pointee, ir.ArrayType
                        ):
                            array_expr_for_gep = self.builder.load(
                                resolved_array_sym, name=f"{raw_array_expr}_arrptr"
                            )
                        else:
                            array_expr_for_gep = resolved_array_sym
                    else:
                        array_expr_for_gep = self.compile(raw_array_expr)
                    if not isinstance(array_expr_for_gep.type, ir.PointerType):
                        raise Exception(
                            f"LHS array expression '{raw_array_expr}' did not evaluate to a pointer."
                        )

                    index_llvm = self.compile(lhs_ast_node.index_expr)
                    if not isinstance(index_llvm.type, ir.IntType):
                        raise Exception("Array index in assignment must be an integer.")

                    zero = ir.Constant(ir.IntType(32), 0)
                    if isinstance(array_expr_for_gep.type.pointee, ir.ArrayType):
                        target_ptr = self.builder.gep(
                            array_expr_for_gep,
                            [zero, index_llvm],
                            inbounds=True,
                            name="assign_idx_ptr",
                        )
                    else:
                        target_ptr = self.builder.gep(
                            array_expr_for_gep,
                            [index_llvm],
                            inbounds=True,
                            name="assign_idx_ptr",
                        )
                elif isinstance(lhs_ast_node, MemberAccess):
                    struct_instance_ptr_val = None

                    if isinstance(lhs_ast_node.struct_name, str):
                        instance_name = lhs_ast_node.struct_name

                        alloca_for_instance_ptr = self.current_scope.resolve(
                            instance_name
                        )
                        if alloca_for_instance_ptr is None or not isinstance(
                            alloca_for_instance_ptr.type, ir.PointerType
                        ):
                            raise Exception(
                                f"Struct instance variable '{instance_name}' not found or not a pointer."
                            )

                        struct_instance_ptr_val = self.builder.load(
                            alloca_for_instance_ptr, name=f"{instance_name}_val_ptr"
                        )

                    else:
                        struct_instance_ptr_val = self.compile(lhs_ast_node.struct_name)

                    if not isinstance(
                        struct_instance_ptr_val.type, ir.PointerType
                    ) or not isinstance(
                        struct_instance_ptr_val.type.pointee, ir.IdentifiedStructType
                    ):
                        raise Exception(
                            f"Expression for struct instance '{lhs_ast_node.struct_name}' "
                            f"did not yield a pointer to a known struct type. Got: {struct_instance_ptr_val.type}"
                        )

                    struct_llvm_type = struct_instance_ptr_val.type.pointee
                    struct_name_str = struct_llvm_type.name
                    field_name_str = lhs_ast_node.member_name

                    field_indices = self.struct_field_indices.get(struct_name_str)
                    if field_indices is None or field_name_str not in field_indices:
                        raise Exception(
                            f"Struct '{struct_name_str}' has no field '{field_name_str}' for assignment."
                        )

                    field_idx_int = field_indices[field_name_str]
                    zero = ir.Constant(ir.IntType(32), 0)
                    llvm_field_idx = ir.Constant(ir.IntType(32), field_idx_int)

                    target_ptr = self.builder.gep(
                        struct_instance_ptr_val,
                        [zero, llvm_field_idx],
                        inbounds=True,
                        name=f"{lhs_ast_node.struct_name}_{field_name_str}_ptr",
                    )
                else:
                    raise Exception(
                        f"Invalid Left-Hand-Side for assignment: {type(lhs_ast_node)}"
                    )

                if target_ptr is None:
                    raise Exception(
                        f"Internal error: target_ptr not set for assignment to {lhs_ast_node}"
                    )

                expected_lhs_val_type = target_ptr.type.pointee
                value_to_store = rhs_llvm_val

                if value_to_store.type != expected_lhs_val_type:

                    if isinstance(expected_lhs_val_type, ir.FloatType) and isinstance(
                        value_to_store.type, ir.IntType
                    ):
                        value_to_store = self.builder.sitofp(
                            value_to_store, expected_lhs_val_type, "assign_conv"
                        )

                    elif isinstance(
                        expected_lhs_val_type, ir.PointerType
                    ) and isinstance(value_to_store.type, ir.PointerType):

                        if expected_lhs_val_type != value_to_store.type:
                            print(
                                f"Warning: Assigning pointers of different types: {value_to_store.type} to {expected_lhs_val_type}. May require bitcast."
                            )

                            raise Exception(
                                f"Pointer type mismatch in assignment. Expected {expected_lhs_val_type}, got {value_to_store.type}"
                            )
                    else:
                        raise Exception(
                            f"Type mismatch in assignment. Cannot assign {value_to_store.type} to {expected_lhs_val_type}."
                        )

                if value_to_store.type != expected_lhs_val_type:

                    if isinstance(expected_lhs_val_type, ir.FloatType) and isinstance(
                        value_to_store.type, ir.IntType
                    ):
                        value_to_store = self.builder.sitofp(
                            value_to_store, expected_lhs_val_type, "assign_conv"
                        )

                    else:
                        raise Exception(
                            f"Type mismatch in assignment. Cannot assign {value_to_store.type} to {expected_lhs_val_type} for LHS {lhs_ast_node}."
                        )

                if assign_op == "=":
                    self.builder.store(value_to_store, target_ptr)
                else:

                    current_val = self.builder.load(target_ptr, "compound_old_val")

                    new_val = None
                    if isinstance(current_val.type, ir.IntType):
                        if assign_op == "+=":
                            new_val = self.builder.add(
                                current_val, value_to_store, "compound_add"
                            )
                        elif assign_op == "-=":
                            new_val = self.builder.sub(
                                current_val, value_to_store, "compound_sub"
                            )
                        elif assign_op == "*=":
                            new_val = self.builder.mul(
                                current_val, value_to_store, "compound_mul"
                            )
                        elif assign_op == "/=":
                            new_val = self.builder.sdiv(
                                current_val, value_to_store, "compound_div"
                            )

                        else:
                            raise Exception(
                                f"Unsupported compound assignment operator: {assign_op}"
                            )
                    elif isinstance(current_val.type, ir.FloatType):

                        if assign_op == "+=":
                            new_val = self.builder.fadd(
                                current_val, value_to_store, "compound_fadd"
                            )

                        else:
                            raise Exception(
                                f"Unsupported compound assignment operator '{assign_op}' for floats."
                            )
                    else:
                        raise Exception(
                            f"Compound assignment not supported for type {current_val.type}"
                        )

                    self.builder.store(new_val, target_ptr)

                return
            elif type(ast) == Parameter:

                param_name = ast.name

                param_type = ast.type

                self.create_variable_mut(param_name, param_type)

            elif isinstance(ast, FunctionCall):

                func_to_call_llvm = None
                is_method_call_on_instance = False
                instance_ptr_for_method = None

                if isinstance(ast.call_name, str):
                    call_name_str = ast.call_name
                    resolved_symbol = self.current_scope.resolve(call_name_str)

                    if resolved_symbol is None:
                        try:
                            resolved_symbol = self.module.get_global(call_name_str)
                        except KeyError:
                            pass

                    if resolved_symbol is None:
                        raise Exception(
                            f"Function or generic template '{call_name_str}' not found."
                        )

                    if isinstance(resolved_symbol, ir.Function):
                        func_to_call_llvm = resolved_symbol
                    elif isinstance(resolved_symbol, dict) and resolved_symbol.get(
                        "_is_generic_template"
                    ):

                        (
                            generic_param_names,
                            generic_func_ast,
                        ) = self.generic_function_templates[call_name_str]

                        arg_llvm_values = [
                            self.compile(arg_expr) for arg_expr in ast.params
                        ]
                        arg_llvm_types = [val.type for val in arg_llvm_values]

                        inferred_bindings = {}

                        if not generic_param_names:
                            raise Exception(
                                f"Generic template '{call_name_str}' has no type parameters listed."
                            )

                        for tp_name in generic_param_names:
                            found_binding_for_tp = False
                            for i, ast_param in enumerate(generic_func_ast.params):
                                param_type_repr = ast_param.var_type
                                param_type_name_str = None
                                if isinstance(param_type_repr, str):
                                    param_type_name_str = param_type_repr
                                elif isinstance(param_type_repr, TypeParameterNode):
                                    param_type_name_str = param_type_repr.name

                                if param_type_name_str == tp_name:
                                    if i < len(arg_llvm_types):
                                        inferred_type_for_tp = arg_llvm_types[i]
                                        if (
                                            tp_name in inferred_bindings
                                            and inferred_bindings[tp_name]
                                            != inferred_type_for_tp
                                        ):
                                            raise Exception(
                                                f"Type inference conflict for type parameter '{tp_name}' in call to '{call_name_str}'. "
                                                f"Inferred as {inferred_bindings[tp_name]} then as {inferred_type_for_tp}."
                                            )
                                        inferred_bindings[
                                            tp_name
                                        ] = inferred_type_for_tp
                                        found_binding_for_tp = True

                            if not found_binding_for_tp:
                                raise Exception(
                                    f"Could not infer type for type parameter '{tp_name}' in call to '{call_name_str}'."
                                )

                        concrete_types_tuple = tuple(
                            inferred_bindings[tp_name]
                            for tp_name in generic_param_names
                        )
                        instantiation_key = (call_name_str, concrete_types_tuple)

                        if instantiation_key in self.instantiated_functions:
                            func_to_call_llvm = self.instantiated_functions[
                                instantiation_key
                            ]
                        else:
                            func_to_call_llvm = self._instantiate_and_compile_generic(
                                call_name_str,
                                generic_func_ast,
                                inferred_bindings,
                                concrete_types_tuple,
                            )

                    else:
                        raise Exception(
                            f"Symbol '{call_name_str}' resolved to an unknown callable type: {type(resolved_symbol)}"
                        )

                elif isinstance(ast.call_name, ModuleAccess):

                    resolved_ma_symbol = self.compile(ast.call_name)
                    if isinstance(resolved_ma_symbol, ir.Function):
                        func_to_call_llvm = resolved_ma_symbol

                    else:
                        raise Exception(
                            f"Module access call '{ast.call_name}' did not resolve to a function."
                        )

                elif isinstance(ast.call_name, MemberAccess):

                    raise NotImplementedError(
                        "Direct MemberAccess in function call position not fully handled, use StructMethodCall AST node."
                    )

                else:
                    func_to_call_llvm = self.compile(ast.call_name)
                    if not isinstance(
                        func_to_call_llvm.type, ir.PointerType
                    ) or not isinstance(
                        func_to_call_llvm.type.pointee, ir.FunctionType
                    ):
                        raise Exception(
                            f"Expression '{ast.call_name}' used as function does not evaluate to a function pointer."
                        )

                if func_to_call_llvm and not (
                    isinstance(resolved_symbol, dict)
                    and resolved_symbol.get("_is_generic_template")
                ):

                    arg_llvm_values = [
                        self.compile(arg_expr) for arg_expr in ast.params
                    ]

                if func_to_call_llvm is None:
                    raise Exception(
                        f"Could not resolve function for call: {ast.call_name}"
                    )

                final_args_for_call = []
                if is_method_call_on_instance:
                    final_args_for_call.append(instance_ptr_for_method)
                final_args_for_call.extend(arg_llvm_values)

                fn_llvm_type = func_to_call_llvm.function_type

                num_expected_fixed_args = len(fn_llvm_type.args)
                num_provided_args = len(final_args_for_call)

                if num_provided_args < num_expected_fixed_args:
                    raise Exception(
                        f"Function '{func_to_call_llvm.name}' expects at least {num_expected_fixed_args} args, got {num_provided_args}."
                    )
                if (
                    not fn_llvm_type.var_arg
                    and num_provided_args > num_expected_fixed_args
                ):
                    raise Exception(
                        f"Function '{func_to_call_llvm.name}' takes {num_expected_fixed_args} args, but {num_provided_args} were given."
                    )

                processed_final_args = []
                for i in range(num_provided_args):
                    arg_val = final_args_for_call[i]
                    if i < num_expected_fixed_args:
                        expected_llvm_type = fn_llvm_type.args[i]
                        if arg_val.type != expected_llvm_type:

                            if isinstance(
                                expected_llvm_type, ir.FloatType
                            ) and isinstance(arg_val.type, ir.IntType):
                                arg_val = self.builder.sitofp(
                                    arg_val, expected_llvm_type, f"call_arg{i}_conv"
                                )
                            elif (
                                isinstance(expected_llvm_type, ir.PointerType)
                                and isinstance(arg_val.type, ir.PointerType)
                                and expected_llvm_type != arg_val.type
                            ):
                                print(
                                    f"Warning: Call to '{func_to_call_llvm.name}' arg {i} pointer type mismatch. Expected {expected_llvm_type}, got {arg_val.type}. Bitcasting."
                                )
                                arg_val = self.builder.bitcast(
                                    arg_val, expected_llvm_type
                                )
                            else:
                                raise Exception(
                                    f"Call to '{func_to_call_llvm.name}', argument {i} type mismatch. Expected {expected_llvm_type}, got {arg_val.type}."
                                )
                    elif fn_llvm_type.var_arg:
                        if isinstance(arg_val.type, ir.FloatType):
                            arg_val = self.builder.fpext(
                                arg_val, ir.DoubleType(), f"call_vararg{i}_fpext"
                            )

                    processed_final_args.append(arg_val)

                return self.builder.call(func_to_call_llvm, processed_final_args)

            elif isinstance(ast, MacroDeclaration):
                if ast.name in self.macros:
                    raise Exception(f"Macro '{ast.name}' already declared.")
                self.macros[ast.name] = (ast.params, ast.body)
                return

            elif isinstance(ast, MacroCall):
                if ast.name not in self.macros:
                    raise Exception(f"Macro '{ast.name}' not declared.")

                param_names, body_stmts = self.macros[ast.name]
                if len(param_names) != len(ast.args):
                    raise Exception(
                        f"Macro '{ast.name}' expects {len(param_names)} args, got {len(ast.args)}."
                    )

                mapping = dict(zip(param_names, ast.args))

                result_val = None
                for stmt in body_stmts:
                    expanded = _substitute(stmt, mapping)
                    if isinstance(expanded, ReturnStatement):

                        result_val = self.compile(expanded.value)
                    else:

                        self.compile(expanded)

                if result_val is None:
                    raise Exception(f"Macro '{ast.name}' did not return a value.")
                return result_val

            elif isinstance(ast, TypeOf):

                if isinstance(ast.expr, ModuleAccess):
                    mod_alias, name = ast.expr.alias, ast.expr.name
                    if alias not in self.module_aliases:
                        raise Exception(f"Module '{mod_alias}' not imported.")
                    path = self.module_aliases[mod_alias]

                    if name in self.module_enum_types.get(path, {}):
                        tname = name

                    elif name in self.module_struct_types.get(path, {}):
                        tname = name
                    else:

                        val = self.loaded_modules[path].get(name)
                        if val is None:
                            raise Exception(
                                f"Module '{mod_alias}' has no symbol '{name}'."
                            )

                        llvm_val = (
                            self.builder.call(val, [])
                            if isinstance(val, ir.Function)
                            else self.compile_member_access(ast.expr)
                        )

                        llvm_t = llvm_val.type
                        goto_expr = True

                elif isinstance(ast.expr, str):

                    if ast.expr in self.type_codes:

                        tname = ast.expr
                    else:

                        llvm_val = self.get_variable(ast.expr)
                        llvm_t = llvm_val.type
                        goto_expr = True

                else:

                    llvm_val = self.compile(ast.expr)
                    llvm_t = llvm_val.type
                    goto_expr = True

                if "goto_expr" in locals() and goto_expr:

                    if isinstance(llvm_t, ir.IntType) and llvm_t.width == 32:

                        found = False
                        for en, ty in self.enum_types.items():
                            if ty == llvm_t:
                                tname = en
                                found = True
                                break
                        if not found:
                            tname = "int"
                    elif isinstance(llvm_t, ir.FloatType):
                        tname = "float"
                    elif isinstance(llvm_t, ir.IntType) and llvm_t.width == 1:
                        tname = "bool"
                    elif isinstance(
                        llvm_t, ir.PointerType
                    ) and llvm_t.pointee == ir.IntType(8):
                        tname = "string"
                    elif isinstance(llvm_t, ir.PointerType) and isinstance(
                        llvm_t.pointee, ir.IdentifiedStructType
                    ):
                        tname = str(llvm_t.pointee.name)
                    else:
                        raise Exception(
                            f"typeof(): unsupported value expression type {llvm_t}"
                        )

                code = self.type_codes.get(tname)
                if code is None:
                    raise Exception(f"typeof(): no code for type '{tname}'")
                return ir.Constant(ir.IntType(32), code)

            elif type(ast) == SpecialDeclaration:

                special_name = ast.name

                special_args = [arg.name for arg in ast.arguments]

                self.symbol_table[special_name] = (special_args, ast.body)

                self.compile(ast.body)
            elif type(ast) == StructDeclaration:
                return self.compile_struct(ast)
            elif type(ast) == StructInstantiation:
                return self.compile_struct_instantiation(ast)
            elif isinstance(ast, MemberAccess):

                struct_instance_ptr = None
                if isinstance(ast.struct_name, str):

                    resolved_d = self.current_scope.resolve(ast.struct_name)
                    if resolved_d is None:
                        raise Exception(
                            f"Struct instance variable '{ast.struct_name}' not found."
                        )
                    if not isinstance(resolved_d.type, ir.PointerType):
                        raise Exception(
                            f"Symbol '{ast.struct_name}' is not a pointer (expected alloca)."
                        )

                    if isinstance(
                        resolved_d.type.pointee, ir.PointerType
                    ) and isinstance(
                        resolved_d.type.pointee.pointee, ir.IdentifiedStructType
                    ):

                        struct_instance_ptr = self.builder.load(
                            resolved_d, name=f"{ast.struct_name}_ptr_val"
                        )
                    elif isinstance(resolved_d.type.pointee, ir.IdentifiedStructType):

                        struct_instance_ptr = resolved_d
                    else:
                        raise Exception(
                            f"Variable '{ast.struct_name}' is not a struct instance or pointer to one. Pointee type: {resolved_d.type.pointee}"
                        )

                else:
                    struct_instance_ptr = self.compile(ast.struct_name)

                if not isinstance(
                    struct_instance_ptr.type, ir.PointerType
                ) or not isinstance(
                    struct_instance_ptr.type.pointee, ir.IdentifiedStructType
                ):
                    raise Exception(
                        f"Expression for struct instance '{ast.struct_name}' did not yield a pointer to a known struct type. Got: {struct_instance_ptr.type}"
                    )

                struct_llvm_type = struct_instance_ptr.type.pointee
                struct_name_str = struct_llvm_type.name
                member_name_str = ast.member_name

                field_indices = self.struct_field_indices.get(struct_name_str)
                if field_indices is None or member_name_str not in field_indices:
                    raise Exception(
                        f"Struct type '{struct_name_str}' (from '{ast.struct_name}') has no member '{member_name_str}'."
                    )

                idx = field_indices[member_name_str]

                zero = ir.Constant(ir.IntType(32), 0)
                llvm_idx = ir.Constant(ir.IntType(32), idx)

                field_ptr = self.builder.gep(
                    struct_instance_ptr,
                    [zero, llvm_idx],
                    inbounds=True,
                    name=f"{str(ast.struct_name)}_{member_name_str}_fieldptr",
                )

                field_llvm_type = struct_llvm_type.elements[idx]

                if isinstance(field_llvm_type, (ir.ArrayType, ir.IdentifiedStructType)):

                    return field_ptr
                else:

                    return self.builder.load(
                        field_ptr,
                        name=f"{str(ast.struct_name)}_{member_name_str}_fieldval",
                    )
            elif type(ast) == StructInstantiation:

                struct_name = ast.name

                struct_fields = [field.name for field in ast.fields]

                if struct_name not in self.symbol_table:
                    raise Exception(f"Struct '{struct_name}' not declared.")
                struct_fields, struct_body = self.symbol_table[struct_name]
                if len(struct_fields) != len(struct_body):
                    raise Exception(
                        f"Struct '{struct_name}' takes {len(struct_body)} arguments, but {len(struct_fields)} were given."
                    )
                for i, field in enumerate(struct_fields):
                    self.set_variable(field, struct_fields[i])
                self.compile(struct_body)
            elif isinstance(ast, ImportC):

                lib_name = (
                    ast.path_or_name.strip('"').rstrip('"').strip("'").rstrip("'")
                )
                self.imported_libs.append(lib_name)
                return
            elif type(ast) == FieldAssignment:

                struct_name = ast.name

                field_name = ast.field

                field_value = ast.value

                if struct_name in self.symbol_table:
                    struct_fields, struct_body = self.symbol_table[struct_name]
                    if field_name in struct_fields:
                        self.set_variable(field_name, field_value)
                    else:
                        raise Exception(
                            f"Field '{field_name}' not declared in struct '{struct_name}'."
                        )
                else:
                    raise Exception(f"Struct '{struct_name}' not declared.")
            elif type(ast) == StructMember:

                struct_name = ast.name

                field_name = ast.field

                field_value = ast.value

                if struct_name in self.symbol_table:
                    struct_fields, struct_body = self.symbol_table[struct_name]
                    if field_name in struct_fields:
                        return self.get_variable(field_value)
                    else:
                        raise Exception(
                            f"Field '{field_name}' not declared in struct '{struct_name}'."
                        )
                else:
                    raise Exception(f"Struct '{struct_name}' not declared.")
            elif type(ast) == StructMethodCall:
                return self.compile_struct_method_call(ast)
            elif type(ast) == MemberAccess:

                struct_name = ast.name

                field_name = ast.field

                field_value = ast.value

                if struct_name in self.symbol_table:
                    struct_fields, struct_body = self.symbol_table[struct_name]
                    if field_name in struct_fields:
                        return self.get_variable(field_value)
                    else:
                        raise Exception(
                            f"Field '{field_name}' not declared in struct '{struct_name}'."
                        )
                else:
                    raise Exception(f"Struct '{struct_name}' not declared.")

            elif isinstance(ast, ForLoop):

                self.enter_scope()

                if ast.init is not None:

                    self.compile(ast.init)

                cond_block = self.function.append_basic_block("for_cond")
                body_block = self.function.append_basic_block("for_body")

                inc_block = self.function.append_basic_block("for_increment")
                end_block = self.function.append_basic_block("for_end")

                self.builder.branch(cond_block)

                self.builder.position_at_end(cond_block)
                cond_val = self.compile(ast.condition)
                if (
                    not isinstance(cond_val.type, ir.IntType)
                    or cond_val.type.width != 1
                ):
                    raise Exception(
                        f"For loop condition must evaluate to a boolean (i1), got {cond_val.type}"
                    )
                self.builder.cbranch(cond_val, body_block, end_block)

                self.builder.position_at_end(body_block)

                self.enter_scope(
                    is_loop_scope=True,
                    loop_cond_block=inc_block,
                    loop_end_block=end_block,
                )

                self.compile(ast.body)

                if not self.builder.block.is_terminated:
                    self.builder.branch(inc_block)

                self.exit_scope()

                self.builder.position_at_end(inc_block)
                if ast.increment is not None:

                    self.compile(ast.increment)

                if not self.builder.block.is_terminated:
                    self.builder.branch(cond_block)

                self.builder.position_at_end(end_block)

                self.exit_scope()
                return
            elif type(ast) == ForeachLoop:
                loop_var = ast.identifier
                collection = self.compile(ast.collection)
                body = ast.body

                loop_block = self.create_block("foreach_loop")
                end_block = self.create_block("foreach_end")

                self.loop_stack.append((loop_block, end_block))

                self.builder.branch(loop_block)
                self.builder.position_at_end(loop_block)
                self.compile(body)

                self.builder.branch(loop_block)

                self.loop_stack.pop()
                self.builder.position_at_end(end_block)
            elif type(ast) == Extern:
                va_args = False

                actual_func_args_ast = list(ast.func_args)

                if (
                    actual_func_args_ast
                    and isinstance(actual_func_args_ast[-1], str)
                    and actual_func_args_ast[-1] == "..."
                ):
                    va_args = True
                    actual_func_args_ast.pop()

                param_llvm_types = []
                for param_ast_node in actual_func_args_ast:
                    if not isinstance(param_ast_node, Parameter):

                        param_llvm_types.append(self.convert_type(param_ast_node))
                    else:
                        param_llvm_types.append(
                            self.convert_type(param_ast_node.var_type)
                        )

                fn_ty = ir.FunctionType(
                    self.convert_type(ast.func_return_type),
                    param_llvm_types,
                    var_arg=va_args,
                )
                fn = ir.Function(self.module, fn_ty, name=ast.func_name)

                try:
                    self.global_scope.define(ast.func_name, fn)
                except Exception as e:

                    existing_fn = self.global_scope.resolve(ast.func_name)
                    if (
                        isinstance(existing_fn, ir.Function)
                        and existing_fn.function_type == fn_ty
                    ):

                        print(
                            f"Warning: Extern function '{ast.func_name}' re-declared with compatible signature."
                        )

                        self.global_scope.symbols[ast.func_name] = fn
                    else:
                        raise Exception(
                            f"Error declaring extern function '{ast.func_name}': {e}"
                        ) from e
                return

            elif isinstance(ast, ControlStatement):
                control_type = ast.control_type

                active_loop_scope = self.current_scope.find_loop_scope()
                if active_loop_scope is None:
                    raise Exception(
                        f"'{control_type}' statement found outside of any loop construct."
                    )

                if control_type == "break":
                    if active_loop_scope.loop_end_block is None:
                        raise Exception(
                            "Internal Compiler Error: Loop scope active, but no 'loop_end_block' defined for break."
                        )
                    self.builder.branch(active_loop_scope.loop_end_block)
                elif control_type == "continue":
                    if active_loop_scope.loop_cond_block is None:
                        raise Exception(
                            "Internal Compiler Error: Loop scope active, but no 'loop_cond_block' defined for continue."
                        )
                    self.builder.branch(active_loop_scope.loop_cond_block)
                else:

                    raise Exception(
                        f"Unsupported control statement type: '{control_type}' encountered."
                    )

                return

            elif isinstance(ast, QualifiedAccess):

                lhs = ast.left
                if isinstance(lhs, str) and lhs in self.enum_types:

                    return self.compile_enum_access(ast)
                elif isinstance(lhs, str) and lhs in self.module_aliases:

                    return self.compile_module_access(ast)
                else:
                    raise Exception(f"Unknown qualifier '{lhs}' for {ast.name}")
            elif isinstance(ast, DeleteStatementNode):
                pointer_expr_ast = ast.pointer_expr_ast

                ptr_to_free_llvm = self.compile(pointer_expr_ast)

                if not isinstance(ptr_to_free_llvm.type, ir.PointerType):
                    raise Exception(
                        f"'delete' expects a pointer, got type {ptr_to_free_llvm.type} from expression '{pointer_expr_ast}'."
                    )

                void_ptr_type = ir.IntType(8).as_pointer()
                if ptr_to_free_llvm.type != void_ptr_type:
                    ptr_to_free_llvm_casted = self.builder.bitcast(
                        ptr_to_free_llvm, void_ptr_type, name="ptr_for_free"
                    )
                else:
                    ptr_to_free_llvm_casted = ptr_to_free_llvm

                free_func_type = ir.FunctionType(ir.VoidType(), [void_ptr_type])
                try:
                    free_func = self.module.get_global("free")
                    if (
                        not isinstance(free_func, ir.Function)
                        or free_func.function_type != free_func_type
                    ):
                        raise Exception("free declared with incompatible signature.")
                except KeyError:
                    free_func = ir.Function(self.module, free_func_type, name="free")

                self.builder.call(free_func, [ptr_to_free_llvm_casted])
                return

            elif isinstance(ast, NewExpressionNode):
                alloc_type_ast = ast.alloc_type_ast

                llvm_type_to_allocate = self.convert_type(alloc_type_ast)

                if not isinstance(llvm_type_to_allocate, ir.Type):

                    raise Exception(
                        f"Type '{alloc_type_ast}' specified in 'new' expression did not resolve to a valid LLVM type. "
                        f"Resolved to: {llvm_type_to_allocate} (type: {type(llvm_type_to_allocate)})"
                    )

                if self.data_layout_obj is None:

                    raise Exception(
                        "Compiler's DataLayout object not initialized. "
                        "Cannot determine type size for 'new' expression."
                    )

                try:

                    type_alloc_size_bytes = llvm_type_to_allocate.get_abi_size(
                        self.data_layout_obj
                    )
                except Exception as e:
                    raise Exception(
                        f"Error getting ABI size for type '{llvm_type_to_allocate}' "
                        f"with DataLayout: {e}"
                    )

                if (
                    type_alloc_size_bytes == 0
                    and not llvm_type_to_allocate.is_zero_sized
                ):

                    opaque_info = (
                        f", is_opaque={llvm_type_to_allocate.is_opaque}"
                        if hasattr(llvm_type_to_allocate, "is_opaque")
                        else ""
                    )
                    name_info = (
                        f", name='{llvm_type_to_allocate.name}'"
                        if hasattr(llvm_type_to_allocate, "name")
                        else ""
                    )
                    raise Exception(
                        f"Could not determine a non-zero ABI size for type '{llvm_type_to_allocate}'{name_info}{opaque_info} "
                        "used with 'new'. The type definition might be incomplete."
                    )

                size_arg_llvm = ir.Constant(ir.IntType(64), type_alloc_size_bytes)

                malloc_arg_type = ir.IntType(64)
                malloc_return_type = ir.IntType(8).as_pointer()
                malloc_func_type = ir.FunctionType(
                    malloc_return_type, [malloc_arg_type]
                )

                try:
                    malloc_func = self.module.get_global("malloc")
                    if not isinstance(malloc_func, ir.Function):
                        raise Exception("'malloc' symbol exists but is not a function.")
                    if malloc_func.function_type != malloc_func_type:
                        print(
                            f"Warning: Existing 'malloc' declaration has a different signature "
                            f"({malloc_func.function_type}) than expected ({malloc_func_type}). Re-declaring."
                        )

                        malloc_func = ir.Function(
                            self.module, malloc_func_type, name="malloc"
                        )
                except KeyError:
                    malloc_func = ir.Function(
                        self.module, malloc_func_type, name="malloc"
                    )

                if self.builder is None:
                    raise Exception(
                        "Builder not available for 'new' expression (likely not inside a function)."
                    )
                raw_heap_ptr_llvm = self.builder.call(
                    malloc_func, [size_arg_llvm], name="raw_heap_ptr"
                )

                target_pointer_llvm_type = llvm_type_to_allocate.as_pointer()
                typed_heap_ptr_llvm = self.builder.bitcast(
                    raw_heap_ptr_llvm, target_pointer_llvm_type, name="typed_heap_ptr"
                )

                return typed_heap_ptr_llvm

            elif isinstance(ast, EnumDeclaration):
                enum_name_str = ast.name

                if enum_name_str in self.enum_types:
                    raise Exception(f"Enum type '{enum_name_str}' already declared.")

                llvm_enum_underlying_type = ir.IntType(32)
                self.enum_types[enum_name_str] = llvm_enum_underlying_type

                member_map_for_enum_members_registry = {}

                next_value = 0
                for member_name_str, value_expr_ast in ast.values:
                    member_llvm_const_val = None
                    if value_expr_ast is not None:

                        if not isinstance(value_expr_ast, Literal) or not isinstance(
                            value_expr_ast.value, int
                        ):
                            raise Exception(
                                f"Enum member '{enum_name_str}::{member_name_str}' value must be an integer literal."
                            )
                        assigned_value = value_expr_ast.value
                        member_llvm_const_val = ir.Constant(
                            llvm_enum_underlying_type, assigned_value
                        )
                        next_value = assigned_value + 1
                    else:
                        member_llvm_const_val = ir.Constant(
                            llvm_enum_underlying_type, next_value
                        )
                        next_value += 1

                    member_map_for_enum_members_registry[
                        member_name_str
                    ] = member_llvm_const_val

                self.enum_members[enum_name_str] = member_map_for_enum_members_registry

                if enum_name_str not in self.type_codes:
                    self.type_codes[enum_name_str] = self._next_type_code
                    self._next_type_code += 1

                return
            elif type(ast) == EnumAccess:
                enum_name = ast.enum_name
                member = ast.value

                if enum_name not in self.enum_members:
                    raise Exception(f"Enum '{enum_name}' is not defined.")

                members = self.enum_members[enum_name]
                if member not in members:
                    raise Exception(f"Enum '{enum_name}' has no member '{member}'.")

                return members[member]

            elif isinstance(ast, LogicalOperator):
                left = self.compile(ast.left)

                op = ast.operator
                if op == "&&":

                    entry = self.builder.block
                    true_bb = self.function.append_basic_block("and_true")
                    cont_bb = self.function.append_basic_block("and_cont")

                    self.builder.cbranch(left, true_bb, cont_bb)

                    self.builder.position_at_end(true_bb)
                    right = self.compile(ast.right)
                    self.builder.branch(cont_bb)

                    self.builder.position_at_end(cont_bb)
                    phi = self.builder.phi(left.type, name="andtmp")
                    phi.add_incoming(ir.Constant(left.type, 0), entry)
                    phi.add_incoming(right, true_bb)
                    return phi

                elif op == "||":
                    entry = self.builder.block
                    false_bb = self.function.append_basic_block("or_false")
                    cont_bb = self.function.append_basic_block("or_cont")
                    self.builder.cbranch(left, cont_bb, false_bb)

                    self.builder.position_at_end(false_bb)
                    right = self.compile(ast.right)
                    self.builder.branch(cont_bb)

                    self.builder.position_at_end(cont_bb)
                    phi = self.builder.phi(left.type, name="ortmp")
                    phi.add_incoming(ir.Constant(left.type, 1), entry)
                    phi.add_incoming(right, false_bb)
                    return phi

                elif op == "!":
                    val = self.compile(ast.right)

                    return self.builder.icmp_equal(
                        val, ir.Constant(val.type, 0), name="nottmp"
                    )

                else:
                    raise Exception(f"Unsupported logical operator: {op}")

            elif isinstance(ast, ComparisonOperator):
                left = self.compile(ast.left)
                right = self.compile(ast.right)
                op = ast.operator

                if isinstance(left.type, ir.FloatType) or isinstance(
                    right.type, ir.FloatType
                ):

                    if isinstance(left.type, ir.IntType):
                        left = self.builder.sitofp(left, right.type, name="cvt_fp_l")
                    if isinstance(right.type, ir.IntType):
                        right = self.builder.sitofp(right, left.type, name="cvt_fp_r")

                    mapping = {
                        "==": "oeq",
                        "!=": "one",
                        "<": "olt",
                        "<=": "ole",
                        ">": "ogt",
                        ">=": "oge",
                    }
                    if op not in mapping:
                        raise Exception(f"Unsupported float cmp op: {op}")
                    return self.builder.fcmp_ordered(
                        mapping[op], left, right, name="fcmp"
                    )

                else:

                    mapping = {
                        "==": "eq",
                        "!=": "ne",
                        "<": "slt",
                        "<=": "sle",
                        ">": "sgt",
                        ">=": "sge",
                    }
                    if op not in mapping:
                        raise Exception(f"Unsupported int cmp op: {op}")
                    return self.builder.icmp_signed(op, left, right, name="icmp")

            elif isinstance(ast, UnaryOperator):
                val = self.compile(ast.operand)
                op = ast.operator
                if op == "-":
                    if isinstance(val.type, ir.FloatType):
                        return self.builder.fneg(val, name="fnegtmp")
                    else:
                        return self.builder.neg(val, name="negtmp")
                elif op in ("~",):
                    mask = ir.Constant(val.type, -1)
                    return self.builder.xor(val, mask, name="nottmp")
                else:
                    raise Exception(f"Unsupported unary operator: {op}")

            elif isinstance(ast, PostfixOperator):

                if not isinstance(ast.operand, str):

                    raise Exception(
                        f"Postfix ++/-- currently only supports simple identifiers, got {type(ast.operand)}."
                    )

                var_name = ast.operand

                var_ptr = self.current_scope.resolve(var_name)

                if var_ptr is None:
                    raise Exception(
                        f"Undeclared variable '{var_name}' for postfix {ast.operator}."
                    )

                if not isinstance(var_ptr, (ir.AllocaInstr, ir.GlobalVariable)):
                    raise Exception(
                        f"Symbol '{var_name}' is not a variable pointer, cannot apply postfix {ast.operator}."
                    )

                if isinstance(var_ptr, ir.GlobalVariable) and var_ptr.global_constant:
                    raise Exception(
                        f"Cannot apply postfix {ast.operator} to global constant '{var_name}'."
                    )

                old_val = self.builder.load(var_ptr, name=f"{var_name}_old_val")

                one_const = None
                if isinstance(old_val.type, ir.IntType):
                    one_const = ir.Constant(old_val.type, 1)
                elif isinstance(old_val.type, ir.FloatType):
                    one_const = ir.Constant(old_val.type, 1.0)

                else:
                    raise Exception(
                        f"Postfix {ast.operator} not supported for type {old_val.type} of variable '{var_name}'."
                    )

                new_val = None
                if ast.operator == "++":
                    if isinstance(old_val.type, ir.FloatType):
                        new_val = self.builder.fadd(
                            old_val, one_const, name=f"{var_name}_inc_val"
                        )

                    else:
                        new_val = self.builder.add(
                            old_val, one_const, name=f"{var_name}_inc_val"
                        )
                elif ast.operator == "--":
                    if isinstance(old_val.type, ir.FloatType):
                        new_val = self.builder.fsub(
                            old_val, one_const, name=f"{var_name}_dec_val"
                        )

                    else:
                        new_val = self.builder.sub(
                            old_val, one_const, name=f"{var_name}_dec_val"
                        )
                else:

                    raise Exception(f"Unknown postfix operator: {ast.operator}")

                self.builder.store(new_val, var_ptr)

                return old_val

            elif isinstance(ast, AdditiveOperator):
                left = self.compile(ast.left)
                right = self.compile(ast.right)
                if left.type != right.type:
                    raise Exception(
                        f"Type mismatch in + operands: {left.type} vs {right.type}"
                    )
                if isinstance(left.type, ir.FloatType):
                    return self.builder.fadd(left, right, name="faddtmp")
                else:
                    return self.builder.add(left, right, name="addtmp")

            elif isinstance(ast, MultiplicativeOperator):
                left = self.compile(ast.left)
                right = self.compile(ast.right)
                if left.type != right.type:
                    raise Exception(
                        f"Type mismatch in * operands: {left.type} vs {right.type}"
                    )
                op = ast.operator
                if isinstance(left.type, ir.FloatType):
                    if op == "*":
                        return self.builder.fmul(left, right, name="fmultmp")
                    elif op == "/":
                        return self.builder.fdiv(left, right, name="fdivtmp")
                    else:

                        raise Exception("Floating-point remainder not supported")
                else:
                    if op == "*":
                        return self.builder.mul(left, right, name="multmp")
                    elif op == "/":
                        return self.builder.sdiv(left, right, name="sdivtmp")
                    else:
                        return self.builder.srem(left, right, name="sremtmp")
            elif isinstance(ast, ArrayLiteralNode):

                if not ast.elements:

                    raise Exception(
                        "Cannot compile empty array literal [] as a standalone expression without type context."
                    )

                llvm_elements = []
                for elem_ast in ast.elements:
                    llvm_elements.append(self.compile(elem_ast))

                if not llvm_elements:
                    raise Exception(
                        "Internal error: llvm_elements list is empty after compiling array literal elements."
                    )

                llvm_element_type = llvm_elements[0].type
                array_len = len(llvm_elements)

                final_constant_elements = []
                for i, elem_llvm_val in enumerate(llvm_elements):
                    if not isinstance(elem_llvm_val, ir.Constant):
                        raise Exception(
                            f"Array literal element {i} ('{ast.elements[i]}') did not compile to a constant value. Got {type(elem_llvm_val)}."
                        )

                    if elem_llvm_val.type != llvm_element_type:

                        raise Exception(
                            f"Array literal has inconsistent element types. "
                            f"Expected type {llvm_element_type} (from first element), "
                            f"but element {i} ('{ast.elements[i]}') has type {elem_llvm_val.type}."
                        )
                    final_constant_elements.append(elem_llvm_val)

                llvm_array_type = ir.ArrayType(llvm_element_type, array_len)
                return ir.Constant(llvm_array_type, final_constant_elements)

            elif type(ast) == Program:

                for node in ast.statements:
                    self.compile(node)
            elif type(ast) == list:

                for node in ast:
                    self.compile(node)
            elif ast == [None]:
                self.builder.ret_void()
            elif isinstance(ast, str):
                var_name = ast
                resolved_symbol = self.current_scope.resolve(var_name)
                if isinstance(resolved_symbol, (ir.AllocaInstr, ir.GlobalVariable)):
                    return self.builder.load(resolved_symbol, name=var_name + "_val")
                return self.get_variable(ast)
            else:
                breakpoint()
                raise Exception(f"Unsupported AST node type: {type(ast)}")
        return self.module

    def generate_ir(self):
        return str(self.module)

    def generate_object_code(self, output_filename: str = "output.o"):
        """
        Compiles the LLVM IR in self.module to a platform-specific object file.
        """
        print(f"--- Generating Object Code: {output_filename} ---")
        final_ir_str = str(self.module)

        try:
            llvm_module_parsed = binding.parse_assembly(final_ir_str)
            llvm_module_parsed.verify()
        except RuntimeError as e:
            print("LLVM IR Parsing Error during object code generation:")
            print(final_ir_str)
            raise Exception(f"Failed to parse LLVM IR: {e}")

        if self.target_machine is None:

            raise Exception(
                "TargetMachine not initialized. Cannot generate object code."
            )

        try:
            with open(output_filename, "wb") as f:
                f.write(self.target_machine.emit_object(llvm_module_parsed))
            print(f"Object file '{output_filename}' generated successfully.")
        except Exception as e:
            raise Exception(f"Failed to emit object file '{output_filename}': {e}")

    def generate_executable(
        self,
        output_executable_name: str = None,
        source_object_filename: str = "output.o",
        keep_object_file: bool = False,
    ):
        """
        Generates an object file (if needed) and then links it into an executable
        using an external linker (clang or gcc).
        """
        if output_executable_name is None:

            if platform.system() == "Windows":
                output_executable_name = "output.exe"
            else:
                output_executable_name = "output"

        print(f"--- Generating Executable: {output_executable_name} ---")

        if not os.path.exists(source_object_filename):
            print(
                f"Object file '{source_object_filename}' not found. Generating it first..."
            )
            self.generate_object_code(output_filename=source_object_filename)
        elif source_object_filename == "output.o" and not os.path.exists("output.o"):

            self.generate_object_code(output_filename="output.o")
            source_object_filename = "output.o"

        linker_cmd = None
        c_libraries_to_link = []

        for lib_name_in_popo in self.imported_libs:
            if lib_name_in_popo.lower() == "m":
                if platform.system() != "Windows":
                    c_libraries_to_link.append("-lm")

        linker_flags = []
        if platform.system() == "Windows":

            linkers_to_try = ["clang", "gcc"]

        elif platform.system() == "Darwin":
            linkers_to_try = ["clang", "gcc"]
            linker_flags.extend(["-L/usr/local/lib"])

        else:
            linkers_to_try = ["clang", "gcc"]

        for linker in linkers_to_try:
            try:

                subprocess.check_output([linker, "--version"], stderr=subprocess.STDOUT)
                linker_cmd_list = (
                    [linker, source_object_filename]
                    + linker_flags
                    + c_libraries_to_link
                    + ["-o", output_executable_name]
                )

                linker_cmd = " ".join(linker_cmd_list)
                print(f"Attempting to link using: {linker_cmd}")

                proc = subprocess.Popen(
                    linker_cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                stdout, stderr = proc.communicate()

                if proc.returncode == 0:
                    print(
                        f"Executable '{output_executable_name}' generated successfully using {linker}."
                    )
                    break
                else:
                    print(
                        f"Linking with {linker} failed. Return code: {proc.returncode}"
                    )
                    print(f"Stdout:\n{stdout.decode(errors='replace')}")
                    print(f"Stderr:\n{stderr.decode(errors='replace')}")
                    linker_cmd = None
            except FileNotFoundError:
                print(f"Linker '{linker}' not found in PATH. Trying next...")
                linker_cmd = None
            except subprocess.CalledProcessError as e:
                print(f"Checking version of '{linker}' failed: {e}")
                linker_cmd = None

        if linker_cmd is None:
            raise Exception(
                f"Failed to link object file. No suitable linker (clang or gcc) found or linking failed. "
                f"Ensure a C compiler (clang or gcc) is installed and in your system PATH."
            )

        if not keep_object_file and os.path.exists(source_object_filename):
            try:
                os.remove(source_object_filename)
                print(f"Removed temporary object file: {source_object_filename}")
            except OSError as e:
                print(
                    f"Warning: Could not remove object file '{source_object_filename}': {e}"
                )

    def runwithjit(self, entry_function_name="main"):

        print(str(self.module))
        for lib in self.imported_libs:
            real = resolve_c_library(lib)
            if not os.path.exists(real) and os.path.isabs(real):
                raise Exception(f"Cannot locate C library '{lib}' (tried '{real}')")
            binding.load_library_permanently(real)

        llvm_module = binding.parse_assembly(str(self.module))
        llvm_module.verify()

        target_machine = binding.Target.from_default_triple().create_target_machine()
        engine = binding.create_mcjit_compiler(llvm_module, target_machine)

        engine.finalize_object()
        engine.run_static_constructors()

        if self.main_function is None:
            raise Exception("No 'main' function found to JIT.")
        func_ptr = engine.get_function_address(self.main_function.name)
        func = ctypes.CFUNCTYPE(None)(func_ptr)
        func()

    def shutdown(self):
        binding.shutdown()


if __name__ == "__main__":
    prs = argparse.ArgumentParser(description="Popo Compiler")
    prs.add_argument("input", type=str, help="Input Popo file")

    prs.add_argument(
        "-o", "--output", type=str, help="Name of the output executable file"
    )
    prs.add_argument(
        "--obj", action="store_true", help="Generate object code only (output.o)"
    )
    prs.add_argument(
        "-O", "--optimization-level", help="LLVM Optimization level", type=int
    )
    prs.add_argument(
        "-C", "--codemodel", help="LLVM CodeModel (default, small,...)", type=str
    )
    prs.add_argument(
        "--keep-obj",
        action="store_true",
        help="Keep intermediate object file when generating executable",
    )

    prs.add_argument(
        "-r", "--run", action="store_true", help="Run the program using JIT"
    )
    prs.add_argument(
        "--ir", "-i", action="store_true", help="Generate and print LLVM IR code"
    )

    args = prs.parse_args()

    with open(args.input, "r") as f:
        code = f.read()

    print("Parsing code...")
    ast = parse_code(code)
    if ast is None:
        print("Parsing failed, AST is None.")
        exit(1)
    print("Parsing successful.")

    compiler = Compiler(args.optimization_level, args.codemodel, args.run)

    print("Compiling AST to LLVM IR...")
    try:
        compiler.compile(ast)
        print("LLVM IR generation successful.")
    except Exception as e:
        print(f"Error during compilation to LLVM IR: {e}")

        exit(1)

    if args.ir:
        print("--- Generated LLVM IR ---")
        print(compiler.generate_ir())
        print("-------------------------")

    if args.obj:
        obj_file = args.output if args.output else "output.o"
        if args.output and not (
            args.output.endswith(".o") or args.output.endswith(".obj")
        ):
            obj_file = args.output + ".o"
        try:
            compiler.generate_object_code(output_filename=obj_file)
        except Exception as e:
            print(f"Error generating object code: {e}")
            exit(1)
    elif args.run:
        try:
            print("Running with JIT...")
            compiler.runwithjit("main")
        except Exception as e:
            print(f"Error during JIT execution: {e}")

            exit(1)
    else:
        exe_name = args.output
        obj_name = "temp_output.o"

        try:
            compiler.generate_executable(
                output_executable_name=exe_name,
                source_object_filename=obj_name,
                keep_object_file=args.keep_obj,
            )
        except Exception as e:
            print(f"Error generating executable: {e}")

            exit(1)

    compiler.shutdown()
    print("Compilation process finished.")
