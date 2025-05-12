import ply.yacc as yacc
from PopoLexer import tokens, lexer


class Node:
    pass


class BuiltinFunction(Node):
    def __init__(self):
        pass


__print_function = BuiltinFunction()
__print_function.call = lambda evaluated_value: print(evaluated_value[0])

__MBuiltins__ = {"print_function": __print_function}


class Program(Node):
    def __init__(self, statements):
        self.statements = statements

    def __repr__(self):
        t = ""
        for i in self.statements:
            t += str(i) + "\n\n"
        return f"Program({t})"


class ImportC(Node):
    def __init__(self, path_or_name):
        self.path_or_name = path_or_name

    def __repr__(self):
        return f"ImportC({self.path_or_name})"


class ImportModule(Node):
    def __init__(self, path, alias=None):

        self.path = path.strip('"')
        self.alias = alias

    def __repr__(self):
        return f"ImportModule(path={self.path}, alias={self.alias})"


class ModuleAccess(Node):
    def __init__(self, alias, name):
        self.alias = alias
        self.name = name

    def __repr__(self):
        return f"ModuleAccess({self.alias}.{self.name})"


class GenericTypeParameterDeclarationNode(Node):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"GenericTypeParameterDeclarationNode(Name='{self.name}')"


class TypeParameterNode(Node):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"TypeParameterNode(Name='{self.name}')"


class SizeofNode(Node):
    def __init__(self, target_ast_node):

        self.target_ast_node = target_ast_node

    def __repr__(self):
        return f"SizeofNode(Target={self.target_ast_node})"


class Extern(Node):
    def __init__(self, func_name, func_args, func_return_type):
        self.func_name = func_name
        self.func_args = func_args
        self.func_return_type = func_return_type

    def __repr__(self):
        return f"Extern(Name: {self.func_name}, Args: {self.func_args}, Return Type: {self.func_return_type})"


class AsPtrNode(Node):
    def __init__(self, expression_ast):
        self.expression_ast = expression_ast

    def __repr__(self):
        return f"AsPtrNode(Expression={self.expression_ast})"


class QualifiedAccess(Node):
    def __init__(self, left, name):
        self.left = left
        self.name = name

    def __repr__(self):
        return f"QualifiedAccess({self.left}, {self.name})"


class Literal(Node):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"Literal({self.value})"


class Assignment(Node):
    def __init__(self, identifier, operator, value):
        self.identifier = identifier
        self.operator = operator
        self.value = value

    def __repr__(self):
        return f"Assignment(ID: {self.identifier}, Operator: {self.operator}, Value: {self.value})"


class VariableDeclaration(Node):
    def __init__(self, is_mutable, identifier, var_type, value):
        self.is_mutable = is_mutable
        self.identifier = identifier
        self.type = var_type
        self.value = value

    def __repr__(self):
        return f"VariableDeclaration(Mutable: {self.is_mutable}, ID: {self.identifier}, Type: {self.type}, Value: {self.value})"


class FunctionDeclaration(Node):
    def __init__(
        self, name, params, return_type, body, is_static=False, type_parameters=None
    ):
        self.name = name
        self.params = params
        self.return_type = return_type
        self.body = body
        self.is_static = is_static
        self.type_parameters = type_parameters if type_parameters else []

    def __repr__(self):
        static_str = "STATIC " if self.is_static else ""
        type_param_repr = ""
        if self.type_parameters:
            type_param_repr = f"<{', '.join(self.type_parameters)}>"
        return f"{static_str}FunctionDeclaration{type_param_repr}(Name: {self.name}, Params: {self.params}, Return Type: {self.return_type}, Body: {self.body})"


class Parameter(Node):
    def __init__(self, identifier, var_type):
        self.identifier = identifier
        self.var_type = var_type

    def __repr__(self):
        return f"Parameter(ID: {self.identifier}, Type: {self.var_type})"


class MacroDeclaration(Node):
    def __init__(self, name, params, body):
        self.name = name
        self.params = params or []
        self.body = body

    def __repr__(self):
        return f"MacroDeclaration({self.name}, {self.params}, {self.body})"


class MacroCall(Node):
    def __init__(self, name, args):
        self.name = name
        self.args = args or []

    def __repr__(self):
        return f"MacroCall({self.name}, {self.args})"


class SpecialDeclaration(Node):
    def __init__(self, name, body):
        self.name = name
        self.body = body

    def __repr__(self):
        return f"SpecialDeclaration(Name: {self.name}, Body: {self.body})"


class FunctionCall(Node):
    def __init__(self, call_name, params):
        self.call_name = call_name
        self.params = params

    def __repr__(self):
        return f"FunctionCall(Name: {self.call_name}, Params: {self.params})"


class StructMethodCall(Node):
    def __init__(self, struct_name, method_name, params):
        self.struct_name = struct_name
        self.method_name = method_name
        self.params = params

    def __repr__(self):
        return f"StructMethodCall(Struct: {self.struct_name}, Method: {self.method_name}, Params: {self.params})"


class StructDeclaration(Node):
    def __init__(self, name, members, methods):
        self.name = name
        self.members = members
        self.methods = methods

    def __repr__(self):
        return f"StructDeclaration(Name: {self.name}, Members: {self.members}, Methods: {self.methods})"

    def get_member_by_name(self, name):
        for member in self.members:
            if member.identifier == name:
                return member
        return None


class FieldAssignment(Node):
    def __init__(self, identifier, value):
        self.identifier = identifier
        self.value = value

    def __repr__(self):
        return f"FieldAssignment(ID: {self.identifier}, Value: {self.value})"


class StructMember(Node):
    def __init__(self, identifier, var_type):
        self.identifier = identifier
        self.var_type = var_type

    def __repr__(self):
        return f"StructMember(ID: {self.identifier}, Type: {self.var_type})"


class StructInstantiation(Node):
    def __init__(self, struct_name, field_assignments):
        self.struct_name = struct_name
        self.field_assignments = field_assignments

    def __repr__(self):
        return f"StructInstantiation(Name: {self.struct_name}, Fields: {self.field_assignments})"


class MemberAccess(Node):
    def __init__(self, struct_name, member_name):
        self.struct_name = struct_name
        self.member_name = member_name

    def __repr__(self):
        return f"MemberAccess(Struct: {self.struct_name}, Member: {self.member_name})"


class TypeConv(Node):
    def __init__(self, target_type, expr):
        self.target_type = target_type
        self.expr = expr

    def __repr__(self):
        return f"TypeConv(To: {self.target_type}, Expr: {self.expr})"


class TypeAnnotation(Node):
    def __init__(self, base: str, bits: int):
        self.base = base
        self.bits = bits

    def __repr__(self):
        return f"TypeAnnotation({self.base}, {self.bits})"


class NewExpressionNode(Node):
    def __init__(self, alloc_type_ast):
        self.alloc_type_ast = alloc_type_ast

    def __repr__(self):
        return f"NewExpressionNode(AllocType={self.alloc_type_ast})"


class DeleteStatementNode(Node):
    def __init__(self, pointer_expr_ast):
        self.pointer_expr_ast = pointer_expr_ast

    def __repr__(self):
        return f"DeleteStatementNode(PointerExpr={self.pointer_expr_ast})"


class IfStatement(Node):
    def __init__(self, condition, body, elifs, else_body):
        self.condition = condition
        self.body = body
        self.elifs = elifs
        self.else_body = else_body

    def __repr__(self):
        return f"IfStatement(Condition: {self.condition}, Body: {self.body}, Elifs: {self.elifs}, Else: {self.else_body})"


class ForLoop(Node):
    def __init__(self, init, condition, increment, body):
        self.init = init
        self.condition = condition
        self.increment = increment
        self.body = body

    def __repr__(self):
        return f"ForLoop(Init: {self.init}, Condition: {self.condition}, Increment: {self.increment}, Body: {self.body})"


class WhileLoop(Node):
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body

    def __repr__(self):
        return f"WhileLoop(Condition: {self.condition}, Body: {self.body})"


class ForeachLoop(Node):
    def __init__(self, identifier, var_type, iterable, body):
        self.identifier = identifier
        self.var_type = var_type
        self.iterable = iterable
        self.body = body

    def __repr__(self):
        return f"ForeachLoop(ID: {self.identifier}, Type: {self.var_type}, Iterable: {self.iterable}, Body: {self.body})"


class ControlStatement(Node):
    def __init__(self, control_type):
        self.control_type = control_type

    def __repr__(self):
        return f"ControlStatement(Type: {self.control_type})"


class EnumDeclaration(Node):
    def __init__(self, name, values):
        self.name = name
        self.values = values

    def __repr__(self):
        return f"EnumDeclaration(Name: {self.name}, Values: {self.values})"


class EnumAccess(Node):
    def __init__(self, enum_name, value):
        self.enum_name = enum_name
        self.value = value

    def __repr__(self):
        return f"EnumAccess(Enum: {self.enum_name}, Value: {self.value})"


class LogicalOperator(Node):
    def __init__(self, operator, left, right):
        self.operator = operator
        self.left = left
        self.right = right

    def __repr__(self):
        return f"LogicalOperator(Operator: {self.operator}, Left: {self.left}, Right: {self.right})"


class ComparisonOperator(Node):
    def __init__(self, operator, left, right):
        self.operator = operator
        self.left = left
        self.right = right

    def __repr__(self):
        return f"ComparisonOperator(Operator: {self.operator}, Left: {self.left}, Right: {self.right})"


class UnaryOperator(Node):
    def __init__(self, operator, operand):
        self.operator = operator
        self.operand = operand

    def __repr__(self):
        return f"UnaryOperator(Operator: {self.operator}, Operand: {self.operand})"


class PostfixOperator(Node):
    def __init__(self, operator, operand):
        self.operator = operator
        self.operand = operand

    def __repr__(self):
        return f"PostfixOperator(Operator: {self.operator}, Operand: {self.operand})"


class AdditiveOperator(Node):
    def __init__(self, operator, left, right):
        self.operator = operator
        self.left = left
        self.right = right

    def __repr__(self):
        return f"AdditiveOperator(Operator: {self.operator}, Left: {self.left}, Right: {self.right})"


class MultiplicativeOperator(Node):
    def __init__(self, operator, left, right):
        self.operator = operator
        self.left = left
        self.right = right

    def __repr__(self):
        return f"MultiplicativeOperator(Operator: {self.operator}, Left: {self.left}, Right: {self.right})"


class TypeOf(Node):
    def __init__(self, expr):
        self.expr = expr

    def __repr__(self):
        return f"TypeOf({self.expr})"


class ArrayTypeNode(Node):
    def __init__(self, element_type, size_expr=None):
        self.element_type = element_type
        self.size_expr = size_expr

    def __repr__(self):
        size_str = f", SizeExpr={self.size_expr}" if self.size_expr else ""
        return f"ArrayTypeNode(ElementType={self.element_type}{size_str})"


class PointerTypeNode(Node):
    def __init__(self, pointee_type):
        self.pointee_type = pointee_type

    def __repr__(self):
        return f"PointerTypeNode(PointeeType={self.pointee_type})"


class ArrayLiteralNode(Node):
    def __init__(self, elements):
        self.elements = elements

    def __repr__(self):
        return f"ArrayLiteralNode(Elements={self.elements})"


class ArrayIndexNode(Node):
    def __init__(self, array_expr, index_expr):
        self.array_expr = array_expr
        self.index_expr = index_expr

    def __repr__(self):
        return f"ArrayIndexNode(Array={self.array_expr}, Index={self.index_expr})"


class AddressOfNode(Node):
    def __init__(self, expression):
        self.expression = expression

    def __repr__(self):
        return f"AddressOfNode({self.expression})"


class DereferenceNode(Node):
    def __init__(self, expression):
        self.expression = expression

    def __repr__(self):
        return f"DereferenceNode({self.expression})"


class ReturnStatement(Node):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"ReturnStatement(Value: {self.value})"


def p_program(p):
    "program : statements"
    p[0] = Program(p[1])


def p_statements_single(p):
    """statements : statement"""
    p[0] = [p[1]]


def p_statements_multiple(p):
    "statements : statements statement"
    p[0] = p[1] + [p[2]]


def p_statement(p):
    """statement : macro_declaration
    | variable_declaration
    | function_declaration
    | struct_declaration
    | delete_statement
    | enum_declaration
    | special_programming
    | if_statement
    | loop_statement
    | control_statement
    | return_statement
    | expression_statement
    | import_c_statement
    | import_statement
    | extern_statement
    | empty"""
    p[0] = p[1]


def p_import_statement(p):
    """import_statement : IMPORT STRING_LITERAL SEMICOLON
    | IMPORT STRING_LITERAL AS IDENTIFIER SEMICOLON"""
    if len(p) == 4:
        p[0] = ImportModule(path=p[2])
    else:
        p[0] = ImportModule(path=p[2], alias=p[4])


def p_import_c_statement(p):
    """import_c_statement : IMPORT_C STRING_LITERAL SEMICOLON"""
    p[0] = ImportC(p[2])


def p_extern_statement(p):
    """extern_statement : EXTERN FUN IDENTIFIER LPAREN extern_params_content RPAREN extern_return_type SEMICOLON"""

    param_list, has_va_args = p[5]

    processed_args = param_list
    if has_va_args and param_list and param_list[-1] == "...":
        pass
    elif has_va_args:
        processed_args = (param_list or []) + ["..."]

    p[0] = Extern(func_name=p[3], func_args=processed_args, func_return_type=p[7])


def p_extern_params_content(p):
    """extern_params_content : extern_param_list COMMA ELLIPSIS
    | extern_param_list
    | empty"""
    if len(p) == 4:
        p[0] = (p[1], True)
    elif len(p) == 2:
        if p[1] is None:
            p[0] = ([], False)
        else:
            p[0] = (p[1], False)
    else:
        p[0] = ([], False)


def p_extern_param_list_single(p):
    """extern_param_list : extern_param"""
    p[0] = [p[1]]


def p_extern_param_list_multiple(p):
    """extern_param_list : extern_param_list COMMA extern_param"""
    p[0] = p[1] + [p[3]]


def p_extern_param(p):
    """extern_param : IDENTIFIER COLON LT type GT"""
    p[0] = Parameter(identifier=p[1], var_type=p[4])


def p_extern_return_type(p):
    """extern_return_type : LT type GT
    | NORET"""
    if len(p) == 4:
        p[0] = p[2]
    else:
        p[0] = p[1]


def p_expression_statement(p):
    "expression_statement : expression SEMICOLON"
    p[0] = p[1]


def p_generic_param_list_decl_opt(p):
    """generic_param_list_decl_opt : LT generic_param_list_items GT
    | empty"""
    if len(p) == 4:
        p[0] = p[2]
    else:
        p[0] = []


def p_generic_param_list_items(p):
    """generic_param_list_items : IDENTIFIER
    | generic_param_list_items COMMA IDENTIFIER"""
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]


def p_block(p):
    """block : LBRACE statements RBRACE"""
    p[0] = p[2]


def p_if_statement(p):
    """if_statement : IF LPAREN expression RPAREN block else_clause_opt"""

    p[0] = IfStatement(
        condition=p[3], body=p[5], elifs=p[6]["elifs"], else_body=p[6]["else"]
    )


def p_else_clause_opt(p):
    """else_clause_opt : else_if_list else_block_opt
    | else_block_opt"""
    if len(p) == 3:
        p[0] = {"elifs": p[1], "else": p[2]}
    else:
        p[0] = {"elifs": [], "else": p[1]}


def p_else_if_list(p):
    """else_if_list : ELSEIF LPAREN expression RPAREN block
    | else_if_list ELSEIF LPAREN expression RPAREN block"""
    if len(p) == 6:
        p[0] = [(p[3], p[5])]
    else:
        p[0] = p[1] + [(p[4], p[6])]


def p_else_block_opt(p):
    """else_block_opt : ELSE block
    | empty"""
    if len(p) == 3:
        p[0] = p[2]
    else:
        p[0] = None


def p_loop_statement(p):
    """loop_statement : while_loop
    | for_loop
    | foreach_loop"""
    p[0] = p[1]


def p_while_loop(p):
    "while_loop : WHILE LPAREN expression RPAREN block"
    p[0] = WhileLoop(p[3], p[5])


def p_for_loop(p):
    "for_loop : FOR LPAREN variable_declaration expression SEMICOLON expression RPAREN block"
    p[0] = ForLoop(init=p[3], condition=p[4], increment=p[6], body=p[8])


def p_foreach_loop(p):
    "foreach_loop : FOREACH IDENTIFIER LT type GT IN expression block"
    p[0] = ForeachLoop(p[2], p[4], p[7], p[8])


def p_control_statement(p):
    """control_statement : BREAK SEMICOLON
    | CONTINUE SEMICOLON"""
    p[0] = ControlStatement(p[1])


def p_return_statement(p):
    """return_statement : RETURN expression SEMICOLON
    | RETURN SEMICOLON"""
    if len(p) == 4:
        p[0] = ReturnStatement(p[2])
    else:
        p[0] = ReturnStatement(None)


def p_struct_declaration(p):
    "struct_declaration : STRUCT IDENTIFIER LBRACE struct_body RBRACE"
    p[0] = StructDeclaration(
        name=p[2], members=p[4]["members"], methods=p[4]["methods"]
    )


def p_struct_body(p):
    """struct_body : struct_members struct_methods
    | struct_members
    | struct_methods
    | empty"""
    if len(p) == 3:
        p[0] = {"members": p[1], "methods": p[2]}
    elif len(p) == 2 and isinstance(p[1], list):
        p[0] = {"members": p[1], "methods": []}
    elif len(p) == 2 and isinstance(p[1], dict):
        p[0] = {"members": [], "methods": p[1]["methods"]}
    elif len(p) == 2 and p[1] is None:
        p[0] = {"members": [], "methods": []}
    else:
        p[0] = {"members": [], "methods": p[1]}


def p_struct_methods(p):
    """struct_methods : struct_methods function_declaration
    | function_declaration"""
    if len(p) == 3:
        p[0] = p[1] + [p[2]]
    else:
        p[0] = [p[1]]


def p_struct_members_single(p):
    """struct_members : struct_member"""
    p[0] = [p[1]]


def p_struct_members_multiple(p):
    """struct_members : struct_members COMMA struct_member"""
    p[0] = p[1] + [p[3]]


def p_struct_members_trailing_comma(p):
    """struct_members : struct_members COMMA"""
    p[0] = p[1]


def p_struct_member(p):
    """struct_member : IDENTIFIER LT type GT"""
    p[0] = StructMember(p[1], p[3])


def p_enum_declaration(p):
    "enum_declaration : ENUM IDENTIFIER LBRACE enum_values RBRACE"
    p[0] = EnumDeclaration(name=p[2], values=p[4])


def p_enum_values_single(p):
    "enum_values : enum_value"
    p[0] = [p[1]]


def p_enum_values_multiple(p):
    "enum_values : enum_values COMMA enum_value"
    p[0] = p[1] + [p[3]]


def p_enum_value(p):
    """enum_value : IDENTIFIER
    | IDENTIFIER EQUAL expression"""
    if len(p) == 2:
        p[0] = (p[1], None)
    else:
        p[0] = (p[1], p[3])


def p_variable_declaration(p):
    """variable_declaration : mutable_declaration
    | immutable_declaration
    | declared_not_assigned_declaration"""
    p[0] = p[1]


def p_declared_not_assigned_declaration(p):
    """declared_not_assigned_declaration : LET IDENTIFIER LT type GT SEMICOLON
    | BEZ IDENTIFIER LT type GT SEMICOLON
    | CONST IDENTIFIER LT type GT SEMICOLON
    | BETON IDENTIFIER LT type GT SEMICOLON"""
    p[0] = VariableDeclaration(
        is_mutable=True, identifier=p[2], var_type=p[4], value=None
    )


def p_mutable_declaration(p):
    """mutable_declaration : LET IDENTIFIER LT type GT EQUAL expression SEMICOLON
    | BEZ IDENTIFIER LT type GT EQUAL expression SEMICOLON"""
    p[0] = VariableDeclaration(
        is_mutable=True, identifier=p[2], var_type=p[4], value=p[7]
    )


def p_immutable_declaration(p):
    """immutable_declaration : CONST IDENTIFIER LT type GT EQUAL expression SEMICOLON
    | BETON IDENTIFIER LT type GT EQUAL expression SEMICOLON"""
    p[0] = VariableDeclaration(
        is_mutable=False, identifier=p[2], var_type=p[4], value=p[7]
    )


def p_type(p):
    """type : base_type
    | pointer_type
    | array_type"""
    p[0] = p[1]


def p_base_type(p):
    """base_type : IDENTIFIER
    | IDENTIFIER COLONCOLON IDENTIFIER
    | AUTO
    | IDENTIFIER LPAREN INTEGER RPAREN
    | LPAREN type RPAREN"""

    if (
        p.slice[1].type == "LPAREN"
        and p.slice[len(p) - 1].type == "RPAREN"
        and len(p) == 4
    ):

        p[0] = p[2]
    elif len(p) == 2:
        if p[1] == "auto":
            p[0] = "auto"
        else:
            p[0] = p[1]
    elif len(p) == 4 and p.slice[2].type == "COLONCOLON":
        p[0] = ModuleAccess(p[1], p[3])
    elif (
        len(p) == 5 and p.slice[1].type == "IDENTIFIER" and p.slice[2].type == "LPAREN"
    ):
        base = p[1]
        bits = p[3]
        p[0] = TypeAnnotation(base, bits)


def p_pointer_type(p):
    """pointer_type : AMPERSAND type"""
    p[0] = PointerTypeNode(p[2])


def p_array_type(p):
    """array_type : LBRACKET type RBRACKET
    | LBRACKET type COMMA expression RBRACKET"""

    if len(p) == 4:
        p[0] = ArrayTypeNode(element_type=p[2], size_expr=None)
    elif len(p) == 6:

        p[0] = ArrayTypeNode(element_type=p[2], size_expr=p[4])


def p_type_param(p):
    """type : IDENTIFIER LPAREN INTEGER RPAREN"""

    base = p[1]
    bits = int(p[3])
    p[0] = TypeAnnotation(base, bits)


def p_function_declaration(p):
    """function_declaration : FUN IDENTIFIER generic_param_list_decl_opt LPAREN params RPAREN return_type LBRACE statements RBRACE
    | STATIC FUN IDENTIFIER generic_param_list_decl_opt LPAREN params RPAREN return_type LBRACE statements RBRACE"""
    is_static_decl = p.slice[1].type == "STATIC"

    name_idx = 2
    generic_params_idx = 3
    params_lparen_idx = 4

    if is_static_decl:
        name_idx = 3
        generic_params_idx = 4
        params_lparen_idx = 5

    func_name = p[name_idx]
    type_params_list = p[generic_params_idx]

    actual_params = p[params_lparen_idx + 1]
    return_type = p[params_lparen_idx + 3]
    body = p[params_lparen_idx + 5]

    body_stmts = p[params_lparen_idx + 5]

    p[0] = FunctionDeclaration(
        name=func_name,
        params=actual_params,
        return_type=return_type,
        body=body_stmts,
        is_static=is_static_decl,
        type_parameters=type_params_list,
    )


def p_params(p):
    """params : param_list
    | empty"""
    p[0] = p[1] if p[1] is not None else []


def p_param_list_single(p):
    "param_list : param"
    p[0] = [p[1]]


def p_param_list_multiple(p):
    "param_list : param_list COMMA param"
    p[0] = p[1] + [p[3]]


def p_param(p):
    "param : IDENTIFIER COLON LT type GT"
    p[0] = Parameter(identifier=p[1], var_type=p[4])


def p_return_type(p):
    """return_type : LT type GT
    | NORET"""
    if len(p) == 4:
        p[0] = p[2]
    else:
        p[0] = p[1]


def p_special_programming(p):
    """special_programming : AT IDENTIFIER LBRACE statements RBRACE"""
    p[0] = SpecialDeclaration(p[2], p[4])


def p_macro_declaration(p):
    """macro_declaration : AT MACRO IDENTIFIER LPAREN macro_param_list RPAREN LBRACE statements RBRACE"""
    p[0] = MacroDeclaration(name=p[3], params=p[5], body=p[8])


def p_primary_macro_call(p):
    """primary : DOLLAR IDENTIFIER LPAREN arguments RPAREN"""

    p[0] = MacroCall(p[2], p[4] if p[4] is not None else [])


def p_reserved_kw_tconv(p):
    """reserved_kw_tconv : STD_CONV"""
    p[0] = p[1]


def p_primary_type_conv(p):
    """primary : reserved_kw_tconv LT type GT LPAREN expression RPAREN"""

    name = p[1]
    to_ty = p[3]
    expr = p[6]
    p[0] = TypeConv(to_ty, expr)


def p_macro_param_list(p):
    """macro_param_list : IDENTIFIER
    | macro_param_list COMMA IDENTIFIER
    | empty"""
    if len(p) == 2:
        p[0] = [] if p[1] is None else [p[1]]
    else:
        p[0] = p[1] + [p[3]]


def p_expression(p):
    """expression : assignment_expression"""
    p[0] = p[1]


def p_assignment_expression(p):
    """assignment_expression : conditional_expression
    | unary assignment_operator assignment_expression
    """

    if len(p) == 2:
        p[0] = p[1]
    else:

        p[0] = Assignment(p[1], p[2], p[3])


def p_assignment_operator(p):
    """assignment_operator : EQUAL
    | PLUSEQUAL
    | MINUSEQUAL
    | MULTEQUAL
    | DIVEQUAL"""
    p[0] = p[1]


def p_conditional_expression(p):
    "conditional_expression : logical_or"
    p[0] = p[1]


def p_logical_or(p):
    """logical_or : logical_and
    | logical_or OR logical_and"""
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = LogicalOperator("||", p[1], p[3])


def p_logical_and(p):
    """logical_and : equality
    | logical_and AND equality"""
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = LogicalOperator("&&", p[1], p[3])


def p_equality(p):
    """equality : comparison
    | equality EQEQ comparison
    | equality NOTEQ comparison"""
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = ComparisonOperator(p[2], p[1], p[3])


def p_comparison(p):
    """comparison : additive
    | comparison LT additive
    | comparison GT additive
    | comparison LTEQ additive
    | comparison GTEQ additive"""
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = ComparisonOperator(p[2], p[1], p[3])


def p_additive(p):
    """additive : multiplicative
    | additive PLUS multiplicative
    | additive MINUS multiplicative"""
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = AdditiveOperator(p[2], p[1], p[3])


def p_multiplicative(p):
    """multiplicative : unary
    | multiplicative MULT unary
    | multiplicative DIV unary
    | multiplicative MOD unary"""
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = MultiplicativeOperator(p[2], p[1], p[3])


def p_unary(p):
    """unary : PLUS unary
    | MINUS unary %prec UMINUS
    | AMPERSAND unary %prec ADDRESSOF_PREC
    | MULT unary %prec DEREFERENCE_PREC
    | NOT unary
    | postfix"""
    if len(p) == 3:
        op_token_type = p.slice[1].type
        op_value = p[1]

        if op_token_type == "PLUS":
            p[0] = p[2]
        elif op_token_type == "MINUS":
            p[0] = UnaryOperator(op_value, p[2])
        elif op_token_type == "AMPERSAND":
            p[0] = AddressOfNode(p[2])
        elif op_token_type == "MULT":
            p[0] = DereferenceNode(p[2])
        elif op_token_type == "NOT":

            p[0] = UnaryOperator(op_value, p[2])
        else:
            raise SyntaxError(f"Unknown unary operator token type: {op_token_type}")
    else:
        p[0] = p[1]


def p_postfix(p):
    """
    postfix : primary
            | postfix postfix_suffix
    """
    if len(p) == 2:
        p[0] = p[1]
    else:
        left = p[1]
        kind, data = p[2]

        if kind == "inc" or kind == "dec":
            p[0] = PostfixOperator(data, left)
        elif kind == "call":
            args = data
            p[0] = FunctionCall(left, args)
        elif kind == "method":
            method_name, args = data
            p[0] = StructMethodCall(left, method_name, args)
        elif kind == "field":
            p[0] = MemberAccess(left, data)
        elif kind == "qual":

            if isinstance(left, str):
                p[0] = ModuleAccess(left, data)
            elif isinstance(left, ModuleAccess):

                new_alias = f"{left.alias}::{left.name}" if left.name else left.alias
                p[0] = ModuleAccess(new_alias, data)

            else:
                raise SyntaxError(f"Invalid left-hand side for COLONCOLON: {left}")
        elif kind == "index":
            index_expr = data
            p[0] = ArrayIndexNode(left, index_expr)
        else:
            raise SyntaxError(f"Unknown postfix kind: {kind}")


def p_postfix_suffix(p):
    """
    postfix_suffix : INCREMENT
                   | DECREMENT
                   | LPAREN arguments RPAREN
                   | DOT IDENTIFIER LPAREN arguments RPAREN
                   | DOT IDENTIFIER
                   | COLONCOLON IDENTIFIER
                   | LBRACKET expression RBRACKET"""
    tok = p.slice[1].type
    if tok == "INCREMENT":
        p[0] = ("inc", "++")
    elif tok == "DECREMENT":
        p[0] = ("dec", "--")
    elif tok == "LPAREN":
        p[0] = ("call", p[2])
    elif tok == "DOT":
        if len(p) == 3:
            p[0] = ("field", p[2])
        else:
            p[0] = ("method", (p[2], p[4]))
    elif tok == "COLONCOLON":
        p[0] = ("qual", p[2])
    elif tok == "LBRACKET":
        p[0] = ("index", p[2])

    else:

        raise SyntaxError(
            f"Unknown postfix_suffix starting with token {tok} value {p[1]}"
        )


def p_primary(p):
    """primary : literal
    | IDENTIFIER
    | LPAREN expression RPAREN
    | IDENTIFIER LBRACE field_assignments RBRACE
    | typeof_expression
    | array_literal
    | new_heap_allocation_expression
    | sizeof_expression
    | as_ptr_expression"""
    if len(p) == 2:
        p[0] = p[1]
    elif p.slice[1].type == "LPAREN":
        p[0] = p[2]
    elif p.slice[1].type == "IDENTIFIER" and p.slice[2].type == "LBRACE":
        p[0] = StructInstantiation(p[1], p[3])

    if len(p) == 2 and p.slice[1].type == "NEW":
        p[0] = p[1]


def p_as_ptr_expression(p):
    """as_ptr_expression : AS_PTR LPAREN expression RPAREN"""
    p[0] = AsPtrNode(expression_ast=p[3])


def p_sizeof_expression(p):
    """sizeof_expression : SIZEOF LPAREN sizeof_target RPAREN"""
    p[0] = SizeofNode(target_ast_node=p[3])


def p_sizeof_target(p):
    """sizeof_target : LT type GT
    | expression
    """
    if len(p) == 4:
        p[0] = p[2]
    else:
        p[0] = p[1]


def p_primary_new_angle_brackets(p):
    """primary : NEW LT type GT"""
    p[0] = NewExpressionNode(alloc_type_ast=p[3])


def p_new_heap_allocation_expression(p):
    """new_heap_allocation_expression : NEW LT type GT"""
    p[0] = NewExpressionNode(alloc_type_ast=p[3])


def p_delete_statement(p):
    """delete_statement : DELETE expression SEMICOLON"""
    p[0] = DeleteStatementNode(pointer_expr_ast=p[2])


def p_array_literal(p):
    """array_literal : LBRACKET arguments RBRACKET"""
    p[0] = ArrayLiteralNode(p[2])


def p_typeof_expression(p):
    """typeof_expression : TYPEOF LPAREN expression RPAREN"""
    p[0] = TypeOf(p[3])


def p_field_assignments_single(p):
    """field_assignments : field_assignment
    | empty"""
    p[0] = [p[1]]


def p_field_assignments_multiple(p):
    "field_assignments : field_assignments COMMA field_assignment"
    p[0] = p[1] + [p[3]]


def p_field_assignment(p):
    "field_assignment : IDENTIFIER COLON expression"
    p[0] = FieldAssignment(p[1], p[3])


def p_arguments(p):
    """arguments : expression_list
    | empty"""
    p[0] = p[1] if p[1] is not None else []


def p_expression_list_single(p):
    "expression_list : expression"
    p[0] = [p[1]]


def p_expression_list_multiple(p):
    "expression_list : expression_list COMMA expression"
    p[0] = p[1] + [p[3]]


def p_literal(p):
    """literal : INTEGER
    | FLOAT
    | STRING_LITERAL
    | CHAR_LITERAL"""
    p[0] = Literal(p[1])


def p_empty(p):
    "empty :"
    p[0] = None


def p_error(p):
    if p:
        print(f"Syntax error at '{p.value}' (line {p.lineno})")
    else:
        print("Syntax error at EOF")


precedence = (
    ("right", "EQUAL", "PLUSEQUAL", "MINUSEQUAL", "MULTEQUAL", "DIVEQUAL"),
    ("left", "OR"),
    ("left", "AND"),
    ("nonassoc", "EQEQ", "NOTEQ", "LT", "GT", "LTEQ", "GTEQ"),
    ("left", "PLUS", "MINUS"),
    ("left", "MULT", "DIV", "MOD"),
    ("right", "NOT", "UMINUS", "ADDRESSOF_PREC", "DEREFERENCE_PREC"),
    ("left", "LPAREN", "LBRACKET", "DOT"),
    ("left", "INCREMENT", "DECREMENT"),
)


parser = yacc.yacc()

if __name__ == "__main__":
    test_code = """

    enum Status {
        Ok,
        NotOk,
        ProbablyOk
    }

    struct MyStr{
        f <char>,
        b <int>,

        fun lol() <noret> {
            printf("Ok");
        }

        fun set(self: <MyStr>, new_f: <char>) <MyStr> {
            self.f = new_f;
        }
    }
    
    @macro mymacro (c, a) {
        print(c % a);
    }

    @macro todo () {
    
    }

    fun main() <noret> {
        let x <Status> = Status::Ok;
        let mys <auto> = MyStr{f: 'a', b: 1};
        const y <auto> = 1;

        mymacro:{10, 20};

        while (x >= 10){
            output(x);
        }
        for (let i <int> = 0; i < 10; i++){
           output(i);
        }

        if (x == 1){
            todo:{};
        } elseif (x == 2){
            todo:{};
        } else {
            todo:{};
        }

    }

    fun add(a: <int>, b: <int>) <int> {
        return a + b;
    }
    """

    while True:
        try:
            s = input("Popo > ")
        except EOFError:
            break
        if not s:
            continue
        lexer.input(s)
        ast = parser.parse(s)
        print(ast)
