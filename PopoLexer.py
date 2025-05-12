import ply.lex as lex


tokens = (
    
    'LET', 'BEZ', 'CONST', 'BETON', 'AUTO',
    
    'FUN', 'NORET', 'RETURN',
    
    'STRUCT', 'ENUM',
    
    'MACRO', 'STATIC', 'AT',
    
    'WHILE', 'FOR', 'FOREACH', 'BREAK', 'CONTINUE',
    'IF', 'ELSE', 'ELSEIF', 'IN',
    
    'LT', 'GT', 'EQUAL', 'SEMICOLON', 'COLON','COLONCOLON', 'COMMA',
    'LPAREN', 'RPAREN', 'LBRACE', 'RBRACE', 'DOT',
    
    'OR', 'AND', 'NOT',
    
    'PLUS', 'MINUS', 'MULT', 'DIV', 'MOD', 'EQEQ', 'NOTEQ', 'LTEQ', 'GTEQ',
    'INCREMENT', 'DECREMENT',
    'PLUSEQUAL', 'MINUSEQUAL', 'MULTEQUAL', 'DIVEQUAL',
    
    'INTEGER', 'FLOAT', 'STRING_LITERAL', 'CHAR_LITERAL',
    
    'IDENTIFIER', 'ELLIPSIS',
    
    'STD_CONV', 'NEW', 'DELETE','SIZEOF',
    
    'AMPERSAND','LBRACKET', 'RBRACKET', 
    'TYPEOF','DOLLAR',
    'IMPORT_C','IMPORT', 'AS', 'EXTERN', 'AS_PTR'
)


t_LT = r'<'
t_GT = r'>'
t_EQUAL = r'='
t_SEMICOLON = r';'
t_COLON = r':'
t_COLONCOLON = r'::'
t_COMMA = r','
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_LBRACE = r'\{'
t_RBRACE = r'\}'
t_DOT = r'\.'
t_PLUS = r'\+'
t_MINUS = r'-'
t_MULT = r'\*'
t_DIV = r'/'
t_MOD = r'%'
t_AT = r'@'
t_ELLIPSIS = r'\.\.\.'
t_DOLLAR = r'\$'
t_LBRACKET = r'\['
t_RBRACKET = r'\]'
t_AMPERSAND = r'&'




def t_OR(t):
    r'\|\|'
    return t

def t_AND(t):
    r'&&'
    return t

def t_NOT(t):
    r'!'
    return t
def t_EQEQ(t):
    r'=='
    return t

def t_NOTEQ(t):
    r'!='
    return t

def t_LTEQ(t):
    r'<='
    return t

def t_GTEQ(t):
    r'>='
    return t

def t_INCREMENT(t):
    r'\+\+'
    return t

def t_DECREMENT(t):
    r'--'
    return t

def t_PLUSEQUAL(t):
    r'\+='
    return t

def t_MINUSEQUAL(t):
    r'-='
    return t

def t_MULTEQUAL(t):
    r'\*='
    return t

def t_DIVEQUAL(t):
    r'/='
    return t


keywords = {
    
    'let': 'LET',
    'bez': 'BEZ',
    'const': 'CONST',
    'beton': 'BETON',
    'auto': 'AUTO',
    
    'fun': 'FUN',
    '<noret>': 'NORET',
    'return': 'RETURN',
    
    'struct': 'STRUCT',
    'enum': 'ENUM',
    
    'macro': 'MACRO',
    'static': 'STATIC',
    
    'while': 'WHILE',
    'for': 'FOR',
    'foreach': 'FOREACH',
    'break': 'BREAK',
    'continue': 'CONTINUE',
    'if': 'IF',
    'else': 'ELSE',
    'elseif': 'ELSEIF',
    'in': 'IN',
    
    'import_c': 'IMPORT_C',
    'import': 'IMPORT',
    'as' : 'AS',
    'extern': 'EXTERN',
    'new': 'NEW',
    'delete': 'DELETE',
    'typeof': 'TYPEOF',
    'std_conv' : 'STD_CONV',
    'sizeof':'SIZEOF',
    'as_ptr':'AS_PTR',
}

def t_IDENTIFIER(t):
    r'[a-zA-Z_][a-zA-Z0-9_]*'
    t.type = keywords.get(t.value, 'IDENTIFIER')  
    return t


def t_FLOAT(t):
    r'\d+\.\d+'
    t.value = float(t.value)
    return t

def t_INTEGER(t):
    r'\d+'
    t.value = int(t.value)
    return t

def t_STRING_LITERAL(t):
    r'\"([^\\\"]|\\.)*\"'
    t.value = t.value[1:-1]  
    return t

def t_CHAR_LITERAL(t):
    r'\'([^\\\']|\\.)\''
    t.value = t.value[1:-1]  
    return t


t_ignore = ' \t\r'
t_ignore_LINECOMMENT = r'//.*' 



def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)


def t_error(t):
    print(f"Illegal character '{t.value[0]}' at line {t.lineno}")
    t.lexer.skip(1)


lexer = lex.lex()


if __name__ == "__main__":
    data = """
    let x <int> = 1;
    bez z <float> = 1.1;
    fun add(x: <int>, y: <int>) <int> {
        return x + y;
    }
    """
    while True:
        lexer.input(input("> "))
        while True:
            tok = lexer.token()
            if not tok:
                break
            print(tok)
