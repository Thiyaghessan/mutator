"""Find mutation sites.

Clojure follows clj-mutate's symbol rules. Go follows mutate4go, including
one-way `*` to `/`. Java follows mutate4java. TypeScript, Rust, and Python
use the same decisions as Java, spelled in that language. TypeScript also
mutates `??` to `||` and `?.` to `.` (removed on a call or index). JavaScript
files use the TypeScript rules.

R has its own set, listed above `_visit_r`.
"""

from __future__ import annotations

import re

from mutator.crapper_link import ensure_crapper

_BINARY = {
    "+": "-",
    "-": "+",
    "*": "/",
    "/": "*",
    ">": ">=",
    ">=": ">",
    "<": "<=",
    "<=": "<",
    "==": "!=",
    "!=": "==",
    "===": "!==",
    "!==": "===",
    "&&": "||",
    "||": "&&",
    "??": "||",
    "and": "or",
    "or": "and",
}

_BINARY_PARENTS = {
    "binary_expression",
    "binary_operator",
    "comparison_operator",
    "boolean_operator",
}

_UNARY_PARENTS = {"unary_expression", "unary_operator", "not_operator"}

_BOOLEANS = {"true": "false", "false": "true", "True": "False", "False": "True"}

_NUMBERS = {
    "decimal_integer_literal",
    "integer_literal",
    "int_literal",
    "integer",
    "number",
    "num_lit",
}

_CLOJURE_HEAD = {
    "+": ("-", "arithmetic"),
    "-": ("+", "arithmetic"),
    "*": ("/", "arithmetic"),
    "inc": ("dec", "arithmetic"),
    "dec": ("inc", "arithmetic"),
    ">": (">=", "comparison"),
    ">=": (">", "comparison"),
    "<": ("<=", "comparison"),
    "<=": ("<", "comparison"),
    "=": ("not=", "equality"),
    "not=": ("=", "equality"),
    "if": ("if-not", "conditional"),
    "if-not": ("if", "conditional"),
    "when": ("when-not", "conditional"),
    "when-not": ("when", "conditional"),
    "and": ("or", "logical"),
    "or": ("and", "logical"),
    "double": ("int", "coercion"),
    "int": ("double", "coercion"),
    "first": ("second", "seq"),
    "second": ("first", "seq"),
    "filter": ("remove", "seq"),
    "remove": ("filter", "seq"),
    "take": ("drop", "seq"),
    "drop": ("take", "seq"),
    "rest": ("next", "seq"),
    "next": ("rest", "seq"),
    "every?": ("some", "seq"),
    "some": ("every?", "seq"),
    "min": ("max", "numeric"),
    "max": ("min", "numeric"),
    "pos?": ("neg?", "predicate"),
    "neg?": ("pos?", "predicate"),
    "even?": ("odd?", "predicate"),
    "odd?": ("even?", "predicate"),
    "nil?": ("some?", "predicate"),
    "some?": ("nil?", "predicate"),
}


class RawSite:
    def __init__(
        self,
        line: int,
        start: int,
        end: int,
        original: str,
        mutant: str,
        category: str,
        anchor_line: int = 0,
        within: tuple[int, int] | None = None,
    ):
        self.line = line
        # The line its statement starts on. LCOV marks only that line of a multi-line statement.
        self.anchor_line = anchor_line or line
        self.start = start
        self.end = end
        self.original = original
        self.mutant = mutant
        self.category = category
        # See Site.within.
        self.within = within


def _grammar(language: str, path: str) -> str:
    if language == "typescript":
        if path.endswith(".tsx"):
            return "tsx"
        if path.endswith((".js", ".jsx", ".mjs", ".cjs")):
            return "javascript"
        return "typescript"
    if language == "go":
        return "go"
    return language


def _text(data: bytes, node) -> str:
    return data[node.start_byte : node.end_byte].decode("utf-8")


def _category(token: str) -> str:
    if token in {"+", "-", "*", "/"}:
        return "arithmetic"
    if token in {">", ">=", "<", "<="}:
        return "comparison"
    if token in {"==", "!=", "===", "!=="}:
        return "equality"
    if token in {"&&", "||", "??", "and", "or"}:
        return "logical"
    return "arithmetic"


def _binary_mutant(language: str, token: str) -> str | None:
    if language == "go" and token == "/":
        return None
    if token == "??" and language != "typescript":
        return None
    return _BINARY.get(token)


def _optional_mutant(node) -> str | None:
    """`a?.b` becomes `a.b`. A call or index drops `?.`, so `a?.()` becomes `a()`."""

    parent = node.parent
    if parent is None:
        return None
    host = parent.parent if parent.type == "optional_chain" else parent
    if host is None:
        return None
    if host.type == "member_expression":
        return "."
    if host.type in {"call_expression", "subscript_expression"}:
        return ""
    return None


def _add(found: list[RawSite], node, original: str, mutant: str, category: str) -> None:
    if mutant == original:
        return
    found.append(
        RawSite(
            line=node.start_point[0] + 1,
            start=node.start_byte,
            end=node.end_byte,
            original=original,
            mutant=mutant,
            category=category,
            anchor_line=_statement_line(node),
        )
    )


def _is_statement(node_type: str) -> bool:
    return node_type.endswith("_statement") or node_type.endswith("_declaration")


def _statement_line(node) -> int:
    current = node.parent
    while current is not None:
        if _is_statement(current.type):
            return current.start_point[0] + 1
        current = current.parent
    return node.start_point[0] + 1


# A literal in a TypeScript type, such as `{ ok: true }` in a return type, never runs.
_TYPE_NODES = {
    "type_annotation",
    "type_alias_declaration",
    "interface_declaration",
    "type_arguments",
    "type_parameters",
    "literal_type",
}


def _in_type(node) -> bool:
    current = node.parent
    while current is not None:
        if current.type in _TYPE_NODES:
            return True
        current = current.parent
    return False


def _clojure_head(node, data: bytes) -> str | None:
    for child in node.children:
        if child.type == "sym_lit":
            return _text(data, child)
    return None


def _is_head(node) -> bool:
    parent = node.parent
    if parent is None or parent.type != "list_lit":
        return False
    for child in parent.children:
        if child.type == "sym_lit":
            return child.id == node.id
    return False


def _clojure_skipped(node, data: bytes) -> bool:
    current = node
    while current is not None:
        if any(part in current.type for part in ("comment", "quot", "string", "str_lit", "char_lit")):
            return True
        if current.type == "list_lit":
            head = _clojure_head(current, data)
            if head in {"comment", "quote"}:
                return True
        current = current.parent
    return False


def _clojure_text(node, data: bytes) -> str:
    if node.child_count == 0 or node.type in {"sym_lit", "bool_lit", "num_lit"}:
        return _text(data, node)
    return ""


def _flip01(text: str) -> str:
    if text == "0":
        return "1"
    return "0"


def _clojure_node(node, data: bytes, found: list[RawSite]) -> None:
    if _clojure_skipped(node, data):
        return
    text = _clojure_text(node, data)
    if node.type == "sym_lit" and _is_head(node) and text in _CLOJURE_HEAD:
        mutant, category = _CLOJURE_HEAD[text]
        _add(found, node, text, mutant, category)
        return
    if node.type == "bool_lit" and text in _BOOLEANS:
        _add(found, node, text, _BOOLEANS[text], "boolean")
        return
    if node.type == "num_lit" and text in {"0", "1"}:
        _add(found, node, text, _flip01(text), "constant")


def _visit_clojure(node, data: bytes, found: list[RawSite]) -> None:
    # An explicit stack: a 1,200-deep form must not raise RecursionError.
    stack = [node]
    while stack:
        current = stack.pop()
        _clojure_node(current, data, found)
        stack.extend(reversed(current.children))


def _parent_type(node) -> str:
    if node.parent is None:
        return ""
    return node.parent.type


def _visit_binary(node, token: str, parent: str, language: str, found: list[RawSite]) -> None:
    if parent not in _BINARY_PARENTS:
        return
    mutant = _binary_mutant(language, token)
    if mutant is not None:
        _add(found, node, token, mutant, _category(token))


def _visit_optional(node, token: str, language: str, found: list[RawSite]) -> None:
    if language != "typescript" or token != "?.":
        return
    optional = _optional_mutant(node)
    if optional is not None:
        _add(found, node, "?.", optional, "optional")


def _visit_not(node, token: str, parent: str, found: list[RawSite]) -> None:
    if token in {"!", "not"} and parent in _UNARY_PARENTS | {"comparison_operator"}:
        _add(found, node, token, "", "unary")


def _visit_negation(node, token: str, parent: str, found: list[RawSite]) -> None:
    if token == "-" and parent in _UNARY_PARENTS:
        _add(found, node, token, "", "unary")


def _visit_boolean_token(node, token: str, found: list[RawSite]) -> None:
    if node.type in {"true", "false"} and token in _BOOLEANS:
        _add(found, node, token, _BOOLEANS[token], "boolean")


def _visit_constant(node, token: str, found: list[RawSite]) -> None:
    if node.type in _NUMBERS and token in {"0", "1"}:
        _add(found, node, token, _flip01(token), "constant")


def _visit_bool_lit(node, data: bytes, found: list[RawSite]) -> None:
    if node.type != "bool_lit":
        return
    text = _text(data, node)
    if text in _BOOLEANS:
        _add(found, node, text, _BOOLEANS[text], "boolean")


def _visit_leaf(node, data: bytes, language: str, found: list[RawSite]) -> None:
    # A literal sits inside the same expression as an operator, so these
    # checks are independent. `0` is a child of `a > 0`, not an operator.
    token = _text(data, node)
    parent = _parent_type(node)
    _visit_binary(node, token, parent, language, found)
    _visit_optional(node, token, language, found)
    _visit_not(node, token, parent, found)
    _visit_negation(node, token, parent, found)
    _visit_boolean_token(node, token, found)
    _visit_constant(node, token, found)


def _visit_node(node, data: bytes, language: str, found: list[RawSite]) -> None:
    if language == "typescript" and _in_type(node):
        return
    if node.child_count == 0:
        _visit_leaf(node, data, language, found)
    else:
        _visit_bool_lit(node, data, found)


def _visit(node, data: bytes, language: str, found: list[RawSite]) -> None:
    # An explicit stack: a 1,200-term expression must not raise RecursionError.
    stack = [node]
    while stack:
        current = stack.pop()
        _visit_node(current, data, language, found)
        stack.extend(reversed(current.children))


# R mutations, one site each:
#
# - a comparison is flipped: `==` and `!=`, `<` and `<=`, `>` and `>=`
# - `&&` becomes `||`, and `||` becomes `&&`
# - `TRUE` becomes `FALSE`, and `FALSE` becomes `TRUE`
# - a plain number goes up by one: `0` to `1`, `12L` to `13L`, `2.5` to `3.5`
# - the condition of an `if` or a `while` is negated: `x` to `!(x)`
# - a statement inside braces is deleted, unless it is a return
#
# A return is one of two things. It is a call to `return(...)`, also when
# written `` `return`(...) `` or `"return"(...)`, put in parentheses, or
# wrapped as in `invisible(return(x))`. Or it
# is an expression whose value the function gives back: the last expression
# of the function body, and, when that is an `if` or a `switch`, the last
# expression of each of its branches. Arithmetic operators are left alone.
_R_FLIPPED_OPERATORS = {"==", "!=", "<", "<=", ">", ">=", "&&", "||"}
_R_BOOLEANS = {"TRUE": "FALSE", "FALSE": "TRUE"}
_R_NUMBERS = {"float", "integer"}
_R_NEGATED_CONDITIONS = {"if_statement", "while_statement"}
# Digits, an optional fraction, and an optional L. Hex and 1e3 do not match.
_R_PLAIN_NUMBER = re.compile(r"(\d*)(\.\d*)?(L?)")


def _r_field_of(node) -> str:
    """The name of the field `node` fills in its parent, or an empty string."""

    parent = node.parent
    for index, child in enumerate(parent.children):
        if child.id == node.id:
            return parent.field_name_for_child(index) or ""
    return ""


# covr counts a statement inside braces by itself. A body written without
# braces after `while`, `repeat`, or `function` is counted only as part of the
# statement that holds it, so that statement having run does not show the
# body ran. `for` is listed for the same reason, though covr does count its
# body.
#
# A function is the one case with a count of its own: for a function assigned
# at the top of a file, covr counts the body under the whole `function(...)`
# expression. So for a function the bytes that must hold a count are those of
# the `function(...)` expression, and for a loop they are those of the body.
_R_BODY_OWNERS = {"while_statement", "repeat_statement", "for_statement", "function_definition"}


def _r_body_without_braces(node) -> tuple[int, int] | None:
    """Bytes that must hold a count for `node`, when its nearest body has no braces."""

    current = node
    while current.parent is not None:
        parent = current.parent
        if (
            parent.type in _R_BODY_OWNERS
            and current.type != "braced_expression"
            and _r_field_of(current) == "body"
        ):
            measured = parent if parent.type == "function_definition" else current
            return measured.start_byte, measured.end_byte
        current = parent
    return None


def _add_r(found: list[RawSite], node, original: str, mutant: str, category: str) -> None:
    found.append(
        RawSite(
            line=node.start_point[0] + 1,
            start=node.start_byte,
            end=node.end_byte,
            original=original,
            mutant=mutant,
            category=category,
            within=_r_body_without_braces(node),
        )
    )


def _r_number_plus_one(text: str) -> str | None:
    match = _R_PLAIN_NUMBER.fullmatch(text)
    if match is None or not any(character.isdigit() for character in text):
        return None
    whole, fraction, suffix = match.groups()
    return f"{int(whole or '0') + 1}{fraction or ''}{suffix}"


def _r_is_operator(node) -> bool:
    parent = node.parent
    return parent is not None and parent.type == "binary_operator" and _r_field_of(node) == "operator"


def _r_called_name(node, data: bytes) -> str:
    """The name a call names, read the way crapper reads R names, or "" when it has none."""

    crapper = ensure_crapper()
    return crapper.languages.r.plain_name(data, node.child_by_field_name("function"))


def _r_argument_values(call) -> list:
    arguments = call.child_by_field_name("arguments")
    if arguments is None:
        return []
    values = []
    for argument in arguments.children:
        value = argument.child_by_field_name("value") if argument.type == "argument" else None
        if value is not None:
            values.append(value)
    return values


def _r_is_return_call(node, data: bytes) -> bool:
    """`return(x)`, also in parentheses, or handed straight to another call.

    `(return(x))` and `invisible(return(x))` both leave the function.
    """

    pending = [node]
    while pending:
        current = pending.pop()
        if current.type == "parenthesized_expression":
            pending.extend(current.named_children)
            continue
        if current.type != "call":
            continue
        if _r_called_name(current, data) == "return":
            return True
        pending.extend(_r_argument_values(current))
    return False


def _r_statements(block) -> list:
    return [
        child
        for index, child in enumerate(block.children)
        if block.field_name_for_child(index) == "body"
    ]


def _r_passes_its_value_up(node, data: bytes):
    """The node whose value is the value of `node`, or None when there is none.

    The last statement of a block is the value of the block. A branch of an
    `if` is the value of the `if`. An alternative of a `switch` is the value
    of the `switch`.
    """

    parent = node.parent
    if parent.type == "braced_expression":
        statements = _r_statements(parent)
        if statements and statements[-1].id == node.id:
            return parent
        return None
    if parent.type == "if_statement" and _r_field_of(node) in {"consequence", "alternative"}:
        return parent
    if parent.type == "parenthesized_expression":
        return parent
    if parent.type == "argument":
        call = parent.parent.parent
        if call.type == "call" and _r_called_name(call, data) == "switch":
            return call
    return None


def _r_gives_the_function_value(node, data: bytes) -> bool:
    """True when the value of `node` is what its function gives back."""

    current = node
    while current is not None:
        parent = current.parent
        if parent is None:
            return False
        if parent.type == "function_definition":
            return _r_field_of(current) == "body"
        current = _r_passes_its_value_up(current, data)
    return False


def _r_is_return(statement, data: bytes) -> bool:
    return _r_is_return_call(statement, data) or _r_gives_the_function_value(statement, data)


def _r_deletable_statements(block, data: bytes) -> list:
    return [
        statement for statement in _r_statements(block) if not _r_is_return(statement, data)
    ]


def _visit_r_token(node, data: bytes, found: list[RawSite]) -> None:
    token = _text(data, node)
    if token in _R_FLIPPED_OPERATORS and _r_is_operator(node):
        _add_r(found, node, token, _BINARY[token], _category(token))
    elif node.type in {"true", "false"} and token in _R_BOOLEANS:
        _add_r(found, node, token, _R_BOOLEANS[token], "boolean")


def _visit_r_number(node, data: bytes, found: list[RawSite]) -> None:
    text = _text(data, node)
    raised = _r_number_plus_one(text)
    if raised is not None:
        _add_r(found, node, text, raised, "constant")


def _visit_r_condition(node, data: bytes, found: list[RawSite]) -> None:
    condition = node.child_by_field_name("condition")
    if condition is not None:
        text = _text(data, condition)
        _add_r(found, condition, text, f"!({text})", "conditional")


def _visit_r_block(node, data: bytes, found: list[RawSite]) -> None:
    for statement in _r_deletable_statements(node, data):
        _add_r(found, statement, _text(data, statement), "", "statement")


def _visit_r_node(node, data: bytes, found: list[RawSite]) -> None:
    if node.type in _R_NUMBERS:
        _visit_r_number(node, data, found)
    elif node.child_count == 0:
        _visit_r_token(node, data, found)
    elif node.type in _R_NEGATED_CONDITIONS:
        _visit_r_condition(node, data, found)
    elif node.type == "braced_expression":
        _visit_r_block(node, data, found)


def _visit_r(node, data: bytes, found: list[RawSite]) -> None:
    # An explicit stack: a 1,200-term expression must not raise RecursionError.
    stack = [node]
    while stack:
        current = stack.pop()
        _visit_r_node(current, data, found)
        stack.extend(reversed(current.children))


def discover_raw(source: str, language: str, path: str) -> list[RawSite]:
    """Mutation sites in source order. Offsets are UTF-8 byte offsets."""

    crapper = ensure_crapper()
    data = source.encode("utf-8")
    grammar = _grammar(language, path)
    _data, tree = crapper.languages.treesitter.parse(source, grammar)
    found: list[RawSite] = []
    if language == "clojure":
        _visit_clojure(tree.root_node, data, found)
    elif language == "r":
        _visit_r(tree.root_node, data, found)
    else:
        _visit(tree.root_node, data, language, found)
    found.sort(key=lambda site: (site.line, site.start, site.end))
    return found


def apply_site(source: str, start: int, end: int, mutant: str) -> str:
    data = source.encode("utf-8")
    updated = data[:start] + mutant.encode("utf-8") + data[end:]
    return updated.decode("utf-8")
