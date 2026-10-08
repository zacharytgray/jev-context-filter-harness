"""DocFinQA filings and the usable-question rule the task draw reads (draw_full.py): test rows grouped by filing, and
a question is usable when its gold parses under DFQ-1 and its gold program gives a value that matches it. the gold
program is read with a small ast evaluator, never exec"""
import ast
import hashlib

from experiments.bench import score_docfinqa as S


def filing_id(context: str) -> str:
    return hashlib.sha256(context.encode()).hexdigest()[:8]


def filings(rows: list[dict]) -> dict[str, list[int]]:
    """test rows grouped by the sha256 of Context, keyed by its first 8 hex chars, rows in order"""
    out, full = {}, {}
    for i, r in enumerate(rows):
        h = hashlib.sha256(r["Context"].encode()).hexdigest()
        if full.setdefault(h[:8], h) != h:
            raise ValueError(f"two filings share the id {h[:8]}")
        out.setdefault(h[:8], []).append(i)
    return out


# the gold program's value, read with a small ast evaluator, never exec. the grammar: assignments to a name, numbers,
# names, + - * / ** and unary minus, parentheses, and max, min, sum and len over a list literal or a name (26 test
# programs use them, as in `table_row = [829, 480, 104]` then `answer = max(table_row)`). the value is `answer`
_OPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b, ast.Pow: lambda a, b: a ** b}
_FUNCS = {"max": max, "min": min, "sum": sum, "len": len}


def _eval(node, env: dict):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.Name) and node.id in env:
        return env[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        a, b = _eval(node.left, env), _eval(node.right, env)
        if isinstance(a, list) or isinstance(b, list):
            raise ValueError("arithmetic takes numbers")
        if isinstance(node.op, ast.Pow) and abs(b) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](a, b)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _eval(node.operand, env)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.List):
        return [_eval(x, env) for x in node.elts]
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS
            and not node.keywords and len(node.args) == 1):
        v = _eval(node.args[0], env)
        if not isinstance(v, list):
            raise ValueError("a call takes a list")
        return _FUNCS[node.func.id](v)
    raise ValueError(f"not allowed: {type(node).__name__}")


def program_value(program: str) -> float | None:
    """the value the program assigns to `answer`. None for an empty program, anything outside the grammar, or no
    numeric answer"""
    src = "\n".join(line.strip() for line in (program or "").splitlines() if line.strip())
    if not src:
        return None
    try:
        tree = ast.parse(src)
        env = {}
        for st in tree.body:
            if not (isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name)):
                raise ValueError(f"not allowed: {type(st).__name__}")
            env[st.targets[0].id] = _eval(st.value, env)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, TypeError):
        return None
    v = env.get("answer")
    return v if type(v) in (int, float) else None


def usable(row: dict) -> bool:
    """the gold parses under DFQ-1, and the gold program's value matches it under DFQ-1 strict"""
    if S.parse(row["Answer"]) is None:
        return False
    v = program_value(row["Program"])
    return v is not None and S.judge(repr(v), row["Answer"])["strict"]
