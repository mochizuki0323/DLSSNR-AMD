#!/usr/bin/env python3
"""Fully unroll constant-trip-count `for` loops in preprocessed GLSL (glslang -E output).

    unroll_glsl.py IN OUT [--max-trip N]

Why: LLPC (the AMD Windows driver's compiler) leaves the large per-fragment loops of
fswin_t.comp rolled, so the small fragment arrays they index (qb[m], eq[m][h], ...) get a
dynamic index and live in scratch. NIR unrolls them before ACO sees them. Doing the same
unroll in the source gives LLPC straight-line code with constant indices.

A loop is unrolled when its header is `for (int|uint V = A; V < B; V++ | ++V | V += S)` with A, B,
S integer constant expressions (`16u` and `uint(...)` are integers too), its body neither assigns V nor
contains break/return at its own level, and the trip count is at most --max-trip. Each copy is
`{ const int V = value; body }`, or `{ const int V = value; do body while (false); }` when the body has a
`continue` (which then ends its own copy), and the copies sit in one more block, since the loop can be
the unbraced body of an `if`. Inner loops are unrolled first; in each copy, the loops left whose headers
use V get V's value written in and are tried again. `.length()` of a 16x16 subgroup cooperative matrix
is written as 8 everywhere (8 components a lane at the subgroup size 32 the runtime requires).
"""
import argparse
import re
import sys

HEAD = re.compile(r'\bfor\s*\(')
GLOBALS = set()          # arrays that live in buffers or shared memory (filled by main)
PRIVATE_ONLY = False     # --private-index


def indexes_private(body, v):
    """True if `v` appears inside the subscript of an array that is not a buffer/shared array."""
    for m in re.finditer(r'\b(\w+)\s*\[', body):
        name = m.group(1)
        if name in GLOBALS:
            continue
        try:
            close = match_paren(body, m.end() - 1, '[', ']')
        except ValueError:
            continue
        if re.search(r'\b%s\b' % v, body[m.end():close]):
            return True
    return False
CONST = re.compile(r'^[\d\s()+\-*/%]+$')


def match_paren(s, i, open_='(', close=')'):
    depth = 0
    while i < len(s):
        if s[i] == open_:
            depth += 1
        elif s[i] == close:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError('unbalanced ' + open_)


def skip_ws(s, i):
    while i < len(s) and s[i].isspace():
        i += 1
    return i


def statement_end(s, i):
    """End (exclusive) of the statement starting at s[i]."""
    i = skip_ws(s, i)
    if s[i] == '{':
        return match_paren(s, i, '{', '}') + 1
    m = re.match(r'(for|while|if|switch)\b', s[i:])
    if m:
        p = match_paren(s, s.index('(', i))
        end = statement_end(s, p + 1)
        if m.group(1) == 'if':
            j = skip_ws(s, end)
            if re.match(r'else\b', s[j:]):
                end = statement_end(s, j + 4)
        return end
    if re.match(r'do\b', s[i:]):
        end = statement_end(s, i + 2)
        j = skip_ws(s, end)
        p = match_paren(s, s.index('(', j))
        return s.index(';', p) + 1
    depth = 0
    while True:                      # a simple statement; parens/braces inside initialisers
        c = s[i]
        if c in '({[':
            depth += 1
        elif c in ')}]':
            depth -= 1
        elif c == ';' and depth == 0:
            return i + 1
        i += 1


def const_eval(e):
    e = re.sub(r'\buint\s*\(', '(', e.strip())
    e = re.sub(r'(\d)[uU]\b', r'\1', e)
    if not CONST.match(e):
        return None
    try:
        return int(eval(e.replace('/', '//'), {'__builtins__': {}}))
    except Exception:
        return None


def own_level_escape(body):
    """The kinds of break/continue/return that would leave this loop (nested loops/switches skipped)."""
    s, i, flat = body, 0, []
    loop = re.compile(r'\b(for|while|switch|do)\b')
    while i < len(s):
        m = loop.search(s, i)
        if not m:
            flat.append(s[i:])
            break
        flat.append(s[i:m.start()])
        try:
            i = statement_end(s, m.start())
        except Exception:
            return {'?'}
    return set(re.findall(r'\b(break|continue|return)\b', ''.join(flat)))


def bind_outer(body, var, lit, max_trip, stats):
    """`body` as one copy of an unrolled loop over `var` = `lit`: the headers of the loops still in it that
    use `var` get the value written in, and are offered to transform() again (a copy may make them constant).
    Untouched when no header uses `var`, or when the body declares a `var` of its own."""
    if not re.search(r'\bfor\s*\([^;]*\b%s\b' % var, body) or re.search(r'\bu?int\s+%s\b' % var, body):
        return body
    pieces, at = [], 0
    for h in HEAD.finditer(body):
        lo = body.index('(', h.start())
        hi = match_paren(body, lo)
        pieces.append(body[at:lo] + re.sub(r'\b%s\b' % var, '(%s)' % lit, body[lo:hi + 1]))
        at = hi + 1
    pieces.append(body[at:])
    return transform(''.join(pieces), max_trip, stats)


def transform(s, max_trip, stats):
    out, pos = [], 0
    while True:
        m = HEAD.search(s, pos)
        if not m:
            out.append(s[pos:])
            return ''.join(out)
        p_open = s.index('(', m.start())
        p_close = match_paren(s, p_open)
        end = statement_end(s, p_close + 1)
        header = s[p_open + 1:p_close]
        body = transform(s[p_close + 1:end], max_trip, stats)       # inner loops first
        parts = header.split(';')
        unrolled = None
        if len(parts) == 3:
            h0 = re.match(r'\s*(int|uint)\s+(\w+)\s*=\s*(.+)$', parts[0], re.S)
            if h0:
                kind, v = h0.group(1), h0.group(2)
                lit = 'u' if kind == 'uint' else ''
                a = const_eval(h0.group(3))
                hc = re.match(r'\s*%s\s*(<|<=)\s*(.+)$' % v, parts[1], re.S)
                b = const_eval(hc.group(2)) if hc else None
                inc = parts[2].strip()
                step = None
                if re.fullmatch(r'\+\+\s*%s|%s\s*\+\+' % (v, v), inc):
                    step = 1
                else:
                    hs = re.fullmatch(r'%s\s*\+=\s*(.+)' % v, inc, re.S)
                    step = const_eval(hs.group(1)) if hs else None
                if a is not None and b is not None and step and step > 0:
                    if hc.group(1) == '<=':
                        b += 1
                    values = list(range(a, b, step))
                    assigns = re.search(r'(?<![\w.])%s\s*(=[^=]|\+\+|--|[-+*/]=)|(\+\+|--)\s*%s\b' % (v, v), body)
                    wanted = not PRIVATE_ONLY or indexes_private(body, v)
                    leaves = own_level_escape(body)
                    if (wanted and len(values) <= max_trip and not assigns and leaves <= {'continue'}
                            and stats['unrolled'] < stats.get('limit', 1 << 30)):
                        # One block around the copies: the loop may be the body of an `if`. With a
                        # `continue`, `do ... while (false)` around each copy ends that copy.
                        copy = '{ const %s %s = %d%s; do %s while (false); }' if leaves else '{ const %s %s = %d%s; %s }'
                        unrolled = '{ ' + ''.join(copy % (kind, v, x, lit, bind_outer(body, v, '%d%s' % (x, lit), max_trip, stats))
                                                  for x in values) + ' }'
                        stats['unrolled'] += 1
        if unrolled is None:
            stats['kept'] += 1
            unrolled = s[m.start():p_close + 1] + body
        out.append(s[pos:m.start()] + unrolled)
        pos = end


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--max-trip', type=int, default=64)
    ap.add_argument('--limit', type=int, default=1 << 30, help='unroll at most N loops (bisection)')
    ap.add_argument('--private-index', action='store_true',
                    help='only unroll loops whose variable indexes a local (non-buffer, non-shared) array')
    a = ap.parse_args()
    text = open(a.src).read()
    global PRIVATE_ONLY
    PRIVATE_ONLY = a.private_index
    # buffer block members declared as arrays, and shared arrays
    for m in re.finditer(r'\bbuffer\s+\w+\s*\{([^}]*)\}', text):
        GLOBALS.update(re.findall(r'(\w+)\s*\[\s*\]', m.group(1)))
    GLOBALS.update(re.findall(r'\bshared\s+[\w<>, ]+?\s+(\w+)\s*\[', text))
    mats = set(re.findall(r'\bcoopmat\s*<[^<>]*gl_ScopeSubgroup\s*,\s*16\s*,\s*16\s*,[^<>]*>\s+(\w+)', text))
    if mats:   # 16x16 subgroup cooperative matrices: 8 components a lane (subgroup size 32)
        text = re.sub(r'\b(%s)((?:\[[^\[\]]*\])*)\.length\(\)' % '|'.join(sorted(mats)), '8', text)
    stats = {'unrolled': 0, 'kept': 0, 'limit': a.limit}
    sys.setrecursionlimit(10000)
    out = transform(text, a.max_trip, stats)
    open(a.dst, 'w').write(out)
    print(f"{a.src}: {stats['unrolled']} loops unrolled, {stats['kept']} kept, {len(text)} -> {len(out)} bytes")


if __name__ == '__main__':
    main()
