#!/usr/bin/env python3
"""Default dlssnr-amd.ini files for the installer, written with the exact text the code writes (the format strings are
read out of windows/src/pe/nr_pe_config.cpp, so the two cannot drift): <out>/dlssnr-amd-optiscaler.ini ([Int4Mixed],
[Preprocess], [Network] and [Log], what the OptiScaler route reads) and <out>/dlssnr-amd-addon.ini (the ReShade add-on's whole file).
usage: make_ini.py <nr_pe_config.cpp> <out dir> [--int4]   (--int4: with [Int4Mixed], for an install with int4 mixed)"""
import re, sys
from pathlib import Path

src = Path(sys.argv[1]).read_text(encoding="utf-8")
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)


def formats(body):
    """the string literals of every std::fprintf(f, "..." "...", ...) in body, joined and unescaped"""
    res = []
    for m in re.finditer(r'std::fprintf\(f,\s*((?:"(?:[^"\\]|\\.)*"\s*)+)', body):
        lits = re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))
        res.append("".join(lits).encode("utf-8").decode("unicode_escape").encode("latin-1").decode("utf-8"))
    return res


def body_of(sig):
    i = src.index(sig); j = src.index("\n}\n", i)
    return src[i:j]


# [Int4Mixed] only for an install with int4 mixed (the programs built with NR_INT4=1 write it)
perf = formats(body_of("void write_int4mixed("))[0] if "--int4" in sys.argv[3:] else None
prep = formats(body_of("void write_preprocess("))[0]
logsec = formats(body_of("void write_log("))[0] % (1, 1)
network = formats(body_of("void write_network("))[0] % (0,)   # [Network] ACO Mode = 0: the driver's compiler
save = formats(body_of("void Config::save("))
assert prep and len(save) == 8, len(save)   # [DlssNr], six Pass<N> lines (none by default), own keys

# defaults (nr_runtime.hpp Controls / Preprocess, nr_pe_config.hpp Config / PreprocessConfig)
int4mixed = perf % (1, "Ctrl+F11", 0, 1) if perf else ""   # chosen at install: on
preprocess = prep % (0, "auto", 0.0, "filmic", 1.0, 1.0, "Ctrl+F10", 1)
dlssnr = save[0] % ("true", "true", 1, "false", 1.0, 0, 1.0, 1.0, 1.0, -1.0, "true", 1.0, 1.0, 2.0, 0, "catmullrom")
own = save[-1] % (1.0, 1.0)

(out / "dlssnr-amd-optiscaler.ini").write_text(int4mixed + preprocess + network + logsec, encoding="utf-8")
(out / "dlssnr-amd-addon.ini").write_text(dlssnr + own + "\n" + int4mixed + preprocess + network + logsec, encoding="utf-8")
print("wrote", out / "dlssnr-amd-optiscaler.ini", out / "dlssnr-amd-addon.ini")
