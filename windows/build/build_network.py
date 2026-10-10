#!/usr/bin/env python3
"""Build the Windows network (AMD proprietary driver, LLPC) from windows/shaders/<arch>/pipelines.json.

    build_network.py <arch> [--out DIR] [--check DIR]

Writes (default build/windows/<arch>/network):
    g_<name>.spv + the four marker files   the network, what --spv-dir points at
    runtime/                                 the passes around it (windows/shaders/passes)
    temporal/                                the temporal variants and the motion estimator

LLPC leaves the large per-fragment loops of fswin_t.comp rolled and spills the arrays they
index to scratch, so every fswin_t pipeline is preprocessed (ffwd3_t, attn and vit_attn too: UNROLLED below),
its pair quantisation loops made
four-wide (windows/build/quad_quant_glsl.py; --no-quad keeps them; it also drops the f32 -> f16 -> f32
round trips RADV deletes anyway, NR_Q32_DIRECT=0 in the environment keeps them), fully unrolled
(windows/build/unroll_glsl.py) and compiled from the result.

The host must be compiled with the same constants: windows/build/arch/<arch>.sh.
--check DIR compares every network SPV and marker with DIR byte for byte and
lists the differences (exit 1 if any).
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[2]
RUNTIME = ['runtime_alpha', 'runtime_encode', 'runtime_transfer', 'runtime_upscale', 'runtime_prep', 'runtime_depth', 'cascade_lograt', 'cascade_blur', 'cascade_feed',
           'runtime_downscale', 'runtime_taps']
# runtime passes built from another pass's source with defines: name -> (source, defines)
RUNTIME_VARIANTS = {'runtime_transfer_store': ('runtime_transfer', ['NR_STORE_NATIVE'])}
MOTION = ['motion_luma', 'motion_estimate']


QUAD = True   # --no-quad: the pair loops as written (A/B)
# The sources whose pipelines are unrolled (unroll_glsl.py) before glslang: fswin_t (LLPC spills the arrays its rolled
# loops index), and ffwd3_t, attn and vit_attn, so that no loop indexes an array of cooperative matrices with its
# loop variable.
UNROLLED = {'fswin_t.comp', 'ffwd3_t.comp', 'attn.comp', 'vit_attn.comp'}


def glslang(arch, src, defines, out, unroll=False, network=False):
    if unroll or (network and QUAD):
        pre = out.with_suffix('.pre.comp')
        r = subprocess.run([str(R / 'toolchain/glslang/bin/glslang'), '-E', '-I' + str(R / 'windows/shaders' / arch / 'include'),
                            '-I' + str(R / 'windows/shaders/passes/include')] + ['-D' + d for d in defines] + [str(src)],
                           capture_output=True, text=True)
        if r.returncode:
            sys.exit(f'glslang -E failed on {src}:\n{r.stdout}{r.stderr}')
        pre.write_text(r.stdout)
        if QUAD:
            subprocess.run([sys.executable, str(R / 'windows/build/quad_quant_glsl.py'), str(pre), str(pre)],
                           check=True, stdout=subprocess.DEVNULL)
        if unroll:
            subprocess.run([sys.executable, str(R / 'windows/build/unroll_glsl.py'), str(pre), str(pre)],
                           check=True, stdout=subprocess.DEVNULL)
        src, defines = pre, []
    cmd = [str(R / 'toolchain/glslang/bin/glslang'), '-V', '--target-env', 'vulkan1.3',
           '-I' + str(R / 'windows/shaders' / arch / 'include'), '-I' + str(R / 'windows/shaders/passes/include')]
    cmd += ['-D' + d for d in defines] + [str(src), '-o', str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f'glslang failed on {src}:\n{r.stdout}{r.stderr}')
    if unroll or (network and QUAD):
        src.unlink()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('arch')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--define', action='append', default=[],
                    help='extra define for every network pipeline (diagnostics, e.g. NR_PROF_OFF=...)')
    ap.add_argument('--check', type=Path)
    ap.add_argument('--no-quad', action='store_true',
                    help='keep the pair quantisation loops (windows/build/quad_quant_glsl.py off)')
    a = ap.parse_args()
    global QUAD
    QUAD = not a.no_quad
    table = json.loads((R / 'windows/shaders' / a.arch / 'pipelines.json').read_text())
    out = a.out or R / 'build/windows' / a.arch / 'network'
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    pipelines = table['pipelines']
    for name, e in pipelines.items():
        glslang(a.arch, R / 'windows/shaders' / a.arch / e['source'], e['defines'] + a.define, out / f'g_{name}.spv',
                unroll=e['source'] in UNROLLED, network=True)
    for name, text in table['markers'].items():
        (out / name).write_text('\n'.join(text) + '\n' if isinstance(text, list) else text + '\n')
    (out / 'runtime').mkdir()
    (out / 'temporal').mkdir()
    for name, v in table['variants'].items():
        base = pipelines[v['base']]
        glslang(a.arch, R / 'windows/shaders' / a.arch / base['source'], base['defines'] + v['add'],
                out / 'temporal' / f'{name}.spv', unroll=base['source'] in UNROLLED, network=True)
    for k in MOTION:
        glslang(a.arch, R / 'windows/shaders/passes' / f'{k}.comp', [], out / 'temporal' / f'{k}.spv')
    shutil.copy2(out / 'shader-constants.txt', out / 'temporal' / 'shader-constants.txt')
    for k in RUNTIME:
        glslang(a.arch, R / 'windows/shaders/passes' / f'{k}.comp', [], out / 'runtime' / f'{k}.spv')
    for k, (src, defines) in RUNTIME_VARIANTS.items():
        glslang(a.arch, R / 'windows/shaders/passes' / f'{src}.comp', defines, out / 'runtime' / f'{k}.spv')
    # runtime_upscale.spv carries AMD FidelityFX Super Resolution 1 (MIT): its notice goes with it
    shutil.copy2(R / 'windows/shaders/passes/include/fsr1/LICENSE.txt', out / 'runtime' / 'FidelityFX-FSR1-LICENSE.txt')
    print(f'{out}: {len(pipelines)} network pipelines, {len(table["variants"]) + len(MOTION)} temporal, '
          f'{len(RUNTIME) + len(RUNTIME_VARIANTS)} runtime')

    if a.check:
        bad = [p.name for p in sorted(out.iterdir()) if p.is_file()
               if not (a.check / p.name).is_file() or (a.check / p.name).read_bytes() != p.read_bytes()]
        print('check against', a.check, ':', 'identical' if not bad else 'DIFFERENT: ' + ' '.join(bad))
        sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
