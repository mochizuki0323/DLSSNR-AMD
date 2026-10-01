#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Instalador com UI para DLSSNR-AMD-Vulkan.

Reproduz install.sh com interface grafica (tkinter, apenas stdlib):
- identifica o executavel na pasta do jogo (bits 32/64, aviso Unreal top-folder)
- extrai os pesos da DLL da NVIDIA uma vez so e reutiliza o caminho
  (config em ~/.config/dlssnr-amd-installer/config.json + dlssnr-amd/dlssnr.bin no pacote)
- instala / desinstala (via dlssnr-amd-install.txt)
- verifica o repositorio upstream https://github.com/mochizuki0323/DLSSNR-AMD
  por atualizacoes e atualiza o pacote (com progresso)
- barra de progressao + janela de log
- pre-configuracao do OptiScaler.ini antes da instalacao

Uso:  python3 installer-gui.py   (ou bash install-gui.sh)
"""
import glob
import hashlib
import json
import os
import queue
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
UPSTREAM_OWNER = "mochizuki0323"
UPSTREAM_REPO = "DLSSNR-AMD"
UPSTREAM_API = f"https://api.github.com/repos/{UPSTREAM_OWNER}/{UPSTREAM_REPO}/releases/latest"
UPSTREAM_URL = f"https://github.com/{UPSTREAM_OWNER}/{UPSTREAM_REPO}"

MODEL_NAME = "dlssnr.bin"
MODEL_SHA256 = "2b41c888cf4155b8958c665ba64018ab0bd25c85fc71a2b6db86d0d04d1f7fbd"
CURRENT_VERSION = "0.0.2.5"  # atualizado a cada release do pacote

CONFIG_DIR = Path.home() / ".config" / "dlssnr-amd-installer"
CONFIG_FILE = CONFIG_DIR / "config.json"

ROUTES = {
    "gamescope": "Gamescope + OptiScaler (reconstrução neural no Gamescope)",
    "optiscaler": "jogo tem DLSS/FSR/XeSS (OptiScaler, so 64-bit)",
    "reshade": "D3D10/11/12 sem upscaler utilizavel",
    "vulkan": "jogo Vulkan, ou D3D9 (via DXVK)",
    "dx9": "como vulkan + depth de D3D9 antigos",
}


# --------------------------------------------------------------------------
# config persistente (reutiliza caminho da DLL entre instalacoes)
# --------------------------------------------------------------------------

def load_config() -> dict:
    try:
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_config(cfg: dict) -> None:
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def package_arch() -> str:
    if (HERE / "optiscaler").is_dir():
        return "x86_64"
    if (HERE / "gamescope").is_dir() or (HERE / "DLSSNR-AMD").is_dir():
        return "x86_64"
    return "i686"


def package_bits() -> int:
    return 32 if (HERE / "reshade" / "dlssnr_amd.addon32").exists() else 64


# --------------------------------------------------------------------------
# deteccao de executaveis (igual ao install.sh)
# --------------------------------------------------------------------------

def pe_bits(exe: Path):
    try:
        b = exe.read_bytes()
        if len(b) < 0x40 or b[:2] != b"MZ":
            return "?"
        pe = struct.unpack_from("<I", b, 0x3C)[0]
        machine = struct.unpack_from("<H", b, pe + 4)[0]
        return {0x14C: "32", 0x8664: "64"}.get(machine, "?")
    except Exception:
        return "?"


def scan_game_folder(game: Path):
    """Retorna (exes, ue_warning). exes = [(nome, bits)]."""
    exes = []
    if game.is_dir():
        for exe in sorted(game.glob("*.exe")):
            if exe.is_file():
                exes.append((exe.name, pe_bits(exe)))
    ue = bool(glob.glob(str(game / "*" / "Binaries" / "Win64" / "*.exe")))
    return exes, ue


_MODEL_CACHE: dict = {}


def model_ok(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        st = path.stat()
    except Exception:
        return False
    key = (str(path), st.st_size, st.st_mtime_ns)
    # Evita re-hashear (SHA256 de dezenas de MiB) a cada tecla digitada:
    # refresh_scan() e chamado em todo trace_add("write").
    if _MODEL_CACHE.get("key") == key and "ok" in _MODEL_CACHE:
        return _MODEL_CACHE["ok"]
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
                h.update(chunk)
        ok = h.hexdigest() == MODEL_SHA256
    except Exception:
        ok = False
    _MODEL_CACHE["key"] = key
    _MODEL_CACHE["ok"] = ok
    return ok


# --------------------------------------------------------------------------
# manifest install/uninstall
# --------------------------------------------------------------------------

def read_manifest(game: Path):
    m = game / "dlssnr-amd-install.txt"
    if not m.is_file():
        return []
    return [l.strip() for l in m.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]


def uninstall_game(game: Path, log) -> None:
    manifest = game / "dlssnr-amd-install.txt"
    if not manifest.is_file():
        log(f"Nada a desinstalar: {manifest} nao encontrado.")
        return
    for entry in read_manifest(game):
        if not entry or entry.startswith("/") or entry.startswith("..") or "/../" in entry:
            log(f"  ignorando entrada suspeita: {entry}")
            continue
        target = game / entry
        try:
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            elif target.exists() or target.is_symlink():
                target.unlink()
            log(f"  removido: {entry}")
        except Exception as e:
            log(f"  ERRO ao remover {entry}: {e}")
    try:
        manifest.unlink()
    except Exception:
        pass
    log(f"Desinstalado de {game}.")


# --------------------------------------------------------------------------
# OptiScaler.ini
# --------------------------------------------------------------------------

def windows_path(p: str) -> str:
    p = p.replace("/", "\\")
    if len(p) > 1 and p[1] == ":":
        return p
    return "Z:" + p if p.startswith("\\") else p


def default_optiscaler_ini() -> str:
    """Texto padrao do OptiScaler.ini lido do zip da release (como install.sh instala)."""
    try:
        zips = sorted((HERE / "optiscaler").glob("OptiScaler*.zip"))
        if not zips:
            zips = sorted((HERE / "DLSSNR-AMD" / "artifacts" / "ref" / "downloads").glob("OptiScaler*.zip"))
        if not zips:
            zips = sorted(HERE.glob("**/OptiScaler*.zip"))
        if zips:
            with zipfile.ZipFile(zips[0]) as z:
                for info in z.infolist():
                    if info.filename.replace("\\", "/").endswith("OptiScaler.ini"):
                        return z.read(info).decode("utf-8", errors="replace")
    except Exception:
        pass
    return ""


def apply_ini_keys(ini_path: Path, overrides: dict, log) -> None:
    """overrides: {(secao, chave): valor}. Reescreve se existir, adiciona se faltar."""
    try:
        text = ini_path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        log(f"  AVISO: {ini_path.name} nao encontrado, pre-config ignorada.")
        return
    lines = text.splitlines()
    section = None
    seen = set()
    for i, line in enumerate(lines):
        m = re.match(r"\s*\[([^\]]+)\]", line)
        if m:
            section = m.group(1)
            continue
        e = re.match(r"\s*([A-Za-z0-9_]+)\s*=", line)
        if not e or section is None or line.lstrip().startswith(("#", ";")):
            continue
        key = (section, e.group(1))
        if key in overrides and key not in seen:
            seen.add(key)
            lines[i] = f"{e.group(1)}={overrides[key]}"
            log(f"  [{section}] {e.group(1)}={overrides[key]}")
    # adiciona chaves que nao existiam
    missing = {k: v for k, v in overrides.items() if k not in seen}
    if missing:
        by_section = {}
        for (s, k), v in missing.items():
            by_section.setdefault(s, []).append((k, v))
        for s, kvs in by_section.items():
            header = f"[{s}]"
            if header in lines:
                idx = lines.index(header) + 1
                for k, v in kvs:
                    lines.insert(idx, f"{k}={v}")
                    idx += 1
            else:
                lines += ["", header] + [f"{k}={v}" for k, v in kvs]
            for k, v in kvs:
                log(f"  [{s}] {k}={v} (adicionado)")
    ini_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def patch_ini_minimal(game: Path, log) -> None:
    """Equivalente a optiscaler/patch_ini.py: Enabled=true + NvngxPath."""
    ini = game / "OptiScaler.ini"
    wanted = {
        ("DlssNr", "Enabled"): "true",
        ("Libraries", "NvngxPath"): windows_path(str(game / "dlssnr_core.dll")),
    }
    apply_ini_keys(ini, wanted, log)


# --------------------------------------------------------------------------
# instalacao (espelha install.sh)
# --------------------------------------------------------------------------

class Installer:
    def __init__(self, here: Path = HERE):
        self.here = here

    def extract_optiscaler_release(self, tmp: Path, log) -> None:
        zips = sorted((self.here / "optiscaler").glob("OptiScaler*.zip"))
        if not zips:
            raise RuntimeError("Pacote sem OptiScaler*.zip em optiscaler/.")
        src = zips[0]
        root = tmp.resolve()
        count = 0
        with zipfile.ZipFile(src) as z:
            for info in z.infolist():
                name = info.filename.replace("\\", "/")
                target = os.path.normpath(os.path.join(str(root), name))
                if target != str(root) and not target.startswith(str(root) + os.sep):
                    raise RuntimeError(f"entrada do zip escapa do destino: {info.filename}")
                if name.endswith("/") or info.is_dir():
                    os.makedirs(target, exist_ok=True)
                    continue
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with z.open(info) as fsrc, open(target, "wb") as fdst:
                    shutil.copyfileobj(fsrc, fdst)
                count += 1
        log(f"  extraidos {count} arquivos de {src.name}")

    def run(self, game: Path, route: str, dll: str | None, ini_overrides: dict,
            log, progress, remember_dll: bool = True) -> str:
        """Executa a instalacao. progress(fracao 0..1, texto). Retorna launch options."""
        if route not in ROUTES:
            raise RuntimeError(f"rota desconhecida: {route}")
        if not game.is_dir():
            raise RuntimeError(f"pasta nao encontrada: {game}")
        game = game.resolve()
        if route in ("optiscaler", "gamescope") and not (self.here / "optiscaler").is_dir():
            raise RuntimeError("A rota OptiScaler / Gamescope so existe no pacote 64-bit.")
        bits = package_bits()
        pkg_model = self.here / "dlssnr-amd" / MODEL_NAME

        progress(0.02, "Verificando modelo...")
        model_src = None  # arquivo temporario quando o pacote nao pode guardar o modelo
        if model_ok(pkg_model):
            log(f"Modelo ok no pacote ({pkg_model.name}, SHA256 confere). Sem --dll.")
        elif dll:
            model_src = self._extract_and_keep(dll, pkg_model, log, progress)
        else:
            if pkg_model.exists():
                raise RuntimeError(f"{pkg_model} danificado ou de outra versao; extraia de novo com --dll (selecione a DLL).")
            raise RuntimeError("O pacote nao tem o modelo: selecione nvngx_dlssnr.dll (310.8.0) ou seu zip.")
        if remember_dll and dll:
            cfg = load_config()
            cfg["last_dll"] = dll
            cfg["last_game"] = str(game)
            cfg["last_route"] = route
            save_config(cfg)
            log(f"Caminho da DLL guardado para as proximas instalacoes: {dll}")

        manifest = game / "dlssnr-amd-install.txt"
        if manifest.exists():
            log("Instalacao anterior encontrada, removendo antes...")
            uninstall_game(game, log)

        progress(0.32, "Copiando dlssnr-amd/...")
        manifest.write_text("dlssnr-amd-install.txt\n", encoding="utf-8")
        records = ["dlssnr-amd-install.txt"]

        def record(e):
            records.append(e)
            with open(manifest, "a", encoding="utf-8") as f:
                f.write(e + "\n")

        def put_file(src: Path, name: str):
            shutil.copy2(src, game / name)
            record(name)
            log(f"  arquivo: {name}")

        def put_tree(src: Path, name: str):
            shutil.copytree(src, game / name, dirs_exist_ok=True)
            record(name + "/")
            log(f"  pasta: {name}/")

        put_tree(self.here / "dlssnr-amd", "dlssnr-amd")
        if model_src:
            shutil.copy2(model_src, game / "dlssnr-amd" / MODEL_NAME)
            log("  modelo copiado do temporario (pacote sem escrita).")

        progress(0.50, f"Instalando rota {route}...")
        if route in ("optiscaler", "gamescope"):
            with tempfile.TemporaryDirectory() as t:
                tmp = Path(t)
                self.extract_optiscaler_release(tmp, log)
                for junk in ("!! EXTRACT ALL FILES TO GAME FOLDER !!", "setup_windows.bat",
                             "setup_linux.sh", "nvngx.dll_dlssnr.dll"):
                    p = tmp / junk
                    if p.exists():
                        p.unlink()
                if (tmp / "OptiScaler.dll").exists():
                    (tmp / "OptiScaler.dll").rename(tmp / "dxgi.dll")
                items = sorted(tmp.iterdir(), key=lambda p: p.name)
                n = len(items)
                for i, f in enumerate(items):
                    if f.is_dir():
                        put_tree(f, f.name)
                    else:
                        put_file(f, f.name)
                    progress(0.50 + 0.30 * (i + 1) / max(n, 1), f"OptiScaler: {f.name}")
            for f in ("nvngx.dll_dlssnr.dll", "nvngx_dlssnr.dll", "dlssnr_core.dll"):
                put_file(self.here / "optiscaler" / f, f)
            log("Aplicando patch base do OptiScaler.ini (DlssNr Enabled + NvngxPath)...")
            patch_ini_minimal(game, log)
            if ini_overrides:
                log("Aplicando pre-configuracao do OptiScaler.ini...")
                apply_ini_keys(game / "OptiScaler.ini", ini_overrides, log)
            record("OptiScaler.log")
            record("dlssnr-amd.log")
            record("dlssnr-amd.ini")
            overrides = "dxgi=n,b"

            if route == "gamescope":
                log("Configurando rota Gamescope + OptiScaler...")
                # 1. Configura dlssnr-amd.ini com a secao [Gamescope] Enabled=1
                dlssnr_ini = game / "dlssnr-amd.ini"
                gamescope_cfg = {
                    ("Gamescope", "Enabled"): "1",
                    ("Gamescope", "Verbose"): "0",
                }
                if not dlssnr_ini.exists():
                    dlssnr_ini.write_text("[Gamescope]\nEnabled=1\nVerbose=0\n\n[DlssNr]\nEnabled=true\n", encoding="utf-8")
                    log("  criado dlssnr-amd.ini com [Gamescope] Enabled=1")
                else:
                    apply_ini_keys(dlssnr_ini, gamescope_cfg, log)

                # 2. Instala / disponibiliza o binario customizado do Gamescope
                gs_src = self.here / "gamescope" / "build" / "src" / "gamescope"
                if not gs_src.is_file():
                    gs_src = self.here / "gamescope" / "src" / "gamescope"
                if gs_src.is_file():
                    local_bin = Path.home() / ".local" / "bin"
                    local_bin.mkdir(parents=True, exist_ok=True)
                    gs_dst = local_bin / "gamescope-dlssnr"
                    try:
                        shutil.copy2(gs_src, gs_dst)
                        gs_dst.chmod(0o755)
                        log(f"  Binario Gamescope (DLSS-NR) instalado em: {gs_dst}")
                    except Exception as e:
                        log(f"  AVISO ao copiar binario gamescope ({e}); use o caminho completo.")
                else:
                    log("  AVISO: Binario gamescope compilado nao encontrado em gamescope/build/src/gamescope.")
        elif route == "reshade":
            items = sorted((self.here / "reshade").iterdir(), key=lambda p: p.name)
            for i, f in enumerate(items):
                if f.is_dir():
                    put_tree(f, f.name)
                else:
                    put_file(f, f.name)
                progress(0.50 + 0.30 * (i + 1) / max(len(items), 1), f"ReShade: {f.name}")
            overrides = "dxgi=n,b"
        else:  # vulkan / dx9
            items = sorted(list((self.here / "reshade").iterdir()) + list((self.here / "vulkan").iterdir()),
                           key=lambda p: p.name)
            items = [f for f in items if f.name not in ("dxgi.dll", "ReShadePreset-d3d9.ini")]
            for i, f in enumerate(items):
                if f.is_dir():
                    put_tree(f, f.name)
                else:
                    put_file(f, f.name)
                progress(0.50 + 0.30 * (i + 1) / max(len(items), 1), f"{f.name}")
            if route == "dx9":
                shutil.copy2(self.here / "vulkan" / "ReShadePreset-d3d9.ini", game / "ReShadePreset.ini")
                record("ReShadePreset.ini")
            overrides = "winevulkan=n,b;vulkan-1=n,b"
        if route not in ("optiscaler", "gamescope"):
            record("dlssnr-amd.ini")
            record("dlssnr-amd.log")
            record("ReShade.log")

        progress(0.97, "Finalizando...")
        try:
            if model_src and os.path.exists(model_src):
                os.unlink(model_src)
        except Exception:
            pass
        if route == "gamescope":
            gs_bin = Path.home() / ".local" / "bin" / "gamescope-dlssnr"
            launch = f'WINEDLLOVERRIDES="{overrides}" NR_GAMESCOPE_BRIDGE=1 "{gs_bin}" -W 1920 -H 1080 -w 1920 -h 1080 -f -- %command%'
        else:
            launch = f'WINEDLLOVERRIDES="{overrides}" %command%'
        log(f"Instalada a rota {route} em: {game}")
        log(f"Steam launch options:  {launch}")
        if route == "gamescope":
            log("Gamescope + OptiScaler instalado com sucesso!")
            log("  1. No Steam, configure os Launch Options do jogo:")
            log(f"     {launch}")
            log("  2. Inicie o jogo pelo Steam.")
            log("     O Gamescope cria o socket IPC e exporta o frame com external memory;")
            log("     O OptiScaler conecta automaticamente e processa a imagem;")
            log("     O Gamescope recebe o frame filtrado pelo modelo e apresenta na tela.")
            log("  3. Ajuste a resolucao (-W -H) conforme seu monitor.")
        elif route == "optiscaler":
            log("No jogo, ative DLSS/FSR/XeSS; Insert abre o menu; NR tem pagina propria.")
        else:
            log("No jogo, Home abre o ReShade (Add-ons), ou edite dlssnr-amd.ini.")
        progress(1.0, "Concluido.")
        return launch

    def _extract_and_keep(self, dll: str, pkg_model: Path, log, progress):
        log(f"Extraindo o modelo de {dll} ... (~20s)")
        progress(0.08, "Extraindo modelo da DLL NVIDIA...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tf:
            model_tmp = tf.name
        part = str(pkg_model) + ".part"
        try:
            proc = subprocess.Popen(
                ["bash", str(self.here / "model-tools" / "extract_model.sh"), dll, model_tmp],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                log("  [extract] " + line.rstrip())
            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError("extract_model.sh falhou (DLL precisa ser nvngx_dlssnr 310.8.0).")
            if not model_ok(Path(model_tmp)):
                raise RuntimeError("Modelo extraido nao confere (SHA256); DLL errada?")
            try:
                shutil.copy2(model_tmp, part)
                os.replace(part, pkg_model)
                os.chmod(pkg_model, 0o644)
                log(f"Modelo guardado em {pkg_model}; proximas instalacoes nao precisam de --dll.")
                return None
            except Exception as e:
                try:
                    if os.path.exists(part):
                        os.unlink(part)
                except Exception:
                    pass
                log(f"AVISO: nao foi possivel guardar no pacote ({e}); usando temporario.")
                return model_tmp
        except Exception:
            try:
                if os.path.exists(model_tmp):
                    os.unlink(model_tmp)
            except Exception:
                pass
            raise


# --------------------------------------------------------------------------
# upstream: verificar / atualizar
# --------------------------------------------------------------------------

def fetch_json(url: str, timeout: int = 20) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "dlssnr-amd-installer-gui",
                                               "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", errors="replace"))


def check_upstream(log=None):
    """Retorna dict com tag, assets, body ou levanta RuntimeError."""
    data = fetch_json(UPSTREAM_API)
    return {"tag": data.get("tag_name", "?"), "name": data.get("name", ""),
            "body": data.get("body", ""), "html_url": data.get("html_url", UPSTREAM_URL),
            "assets": [{"name": a.get("name", ""), "url": a.get("browser_download_url", ""),
                        "size": a.get("size", 0)} for a in data.get("assets", [])]}


def download_with_progress(url: str, dest: Path, log, progress):
    req = urllib.request.Request(url, headers={"User-Agent": "dlssnr-amd-installer-gui"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1024 * 256)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                progress(done / total, f"Baixando... {done/1e6:.1f}/{total/1e6:.1f} MiB")
            else:
                progress(0.0, f"Baixando... {done/1e6:.1f} MiB")
            if log and done % (16 * 1024 * 1024) < 256 * 1024:
                log(f"  {done/1e6:.1f} MiB" + (f" de {total/1e6:.1f}" if total else ""))


def update_package(asset_url: str, log, progress) -> None:
    """Baixa o tarball, preserva dlssnr.bin e extrai por cima do pacote."""
    import tarfile
    pkg_model = HERE / "dlssnr-amd" / MODEL_NAME
    kept = None
    if model_ok(pkg_model):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tf:
            kept = tf.name
        shutil.copy2(pkg_model, kept)
        log("Modelo atual preservado para reuso apos a atualizacao.")
    with tempfile.TemporaryDirectory() as t:
        tgz = Path(t) / "update.tar.gz"
        progress(0.02, "Baixando atualizacao...")
        download_with_progress(asset_url, tgz, log,
                               lambda f, m: progress(0.02 + 0.75 * f, m))
        progress(0.80, "Extraindo atualizacao...")
        with tarfile.open(tgz, "r:gz") as tar:
            members = tar.getmembers()
            n = len(members)
            for i, m in enumerate(members):
                # releases sao empacotados com prefixo ./ e sem pasta raiz;
                # normaliza e ignora entradas perigosas/vazias
                name = m.name
                if name.startswith("/") or name.startswith("\\"):
                    continue  # absoluto: ignora
                if name.startswith("./"):
                    name = name[2:]
                name = name.lstrip("/")
                if not name or name == "." or name.startswith("..") or "/../" in name:
                    continue
                m.name = name
                tar.extract(m, path=str(HERE))
                if i % 50 == 0:
                    progress(0.80 + 0.15 * (i + 1) / max(n, 1), f"Extraindo... ({i+1}/{n})")
                    log(f"  ... {i+1}/{n}")
        log(f"Pacote atualizado a partir de {asset_url}")
    if kept:
        if not model_ok(pkg_model):
            try:
                shutil.copy2(kept, pkg_model)
                log("Modelo anterior restaurado no pacote atualizado.")
            except Exception as e:
                log(f"AVISO: nao deu para restaurar o modelo ({e}); use --dll uma vez.")
        else:
            log("Pacote novo ja traz modelo valido; o anterior nao foi necessario.")
        try:
            os.unlink(kept)
        except Exception:
            pass
    progress(1.0, "Atualizacao concluida. Reinicie o instalador.")


# ==========================================================================
# GUI (tkinter)
# ==========================================================================

PRECONFIG_FIELDS = [
    # (rotulo, secao, chave, tipo, opcoes)
    ("Gamescope Bridge", "Gamescope", "Enabled", "combo", ["auto", "1", "0"]),
    ("Gamescope Verbose", "Gamescope", "Verbose", "combo", ["auto", "0", "1"]),
    ("Upscaler DX12", "Upscalers", "Dx12Upscaler", "combo", ["auto", "dlss", "ffx", "fsr21", "fsr22", "xess"]),
    ("Upscaler DX11", "Upscalers", "Dx11Upscaler", "combo", ["auto", "fsr22", "fsr31", "xess", "dlss", "ffx_12", "dlss_12"]),
    ("Upscaler Vulkan", "Upscalers", "VulkanUpscaler", "combo", ["auto", "fsr22", "fsr21", "ffx", "xess"]),
    ("Spoofing DXGI (false = tenta se NR ficar em Waiting)", "Spoofing", "Dxgi", "combo", ["auto", "true", "false"]),
    ("Streamline spoofing", "Spoofing", "StreamlineSpoofing", "combo", ["auto", "true", "false"]),
    ("NR antes do upscaler (RunBeforeSR)", "DlssNr", "RunBeforeSR", "combo", ["auto", "true", "false"]),
    ("FinishedPicture (apos efeitos, DX12)", "DlssNr", "FinishedPicture", "combo", ["false", "true"]),
    ("Upscaler privado (DeferredDLSS)", "DlssNr", "PrivateUpscaler", "combo", ["auto", "0", "1", "2", "3"]),
    ("Frame Generation", "FrameGen", "Enabled", "combo", ["auto", "true", "false"]),
    ("FG input", "FrameGen", "FGInput", "combo", ["auto", "nofg", "dlssg", "nvngxfg", "fsrfg", "upscaler", "fsrfg30"]),
    ("FG output", "FrameGen", "FGOutput", "combo", ["auto", "nofg", "fsrfg", "xefg", "dlssg"]),
]

_gs_bin = Path.home() / ".local" / "bin" / "gamescope-dlssnr"
LAUNCH_HINTS = {
    "gamescope": f'WINEDLLOVERRIDES="dxgi=n,b" NR_GAMESCOPE_BRIDGE=1 "{_gs_bin}" -W 1920 -H 1080 -w 1920 -h 1080 -f -- %command%',
    "optiscaler": 'WINEDLLOVERRIDES="dxgi=n,b" %command%',
    "reshade": 'WINEDLLOVERRIDES="dxgi=n,b" %command%',
    "vulkan": 'WINEDLLOVERRIDES="winevulkan=n,b;vulkan-1=n,b" %command%',
    "dx9": 'WINEDLLOVERRIDES="winevulkan=n,b;vulkan-1=n,b" %command%',
}


def run_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    cfg = load_config()
    inst = Installer()

    root = tk.Tk()
    root.title(f"DLSSNR-AMD Installer GUI  v{CURRENT_VERSION}-{package_arch()}")
    root.geometry("860x760")

    msg_q = queue.Queue()

    # ---- estado ----
    var_game = tk.StringVar(value=cfg.get("last_game", ""))
    var_route = tk.StringVar(value=cfg.get("last_route", "gamescope"))
    var_dll = tk.StringVar(value=cfg.get("last_dll", ""))
    var_remember = tk.BooleanVar(value=True)
    var_exe = tk.StringVar(value="")
    pre_vars = {}
    for label, sec, key, _typ, opts in PRECONFIG_FIELDS:
        dflt = cfg.get("ini", {}).get(f"{sec}.{key}", "auto" if "auto" in opts else opts[0])
        pre_vars[(sec, key)] = tk.StringVar(value=dflt)

    # ---- layout ----
    main = ttk.Frame(root, padding=10)
    main.pack(fill="both", expand=True)
    main.columnconfigure(1, weight=1)

    r = 0
    ttk.Label(main, text="Pasta do jogo (com o .exe):").grid(row=r, column=0, sticky="w")
    f_game = ttk.Frame(main)
    f_game.grid(row=r, column=1, columnspan=2, sticky="ew")
    f_game.columnconfigure(0, weight=1)
    ent_game = ttk.Entry(f_game, textvariable=var_game)
    ent_game.grid(row=0, column=0, sticky="ew")
    ttk.Button(f_game, text="Procurar...",
               command=lambda: var_game.set(filedialog.askdirectory(title="Pasta do jogo") or var_game.get())
               ).grid(row=0, column=1, padx=(6, 0))
    r += 1
    lbl_exe = ttk.Label(main, text="Executaveis: (escolha para confirmar)", foreground="gray")
    lbl_exe.grid(row=r, column=0, sticky="w", pady=(4, 0))
    lst_exe = tk.Listbox(main, height=3)
    lst_exe.grid(row=r + 1, column=0, columnspan=3, sticky="ew")
    r += 2
    lbl_warn = ttk.Label(main, text="", foreground="red", wraplength=800, justify="left")
    lbl_warn.grid(row=r, column=0, columnspan=3, sticky="w")
    r += 1

    ttk.Separator(main, orient="horizontal").grid(row=r, column=0, columnspan=3, sticky="ew", pady=8)
    r += 1
    ttk.Label(main, text="Rota:").grid(row=r, column=0, sticky="nw")
    f_route = ttk.Frame(main)
    f_route.grid(row=r, column=1, columnspan=2, sticky="w")
    for i, (rid, desc) in enumerate(ROUTES.items()):
        ttk.Radiobutton(f_route, text=f"{rid}  —  {desc}", variable=var_route, value=rid).grid(
            row=i, column=0, sticky="w")
    r += 1
    lbl_launch = ttk.Label(main, text="", foreground="blue")
    lbl_launch.grid(row=r, column=0, columnspan=3, sticky="w")
    r += 1

    ttk.Separator(main, orient="horizontal").grid(row=r, column=0, columnspan=3, sticky="ew", pady=8)
    r += 1
    ttk.Label(main, text="DLL NVIDIA (nvngx_dlssnr 310.8.0):").grid(row=r, column=0, sticky="w")
    f_dll = ttk.Frame(main)
    f_dll.grid(row=r, column=1, columnspan=2, sticky="ew")
    f_dll.columnconfigure(0, weight=1)
    ttk.Entry(f_dll, textvariable=var_dll).grid(row=0, column=0, sticky="ew")
    ttk.Button(f_dll, text="Procurar...",
               command=lambda: var_dll.set(filedialog.askopenfilename(
                   title="nvngx_dlssnr.dll ou zip",
                   filetypes=[("DLL/zip", "*.dll *.zip"), ("Todos", "*.*")]) or var_dll.get())
               ).grid(row=0, column=1, padx=(6, 0))
    r += 1
    lbl_model = ttk.Label(main, text="", foreground="gray")
    lbl_model.grid(row=r, column=1, columnspan=2, sticky="w")
    ttk.Checkbutton(main, text="Lembrar caminho da DLL (extrai 1x e reutiliza)",
                    variable=var_remember).grid(row=r + 1, column=1, sticky="w")
    r += 2

    ttk.Separator(main, orient="horizontal").grid(row=r, column=0, columnspan=3, sticky="ew", pady=8)
    r += 1
    ttk.Label(main, text="Pre-config OptiScaler.ini (rota optiscaler):").grid(row=r, column=0, columnspan=3, sticky="w")
    r += 1
    f_pre = ttk.Frame(main)
    f_pre.grid(row=r, column=0, columnspan=3, sticky="ew")
    for i, (label, sec, key, _typ, opts) in enumerate(PRECONFIG_FIELDS):
        ttk.Label(f_pre, text=label).grid(row=i, column=0, sticky="w", padx=(0, 6))
        ttk.Combobox(f_pre, textvariable=pre_vars[(sec, key)], values=opts, width=14).grid(row=i, column=1, sticky="w")
        ttk.Label(f_pre, text=f"[{sec}] {key}", foreground="gray").grid(row=i, column=2, sticky="w", padx=(8, 0))
    f_pre.columnconfigure(0, weight=1)
    r += 1
    ttk.Button(main, text="Editar OptiScaler.ini completo...",
               command=lambda: open_full_ini_editor(root, pre_vars)).grid(row=r, column=0, columnspan=3, sticky="w", pady=4)
    r += 1

    ttk.Separator(main, orient="horizontal").grid(row=r, column=0, columnspan=3, sticky="ew", pady=8)
    r += 1
    f_btn = ttk.Frame(main)
    f_btn.grid(row=r, column=0, columnspan=3, sticky="ew", pady=4)
    btn_install = ttk.Button(f_btn, text="Instalar")
    btn_remove = ttk.Button(f_btn, text="Desinstalar")
    btn_check = ttk.Button(f_btn, text="Verificar atualização")
    btn_update = ttk.Button(f_btn, text="Atualizar pacote", state="disabled")
    for b in (btn_install, btn_remove, btn_check, btn_update):
        b.pack(side="left", padx=4)
    lbl_up = ttk.Label(main, text=f"Pacote local: v{CURRENT_VERSION}-{package_arch()}  |  {UPSTREAM_URL}",
                       foreground="gray", wraplength=800)
    lbl_up.grid(row=r + 1, column=0, columnspan=3, sticky="w")
    r += 2

    prog = ttk.Progressbar(main, mode="determinate", maximum=100)
    prog.grid(row=r, column=0, columnspan=3, sticky="ew")
    r += 1
    txt = tk.Text(main, height=14, wrap="word", state="disabled")
    txt.grid(row=r, column=0, columnspan=3, sticky="nsew")
    main.rowconfigure(r, weight=1)
    scr = ttk.Scrollbar(main, command=txt.yview)
    scr.grid(row=r, column=3, sticky="ns")
    txt.configure(yscrollcommand=scr.set)

    # ---- helpers de UI ----
    def log(msg):
        msg_q.put(("log", str(msg)))

    def progress(frac, msg=""):
        msg_q.put(("progress", (max(0.0, min(1.0, frac)), str(msg))))

    def flush_queue():
        try:
            while True:
                kind, payload = msg_q.get_nowait()
                if kind == "exec":
                    # Callable criado na worker, mas EXECUTADO aqui (thread da UI).
                    # Nunca toque em widgets tkinter fora desta funcao.
                    try:
                        payload()
                    except Exception as e:
                        try:
                            txt.configure(state="normal")
                            txt.insert("end", f"ERRO interno UI: {e}\n")
                            txt.see("end")
                            txt.configure(state="disabled")
                        except Exception:
                            pass
                    continue
                txt.configure(state="normal")
                if kind == "log":
                    txt.insert("end", payload + "\n")
                    txt.see("end")
                elif kind == "progress":
                    frac, m = payload
                    prog["value"] = frac * 100
                    if m:
                        root.title(f"DLSSNR-AMD Installer — {m}")
                txt.configure(state="disabled")
        except queue.Empty:
            pass
        root.after(120, flush_queue)

    _scan_job = {"id": None}

    def _schedule_scan(*_a):
        # Debounce: digitar nao re-escaneia nem re-hasheia a cada tecla.
        try:
            if _scan_job["id"] is not None:
                root.after_cancel(_scan_job["id"])
        except Exception:
            pass
        _scan_job["id"] = root.after(250, refresh_scan)

    def refresh_scan(*_a):
        game = Path(var_game.get()).expanduser() if var_game.get() else None
        lst_exe.delete(0, "end")
        warns = []
        if game and game.is_dir():
            exes, ue = scan_game_folder(game)
            for name, b in exes:
                lst_exe.insert("end", f"{name}  [{b}-bit]")
            if not exes:
                warns.append("Nenhum *.exe na pasta. Confira se e a pasta do executavel.")
            if ue:
                warns.append("Parece a pasta raiz de um jogo Unreal Engine; "
                             "instale em <projeto>/Binaries/Win64 (*-Shipping.exe).")
            bits = package_bits()
            ebits = sorted({b for _, b in exes if b in ("32", "64")})
            if ebits and str(bits) not in ebits:
                warns.append(f"Pacote {bits}-bit, mas o exe e {'/'.join(ebits)}-bit; "
                             "DLLs da outra arquitetura nao carregam.")
            man = game / "dlssnr-amd-install.txt"
            if man.exists():
                warns.append(f"Instalacao anterior detectada ({man.name}); sera removida antes.")
        lbl_warn.config(text="\n".join(warns))
        lbl_launch.config(text="Steam launch options:  " + LAUNCH_HINTS.get(var_route.get(), ""))
        pm = HERE / "dlssnr-amd" / MODEL_NAME
        if model_ok(pm):
            lbl_model.config(text=f"Modelo no pacote: OK ({pm.stat().st_size/1e6:.1f} MiB, SHA256 confere). DLL nao necessaria.",
                             foreground="green")
        elif pm.exists():
            lbl_model.config(text="Modelo no pacote: DANIFICADO — selecione a DLL para extrair de novo.",
                             foreground="red")
        else:
            saved = var_dll.get().strip()
            hint = f" Caminho guardado: {saved}" if saved and Path(saved).exists() else ""
            lbl_model.config(text="Modelo no pacote: ausente — selecione a DLL (extrai 1x e guarda)." + hint,
                             foreground="orange")

    def set_busy(busy):
        state = "disabled" if busy else "normal"
        for b in (btn_install, btn_remove, btn_check, btn_update):
            b.config(state=state)
        if not busy:
            btn_update.config(state="normal" if getattr(btn_update, "_has_update", False) else "disabled")

    def collect_ini_overrides():
        ov = {}
        for (sec, key), v in pre_vars.items():
            val = v.get().strip()
            if val and val != "auto":
                ov[(sec, key)] = val
        full = getattr(root, "_full_ini_text", None)
        if full:
            # o editor completo tem prioridade: extrai pares secao/chave
            for m in re.finditer(r"\[([^\]]+)\](.*?)(?=\n\[|\Z)", full, re.S):
                sec = m.group(1).strip()
                for line in m.group(2).splitlines():
                    e = re.match(r"\s*([A-Za-z0-9_]+)\s*=\s*(.+?)\s*$", line)
                    if e and not line.lstrip().startswith(("#", ";")):
                        ov[(sec, e.group(1))] = e.group(2)
        cfg2 = load_config()
        cfg2["ini"] = {f"{s}.{k}": v for (s, k), v in ov.items()}
        save_config(cfg2)
        return ov

    def do_install():
        game = Path(var_game.get()).expanduser()
        route = var_route.get()
        dll = var_dll.get().strip() or None
        if not var_game.get().strip():
            messagebox.showwarning("Instalador", "Escolha a pasta do jogo.")
            return
        # reutiliza caminho guardado se o campo estiver vazio
        if not dll:
            saved = load_config().get("last_dll", "")
            if saved and Path(saved).exists():
                dll = saved
                var_dll.set(saved)
                log(f"Reutilizando caminho guardado da DLL: {dll}")
        ov = collect_ini_overrides()
        set_busy(True)
        progress(0.0, "Iniciando instalacao...")
        def worker():
            try:
                launch = inst.run(game, route, dll, ov, log, progress,
                                  remember_dll=var_remember.get())
                msg_q.put(("log", f"\nOK. Use: {launch}"))
            except Exception as e:
                msg_q.put(("log", f"\nERRO: {e}"))
            finally:
                msg_q.put(("progress", (1.0, "Pronto.")))
                # NUNCA root.after() a partir da worker (tkinter nao e thread-safe
                # e o after pode nunca disparar -> botoes presos em disabled).
                msg_q.put(("exec", lambda: (set_busy(False), refresh_scan())))
        threading.Thread(target=worker, daemon=True).start()

    def do_remove():
        game = Path(var_game.get()).expanduser()
        if not var_game.get().strip() or not game.is_dir():
            messagebox.showwarning("Instalador", "Escolha a pasta do jogo.")
            return
        set_busy(True)
        def worker():
            try:
                uninstall_game(game, log)
            except Exception as e:
                msg_q.put(("log", f"ERRO: {e}"))
            finally:
                msg_q.put(("exec", lambda: (set_busy(False), refresh_scan())))
        threading.Thread(target=worker, daemon=True).start()

    latest = {}

    def do_check():
        set_busy(True)
        log(f"Verificando {UPSTREAM_API} ...")
        def worker():
            try:
                data = check_upstream()
                tag = data["tag"]
                msg_q.put(("log", f"Upstream: {tag} ({data['name']}) — {data['html_url']}"))
                if data.get("body"):
                    msg_q.put(("log", "Notas: " + data["body"][:600].replace(chr(10), " ")))
                cur = f"v{CURRENT_VERSION}"
                if tag and tag != cur:
                    msg_q.put(("log", f"ATUALIZACAO disponivel: {cur} -> {tag}"))
                    arch = package_arch()
                    cands = [a for a in data["assets"] if arch in a["name"] and a["name"].endswith(".tar.gz")]
                    if cands:
                        asset = cands[0]
                        msg_q.put(("log", f"  pacote: {asset['name']} ({asset['size']/1e6:.1f} MiB)"))

                        def _apply():
                            latest.update(data)
                            latest["asset"] = asset
                            btn_update._has_update = True
                            set_busy(False)
                        msg_q.put(("exec", _apply))
                    else:
                        msg_q.put(("log", f"  nenhum asset *{arch}.tar.gz no release."))

                        def _no_asset():
                            latest.update(data)
                            set_busy(False)
                        msg_q.put(("exec", _no_asset))
                else:
                    msg_q.put(("log", "Ja esta na versao mais recente."))

                    def _same():
                        latest.update(data)
                        set_busy(False)
                    msg_q.put(("exec", _same))
            except Exception as e:
                msg_q.put(("log", f"ERRO ao verificar upstream: {e}"))
                msg_q.put(("exec", lambda: set_busy(False)))
        threading.Thread(target=worker, daemon=True).start()

    def do_update():
        asset = latest.get("asset")
        if not asset:
            messagebox.showinfo("Atualizar", "Clique em Verificar atualização primeiro.")
            return
        if not messagebox.askyesno("Atualizar",
                                    f"Baixar e aplicar {asset['name']}?\nO modelo dlssnr.bin sera preservado."):
            return
        set_busy(True)
        def worker():
            try:
                update_package(asset["url"], log, progress)
                msg_q.put(("log", "Reinicie o instalador para usar o pacote novo."))
            except Exception as e:
                msg_q.put(("log", f"ERRO na atualizacao: {e}"))
            finally:
                msg_q.put(("exec", lambda: (set_busy(False), refresh_scan())))
        threading.Thread(target=worker, daemon=True).start()

    btn_install.config(command=do_install)
    btn_remove.config(command=do_remove)
    btn_check.config(command=do_check)
    btn_update.config(command=do_update)
    var_game.trace_add("write", _schedule_scan)
    var_route.trace_add("write", _schedule_scan)
    var_dll.trace_add("write", _schedule_scan)

    refresh_scan()
    log(f"DLSSNR-AMD installer GUI — pacote v{CURRENT_VERSION}-{package_arch()} ({package_bits()}-bit).")
    log("1) Escolha a pasta do jogo (a do .exe). 2) Escolha a rota. 3) DLL 1x. 4) Pre-configure e Instale.")
    flush_queue()
    root.mainloop()


# ==========================================================================
# GUI moderna (CustomTkinter). Cai para run_gui() classica se indisponivel.
# ==========================================================================

def run_gui_ctk():
    import tkinter as tk
    from tkinter import filedialog, messagebox

    import customtkinter as ctk

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    cfg = load_config()
    inst = Installer()

    root = ctk.CTk()
    root.title(f"DLSSNR-AMD  ·  v{CURRENT_VERSION}-{package_arch()}")
    root.geometry("1020x700")
    root.minsize(900, 620)
    root.grid_columnconfigure(1, weight=1)
    root.grid_rowconfigure(0, weight=1)

    msg_q = queue.Queue()

    var_game = tk.StringVar(value=cfg.get("last_game", ""))
    var_route = tk.StringVar(value=cfg.get("last_route", "gamescope"))
    var_dll = tk.StringVar(value=cfg.get("last_dll", ""))
    var_exe = tk.StringVar(value="")
    var_remember = tk.BooleanVar(value=True)
    pre_vars = {}
    for _label, sec, key, _typ, opts in PRECONFIG_FIELDS:
        dflt = cfg.get("ini", {}).get(f"{sec}.{key}", "auto" if "auto" in opts else opts[0])
        pre_vars[(sec, key)] = tk.StringVar(value=dflt)

    # ---- sidebar ----
    side = ctk.CTkFrame(root, width=230, corner_radius=0)
    side.grid(row=0, column=0, sticky="nsew")
    side.grid_propagate(False)

    ctk.CTkLabel(side, text="DLSSNR-AMD", font=("Sans", 22, "bold")).pack(padx=16, pady=(20, 0), anchor="w")
    ctk.CTkLabel(side, text=f"v{CURRENT_VERSION}-{package_arch()}  ·  {package_bits()}-bit",
                 font=("Sans", 12), text_color="gray").pack(padx=16, pady=(0, 16), anchor="w")

    btn_install = ctk.CTkButton(side, text="Instalar", height=38,
                                font=("Sans", 14, "bold"))
    btn_install.pack(padx=16, pady=4, fill="x")
    btn_remove = ctk.CTkButton(side, text="Desinstalar", height=34, fg_color="transparent",
                               border_width=1, text_color=("gray10", "gray90"))
    btn_remove.pack(padx=16, pady=4, fill="x")
    btn_check = ctk.CTkButton(side, text="Verificar atualização", height=34, fg_color="transparent",
                              border_width=1, text_color=("gray10", "gray90"))
    btn_check.pack(padx=16, pady=4, fill="x")
    btn_update = ctk.CTkButton(side, text="Atualizar pacote", height=34, fg_color="transparent",
                               border_width=1, text_color=("gray10", "gray90"))
    btn_update.pack(padx=16, pady=4, fill="x")
    btn_update.configure(state="disabled")

    lbl_up = ctk.CTkLabel(side, text="Upstream: não verificado", font=("Sans", 11),
                          text_color="gray", wraplength=198, justify="left")
    lbl_up.pack(padx=16, pady=(12, 0), anchor="w")

    ctk.CTkLabel(side, text="", height=10).pack(fill="x", expand=True)  # espaco
    ctk.CTkLabel(side, text="Aparência", font=("Sans", 11), text_color="gray").pack(padx=16, anchor="w")
    ctk.CTkOptionMenu(side, values=["Dark", "Light", "System"],
                      command=ctk.set_appearance_mode).pack(padx=16, pady=(0, 16), fill="x")

    # ---- area principal ----
    main = ctk.CTkFrame(root, corner_radius=0, fg_color="transparent")
    main.grid(row=0, column=1, sticky="nsew", padx=16, pady=12)
    main.grid_columnconfigure(0, weight=1)
    main.grid_rowconfigure(1, weight=1)

    ctk.CTkLabel(main, text="Instalador", font=("Sans", 20, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 8))

    tabs = ctk.CTkTabview(main)
    tabs.grid(row=1, column=0, sticky="nsew")
    for name in ("Jogo", "Modelo", "OptiScaler.ini", "Log"):
        tabs.add(name)
        tabs.tab(name).grid_columnconfigure(0, weight=1)

    # ---- aba Jogo ----
    tj = tabs.tab("Jogo")
    ctk.CTkLabel(tj, text="Pasta do jogo (a que contém o .exe)", font=("Sans", 13, "bold")).pack(anchor="w", padx=12, pady=(12, 4))
    row_game = ctk.CTkFrame(tj, fg_color="transparent")
    row_game.pack(fill="x", padx=12)
    row_game.grid_columnconfigure(0, weight=1)
    ent_game = ctk.CTkEntry(row_game, textvariable=var_game, placeholder_text="/path/to/steamapps/common/<jogo>/Binaries/Win64")
    ent_game.grid(row=0, column=0, sticky="ew")
    ctk.CTkButton(row_game, text="Procurar", width=100,
                  command=lambda: var_game.set(filedialog.askdirectory(title="Pasta do jogo") or var_game.get())
                  ).grid(row=0, column=1, padx=(8, 0))

    ctk.CTkLabel(tj, text="Executáveis encontrados", font=("Sans", 13, "bold")).pack(anchor="w", padx=12, pady=(12, 4))
    exe_box = ctk.CTkScrollableFrame(tj, height=90)
    exe_box.pack(fill="x", padx=12)
    lbl_warn = ctk.CTkLabel(tj, text="", font=("Sans", 12), text_color="#E74C3C",
                            wraplength=640, justify="left")
    lbl_warn.pack(anchor="w", padx=12, pady=6)

    ctk.CTkLabel(tj, text="Rota", font=("Sans", 13, "bold")).pack(anchor="w", padx=12, pady=(4, 4))
    route_box = ctk.CTkFrame(tj, fg_color="transparent")
    route_box.pack(fill="x", padx=12, pady=(0, 4))
    for rid, desc in ROUTES.items():
        r = ctk.CTkRadioButton(route_box, text=f"{rid}   —   {desc}", variable=var_route, value=rid,
                               font=("Sans", 13))
        r.pack(anchor="w", pady=2)
    lbl_launch = ctk.CTkLabel(tj, text="", font=("Sans", 12, "bold"), text_color="#3498DB",
                              wraplength=640, justify="left")
    lbl_launch.pack(anchor="w", padx=12, pady=6)

    # ---- aba Modelo ----
    tm = tabs.tab("Modelo")
    card = ctk.CTkFrame(tm)
    card.pack(fill="x", padx=12, pady=12)
    lbl_model = ctk.CTkLabel(card, text="", font=("Sans", 13), wraplength=600, justify="left")
    lbl_model.pack(anchor="w", padx=14, pady=12)
    ctk.CTkLabel(tm, text="DLL da NVIDIA (nvngx_dlssnr 310.8.0 — .dll ou .zip, só na 1ª vez)",
                 font=("Sans", 13, "bold")).pack(anchor="w", padx=12, pady=(0, 4))
    row_dll = ctk.CTkFrame(tm, fg_color="transparent")
    row_dll.pack(fill="x", padx=12)
    row_dll.grid_columnconfigure(0, weight=1)
    ctk.CTkEntry(row_dll, textvariable=var_dll, placeholder_text="/caminho/nvngx_dlssnr_310.8.0.zip").grid(
        row=0, column=0, sticky="ew")
    ctk.CTkButton(row_dll, text="Procurar", width=100,
                  command=lambda: var_dll.set(filedialog.askopenfilename(
                      title="nvngx_dlssnr.dll ou zip",
                      filetypes=[("DLL/zip", "*.dll *.zip"), ("Todos", "*.*")]) or var_dll.get())
                  ).grid(row=0, column=1, padx=(8, 0))
    ctk.CTkCheckBox(tm, text="Lembrar caminho da DLL (extrai 1x e reutiliza)", variable=var_remember,
                    font=("Sans", 13)).pack(anchor="w", padx=12, pady=10)
    ctk.CTkLabel(tm, text="O modelo extraído fica em dlssnr-amd/dlssnr.bin no pacote\n"
                          "e é verificado por SHA256 a cada instalação.",
                 font=("Sans", 12), text_color="gray", justify="left").pack(anchor="w", padx=12)

    # ---- aba OptiScaler.ini ----
    ti = tabs.tab("OptiScaler.ini")
    ctk.CTkLabel(ti, text="Pré-configuração (aplicada após o patch base na rota optiscaler)",
                 font=("Sans", 13, "bold")).pack(anchor="w", padx=12, pady=(12, 4))
    pre_box = ctk.CTkScrollableFrame(ti)
    pre_box.pack(fill="both", expand=True, padx=12, pady=(0, 8))
    pre_box.grid_columnconfigure(1, weight=1)
    for i, (label, sec, key, _typ, opts) in enumerate(PRECONFIG_FIELDS):
        ctk.CTkLabel(pre_box, text=label, font=("Sans", 12)).grid(row=i, column=0, sticky="w", padx=(4, 8), pady=3)
        ctk.CTkComboBox(pre_box, variable=pre_vars[(sec, key)], values=opts, width=130).grid(
            row=i, column=1, sticky="w", pady=3)
        ctk.CTkLabel(pre_box, text=f"[{sec}] {key}", font=("Sans", 11),
                     text_color="gray").grid(row=i, column=2, sticky="w", padx=(8, 4), pady=3)
    ctk.CTkButton(ti, text="Editar OptiScaler.ini completo…", fg_color="transparent", border_width=1,
                  text_color=("gray10", "gray90"),
                  command=lambda: open_full_ini_editor_ctk(root, pre_vars)).pack(anchor="w", padx=12, pady=(0, 12))

    # ---- aba Log ----
    tl = tabs.tab("Log")
    tl.grid_rowconfigure(0, weight=1)
    tl.grid_columnconfigure(0, weight=1)
    txt = ctk.CTkTextbox(tl, font=("Mono", 12))
    txt.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
    txt.configure(state="disabled")

    # ---- barra de progresso (sempre visivel) ----
    pframe = ctk.CTkFrame(main, fg_color="transparent")
    pframe.grid(row=2, column=0, sticky="ew", pady=(8, 0))
    pframe.grid_columnconfigure(0, weight=1)
    lbl_prog = ctk.CTkLabel(pframe, text="Pronto.", font=("Sans", 12), text_color="gray")
    lbl_prog.grid(row=0, column=0, sticky="w")
    prog = ctk.CTkProgressBar(pframe, height=12)
    prog.grid(row=1, column=0, sticky="ew", pady=(2, 0))
    prog.set(0.0)

    # ---- logica ----
    def log(msg):
        msg_q.put(("log", str(msg)))

    def progress(frac, msg=""):
        msg_q.put(("progress", (max(0.0, min(1.0, frac)), str(msg))))

    def flush_queue():
        try:
            while True:
                kind, payload = msg_q.get_nowait()
                if kind == "exec":
                    # Executa na thread da UI; workers NUNCA tocam em widgets.
                    try:
                        payload()
                    except Exception as e:
                        try:
                            txt.configure(state="normal")
                            txt.insert("end", f"ERRO interno UI: {e}\n")
                            txt.see("end")
                            txt.configure(state="disabled")
                        except Exception:
                            pass
                    continue
                if kind == "log":
                    txt.configure(state="normal")
                    txt.insert("end", payload + "\n")
                    txt.see("end")
                    txt.configure(state="disabled")
                elif kind == "progress":
                    frac, m = payload
                    prog.set(frac)
                    if m:
                        lbl_prog.configure(text=m)
        except queue.Empty:
            pass
        root.after(120, flush_queue)

    _scan_job = {"id": None}

    def _schedule_scan(*_a):
        try:
            if _scan_job["id"] is not None:
                root.after_cancel(_scan_job["id"])
        except Exception:
            pass
        _scan_job["id"] = root.after(250, refresh_scan)

    def refresh_scan(*_a):
        for w in exe_box.winfo_children():
            w.destroy()
        game = Path(var_game.get()).expanduser() if var_game.get().strip() else None
        warns = []
        if game and game.is_dir():
            exes, ue = scan_game_folder(game)
            if exes:
                if not var_exe.get() or var_exe.get() not in [n for n, _ in exes]:
                    var_exe.set(exes[0][0])
                for name, b in exes:
                    ctk.CTkRadioButton(exe_box, text=f"{name}   [{b}-bit]",
                                       variable=var_exe, value=name,
                                       font=("Sans", 12)).pack(anchor="w", pady=1)
            else:
                ctk.CTkLabel(exe_box, text="Nenhum *.exe na pasta.", font=("Sans", 12),
                             text_color="gray").pack(anchor="w")
                warns.append("Nenhum *.exe na pasta. Confira se é a pasta do executável.")
            if ue:
                warns.append("Parece a pasta raiz de um jogo Unreal Engine — "
                             "instale em <projeto>/Binaries/Win64 (*-Shipping.exe).")
            bits = package_bits()
            ebits = sorted({b for _, b in exes if b in ("32", "64")})
            if ebits and str(bits) not in ebits:
                warns.append(f"Pacote {bits}-bit, mas o exe é {'/'.join(ebits)}-bit.")
            if (game / "dlssnr-amd-install.txt").exists():
                warns.append("Instalação anterior detectada; será removida antes.")
        elif var_game.get().strip():
            warns.append("Pasta não encontrada.")
        lbl_warn.configure(text="\n".join(warns))
        lbl_launch.configure(text="Steam launch options:  " + LAUNCH_HINTS.get(var_route.get(), ""))
        pm = HERE / "dlssnr-amd" / MODEL_NAME
        if model_ok(pm):
            lbl_model.configure(
                text=f"Modelo no pacote: OK ({pm.stat().st_size/1e6:.1f} MiB, SHA256 confere).\nDLL não necessária.")
        elif pm.exists():
            lbl_model.configure(text="Modelo no pacote: DANIFICADO — selecione a DLL para extrair de novo.")
        else:
            saved = var_dll.get().strip()
            hint = f"\nCaminho guardado: {saved}" if saved and Path(saved).exists() else ""
            lbl_model.configure(text="Modelo no pacote: ausente — selecione a DLL (extrai 1x e guarda)." + hint)

    def set_busy(busy):
        state = "disabled" if busy else "normal"
        for b in (btn_install, btn_remove, btn_check, btn_update):
            b.configure(state=state)
        if not busy:
            btn_update.configure(state="normal" if getattr(btn_update, "_has_update", False) else "disabled")

    def collect_ini_overrides():
        ov = {}
        for (sec, key), v in pre_vars.items():
            val = v.get().strip()
            if val and val != "auto":
                ov[(sec, key)] = val
        full = getattr(root, "_full_ini_text", None)
        if full:
            for m in re.finditer(r"\[([^\]]+)\](.*?)(?=\n\[|\Z)", full, re.S):
                sec = m.group(1).strip()
                for line in m.group(2).splitlines():
                    e = re.match(r"\s*([A-Za-z0-9_]+)\s*=\s*(.+?)\s*$", line)
                    if e and not line.lstrip().startswith(("#", ";")):
                        ov[(sec, e.group(1))] = e.group(2)
        cfg2 = load_config()
        cfg2["ini"] = {f"{s}.{k}": v for (s, k), v in ov.items()}
        save_config(cfg2)
        return ov

    def do_install():
        game = Path(var_game.get()).expanduser()
        route = var_route.get()
        dll = var_dll.get().strip() or None
        if not var_game.get().strip():
            messagebox.showwarning("Instalador", "Escolha a pasta do jogo.")
            return
        if not dll:
            saved = load_config().get("last_dll", "")
            if saved and Path(saved).exists():
                dll = saved
                var_dll.set(saved)
                log(f"Reutilizando caminho guardado da DLL: {dll}")
        ov = collect_ini_overrides()
        set_busy(True)
        tabs.set("Log")
        progress(0.0, "Iniciando instalação…")

        def worker():
            try:
                launch = inst.run(game, route, dll, ov, log, progress,
                                  remember_dll=var_remember.get())
                msg_q.put(("log", f"\nOK. Use: {launch}"))
            except Exception as e:
                msg_q.put(("log", f"\nERRO: {e}"))
            finally:
                msg_q.put(("progress", (1.0, "Pronto.")))
                msg_q.put(("exec", lambda: (set_busy(False), refresh_scan())))
        threading.Thread(target=worker, daemon=True).start()

    def do_remove():
        game = Path(var_game.get()).expanduser()
        if not var_game.get().strip() or not game.is_dir():
            messagebox.showwarning("Instalador", "Escolha a pasta do jogo.")
            return
        set_busy(True)
        tabs.set("Log")

        def worker():
            try:
                uninstall_game(game, log)
            except Exception as e:
                msg_q.put(("log", f"ERRO: {e}"))
            finally:
                msg_q.put(("exec", lambda: (set_busy(False), refresh_scan())))
        threading.Thread(target=worker, daemon=True).start()

    latest = {}

    def do_check():
        set_busy(True)
        log(f"Verificando {UPSTREAM_API} …")
        lbl_up.configure(text="Upstream: verificando…")

        def worker():
            try:
                data = check_upstream()
                tag = data["tag"]
                msg_q.put(("log", f"Upstream: {tag} ({data['name']})"))
                cur = f"v{CURRENT_VERSION}"
                if tag and tag != cur:
                    msg_q.put(("log", f"ATUALIZAÇÃO disponível: {cur} -> {tag}"))
                    arch = package_arch()
                    cands = [a for a in data["assets"] if arch in a["name"] and a["name"].endswith(".tar.gz")]
                    if cands:
                        asset = cands[0]
                        msg_q.put(("log", f"  pacote: {asset['name']} ({asset['size']/1e6:.1f} MiB)"))

                        def _has():
                            latest.update(data)
                            latest["asset"] = asset
                            btn_update._has_update = True
                            lbl_up.configure(text=f"Upstream: {tag} disponível")
                            set_busy(False)
                        msg_q.put(("exec", _has))
                    else:
                        msg_q.put(("log", f"  nenhum asset *{arch}.tar.gz no release."))

                        def _no():
                            latest.update(data)
                            lbl_up.configure(text=f"Upstream: {tag} (sem pacote {arch})")
                            set_busy(False)
                        msg_q.put(("exec", _no))
                else:
                    msg_q.put(("log", "Já está na versão mais recente."))

                    def _same():
                        latest.update(data)
                        lbl_up.configure(text=f"Upstream: {tag} (atualizado)")
                        set_busy(False)
                    msg_q.put(("exec", _same))
            except Exception as e:
                msg_q.put(("log", f"ERRO ao verificar upstream: {e}"))

                def _err():
                    lbl_up.configure(text="Upstream: erro de rede")
                    set_busy(False)
                msg_q.put(("exec", _err))
        threading.Thread(target=worker, daemon=True).start()

    def do_update():
        asset = latest.get("asset")
        if not asset:
            messagebox.showinfo("Atualizar", "Clique em Verificar atualização primeiro.")
            return
        if not messagebox.askyesno("Atualizar",
                                    f"Baixar e aplicar {asset['name']}?\nO modelo dlssnr.bin será preservado."):
            return
        set_busy(True)
        tabs.set("Log")

        def worker():
            try:
                update_package(asset["url"], log, progress)
                msg_q.put(("log", "Reinicie o instalador para usar o pacote novo."))
            except Exception as e:
                msg_q.put(("log", f"ERRO na atualização: {e}"))
            finally:
                msg_q.put(("exec", lambda: (set_busy(False), refresh_scan())))
        threading.Thread(target=worker, daemon=True).start()

    btn_install.configure(command=do_install)
    btn_remove.configure(command=do_remove)
    btn_check.configure(command=do_check)
    btn_update.configure(command=do_update)
    var_game.trace_add("write", _schedule_scan)
    var_route.trace_add("write", _schedule_scan)
    var_dll.trace_add("write", _schedule_scan)

    refresh_scan()
    log(f"DLSSNR-AMD — pacote v{CURRENT_VERSION}-{package_arch()} ({package_bits()}-bit).")
    log("1) Aba Jogo: pasta + rota.  2) Aba Modelo: DLL (1x).  3) Aba OptiScaler.ini: pré-config.  4) Instalar.")
    flush_queue()
    root.mainloop()


# ==========================================================================
# GUI Qt (PySide6). Preferida quando disponivel; cai para CTk/classica.
# ==========================================================================

QT_DARK_QSS = """
QMainWindow, QWidget { background-color: #232629; color: #EFF0F1; font-size: 13px; }
QFrame#sidebar { background-color: #1B1D20; border: none; }
QFrame#sidebar QLabel { background-color: transparent; }
QLabel#logo { font-size: 24px; font-weight: 800; color: #FFFFFF; }
QLabel#version { color: #9AA0A6; font-size: 12px; }
QLabel#upstream { color: #9AA0A6; font-size: 11px; }
QPushButton { background-color: #31363B; border: 1px solid #3F444A; border-radius: 6px;
              padding: 8px 12px; color: #EFF0F1; }
QPushButton:hover { background-color: #3A4046; border-color: #4D545C; }
QPushButton:pressed { background-color: #2A2E33; }
QPushButton:disabled { color: #7A7E83; background-color: #2A2D31; border-color: #33373C; }
QPushButton#primary { background-color: #2D7D9A; border: none; font-weight: 700; font-size: 14px; }
QPushButton#primary:hover { background-color: #3592B5; }
QPushButton#primary:pressed { background-color: #256B83; }
QLineEdit { background-color: #2F3338; border: 1px solid #3F444A; border-radius: 6px;
            padding: 7px 9px; selection-background-color: #2D7D9A; }
QLineEdit:focus { border: 1px solid #2D7D9A; }
QTabWidget::pane { border: 1px solid #3A3F45; border-radius: 6px; background: #232629; }
QTabBar::tab { background: #2A2E33; color: #B0B4B9; padding: 9px 22px; margin-right: 2px;
               border-top-left-radius: 6px; border-top-right-radius: 6px; }
QTabBar::tab:selected { background: #31363B; color: #FFFFFF; }
QTabBar::tab:hover:!selected { background: #2E3338; color: #FFFFFF; }
QListWidget { background-color: #2A2E33; border: 1px solid #3A3F45; border-radius: 6px; padding: 4px; }
QListWidget::item { padding: 6px 8px; border-radius: 4px; }
QListWidget::item:selected { background-color: #2D7D9A; color: white; }
QPlainTextEdit { background-color: #1E2023; border: 1px solid #3A3F45; border-radius: 6px;
                 font-family: monospace; font-size: 12px; }
QComboBox { background-color: #2F3338; border: 1px solid #3F444A; border-radius: 6px; padding: 6px 9px; }
QComboBox QAbstractItemView { background-color: #2F3338; selection-background-color: #2D7D9A; }
QRadioButton { spacing: 8px; padding: 3px 0; }
QRadioButton::indicator { width: 16px; height: 16px; border-radius: 8px; border: 2px solid #6A7076; }
QRadioButton::indicator:checked { border-color: #2D7D9A; background-color: #2D7D9A; }
QCheckBox { spacing: 8px; }
QCheckBox::indicator { width: 16px; height: 16px; border-radius: 4px; border: 2px solid #6A7076; }
QCheckBox::indicator:checked { background-color: #2D7D9A; border-color: #2D7D9A; }
QProgressBar { background-color: #2A2E33; border: none; border-radius: 6px; height: 14px; text-align: center; }
QProgressBar::chunk { background-color: #2D7D9A; border-radius: 6px; }
QLabel#warn { color: #E74C3C; }
QLabel#launch { color: #5DADE2; font-weight: 600; }
QScrollArea { border: none; }
"""


def run_gui_qt():
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox,
                                   QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit,
                                   QListWidget, QMainWindow, QMessageBox, QPlainTextEdit,
                                   QProgressBar, QPushButton, QRadioButton, QScrollArea,
                                   QSizePolicy, QSpacerItem, QTabWidget, QVBoxLayout, QWidget)

    class Window(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle(f"DLSSNR-AMD  ·  v{CURRENT_VERSION}-{package_arch()}")
            self.resize(1020, 700)
            self.msg_q = queue.Queue()
            self.cfg = load_config()
            self.inst = Installer()
            self.route = self.cfg.get("last_route", "gamescope")
            self.pre_combos = {}
            self._full_ini_text = None
            self.latest = {}
            self._build()
            self.refresh_scan()
            self.log(f"DLSSNR-AMD — pacote v{CURRENT_VERSION}-{package_arch()} ({package_bits()}-bit).")
            self.log("1) Aba Jogo: pasta + rota.  2) Aba Modelo: DLL (1x).  "
                     "3) Aba OptiScaler.ini: pré-config.  4) Instalar.")
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.flush_queue)
            self.timer.start(120)

        # -- construcao -------------------------------------------------
        def _build(self):
            central = QWidget()
            self.setCentralWidget(central)
            lay = QHBoxLayout(central)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(0)

            side = QFrame()
            side.setObjectName("sidebar")
            side.setFixedWidth(240)
            sl = QVBoxLayout(side)
            sl.setContentsMargins(18, 20, 18, 16)
            logo = QLabel("DLSSNR-AMD")
            logo.setObjectName("logo")
            sl.addWidget(logo)
            ver = QLabel(f"v{CURRENT_VERSION}-{package_arch()}  ·  {package_bits()}-bit")
            ver.setObjectName("version")
            sl.addWidget(ver)
            sl.addSpacing(14)

            self.btn_install = QPushButton("Instalar")
            self.btn_install.setObjectName("primary")
            self.btn_install.setMinimumHeight(40)
            self.btn_install.clicked.connect(self.do_install)
            sl.addWidget(self.btn_install)
            self.btn_remove = QPushButton("Desinstalar")
            self.btn_remove.clicked.connect(self.do_remove)
            sl.addWidget(self.btn_remove)
            self.btn_check = QPushButton("Verificar atualização")
            self.btn_check.clicked.connect(self.do_check)
            sl.addWidget(self.btn_check)
            self.btn_update = QPushButton("Atualizar pacote")
            self.btn_update.clicked.connect(self.do_update)
            self.btn_update.setEnabled(False)
            sl.addWidget(self.btn_update)
            sl.addSpacing(10)
            self.lbl_up = QLabel("Upstream: não verificado")
            self.lbl_up.setObjectName("upstream")
            self.lbl_up.setWordWrap(True)
            sl.addWidget(self.lbl_up)
            sl.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))
            lay.addWidget(side)

            main = QWidget()
            ml = QVBoxLayout(main)
            ml.setContentsMargins(18, 14, 18, 12)
            title = QLabel("Instalador")
            title.setStyleSheet("font-size: 20px; font-weight: 800;")
            ml.addWidget(title)

            self.tabs = QTabWidget()
            ml.addWidget(self.tabs, 1)
            self._tab_game()
            self._tab_model()
            self._tab_ini()
            self._tab_log()
            lay.addWidget(main, 1)

            prow = QHBoxLayout()
            self.lbl_prog = QLabel("Pronto.")
            self.lbl_prog.setStyleSheet("color: #9AA0A6;")
            self.prog = QProgressBar()
            self.prog.setRange(0, 100)
            self.prog.setValue(0)
            prow.addWidget(QLabel("Progresso:"))
            prow.addWidget(self.prog, 1)
            prow.addWidget(self.lbl_prog)
            ml.addLayout(prow)

        def _tab_game(self):
            w = QWidget()
            l = QVBoxLayout(w)
            l.addWidget(QLabel("<b>Pasta do jogo</b> (a que contém o .exe)"))
            row = QHBoxLayout()
            self.ed_game = QLineEdit(self.cfg.get("last_game", ""))
            self.ed_game.setPlaceholderText("/path/to/steamapps/common/<jogo>/Binaries/Win64")
            self.ed_game.textChanged.connect(self.refresh_scan)
            row.addWidget(self.ed_game, 1)
            b = QPushButton("Procurar…")
            b.clicked.connect(self._browse_game)
            row.addWidget(b)
            l.addLayout(row)
            l.addWidget(QLabel("<b>Executáveis encontrados</b>"))
            self.exe_list = QListWidget()
            self.exe_list.setMaximumHeight(96)
            l.addWidget(self.exe_list)
            self.lbl_warn = QLabel("")
            self.lbl_warn.setObjectName("warn")
            self.lbl_warn.setWordWrap(True)
            l.addWidget(self.lbl_warn)
            l.addWidget(QLabel("<b>Rota</b>"))
            self.lbl_launch = QLabel("")
            self.lbl_launch.setObjectName("launch")
            self.lbl_launch.setWordWrap(True)
            self.route_group = QButtonGroup(self)
            self.route_btns = {}
            for i, (rid, desc) in enumerate(ROUTES.items()):
                rb = QRadioButton(f"{rid}   —   {desc}")
                rb.toggled.connect(self._route_changed)
                self.route_group.addButton(rb, i)
                self.route_btns[rid] = rb
                l.addWidget(rb)
            if self.route in self.route_btns:
                self.route_btns[self.route].setChecked(True)
            l.addWidget(self.lbl_launch)
            l.addStretch(1)
            self.tabs.addTab(w, "Jogo")

        def _tab_model(self):
            w = QWidget()
            l = QVBoxLayout(w)
            self.lbl_model = QLabel("")
            self.lbl_model.setWordWrap(True)
            self.lbl_model.setStyleSheet("background:#2A2E33; border-radius:6px; padding:12px;")
            l.addWidget(self.lbl_model)
            l.addWidget(QLabel("<b>DLL da NVIDIA</b> (nvngx_dlssnr 310.8.0 — .dll ou .zip, só na 1ª vez)"))
            row = QHBoxLayout()
            self.ed_dll = QLineEdit(self.cfg.get("last_dll", ""))
            self.ed_dll.setPlaceholderText("/caminho/nvngx_dlssnr_310.8.0.zip")
            self.ed_dll.textChanged.connect(self.refresh_scan)
            row.addWidget(self.ed_dll, 1)
            b = QPushButton("Procurar…")
            b.clicked.connect(self._browse_dll)
            row.addWidget(b)
            l.addLayout(row)
            self.ck_remember = QCheckBox("Lembrar caminho da DLL (extrai 1x e reutiliza)")
            self.ck_remember.setChecked(True)
            l.addWidget(self.ck_remember)
            info = QLabel("O modelo extraído fica em dlssnr-amd/dlssnr.bin no pacote\ne é verificado por SHA256 a cada instalação.")
            info.setStyleSheet("color: #9AA0A6;")
            l.addWidget(info)
            l.addStretch(1)
            self.tabs.addTab(w, "Modelo")

        def _tab_ini(self):
            w = QWidget()
            l = QVBoxLayout(w)
            l.addWidget(QLabel("<b>Pré-configuração</b> (aplicada após o patch base na rota optiscaler)"))
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            inner = QWidget()
            form = QVBoxLayout(inner)
            for label, sec, key, _typ, opts in PRECONFIG_FIELDS:
                row = QHBoxLayout()
                lab = QLabel(label)
                lab.setMinimumWidth(340)
                row.addWidget(lab)
                cb = QComboBox()
                cb.addItems(opts)
                dflt = self.cfg.get("ini", {}).get(f"{sec}.{key}", "auto" if "auto" in opts else opts[0])
                if dflt in opts:
                    cb.setCurrentText(dflt)
                cb.setMinimumWidth(120)
                row.addWidget(cb)
                hint = QLabel(f"[{sec}] {key}")
                hint.setStyleSheet("color: #9AA0A6;")
                row.addWidget(hint, 1)
                form.addLayout(row)
                self.pre_combos[(sec, key)] = cb
            form.addStretch(1)
            scroll.setWidget(inner)
            l.addWidget(scroll, 1)
            b = QPushButton("Editar OptiScaler.ini completo…")
            b.clicked.connect(self._full_ini_editor)
            l.addWidget(b)
            self.tabs.addTab(w, "OptiScaler.ini")

        def _tab_log(self):
            w = QWidget()
            l = QVBoxLayout(w)
            self.txt = QPlainTextEdit()
            self.txt.setReadOnly(True)
            l.addWidget(self.txt, 1)
            self.tabs.addTab(w, "Log")

        # -- eventos ----------------------------------------------------
        def _browse_game(self):
            d = QFileDialog.getExistingDirectory(self, "Pasta do jogo")
            if d:
                self.ed_game.setText(d)

        def _browse_dll(self):
            f, _ = QFileDialog.getOpenFileName(self, "nvngx_dlssnr.dll ou zip",
                                               "", "DLL/zip (*.dll *.zip);;Todos (*.*)")
            if f:
                self.ed_dll.setText(f)

        def _route_changed(self):
            for rid, rb in self.route_btns.items():
                if rb.isChecked():
                    self.route = rid
            self.lbl_launch.setText("Steam launch options:  " + LAUNCH_HINTS.get(self.route, ""))

        def log(self, msg):
            self.msg_q.put(("log", str(msg)))

        def progress(self, frac, msg=""):
            self.msg_q.put(("progress", (max(0.0, min(1.0, frac)), str(msg))))

        def flush_queue(self):
            try:
                while True:
                    kind, payload = self.msg_q.get_nowait()
                    if kind == "exec":
                        # Workers NUNCA tocam em widgets Qt; o callable roda aqui,
                        # na thread da GUI (QTimer deste Window).
                        try:
                            payload()
                        except Exception as e:
                            try:
                                self.txt.appendPlainText(f"ERRO interno UI: {e}")
                            except Exception:
                                pass
                        continue
                    if kind == "log":
                        self.txt.appendPlainText(payload)
                    elif kind == "progress":
                        frac, m = payload
                        self.prog.setValue(int(frac * 100))
                        if m:
                            self.lbl_prog.setText(m)
            except queue.Empty:
                pass

        def refresh_scan(self):
            self.exe_list.clear()
            game = Path(self.ed_game.text()).expanduser() if self.ed_game.text().strip() else None
            warns = []
            if game and game.is_dir():
                exes, ue = scan_game_folder(game)
                for name, b in exes:
                    self.exe_list.addItem(f"{name}   [{b}-bit]")
                if self.exe_list.count():
                    self.exe_list.setCurrentRow(0)
                if not exes:
                    warns.append("Nenhum *.exe na pasta. Confira se é a pasta do executável.")
                if ue:
                    warns.append("Parece a pasta raiz de um jogo Unreal Engine — "
                                 "instale em <projeto>/Binaries/Win64 (*-Shipping.exe).")
                bits = package_bits()
                ebits = sorted({b for _, b in exes if b in ("32", "64")})
                if ebits and str(bits) not in ebits:
                    warns.append(f"Pacote {bits}-bit, mas o exe é {'/'.join(ebits)}-bit.")
                if (game / "dlssnr-amd-install.txt").exists():
                    warns.append("Instalação anterior detectada; será removida antes.")
            elif self.ed_game.text().strip():
                warns.append("Pasta não encontrada.")
            self.lbl_warn.setText("\n".join(warns))
            self.lbl_launch.setText("Steam launch options:  " + LAUNCH_HINTS.get(self.route, ""))
            pm = HERE / "dlssnr-amd" / MODEL_NAME
            if model_ok(pm):
                self.lbl_model.setText(
                    f"Modelo no pacote: OK ({pm.stat().st_size/1e6:.1f} MiB, SHA256 confere).\nDLL não necessária.")
            elif pm.exists():
                self.lbl_model.setText("Modelo no pacote: DANIFICADO — selecione a DLL para extrair de novo.")
            else:
                saved = self.ed_dll.text().strip()
                hint = f"\nCaminho guardado: {saved}" if saved and Path(saved).exists() else ""
                self.lbl_model.setText("Modelo no pacote: ausente — selecione a DLL (extrai 1x e guarda)." + hint)

        def set_busy(self, busy):
            for b in (self.btn_install, self.btn_remove, self.btn_check, self.btn_update):
                b.setEnabled(not busy)
            if not busy:
                self.btn_update.setEnabled(bool(getattr(self.btn_update, "_has_update", False)))

        def collect_ini_overrides(self):
            ov = {}
            for (sec, key), cb in self.pre_combos.items():
                val = cb.currentText().strip()
                if val and val != "auto":
                    ov[(sec, key)] = val
            if self._full_ini_text:
                for m in re.finditer(r"\[([^\]]+)\](.*?)(?=\n\[|\Z)", self._full_ini_text, re.S):
                    sec = m.group(1).strip()
                    for line in m.group(2).splitlines():
                        e = re.match(r"\s*([A-Za-z0-9_]+)\s*=\s*(.+?)\s*$", line)
                        if e and not line.lstrip().startswith(("#", ";")):
                            ov[(sec, e.group(1))] = e.group(2)
            cfg2 = load_config()
            cfg2["ini"] = {f"{s}.{k}": v for (s, k), v in ov.items()}
            save_config(cfg2)
            return ov

        def _full_ini_editor(self):
            from PySide6.QtWidgets import QDialog, QDialogButtonBox
            dlg = QDialog(self)
            dlg.setWindowTitle("OptiScaler.ini — editor completo")
            dlg.resize(700, 560)
            l = QVBoxLayout(dlg)
            l.addWidget(QLabel("Os pares [seção] chave=valor serão aplicados após o patch base."))
            ed = QPlainTextEdit()
            base = self._full_ini_text or default_optiscaler_ini()
            ed.setPlainText(base or "; Cole aqui as linhas [Secao] chave=valor.\n")
            l.addWidget(ed, 1)
            btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            btns.accepted.connect(dlg.accept)
            btns.rejected.connect(dlg.reject)
            l.addWidget(btns)
            if dlg.exec():
                self._full_ini_text = ed.toPlainText()

        def _run_worker(self, fn):
            self.set_busy(True)
            self.tabs.setCurrentIndex(3)
            threading.Thread(target=self._wrap(fn), daemon=True).start()

        def _wrap(self, fn):
            def go():
                try:
                    fn()
                except Exception as e:
                    self.msg_q.put(("log", f"ERRO: {e}"))
                finally:
                    self.msg_q.put(("progress", (1.0, "Pronto.")))
                    # QTimer.singleShot() a partir da worker nunca dispara (sem
                    # event-loop nessa thread) -> UI presa em disabled. Vai pela fila.
                    self.msg_q.put(("exec", lambda: (self.set_busy(False), self.refresh_scan())))
            return go

        def do_install(self):
            game = Path(self.ed_game.text()).expanduser()
            if not self.ed_game.text().strip():
                QMessageBox.warning(self, "Instalador", "Escolha a pasta do jogo.")
                return
            dll = self.ed_dll.text().strip() or None
            if not dll:
                saved = load_config().get("last_dll", "")
                if saved and Path(saved).exists():
                    dll = saved
                    self.ed_dll.setText(saved)
                    self.log(f"Reutilizando caminho guardado da DLL: {dll}")
            ov = self.collect_ini_overrides()
            route = self.route
            remember = self.ck_remember.isChecked()
            self.progress(0.0, "Iniciando instalação…")

            def fn():
                launch = self.inst.run(game, route, dll, ov, self.log, self.progress,
                                       remember_dll=remember)
                self.msg_q.put(("log", f"\nOK. Use: {launch}"))
            self._run_worker(fn)

        def do_remove(self):
            game = Path(self.ed_game.text()).expanduser()
            if not self.ed_game.text().strip() or not game.is_dir():
                QMessageBox.warning(self, "Instalador", "Escolha a pasta do jogo.")
                return

            def fn():
                uninstall_game(game, self.log)
            self._run_worker(fn)

        def do_check(self):
            self.set_busy(True)
            self.log(f"Verificando {UPSTREAM_API} …")
            self.lbl_up.setText("Upstream: verificando…")

            def fn():
                try:
                    data = check_upstream()
                    tag = data["tag"]
                    self.msg_q.put(("log", f"Upstream: {tag} ({data['name']})"))
                    cur = f"v{CURRENT_VERSION}"
                    if tag and tag != cur:
                        self.msg_q.put(("log", f"ATUALIZAÇÃO disponível: {cur} -> {tag}"))
                        arch = package_arch()
                        cands = [a for a in data["assets"]
                                 if arch in a["name"] and a["name"].endswith(".tar.gz")]
                        if cands:
                            asset = cands[0]
                            self.msg_q.put(("log",
                                            f"  pacote: {asset['name']} ({asset['size']/1e6:.1f} MiB)"))

                            def _has(d=data, a=asset, t=tag):
                                self.latest.update(d)
                                self.latest["asset"] = a
                                self.btn_update._has_update = True
                                self.lbl_up.setText(f"Upstream: {t} disponível")
                                self.set_busy(False)
                            self.msg_q.put(("exec", _has))
                        else:
                            self.msg_q.put(("log", f"  nenhum asset *{arch}.tar.gz no release."))

                            def _no(d=data, t=tag, ar=arch):
                                self.latest.update(d)
                                self.lbl_up.setText(f"Upstream: {t} (sem pacote {ar})")
                                self.set_busy(False)
                            self.msg_q.put(("exec", _no))
                    else:
                        self.msg_q.put(("log", "Já está na versão mais recente."))

                        def _same(d=data, t=tag):
                            self.latest.update(d)
                            self.lbl_up.setText(f"Upstream: {t} (atualizado)")
                            self.set_busy(False)
                        self.msg_q.put(("exec", _same))
                except Exception as e:
                    self.msg_q.put(("log", f"ERRO ao verificar upstream: {e}"))

                    def _err():
                        self.lbl_up.setText("Upstream: erro de rede")
                        self.set_busy(False)
                    self.msg_q.put(("exec", _err))
            threading.Thread(target=fn, daemon=True).start()

        def do_update(self):
            asset = self.latest.get("asset")
            if not asset:
                QMessageBox.information(self, "Atualizar", "Clique em Verificar atualização primeiro.")
                return
            if QMessageBox.question(self, "Atualizar",
                                    f"Baixar e aplicar {asset['name']}?\nO modelo dlssnr.bin será preservado."
                                    ) != QMessageBox.Yes:
                return

            def fn():
                try:
                    update_package(asset["url"], self.log, self.progress)
                    self.msg_q.put(("log", "Reinicie o instalador para usar o pacote novo."))
                except Exception as e:
                    self.msg_q.put(("log", f"ERRO na atualização: {e}"))
            self._run_worker(fn)

    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(QT_DARK_QSS)
    win = Window()
    win.show()
    app.exec()


def open_full_ini_editor_ctk(root, pre_vars):
    import customtkinter as ctk
    win = ctk.CTkToplevel(root)
    win.title("OptiScaler.ini — editor completo")
    win.geometry("700x560")
    ctk.CTkLabel(win, text="Os pares [seção] chave=valor serão aplicados após o patch base.",
                 font=("Sans", 12), text_color="gray").pack(padx=12, pady=8, anchor="w")
    txt = ctk.CTkTextbox(win, font=("Mono", 12))
    txt.pack(fill="both", expand=True, padx=12, pady=4)
    base = getattr(root, "_full_ini_text", None) or default_optiscaler_ini()
    if not base:
        base = "; OptiScaler.ini padrão não encontrado no pacote.\n; Cole aqui as linhas [Secao] chave=valor.\n"
    txt.insert("1.0", base)

    def save():
        root._full_ini_text = txt.get("1.0", "end")
        win.destroy()
    ctk.CTkButton(win, text="Usar este texto na instalação", command=save).pack(pady=10)


def open_full_ini_editor(root, pre_vars):
    import tkinter as tk
    from tkinter import ttk
    win = tk.Toplevel(root)
    win.title("OptiScaler.ini — editor completo (aplica na instalacao)")
    win.geometry("700x560")
    ttk.Label(win, text="Edite os valores; [secao] chave=valor sera aplicado apos o patch base.",
              wraplength=660).pack(padx=8, pady=6)
    txt = tk.Text(win, wrap="none")
    txt.pack(fill="both", expand=True, padx=8, pady=4)
    base = getattr(root, "_full_ini_text", None) or default_optiscaler_ini()
    if not base:
        base = "; OptiScaler.ini padrao nao encontrado no pacote.\n; Cole aqui as linhas [Secao] chave=valor.\n"
    txt.insert("1.0", base)

    def save():
        root._full_ini_text = txt.get("1.0", "end")
        win.destroy()
    ttk.Button(win, text="Usar este texto na instalacao", command=save).pack(pady=6)


if __name__ == "__main__":
    if "--version" in sys.argv:
        print(f"{CURRENT_VERSION}-{package_arch()}")
        sys.exit(0)
    if "--classic" in sys.argv:
        run_gui()
    elif "--ctk" in sys.argv:
        run_gui_ctk()
    else:
        try:
            import PySide6  # noqa: F401
            run_gui_qt()
        except ImportError:
            try:
                import customtkinter  # noqa: F401
                run_gui_ctk()
            except ImportError:
                print("PySide6/CustomTkinter nao encontrados; usando UI classica "
                      "(rode install-gui.sh para instalar a Qt).")
                run_gui()
