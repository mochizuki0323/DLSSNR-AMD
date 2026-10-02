#!/usr/bin/env bash
# Lanca o instalador com UI Qt (PySide6, tema escuro).
# Na primeira vez cria .venv-gui/ e instala a lib (precisa de internet
# uma vez); sem a lib, cai para as UIs de reserva (--ctk/--classic).
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
venv=$here/.venv-gui

have_qt() { "$venv/bin/python" -c "import PySide6" 2>/dev/null; }

if [[ ! -x "$venv/bin/python" ]]; then
    echo "Criando ambiente da UI em $venv ..."
    if ! python3 -m venv "$venv"; then
        echo "AVISO: nao deu para criar o venv; usando UI de reserva." >&2
        exec python3 "$here/installer-gui.py" --classic "$@"
    fi
fi
if ! have_qt; then
    echo "Instalando PySide6 (uma vez, ~100 MiB) ..."
    if ! "$venv/bin/pip" install -q PySide6; then
        echo "AVISO: sem internet/lib; usando UI de reserva." >&2
        exec python3 "$here/installer-gui.py" --classic "$@"
    fi
fi
exec "$venv/bin/python" "$here/installer-gui.py" "$@"
