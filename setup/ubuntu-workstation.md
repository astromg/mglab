# Stanowisko Ubuntu (26.04) z dostępem do OCM — instalacja krok po kroku

Lokalne konto na Ubuntu, z którego chcę mieć: GitHub przez SSH, dyski OCM (`/data/fits`, `/data/misc`), NATS,
pełne TOI, JupyterLab, PyCharm i Claude Code z dostępem do vault. Źródła: oca-manual `docs/network.md`
(NFS, środowisko Pythona) i notatka vault „New Linux Machine Setup”.

Układ katalogów jak na Macu: wszystkie repo w `~/work/src/` (TOI wymaga, żeby `tpg` leżał obok niego).

## 0. Najpierw to repo

Klucz SSH wygenerowany na tej maszynie i dodany w GitHub → Settings → SSH and GPG keys.

```bash
ssh -T git@github.com                       # "Hi astromg! You've successfully authenticated..."
mkdir -p ~/work/src && cd ~/work/src
git clone git@github.com:astromg/mglab.git
```

Własny klucz na każdą maszynę zamiast kopii z pracy: przy utracie laptopa unieważniam tylko jego klucz.
Warto za to skopiować `~/.ssh/config` z komputera w pracy (aliasy hostów OCM: `nats.oca`, `tic.oca`…).

## 1. Git

```bash
git config --global user.name "Marek Górski"
git config --global user.email "mgorski@camk.edu.pl"
# prywatne zależności TOI (ocabox, ocabox-common, ctc, pyaraucaria) mają adresy https:// - idą przez SSH, bez tokena
git config --global url."git@github.com:".insteadOf "https://github.com/"
git ls-remote git@github.com:araucaria-project/ocabox.git | head -1     # test dostępu do prywatnego repo
```

## 2. Sieć OCM

```bash
ping -c1 192.168.7.38                         # NATS
getent hosts perun.oca.lan                    # DNS OCM, serwer plików
curl -sI http://docker.oca.lan:8081 | head -1 # manual OCA (zakładka Telescope w TOI)
```

Brak odpowiedzi: komputer nie jest w LAN OCM. Wtedy najpierw Tailscale (router `ts-oca`, 192.168.7.0/24).

## 3. Pakiety systemowe

```bash
sudo apt update
sudo apt install -y git curl build-essential pipx nfs-common autofs rsync \
    libxcb-cursor0 libxkbcommon-x11-0 libegl1
pipx ensurepath          # potem wylogować się i zalogować
```

`libxcb-*` i `libegl1` są potrzebne Qt (TOI pod Waylandem sam przełącza się na xcb).

## 4. Dyski `/data/fits` i `/data/misc` (autofs, jak na maszynach OCM)

```bash
sudo mkdir -p /opt/autofs
sudo rsync -av perun.oca.lan::autofs/ /opt/autofs/
[ -f /etc/autofs.conf ] && [ ! -L /etc/autofs.conf ] && sudo mv /etc/autofs.conf /etc/autofs.conf.bak
sudo ln -sf /opt/autofs/autofs.conf /etc/autofs.conf
sudo systemctl restart autofs
ls /data/misc     # samo `ls /data` jest puste - autofs montuje dopiero przy wejściu w podkatalog
```

Później, gdy konfiguracja na perun się zmieni: `sudo /opt/autofs/resync-from-perun.sh`.
Uprawnienia zależą od UID: na maszynach OCM `poweruser` = 1000, `observer` = 1001. Konto lokalne z innym UID
zwykle tylko czyta.

## 5. Narzędzia Pythona

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
pipx install poetry
poetry config virtualenvs.in-project true      # .venv w katalogu projektu (PyCharm, Jupyter)
poetry config keyring.enabled false            # bez tego poetry potrafi zawisnąć bez keyringu
poetry config system-git-client true           # poetry używa gita z pkt 1, czyli SSH
```

## 6. TOI

```bash
cd ~/work/src
git clone git@github.com:astromg/TOI.git
git clone git@github.com:araucaria-project/tpg.git     # zależność TOI ze ścieżki ../tpg
cd TOI
poetry env use python3.14      # TOI i tpg dopuszczają 3.11-3.14
poetry install
poetry run toi                 # zawsze z katalogu TOI: czyta toi_config.yaml i observers.yaml
```

Na nowym ekranie TOI buduje domyślny układ paneli; menu ⚙ → Reset layout, gdy coś się rozjedzie.

## 7. JupyterLab

```bash
uv tool install jupyterlab
# kernel ze środowiskiem TOI (ocabox, pyaraucaria...), bez zmiany pyproject.toml:
cd ~/work/src/TOI && .venv/bin/python -m pip install ipykernel
.venv/bin/python -m ipykernel install --user --name toi --display-name "TOI (poetry)"
# notebooki z mglab mają własne środowisko:
cd ~/work/src/mglab && uv sync && uv run jupyter lab
```

## 8. PyCharm

```bash
sudo snap install pycharm --classic
```

Od 2025 Community i Professional to jeden PyCharm; jeśli nazwa snapa się nie zgadza — `snap find pycharm`
albo JetBrains Toolbox. Interpreter projektu: Add Interpreter → Existing → `~/work/src/TOI/.venv/bin/python`.

## 9. Claude Code i vault

```bash
curl -fsSL https://claude.ai/install.sh | bash
git clone git@github.com:araucaria-project/knowledge-base.git ~/work/src/knowledge-base
mkdir -p ~/.claude
sed "s#__HOME__#$HOME#g" ~/work/src/mglab/setup/claude-global-CLAUDE.md > ~/.claude/CLAUDE.md
```

`setup/claude-global-CLAUDE.md` to kopia mojego globalnego `~/.claude/CLAUDE.md` z Maca (instrukcje o vault),
z `__HOME__` zamiast ścieżki. Alternatywa: `~/work/src/knowledge-base/sync.sh onboard` wypisuje ten blok
z właściwą ścieżką. Pamięć projektów Claude (np. przypomnienia) jest lokalna i z Maca się nie przenosi.
Pierwsze uruchomienie: `cd ~/work/src/TOI && claude`, logowanie w przeglądarce.

## Typowe problemy

- `Permission denied (publickey)` przy `poetry install` → brak pkt 1 (`insteadOf`) albo `system-git-client`.
- `qt.qpa.plugin: Could not load the Qt platform plugin "xcb"` → brak pakietów z pkt 3 (`libxcb-cursor0`).
- TOI kończy się od razu z „Can't connect to NATS” → pkt 2 (sieć / Tailscale).
