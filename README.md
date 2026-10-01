# LaRS label editor with SAM 3 autolabeling

A browser-based editor for panoptic masks in the [LaRS](https://lojzezust.github.io/lars-dataset/) dataset format, with a SAM 3 tool: draw a box around one object and every similar object in the image is added as a new instance of the chosen class.

It runs as two local servers:

- `server.py` is the editor. It serves the web page and reads and writes the dataset files. It needs only `numpy` and `pillow`.
- `sam_server.py` runs SAM 3 on the GPU. The editor forwards SAM requests to it, so the browser only ever talks to the editor.

The editor works without the SAM server; only the SAM tool needs it.

## Requirements

- An NVIDIA GPU with about 5 GB of free VRAM and a recent driver (one that supports CUDA 12.8).
- Python 3.12 (3.10 is known to work too).
- Git.
- The SAM 3 checkpoint `sam3.pt`. It is gated on Hugging Face: request access at <https://huggingface.co/facebook/sam3>, then download `sam3.pt` from the "Files" tab. Not `sam3.1_multiplex.pt`, which is the video model.

## Folder layout

After setup, the folder should look like this. The dataset, the checkpoint, the SAM 3 clone and the virtual environment are not part of this repository (see `.gitignore`).

```
lars-editor/
	server.py
	sam_server.py
	index.html
	sam3.pt            (download manually)
	LaRS_v1.1.0/       (copy manually)
		train/images/...
		val/images/...
	sam3repo/          (cloned during setup)
	.venv/             (created during setup)
```

The SAM 3 clone must not be called `sam3`. A folder with that name next to `sam_server.py` hides the installed `sam3` package from Python and causes a confusing `TypeError ... not NoneType` at startup.

## Installation

All commands are run from inside the `lars-editor` folder.

### Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate

pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128

git clone https://github.com/facebookresearch/sam3.git sam3repo
git -C sam3repo checkout 2345a4ad109ac29c569da749c91d84f10dc08c40
pip install -e sam3repo
pip install einops pycocotools psutil
```

### Windows

In PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1

pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "triton-windows<3.7"

git clone https://github.com/facebookresearch/sam3.git sam3repo
git -C sam3repo checkout 2345a4ad109ac29c569da749c91d84f10dc08c40
pip install -e sam3repo
pip install einops pycocotools psutil
```

If PowerShell refuses to run `Activate.ps1`, allow local scripts once with `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or use Command Prompt and run `.venv\Scripts\activate.bat` instead.

`triton-windows` is needed because SAM 3 imports `triton`, which the Linux build of PyTorch includes but the Windows build does not. Its version must match PyTorch: `<3.7` belongs to PyTorch 2.10.

### Notes on the dependencies

- The `git checkout` pins SAM 3 to a known-working version. Skip it to use the latest code, at the risk of breaking changes.
- `einops`, `pycocotools` and `psutil` are imported by SAM 3 but missing from its required dependencies, so they are installed separately.
- The editor itself only needs `numpy` and `pillow`, which the steps above already install. Both servers use the same virtual environment.

### Check the install

```bash
python -c "import torch, sam3; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
```

This should print `True` and your GPU name.

## Running

Open two terminals in the `lars-editor` folder and activate the virtual environment in both (`source .venv/bin/activate` on Linux, `.venv\Scripts\Activate.ps1` on Windows).

Terminal 1, the SAM server (takes a few seconds to load the model):

```bash
python sam_server.py
```

Terminal 2, the editor:

```bash
python server.py
```

Then open <http://localhost:8000> in a browser.

### Defaults and options

Both scripts run without arguments; these options override the defaults.

| Script | Option | Default |
| --- | --- | --- |
| `server.py` | dataset folder (positional) | `LaRS_v1.1.0` |
| `server.py` | `--port` | `8000` |
| `server.py` | `--sam` | `http://127.0.0.1:8001` |
| `sam_server.py` | `--checkpoint` | `sam3.pt` |
| `sam_server.py` | `--port` | `8001` |

Example: `python server.py /data/LaRS_v1.1.0 --port 8080`.

The dataset folder must contain one or more split folders (e.g. `train`, `val`), each with an `images` subfolder. Masks are written next to them in `panoptic_masks`, `semantic_masks` and `panoptic_annotations.json`.

## Using the SAM tool

1. Choose the class under "New object" in the right panel.
2. Press `X` or click "SAM".
3. Drag a box around one object.

Every object SAM considers similar is shown as a pending mask with a white border; the labels are not changed yet. Each mask has a ✓ / ✕ button pair at its top-right corner:

- ✓ adds that mask as a new instance of the chosen class, painted over whatever is under it. Each apply is one undo step (`Ctrl+Z`).
- ✕ discards that mask.
- Hovering a button pair turns its mask's border yellow, to show which mask it belongs to.

The panel shows how many masks are pending, with buttons for all of them at once:

| Button | Key | Action |
| --- | --- | --- |
| Apply | `Enter` | Apply all pending masks (one undo step) |
| Discard | `Esc` | Discard all pending masks |
| Hide / Show | `H` | Hide the pending masks to see the labels below |

A new SAM box replaces any masks still pending, and switching images discards them. Saving keeps them.

The "SAM conf." slider sets the minimum detection confidence for the next box: lower finds more objects, including false ones; higher finds fewer.

The first box on an image takes about 0.4 s while SAM processes the image; further boxes on the same image take under 0.1 s. Each request prints its time and peak VRAM use in the SAM server's terminal.

## Troubleshooting

- **`TypeError: expected str, bytes or os.PathLike object, not NoneType` when starting `sam_server.py`:** the SAM 3 clone is in a folder named `sam3` next to `sam_server.py`. Rename it and reinstall: `mv sam3 sam3repo` (Windows: `ren sam3 sam3repo`), then `pip install -e sam3repo`.
- **`ModuleNotFoundError` when starting `sam_server.py`:** install the missing package with `pip install <name>`. On Windows, a missing `triton` means `triton-windows` is not installed.
- **"cannot reach SAM server" in the editor:** `sam_server.py` is not running, or is still loading the model.
- **`OutOfMemoryError`:** close other programs using the GPU. The SAM server needs about 5 GB of VRAM at peak.
- **`FutureWarning: Importing from timm.models.layers is deprecated`:** harmless, it comes from SAM 3's own code.
