# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec: a Windows one-folder, windowed build of the desktop app.

Build it with ``scripts/build_exe.ps1`` (it also runs the frozen ``--selftest``).

What is bundled: ``app`` and ``common`` (found by following the imports from the launcher), the
imageio-ffmpeg binary, the keyring Windows backend and the Qt plugins a plain window needs.
``remote/`` is deliberately NOT bundled: the app never imports or uploads it. The rented machine
runs the copy that is baked into the container image (``/app/remote``).
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

ROOT = Path(SPECPATH).parent  # noqa: F821 - SPECPATH is provided by PyInstaller

# imageio_ffmpeg.get_ffmpeg_exe() looks in <package>/binaries/, so ship the package data.
datas = collect_data_files("imageio_ffmpeg")
# keyring finds its backends through package metadata (entry points).
datas += copy_metadata("keyring")

hiddenimports = (
    collect_submodules("keyring.backends")  # includes keyring.backends.Windows
    + ["win32ctypes.core", "win32ctypes.pywin32"]  # used by the Windows backend
    + collect_submodules("paramiko")
    + collect_submodules("cryptography.hazmat.primitives.asymmetric")
)

# Packages the app never imports (checked against app/ and common/), kept out of the analysis.
# This does not make the folder smaller: PySide6's PyInstaller hook still ships the Qt DLLs
# (Qt6Network, Qt6Qml, ...) and the ffmpeg binary alone is about 100 MB of the total.
excludes = [
    "tkinter",
    "matplotlib",
    "numpy",
    "scipy",
    "pandas",
    "torch",
    "IPython",
    "pytest",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebEngineQuick",
    "PySide6.QtPdf",
    "PySide6.QtPdfWidgets",
    "PySide6.QtQml",
    "PySide6.QtQuick",
    "PySide6.QtQuickWidgets",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "PySide6.Qt3DCore",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.QtNetwork",
    "PySide6.QtSql",
    "PySide6.QtSvg",
    "PySide6.QtSvgWidgets",
    "PySide6.QtOpenGL",
    "PySide6.QtOpenGLWidgets",
    "PySide6.QtDesigner",
    "PySide6.QtHelp",
    "PySide6.QtTest",
]

a = Analysis(  # noqa: F821
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="gpu-transcriber",
    debug=False,
    strip=False,
    upx=False,
    console=False,  # windowed: no console window behind the app
)

coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="gpu-transcriber",
)
