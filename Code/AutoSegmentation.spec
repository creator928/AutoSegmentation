# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

# Conda의 ctypes 의존 DLL을 포함하여 배포 EXE에서도 Windows 환경 점검이 동작하게 합니다.
ffi_path = Path(sys.prefix) / 'Library' / 'bin' / 'ffi.dll'
runtime_binaries = [(str(ffi_path), '.')] if ffi_path.exists() else []

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=runtime_binaries,
    datas=[('training_runner.py', '.'), ('auto_label_runner.py', '.'), ('validation_runner.py', '.')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['qt_material'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='AutoSegmentation',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
