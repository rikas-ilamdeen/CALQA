# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['src\\main.py'],
    pathex=['src'],
    binaries=[],
    datas=[],
    hiddenimports=['PyQt6.QtCore', 'PyQt6.QtGui', 'PyQt6.QtWidgets', 'PyQt6.sip', 'numba.cuda', 'numba.cuda.cudadrv.driver', 'numba.cuda.cudadrv.devices', 'numba.cuda.cudadrv.runtime', 'numba.cuda.cudadrv.libs'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['cupy', 'cupy_backends', 'cv2', 'matplotlib', 'tkinter', 'numba.np.ufunc.tbbpool', 'numba.tests', 'numba.cuda.tests'],
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
    name='CALQA',
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
