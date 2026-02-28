# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for Growing Up macOS app.

Usage:
    pyinstaller packaging/macos/GrowingUp.spec

Or use build.sh which handles everything including code signing and DMG.
"""

import os
import glob
from pathlib import Path

block_cipher = None

# Find mediapipe's tasks/c directory (contains libmediapipe.dylib)
_site_packages = Path(os.popen("python -c \"import mediapipe; print(mediapipe.__path__[0])\"").read().strip())
_mediapipe_tasks_c = str(_site_packages / "tasks" / "c")

PROJECT_ROOT = Path(os.path.abspath(SPECPATH)).parent.parent
PACKAGING_DIR = PROJECT_ROOT / "packaging" / "macos"

a = Analysis(
    [str(PACKAGING_DIR / "launcher.py")],
    pathex=[str(PROJECT_ROOT / "webapp")],
    binaries=[
        # mediapipe native library (loaded via importlib.resources)
        (_mediapipe_tasks_c + "/libmediapipe.dylib", "mediapipe/tasks/c"),
    ],
    datas=[
        # Code/ pipeline scripts and ML model files
        (str(PROJECT_ROOT / "Code" / "*.py"), "Code"),
        (str(PROJECT_ROOT / "Code" / "face_landmarker_v2_with_blendshapes.task"), "Code"),
        (str(PROJECT_ROOT / "Code" / "selfie_segmenter.tflite"), "Code"),
        # webapp/ (Flask app, templates, static)
        (str(PROJECT_ROOT / "webapp" / "app.py"), "webapp"),
        (str(PROJECT_ROOT / "webapp" / "pipeline_runner.py"), "webapp"),
        (str(PROJECT_ROOT / "webapp" / "templates"), "webapp/templates"),
        (str(PROJECT_ROOT / "webapp" / "static"), "webapp/static"),
        # mediapipe native bindings package (for importlib.resources to find the dylib)
        (_mediapipe_tasks_c + "/__init__.py", "mediapipe/tasks/c"),
        # Bundled ffmpeg/ffprobe (added by build.sh into packaging/macos/bin/)
        (str(PACKAGING_DIR / "bin" / "ffmpeg"), "bin"),
        (str(PACKAGING_DIR / "bin" / "ffprobe"), "bin"),
    ],
    hiddenimports=[
        # Flask
        "flask",
        "flask.json",
        "jinja2",
        "markupsafe",
        "werkzeug",
        "werkzeug.serving",
        "werkzeug.debug",
        # insightface + ONNX
        "insightface",
        "insightface.app",
        "insightface.app.face_analysis",
        "insightface.model_zoo",
        "insightface.utils",
        "onnxruntime",
        "onnx",
        # mediapipe
        "mediapipe",
        "mediapipe.tasks",
        "mediapipe.tasks.c",
        "mediapipe.tasks.python",
        "mediapipe.tasks.python.vision",
        # scipy (used by insightface)
        "scipy",
        "scipy.spatial",
        "scipy.special",
        # Image/video processing
        "cv2",
        "numpy",
        "PIL",
        "PIL.Image",
        "PIL.ExifTags",
        # Cython stubs for insightface on Apple Silicon
        "cython",
        # Standard library modules sometimes missed
        "json",
        "pathlib",
        "threading",
        "subprocess",
        "datetime",
        "shutil",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "PyQt5",
        "PyQt6",
        "PySide2",
        "PySide6",
        # Exclude the x86_64-only Cython module (stubbed at runtime on Apple Silicon)
        "insightface.thirdparty.face3d.mesh.cython.mesh_core_cython",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Filter out the x86_64-only insightface Cython .so (it's stubbed at runtime)
a.binaries = [b for b in a.binaries if 'mesh_core_cython' not in b[0]]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Growing Up",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="Growing Up",
)

app = BUNDLE(
    coll,
    name="Growing Up.app",
    icon=str(PACKAGING_DIR / "icon.icns"),
    bundle_identifier="com.barnettlabs.growingup",
    info_plist={
        "CFBundleDisplayName": "Growing Up",
        "CFBundleName": "Growing Up",
        "CFBundleShortVersionString": "1.0.0",
        "CFBundleVersion": "1",
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "12.0",
        "NSAppleEventsUsageDescription": "Growing Up needs to open your browser.",
    },
)
