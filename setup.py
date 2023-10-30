from cx_Freeze import setup, Executable
import sys
sys.setrecursionlimit(5000)
build_options = {
    'packages': ['flask', 'transformers', 'tqdm', 'torch'],
    'excludes': []
}

base = None

executables = [
    Executable('main.py', base=base, target_name='app_name')
]

setup(
    name='Model Inference API',
    version='0.1',
    description='API for model inference using BERT models',
    options={'build_exe': build_options},
    executables=executables
)
