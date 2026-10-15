"""Dependency-free checks for files included in a public checkout."""
import ast
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
paths = subprocess.check_output(
    ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=root
).decode().split('\0')
errors = []
for name in filter(None, paths):
    path = root / name
    if not path.is_file():
        continue
    if path.suffix in {'.ckpt', '.pth', '.pkl', '.pem'} or name.startswith(('logdir/', 'results/')):
        errors.append(f'{name}: generated or sensitive artifact')
    if path.suffix == '.py':
        ast.parse(path.read_text(), filename=name)
    if path.suffix == '.json':
        json.loads(path.read_text())
    if path.suffix == '.ipynb':
        notebook = json.loads(path.read_text())
        for index, cell in enumerate(notebook['cells']):
            if cell.get('outputs') or cell.get('execution_count') is not None:
                errors.append(f'{name} cell {index}: clear execution outputs/counts')
            if cell.get('attachments'):
                errors.append(f'{name} cell {index}: review embedded attachments')
if errors:
    raise SystemExit('\n'.join(errors))
print('Python syntax, JSON, notebook hygiene, and artifact checks passed.')
