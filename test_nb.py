import json
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for testing

with open('student_performance_tracking_system.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells in notebook: {len(nb['cells'])}")
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
print(f"Total code cells: {len(code_cells)}")

full_code = '\n\n# --- NEXT CELL ---\n'.join([''.join(c['source']) for c in code_cells])
exec_globals = {}
exec(full_code, exec_globals)
print("\n>>> SUCCESS: ALL NOTEBOOK CELLS EXECUTED WITH ZERO ERRORS! <<<")
