import json

nb_path = r'CNN_assignment_template (1).ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# List of code snippets that indicate a cell should be of type 'code'
# These match the starts of the replacements we made in the previous step
code_markers = [
    "# Download Cats and Dogs dataset",
    "plt.figure(figsize=(10, 10))",
    "acc = history_custom.history['accuracy']",
    "acc = history_tl.history['accuracy']",
    "comparison_df.set_index('Metric')"
]

fixed_count = 0

for cell in cells:
    if 'source' in cell and len(cell['source']) > 0:
        first_line = cell['source'][0]
        
        # Check if this cell starts with one of our code markers
        is_code_content = False
        for marker in code_markers:
            if marker in first_line:
                is_code_content = True
                break
        
        # If it looks like code but is marked as markdown, fix it
        if is_code_content and cell['cell_type'] == 'markdown':
            print(f"Fixing cell starting with: {first_line.strip()}...")
            cell['cell_type'] = 'code'
            cell['execution_count'] = None
            cell['outputs'] = []
            fixed_count += 1

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"Fixed {fixed_count} cells.")
