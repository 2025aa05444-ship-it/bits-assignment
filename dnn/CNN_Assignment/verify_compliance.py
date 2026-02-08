import json
import re

nb_path = r'CNN_assignment_template (1).ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

source_code = ""
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_code += "".join(cell['source']) + "\n"

print("--- STATIC COMPLIANCE CHECK REPORT ---\n")

# 1. Filename / BITS ID Check (Manual)
print("[MANUAL CHECK] Filename must be <BITS_ID>_cnn_assignment.ipynb")
print("[MANUAL CHECK] BITS ID in top cell must match filename")

# 2. Custom CNN Checks
print("\n[CHECK] Custom CNN Architecture:")
has_conv = "Conv2D" in source_code or "nn.Conv2d" in source_code
has_gap = "GlobalAveragePooling2D" in source_code or "AdaptiveAvgPool" in source_code
has_flatten_dense = ("Flatten" in source_code or "flatten" in source_code) and ("Dense" in source_code or "Linear" in source_code)
# Note: Flatten might be used in logic but NOT in the model path if GAP is present.
# The algo says "IF (Flatten.. AND Dense..) AND NOT has_gap THEN uses_flatten_dense = TRUE" meaning if GAP is present, Flatten+Dense is allowed (maybe for something else?) 
# But let's check if Flatten is used likely in the model.
# Actually, the algo says: IF has_conv AND has_gap AND NOT uses_flatten_dense THEN score += 2.
# So if GAP is present, we are good, UNLESS Flatten+Dense is ALSO present? 
# "IF (Flatten...) AND ... AND NOT has_gap" -> This implies if has_gap is TRUE, then uses_flatten_dense is FALSE.
# So as long as GAP is present, we are fine.

if has_conv:
    print("  [PASS] Conv2D layers found")
else:
    print("  [FAIL] Conv2D layers NOT found")

if has_gap:
    print("  [PASS] Global Average Pooling found")
else:
    print("  [FAIL] Global Average Pooling NOT found (CRITICAL)")

# 3. Transfer Learning Checks
print("\n[CHECK] Transfer Learning:")
has_resnet_vgg = any(x in source_code for x in ['ResNet', 'VGG', 'resnet', 'vgg'])
has_frozen = "trainable = False" in source_code or "requires_grad = False" in source_code

if has_resnet_vgg:
    print("  [PASS] Pre-trained model (ResNet/VGG) found")
else:
    print("  [FAIL] Pre-trained model NOT found")

if has_frozen:
    print("  [PASS] Layer freezing found")
else:
    print("  [FAIL] Layer freezing NOT found")

# 4. JSON Structure
print("\n[CHECK] JSON Output:")
has_json_func = "get_assignment_results" in source_code
has_json_keys = all(x in source_code for x in ['custom_cnn', 'transfer_learning', 'analysis', 'dataset_info'])

if has_json_func and has_json_keys:
    print("  [PASS] JSON output generation code found")
else:
    print("  [FAIL] JSON output code missing or incomplete")

# 5. Analysis
print("\n[CHECK] Analysis:")
analysis_keywords = ['accuracy', 'precision', 'recall', 'f1', 'transfer', 'gap', 'convergence']
found_keywords = [k for k in analysis_keywords if k in source_code.lower()]

if len(found_keywords) >= 5:
    print(f"  [PASS] Analysis keywords found: {len(found_keywords)}/7")
else:
    print(f"  [WARN] Analysis keywords scarce: {len(found_keywords)}/7 found")

print("\n--- END REPORT ---")
