import json

notebook_path = r"c:\Users\kulkarp7\OneDrive - TRADINGANDCOMMAND(BT)\Documents\Prasad\BITS\GITRepository\bits-assignment\dnn\RNN_Assignment\2025AA05444_rnn_assignment.ipynb"

def create_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source
    }

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# First, remove any existing visualization cells
cells_to_remove = []
for i, cell in enumerate(cells):
    source_text = "".join(cell.get('source', []))
    
    # Remove cells that contain visualization code
    if any(keyword in source_text for keyword in [
        'fig, axes = plt.subplots',
        'Visualize RNN Results',
        'Visualize Transformer Results',
        'Comparative Visualization',
        'RNN Training Loss',
        'Transformer Training Loss',
        'Model Performance Comparison'
    ]):
        cells_to_remove.append(i)

# Remove cells in reverse order to maintain indices
for i in reversed(cells_to_remove):
    print(f"Removing visualization cell at index {i}")
    cells.pop(i)

# Now find the correct insertion points
insert_indices = []
for i, cell in enumerate(cells):
    source_text = "".join(cell.get('source', []))
    
    # After RNN metrics print
    if 'print(f"R² Score: {rnn_r2:.4f}")' in source_text:
        insert_indices.append(('rnn', i + 1))
        print(f"Found RNN metrics at index {i}")
    
    # After Transformer metrics print
    if 'print(f"R² Score: {transformer_r2:.4f}")' in source_text:
        insert_indices.append(('transformer', i + 1))
        print(f"Found Transformer metrics at index {i}")
    
    # After comparison dataframe print
    if 'print(comparison_df.to_string(index=False))' in source_text:
        insert_indices.append(('comparison', i + 1))
        print(f"Found comparison at index {i}")

# RNN Visualization - SHOWING ALL DATA
rnn_viz = [
    "# ============================================================\n",
    "# RNN VISUALIZATION - ALL TEST DATA\n",
    "# ============================================================\n",
    "fig, axes = plt.subplots(1, 2, figsize=(15, 5))\n",
    "\n",
    "# Plot 1: Training Loss\n",
    "axes[0].plot(history_rnn.history['loss'], label='Training Loss', linewidth=2)\n",
    "axes[0].plot(history_rnn.history['val_loss'], label='Validation Loss', linewidth=2)\n",
    "axes[0].set_title('RNN Training Loss', fontsize=14, fontweight='bold')\n",
    "axes[0].set_xlabel('Epoch')\n",
    "axes[0].set_ylabel('Loss')\n",
    "axes[0].legend()\n",
    "axes[0].grid(True, alpha=0.3)\n",
    "\n",
    "# Plot 2: ALL Test Data Predictions\n",
    "axes[1].plot(y_test_inv[:, 0], label='Actual', alpha=0.7, linewidth=2)\n",
    "axes[1].plot(y_pred_rnn_inv[:, 0], label='Predicted', alpha=0.7, linewidth=2)\n",
    "axes[1].set_title(f'RNN: Actual vs Predicted - ALL {len(y_test_inv)} Test Samples', \n",
    "                  fontsize=14, fontweight='bold')\n",
    "axes[1].set_xlabel('Sample Index')\n",
    "axes[1].set_ylabel('Stock Price')\n",
    "axes[1].legend()\n",
    "axes[1].grid(True, alpha=0.3)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "print(f\"\\nDisplaying predictions for ALL {len(y_test_inv)} test samples\")\n"
]

# Transformer Visualization - SHOWING ALL DATA
transformer_viz = [
    "# ============================================================\n",
    "# TRANSFORMER VISUALIZATION - ALL TEST DATA\n",
    "# ============================================================\n",
    "fig, axes = plt.subplots(1, 2, figsize=(15, 5))\n",
    "\n",
    "# Plot 1: Training Loss\n",
    "axes[0].plot(history_trans.history['loss'], label='Training Loss', linewidth=2)\n",
    "axes[0].plot(history_trans.history['val_loss'], label='Validation Loss', linewidth=2)\n",
    "axes[0].set_title('Transformer Training Loss', fontsize=14, fontweight='bold')\n",
    "axes[0].set_xlabel('Epoch')\n",
    "axes[0].set_ylabel('Loss')\n",
    "axes[0].legend()\n",
    "axes[0].grid(True, alpha=0.3)\n",
    "\n",
    "# Plot 2: ALL Test Data Predictions\n",
    "axes[1].plot(y_test_inv[:, 0], label='Actual', alpha=0.7, linewidth=2)\n",
    "axes[1].plot(y_pred_trans_inv[:, 0], label='Predicted', alpha=0.7, linewidth=2)\n",
    "axes[1].set_title(f'Transformer: Actual vs Predicted - ALL {len(y_test_inv)} Test Samples', \n",
    "                  fontsize=14, fontweight='bold')\n",
    "axes[1].set_xlabel('Sample Index')\n",
    "axes[1].set_ylabel('Stock Price')\n",
    "axes[1].legend()\n",
    "axes[1].grid(True, alpha=0.3)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "print(f\"\\nDisplaying predictions for ALL {len(y_test_inv)} test samples\")\n"
]

# Comparison Visualization - SHOWING ALL DATA
comparison_viz = [
    "# ============================================================\n",
    "# COMPARATIVE VISUALIZATION - ALL TEST DATA\n",
    "# ============================================================\n",
    "fig, axes = plt.subplots(1, 2, figsize=(15, 5))\n",
    "\n",
    "# Plot 1: Metrics Comparison\n",
    "metrics = ['MAE', 'RMSE', 'R²']\n",
    "rnn_values = [rnn_mae, rnn_rmse, rnn_r2]\n",
    "transformer_values = [transformer_mae, transformer_rmse, transformer_r2]\n",
    "\n",
    "x = np.arange(len(metrics))\n",
    "width = 0.35\n",
    "\n",
    "bars1 = axes[0].bar(x - width/2, rnn_values, width, label='RNN', alpha=0.8, color='steelblue')\n",
    "bars2 = axes[0].bar(x + width/2, transformer_values, width, label='Transformer', alpha=0.8, color='coral')\n",
    "axes[0].set_xlabel('Metrics', fontsize=12)\n",
    "axes[0].set_ylabel('Value', fontsize=12)\n",
    "axes[0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')\n",
    "axes[0].set_xticks(x)\n",
    "axes[0].set_xticklabels(metrics)\n",
    "axes[0].legend()\n",
    "axes[0].grid(True, alpha=0.3, axis='y')\n",
    "\n",
    "# Add value labels on bars\n",
    "for bars in [bars1, bars2]:\n",
    "    for bar in bars:\n",
    "        height = bar.get_height()\n",
    "        axes[0].text(bar.get_x() + bar.get_width()/2., height,\n",
    "                    f'{height:.4f}', ha='center', va='bottom', fontsize=9)\n",
    "\n",
    "# Plot 2: ALL Test Data Predictions Overlay\n",
    "axes[1].plot(y_test_inv[:, 0], label='Actual', linewidth=2.5, alpha=0.8, color='black')\n",
    "axes[1].plot(y_pred_rnn_inv[:, 0], label='RNN Predicted', alpha=0.7, linestyle='--', \n",
    "            linewidth=2, color='steelblue')\n",
    "axes[1].plot(y_pred_trans_inv[:, 0], label='Transformer Predicted', alpha=0.7, \n",
    "            linestyle='-.', linewidth=2, color='coral')\n",
    "axes[1].set_title(f'Model Predictions Comparison - ALL {len(y_test_inv)} Test Samples', \n",
    "                  fontsize=14, fontweight='bold')\n",
    "axes[1].set_xlabel('Sample Index', fontsize=12)\n",
    "axes[1].set_ylabel('Stock Price', fontsize=12)\n",
    "axes[1].legend(loc='best')\n",
    "axes[1].grid(True, alpha=0.3)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "print(f\"\\n\" + \"=\"*70)\n",
    "print(f\"COMPLETE DATASET VISUALIZATION\")\n",
    "print(f\"Total test samples displayed: {len(y_test_inv)}\")\n",
    "print(f\"=\"*70)\n"
]

# Insert the visualization cells
offset = 0
for viz_type, idx in sorted(insert_indices, key=lambda x: x[1]):
    actual_idx = idx + offset
    if viz_type == 'rnn':
        cells.insert(actual_idx, create_code_cell(rnn_viz))
        print(f"Inserted RNN visualization at index {actual_idx}")
        offset += 1
    elif viz_type == 'transformer':
        cells.insert(actual_idx, create_code_cell(transformer_viz))
        print(f"Inserted Transformer visualization at index {actual_idx}")
        offset += 1
    elif viz_type == 'comparison':
        cells.insert(actual_idx, create_code_cell(comparison_viz))
        print(f"Inserted Comparison visualization at index {actual_idx}")
        offset += 1

# Save the notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("\n" + "="*70)
print("SUCCESS! All visualizations updated to show COMPLETE dataset")
print("="*70)
print("\nKey changes:")
print("- Removed all old visualization cells")
print("- Added new cells showing ALL test data (not limited to 100)")
print("- Added sample count in titles")
print("- Enhanced styling and labels")
print("\nPlease restart your notebook kernel and run all cells to see the changes!")
