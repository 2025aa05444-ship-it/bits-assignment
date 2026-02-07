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

# Find the index where we should insert visualization cells
# We'll insert after the RNN metrics calculation
insert_indices = []
for i, cell in enumerate(cells):
    source_text = "".join(cell.get('source', []))
    
    # After RNN metrics
    if 'rnn_r2 = r2_score' in source_text:
        insert_indices.append(('rnn', i + 1))
    
    # After Transformer metrics
    if 'transformer_r2 = r2_score' in source_text and 'y_test, y_pred_trans' in source_text:
        insert_indices.append(('transformer', i + 1))
    
    # After comparison dataframe
    if 'comparison_df.to_string(index=False)' in source_text:
        insert_indices.append(('comparison', i + 1))

# RNN Visualization
rnn_viz = [
    "# Visualize RNN Results\n",
    "fig, axes = plt.subplots(1, 2, figsize=(15, 5))\n",
    "\n",
    "# Plot 1: Training Loss\n",
    "axes[0].plot(history_rnn.history['loss'], label='Training Loss')\n",
    "axes[0].plot(history_rnn.history['val_loss'], label='Validation Loss')\n",
    "axes[0].set_title('RNN Training Loss')\n",
    "axes[0].set_xlabel('Epoch')\n",
    "axes[0].set_ylabel('Loss')\n",
    "axes[0].legend()\n",
    "axes[0].grid(True)\n",
    "\n",
    "# Plot 2: Predictions vs Actual (showing ALL test data)\n",
    "axes[1].plot(y_test_inv[:, 0], label='Actual', alpha=0.7)\n",
    "axes[1].plot(y_pred_rnn_inv[:, 0], label='Predicted', alpha=0.7)\n",
    "axes[1].set_title('RNN: Actual vs Predicted (Step 1) - All Test Data')\n",
    "axes[1].set_xlabel('Sample')\n",
    "axes[1].set_ylabel('Value')\n",
    "axes[1].legend()\n",
    "axes[1].grid(True)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
]

# Transformer Visualization
transformer_viz = [
    "# Visualize Transformer Results\n",
    "fig, axes = plt.subplots(1, 2, figsize=(15, 5))\n",
    "\n",
    "# Plot 1: Training Loss\n",
    "axes[0].plot(history_trans.history['loss'], label='Training Loss')\n",
    "axes[0].plot(history_trans.history['val_loss'], label='Validation Loss')\n",
    "axes[0].set_title('Transformer Training Loss')\n",
    "axes[0].set_xlabel('Epoch')\n",
    "axes[0].set_ylabel('Loss')\n",
    "axes[0].legend()\n",
    "axes[0].grid(True)\n",
    "\n",
    "# Plot 2: Predictions vs Actual (showing ALL test data)\n",
    "axes[1].plot(y_test_inv[:, 0], label='Actual', alpha=0.7)\n",
    "axes[1].plot(y_pred_trans_inv[:, 0], label='Predicted', alpha=0.7)\n",
    "axes[1].set_title('Transformer: Actual vs Predicted (Step 1) - All Test Data')\n",
    "axes[1].set_xlabel('Sample')\n",
    "axes[1].set_ylabel('Value')\n",
    "axes[1].legend()\n",
    "axes[1].grid(True)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
]

# Comparison Visualization
comparison_viz = [
    "# Comparative Visualization\n",
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
    "axes[0].bar(x - width/2, rnn_values, width, label='RNN', alpha=0.8)\n",
    "axes[0].bar(x + width/2, transformer_values, width, label='Transformer', alpha=0.8)\n",
    "axes[0].set_xlabel('Metrics')\n",
    "axes[0].set_ylabel('Value')\n",
    "axes[0].set_title('Model Performance Comparison')\n",
    "axes[0].set_xticks(x)\n",
    "axes[0].set_xticklabels(metrics)\n",
    "axes[0].legend()\n",
    "axes[0].grid(True, alpha=0.3)\n",
    "\n",
    "# Plot 2: Predictions Overlay (showing ALL test data)\n",
    "axes[1].plot(y_test_inv[:, 0], label='Actual', linewidth=2, alpha=0.7)\n",
    "axes[1].plot(y_pred_rnn_inv[:, 0], label='RNN Predicted', alpha=0.6, linestyle='--')\n",
    "axes[1].plot(y_pred_trans_inv[:, 0], label='Transformer Predicted', alpha=0.6, linestyle='-.')\n",
    "axes[1].set_title('Model Predictions Comparison - All Test Data')\n",
    "axes[1].set_xlabel('Sample')\n",
    "axes[1].set_ylabel('Value')\n",
    "axes[1].legend()\n",
    "axes[1].grid(True, alpha=0.3)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
]

# Insert the visualization cells
offset = 0
for viz_type, idx in sorted(insert_indices, key=lambda x: x[1]):
    if viz_type == 'rnn':
        cells.insert(idx + offset, create_code_cell(rnn_viz))
        offset += 1
    elif viz_type == 'transformer':
        cells.insert(idx + offset, create_code_cell(transformer_viz))
        offset += 1
    elif viz_type == 'comparison':
        cells.insert(idx + offset, create_code_cell(comparison_viz))
        offset += 1

# Save the notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Visualizations with all data added successfully!")
