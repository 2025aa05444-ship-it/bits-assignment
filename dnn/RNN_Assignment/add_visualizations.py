
import json

notebook_path = r"c:/Users/kulkarp7/OneDrive - TRADINGANDCOMMAND(BT)/Documents/Prasad/BITS/GITRepository/bits-assignment/dnn/RNN_Assignment/2025AA05444_rnn_assignment.ipynb"

def create_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source
    }

def add_visualizations():
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    new_cells = []
    
    # Define visualization codes
    
    vis_rnn_code = [
        "# 2.4 Visualize RNN Results\n",
        "plt.figure(figsize=(15, 5))\n",
        "\n",
        "# Plot Training Loss\n",
        "plt.subplot(1, 2, 1)\n",
        "plt.plot(history_rnn.history['loss'], label='Train Loss')\n",
        "plt.plot(history_rnn.history['val_loss'], label='Validation Loss')\n",
        "plt.title('RNN Model Training Loss')\n",
        "plt.xlabel('Epochs')\n",
        "plt.ylabel('Loss (MSE)')\n",
        "plt.legend()\n",
        "plt.grid(True)\n",
        "\n",
        "# Plot Predictions (First Step)\n",
        "plt.subplot(1, 2, 2)\n",
        "# Create a sample of the test set for clearer visualization\n",
        "limit = 100\n",
        "plt.plot(y_test[:limit, 0], label='Actual (Step 1)', marker='.')\n",
        "plt.plot(y_pred_rnn[:limit, 0], label='Predicted (Step 1)', marker='x')\n",
        "plt.title('RNN Predictions vs Actual (First 100 Samples)')\n",
        "plt.xlabel('Time Step')\n",
        "plt.ylabel('Normalized Price')\n",
        "plt.legend()\n",
        "plt.grid(True)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "# Residuals Histogram\n",
        "residuals_rnn = y_test - y_pred_rnn\n",
        "plt.figure(figsize=(10, 4))\n",
        "sns.histplot(residuals_rnn.flatten(), kde=True, bins=50)\n",
        "plt.title('RNN Prediction Residuals Distribution')\n",
        "plt.xlabel('Error')\n",
        "plt.ylabel('Frequency')\n",
        "plt.grid(True)\n",
        "plt.show()\n"
    ]

    vis_trans_code = [
        "# 3.6 Visualize Transformer Results\n",
        "plt.figure(figsize=(15, 5))\n",
        "\n",
        "# Plot Training Loss\n",
        "plt.subplot(1, 2, 1)\n",
        "plt.plot(history_transformer.history['loss'], label='Train Loss')\n",
        "plt.plot(history_transformer.history['val_loss'], label='Validation Loss')\n",
        "plt.title('Transformer Model Training Loss')\n",
        "plt.xlabel('Epochs')\n",
        "plt.ylabel('Loss (MSE)')\n",
        "plt.legend()\n",
        "plt.grid(True)\n",
        "\n",
        "# Plot Predictions\n",
        "plt.subplot(1, 2, 2)\n",
        "limit = 100\n",
        "plt.plot(y_test[:limit, 0], label='Actual (Step 1)', marker='.')\n",
        "plt.plot(y_pred_trans[:limit, 0], label='Predicted (Step 1)', marker='x')\n",
        "plt.title('Transformer Predictions vs Actual (First 100 Samples)')\n",
        "plt.xlabel('Time Step')\n",
        "plt.ylabel('Normalized Price')\n",
        "plt.legend()\n",
        "plt.grid(True)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n"
    ]

    vis_comp_code = [
        "# 4.2 Visual Comparison\n",
        "\n",
        "# Bar Chart for Metrics\n",
        "metrics_to_plot = ['MAE', 'RMSE', 'R² Score']\n",
        "rnn_metrics = [rnn_mae, rnn_rmse, rnn_r2]\n",
        "trans_metrics = [transformer_mae, transformer_rmse, transformer_r2]\n",
        "\n",
        "x = np.arange(len(metrics_to_plot))\n",
        "width = 0.35\n",
        "\n",
        "plt.figure(figsize=(10, 6))\n",
        "plt.bar(x - width/2, rnn_metrics, width, label='RNN')\n",
        "plt.bar(x + width/2, trans_metrics, width, label='Transformer')\n",
        "\n",
        "plt.xlabel('Metric')\n",
        "plt.ylabel('Score')\n",
        "plt.title('Model Performance Comparison')\n",
        "plt.xticks(x, metrics_to_plot)\n",
        "plt.legend()\n",
        "plt.grid(axis='y')\n",
        "plt.show()\n",
        "\n",
        "# Prediction Overlay\n",
        "plt.figure(figsize=(15, 6))\n",
        "limit = 150\n",
        "plt.plot(y_test[:limit, 0], label='Actual', color='black', alpha=0.6, linewidth=2)\n",
        "plt.plot(y_pred_rnn[:limit, 0], label='RNN Prediction', linestyle='--', alpha=0.8)\n",
        "plt.plot(y_pred_trans[:limit, 0], label='Transformer Prediction', linestyle='-.', alpha=0.8)\n",
        "plt.title(f'RNN vs Transformer Predictions (First {limit} Test Samples)')\n",
        "plt.xlabel('Time Step')\n",
        "plt.ylabel('Normalized Value')\n",
        "plt.legend()\n",
        "plt.grid(True)\n",
        "plt.show()\n"
    ]

    for cell in nb['cells']:
        new_cells.append(cell)
        source_text = "".join(cell.get('source', []))
        
        # Check for target sections and insert IF NOT already present (simple check)
        
        if "2.4 Visualize RNN Results" in source_text and cell['cell_type'] == 'markdown':
            # Append code cell
            new_cells.append(create_code_cell(vis_rnn_code))
            
        elif "3.6 Visualize Transformer Results" in source_text and cell['cell_type'] == 'markdown':
            new_cells.append(create_code_cell(vis_trans_code))
            
        elif "4.2 Visual Comparison" in source_text and cell['cell_type'] == 'markdown':
            new_cells.append(create_code_cell(vis_comp_code))

    nb['cells'] = new_cells
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print("Visualizations added to notebook.")

if __name__ == "__main__":
    add_visualizations()
