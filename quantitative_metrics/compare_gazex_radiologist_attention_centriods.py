import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import re
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 12  # Optional: Set default font size
# Load CSV and JSON data
coordinate_csv = pd.read_csv("./all_bboxes_and_centroids.csv")
with open('./model_inference_result_attention_centriods.json', 'r') as file:
    data = json.load(file)
label_csv = pd.read_csv("./eye_track_unique_label.csv")

# Define categories from the label CSV (excluding 'Reports' and 'No Finding')
categories = [
    'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion', 'Lung Opacity',
    'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
    'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
]

# Dictionary to store coordinates by category
category_data = {cat: {'gt_x': [], 'gt_y': [], 'pred_x': [], 'pred_y': []} for cat in categories}

# Generate unique colors for each category using a colormap
cmap = cm.get_cmap('tab20')  # tab20 has 20 distinct colors, sufficient for 13 categories
# category_colors = {cat: cmap(i / len(categories)) for i, cat in enumerate(categories)}
category_colors = {cat: '#D2691E' for i, cat in enumerate(categories)}
category_colors['Support Devices'] = 'gray'
category_colors['No Finding'] = 'gray'
def normalize_commas(text):
    """
    Normalize commas in a text string by ensuring there's a space before and after each comma,
    but only if there isn't already a space there.
    
    Args:
        text (str): The input text string to normalize
    
    Returns:
        str: The normalized text with proper spacing around commas
    """
    result = ""
    i = 0
    while i < len(text):
        if text[i] == ',' and i + 1 < len(text) and text[i + 1] != ' ':
            result += ', '
            i += 1
        else:
            result += text[i]
            i += 1
    
    final_result = ""
    i = 0
    while i < len(result):
        if result[i] == ',' and i > 0 and result[i - 1] != ' ':
            final_result += ' ,' 
        else:
            final_result += result[i]
        i += 1
    
    return final_result

# Process each case
for result in data:
    pid = result['image'].split('/')[-1].split('.')[0]
    temp_df = coordinate_csv[coordinate_csv['id'] == pid]
    
    gt_report = [s.strip() for s in result['gt_report'].lower().split('.') if s.strip()]
    # gt_bb_list = result['gt_bb_list']
    
    # Get ground truth coordinates
    gt_x_case = [int(row['centroid_x']/4) for index, row in temp_df.iterrows()]
    gt_y_case = [int(row['centroid_y']/4) for index, row in temp_df.iterrows()]
    # gt_bb_list = [gt_x_case,gt_y_case]
    
    pred_bb_list = result['gt_bb_list']
    # print(len(gt_y_case),len(pred_bb_list))
    for condition, gt_bbx,gt_bby, pred_bb in zip(gt_report, gt_x_case,gt_y_case, pred_bb_list):
        condition_clean = normalize_commas(condition.strip().replace('.', ''))
        
        label_row = label_csv[label_csv['Reports'].str.strip().str.lower().str.replace('.', '').str.strip() == normalize_commas(condition_clean)]
        if label_row.empty:
            print(f"No match found for condition: '{normalize_commas(condition_clean)}'")
        else:
            active_categories = [cat for cat in categories if label_row[cat].iloc[0] == 1.0]
            if active_categories:
                # gt_coords = gt_bb.split("),(")
                # gt_last_coord = gt_coords[-1].strip("()")
                gt_x, gt_y = gt_bbx,gt_bby
                
                pred_coords = pred_bb.split("),(")
                pred_last_coord = pred_coords[-1].strip("()")
                pred_x, pred_y = map(int, pred_last_coord.split(","))
                
                for cat in active_categories:
                    category_data[cat]['gt_x'].append(gt_x)
                    category_data[cat]['gt_y'].append(gt_y)
                    category_data[cat]['pred_x'].append(pred_x)
                    category_data[cat]['pred_y'].append(pred_y)

# Create a single figure with two subplots (two rows, one column)
fig, axes = plt.subplots(2, 1, figsize=(10, 20))  # Two rows, one column, height matches two stacked graphs
ax_x, ax_y = axes  # Top for x-coordinates, bottom for y-coordinates

# Plot x-coordinates
for cat in categories:
    gt_x = category_data[cat]['gt_x']
    pred_x = category_data[cat]['pred_x']
    
    if cat != 'Support Devices':
        if len(gt_x) > 0:  # Only plot if there is data
            ax_x.scatter(gt_x, pred_x, c=[category_colors[cat]], label=cat, alpha=0.6,s=200)
    else:
        if len(gt_x) > 0:  # Only plot if there is data
            ax_x.scatter(gt_x, pred_x, c=[category_colors[cat]], label=cat, alpha=0.6,s=100)

# Set x-coordinate plot properties
min_val_x = min([min(category_data[cat]['gt_x'] + category_data[cat]['pred_x'], default=0) for cat in categories], default=0)
max_val_x = max([max(category_data[cat]['gt_x'] + category_data[cat]['pred_x'], default=800) for cat in categories], default=800)
ax_x.plot([min_val_x, max_val_x], [min_val_x, max_val_x], 'k--')  # y=x line, not in legend
ax_x.set_xlabel('Ground Truth X', fontsize=25)
ax_x.set_ylabel('Predicted X', fontsize=25)
ax_x.set_title('X-Coordinates', fontsize=30)
ax_x.grid(True)
ax_x.tick_params(axis='both', labelsize=12)

# Plot y-coordinates
for cat in categories:
    gt_y = category_data[cat]['gt_y']
    pred_y = category_data[cat]['pred_y']
    
    if cat != 'Support Devices':
        if len(gt_y) > 0:  # Only plot if there is data
            ax_y.scatter(gt_y, pred_y, c=[category_colors[cat]], label=cat, alpha=0.6,s=200)
    else:
        if len(gt_y) > 0:  # Only plot if there is data
            ax_y.scatter(gt_y, pred_y, c=[category_colors[cat]], label=cat, alpha=0.6,s=100)

# Set y-coordinate plot properties
min_val_y = min([min(category_data[cat]['gt_y'] + category_data[cat]['pred_y'], default=0) for cat in categories], default=0)
max_val_y = max([max(category_data[cat]['gt_y'] + category_data[cat]['pred_y'], default=800) for cat in categories], default=800)
ax_y.plot([min_val_y, max_val_y], [min_val_y, max_val_y], 'k--')  # y=x line, not in legend
ax_y.set_xlabel('Ground Truth Y', fontsize=25)
ax_y.set_ylabel('Predicted Y', fontsize=25)
ax_y.set_title('Y-Coordinates', fontsize=30)
ax_y.grid(True)
ax_y.tick_params(axis='both', labelsize=12)

# Create a single legend outside the right of both graphs
# handles = [plt.scatter([], [], c=category_colors[cat], label=cat, alpha=0.6) for cat in categories]
# fig.legend(handles, categories, loc='center right', bbox_to_anchor=(0.98, 0.5), title='Categories', fontsize=12, title_fontsize=14, borderaxespad=1.0)

# Adjust layout to prevent overlap and ensure legend visibility
plt.tight_layout()
# plt.subplots_adjust(right=0.65)  # Increased space for the legend
plt.savefig('scatter_plots_combined.png',dpi=600)

# Create a new figure for the legend
fig_legend = plt.figure(figsize=(4, 8))  # Adjust size to fit the legend
ax_legend = fig_legend.add_subplot(111)
ax_legend.axis('off')  # Hide axes
categories = ['Disease Findings','Findings']
category_colors = {'Disease Findings':'#D2691E','Findings':'gray'}
# Recreate the legend handles
handles = [plt.scatter([], [], c=category_colors[cat], label=cat, alpha=0.6) for cat in categories]

# Add the legend to the new figure
legend = ax_legend.legend(handles, categories, loc='center', title='Categories', 
                         fontsize=20, title_fontsize=20)

# Save the legend as a separate image
plt.savefig('legend_separate.png', dpi=600, bbox_inches='tight')
plt.close(fig_legend)  # Close the legend figure to free memory
