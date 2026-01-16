import json
import numpy as np
import csv
from collections import defaultdict
def majority_voting(union, sep_rad):
    # Number of radiologists
    num_radiologists = len(sep_rad)
    # Majority threshold (e.g., for 4 radiologists, need > 2 votes)
    majority_threshold = 2
    
    # Count votes for each finding in union
    majority_findings = []
    
    for finding in union:
        # Skip "Support Devices" if present
        # if finding == "Support Devices":
        #     continue
        # Count how many radiologists reported this finding
        vote_count = sum(1 for rad_findings in sep_rad if finding in rad_findings)
        # print(vote_count, finding)
        # Check if finding meets majority threshold
        if vote_count >= majority_threshold:
            majority_findings.append(finding)
    
    return majority_findings
# Load JSON files
with open('rad.json') as f:
    d = json.load(f)
with open('model_raw.json') as f:
    model_data = json.load(f)

# Get unique IDs
id_list = [i for i in d.keys()]
unique_id_list = np.unique([i[:4] for i in id_list])

# Initialize lists and dictionaries
detailed_rows = []
summary_metrics = defaultdict(lambda: {'TP': 0, 'FN': 0, 'FP': 0})
header = ['Unique_ID', 'Source', 'TP', 'FN', 'FP']

for unique_id in unique_id_list:
    union = []
    sep_rad = []
    model = []
    
    # Collect union and separate radiologist values, excluding "Support Devices"
    for key, value in d.items():
        if key[:4] == unique_id:
            filtered_values = [v for v in value ]
            union.extend(filtered_values)
            sep_rad.append(filtered_values)
    
    # Collect model values, excluding "Support Devices"
    for key, value in model_data.items():
        if key[:4] == unique_id:
            model.extend([v for v in value ])
            break
    
    union = np.unique(union)  # Ground truth unique values
    
    union = majority_voting(union,sep_rad)
    print('u:',union)
    print(unique_id)
    
    # Process each sep_rad entry
    for i, rad_values in enumerate(sep_rad):
        rad_unique = np.unique(rad_values)
        
        TP = 0  # True Positives
        FN = 0  # False Negatives
        FP = 0  # False Positives
        
        # Compute confusion matrix for radiologist
        for val in union:
            if val in rad_unique:
                TP += 1
            else:
                FN += 1
        for val in rad_unique:
            if val not in union:
                FP += 1
        
        # Add to detailed results
        source = f'Radiologist_{i+1}'
        detailed_rows.append([unique_id, source, TP, FN, FP])
        
        # Update summary metrics
        summary_metrics[source]['TP'] += TP
        summary_metrics[source]['FN'] += FN
        summary_metrics[source]['FP'] += FP
    
    # Process model
    model_unique = np.unique(model)
    
    TP = 0  # True Positives
    FN = 0  # False Negatives
    FP = 0  # False Positives
    
    # Compute confusion matrix for model
    for val in union:
        if val in model_unique:
            TP += 1
        else:
            FN += 1
    for val in model_unique:
        if val not in union:
            FP += 1
    
    # Add to detailed results
    detailed_rows.append([unique_id, 'Model', TP, FN, FP])
    
    # Update summary metrics
    summary_metrics['Model']['TP'] += TP
    summary_metrics['Model']['FN'] += FN
    summary_metrics['Model']['FP'] += FP

# Prepare CSV output
csv_rows = []
csv_rows.append(header)
csv_rows.extend(detailed_rows)

# Add a blank row to separate detailed and summary sections
csv_rows.append([])

# Add summary header
csv_rows.append(['Source', 'TP', 'FN', 'FP'])

# Add summary rows
for source, metrics in summary_metrics.items():
    csv_rows.append([source, metrics['TP'], metrics['FN'], metrics['FP']])

# Write to CSV
with open('confusion_matrix_combined.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(csv_rows)
