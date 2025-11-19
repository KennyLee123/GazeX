import pandas as pd
import numpy as np
import cv2
import ast
import os
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial import cKDTree
def centriod(raw_hm,y_min,y_max,x_min,x_max):
    sub = raw_hm[y_min:y_max+1, x_min:x_max+1]
    ys_sub, xs_sub = np.indices(sub.shape)
    W = sub.sum()
    if W > 0:
        cx = x_min + (xs_sub * sub).sum() / W
        cy = y_min + (ys_sub * sub).sum() / W
    else:
        cx, cy = (x_min + x_max)/2, (y_min + y_max)/2
    return cx,cy
def get_heatmap_bounding_box(heatmap,heatmap_raw, threshold=0.1):
    """
    Calculate bounding box for non-zero regions in heatmap
    Returns (x_min, y_min, x_max, y_max) or None if no valid region
    """
    # Find non-zero regions above threshold
    valid_regions = heatmap > threshold
    
    if not np.any(valid_regions):
        return None
    
    # Get coordinates of valid regions
    y_coords, x_coords = np.where(valid_regions)
    
    # Calculate bounding box
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    cx,cy = centriod(heatmap_raw,y_min, y_max,x_min, x_max)
    return (x_min, y_min, x_max, y_max,cx,cy)

def get_heatmap_bounding_box_percent(heatmap,heatmap_raw, top_percent=10):
    """
    Calculate bounding box for top percent of values in heatmap
    Args:
        heatmap: numpy array containing heatmap values
        top_percent: percentage of highest values to include (0-100)
    Returns (x_min, y_min, x_max, y_max) or None if no valid region
    """
    if top_percent <= 0 or top_percent > 100:
        raise ValueError("top_percent must be between 0 and 100")
    
    # Flatten heatmap and sort to find threshold for top percent
    flat_heatmap = heatmap.flatten()
    if not np.any(flat_heatmap > 0):
        return None
        
    threshold = np.percentile(flat_heatmap[flat_heatmap > 0], 100 - top_percent)
    
    # Find regions above threshold
    valid_regions = heatmap > threshold
    
    if not np.any(valid_regions):
        return None
    
    # Get coordinates of valid regions
    y_coords, x_coords = np.where(valid_regions)
    
    # Calculate bounding box
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    cx,cy = centriod(heatmap_raw,y_min, y_max,x_min, x_max)
    
    return (x_min, y_min, x_max, y_max,cx,cy)

def plot_heatmap_with_bbox(heatmap, bbox, title, save_path=None, base_image=None, darkness_strength=0.15):
    """
    Plot heatmap with bounding box overlay
    """
    fig, axes = plt.subplots(1, 2 if base_image is not None else 1, figsize=(15, 6))
    if base_image is not None:
        axes = [axes[0], axes[1]] if hasattr(axes, '__len__') else [axes, axes]
    else:
        axes = [axes] if not hasattr(axes, '__len__') else axes
    
    # Plot 1: Original heatmap with bounding box
    im1 = axes[0].imshow(heatmap, cmap='hot', interpolation='bilinear')
    axes[0].set_title(f'{title} - Heatmap')
    plt.colorbar(im1, ax=axes[0], shrink=0.6)
    
    if bbox is not None:
        x_min, y_min, x_max, y_max,cx,cy = bbox
        rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                        linewidth=2, edgecolor='cyan', facecolor='none')
        axes[0].add_patch(rect)
        axes[0].text(x_min, y_min - 5, f'BBox: ({x_min},{y_min}) to ({x_max},{y_max})', 
                    color='cyan', fontsize=8, weight='bold')
    
    # Plot 2: Enhanced image with heatmap overlay (if base_image provided)
    if base_image is not None:
        gray_img = cv2.cvtColor(base_image, cv2.COLOR_RGB2GRAY) if len(base_image.shape) == 3 else base_image
        final_gray = (gray_img * (darkness_strength + (1 - darkness_strength) * heatmap)).astype(np.uint8)
        final_rgb = cv2.cvtColor(final_gray, cv2.COLOR_GRAY2RGB)
        
        axes[1].imshow(final_rgb)
        axes[1].set_title(f'{title} - Enhanced Overlay')
        
        if bbox is not None:
            x_min, y_min, x_max, y_max,cx,cy = bbox
            rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                            linewidth=2, edgecolor='cyan', facecolor='none')
            axes[1].add_patch(rect)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def process_clusters_optimized(clusters_info, patch, new_W, new_H):
    """
    Optimized cluster processing that uses KDTree search only once per cluster
    """
    # Build KDTree once for all operations
    patch_points = patch[['x', 'y']].values
    tree = cKDTree(patch_points)
    
    # Process each cluster
    for cluster_info in clusters_info:
        cluster_points = cluster_info['points']
        cluster_x = cluster_points[:, 0].astype(int)
        cluster_y = cluster_points[:, 1].astype(int)
        
        # Single KDTree query for all points in this cluster
        cluster_coords = np.column_stack([cluster_x, cluster_y])
        _, closest_indices = tree.query(cluster_coords)
        
        # Get all closest rows at once
        closest_rows = patch.iloc[closest_indices]
        
        # Calculate average timestamp using vectorized operation
        cluster_timestamps = closest_rows['timestamp'].values
        cluster_info['avg_timestamp'] = np.mean(cluster_timestamps)
        
        # Generate heatmap using the same closest_rows data
        cluster_heatmap = np.zeros((new_H, new_W), dtype=np.float32)
        
        for i, (closest_row_idx, (x_pt, y_pt)) in enumerate(zip(closest_indices, cluster_coords)):
            closest_row = closest_rows.iloc[i]
            
            # Use the same Gaussian generation method
            x, y = int(closest_row['x']), int(closest_row['y'])
            sigma_x = closest_row['sigma_x'] * 1.25
            sigma_y = closest_row['sigma_y'] * 1.25
            
            if 0 <= x < new_W and 0 <= y < new_H:
                radius = int(max(sigma_x, sigma_y) * 3)
                x1, x2 = max(x - radius, 0), min(x + radius + 1, new_W)
                y1, y2 = max(y - radius, 0), min(y + radius + 1, new_H)
                
                # Optimized Gaussian calculation
                inv_2sigma_x2 = 1.0 / (2 * sigma_x**2)
                inv_2sigma_y2 = 1.0 / (2 * sigma_y**2)
                
                gx = np.arange(x1, x2)
                gy = np.arange(y1, y2)
                dx = (gx - x) ** 2
                dy = ((gy - y) ** 2).reshape(-1, 1)
                
                g = np.exp(-(dx * inv_2sigma_x2 + dy * inv_2sigma_y2))
                cluster_heatmap[y1:y2, x1:x2] += g
        
        # Store the heatmap in cluster_info if needed
        cluster_info['heatmap'] = cluster_heatmap
    
    # Sort by average timestamp
    clusters_info.sort(key=lambda x: x['avg_timestamp'])
    
    return clusters_info

def extract_patches_from_transcription(transcription_path):
    if not os.path.exists(transcription_path):
        return [], []

    try:
        df = pd.read_csv(transcription_path)
        patch_sentences = []
        patch_intervals = []

        start_idx = 0
        for idx, row in df.iterrows():
            if row['word'] == '.':
                segment = df.iloc[start_idx:idx+1]
                sentence = ' '.join(segment['word'].tolist())
                t_start = segment['timestamp_start_word'].iloc[0]
                t_end = segment['timestamp_end_word'].iloc[-1]
                patch_sentences.append(sentence)
                patch_intervals.append((t_start, t_end))
                start_idx = idx + 1

        return patch_sentences, patch_intervals

    except Exception as e:
        print(f"Error reading {transcription_path}: {e}")
        return [], []

def perform_dbscan_clustering(patch_data, new_H, new_W, eps=0.3, min_samples=10, heatmap_threshold=0.1):
    heatmap_for_clustering = np.zeros((new_H, new_W), dtype=np.float32)
    for _, row in patch_data.iterrows():
        x, y = int(row['x']), int(row['y'])
        sigma_x = row['sigma_x'] * 1.5
        sigma_y = row['sigma_y'] * 1.5
        if 0 <= x < new_W and 0 <= y < new_H:
            radius = int(max(sigma_x, sigma_y) * 3)
            x1, x2 = max(x - radius, 0), min(x + radius + 1, new_W)
            y1, y2 = max(y - radius, 0), min(y + radius + 1, new_H)
            gx = np.arange(x1, x2)
            gy = np.arange(y1, y2)
            xx, yy = np.meshgrid(gx, gy)
            g = np.exp(-((xx - x)**2 / (2 * sigma_x**2) + (yy - y)**2 / (2 * sigma_y**2)))
            heatmap_for_clustering[y1:y2, x1:x2] += g


    if np.max(heatmap_for_clustering) > 0:
        heatmap_for_clustering = heatmap_for_clustering / np.max(heatmap_for_clustering)

    y_coords, x_coords = np.where(heatmap_for_clustering > heatmap_threshold)
    if len(x_coords) == 0:
        return [], heatmap_for_clustering


    intensities = heatmap_for_clustering[y_coords, x_coords]
    

    intensity_weight = 0.1
    scaled_intensity = intensities * intensity_weight


    scaler = StandardScaler()
    xy_scaled = scaler.fit_transform(np.column_stack([x_coords, y_coords]))


    cluster_data_scaled = np.column_stack([xy_scaled, scaled_intensity])
    

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(cluster_data_scaled)
    

    unique_labels = set(cluster_labels)
    unique_labels.discard(-1)

    clusters_info = []
    cluster_data = np.column_stack([x_coords, y_coords, intensities])
    
    for cluster_id in unique_labels:
        cluster_mask = cluster_labels == cluster_id
        cluster_points = cluster_data[cluster_mask]
        
        if len(cluster_points) == 0:
            continue
            
        cluster_x = cluster_points[:, 0]
        cluster_y = cluster_points[:, 1]
        cluster_intensities = cluster_points[:, 2]
        
        cluster_heatmap = np.zeros((new_H, new_W), dtype=np.float32)
        for i in range(len(cluster_x)):
            x, y = int(cluster_x[i]), int(cluster_y[i])
            if 0 <= x < new_W and 0 <= y < new_H:
                cluster_heatmap[y, x] = cluster_intensities[i]
        
        clusters_info.append({
            'cluster_id': cluster_id,
            'points': cluster_points,
            'heatmap': cluster_heatmap
        })
    
    return clusters_info, heatmap_for_clustering

def generate_enhanced_gaze_video_by_id(csv_path: str, sample_id: str, output_path: str = None, 
                                     darkness_strength: float = 0.15, rgb_output_path: str = None,
                                     eps: float = 0.3, min_samples: int = 10, heatmap_threshold: float = 0.1):

    df = pd.read_csv(csv_path)
    row = df[df['id'] == sample_id]
    if row.empty:
        print(f"❌ ID '{sample_id}' 不在 CSV 中。")
        return [], []
    row = row.iloc[0]

    gaze_csv = row['gaze_csv']
    image_path = row['image_path']
    patch_intervals = ast.literal_eval(row['patch_intervals'])
    patch_sentences = ast.literal_eval(row['patch_sentences'])


    gaze_df = pd.read_csv(gaze_csv)
    gaze_df = gaze_df.rename(columns=lambda x: x.strip())
    gaze_df = gaze_df.dropna(subset=[
        'timestamp_sample', 'x_position', 'y_position',
        'angular_resolution_x_pixels_per_degree',
        'angular_resolution_y_pixels_per_degree'
    ])
    gaze_df = gaze_df.rename(columns={
        'timestamp_sample': 'timestamp',
        'x_position': 'x',
        'y_position': 'y',
        'angular_resolution_x_pixels_per_degree': 'sigma_x',
        'angular_resolution_y_pixels_per_degree': 'sigma_y'
    })


    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]
    new_H, new_W = H // 4, W // 4
    img_rgb = cv2.resize(img_rgb, (new_W, new_H), interpolation=cv2.INTER_AREA)
    
    if rgb_output_path:
        cv2.imwrite(rgb_output_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    
    gray_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Store bounding box information
    cluster_bboxes = []
    holistic_bboxes = []
    
    # Create output directory for visualizations
    viz_dir = f'./visualizations/{sample_id}'
    os.makedirs(viz_dir, exist_ok=True)
    
    for idx, ((start, end), sentence) in enumerate(zip(patch_intervals, patch_sentences)):
        start = start - 2  
        patch = gaze_df[(gaze_df['timestamp'] >= start) & (gaze_df['timestamp'] <= end)].copy()

        if patch.empty:
            print(f"⚠️ Patch {idx+1} 无有效数据")
            continue


        patch['x'] = patch['x'] / 4
        patch['y'] = patch['y'] / 4
        patch['sigma_x'] = patch['sigma_x'] / 4
        patch['sigma_y'] = patch['sigma_y'] / 4


        x_low, x_high = np.percentile(patch['x'], [5, 95])
        y_low, y_high = np.percentile(patch['y'], [5, 95])
        patch = patch[
            (patch['x'] >= x_low) & (patch['x'] <= x_high) &
            (patch['y'] >= y_low) & (patch['y'] <= y_high)
        ]

        if patch.empty:
            print(f"⚠️ Patch {idx+1} 清洗后无有效数据")
            continue

        clusters_info, _ = perform_dbscan_clustering(
            patch, new_H, new_W, eps, min_samples, heatmap_threshold
        )

        print(f"Patch {idx+1}: 发现 {len(clusters_info)} 个聚类")

        # Process cluster heatmaps and save bounding boxes
        patch_cluster_bboxes = []
        if len(clusters_info) > 1:
            clusters_info = process_clusters_optimized(clusters_info, patch, new_W, new_H)
            for cluster_idx, temp_clusters in enumerate(clusters_info):
                cluster_heatmap = temp_clusters['heatmap']
                if np.max(cluster_heatmap) > 0:
                    # Apply log transformation and normalization
                    cluster_heatmap_processed = np.log1p(cluster_heatmap + 1e-6)
                    cluster_heatmap_processed = cluster_heatmap_processed / np.max(cluster_heatmap_processed)
                    
                    # Get bounding box
                    bbox = get_heatmap_bounding_box(cluster_heatmap_processed,cluster_heatmap, threshold=0.1)
                    
                    cluster_info = {
                        'patch_id': idx + 1,
                        'cluster_id': cluster_idx,
                        'bbox': bbox,
                        'sentence': sentence
                    }
                    patch_cluster_bboxes.append(cluster_info)
                    
                    # Plot and save visualization
                    plot_title = f'Patch {idx+1} Cluster {cluster_idx}'
                    plot_path = os.path.join(viz_dir, f'patch_{idx+1}_cluster_{cluster_idx}.png')
                    plot_heatmap_with_bbox(cluster_heatmap_processed, bbox, plot_title, 
                                         plot_path, img_rgb, darkness_strength)
                    
                    print(f"  Cluster {cluster_idx} BBox: {bbox}")

        # Generate and process holistic heatmap
        holistic_heatmap = np.zeros((new_H, new_W), dtype=np.float32)
        for _, row2 in patch.iterrows():
            x, y = int(row2['x']), int(row2['y'])
            sigma_x = row2['sigma_x'] * 1.25
            sigma_y = row2['sigma_y'] * 1.25
            if 0 <= x < new_W and 0 <= y < new_H:
                radius = int(max(sigma_x, sigma_y) * 3)
                x1, x2 = max(x - radius, 0), min(x + radius + 1, new_W)
                y1, y2 = max(y - radius, 0), min(y + radius + 1, new_H)
                gx = np.arange(x1, x2)
                gy = np.arange(y1, y2)
                xx, yy = np.meshgrid(gx, gy)
                g = np.exp(-((xx - x)**2 / (2 * sigma_x**2) + (yy - y)**2 / (2 * sigma_y**2)))
                holistic_heatmap[y1:y2, x1:x2] += g

        # Process holistic heatmap and get bounding box
        if np.max(holistic_heatmap) > 0:
            holistic_heatmap_processed = np.log1p(holistic_heatmap + 1e-6)
            holistic_heatmap_processed = holistic_heatmap_processed / np.max(holistic_heatmap_processed)
            
            # Get bounding box for holistic heatmap
            holistic_bbox = get_heatmap_bounding_box_percent(holistic_heatmap_processed,holistic_heatmap,20)
            
            holistic_info = {
                'patch_id': idx + 1,
                'cluster_id': 'holistic_end',
                'bbox': holistic_bbox,
                'sentence': sentence
            }
            holistic_bboxes.append(holistic_info)
            
            # Plot and save holistic visualization
            plot_title = f'Patch {idx+1} Holistic'
            plot_path = os.path.join(viz_dir, f'patch_{idx+1}_holistic.png')
            plot_heatmap_with_bbox(holistic_heatmap_processed, holistic_bbox, plot_title, 
                                 plot_path, img_rgb, darkness_strength)
            
            print(f"  Holistic BBox: {holistic_bbox}")
        
        cluster_bboxes.extend(patch_cluster_bboxes)
        cluster_bboxes.append(holistic_info)

    
    # Convert to DataFrame for easier handling
    cluster_df_data = []
    for bbox_info in cluster_bboxes:
        # print(bbox_info)
        if bbox_info['bbox']:
            x_min, y_min, x_max, y_max,cx,cy = bbox_info['bbox']
            if bbox_info['cluster_id'] == 'holistic_end':
                cluster_df_data.append({
                    'sample_id': sample_id,
                    'type': 'holistic',
                    'patch_id': bbox_info['patch_id'],
                    'cluster_id': -1,
                    'x_min': x_min,
                    'y_min': y_min,
                    'x_max': x_max,
                    'y_max': y_max,
                    'width': x_max - x_min,
                    'height': y_max - y_min,
                    'cx':cx,
                    'cy':cy,
                    'sentence': bbox_info['sentence']
                })
            else:
                cluster_df_data.append({
                'sample_id': sample_id,
                'type': 'cluster',
                'patch_id': bbox_info['patch_id'],
                'cluster_id': bbox_info['cluster_id'],
                'x_min': x_min,
                'y_min': y_min,
                'x_max': x_max,
                'y_max': y_max,
                'width': x_max - x_min,
                'height': y_max - y_min,
                'cx':cx,
                'cy':cy,
                'sentence': bbox_info['sentence']
            })

        
    # holistic_df_data = []
    # for bbox_info in holistic_bboxes:
    #     if bbox_info['bbox']:
    #         x_min, y_min, x_max, y_max = bbox_info['bbox']
    #         holistic_df_data.append({
    #             'sample_id': sample_id,
    #             'type': 'holistic',
    #             'patch_id': bbox_info['patch_id'],
    #             'cluster_id': -1,  # -1 for holistic
    #             'x_min': x_min,
    #             'y_min': y_min,
    #             'x_max': x_max,
    #             'y_max': y_max,
    #             'width': x_max - x_min,
    #             'height': y_max - y_min,
    #             'sentence': bbox_info['sentence']
    #         })
    
    # Combine and save to CSV
    # all_bbox_data = cluster_df_data + holistic_df_data
    all_bbox_data  = cluster_df_data
    if all_bbox_data:
        bbox_df = pd.DataFrame(all_bbox_data)
        bbox_csv_path = os.path.join(viz_dir, f'{sample_id}_bounding_boxes.csv')
        bbox_df.to_csv(bbox_csv_path, index=False)
        print(f"Bounding box data saved to: {bbox_csv_path}")
    
    return cluster_bboxes


if __name__ == "__main__":

    metadata_path = "../reflacx-xray-localization/1.0.0/main_data/metadata_phase_3.csv"
    metadata = pd.read_csv(metadata_path)
    records = []

    for _, row in metadata.iterrows():
        if row['eye_tracking_data_discarded'] == True:
            continue

        patient_id = row['id']
        image_rel_dcm = row['image']


        gaze_csv = os.path.join(
            "../reflacx-xray-localization/1.0.0/gaze_data",
            patient_id,
            "gaze.csv"
        )

        image_path = image_rel_dcm.replace(
            "physionet.org/files/mimic-cxr/2.0.0/files",
            "../mimic-cxr-jpg/2.1.0/files"
        ).replace(".dcm", ".jpg")

        transcription_path = os.path.join(
            "../reflacx-xray-localization/1.0.0/main_data",
            patient_id,
            "timestamps_transcription.csv"
        )

        patch_sentences, patch_intervals = extract_patches_from_transcription(transcription_path)

        records.append({
            "id": patient_id,
            "gaze_csv": gaze_csv,
            "image_path": image_path,
            "patch_intervals": patch_intervals,
            "patch_sentences": patch_sentences
        })

    df_all = pd.DataFrame(records)
    df_all.to_csv("patch_metadata_enhanced_bbox_3.csv", index=False)
    
    # Add new columns for bounding box data
    # df_all['cluster_bboxes'] = [[] for i in range(len(df_all))]
    df_all['bboxes_seq'] = [[] for i in range(len(df_all))]
    
    # Process samples and extract bounding boxes
    for index, row in df_all.iterrows():
        print(f"Processing sample {index + 1}/{len(df_all)}: {row['id']}")
        patient_id = row['id']
        
    
        cluster_bboxes = generate_enhanced_gaze_video_by_id(
            "patch_metadata_enhanced_bbox_3.csv",
            patient_id,
            eps=0.3,  
            min_samples=10,  
            heatmap_threshold=0.1  
        )
        
        df_all.at[index, "bboxes_seq"] = cluster_bboxes
        # if index == 0:
        #     break
    # Save updated DataFrame
    df_all.to_csv("patch_metadata_enhanced_bbox_3.csv", index=False)
    print("Processing complete! Check the ./visualizations/ directory for plots and CSV files.")