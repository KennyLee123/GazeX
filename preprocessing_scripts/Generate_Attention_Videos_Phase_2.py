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
    """读取 timestamps_transcription.csv 并按句号切分返回时间段和句子列表"""
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
    """
    对单个patch的gaze数据进行DBSCAN聚类
    返回聚类信息和用于聚类的heatmap
    """
    # 生成用于聚类的 heatmap (使用第一种normalization)
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

    # 第一种normalization用于聚类
    if np.max(heatmap_for_clustering) > 0:
        heatmap_for_clustering = heatmap_for_clustering / np.max(heatmap_for_clustering)

    # 从 heatmap 中提取高强度点进行聚类
    y_coords, x_coords = np.where(heatmap_for_clustering > heatmap_threshold)
    if len(x_coords) == 0:
        return [], heatmap_for_clustering

    # 使用强度值作为权重
    intensities = heatmap_for_clustering[y_coords, x_coords]
    
    # 准备聚类数据
    intensity_weight = 0.1
    scaled_intensity = intensities * intensity_weight

    # 标准化空间坐标
    scaler = StandardScaler()
    xy_scaled = scaler.fit_transform(np.column_stack([x_coords, y_coords]))

    # 合并数据
    cluster_data_scaled = np.column_stack([xy_scaled, scaled_intensity])
    
    # 进行 DBSCAN 聚类
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(cluster_data_scaled)
    
    # 获取唯一的聚类标签（排除噪声点 -1）
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
        
        # 创建该聚类的单独heatmap
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
    """
    根据 CSV 中的 sample_id，生成包含DBSCAN聚类的 gaze patch 高亮视频
    对于每个patch：如果有多个聚类，先显示各个聚类，再显示holistic view
    
    参数：
        csv_path: 包含 gaze 信息的配置 CSV 路径
        sample_id: 如 "P102R108387"
        output_path: mp4 输出路径；默认保存在当前目录下
        darkness_strength: 非注视区域暗化程度，0 为最暗，1 为不暗化
        eps: DBSCAN 的邻域半径
        min_samples: DBSCAN 的最小样本数
        heatmap_threshold: heatmap 阈值，用于过滤低强度区域
    """
    df = pd.read_csv(csv_path)
    row = df[df['id'] == sample_id]
    if row.empty:
        print(f"❌ ID '{sample_id}' 不在 CSV 中。")
        return
    row = row.iloc[0]

    gaze_csv = row['gaze_csv']
    image_path = row['image_path']
    patch_intervals = ast.literal_eval(row['patch_intervals'])
    patch_sentences = ast.literal_eval(row['patch_sentences'])

    # 加载 gaze 数据
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

    # 加载并缩放图像（宽高除以4）
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]
    new_H, new_W = H // 4, W // 4
    img_rgb = cv2.resize(img_rgb, (new_W, new_H), interpolation=cv2.INTER_AREA)
    
    if rgb_output_path:
        cv2.imwrite(rgb_output_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    
    gray_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    frames = []
    frame_info = []  # 存储每帧的信息
    holistic_frame = []
    for idx, ((start, end), sentence) in enumerate(zip(patch_intervals, patch_sentences)):
        start = start - 2  # 与原始代码保持一致
        patch = gaze_df[(gaze_df['timestamp'] >= start) & (gaze_df['timestamp'] <= end)].copy()

        if patch.empty:
            print(f"⚠️ Patch {idx+1} 无有效数据")
            continue

        # 缩放 gaze 坐标和 sigma（除以4）
        patch['x'] = patch['x'] / 4
        patch['y'] = patch['y'] / 4
        patch['sigma_x'] = patch['sigma_x'] / 4
        patch['sigma_y'] = patch['sigma_y'] / 4

        # 数据清洗：移除异常值
        x_low, x_high = np.percentile(patch['x'], [5, 95])
        y_low, y_high = np.percentile(patch['y'], [5, 95])
        patch = patch[
            (patch['x'] >= x_low) & (patch['x'] <= x_high) &
            (patch['y'] >= y_low) & (patch['y'] <= y_high)
        ]

        if patch.empty:
            print(f"⚠️ Patch {idx+1} 清洗后无有效数据")
            continue

        # 执行DBSCAN聚类
        clusters_info, _ = perform_dbscan_clustering(
            patch, new_H, new_W, eps, min_samples, heatmap_threshold
        )

        print(f"Patch {idx+1}: 发现 {len(clusters_info)} 个聚类")

        # 如果有多个聚类，按时间顺序显示各个聚类
        if len(clusters_info) > 1:
            # 按每个聚类的平均时间戳排序
            # for cluster_info in clusters_info:
            #     cluster_points = cluster_info['points']
            #     cluster_x = cluster_points[:, 0].astype(int)
            #     cluster_y = cluster_points[:, 1].astype(int)
                
            #     # 找到对应的时间戳
            #     cluster_timestamps = []
            #     for x, y in zip(cluster_x, cluster_y):
            #         # 找到最接近的gaze点的时间戳
            #         distances = np.sqrt((patch['x'] - x)**2 + (patch['y'] - y)**2)
            #         closest_idx = distances.idxmin()
            #         cluster_timestamps.append(patch.loc[closest_idx, 'timestamp'])
                
            #     cluster_info['avg_timestamp'] = np.mean(cluster_timestamps)
            
            # # 按平均时间戳排序
            # clusters_info.sort(key=lambda x: x['avg_timestamp'])
            
            # for cluster_info in clusters_info:
            #     cluster_points = cluster_info['points']
            #     cluster_x = cluster_points[:, 0].astype(int)
            #     cluster_y = cluster_points[:, 1].astype(int)
                
            #     # 重新生成cluster heatmap，使用原始gaze数据的Gaussian方法
            #     cluster_heatmap = np.zeros((new_H, new_W), dtype=np.float32)
                
            #     # 找到属于这个cluster的原始gaze点
            #     for x_pt, y_pt in zip(cluster_x, cluster_y):
            #         # 在原始patch数据中找到最接近的点，获取其sigma信息
            #         distances = np.sqrt((patch['x'] - x_pt)**2 + (patch['y'] - y_pt)**2)
            #         closest_idx = distances.idxmin()
            #         closest_row = patch.loc[closest_idx]
                    
            #         # 使用相同的Gaussian生成方法
            #         x, y = int(closest_row['x']), int(closest_row['y'])
            #         sigma_x = closest_row['sigma_x'] * 1.25
            #         sigma_y = closest_row['sigma_y'] * 1.25
                    
            #         if 0 <= x < new_W and 0 <= y < new_H:
            #             radius = int(max(sigma_x, sigma_y) * 3)
            #             x1, x2 = max(x - radius, 0), min(x + radius + 1, new_W)
            #             y1, y2 = max(y - radius, 0), min(y + radius + 1, new_H)
            #             gx = np.arange(x1, x2)
            #             gy = np.arange(y1, y2)
            #             xx, yy = np.meshgrid(gx, gy)
            #             g = np.exp(-((xx - x)**2 / (2 * sigma_x**2) + (yy - y)**2 / (2 * sigma_y**2)))
            #             cluster_heatmap[y1:y2, x1:x2] += g
            clusters_info = process_clusters_optimized(clusters_info,patch,new_W, new_H)
            for temp_clusters in clusters_info:
                cluster_heatmap = temp_clusters['heatmap']
                if np.max(cluster_heatmap) > 0:
                    cluster_heatmap = np.log1p(cluster_heatmap + 1e-6)
                    cluster_heatmap = cluster_heatmap / np.max(cluster_heatmap)
                # 生成该聚类的frame
                final_gray = (gray_img * (darkness_strength + (1 - darkness_strength) * cluster_heatmap)).astype(np.uint8)
                final_rgb = cv2.cvtColor(final_gray, cv2.COLOR_GRAY2RGB)
                
                frames.append(final_rgb)
                # frame_info.append(f"Patch {idx+1} - Cluster {cluster_info['cluster_id']+1}")

        # 生成holistic view (使用所有gaze数据)
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

        # 使用第二种normalization for holistic view
        if np.max(holistic_heatmap) > 0:
            holistic_heatmap = np.log1p(holistic_heatmap + 1e-6)
            holistic_heatmap = holistic_heatmap / np.max(holistic_heatmap)

        # 生成holistic frame
        final_gray = (gray_img * (darkness_strength + (1 - darkness_strength) * holistic_heatmap)).astype(np.uint8)
        final_rgb = cv2.cvtColor(final_gray, cv2.COLOR_GRAY2RGB)
        
        frames.append(final_rgb)
        # frame_info.append(f"Patch {idx+1} - Holistic View")
        holistic_frame.append(len(frames))

    if not frames:
        print("⚠️ 无有效帧生成，跳过视频保存。")
        return

    # 设置输出路径
    if output_path is None:
        output_path = f"{sample_id}_enhanced_gaze_video.mp4"

    # 写入 mp4 视频（1 FPS，每帧显示 1 秒）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, 1.0, (new_W, new_H))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

    print(f"✅ 增强版视频已保存至：{output_path}")
    print(f"📊 总帧数：{len(frames)}")
    # for i, info in enumerate(frame_info):
    #     print(f"   帧 {i+1}: {info}")
    return holistic_frame

# 使用示例
if __name__ == "__main__":
    # 读取 metadata.csv
    metadata_path = "../reflacx-xray-localization/1.0.0/main_data/metadata_phase_2.csv"
    metadata = pd.read_csv(metadata_path)
    records = []

    for _, row in metadata.iterrows():
        if row['eye_tracking_data_discarded'] == True:
            continue

        patient_id = row['id']
        image_rel_dcm = row['image']

        # 构建路径
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
    df_all.to_csv("patch_metadata_enhanced_2.csv", index=False)
    # all_data_holistic_frame
    # 生成增强版视频（示例：处理第一个样本）
    df_all['frame_index'] = [[] for i in range(len(df_all))]
    for index, row in df_all.iterrows():
        print(index)
        patient_id = row['id']
        output_dir = './data'
        os.makedirs(output_dir, exist_ok=True)
        
        holistic_frame = generate_enhanced_gaze_video_by_id(
            "patch_metadata_enhanced_2.csv",
            patient_id,
            output_path=os.path.join(output_dir, f'{patient_id}.mp4'),
            # rgb_output_path=os.path.join(output_dir, f'{patient_id}.png'),
            eps=0.3,  # 可调整的聚类参数
            min_samples=10,  # 可调整的聚类参数
            heatmap_threshold=0.1  # 可调整的heatmap阈值
        )
        df_all.at[index, "frame_index"] = holistic_frame
        # break  # 仅处理第一个样本作为示例
    df_all.to_csv("patch_metadata_enhanced_2.csv", index=False)
        