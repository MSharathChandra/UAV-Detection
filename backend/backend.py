from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import cv2
import time
import torch
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from ultralytics import YOLO
import pandas as pd
import warnings

# Suppress FutureWarning for YOLOv7 (temporary fix)
warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}}, supports_credentials=True)

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = 'http://localhost:3000'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    return response

WEIGHTS_PATH = "C:/Users/M SHARATH CHANDRA/Desktop/Drone-Detection-Project/"
IMAGES_PATH = "C:/Users/M SHARATH CHANDRA/Desktop/Drone-Detection-Project/drone_dataset/train/images"
LABELS_PATH = "C:/Users/M SHARATH CHANDRA/Desktop/Drone-Detection-Project/drone_dataset/train/labels"

# Load models
try:
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = os.path.join(WEIGHTS_PATH, "model_final_280758.pkl")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    predictor = DefaultPredictor(cfg)
except Exception as e:
    print(f"Failed to load R-CNN: {e}")

try:
    model_v5 = YOLO(os.path.join(WEIGHTS_PATH, "yolov5su.pt"))
except Exception as e:
    print(f"Failed to load YOLOv5: {e}")

try:
    model_v7 = torch.hub.load('WongKinYiu/yolov7', 'custom', os.path.join(WEIGHTS_PATH, "yolov7.pt"))
except Exception as e:
    print(f"Failed to load YOLOv7: {e}")

try:
    model_v9 = YOLO(os.path.join(WEIGHTS_PATH, "yolov9c.pt"))
except Exception as e:
    print(f"Failed to load YOLOv9: {e}")

# Helper functions
def normalize_boxes(boxes):
    return [[float(b[0]), float(b[1]), float(b[2]), float(b[3])] for b in boxes]

def calc_fps_and_time(model, img, iterations=10):
    try:
        start = time.time()
        for _ in range(iterations):
            model(img)
        end = time.time()
        fps = iterations / (end - start)
        inference_time = (end - start) * 1000 / iterations
        return fps, inference_time
    except Exception as e:
        print(f"Failed to calculate FPS: {e}")
        return 0, 0

def small_object_score(detections, img, threshold=0.05):
    if not detections:
        return 0
    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in detections]
    small_objs = sum(1 for area in areas if area < threshold * img.shape[0] * img.shape[1])
    score = small_objs / len(areas) if areas else 0
    return score * 100

def calc_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_gt, y1_gt, x2_gt, y2_gt = box2
    xi1 = max(x1, x1_gt)
    yi1 = max(y1, y1_gt)
    xi2 = min(x2, x2_gt)
    yi2 = min(y2, y2_gt)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def calc_map(pred_boxes_all, gt_boxes_all, iou_threshold=0.5):
    all_preds = []
    all_gts = []
    for preds, gts in zip(pred_boxes_all, gt_boxes_all):
        for pred in preds:
            confidence = pred[4] if len(pred) > 4 else 1.0
            all_preds.append({"box": pred[:4], "confidence": confidence})
        all_gts.extend(gts)

    if not all_gts or not all_preds:
        return 0.0

    all_preds = sorted(all_preds, key=lambda x: x["confidence"], reverse=True)
    tp = [0] * len(all_preds)
    fp = [0] * len(all_preds)
    gt_matched = set()
    for i, pred in enumerate(all_preds):
        best_iou = 0
        best_gt_idx = -1
        for j, gt in enumerate(all_gts):
            if j in gt_matched:
                continue
            iou = calc_iou(pred["box"], gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp[i] = 1
            gt_matched.add(best_gt_idx)
        else:
            fp[i] = 1

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    total_gt = len(all_gts)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    recalls = tp_cumsum / (total_gt + 1e-6)

    precisions_out = []
    for recall_level in np.linspace(0, 1, 11):
        prec = [p for r, p in zip(recalls, precisions) if r >= recall_level]
        precisions_out.append(max(prec) if prec else 0)
    ap = np.mean(precisions_out) * 100
    return min(ap, 100.0)

def calc_precision_recall(pred_boxes_all, gt_boxes_all, iou_threshold=0.5):
    total_tp, total_detections, total_gt = 0, 0, 0
    for pred_boxes, gt_boxes in zip(pred_boxes_all, gt_boxes_all):
        if not pred_boxes or not gt_boxes:
            continue
        tp = sum(1 for det in pred_boxes for gt in gt_boxes if calc_iou(det[:4], gt) >= iou_threshold)
        total_tp += tp
        total_detections += len(pred_boxes)
        total_gt += len(gt_boxes)
    precision = (total_tp / total_detections * 100) if total_detections > 0 else 0
    recall = (total_tp / total_gt * 100) if total_gt > 0 else 0
    return precision, recall

def load_ground_truth(image_name):
    label_file = os.path.join(LABELS_PATH, image_name.replace('.jpg', '.txt').replace('.png', '.txt'))
    gt_boxes = []
    if os.path.exists(label_file):
        img = cv2.imread(os.path.join(IMAGES_PATH, image_name))
        if img is None:
            print(f"Image not found: {image_name}")
            return gt_boxes
        img_h, img_w = img.shape[:2]
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                class_id, x_center, y_center, width, height = map(float, parts[:5])
                x1 = (x_center - width / 2) * img_w
                y1 = (y_center - height / 2) * img_h
                x2 = (x_center + width / 2) * img_w
                y2 = (y_center + height / 2) * img_h
                gt_boxes.append([x1, y1, x2, y2])
    return gt_boxes

def draw_labels(img, detections, algo_name):
    for box in detections:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{algo_name}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img

def generate_algorithm_explanations(df, best_algo):
    explanations = {}
    algorithms = df['Algorithm'].tolist()

    # Define metrics where lower is better
    lower_is_better = ['Inference Time (ms)', 'Model Size (MB)', 'Number of Parameters (M)']
    higher_is_better = ['mAP', 'FPS', 'Precision', 'Recall', 'Small Object Score']

    for index, row in df.iterrows():
        algo = row['Algorithm']
        explanation = []

        # Get the rows for all other algorithms for comparison
        other_rows = df[df['Algorithm'] != algo]

        if algo == best_algo:
            explanation.append(f"Why {algo} is the Best:")
            # Check each metric to see if this algorithm is the best
            for metric in higher_is_better:
                if metric in df.columns:
                    max_val = df[metric].max()
                    algo_val = row[metric]
                    if algo_val == max_val:
                        explanation.append(f"- {metric} ({algo_val:.2f}{'%' if metric in ['mAP', 'Precision', 'Recall', 'Small Object Score'] else ''}): Highest among all algorithms.")
            for metric in lower_is_better:
                if metric in df.columns:
                    min_val = df[metric].min()
                    algo_val = row[metric]
                    if algo_val == min_val:
                        explanation.append(f"- {metric} ({algo_val:.2f}{'' if metric in ['Inference Time (ms)', 'Model Size (MB)'] else 'M'}): Lowest among all algorithms.")

            explanation.append("\nComparisons with Other Algorithms:")
            for _, other_row in other_rows.iterrows():
                other_algo = other_row['Algorithm']
                comparisons = []
                for metric in higher_is_better:
                    if metric in df.columns and row[metric] > other_row[metric]:
                        comparisons.append(f"Higher {metric} ({row[metric]:.2f}{'%' if metric in ['mAP', 'Precision', 'Recall', 'Small Object Score'] else ''} vs {other_row[metric]:.2f}{'%' if metric in ['mAP', 'Precision', 'Recall', 'Small Object Score'] else ''}) than {other_algo}.")
                for metric in lower_is_better:
                    if metric in df.columns and row[metric] < other_row[metric]:
                        comparisons.append(f"Lower {metric} ({row[metric]:.2f}{'' if metric in ['Inference Time (ms)', 'Model Size (MB)'] else 'M'} vs {other_row[metric]:.2f}{'' if metric in ['Inference Time (ms)', 'Model Size (MB)'] else 'M'}) than {other_algo}.")
                if comparisons:
                    explanation.append(f"- {' '.join(comparisons)}")
                else:
                    explanation.append(f"- No significant advantages over {other_algo} in these metrics.")

            explanation.append(f"Overall, {algo} excels in key areas, making it the top choice for drone detection!")
        else:
            explanation.append(f"Why {algo} is Not the Best:")
            best_row = df[df['Algorithm'] == best_algo].iloc[0]
            reasons = []

            for metric in higher_is_better:
                if metric in df.columns and row[metric] < best_row[metric]:
                    reasons.append(f"- Lower {metric} ({row[metric]:.2f}{'%' if metric in ['mAP', 'Precision', 'Recall', 'Small Object Score'] else ''} vs {best_row[metric]:.2f}{'%' if metric in ['mAP', 'Precision', 'Recall', 'Small Object Score'] else ''}): Less than {best_algo}.")
                elif metric in df.columns:
                    explanation.append(f"- {metric} ({row[metric]:.2f}{'%' if metric in ['mAP', 'Precision', 'Recall', 'Small Object Score'] else ''}): Matches or exceeds {best_algo}.")

            for metric in lower_is_better:
                if metric in df.columns and row[metric] > best_row[metric]:
                    reasons.append(f"- Higher {metric} ({row[metric]:.2f}{'' if metric in ['Inference Time (ms)', 'Model Size (MB)'] else 'M'} vs {best_row[metric]:.2f}{'' if metric in ['Inference Time (ms)', 'Model Size (MB)'] else 'M'}): Worse than {best_algo}.")
                elif metric in df.columns:
                    explanation.append(f"- {metric} ({row[metric]:.2f}{'' if metric in ['Inference Time (ms)', 'Model Size (MB)'] else 'M'}): Matches or better than {best_algo}.")

            if reasons:
                explanation.append("\n".join(reasons))

            explanation.append(f"\nComparisons with Other Algorithms:")
            for _, other_row in other_rows.iterrows():
                other_algo = other_row['Algorithm']
                if other_algo == best_algo:
                    continue
                comparisons = []
                for metric in higher_is_better:
                    if metric in df.columns:
                        if row[metric] > other_row[metric]:
                            comparisons.append(f"Higher {metric} ({row[metric]:.2f}{'%' if metric in ['mAP', 'Precision', 'Recall', 'Small Object Score'] else ''} vs {other_row[metric]:.2f}{'%' if metric in ['mAP', 'Precision', 'Recall', 'Small Object Score'] else ''}) than {other_algo}.")
                        elif row[metric] < other_row[metric]:
                            comparisons.append(f"Lower {metric} ({row[metric]:.2f}{'%' if metric in ['mAP', 'Precision', 'Recall', 'Small Object Score'] else ''} vs {other_row[metric]:.2f}{'%' if metric in ['mAP', 'Precision', 'Recall', 'Small Object Score'] else ''}) than {other_algo}.")
                for metric in lower_is_better:
                    if metric in df.columns:
                        if row[metric] < other_row[metric]:
                            comparisons.append(f"Lower {metric} ({row[metric]:.2f}{'' if metric in ['Inference Time (ms)', 'Model Size (MB)'] else 'M'} vs {other_row[metric]:.2f}{'' if metric in ['Inference Time (ms)', 'Model Size (MB)'] else 'M'}) than {other_algo}.")
                        elif row[metric] > other_row[metric]:
                            comparisons.append(f"Higher {metric} ({row[metric]:.2f}{'' if metric in ['Inference Time (ms)', 'Model Size (MB)'] else 'M'} vs {other_row[metric]:.2f}{'' if metric in ['Inference Time (ms)', 'Model Size (MB)'] else 'M'}) than {other_algo}.")
                if comparisons:
                    explanation.append(f"- {' '.join(comparisons)}")
                else:
                    explanation.append(f"- No significant differences with {other_algo} in these metrics.")

            explanation.append(f"Compared to {best_algo}, {algo} falls short in key areas!")

        explanations[algo] = "\n".join(explanation)
    return explanations

def calculate_weighted_scores(df):
    # Define the metrics to normalize
    metrics = ['Precision', 'Recall', 'mAP', 'Inference Time (ms)', 'Model Size (MB)']
    
    # Create a copy of the DataFrame for normalization
    normalized_df = df.copy()
    
    # Ensure the columns are numeric
    for metric in metrics:
        if metric in normalized_df.columns:
            normalized_df[metric] = pd.to_numeric(normalized_df[metric], errors='coerce')
    
    # Store min and max values for each metric for explanation
    min_max_values = {}
    for metric in metrics:
        if metric in normalized_df.columns:
            values = normalized_df[metric]
            min_max_values[metric] = {'min': values.min(), 'max': values.max()}
    
    # Normalize the metrics (Min-Max normalization)
    for metric in metrics:
        if metric in normalized_df.columns:
            values = normalized_df[metric]
            min_val = min_max_values[metric]['min']
            max_val = min_max_values[metric]['max']
            
            # Avoid division by zero
            if max_val == min_val:
                normalized_df[metric] = 0  # If all values are the same, set normalized value to 0
            else:
                normalized_df[metric] = (values - min_val) / (max_val - min_val)
    
    # Define weights for different priorities
    weights = {
        'Balanced': {'Precision': 0.2, 'Recall': 0.2, 'mAP': 0.2, 'Inference Time (ms)': 0.2, 'Model Size (MB)': 0.2},
        'Accuracy': {'Precision': 0.25, 'Recall': 0.25, 'mAP': 0.25, 'Inference Time (ms)': 0.125, 'Model Size (MB)': 0.125},
        'Real-Time Detection': {'Precision': 0.15, 'Recall': 0.15, 'mAP': 0.15, 'Inference Time (ms)': 0.45, 'Model Size (MB)': 0.1},
        'Deployment': {'Precision': 0.15, 'Recall': 0.15, 'mAP': 0.15, 'Inference Time (ms)': 0.15, 'Model Size (MB)': 0.4}
    }
    
    # Invert Inference Time and Model Size (lower is better)
    for metric in ['Inference Time (ms)', 'Model Size (MB)']:
        if metric in normalized_df.columns:
            normalized_df[metric] = 1 - normalized_df[metric]
    
    # Calculate weighted scores for each priority
    tab_results = {}
    for priority, weight_dict in weights.items():
        weighted_scores = pd.Series(0.0, index=normalized_df.index)
        weights_explanation = f"Weights for {priority}:\n"
        for metric, weight in weight_dict.items():
            if metric in normalized_df.columns:
                weighted_scores += normalized_df[metric] * weight
                weights_explanation += f"{metric}: {weight}\n"
        
        # Add min and max values for each metric
        weights_explanation += "\nMin and Max Values Used for Normalization:\n"
        for metric in metrics:
            if metric in min_max_values:
                min_val = min_max_values[metric]['min']
                max_val = min_max_values[metric]['max']
                if metric in ['Precision', 'Recall', 'mAP']:
                    weights_explanation += f"{metric}: Min = {min_val:.2f}%, Max = {max_val:.2f}%\n"
                else:
                    weights_explanation += f"{metric}: Min = {min_val:.2f}, Max = {max_val:.2f}\n"
        
        # Create a table with Algorithm and Weighted Score
        table = pd.DataFrame({
            'Algorithm': normalized_df['Algorithm'],
            'Weighted Score': weighted_scores
        })
        
        # Sort by weighted score in descending order
        table = table.sort_values(by='Weighted Score', ascending=False).reset_index(drop=True)
        
        # Get the best algorithm
        best_algo = table.iloc[0]['Algorithm']
        
        # Merge the sorted table with the original df to get the raw metrics
        display_table = table.merge(df[metrics + ['Algorithm']], on='Algorithm', how='left')
        
        # Add normalized values to the display table
        for metric in metrics:
            if metric in normalized_df.columns:
                # Create a column for normalized values (before inversion for Inference Time and Model Size)
                normalized_values = (df[metric] - min_max_values[metric]['min']) / (min_max_values[metric]['max'] - min_max_values[metric]['min'])
                if metric in ['Inference Time (ms)', 'Model Size (MB)']:
                    # After inversion (since lower is better)
                    normalized_values = 1 - normalized_values
                display_table[f'Normalized {metric}'] = normalized_values
        
        # Format the metrics for display
        for metric in ['Precision', 'Recall', 'mAP']:
            if metric in display_table.columns:
                display_table[metric] = display_table[metric].apply(lambda x: f"{x:.2f}%")
        for metric in ['Inference Time (ms)', 'Model Size (MB)']:
            if metric in display_table.columns:
                display_table[metric] = display_table[metric].apply(lambda x: f"{x:.2f}")
        for metric in [f'Normalized {m}' for m in metrics]:
            if metric in display_table.columns:
                display_table[metric] = display_table[metric].apply(lambda x: f"{x:.4f}")
        display_table['Weighted Score'] = display_table['Weighted Score'].apply(lambda x: f"{x:.2f}")
        
        # Generate explanations for each algorithm, including normalization calculations
        algorithm_explanations = {}
        for _, row in table.iterrows():
            algo = row['Algorithm']
            score = row['Weighted Score']
            explanation = f"{algo} achieved a weighted score of {score:.2f} under {priority} priority.\n"
            for metric in metrics:
                if metric in df.columns:
                    raw_value = df[df['Algorithm'] == algo][metric].iloc[0]
                    min_val = min_max_values[metric]['min']
                    max_val = min_max_values[metric]['max']
                    if max_val == min_val:
                        normalized_value = 0
                    else:
                        normalized_value = (raw_value - min_val) / (max_val - min_val)
                    if metric in ['Precision', 'Recall', 'mAP']:
                        explanation += f"{metric}: {raw_value:.2f}%\n"
                        explanation += f"Normalized {metric}: {normalized_value:.4f} (Formula: ({raw_value:.2f} - {min_val:.2f}) / ({max_val:.2f} - {min_val:.2f}))\n"
                    else:
                        explanation += f"{metric}: {raw_value:.2f}\n"
                        explanation += f"Normalized {metric} (before inversion): {normalized_value:.4f} (Formula: ({raw_value:.2f} - {min_val:.2f}) / ({max_val:.2f} - {min_val:.2f}))\n"
                        if metric in ['Inference Time (ms)', 'Model Size (MB)']:
                            inverted_value = 1 - normalized_value
                            explanation += f"Normalized {metric} (after inversion, since lower is better): {inverted_value:.4f}\n"
            algorithm_explanations[algo] = explanation
        
        tab_results[priority] = {
            'table': display_table.to_dict('records'),
            'best_algo': best_algo,
            'algorithm_explanations': algorithm_explanations,
            'weights_explanation': weights_explanation.strip()
        }
    
    return tab_results

def process_images(image_list, is_custom=False):
    results = {
        "Algorithm": [], "mAP": [], "FPS": [], "Small Object Score": [], "Model Size (MB)": [],
        "Precision": [], "Recall": [], "Inference Time (ms)": [], "Number of Parameters (M)": []
    }
    labeled_images = {"R-CNN": [], "YOLOv5": [], "YOLOv7": [], "YOLOv9": []}
    
    rcnn_preds, v5_preds, v7_preds, v9_preds = [], [], [], []
    gt_boxes_all = []
    fps_sum, inf_time_sum = {"R-CNN": 0, "YOLOv5": 0, "YOLOv7": 0, "YOLOv9": 0}, {"R-CNN": 0, "YOLOv5": 0, "YOLOv7": 0, "YOLOv9": 0}

    all_detections_dir = os.path.join("static", "all_detections")
    os.makedirs(all_detections_dir, exist_ok=True)

    for idx, img_data in enumerate(image_list):
        if is_custom:
            img_name = f"custom_{idx}.jpg"
            img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            gt_boxes = []
        else:
            img_name = img_data
            img = cv2.imread(os.path.join(IMAGES_PATH, img_name))
            gt_boxes = load_ground_truth(img_name)

        if img is None:
            print(f"Failed to load image: {img_name}")
            continue

        gt_boxes_all.append(gt_boxes)

        try:
            outputs_rcnn = predictor(img)
            pred_boxes_rcnn = outputs_rcnn["instances"].pred_boxes.tensor.cpu().numpy().tolist()
            scores_rcnn = outputs_rcnn["instances"].scores.cpu().numpy().tolist()
            pred_boxes_rcnn = [box + [score] for box, score in zip(normalize_boxes(pred_boxes_rcnn), scores_rcnn)]
            rcnn_preds.append(pred_boxes_rcnn)
            fps, inf_time = calc_fps_and_time(predictor, img)
            fps_sum["R-CNN"] += fps
            inf_time_sum["R-CNN"] += inf_time
            img_rcnn = draw_labels(img.copy(), pred_boxes_rcnn, "R-CNN")
            rcnn_file = f"rcnn_{img_name}"
            cv2.imwrite(os.path.join(all_detections_dir, rcnn_file), img_rcnn)
            labeled_images["R-CNN"].append(f"http://localhost:5000/static/all_detections/{rcnn_file}")
        except Exception as e:
            print(f"R-CNN failed on {img_name}: {e}")
            rcnn_preds.append([])

        try:
            results_v5 = model_v5(img)
            pred_boxes_v5 = results_v5[0].boxes.xyxy.cpu().numpy().tolist()
            scores_v5 = results_v5[0].boxes.conf.cpu().numpy().tolist()
            pred_boxes_v5 = [box + [score] for box, score in zip(normalize_boxes(pred_boxes_v5), scores_v5)]
            v5_preds.append(pred_boxes_v5)
            fps, inf_time = calc_fps_and_time(model_v5, img)
            fps_sum["YOLOv5"] += fps
            inf_time_sum["YOLOv5"] += inf_time
            img_v5 = draw_labels(img.copy(), pred_boxes_v5, "YOLOv5")
            v5_file = f"yolov5_{img_name}"
            cv2.imwrite(os.path.join(all_detections_dir, v5_file), img_v5)
            labeled_images["YOLOv5"].append(f"http://localhost:5000/static/all_detections/{v5_file}")
        except Exception as e:
            print(f"YOLOv5 failed on {img_name}: {e}")
            v5_preds.append([])

        try:
            detections_v7 = model_v7(img)
            pred_boxes_v7 = detections_v7.xyxy[0].cpu().numpy().tolist()
            pred_boxes_v7 = [box[:4] + [box[4]] for box in pred_boxes_v7]
            v7_preds.append(pred_boxes_v7)
            fps, inf_time = calc_fps_and_time(model_v7, img)
            fps_sum["YOLOv7"] += fps
            inf_time_sum["YOLOv7"] += inf_time
            img_v7 = draw_labels(img.copy(), pred_boxes_v7, "YOLOv7")
            v7_file = f"yolov7_{img_name}"
            cv2.imwrite(os.path.join(all_detections_dir, v7_file), img_v7)
            labeled_images["YOLOv7"].append(f"http://localhost:5000/static/all_detections/{v7_file}")
        except Exception as e:
            print(f"YOLOv7 failed on {img_name}: {e}")
            v7_preds.append([])

        try:
            results_v9 = model_v9(img)
            pred_boxes_v9 = results_v9[0].boxes.xyxy.cpu().numpy().tolist()
            scores_v9 = results_v9[0].boxes.conf.cpu().numpy().tolist()
            pred_boxes_v9 = [box + [score] for box, score in zip(normalize_boxes(pred_boxes_v9), scores_v9)]
            v9_preds.append(pred_boxes_v9)
            fps, inf_time = calc_fps_and_time(model_v9, img)
            fps_sum["YOLOv9"] += fps
            inf_time_sum["YOLOv9"] += inf_time
            img_v9 = draw_labels(img.copy(), pred_boxes_v9, "YOLOv9")
            v9_file = f"yolov9_{img_name}"
            cv2.imwrite(os.path.join(all_detections_dir, v9_file), img_v9)
            labeled_images["YOLOv9"].append(f"http://localhost:5000/static/all_detections/{v9_file}")
        except Exception as e:
            print(f"YOLOv9 failed on {img_name}: {e}")
            v9_preds.append([])

    num_images = len(image_list)
    if num_images == 0:
        return None, None, 0

    for algo, preds in [("R-CNN", rcnn_preds), ("YOLOv5", v5_preds), ("YOLOv7", v7_preds), ("YOLOv9", v9_preds)]:
        mAP = calc_map(preds, gt_boxes_all)
        precision, recall = calc_precision_recall(preds, gt_boxes_all)
        avg_fps = fps_sum[algo] / num_images
        avg_inf_time = inf_time_sum[algo] / num_images
        small_score = np.mean([small_object_score(p, cv2.imread(os.path.join(IMAGES_PATH, f)) if not is_custom else cv2.imdecode(np.frombuffer(image_list[i], np.uint8), cv2.IMREAD_COLOR)) for i, (p, f) in enumerate(zip(preds, image_list))])

        model_size = os.path.getsize(os.path.join(WEIGHTS_PATH, {
            "R-CNN": "model_final_280758.pkl",
            "YOLOv5": "yolov5su.pt",
            "YOLOv7": "yolov7.pt",
            "YOLOv9": "yolov9c.pt"
        }[algo])) / (1024 * 1024)

        num_params = sum(p.numel() for p in {
            "R-CNN": predictor.model,
            "YOLOv5": model_v5.model,
            "YOLOv7": model_v7,
            "YOLOv9": model_v9.model
        }[algo].parameters()) / 1e6

        results["Algorithm"].append(algo)
        results["mAP"].append(mAP)  # Store as float
        results["FPS"].append(avg_fps)
        results["Small Object Score"].append(small_score)  # Store as float
        results["Model Size (MB)"].append(model_size)
        results["Precision"].append(precision)  # Store as float
        results["Recall"].append(recall)  # Store as float
        results["Inference Time (ms)"].append(avg_inf_time)
        results["Number of Parameters (M)"].append(num_params)

    df = pd.DataFrame(results)
    return df, labeled_images, num_images

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(IMAGES_PATH, filename)

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory("static", filename)

@app.route('/images', methods=['GET'])
def get_images():
    try:
        image_files = [f"http://localhost:5000/images/{f}" for f in os.listdir(IMAGES_PATH) if f.endswith(('.jpg', '.png'))]
        return jsonify({"images": image_files})
    except Exception as e:
        return jsonify({"error": f"Failed to fetch images: {e}"}), 500

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided!"}), 400
    
    file = request.files['image']
    image_name = file.filename
    img_data = file.read()
    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
    
    if img is None:
        return jsonify({"error": "Invalid image!"}), 400

    df, labeled_images, num_processed = process_images([image_name])
    if df is None:
        return jsonify({"error": "Processing failed!"}), 500

    # Format the DataFrame for display
    display_df = df.copy()
    for col in ['mAP', 'Precision', 'Recall', 'Small Object Score']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%")
    for col in ['FPS', 'Inference Time (ms)', 'Model Size (MB)', 'Number of Parameters (M)']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")

    best_algo = df.loc[df['mAP'].idxmax()]['Algorithm'] if not df.empty else "None"
    algorithm_explanations = generate_algorithm_explanations(df, best_algo)
    metric_explanations = {
        "mAP": "Mean Average Precision (%): Detection accuracy across IoU thresholds (higher is better).",
        "FPS": "Frames Per Second: Speed of detection (higher is better).",
        "Small Object Score": "Small objects detected (%): Higher means better small object detection.",
        "Model Size (MB)": "Size of model file (smaller is better for deployment).",
        "Precision": "Correct detections out of all detections (%): Higher is better.",
        "Recall": "Correct detections out of all ground truths (%): Higher is better.",
        "Inference Time (ms)": "Time per inference (lower is best).",
        "Number of Parameters (M)": "Model complexity (fewer is better for efficiency)."
    }
    selection_reason = (
        "Why These Algorithms?\n"
        "We selected R-CNN, YOLOv5, YOLOv7, and YOLOv9 for drone detection research:\n"
        "- R-CNN: High accuracy two-stage detector, great for small objects like drones.\n"
        "- YOLOv5: Balanced speed and accuracy, lightweight for real-time use.\n"
        "- YOLOv7: Improved small object detection and efficiency.\n"
        "- YOLOv9: State-of-the-art with top mAP and FPS.\n"
        "These cover accuracy, speed, and deployment needs!"
    )

    tab_results = calculate_weighted_scores(df)

    return jsonify({
        "images": {
            "rcnn": labeled_images["R-CNN"][0],
            "yolov5": labeled_images["YOLOv5"][0],
            "yolov7": labeled_images["YOLOv7"][0],
            "yolov9": labeled_images["YOLOv9"][0]
        },
        "table": display_df.to_dict(orient="records"),
        "best_algo": best_algo,
        "metric_explanations": metric_explanations,
        "algorithm_explanations": algorithm_explanations,
        "selection_reason": selection_reason,
        "tab_results": tab_results
    })

@app.route('/detect_all', methods=['GET'])
def detect_all():
    image_files = [f for f in os.listdir(IMAGES_PATH) if f.endswith(('.jpg', '.png'))]
    if not image_files:
        return jsonify({"error": "No images found!"}), 400

    df, labeled_images, num_processed = process_images(image_files)
    if df is None:
        return jsonify({"error": "Processing failed!"}), 500

    # Format the DataFrame for display
    display_df = df.copy()
    for col in ['mAP', 'Precision', 'Recall', 'Small Object Score']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%")
    for col in ['FPS', 'Inference Time (ms)', 'Model Size (MB)', 'Number of Parameters (M)']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")

    best_algo = df.loc[df['mAP'].idxmax()]['Algorithm'] if not df.empty else "None"
    algorithm_explanations = generate_algorithm_explanations(df, best_algo)
    metric_explanations = {
        "mAP": "Mean Average Precision (%): Detection accuracy across IoU thresholds (higher is better).",
        "FPS": "Frames Per Second: Speed of detection (higher is better).",
        "Small Object Score": "Small objects detected (%): Higher means better small object detection.",
        "Model Size (MB)": "Size of model file (smaller is better for deployment).",
        "Precision": "Correct detections out of all detections (%): Higher is better.",
        "Recall": "Correct detections out of all ground truths (%): Higher is better.",
        "Inference Time (ms)": "Time per inference (lower is best).",
        "Number of Parameters (M)": "Model complexity (fewer is better for efficiency)."
    }
    selection_reason = (
        "Why These Algorithms?\n"
        "We selected R-CNN, YOLOv5, YOLOv7, and YOLOv9 for drone detection research:\n"
        "- R-CNN: High accuracy two-stage detector, great for small objects like drones.\n"
        "- YOLOv5: Balanced speed and accuracy, lightweight for real-time use.\n"
        "- YOLOv7: Improved small object detection and efficiency.\n"
        "- YOLOv9: State-of-the-art with top mAP and FPS.\n"
        "These cover accuracy, speed, and deployment needs!"
    )

    tab_results = calculate_weighted_scores(df)

    return jsonify({
        "table": display_df.to_dict(orient="records"),
        "best_algo": best_algo,
        "metric_explanations": metric_explanations,
        "algorithm_explanations": algorithm_explanations,
        "selection_reason": selection_reason,
        "labeled_images": labeled_images,
        "num_images_processed": num_processed,
        "tab_results": tab_results
    })

@app.route('/detect_range', methods=['POST'])
def detect_range():
    data = request.get_json()
    num_images = data.get('num_images', 0)
    image_files = [f for f in os.listdir(IMAGES_PATH) if f.endswith(('.jpg', '.png'))]
    
    if not image_files:
        return jsonify({"error": "No images found!"}), 400
    if num_images <= 0 or num_images > len(image_files):
        return jsonify({"error": f"Invalid range! Max available: {len(image_files)}"}), 400

    selected_images = image_files[:num_images]
    df, labeled_images, num_processed = process_images(selected_images)
    if df is None:
        return jsonify({"error": "Processing failed!"}), 500

    # Format the DataFrame for display
    display_df = df.copy()
    for col in ['mAP', 'Precision', 'Recall', 'Small Object Score']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%")
    for col in ['FPS', 'Inference Time (ms)', 'Model Size (MB)', 'Number of Parameters (M)']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")

    best_algo = df.loc[df['mAP'].idxmax()]['Algorithm'] if not df.empty else "None"
    algorithm_explanations = generate_algorithm_explanations(df, best_algo)
    metric_explanations = {
        "mAP": "Mean Average Precision (%): Detection accuracy across IoU thresholds (higher is better).",
        "FPS": "Frames Per Second: Speed of detection (higher is better).",
        "Small Object Score": "Small objects detected (%): Higher means better small object detection.",
        "Model Size (MB)": "Size of model file (smaller is better for deployment).",
        "Precision": "Correct detections out of all detections (%): Higher is better.",
        "Recall": "Correct detections out of all ground truths (%): Higher is better.",
        "Inference Time (ms)": "Time per inference (lower is best).",
        "Number of Parameters (M)": "Model complexity (fewer is better for efficiency)."
    }
    selection_reason = (
        "Why These Algorithms?\n"
        "We selected R-CNN, YOLOv5, YOLOv7, and YOLOv9 for drone detection research:\n"
        "- R-CNN: High accuracy two-stage detector, great for small objects like drones.\n"
        "- YOLOv5: Balanced speed and accuracy, lightweight for real-time use.\n"
        "- YOLOv7: Improved small object detection and efficiency.\n"
        "- YOLOv9: State-of-the-art with top mAP and FPS.\n"
        "These cover accuracy, speed, and deployment needs!"
    )

    tab_results = calculate_weighted_scores(df)

    return jsonify({
        "table": display_df.to_dict(orient="records"),
        "best_algo": best_algo,
        "metric_explanations": metric_explanations,
        "algorithm_explanations": algorithm_explanations,
        "selection_reason": selection_reason,
        "labeled_images": labeled_images,
        "num_images_processed": num_processed,
        "tab_results": tab_results
    })

@app.route('/detect_custom', methods=['POST'])
def detect_custom():
    data = request.get_json()
    image_names = data.get('images', [])
    if not image_names:
        return jsonify({"error": "No images provided!"}), 400

    df, labeled_images, num_processed = process_images(image_names)
    if df is None:
        return jsonify({"error": "Processing failed!"}), 500

    # Format the DataFrame for display
    display_df = df.copy()
    for col in ['mAP', 'Precision', 'Recall', 'Small Object Score']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%")
    for col in ['FPS', 'Inference Time (ms)', 'Model Size (MB)', 'Number of Parameters (M)']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")

    best_algo = df.loc[df['mAP'].idxmax()]['Algorithm'] if not df.empty else "None"
    algorithm_explanations = generate_algorithm_explanations(df, best_algo)
    metric_explanations = {
        "mAP": "Mean Average Precision (%): Detection accuracy across IoU thresholds (higher is better).",
        "FPS": "Frames Per Second: Speed of detection (higher is better).",
        "Small Object Score": "Small objects detected (%): Higher means better small object detection.",
        "Model Size (MB)": "Size of model file (smaller is better for deployment).",
        "Precision": "Correct detections out of all detections (%): Higher is better.",
        "Recall": "Correct detections out of all ground truths (%): Higher is better.",
        "Inference Time (ms)": "Time per inference (lower is best).",
        "Number of Parameters (M)": "Model complexity (fewer is better for efficiency)."
    }
    selection_reason = (
        "Why These Algorithms?\n"
        "We selected R-CNN, YOLOv5, YOLOv7, and YOLOv9 for drone detection research:\n"
        "- R-CNN: High accuracy two-stage detector, great for small objects like drones.\n"
        "- YOLOv5: Balanced speed and accuracy, lightweight for real-time use.\n"
        "- YOLOv7: Improved small object detection and efficiency.\n"
        "- YOLOv9: State-of-the-art with top mAP and FPS.\n"
        "These cover accuracy, speed, and deployment needs!"
    )

    tab_results = calculate_weighted_scores(df)

    return jsonify({
        "table": display_df.to_dict(orient="records"),
        "best_algo": best_algo,
        "metric_explanations": metric_explanations,
        "algorithm_explanations": algorithm_explanations,
        "selection_reason": selection_reason,
        "labeled_images": labeled_images,
        "num_images_processed": num_processed,
        "tab_results": tab_results
    })

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)