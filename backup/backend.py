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

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}}, supports_credentials=True)

WEIGHTS_PATH = "C:/Users/M SHARATH CHANDRA/Desktop/Drone-Detection-Project/"
IMAGES_PATH = "C:/Users/M SHARATH CHANDRA/Desktop/Drone-Detection-Project/drone_dataset/valid/images"
LABELS_PATH = "C:/Users/M SHARATH CHANDRA/Desktop/Drone-Detection-Project/drone_dataset/valid/labels"

# Models load chestham ra
try:
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = os.path.join(WEIGHTS_PATH, "model_final_280758.pkl")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    predictor = DefaultPredictor(cfg)
except Exception as e:
    print(f"R-CNN load fail ra: {e}")

try:
    model_v5 = YOLO(os.path.join(WEIGHTS_PATH, "yolov5su.pt"))
except Exception as e:
    print(f"YOLOv5 load fail ra: {e}")

try:
    model_v7 = torch.hub.load('WongKinYiu/yolov7', 'custom', os.path.join(WEIGHTS_PATH, "yolov7.pt"))
except Exception as e:
    print(f"YOLOv7 load fail ra: {e}")

try:
    model_v9 = YOLO(os.path.join(WEIGHTS_PATH, "yolov9c.pt"))
except Exception as e:
    print(f"YOLOv9 load fail ra: {e}")

# Metric calculation functions ra
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
        print(f"FPS calc fail ra: {e}")
        return 0, 0

def small_object_score(detections, img, threshold=0.05):
    if not detections:
        return 0
    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in detections]
    small_objs = sum(1 for area in areas if area < threshold * img.shape[0] * img.shape[1])
    return small_objs / len(areas) if areas else 0

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
    iou = inter_area / union_area if union_area > 0 else 0
    print(f"IoU between {box1} and {box2}: {iou}")
    return iou

def calc_map(pred_boxes, gt_boxes, iou_threshold=0.3):
    if not gt_boxes or not pred_boxes:
        print("No ground truth or predictions ra, mAP = 0")
        return 0.0
    
    pred_boxes = sorted(pred_boxes, key=lambda x: x[-1] if len(x) > 4 else 1.0, reverse=True)
    tp = [0] * len(pred_boxes)
    fp = [0] * len(pred_boxes)
    gt_matched = set()
    
    for i, pred in enumerate(pred_boxes):
        best_iou = 0
        best_gt_idx = -1
        for j, gt in enumerate(gt_boxes):
            if j in gt_matched:
                continue
            iou = calc_iou(pred[:4], gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        print(f"Best IoU for pred {pred[:4]}: {best_iou}")
        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp[i] = 1
            gt_matched.add(best_gt_idx)
        else:
            fp[i] = 1
    
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    recalls = tp_cumsum / (len(gt_boxes) + 1e-6)
    precisions = np.concatenate(([1.0], precisions, [0.0]))
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    ap = np.trapz(precisions, recalls)
    print(f"TP: {tp}, FP: {fp}, Precisions: {precisions}, Recalls: {recalls}, mAP: {ap}")
    return ap

def calc_precision_recall(detections, gt_boxes):
    if not detections or not gt_boxes:
        return 0, 0
    tp = sum(1 for det in detections for gt in gt_boxes if calc_iou(det[:4], gt) > 0.3)
    precision = tp / len(detections) if detections else 0
    recall = tp / len(gt_boxes) if gt_boxes else 0
    return precision, recall

def load_ground_truth(image_name):
    label_file = os.path.join(LABELS_PATH, image_name.replace('.jpg', '.txt').replace('.png', '.txt'))
    gt_boxes = []
    if os.path.exists(label_file):
        img = cv2.imread(os.path.join(IMAGES_PATH, image_name))
        if img is None:
            print(f"Image not found ra: {image_name}")
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
        print(f"Ground truth boxes for {image_name} ({img_w}x{img_h}): {gt_boxes}")
    else:
        print(f"Label file not found ra: {label_file}")
    return gt_boxes

def draw_labels(img, detections, algo_name):
    for box in detections:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{algo_name}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img

def generate_algorithm_explanations(df, best_algo):
    explanations = {}
    best_row = df[df['Algorithm'] == best_algo].iloc[0]
    best_map = best_row['mAP']
    
    for index, row in df.iterrows():
        algo = row['Algorithm']
        explanation = ""
        if algo == best_algo:
            explanation = (
                f"Why {algo} is Best Ra?\n"
                f"- Highest mAP ({best_map:.2f}): Most accurate drone detection.\n"
                f"- FPS ({row['FPS']:.2f}): Fast enough for real-time use.\n"
                f"- Small Object Score ({row['Small Object Score']:.2f}): Great at detecting small drones.\n"
                f"- Inference Time ({row['Inference Time (ms)']:.2f}ms): Quick processing.\n"
                f"Overall, {algo} balances accuracy, speed, and efficiency ra!"
            )
        else:
            reasons = []
            if row['mAP'] < best_map:
                reasons.append(f"Lower mAP ({row['mAP']:.2f} vs {best_map:.2f}): Less accurate than {best_algo}.")
            if row['FPS'] < best_row['FPS']:
                reasons.append(f"Lower FPS ({row['FPS']:.2f} vs {best_row['FPS']:.2f}): Slower detection speed.")
            if row['Small Object Score'] < best_row['Small Object Score']:
                reasons.append(f"Lower Small Object Score ({row['Small Object Score']:.2f} vs {best_row['Small Object Score']:.2f}): Weaker small drone detection.")
            if row['Inference Time (ms)'] > best_row['Inference Time (ms)']:
                reasons.append(f"Higher Inference Time ({row['Inference Time (ms)']:.2f}ms vs {best_row['Inference Time (ms)']:.2f}ms): Slower processing.")
            explanation = (
                f"Why {algo} is Not Best Ra?\n"
                f"- {' '.join(reasons)}\n"
                f"Compared to {best_algo}, {algo} falls short in these key areas ra!"
            )
        explanations[algo] = explanation
    return explanations

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(IMAGES_PATH, filename)

@app.route('/images', methods=['GET'])
def get_images():
    try:
        image_files = [f"http://localhost:5000/images/{f}" for f in os.listdir(IMAGES_PATH) if f.endswith(('.jpg', '.png'))]
        return jsonify({"images": image_files})
    except Exception as e:
        return jsonify({"error": f"Image fetch fail ra: {e}"}), 500

@app.route('/detect', methods=['POST'])
def detect():
    results = {
        "Algorithm": [], "mAP": [], "FPS": [], "Small Object Score": [], "Model Size (MB)": [],
        "Precision": [], "Recall": [], "Inference Time (ms)": [], "Number of Parameters (M)": []
    }
    metric_explanations = {
        "mAP": "Mean Average Precision: Measures detection accuracy across IoU thresholds.",
        "FPS": "Frames Per Second: Speed of detection (higher is better).",
        "Small Object Score": "Ratio of small objects detected (higher means better small object detection).",
        "Model Size (MB)": "Size of model file (smaller is better for deployment).",
        "Precision": "Correct detections out of all detections (higher is better).",
        "Recall": "Correct detections out of all ground truths (higher is better).",
        "Inference Time (ms)": "Time per inference (lower is better).",
        "Number of Parameters (M)": "Model complexity (fewer is better for efficiency)."
    }
    
    if 'image' not in request.files:
        return jsonify({"error": "No image file ra!"}), 400
    
    file = request.files['image']
    image_name = file.filename
    img_data = file.read()
    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
    
    if img is None:
        return jsonify({"error": "Invalid image ra!"}), 400

    gt_boxes = load_ground_truth(image_name)
    if not gt_boxes:
        print(f"No ground truth boxes loaded for {image_name} ra!")

    # R-CNN
    try:
        outputs_rcnn = predictor(img)
        pred_boxes_rcnn = outputs_rcnn["instances"].pred_boxes.tensor.cpu().numpy().tolist()
        print(f"R-CNN predictions: {pred_boxes_rcnn}")
        fps, inf_time = calc_fps_and_time(predictor, img)
        mAP = calc_map(pred_boxes_rcnn, gt_boxes)
        small_score = small_object_score(pred_boxes_rcnn, img)
        model_size = os.path.getsize(cfg.MODEL.WEIGHTS) / (1024 * 1024)
        precision, recall = calc_precision_recall(pred_boxes_rcnn, gt_boxes)
        num_params = sum(p.numel() for p in predictor.model.parameters()) / 1e6
        img_rcnn = draw_labels(img.copy(), pred_boxes_rcnn, "R-CNN")
        results["Algorithm"].append("R-CNN")
        results["mAP"].append(mAP)
        results["FPS"].append(fps)
        results["Small Object Score"].append(small_score)
        results["Model Size (MB)"].append(model_size)
        results["Precision"].append(precision)
        results["Recall"].append(recall)
        results["Inference Time (ms)"].append(inf_time)
        results["Number of Parameters (M)"].append(num_params)
    except Exception as e:
        print(f"R-CNN detection fail ra: {e}")
        results["Algorithm"].append("R-CNN")
        results["mAP"].append(0.0)
        results["FPS"].append(0)
        results["Small Object Score"].append(0)
        results["Model Size (MB)"].append(0)
        results["Precision"].append(0)
        results["Recall"].append(0)
        results["Inference Time (ms)"].append(0)
        results["Number of Parameters (M)"].append(0)

    # YOLOv5
    try:
        results_v5 = model_v5(img)
        pred_boxes_v5 = results_v5[0].boxes.xyxy.cpu().numpy().tolist()
        print(f"YOLOv5 predictions: {pred_boxes_v5}")
        fps, inf_time = calc_fps_and_time(model_v5, img)
        mAP = calc_map(pred_boxes_v5, gt_boxes)
        small_score = small_object_score(pred_boxes_v5, img)
        model_size = os.path.getsize(os.path.join(WEIGHTS_PATH, "yolov5su.pt")) / (1024 * 1024)
        precision, recall = calc_precision_recall(pred_boxes_v5, gt_boxes)
        num_params = sum(p.numel() for p in model_v5.model.parameters()) / 1e6
        img_v5 = draw_labels(img.copy(), pred_boxes_v5, "YOLOv5")
        results["Algorithm"].append("YOLOv5")
        results["mAP"].append(mAP)
        results["FPS"].append(fps)
        results["Small Object Score"].append(small_score)
        results["Model Size (MB)"].append(model_size)
        results["Precision"].append(precision)
        results["Recall"].append(recall)
        results["Inference Time (ms)"].append(inf_time)
        results["Number of Parameters (M)"].append(num_params)
    except Exception as e:
        print(f"YOLOv5 detection fail ra: {e}")
        results["Algorithm"].append("YOLOv5")
        results["mAP"].append(0.0)
        results["FPS"].append(0)
        results["Small Object Score"].append(0)
        results["Model Size (MB)"].append(0)
        results["Precision"].append(0)
        results["Recall"].append(0)
        results["Inference Time (ms)"].append(0)
        results["Number of Parameters (M)"].append(0)

    # YOLOv7
    try:
        detections_v7 = model_v7(img)
        pred_boxes_v7 = detections_v7.xyxy[0].cpu().numpy().tolist()
        print(f"YOLOv7 predictions: {pred_boxes_v7}")
        fps, inf_time = calc_fps_and_time(model_v7, img)
        mAP = calc_map(pred_boxes_v7, gt_boxes)
        small_score = small_object_score(pred_boxes_v7, img)
        model_size = os.path.getsize(os.path.join(WEIGHTS_PATH, "yolov7.pt")) / (1024 * 1024)
        precision, recall = calc_precision_recall(pred_boxes_v7, gt_boxes)
        num_params = sum(p.numel() for p in model_v7.parameters()) / 1e6
        img_v7 = draw_labels(img.copy(), pred_boxes_v7, "YOLOv7")
        results["Algorithm"].append("YOLOv7")
        results["mAP"].append(mAP)
        results["FPS"].append(fps)
        results["Small Object Score"].append(small_score)
        results["Model Size (MB)"].append(model_size)
        results["Precision"].append(precision)
        results["Recall"].append(recall)
        results["Inference Time (ms)"].append(inf_time)
        results["Number of Parameters (M)"].append(num_params)
    except Exception as e:
        print(f"YOLOv7 detection fail ra: {e}")
        results["Algorithm"].append("YOLOv7")
        results["mAP"].append(0.0)
        results["FPS"].append(0)
        results["Small Object Score"].append(0)
        results["Model Size (MB)"].append(0)
        results["Precision"].append(0)
        results["Recall"].append(0)
        results["Inference Time (ms)"].append(0)
        results["Number of Parameters (M)"].append(0)

    # YOLOv9
    try:
        results_v9 = model_v9(img)
        pred_boxes_v9 = results_v9[0].boxes.xyxy.cpu().numpy().tolist()
        print(f"YOLOv9 predictions: {pred_boxes_v9}")
        fps, inf_time = calc_fps_and_time(model_v9, img)
        mAP = calc_map(pred_boxes_v9, gt_boxes)
        small_score = small_object_score(pred_boxes_v9, img)
        model_size = os.path.getsize(os.path.join(WEIGHTS_PATH, "yolov9c.pt")) / (1024 * 1024)
        precision, recall = calc_precision_recall(pred_boxes_v9, gt_boxes)
        num_params = sum(p.numel() for p in model_v9.model.parameters()) / 1e6
        img_v9 = draw_labels(img.copy(), pred_boxes_v9, "YOLOv9")
        results["Algorithm"].append("YOLOv9")
        results["mAP"].append(mAP)
        results["FPS"].append(fps)
        results["Small Object Score"].append(small_score)
        results["Model Size (MB)"].append(model_size)
        results["Precision"].append(precision)
        results["Recall"].append(recall)
        results["Inference Time (ms)"].append(inf_time)
        results["Number of Parameters (M)"].append(num_params)
    except Exception as e:
        print(f"YOLOv9 detection fail ra: {e}")
        results["Algorithm"].append("YOLOv9")
        results["mAP"].append(0.0)
        results["FPS"].append(0)
        results["Small Object Score"].append(0)
        results["Model Size (MB)"].append(0)
        results["Precision"].append(0)
        results["Recall"].append(0)
        results["Inference Time (ms)"].append(0)
        results["Number of Parameters (M)"].append(0)

    # Save images ra
    static_dir = "static"
    os.makedirs(static_dir, exist_ok=True)
    cv2.imwrite(os.path.join(static_dir, "rcnn.jpg"), img_rcnn)
    cv2.imwrite(os.path.join(static_dir, "yolov5.jpg"), img_v5)
    cv2.imwrite(os.path.join(static_dir, "yolov7.jpg"), img_v7)
    cv2.imwrite(os.path.join(static_dir, "yolov9.jpg"), img_v9)

    df = pd.DataFrame(results)
    best_algo = df.loc[df['mAP'].idxmax()]['Algorithm'] if not df.empty else "None"
    algorithm_explanations = generate_algorithm_explanations(df, best_algo)
    
    selection_reason = (
        "Why These Algorithms Ra?\n"
        "We selected R-CNN, YOLOv5, YOLOv7, and YOLOv9 for drone detection research ra:\n"
        "- R-CNN: High accuracy two-stage detector, great for small objects like drones, used as a baseline.\n"
        "- YOLOv5: Balanced speed and accuracy, lightweight and customizable for real-time use.\n"
        "- YOLOv7: Latest YOLO with improved small object detection and efficiency.\n"
        "- YOLOv9: State-of-the-art model with top mAP and FPS, future-ready for drones.\n"
        "These cover accuracy, speed, and deployment needs ra, allowing us to find the best fit!"
    )

    return jsonify({
        "images": {
            "rcnn": "http://localhost:5000/static/rcnn.jpg",
            "yolov5": "http://localhost:5000/static/yolov5.jpg",
            "yolov7": "http://localhost:5000/static/yolov7.jpg",
            "yolov9": "http://localhost:5000/static/yolov9.jpg"
        },
        "table": df.to_dict(orient="records"),
        "best_algo": best_algo,
        "metric_explanations": metric_explanations,
        "algorithm_explanations": algorithm_explanations,
        "selection_reason": selection_reason
    })

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
