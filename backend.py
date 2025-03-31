# from flask import Flask, request, jsonify, send_from_directory
# from flask_cors import CORS
# import os
# import cv2
# import time
# import torch
# import numpy as np
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2 import model_zoo
# from ultralytics import YOLO
# import pandas as pd

# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}}, supports_credentials=True)

# WEIGHTS_PATH = "C:/Users/M SHARATH CHANDRA/Desktop/Drone-Detection-Project/"
# IMAGES_PATH = "C:/Users/M SHARATH CHANDRA/Desktop/Drone-Detection-Project/drone_dataset/valid/images"
# LABELS_PATH = "C:/Users/M SHARATH CHANDRA/Desktop/Drone-Detection-Project/drone_dataset/valid/labels"

# # Models load chestham ra
# try:
#     cfg = get_cfg()
#     cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
#     cfg.MODEL.WEIGHTS = os.path.join(WEIGHTS_PATH, "model_final_280758.pkl")
#     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
#     predictor = DefaultPredictor(cfg)
# except Exception as e:
#     print(f"R-CNN load fail ra: {e}")

# try:
#     model_v5 = YOLO(os.path.join(WEIGHTS_PATH, "yolov5su.pt"))
# except Exception as e:
#     print(f"YOLOv5 load fail ra: {e}")

# try:
#     model_v7 = torch.hub.load('WongKinYiu/yolov7', 'custom', os.path.join(WEIGHTS_PATH, "yolov7.pt"))
# except Exception as e:
#     print(f"YOLOv7 load fail ra: {e}")

# try:
#     model_v9 = YOLO(os.path.join(WEIGHTS_PATH, "yolov9c.pt"))
# except Exception as e:
#     print(f"YOLOv9 load fail ra: {e}")

# # Metric calculation functions ra
# def calc_fps_and_time(model, img, iterations=10):
#     try:
#         start = time.time()
#         for _ in range(iterations):
#             model(img)
#         end = time.time()
#         fps = iterations / (end - start)
#         inference_time = (end - start) * 1000 / iterations
#         return fps, inference_time
#     except Exception as e:
#         print(f"FPS calc fail ra: {e}")
#         return 0, 0

# def small_object_score(detections, img, threshold=0.05):
#     if not detections:
#         return 0
#     areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in detections]
#     small_objs = sum(1 for area in areas if area < threshold * img.shape[0] * img.shape[1])
#     return small_objs / len(areas) if areas else 0

# def calc_iou(box1, box2):
#     x1, y1, x2, y2 = box1
#     x1_gt, y1_gt, x2_gt, y2_gt = box2
#     xi1 = max(x1, x1_gt)
#     yi1 = max(y1, y1_gt)
#     xi2 = min(x2, x2_gt)
#     yi2 = min(y2, y2_gt)
#     inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
#     box1_area = (x2 - x1) * (y2 - y1)
#     box2_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
#     union_area = box1_area + box2_area - inter_area
#     iou = inter_area / union_area if union_area > 0 else 0
#     print(f"IoU between {box1} and {box2}: {iou}")
#     return iou

# def calc_map(pred_boxes, gt_boxes, iou_threshold=0.3):
#     if not gt_boxes or not pred_boxes:
#         print("No ground truth or predictions ra, mAP = 0")
#         return 0.0
    
#     pred_boxes = sorted(pred_boxes, key=lambda x: x[-1] if len(x) > 4 else 1.0, reverse=True)
#     tp = [0] * len(pred_boxes)
#     fp = [0] * len(pred_boxes)
#     gt_matched = set()
    
#     for i, pred in enumerate(pred_boxes):
#         best_iou = 0
#         best_gt_idx = -1
#         for j, gt in enumerate(gt_boxes):
#             if j in gt_matched:
#                 continue
#             iou = calc_iou(pred[:4], gt)
#             if iou > best_iou:
#                 best_iou = iou
#                 best_gt_idx = j
#         print(f"Best IoU for pred {pred[:4]}: {best_iou}")
#         if best_iou >= iou_threshold and best_gt_idx != -1:
#             tp[i] = 1
#             gt_matched.add(best_gt_idx)
#         else:
#             fp[i] = 1
    
#     tp_cumsum = np.cumsum(tp)
#     fp_cumsum = np.cumsum(fp)
#     precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
#     recalls = tp_cumsum / (len(gt_boxes) + 1e-6)
#     precisions = np.concatenate(([1.0], precisions, [0.0]))
#     recalls = np.concatenate(([0.0], recalls, [1.0]))
#     ap = np.trapz(precisions, recalls)
#     print(f"TP: {tp}, FP: {fp}, Precisions: {precisions}, Recalls: {recalls}, mAP: {ap}")
#     return ap

# def calc_precision_recall(detections, gt_boxes):
#     if not detections or not gt_boxes:
#         return 0, 0
#     tp = sum(1 for det in detections for gt in gt_boxes if calc_iou(det[:4], gt) > 0.3)
#     precision = tp / len(detections) if detections else 0
#     recall = tp / len(gt_boxes) if gt_boxes else 0
#     return precision, recall

# def load_ground_truth(image_name):
#     label_file = os.path.join(LABELS_PATH, image_name.replace('.jpg', '.txt').replace('.png', '.txt'))
#     gt_boxes = []
#     if os.path.exists(label_file):
#         img = cv2.imread(os.path.join(IMAGES_PATH, image_name))
#         if img is None:
#             print(f"Image not found ra: {image_name}")
#             return gt_boxes
#         img_h, img_w = img.shape[:2]
#         with open(label_file, 'r') as f:
#             for line in f:
#                 parts = line.strip().split()
#                 if len(parts) < 5:
#                     continue
#                 class_id, x_center, y_center, width, height = map(float, parts[:5])
#                 x1 = (x_center - width / 2) * img_w
#                 y1 = (y_center - height / 2) * img_h
#                 x2 = (x_center + width / 2) * img_w
#                 y2 = (y_center + height / 2) * img_h
#                 gt_boxes.append([x1, y1, x2, y2])
#         print(f"Ground truth boxes for {image_name} ({img_w}x{img_h}): {gt_boxes}")
#     else:
#         print(f"Label file not found ra: {label_file}")
#     return gt_boxes

# def draw_labels(img, detections, algo_name):
#     for box in detections:
#         x1, y1, x2, y2 = map(int, box[:4])
#         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(img, f"{algo_name}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#     return img

# def generate_algorithm_explanations(df, best_algo):
#     explanations = {}
#     best_row = df[df['Algorithm'] == best_algo].iloc[0]
#     best_map = best_row['mAP']
    
#     for index, row in df.iterrows():
#         algo = row['Algorithm']
#         explanation = ""
#         if algo == best_algo:
#             explanation = (
#                 f"Why {algo} is Best Ra?\n"
#                 f"- Highest mAP ({best_map:.2f}): Most accurate drone detection.\n"
#                 f"- FPS ({row['FPS']:.2f}): Fast enough for real-time use.\n"
#                 f"- Small Object Score ({row['Small Object Score']:.2f}): Great at detecting small drones.\n"
#                 f"- Inference Time ({row['Inference Time (ms)']:.2f}ms): Quick processing.\n"
#                 f"Overall, {algo} balances accuracy, speed, and efficiency ra!"
#             )
#         else:
#             reasons = []
#             if row['mAP'] < best_map:
#                 reasons.append(f"Lower mAP ({row['mAP']:.2f} vs {best_map:.2f}): Less accurate than {best_algo}.")
#             if row['FPS'] < best_row['FPS']:
#                 reasons.append(f"Lower FPS ({row['FPS']:.2f} vs {best_row['FPS']:.2f}): Slower detection speed.")
#             if row['Small Object Score'] < best_row['Small Object Score']:
#                 reasons.append(f"Lower Small Object Score ({row['Small Object Score']:.2f} vs {best_row['Small Object Score']:.2f}): Weaker small drone detection.")
#             if row['Inference Time (ms)'] > best_row['Inference Time (ms)']:
#                 reasons.append(f"Higher Inference Time ({row['Inference Time (ms)']:.2f}ms vs {best_row['Inference Time (ms)']:.2f}ms): Slower processing.")
#             explanation = (
#                 f"Why {algo} is Not Best Ra?\n"
#                 f"- {' '.join(reasons)}\n"
#                 f"Compared to {best_algo}, {algo} falls short in these key areas ra!"
#             )
#         explanations[algo] = explanation
#     return explanations

# @app.route('/images/<path:filename>')
# def serve_image(filename):
#     return send_from_directory(IMAGES_PATH, filename)

# @app.route('/images', methods=['GET'])
# def get_images():
#     try:
#         image_files = [f"http://localhost:5000/images/{f}" for f in os.listdir(IMAGES_PATH) if f.endswith(('.jpg', '.png'))]
#         return jsonify({"images": image_files})
#     except Exception as e:
#         return jsonify({"error": f"Image fetch fail ra: {e}"}), 500

# @app.route('/detect', methods=['POST'])
# def detect():
#     results = {
#         "Algorithm": [], "mAP": [], "FPS": [], "Small Object Score": [], "Model Size (MB)": [],
#         "Precision": [], "Recall": [], "Inference Time (ms)": [], "Number of Parameters (M)": []
#     }
#     metric_explanations = {
#         "mAP": "Mean Average Precision: Measures detection accuracy across IoU thresholds.",
#         "FPS": "Frames Per Second: Speed of detection (higher is better).",
#         "Small Object Score": "Ratio of small objects detected (higher means better small object detection).",
#         "Model Size (MB)": "Size of model file (smaller is better for deployment).",
#         "Precision": "Correct detections out of all detections (higher is better).",
#         "Recall": "Correct detections out of all ground truths (higher is better).",
#         "Inference Time (ms)": "Time per inference (lower is better).",
#         "Number of Parameters (M)": "Model complexity (fewer is better for efficiency)."
#     }
    
#     if 'image' not in request.files:
#         return jsonify({"error": "No image file ra!"}), 400
    
#     file = request.files['image']
#     image_name = file.filename
#     img_data = file.read()
#     img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
    
#     if img is None:
#         return jsonify({"error": "Invalid image ra!"}), 400

#     gt_boxes = load_ground_truth(image_name)
#     if not gt_boxes:
#         print(f"No ground truth boxes loaded for {image_name} ra!")

#     # R-CNN
#     try:
#         outputs_rcnn = predictor(img)
#         pred_boxes_rcnn = outputs_rcnn["instances"].pred_boxes.tensor.cpu().numpy().tolist()
#         print(f"R-CNN predictions: {pred_boxes_rcnn}")
#         fps, inf_time = calc_fps_and_time(predictor, img)
#         mAP = calc_map(pred_boxes_rcnn, gt_boxes)
#         small_score = small_object_score(pred_boxes_rcnn, img)
#         model_size = os.path.getsize(cfg.MODEL.WEIGHTS) / (1024 * 1024)
#         precision, recall = calc_precision_recall(pred_boxes_rcnn, gt_boxes)
#         num_params = sum(p.numel() for p in predictor.model.parameters()) / 1e6
#         img_rcnn = draw_labels(img.copy(), pred_boxes_rcnn, "R-CNN")
#         results["Algorithm"].append("R-CNN")
#         results["mAP"].append(mAP)
#         results["FPS"].append(fps)
#         results["Small Object Score"].append(small_score)
#         results["Model Size (MB)"].append(model_size)
#         results["Precision"].append(precision)
#         results["Recall"].append(recall)
#         results["Inference Time (ms)"].append(inf_time)
#         results["Number of Parameters (M)"].append(num_params)
#     except Exception as e:
#         print(f"R-CNN detection fail ra: {e}")
#         results["Algorithm"].append("R-CNN")
#         results["mAP"].append(0.0)
#         results["FPS"].append(0)
#         results["Small Object Score"].append(0)
#         results["Model Size (MB)"].append(0)
#         results["Precision"].append(0)
#         results["Recall"].append(0)
#         results["Inference Time (ms)"].append(0)
#         results["Number of Parameters (M)"].append(0)

#     # YOLOv5
#     try:
#         results_v5 = model_v5(img)
#         pred_boxes_v5 = results_v5[0].boxes.xyxy.cpu().numpy().tolist()
#         print(f"YOLOv5 predictions: {pred_boxes_v5}")
#         fps, inf_time = calc_fps_and_time(model_v5, img)
#         mAP = calc_map(pred_boxes_v5, gt_boxes)
#         small_score = small_object_score(pred_boxes_v5, img)
#         model_size = os.path.getsize(os.path.join(WEIGHTS_PATH, "yolov5su.pt")) / (1024 * 1024)
#         precision, recall = calc_precision_recall(pred_boxes_v5, gt_boxes)
#         num_params = sum(p.numel() for p in model_v5.model.parameters()) / 1e6
#         img_v5 = draw_labels(img.copy(), pred_boxes_v5, "YOLOv5")
#         results["Algorithm"].append("YOLOv5")
#         results["mAP"].append(mAP)
#         results["FPS"].append(fps)
#         results["Small Object Score"].append(small_score)
#         results["Model Size (MB)"].append(model_size)
#         results["Precision"].append(precision)
#         results["Recall"].append(recall)
#         results["Inference Time (ms)"].append(inf_time)
#         results["Number of Parameters (M)"].append(num_params)
#     except Exception as e:
#         print(f"YOLOv5 detection fail ra: {e}")
#         results["Algorithm"].append("YOLOv5")
#         results["mAP"].append(0.0)
#         results["FPS"].append(0)
#         results["Small Object Score"].append(0)
#         results["Model Size (MB)"].append(0)
#         results["Precision"].append(0)
#         results["Recall"].append(0)
#         results["Inference Time (ms)"].append(0)
#         results["Number of Parameters (M)"].append(0)

#     # YOLOv7
#     try:
#         detections_v7 = model_v7(img)
#         pred_boxes_v7 = detections_v7.xyxy[0].cpu().numpy().tolist()
#         print(f"YOLOv7 predictions: {pred_boxes_v7}")
#         fps, inf_time = calc_fps_and_time(model_v7, img)
#         mAP = calc_map(pred_boxes_v7, gt_boxes)
#         small_score = small_object_score(pred_boxes_v7, img)
#         model_size = os.path.getsize(os.path.join(WEIGHTS_PATH, "yolov7.pt")) / (1024 * 1024)
#         precision, recall = calc_precision_recall(pred_boxes_v7, gt_boxes)
#         num_params = sum(p.numel() for p in model_v7.parameters()) / 1e6
#         img_v7 = draw_labels(img.copy(), pred_boxes_v7, "YOLOv7")
#         results["Algorithm"].append("YOLOv7")
#         results["mAP"].append(mAP)
#         results["FPS"].append(fps)
#         results["Small Object Score"].append(small_score)
#         results["Model Size (MB)"].append(model_size)
#         results["Precision"].append(precision)
#         results["Recall"].append(recall)
#         results["Inference Time (ms)"].append(inf_time)
#         results["Number of Parameters (M)"].append(num_params)
#     except Exception as e:
#         print(f"YOLOv7 detection fail ra: {e}")
#         results["Algorithm"].append("YOLOv7")
#         results["mAP"].append(0.0)
#         results["FPS"].append(0)
#         results["Small Object Score"].append(0)
#         results["Model Size (MB)"].append(0)
#         results["Precision"].append(0)
#         results["Recall"].append(0)
#         results["Inference Time (ms)"].append(0)
#         results["Number of Parameters (M)"].append(0)

#     # YOLOv9
#     try:
#         results_v9 = model_v9(img)
#         pred_boxes_v9 = results_v9[0].boxes.xyxy.cpu().numpy().tolist()
#         print(f"YOLOv9 predictions: {pred_boxes_v9}")
#         fps, inf_time = calc_fps_and_time(model_v9, img)
#         mAP = calc_map(pred_boxes_v9, gt_boxes)
#         small_score = small_object_score(pred_boxes_v9, img)
#         model_size = os.path.getsize(os.path.join(WEIGHTS_PATH, "yolov9c.pt")) / (1024 * 1024)
#         precision, recall = calc_precision_recall(pred_boxes_v9, gt_boxes)
#         num_params = sum(p.numel() for p in model_v9.model.parameters()) / 1e6
#         img_v9 = draw_labels(img.copy(), pred_boxes_v9, "YOLOv9")
#         results["Algorithm"].append("YOLOv9")
#         results["mAP"].append(mAP)
#         results["FPS"].append(fps)
#         results["Small Object Score"].append(small_score)
#         results["Model Size (MB)"].append(model_size)
#         results["Precision"].append(precision)
#         results["Recall"].append(recall)
#         results["Inference Time (ms)"].append(inf_time)
#         results["Number of Parameters (M)"].append(num_params)
#     except Exception as e:
#         print(f"YOLOv9 detection fail ra: {e}")
#         results["Algorithm"].append("YOLOv9")
#         results["mAP"].append(0.0)
#         results["FPS"].append(0)
#         results["Small Object Score"].append(0)
#         results["Model Size (MB)"].append(0)
#         results["Precision"].append(0)
#         results["Recall"].append(0)
#         results["Inference Time (ms)"].append(0)
#         results["Number of Parameters (M)"].append(0)

#     # Save images ra
#     static_dir = "static"
#     os.makedirs(static_dir, exist_ok=True)
#     cv2.imwrite(os.path.join(static_dir, "rcnn.jpg"), img_rcnn)
#     cv2.imwrite(os.path.join(static_dir, "yolov5.jpg"), img_v5)
#     cv2.imwrite(os.path.join(static_dir, "yolov7.jpg"), img_v7)
#     cv2.imwrite(os.path.join(static_dir, "yolov9.jpg"), img_v9)

#     df = pd.DataFrame(results)
#     best_algo = df.loc[df['mAP'].idxmax()]['Algorithm'] if not df.empty else "None"
#     algorithm_explanations = generate_algorithm_explanations(df, best_algo)
    
#     selection_reason = (
#         "Why These Algorithms Ra?\n"
#         "We selected R-CNN, YOLOv5, YOLOv7, and YOLOv9 for drone detection research ra:\n"
#         "- R-CNN: High accuracy two-stage detector, great for small objects like drones, used as a baseline.\n"
#         "- YOLOv5: Balanced speed and accuracy, lightweight and customizable for real-time use.\n"
#         "- YOLOv7: Latest YOLO with improved small object detection and efficiency.\n"
#         "- YOLOv9: State-of-the-art model with top mAP and FPS, future-ready for drones.\n"
#         "These cover accuracy, speed, and deployment needs ra, allowing us to find the best fit!"
#     )

#     return jsonify({
#         "images": {
#             "rcnn": "http://localhost:5000/static/rcnn.jpg",
#             "yolov5": "http://localhost:5000/static/yolov5.jpg",
#             "yolov7": "http://localhost:5000/static/yolov7.jpg",
#             "yolov9": "http://localhost:5000/static/yolov9.jpg"
#         },
#         "table": df.to_dict(orient="records"),
#         "best_algo": best_algo,
#         "metric_explanations": metric_explanations,
#         "algorithm_explanations": algorithm_explanations,
#         "selection_reason": selection_reason
#     })

# if __name__ == "__main__":
#     app.run(debug=True, host='0.0.0.0', port=5000)




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

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = 'http://localhost:3000'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    return response

WEIGHTS_PATH = "C:/Users/M SHARATH CHANDRA/Desktop/Drone-Detection-Project/"
IMAGES_PATH = "C:/Users/M SHARATH CHANDRA/Desktop/Drone-Detection-Project/drone_dataset/valid/images"
LABELS_PATH = "C:/Users/M SHARATH CHANDRA/Desktop/Drone-Detection-Project/drone_dataset/valid/labels"

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

    for index, row in df.iterrows():
        algo = row['Algorithm']
        explanation = []

        # Get the rows for all other algorithms for comparison
        other_rows = df[df['Algorithm'] != algo]

        if algo == best_algo:
            explanation.append(f"Why {algo} is the Best:")
            explanation.append(f"- mAP ({row['mAP']}%): Highest accuracy in drone detection.")
            explanation.append(f"- FPS ({row['FPS']:.2f}): Best speed for real-time detection.")
            explanation.append(f"- Inference Time ({row['Inference Time (ms)']:.2f}ms): Fastest processing per image.")
            explanation.append(f"- Model Size ({row['Model Size (MB)']:.2f}MB): Most efficient for deployment.")
            explanation.append(f"- Number of Parameters ({row['Number of Parameters (M)']:.2f}M): Least complex model.")
            explanation.append(f"- Precision ({row['Precision']}%): Best at minimizing false positives.")
            explanation.append(f"- Recall ({row['Recall']}%): Best at detecting all drones.")
            explanation.append(f"- Small Object Score ({row['Small Object Score']}%): Best at detecting small drones.")

            explanation.append("\nComparisons with Other Algorithms:")
            for _, other_row in other_rows.iterrows():
                other_algo = other_row['Algorithm']
                comparisons = []
                if float(row['mAP'].replace('%', '')) > float(other_row['mAP'].replace('%', '')):
                    comparisons.append(f"Higher mAP ({row['mAP']} vs {other_row['mAP']}) than {other_algo}.")
                if row['FPS'] > other_row['FPS']:
                    comparisons.append(f"Higher FPS ({row['FPS']:.2f} vs {other_row['FPS']:.2f}) than {other_algo}.")
                if row['Inference Time (ms)'] < other_row['Inference Time (ms)']:
                    comparisons.append(f"Lower Inference Time ({row['Inference Time (ms)']:.2f}ms vs {other_row['Inference Time (ms)']:.2f}ms) than {other_algo}.")
                if row['Model Size (MB)'] < other_row['Model Size (MB)']:
                    comparisons.append(f"Smaller Model Size ({row['Model Size (MB)']:.2f}MB vs {other_row['Model Size (MB)']:.2f}MB) than {other_algo}.")
                if row['Number of Parameters (M)'] < other_row['Number of Parameters (M)']:
                    comparisons.append(f"Fewer Parameters ({row['Number of Parameters (M)']:.2f}M vs {other_row['Number of Parameters (M)']:.2f}M) than {other_algo}.")
                if float(row['Precision'].replace('%', '')) > float(other_row['Precision'].replace('%', '')):
                    comparisons.append(f"Higher Precision ({row['Precision']} vs {other_row['Precision']}) than {other_algo}.")
                if float(row['Recall'].replace('%', '')) > float(other_row['Recall'].replace('%', '')):
                    comparisons.append(f"Higher Recall ({row['Recall']} vs {other_row['Recall']}) than {other_algo}.")
                if float(row['Small Object Score'].replace('%', '')) > float(other_row['Small Object Score'].replace('%', '')):
                    comparisons.append(f"Higher Small Object Score ({row['Small Object Score']} vs {other_row['Small Object Score']}) than {other_algo}.")
                if comparisons:
                    explanation.append(f"- {' '.join(comparisons)}")
                else:
                    explanation.append(f"- No significant advantages over {other_algo} in these metrics.")

            explanation.append(f"Overall, {algo} excels in key areas, making it the top choice for drone detection!")
        else:
            explanation.append(f"Why {algo} is Not the Best:")
            best_row = df[df['Algorithm'] == best_algo].iloc[0]
            reasons = []

            if float(row['mAP'].replace('%', '')) < float(best_row['mAP'].replace('%', '')):
                reasons.append(f"- Lower mAP ({row['mAP']} vs {best_row['mAP']}): Less accurate than {best_algo}. This means it struggles to detect drones consistently.")
            else:
                explanation.append(f"- mAP ({row['mAP']}): Matches {best_algo}, showing good accuracy in detection.")

            if row['FPS'] < best_row['FPS']:
                reasons.append(f"- Lower FPS ({row['FPS']:.2f} vs {best_row['FPS']:.2f}): Slower detection speed than {best_algo}. Not ideal for real-time applications like live drone tracking.")
            else:
                explanation.append(f"- FPS ({row['FPS']:.2f}): Matches or exceeds {best_algo}, making it suitable for real-time use.")

            if row['Inference Time (ms)'] > best_row['Inference Time (ms)']:
                reasons.append(f"- Higher Inference Time ({row['Inference Time (ms)']:.2f}ms vs {best_row['Inference Time (ms)']:.2f}ms): Slower processing per image than {best_algo}. This can cause delays in detection.")
            else:
                explanation.append(f"- Inference Time ({row['Inference Time (ms)']:.2f}ms): Matches or better than {best_algo}, ensuring quick processing.")

            if row['Model Size (MB)'] > best_row['Model Size (MB)']:
                reasons.append(f"- Larger Model Size ({row['Model Size (MB)']:.2f}MB vs {best_row['Model Size (MB)']:.2f}MB): Less efficient for deployment than {best_algo}. Bigger models need more storage and memory.")
            else:
                explanation.append(f"- Model Size ({row['Model Size (MB)']:.2f}MB): Smaller or equal to {best_algo}, making it efficient for deployment.")

            if row['Number of Parameters (M)'] > best_row['Number of Parameters (M)']:
                reasons.append(f"- More Parameters ({row['Number of Parameters (M)']:.2f}M vs {best_row['Number of Parameters (M)']:.2f}M): More complex than {best_algo}. Higher complexity can lead to slower training and inference.")
            else:
                explanation.append(f"- Number of Parameters ({row['Number of Parameters (M)']:.2f}M): Fewer or equal to {best_algo}, indicating a less complex model.")

            if float(row['Precision'].replace('%', '')) < float(best_row['Precision'].replace('%', '')):
                reasons.append(f"- Lower Precision ({row['Precision']} vs {best_row['Precision']}): More false positives than {best_algo}. This means it might detect non-drones as drones.")
            else:
                explanation.append(f"- Precision ({row['Precision']}): Matches or exceeds {best_algo}, minimizing false positives effectively.")

            if float(row['Recall'].replace('%', '')) < float(best_row['Recall'].replace('%', '')):
                reasons.append(f"- Lower Recall ({row['Recall']} vs {best_row['Recall']}): Misses more drones than {best_algo}. This can be critical in security applications.")
            else:
                explanation.append(f"- Recall ({row['Recall']}): Matches or exceeds {best_algo}, ensuring most drones are detected.")

            if float(row['Small Object Score'].replace('%', '')) < float(best_row['Small Object Score'].replace('%', '')):
                reasons.append(f"- Lower Small Object Score ({row['Small Object Score']} vs {best_row['Small Object Score']}): Weaker at detecting small drones than {best_algo}. Small drones are often harder to detect and critical in surveillance.")
            else:
                explanation.append(f"- Small Object Score ({row['Small Object Score']}): Matches or exceeds {best_algo}, showing strong small drone detection.")

            if reasons:
                explanation.append("\n".join(reasons))

            explanation.append(f"\nComparisons with Other Algorithms:")
            for _, other_row in other_rows.iterrows():
                other_algo = other_row['Algorithm']
                if other_algo == best_algo:
                    continue
                comparisons = []
                if float(row['mAP'].replace('%', '')) > float(other_row['mAP'].replace('%', '')):
                    comparisons.append(f"Higher mAP ({row['mAP']} vs {other_row['mAP']}) than {other_algo}.")
                elif float(row['mAP'].replace('%', '')) < float(other_row['mAP'].replace('%', '')):
                    comparisons.append(f"Lower mAP ({row['mAP']} vs {other_row['mAP']}) than {other_algo}.")
                if row['FPS'] > other_row['FPS']:
                    comparisons.append(f"Higher FPS ({row['FPS']:.2f} vs {other_row['FPS']:.2f}) than {other_algo}.")
                elif row['FPS'] < other_row['FPS']:
                    comparisons.append(f"Lower FPS ({row['FPS']:.2f} vs {other_row['FPS']:.2f}) than {other_algo}.")
                if row['Inference Time (ms)'] < other_row['Inference Time (ms)']:
                    comparisons.append(f"Lower Inference Time ({row['Inference Time (ms)']:.2f}ms vs {other_row['Inference Time (ms)']:.2f}ms) than {other_algo}.")
                elif row['Inference Time (ms)'] > other_row['Inference Time (ms)']:
                    comparisons.append(f"Higher Inference Time ({row['Inference Time (ms)']:.2f}ms vs {other_row['Inference Time (ms)']:.2f}ms) than {other_algo}.")
                if row['Model Size (MB)'] < other_row['Model Size (MB)']:
                    comparisons.append(f"Smaller Model Size ({row['Model Size (MB)']:.2f}MB vs {other_row['Model Size (MB)']:.2f}MB) than {other_algo}.")
                elif row['Model Size (MB)'] > other_row['Model Size (MB)']:
                    comparisons.append(f"Larger Model Size ({row['Model Size (MB)']:.2f}MB vs {other_row['Model Size (MB)']:.2f}MB) than {other_algo}.")
                if row['Number of Parameters (M)'] < other_row['Number of Parameters (M)']:
                    comparisons.append(f"Fewer Parameters ({row['Number of Parameters (M)']:.2f}M vs {other_row['Number of Parameters (M)']:.2f}M) than {other_algo}.")
                elif row['Number of Parameters (M)'] > other_row['Number of Parameters (M)']:
                    comparisons.append(f"More Parameters ({row['Number of Parameters (M)']:.2f}M vs {other_row['Number of Parameters (M)']:.2f}M) than {other_algo}.")
                if float(row['Precision'].replace('%', '')) > float(other_row['Precision'].replace('%', '')):
                    comparisons.append(f"Higher Precision ({row['Precision']} vs {other_row['Precision']}) than {other_algo}.")
                elif float(row['Precision'].replace('%', '')) < float(other_row['Precision'].replace('%', '')):
                    comparisons.append(f"Lower Precision ({row['Precision']} vs {other_row['Precision']}) than {other_algo}.")
                if float(row['Recall'].replace('%', '')) > float(other_row['Recall'].replace('%', '')):
                    comparisons.append(f"Higher Recall ({row['Recall']} vs {other_row['Recall']}) than {other_algo}.")
                elif float(row['Recall'].replace('%', '')) < float(other_row['Recall'].replace('%', '')):
                    comparisons.append(f"Lower Recall ({row['Recall']} vs {other_row['Recall']}) than {other_algo}.")
                if float(row['Small Object Score'].replace('%', '')) > float(other_row['Small Object Score'].replace('%', '')):
                    comparisons.append(f"Higher Small Object Score ({row['Small Object Score']} vs {other_row['Small Object Score']}) than {other_algo}.")
                elif float(row['Small Object Score'].replace('%', '')) < float(other_row['Small Object Score'].replace('%', '')):
                    comparisons.append(f"Lower Small Object Score ({row['Small Object Score']} vs {other_row['Small Object Score']}) than {other_algo}.")
                if comparisons:
                    explanation.append(f"- {' '.join(comparisons)}")
                else:
                    explanation.append(f"- No significant differences with {other_algo} in these metrics.")

            explanation.append(f"Compared to {best_algo}, {algo} falls short in key areas!")

        explanations[algo] = "\n".join(explanation)
    return explanations

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
        results["mAP"].append(f"{mAP:.2f}%")
        results["FPS"].append(avg_fps)
        results["Small Object Score"].append(f"{small_score:.2f}%")
        results["Model Size (MB)"].append(model_size)
        results["Precision"].append(f"{precision:.2f}%")
        results["Recall"].append(f"{recall:.2f}%")
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

    best_algo = df.loc[df['mAP'].str.replace('%', '').astype(float).idxmax()]['Algorithm'] if not df.empty else "None"
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

    return jsonify({
        "images": {
            "rcnn": labeled_images["R-CNN"][0],
            "yolov5": labeled_images["YOLOv5"][0],
            "yolov7": labeled_images["YOLOv7"][0],
            "yolov9": labeled_images["YOLOv9"][0]
        },
        "table": df.to_dict(orient="records"),
        "best_algo": best_algo,
        "metric_explanations": metric_explanations,
        "algorithm_explanations": algorithm_explanations,
        "selection_reason": selection_reason
    })

@app.route('/detect_all', methods=['GET'])
def detect_all():
    image_files = [f for f in os.listdir(IMAGES_PATH) if f.endswith(('.jpg', '.png'))]
    if not image_files:
        return jsonify({"error": "No images found!"}), 400

    df, labeled_images, num_processed = process_images(image_files)
    if df is None:
        return jsonify({"error": "Processing failed!"}), 500

    best_algo = df.loc[df['mAP'].str.replace('%', '').astype(float).idxmax()]['Algorithm'] if not df.empty else "None"
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

    return jsonify({
        "table": df.to_dict(orient="records"),
        "best_algo": best_algo,
        "metric_explanations": metric_explanations,
        "algorithm_explanations": algorithm_explanations,
        "selection_reason": selection_reason,
        "labeled_images": labeled_images,
        "num_images_processed": num_processed
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

    best_algo = df.loc[df['mAP'].str.replace('%', '').astype(float).idxmax()]['Algorithm'] if not df.empty else "None"
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

    return jsonify({
        "table": df.to_dict(orient="records"),
        "best_algo": best_algo,
        "metric_explanations": metric_explanations,
        "algorithm_explanations": algorithm_explanations,
        "selection_reason": selection_reason,
        "labeled_images": labeled_images,
        "num_images_processed": num_processed
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

    best_algo = df.loc[df['mAP'].str.replace('%', '').astype(float).idxmax()]['Algorithm'] if not df.empty else "None"
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

    return jsonify({
        "table": df.to_dict(orient="records"),
        "best_algo": best_algo,
        "metric_explanations": metric_explanations,
        "algorithm_explanations": algorithm_explanations,
        "selection_reason": selection_reason,
        "labeled_images": labeled_images,
        "num_images_processed": num_processed
    })

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)