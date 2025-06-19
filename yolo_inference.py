import onnxruntime as ort
import numpy as np
import cv2
import os
from typing import List, Tuple, Dict, Optional

class YOLOv11Inference:
    def __init__(self, model_path: str, conf_threshold: float = 0.5, nms_threshold: float = 0.4):
        """
        YOLOv11 ONNX推理模組
        
        Args:
            model_path: ONNX模型檔案路徑
            conf_threshold: 信心度閾值
            nms_threshold: NMS閾值
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = (640, 640)
        
        # 類別對應 (YOLOv11訓練時的實際類別ID)
        self.class_names = {0: "NG"}
        
        # 初始化ONNX Runtime session
        self.session = None
        self.input_name = None
        self.output_names = None
        
        self._load_model()
        self._debug_model_info()
    
    def _load_model(self):
        """載入ONNX模型"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"模型檔案不存在: {self.model_path}")
            
            # 建立ONNX Runtime session
            providers = ['CPUExecutionProvider']
            if ort.get_device() == 'GPU':
                providers.insert(0, 'CUDAExecutionProvider')
            
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            
            # 取得輸入輸出節點資訊
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            print(f"ONNX模型載入成功: {self.model_path}")
            print(f"輸入節點: {self.input_name}")
            print(f"輸出節點: {self.output_names}")
            
        except Exception as e:
            print(f"ONNX模型載入失敗: {e}")
            raise
    
    def _debug_model_info(self):
        """除錯：顯示模型輸入輸出資訊"""
        if self.session:
            print("\n=== 模型輸入輸出資訊 ===")
            for input_meta in self.session.get_inputs():
                print(f"輸入: {input_meta.name}, shape: {input_meta.shape}, type: {input_meta.type}")
            
            for output_meta in self.session.get_outputs():
                print(f"輸出: {output_meta.name}, shape: {output_meta.shape}, type: {output_meta.type}")
            print("========================\n")
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        影像預處理
        
        Args:
            image: 輸入影像 (BGR格式)
            
        Returns:
            processed_image: 處理後的影像
            scale: 縮放比例
            original_size: 原始影像尺寸
        """
        original_h, original_w = image.shape[:2]
        target_w, target_h = self.input_size
        
        # 計算縮放比例並保持長寬比
        scale = min(target_w / original_w, target_h / original_h)
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        
        # 縮放影像
        resized = cv2.resize(image, (new_w, new_h))
        
        # 建立目標尺寸的畫布並置中
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        # 轉換為RGB並正規化
        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        canvas_norm = canvas_rgb.astype(np.float32) / 255.0
        
        # 轉換為模型輸入格式 (1, 3, H, W)
        input_tensor = np.transpose(canvas_norm, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor, scale, (original_w, original_h)
    
    def postprocess_outputs(self, outputs: List[np.ndarray], scale: float, 
                          original_size: Tuple[int, int]) -> List[Dict]:
        """
        後處理YOLOv11模型輸出（Ultralytics格式）
        
        Args:
            outputs: 模型原始輸出
            scale: 縮放比例
            original_size: 原始影像尺寸
            
        Returns:
            detections: 檢測結果列表
        """
        original_w, original_h = original_size
        target_w, target_h = self.input_size
        
        # YOLOv11單類別ONNX輸出格式：(1, 5, 8400)
        # 5 = [x_center, y_center, width, height, confidence]
        predictions = outputs[0]  # shape: (1, 5, 8400)
        
        # 轉置為 (8400, 5) 格式便於處理
        predictions = predictions[0].transpose()  # (8400, 5)
        
        detections = []
        
        for pred in predictions:
            # 單類別模型輸出格式: [x_center, y_center, width, height, confidence]
            x_center, y_center, width, height, confidence = pred
            
            if confidence < self.conf_threshold:
                continue
            
            # 計算偏移量（padding）
            x_offset = (target_w - original_w * scale) / 2
            y_offset = (target_h - original_h * scale) / 2
            
            # 轉換回原始影像座標
            # YOLOv11輸出為640x640空間內的像素座標
            x_center = (x_center - x_offset) / scale
            y_center = (y_center - y_offset) / scale
            width = width / scale
            height = height / scale
            
            # 計算邊界框
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            
            # 限制在影像範圍內
            x1 = max(0, min(x1, original_w))
            y1 = max(0, min(y1, original_h))
            x2 = max(0, min(x2, original_w))
            y2 = max(0, min(y2, original_h))
            
            # 檢查邊界框有效性
            if x2 <= x1 or y2 <= y1:
                continue
            
            # 單類別模型，class_id固定為0（NG）
            class_id = 0
            class_name = self.class_names.get(class_id, f"Class_{class_id}")
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(confidence),
                'class_id': class_id,
                'class_name': class_name
            })
            
            print(f"檢測到: bbox=[{x1},{y1},{x2},{y2}], confidence={confidence:.3f}, class={class_name}")
        
        return detections
    
    def apply_nms(self, detections: List[Dict]) -> List[Dict]:
        """應用非極大值抑制"""
        if not detections:
            return []
        
        boxes = np.array([det['bbox'] for det in detections])
        scores = np.array([det['confidence'] for det in detections])
        
        # OpenCV NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(), 
            self.conf_threshold, self.nms_threshold
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            return [detections[i] for i in indices]
        
        return []
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """在影像上繪製檢測結果"""
        result_image = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            class_name = det['class_name']
            
            # 繪製邊界框 (NG用紅色)
            color = (0, 0, 255) if class_name == "NG" else (0, 255, 0)
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # 繪製標籤
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # 標籤背景
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # 標籤文字
            cv2.putText(result_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result_image
    
    def predict(self, image: np.ndarray) -> Tuple[List[Dict], bool]:
        """
        執行推理
        
        Args:
            image: 輸入影像 (BGR格式)
            
        Returns:
            detections: 檢測結果
            has_ng: 是否檢測到NG
        """
        try:
            # 預處理
            input_tensor, scale, original_size = self.preprocess_image(image)
            
            # 推理
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
            
            # 除錯：顯示原始輸出資訊
            print(f"\n=== 推理輸出除錯資訊 ===")
            for i, output in enumerate(outputs):
                print(f"輸出 {i}: shape={output.shape}, dtype={output.dtype}")
                if len(output.shape) <= 3:
                    print(f"輸出 {i} 前5個值: {output.flatten()[:5]}")
            print("=======================\n")
            
            # 後處理
            detections = self.postprocess_outputs(outputs, scale, original_size)
            detections = self.apply_nms(detections)
            
            # 檢查是否有NG
            has_ng = any(det['class_name'] == 'NG' for det in detections)
            
            return detections, has_ng
            
        except Exception as e:
            print(f"推理過程發生錯誤: {e}")
            return [], False
    
    def predict_and_draw(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict], bool]:
        """
        執行推理並繪製結果
        
        Args:
            image: 輸入影像 (BGR格式)
            
        Returns:
            result_image: 繪製檢測結果的影像
            detections: 檢測結果
            has_ng: 是否檢測到NG
        """
        detections, has_ng = self.predict(image)
        result_image = self.draw_detections(image, detections)
        
        return result_image, detections, has_ng


# 測試腳本
if __name__ == "__main__":
    # 測試推理模組
    model_path = "best.onnx"  # 請替換為實際模型路徑
    
    if os.path.exists(model_path):
        yolo = YOLOv11Inference(model_path)
        
        # 測試影像
        test_image_path = "Image__2025-05-27__10-22-44_009.png"  # 請替換為測試影像路徑
        if os.path.exists(test_image_path):
            image = cv2.imread(test_image_path)
            
            result_image, detections, has_ng = yolo.predict_and_draw(image)
            
            print(f"檢測結果: {detections}")
            print(f"是否檢測到NG: {has_ng}")
            
            # 儲存結果
            cv2.imwrite("detection_result.jpg", result_image)
            print("檢測結果已儲存至 detection_result.jpg")
        else:
            print(f"測試影像不存在: {test_image_path}")
    else:
        print(f"模型檔案不存在: {model_path}")