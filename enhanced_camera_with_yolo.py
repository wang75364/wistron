import cv2
import numpy as np
from pypylon import pylon
import threading
import time
import json
import os
from datetime import datetime
from yolo_inference import YOLOv11Inference

class BaslerCameraController:
    def __init__(self, config_file="camera_config.json", camera_index=0, yolo_model_path=None):
        self.camera = None
        self.camera_index = camera_index
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        
        self.is_streaming = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        self.config_file = config_file
        self.config = self.load_config()
        
        # 軟觸發相關
        self.software_trigger = False
        
        # YOLOv11推理模組
        self.yolo_inference = None
        if yolo_model_path and os.path.exists(yolo_model_path):
            try:
                self.yolo_inference = YOLOv11Inference(yolo_model_path)
                print(f"YOLOv11模型載入成功: {yolo_model_path}")
            except Exception as e:
                print(f"YOLOv11模型載入失敗: {e}")
        
        # 記憶體管理
        self.frame_buffer_size = 5
        self.frame_buffer = []
        
    def list_available_cameras(self):
        """列出可用的相機"""
        try:
            tl_factory = pylon.TlFactory.GetInstance()
            devices = tl_factory.EnumerateDevices()
            camera_list = []
            
            for i, device in enumerate(devices):
                camera_info = {
                    "index": i,
                    "friendly_name": device.GetFriendlyName(),
                    "model_name": device.GetModelName(),
                    "serial_number": device.GetSerialNumber()
                }
                camera_list.append(camera_info)
            
            return camera_list
        except Exception as e:
            print(f"列舉相機失敗: {e}")
            return []
    
    def load_config(self):
        default_config = {
            "resolution": "5496x3672",
            "fps": 5,
            "exposure_time": 10000,
            "gain": 0,
            "roi": {
                "x": 0,
                "y": 0,
                "width": 5496,
                "height": 3672
            }
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    for key in default_config:
                        if key not in config:
                            config[key] = default_config[key]
                    return config
            except:
                pass
        
        return default_config
    
    def save_config(self):
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
    
    def initialize_camera(self, software_trigger=False):
        """初始化相機，支援軟觸發模式"""
        try:
            tl_factory = pylon.TlFactory.GetInstance()
            devices = tl_factory.EnumerateDevices()
            
            if len(devices) == 0:
                print("沒有找到可用的相機")
                return False
            
            if self.camera_index >= len(devices):
                print(f"相機索引 {self.camera_index} 超出範圍，可用相機數量: {len(devices)}")
                return False
            
            selected_device = devices[self.camera_index]
            print(f"選擇相機: {selected_device.GetFriendlyName()}")
            
            self.camera = pylon.InstantCamera(tl_factory.CreateDevice(selected_device))
            self.camera.Open()
            
            # 設定軟觸發模式
            self.software_trigger = software_trigger
            if self.software_trigger:
                self.camera.TriggerMode.SetValue('On')
                self.camera.TriggerSource.SetValue('Software')
                self.camera.TriggerActivation.SetValue('RisingEdge')
                print("軟觸發模式已啟用")
            else:
                self.camera.TriggerMode.SetValue('Off')
                print("連續擷取模式已啟用")
            
            # 設定其他相機參數
            self.apply_config()
            
            return True
        except Exception as e:
            print(f"相機初始化失敗: {e}")
            return False
    
    def apply_config(self):
        if not self.camera or not self.camera.IsOpen():
            return False
        
        try:
            if self.camera.IsGrabbing():
                self.camera.StopGrabbing()
            
            resolution = self.config["resolution"].split("x")
            max_width, max_height = int(resolution[0]), int(resolution[1])
            roi = self.config["roi"]
            
            try:
                self.camera.OffsetX.SetValue(0)
                self.camera.OffsetY.SetValue(0)
                self.camera.Width.SetValue(self.camera.Width.Max)
                self.camera.Height.SetValue(self.camera.Height.Max)
            except:
                pass
            
            # 設定曝光時間
            try:
                if hasattr(self.camera, 'ExposureMode'):
                    self.camera.ExposureMode.SetValue('Timed')
                if hasattr(self.camera, 'ExposureAuto'):
                    self.camera.ExposureAuto.SetValue('Off')
                if hasattr(self.camera, 'ExposureTimeAbs'):
                    self.camera.ExposureTimeAbs.SetValue(self.config["exposure_time"])
            except Exception as e:
                print(f"曝光時間設定失敗: {e}")
            
            # 設定增益
            try:
                if hasattr(self.camera, 'GainAuto'):
                    self.camera.GainAuto.SetValue('Off')
                if hasattr(self.camera, 'GainRaw'):
                    self.camera.GainRaw.SetValue(self.config["gain"])
            except Exception as e:
                print(f"增益設定失敗: {e}")
            
            # 設定幀率（非軟觸發模式）
            if not self.software_trigger:
                try:
                    if hasattr(self.camera, 'AcquisitionFrameRateEnable'):
                        self.camera.AcquisitionFrameRateEnable.SetValue(True)
                    if hasattr(self.camera, 'AcquisitionFrameRateAbs'):
                        self.camera.AcquisitionFrameRateAbs.SetValue(self.config["fps"])
                except Exception as e:
                    print(f"幀率設定失敗: {e}")
            
            # 設定ROI
            if (roi["x"] > 0 or roi["y"] > 0 or 
                roi["width"] < max_width or roi["height"] < max_height):
                try:
                    if roi["width"] > 0 and roi["height"] > 0:
                        self.camera.Width.SetValue(roi["width"])
                        self.camera.Height.SetValue(roi["height"])
                    
                    if roi["x"] >= 0 and roi["y"] >= 0:
                        self.camera.OffsetX.SetValue(roi["x"])
                        self.camera.OffsetY.SetValue(roi["y"])
                except Exception as e:
                    print(f"ROI設定警告: {e}")
            
            return True
        except Exception as e:
            print(f"配置應用失敗: {e}")
            return False
    
    def software_trigger_capture(self, save_path="captures"):
        """軟觸發拍照"""
        if not self.camera or not self.camera.IsOpen() or not self.software_trigger:
            return None, None
        
        try:
            # 確保目錄存在
            os.makedirs(save_path, exist_ok=True)
            
            # 停止現有擷取
            if self.camera.IsGrabbing():
                self.camera.StopGrabbing()
            
            # 啟動單次擷取模式
            self.camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
            
            # 執行軟觸發
            self.camera.TriggerSoftware.Execute()
            
            # 等待並擷取結果
            grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            
            if grab_result.GrabSucceeded():
                # 影像格式轉換
                image = self.converter.Convert(grab_result)
                frame = image.GetArray()
                
                # 生成檔案名稱
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = f"capture_{timestamp}.png"
                filepath = os.path.join(save_path, filename)
                
                # 儲存影像
                success = cv2.imwrite(filepath, frame)
                if success:
                    # 讀取為binary data
                    with open(filepath, 'rb') as f:
                        png_binary = f.read()
                    
                    grab_result.Release()
                    
                    # 停止擷取
                    if self.camera.IsGrabbing():
                        self.camera.StopGrabbing()
                    
                    return filepath, png_binary
            
            grab_result.Release()
            
        except Exception as e:
            print(f"軟觸發拍照錯誤: {e}")
        finally:
            if self.camera.IsGrabbing():
                self.camera.StopGrabbing()
        
        return None, None
    
    def inference_image(self, image_path, save_detection_result=True):
        """
        對指定影像進行YOLOv11推理
        
        Args:
            image_path: 影像檔案路徑
            save_detection_result: 是否儲存檢測結果影像
            
        Returns:
            tuple: (detections, has_ng, result_image_path)
        """
        if not self.yolo_inference:
            return [], False, None
        
        try:
            # 讀取影像
            image = cv2.imread(image_path)
            if image is None:
                print(f"無法讀取影像: {image_path}")
                return [], False, None
            
            # 執行推理
            result_image, detections, has_ng = self.yolo_inference.predict_and_draw(image)
            
            result_image_path = None
            if save_detection_result:
                # 生成檢測結果檔案名稱
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                result_filename = f"{base_name}_detection.png"
                result_dir = os.path.dirname(image_path)
                result_image_path = os.path.join(result_dir, result_filename)
                
                # 儲存檢測結果
                cv2.imwrite(result_image_path, result_image)
            
            return detections, has_ng, result_image_path
            
        except Exception as e:
            print(f"影像推理錯誤: {e}")
            return [], False, None
    
    def capture_and_inference(self, save_path="captures"):
        """
        軟觸發拍照並進行推理
        
        Returns:
            tuple: (original_filepath, detection_filepath, detections, has_ng)
        """
        # 先拍照
        original_filepath, png_binary = self.software_trigger_capture(save_path)
        
        if original_filepath is None:
            return None, None, [], False
        
        # 進行推理
        detections, has_ng, detection_filepath = self.inference_image(original_filepath, True)
        
        return original_filepath, detection_filepath, detections, has_ng
    
    def start_streaming(self):
        """啟動連續串流（僅非軟觸發模式）"""
        if self.software_trigger:
            print("軟觸發模式下無法啟動連續串流")
            return False
        
        if not self.camera or not self.camera.IsOpen():
            if not self.initialize_camera():
                return False
        
        try:
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            self.is_streaming = True
            
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            return True
        except Exception as e:
            print(f"串流啟動失敗: {e}")
            return False
    
    def stop_streaming(self):
        self.is_streaming = False
        if self.camera and self.camera.IsGrabbing():
            self.camera.StopGrabbing()
    
    def _capture_loop(self):
        while self.is_streaming:
            try:
                if self.camera.IsGrabbing():
                    grab_result = self.camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
                    
                    if grab_result.GrabSucceeded():
                        image = self.converter.Convert(grab_result)
                        frame = image.GetArray()
                        
                        with self.frame_lock:
                            self.current_frame = frame.copy()
                            
                            self.frame_buffer.append(frame)
                            if len(self.frame_buffer) > self.frame_buffer_size:
                                self.frame_buffer.pop(0)
                    
                    grab_result.Release()
                    
            except Exception as e:
                print(f"擷取影像錯誤: {e}")
                time.sleep(0.1)
    
    def get_current_frame(self):
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None
    
    def get_camera_info(self):
        if self.camera and self.camera.IsOpen():
            try:
                info = {
                    "device_info": str(self.camera.GetDeviceInfo()),
                    "current_config": self.config,
                    "is_streaming": self.is_streaming,
                    "camera_index": self.camera_index,
                    "software_trigger": self.software_trigger,
                    "yolo_available": self.yolo_inference is not None
                }
                return info
            except:
                pass
        return None
    
    def cleanup(self):
        self.stop_streaming()
        if self.camera and self.camera.IsOpen():
            self.camera.Close()
        
        self.frame_buffer.clear()
        self.current_frame = None


# 測試腳本
if __name__ == "__main__":
    # 測試相機控制器與YOLOv11整合
    yolo_model_path = "yolov11_model.onnx"  # 請替換為實際模型路徑
    controller = BaslerCameraController(camera_index=0, yolo_model_path=yolo_model_path)
    
    # 列出可用相機
    cameras = controller.list_available_cameras()
    print("可用相機:")
    for cam in cameras:
        print(f"  {cam['index']}: {cam['friendly_name']}")
    
    # 初始化相機（軟觸發模式）
    if controller.initialize_camera(software_trigger=True):
        print("相機初始化成功（軟觸發模式）")
        
        # 測試軟觸發拍照
        original_path, png_data = controller.software_trigger_capture()
        if original_path:
            print(f"拍照成功: {original_path}")
            
            # 測試推理
            if controller.yolo_inference:
                detections, has_ng, detection_path = controller.inference_image(original_path)
                print(f"推理結果: 檢測到 {len(detections)} 個物件")
                print(f"是否有NG: {has_ng}")
                if detection_path:
                    print(f"檢測結果已儲存: {detection_path}")
            else:
                print("YOLOv11模型未載入，跳過推理")
        else:
            print("拍照失敗")
    else:
        print("相機初始化失敗")
    
    controller.cleanup()
