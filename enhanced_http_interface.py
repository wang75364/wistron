from flask import Flask, render_template_string, request, jsonify, send_file
import os
import json
import threading
import time
import re
from datetime import datetime, timedelta
from enhanced_camera_with_yolo import BaslerCameraController

app = Flask(__name__)

# 全域相機控制器
camera_controller = None
cleanup_thread = None
cleanup_running = True

class FileManager:
    def __init__(self, captures_dir="captures"):
        self.captures_dir = captures_dir
        
    def parse_timestamp_from_filename(self, filename):
        """從檔案名稱解析時間戳"""
        try:
            # 解析格式: capture_YYYYMMDD_HHMMSS_mmm.png
            pattern = r'capture_(\d{8})_(\d{6})_(\d{3})(?:_detection)?\.png'
            match = re.match(pattern, filename)
            
            if match:
                date_str = match.group(1)  # YYYYMMDD
                time_str = match.group(2)  # HHMMSS
                ms_str = match.group(3)    # mmm
                
                # 轉換為datetime物件
                dt_str = f"{date_str}_{time_str}_{ms_str}"
                dt = datetime.strptime(dt_str, "%Y%m%d_%H%M%S_%f")
                return dt
            
        except Exception as e:
            print(f"解析檔案時間戳失敗 {filename}: {e}")
        
        return None
    
    def get_all_files(self):
        """取得所有檔案並按時間排序"""
        files = []
        
        if not os.path.exists(self.captures_dir):
            return files
        
        for filename in os.listdir(self.captures_dir):
            if filename.endswith('.png') and filename.startswith('capture_'):
                filepath = os.path.join(self.captures_dir, filename)
                timestamp = self.parse_timestamp_from_filename(filename)
                
                if timestamp:
                    is_detection = '_detection' in filename
                    original_name = filename.replace('_detection', '') if is_detection else filename
                    
                    files.append({
                        'filename': filename,
                        'filepath': filepath,
                        'timestamp': timestamp,
                        'is_detection': is_detection,
                        'original_name': original_name
                    })
        
        # 按時間戳排序（最新的在前）
        files.sort(key=lambda x: x['timestamp'], reverse=True)
        return files
    
    def get_latest_files(self, file_type='both'):
        """
        取得最新檔案
        file_type: 'original', 'detection', 'both'
        """
        all_files = self.get_all_files()
        result = {}
        
        if file_type in ['original', 'both']:
            # 找最新的原始檔案
            for file_info in all_files:
                if not file_info['is_detection']:
                    result['latest_original'] = file_info
                    break
        
        if file_type in ['detection', 'both']:
            # 找最新的辨識檔案
            for file_info in all_files:
                if file_info['is_detection']:
                    result['latest_detection'] = file_info
                    break
        
        return result
    
    def get_paired_files(self):
        """取得配對的檔案（原始+辨識）"""
        all_files = self.get_all_files()
        pairs = {}
        
        # 建立檔案配對
        for file_info in all_files:
            original_name = file_info['original_name']
            if original_name not in pairs:
                pairs[original_name] = {'original': None, 'detection': None}
            
            if file_info['is_detection']:
                pairs[original_name]['detection'] = file_info
            else:
                pairs[original_name]['original'] = file_info
        
        # 按時間排序
        sorted_pairs = []
        for original_name, pair in pairs.items():
            if pair['original']:  # 必須有原始檔案
                timestamp = pair['original']['timestamp']
                sorted_pairs.append({
                    'original_name': original_name,
                    'timestamp': timestamp,
                    **pair
                })
        
        sorted_pairs.sort(key=lambda x: x['timestamp'], reverse=True)
        return sorted_pairs
    
    def cleanup_old_files(self, days=7):
        """清理超過指定天數的檔案"""
        cutoff_date = datetime.now() - timedelta(days=days)
        all_files = self.get_all_files()
        deleted_files = []
        
        for file_info in all_files:
            if file_info['timestamp'] < cutoff_date:
                try:
                    os.remove(file_info['filepath'])
                    deleted_files.append(file_info['filename'])
                    print(f"已刪除過期檔案: {file_info['filename']}")
                except Exception as e:
                    print(f"刪除檔案失敗 {file_info['filename']}: {e}")
        
        return deleted_files

# 初始化檔案管理器
file_manager = FileManager()

def cleanup_worker():
    """清理工作執行緒"""
    global cleanup_running
    
    while cleanup_running:
        try:
            # 每小時執行一次清理
            time.sleep(3600)  # 3600秒 = 1小時
            
            if cleanup_running:
                print("開始執行檔案清理...")
                deleted_files = file_manager.cleanup_old_files(days=7)
                if deleted_files:
                    print(f"清理完成，刪除了 {len(deleted_files)} 個檔案")
                else:
                    print("清理完成，沒有需要刪除的檔案")
                    
        except Exception as e:
            print(f"清理執行緒錯誤: {e}")
            time.sleep(60)  # 錯誤後等待1分鐘再繼續

# HTML模板 (新增搜尋功能)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Basler相機軟觸發系統 with YOLOv11</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Microsoft JhengHei', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 20px;
        }
        
        .panel {
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
            border: 1px solid rgba(255, 255, 255, 0.18);
        }
        
        h1 {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 30px;
            grid-column: 1 / -1;
            font-size: 2em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        
        h2 {
            color: #34495e;
            margin-bottom: 20px;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        
        .button-group {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        
        button {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
            min-width: 120px;
        }
        
        .btn-primary {
            background: linear-gradient(45deg, #3498db, #2980b9);
            color: white;
        }
        
        .btn-success {
            background: linear-gradient(45deg, #27ae60, #229954);
            color: white;
        }
        
        .btn-warning {
            background: linear-gradient(45deg, #f39c12, #e67e22);
            color: white;
        }
        
        .btn-danger {
            background: linear-gradient(45deg, #e74c3c, #c0392b);
            color: white;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .parameter-group {
            margin-bottom: 15px;
        }
        
        label {
            display: block;
            margin-bottom: 5px;
            color: #34495e;
            font-weight: 500;
        }
        
        input[type="number"], input[type="range"], select {
            width: 100%;
            padding: 8px 12px;
            border: 2px solid #bdc3c7;
            border-radius: 6px;
            font-size: 14px;
            transition: border-color 0.3s ease;
        }
        
        input:focus, select:focus {
            outline: none;
            border-color: #3498db;
        }
        
        .status {
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-weight: 500;
            text-align: center;
        }
        
        .status.success {
            background: #d5f4e6;
            color: #27ae60;
            border: 1px solid #27ae60;
        }
        
        .status.error {
            background: #fdf2f2;
            color: #e74c3c;
            border: 1px solid #e74c3c;
        }
        
        .status.info {
            background: #ebf8ff;
            color: #3498db;
            border: 1px solid #3498db;
        }
        
        .image-preview {
            text-align: center;
            margin-top: 20px;
        }
        
        .image-preview img {
            max-width: 100%;
            max-height: 300px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        
        .detection-result {
            margin-top: 20px;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
        }
        
        .detection-ok {
            background: #d5f4e6;
            color: #27ae60;
            border: 2px solid #27ae60;
        }
        
        .detection-ng {
            background: #fdf2f2;
            color: #e74c3c;
            border: 2px solid #e74c3c;
        }
        
        .loading {
            display: none;
            text-align: center;
            color: #3498db;
            font-style: italic;
        }
        
        .file-info {
            background: #f8f9fa;
            border-radius: 6px;
            padding: 10px;
            margin-bottom: 10px;
            font-size: 12px;
            color: #6c757d;
        }
        
        @media (max-width: 1200px) {
            .container {
                grid-template-columns: 1fr 1fr;
            }
        }
        
        @media (max-width: 768px) {
            .container {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Basler相機軟觸發系統 with YOLOv11</h1>
        
        <div class="panel">
            <h2>相機控制</h2>
            <div class="button-group">
                <button class="btn-primary" onclick="initializeCamera()">初始化相機</button>
                <button class="btn-success" onclick="triggerCapture()">軟觸發拍照</button>
                <button class="btn-warning" onclick="captureAndInference()">拍照+辨識</button>
                <button class="btn-danger" onclick="cleanupCamera()">清理資源</button>
            </div>
            
            <div class="button-group">
                <button class="btn-primary" onclick="getCameraInfo()">相機資訊</button>
                <button class="btn-success" onclick="inference()">辨識最新影像</button>
            </div>
            
            <div id="status" class="status info">請先初始化相機</div>
            <div id="loading" class="loading">處理中...</div>
            
            <div id="detectionResult" class="detection-result" style="display: none;"></div>
        </div>
        
        <div class="panel">
            <h2>檔案搜尋</h2>
            <div class="parameter-group">
                <label for="searchType">搜尋類型:</label>
                <select id="searchType">
                    <option value="both">最新原始+辨識</option>
                    <option value="original">最新原始檔案</option>
                    <option value="detection">最新辨識檔案</option>
                    <option value="pairs">所有配對檔案</option>
                </select>
            </div>
            
            <div class="button-group">
                <button class="btn-primary" onclick="searchFiles()">搜尋檔案</button>
                <button class="btn-warning" onclick="loadLatestFiles()">載入最新檔案</button>
                <button class="btn-danger" onclick="manualCleanup()">手動清理</button>
            </div>
            
            <div id="searchResults"></div>
        </div>
        
        <div class="panel">
            <h2>相機參數</h2>
            <div class="parameter-group">
                <label for="exposure">曝光時間 (μs):</label>
                <input type="number" id="exposure" min="100" max="100000" value="10000">
            </div>
            
            <div class="parameter-group">
                <label for="gain">增益:</label>
                <input type="range" id="gain" min="0" max="100" value="0" oninput="updateGainValue(this.value)">
                <span id="gainValue">0</span>
            </div>
            
            <button class="btn-primary" onclick="updateParameters()">更新參數</button>
        </div>
        
        <div class="panel">
            <h2>原始影像</h2>
            <div id="imagePreview" class="image-preview">
                <p>尚未拍攝影像</p>
            </div>
            <div id="imageInfo" class="file-info" style="display: none;"></div>
            <div id="downloadLink" style="text-align: center; margin-top: 10px;"></div>
        </div>
        
        <div class="panel">
            <h2>檢測結果影像</h2>
            <div id="detectionPreview" class="image-preview">
                <p>尚未進行辨識</p>
            </div>
            <div id="detectionInfo" class="file-info" style="display: none;"></div>
            <div id="detectionDownloadLink" style="text-align: center; margin-top: 10px;"></div>
        </div>
    </div>

    <script>
        let currentImageFilename = null;
        let currentDetectionFilename = null;
        
        function showLoading() {
            document.getElementById('loading').style.display = 'block';
        }
        
        function hideLoading() {
            document.getElementById('loading').style.display = 'none';
        }
        
        function updateStatus(message, type = 'info') {
            const statusDiv = document.getElementById('status');
            statusDiv.textContent = message;
            statusDiv.className = `status ${type}`;
        }
        
        function updateGainValue(value) {
            document.getElementById('gainValue').textContent = value;
        }
        
        function formatTimestamp(timestamp) {
            return new Date(timestamp).toLocaleString('zh-TW');
        }
        
        async function initializeCamera() {
            showLoading();
            try {
                const response = await fetch('/api/initialize', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ software_trigger: true })
                });
                const result = await response.json();
                
                if (result.success) {
                    updateStatus('相機初始化成功（軟觸發模式）', 'success');
                } else {
                    updateStatus(`初始化失敗: ${result.error}`, 'error');
                }
            } catch (error) {
                updateStatus(`錯誤: ${error.message}`, 'error');
            }
            hideLoading();
        }
        
        async function triggerCapture() {
            showLoading();
            try {
                const response = await fetch('/api/capture', { method: 'POST' });
                const result = await response.json();
                
                if (result.success) {
                    updateStatus(`拍照成功: ${result.filename}`, 'success');
                    currentImageFilename = result.filename;
                    displayCapturedImage(result.filename);
                    hideDetectionResult();
                } else {
                    updateStatus(`拍照失敗: ${result.error}`, 'error');
                }
            } catch (error) {
                updateStatus(`錯誤: ${error.message}`, 'error');
            }
            hideLoading();
        }
        
        async function captureAndInference() {
            showLoading();
            try {
                const response = await fetch('/api/capture_and_inference', { method: 'POST' });
                const result = await response.json();
                
                if (result.success) {
                    updateStatus(`拍照+辨識成功: ${result.original_filename}`, 'success');
                    currentImageFilename = result.original_filename;
                    currentDetectionFilename = result.detection_filename;
                    
                    displayCapturedImage(result.original_filename);
                    if (result.detection_filename) {
                        displayDetectionImage(result.detection_filename);
                    }
                    
                    showDetectionResult(result.has_ng, result.detections.length);
                } else {
                    updateStatus(`拍照+辨識失敗: ${result.error}`, 'error');
                }
            } catch (error) {
                updateStatus(`錯誤: ${error.message}`, 'error');
            }
            hideLoading();
        }
        
        async function inference() {
            if (!currentImageFilename) {
                updateStatus('請先拍攝影像', 'error');
                return;
            }
            
            showLoading();
            try {
                const response = await fetch('/api/inference', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ filename: currentImageFilename })
                });
                const result = await response.json();
                
                if (result.success) {
                    updateStatus(`辨識完成: 檢測到 ${result.detections.length} 個物件`, 'success');
                    currentDetectionFilename = result.detection_filename;
                    
                    if (result.detection_filename) {
                        displayDetectionImage(result.detection_filename);
                    }
                    
                    showDetectionResult(result.has_ng, result.detections.length);
                } else {
                    updateStatus(`辨識失敗: ${result.error}`, 'error');
                }
            } catch (error) {
                updateStatus(`錯誤: ${error.message}`, 'error');
            }
            hideLoading();
        }
        
        async function searchFiles() {
            const searchType = document.getElementById('searchType').value;
            showLoading();
            
            try {
                const response = await fetch(`/api/search_files?type=${searchType}`);
                const result = await response.json();
                
                if (result.success) {
                    displaySearchResults(result.data, searchType);
                    updateStatus(`檔案搜尋完成`, 'success');
                } else {
                    updateStatus(`搜尋失敗: ${result.error}`, 'error');
                }
            } catch (error) {
                updateStatus(`錯誤: ${error.message}`, 'error');
            }
            hideLoading();
        }
        
        async function loadLatestFiles() {
            showLoading();
            
            try {
                const response = await fetch('/api/search_files?type=both');
                const result = await response.json();
                
                if (result.success) {
                    const data = result.data;
                    
                    if (data.latest_original) {
                        currentImageFilename = data.latest_original.filename;
                        displayCapturedImage(data.latest_original.filename, data.latest_original.timestamp);
                    }
                    
                    if (data.latest_detection) {
                        currentDetectionFilename = data.latest_detection.filename;
                        displayDetectionImage(data.latest_detection.filename, data.latest_detection.timestamp);
                    }
                    
                    updateStatus('最新檔案載入完成', 'success');
                } else {
                    updateStatus(`載入失敗: ${result.error}`, 'error');
                }
            } catch (error) {
                updateStatus(`錯誤: ${error.message}`, 'error');
            }
            hideLoading();
        }
        
        async function manualCleanup() {
            if (!confirm('確定要清理超過一周的檔案嗎？此操作無法復原。')) {
                return;
            }
            
            showLoading();
            
            try {
                const response = await fetch('/api/cleanup_files', { method: 'POST' });
                const result = await response.json();
                
                if (result.success) {
                    updateStatus(`清理完成，刪除了 ${result.deleted_count} 個檔案`, 'success');
                    // 重新搜尋檔案
                    searchFiles();
                } else {
                    updateStatus(`清理失敗: ${result.error}`, 'error');
                }
            } catch (error) {
                updateStatus(`錯誤: ${error.message}`, 'error');
            }
            hideLoading();
        }
        
        function displaySearchResults(data, searchType) {
            const resultsDiv = document.getElementById('searchResults');
            let html = '<h3>搜尋結果</h3>';
            
            if (searchType === 'pairs') {
                if (data.length === 0) {
                    html += '<p>沒有找到檔案</p>';
                } else {
                    html += `<p>找到 ${data.length} 組檔案:</p>`;
                    data.slice(0, 5).forEach(pair => {
                        html += `
                            <div class="file-info">
                                <strong>${pair.original_name}</strong><br>
                                時間: ${formatTimestamp(pair.timestamp)}<br>
                                原始: ${pair.original ? '✓' : '✗'} 
                                辨識: ${pair.detection ? '✓' : '✗'}
                            </div>
                        `;
                    });
                    if (data.length > 5) {
                        html += `<p>... 還有 ${data.length - 5} 組檔案</p>`;
                    }
                }
            } else {
                if (data.latest_original) {
                    html += `
                        <div class="file-info">
                            <strong>最新原始檔案:</strong><br>
                            ${data.latest_original.filename}<br>
                            時間: ${formatTimestamp(data.latest_original.timestamp)}
                        </div>
                    `;
                }
                
                if (data.latest_detection) {
                    html += `
                        <div class="file-info">
                            <strong>最新辨識檔案:</strong><br>
                            ${data.latest_detection.filename}<br>
                            時間: ${formatTimestamp(data.latest_detection.timestamp)}
                        </div>
                    `;
                }
                
                if (!data.latest_original && !data.latest_detection) {
                    html += '<p>沒有找到檔案</p>';
                }
            }
            
            resultsDiv.innerHTML = html;
        }
        
        async function getCameraInfo() {
            showLoading();
            try {
                const response = await fetch('/api/info');
                const result = await response.json();
                
                if (result.success) {
                    updateStatus('相機資訊取得成功', 'success');
                    console.log('相機資訊:', result.info);
                } else {
                    updateStatus(`取得資訊失敗: ${result.error}`, 'error');
                }
            } catch (error) {
                updateStatus(`錯誤: ${error.message}`, 'error');
            }
            hideLoading();
        }
        
        async function updateParameters() {
            const exposure = document.getElementById('exposure').value;
            const gain = document.getElementById('gain').value;
            
            showLoading();
            try {
                const response = await fetch('/api/parameters', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        exposure_time: parseInt(exposure),
                        gain: parseInt(gain)
                    })
                });
                const result = await response.json();
                
                if (result.success) {
                    updateStatus('參數更新成功', 'success');
                } else {
                    updateStatus(`參數更新失敗: ${result.error}`, 'error');
                }
            } catch (error) {
                updateStatus(`錯誤: ${error.message}`, 'error');
            }
            hideLoading();
        }
        
        async function cleanupCamera() {
            showLoading();
            try {
                const response = await fetch('/api/cleanup', { method: 'POST' });
                const result = await response.json();
                
                if (result.success) {
                    updateStatus('資源清理完成', 'success');
                    clearPreviews();
                } else {
                    updateStatus(`清理失敗: ${result.error}`, 'error');
                }
            } catch (error) {
                updateStatus(`錯誤: ${error.message}`, 'error');
            }
            hideLoading();
        }
        
        function displayCapturedImage(filename, timestamp = null) {
            const imageUrl = `/api/image/${filename}`;
            const downloadUrl = `/api/download/${filename}`;
            
            document.getElementById('imagePreview').innerHTML = 
                `<img src="${imageUrl}" alt="拍攝影像">`;
            
            if (timestamp) {
                document.getElementById('imageInfo').innerHTML = 
                    `檔案: ${filename}<br>時間: ${formatTimestamp(timestamp)}`;
                document.getElementById('imageInfo').style.display = 'block';
            } else {
                document.getElementById('imageInfo').style.display = 'none';
            }
            
            document.getElementById('downloadLink').innerHTML = 
                `<button class="btn-primary" onclick="window.open('${downloadUrl}')">下載原始影像</button>`;
        }
        
        function displayDetectionImage(filename, timestamp = null) {
            const imageUrl = `/api/image/${filename}`;
            const downloadUrl = `/api/download/${filename}`;
            
            document.getElementById('detectionPreview').innerHTML = 
                `<img src="${imageUrl}" alt="檢測結果">`;
            
            if (timestamp) {
                document.getElementById('detectionInfo').innerHTML = 
                    `檔案: ${filename}<br>時間: ${formatTimestamp(timestamp)}`;
                document.getElementById('detectionInfo').style.display = 'block';
            } else {
                document.getElementById('detectionInfo').style.display = 'none';
            }
            
            document.getElementById('detectionDownloadLink').innerHTML = 
                `<button class="btn-primary" onclick="window.open('${downloadUrl}')">下載檢測結果</button>`;
        }
        
        function showDetectionResult(hasNg, detectionCount) {
            const resultDiv = document.getElementById('detectionResult');
            resultDiv.style.display = 'block';
            
            if (hasNg) {
                resultDiv.className = 'detection-result detection-ng';
                resultDiv.textContent = `檢測結果: NG (發現 ${detectionCount} 個缺陷)`;
            } else {
                resultDiv.className = 'detection-result detection-ok';
                resultDiv.textContent = `檢測結果: OK (無缺陷)`;
            }
        }
        
        function hideDetectionResult() {
            document.getElementById('detectionResult').style.display = 'none';
            document.getElementById('detectionPreview').innerHTML = '<p>尚未進行辨識</p>';
            document.getElementById('detectionDownloadLink').innerHTML = '';
            document.getElementById('detectionInfo').style.display = 'none';
            currentDetectionFilename = null;
        }
        
        function clearPreviews() {
            document.getElementById('imagePreview').innerHTML = '<p>尚未拍攝影像</p>';
            document.getElementById('downloadLink').innerHTML = '';
            document.getElementById('imageInfo').style.display = 'none';
            hideDetectionResult();
            currentImageFilename = null;
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/search_files')
def search_files():
    """搜尋檔案API"""
    try:
        file_type = request.args.get('type', 'both')
        
        if file_type == 'pairs':
            # 回傳所有配對檔案
            pairs = file_manager.get_paired_files()
            return jsonify({
                'success': True,
                'data': pairs,
                'count': len(pairs)
            })
        else:
            # 回傳最新檔案
            latest_files = file_manager.get_latest_files(file_type)
            return jsonify({
                'success': True,
                'data': latest_files
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/cleanup_files', methods=['POST'])
def cleanup_files():
    """手動清理檔案API"""
    try:
        deleted_files = file_manager.cleanup_old_files(days=7)
        return jsonify({
            'success': True,
            'deleted_files': deleted_files,
            'deleted_count': len(deleted_files)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/initialize', methods=['POST'])
def initialize_camera():
    global camera_controller
    
    try:
        data = request.get_json() or {}
        software_trigger = data.get('software_trigger', True)
        yolo_model_path = data.get('yolo_model_path', 'yolo11n.onnx')
        
        if not os.path.exists(yolo_model_path):
            yolo_model_path = None
        
        camera_controller = BaslerCameraController(
            camera_index=0, 
            yolo_model_path=yolo_model_path
        )
        
        if camera_controller.initialize_camera(software_trigger=software_trigger):
            return jsonify({
                'success': True, 
                'message': '相機初始化成功',
                'yolo_available': camera_controller.yolo_inference is not None
            })
        else:
            return jsonify({'success': False, 'error': '相機初始化失敗'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/capture', methods=['POST'])
def capture_image():
    global camera_controller
    
    if not camera_controller:
        return jsonify({'success': False, 'error': '相機未初始化'})
    
    try:
        filepath, png_binary = camera_controller.software_trigger_capture()
        
        if filepath:
            filename = os.path.basename(filepath)
            return jsonify({
                'success': True, 
                'filename': filename,
                'filepath': filepath
            })
        else:
            return jsonify({'success': False, 'error': '拍照失敗'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/capture_and_inference', methods=['POST'])
def capture_and_inference():
    global camera_controller
    
    if not camera_controller:
        return jsonify({'success': False, 'error': '相機未初始化'})
    
    if not camera_controller.yolo_inference:
        return jsonify({'success': False, 'error': 'YOLOv11模型未載入'})
    
    try:
        original_filepath, detection_filepath, detections, has_ng = camera_controller.capture_and_inference()
        
        if original_filepath:
            original_filename = os.path.basename(original_filepath)
            detection_filename = os.path.basename(detection_filepath) if detection_filepath else None
            
            return jsonify({
                'success': True,
                'original_filename': original_filename,
                'detection_filename': detection_filename,
                'detections': detections,
                'has_ng': has_ng,
                'detection_count': len(detections)
            })
        else:
            return jsonify({'success': False, 'error': '拍照失敗'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/inference', methods=['POST'])
def inference_image():
    global camera_controller
    
    if not camera_controller:
        return jsonify({'success': False, 'error': '相機未初始化'})
    
    if not camera_controller.yolo_inference:
        return jsonify({'success': False, 'error': 'YOLOv11模型未載入'})
    
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'success': False, 'error': '未指定檔案名稱'})
        
        filepath = os.path.join(os.getcwd(), "captures", filename)
        
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': f'檔案不存在: {filename}'})
        
        detections, has_ng, detection_filepath = camera_controller.inference_image(filepath, True)
        detection_filename = os.path.basename(detection_filepath) if detection_filepath else None
        
        return jsonify({
            'success': True,
            'detections': detections,
            'has_ng': has_ng,
            'detection_count': len(detections),
            'detection_filename': detection_filename
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/image/<filename>')
def get_image(filename):
    try:
        filepath = os.path.join(os.getcwd(), "captures", filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': f'檔案不存在: {filepath}'}), 404
        
        return send_file(filepath, mimetype='image/png', as_attachment=False)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download/<filename>')
def download_image(filename):
    try:
        filepath = os.path.join(os.getcwd(), "captures", filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': f'檔案不存在: {filepath}'}), 404
        
        return send_file(filepath, mimetype='image/png', as_attachment=True, 
                        download_name=filename)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/parameters', methods=['POST'])
def update_parameters():
    global camera_controller
    
    if not camera_controller:
        return jsonify({'success': False, 'error': '相機未初始化'})
    
    try:
        data = request.get_json()
        
        results = {}
        
        if 'exposure_time' in data:
            results['exposure'] = camera_controller.update_exposure(data['exposure_time'])
        
        if 'gain' in data:
            results['gain'] = camera_controller.update_gain(data['gain'])
        
        if any(results.values()):
            return jsonify({'success': True, 'results': results})
        else:
            return jsonify({'success': False, 'error': '參數更新失敗'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/info')
def get_camera_info():
    global camera_controller
    
    if not camera_controller:
        return jsonify({'success': False, 'error': '相機未初始化'})
    
    try:
        info = camera_controller.get_camera_info()
        if info:
            return jsonify({'success': True, 'info': info})
        else:
            return jsonify({'success': False, 'error': '無法取得相機資訊'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/cleanup', methods=['POST'])
def cleanup_camera():
    global camera_controller
    
    try:
        if camera_controller:
            camera_controller.cleanup()
            camera_controller = None
        
        return jsonify({'success': True, 'message': '資源清理完成'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def start_cleanup_thread():
    """啟動清理執行緒"""
    global cleanup_thread, cleanup_running
    
    if cleanup_thread is None or not cleanup_thread.is_alive():
        cleanup_running = True
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
        print("檔案清理執行緒已啟動")

def stop_cleanup_thread():
    """停止清理執行緒"""
    global cleanup_running
    cleanup_running = False
    print("檔案清理執行緒停止信號已發送")

if __name__ == '__main__':
    # 確保captures目錄存在
    os.makedirs('captures', exist_ok=True)
    
    # 啟動清理執行緒
    start_cleanup_thread()
    
    try:
        # 啟動Flask應用
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("收到中斷信號，正在關閉...")
    finally:
        # 停止清理執行緒
        stop_cleanup_thread()