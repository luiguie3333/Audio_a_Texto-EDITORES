#!/usr/bin/env python3
"""
Servidor web avanzado para Whisper con interfaz moderna y progreso en tiempo real
Versión mejorada con modelos más potentes y mejor UX
"""

import http.server
import socketserver
import json
import os
import tempfile
import urllib.parse
import mimetypes
import whisper
import io
from pathlib import Path
import threading
import time
import queue
import uuid
from datetime import datetime

# Puerto del servidor
PORT = 8000

# Cache de modelos Whisper
model_cache = {}

# Cola para manejar el progreso de transcripciones
progress_queue = {}

def get_model(model_size):
    """Obtener modelo desde cache o cargarlo"""
    if model_size not in model_cache:
        print(f"📥 Cargando modelo Whisper: {model_size}")
        model_cache[model_size] = whisper.load_model(model_size)
        print(f"✅ Modelo {model_size} cargado")
    return model_cache[model_size]

def format_timestamp(seconds):
    """Convierte segundos a formato SRT"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

def create_srt_content(segments, words_per_line=5):
    """Crear contenido SRT con palabras personalizables"""
    srt_content = []
    subtitle_index = 1
    
    for segment in segments:
        if 'words' not in segment:
            continue
            
        words = segment['words']
        i = 0
        
        while i < len(words):
            line_words = []
            start_time = words[i]['start']
            
            word_count = 0
            while i < len(words) and word_count < words_per_line:
                line_words.append(words[i]['word'].strip())
                word_count += 1
                i += 1
            
            end_time = words[i-1]['end'] if i > 0 else start_time + 0.5
            
            text = ' '.join(line_words)
            start_formatted = format_timestamp(start_time)
            end_formatted = format_timestamp(end_time)
            
            srt_content.append(f"{subtitle_index}")
            srt_content.append(f"{start_formatted} --> {end_formatted}")
            srt_content.append(text)
            srt_content.append("")
            
            subtitle_index += 1
    
    return '\n'.join(srt_content)

def transcribe_audio_with_progress(audio_path, model_size, language, words_per_line, task_id):
    """Transcribir audio con seguimiento de progreso"""
    try:
        # Actualizar progreso: Cargando modelo
        progress_queue[task_id] = {
            'status': 'loading_model',
            'progress': 20,
            'message': f'Cargando modelo {model_size}...'
        }
        
        model = get_model(model_size)
        
        # Actualizar progreso: Iniciando transcripción
        progress_queue[task_id] = {
            'status': 'transcribing',
            'progress': 40,
            'message': 'Iniciando transcripción...'
        }
        
        # Transcribir
        result = model.transcribe(
            audio_path,
            language=language,
            word_timestamps=True,
            verbose=False
        )
        
        # Actualizar progreso: Generando SRT
        progress_queue[task_id] = {
            'status': 'generating_srt',
            'progress': 80,
            'message': 'Generando archivo SRT...'
        }
        
        srt_content = create_srt_content(result['segments'], words_per_line)
        
        # Completado
        progress_queue[task_id] = {
            'status': 'completed',
            'progress': 100,
            'message': 'Transcripción completada',
            'result': srt_content,
            'segments_count': len(result['segments']),
            'detected_language': result.get('language', 'unknown')
        }
        
    except Exception as e:
        progress_queue[task_id] = {
            'status': 'error',
            'progress': 0,
            'message': f'Error: {str(e)}'
        }

class WhisperRequestHandler(http.server.BaseHTTPRequestHandler):
    
    def do_GET(self):
        """Manejar peticiones GET"""
        if self.path == '/':
            self.serve_html()
        elif self.path == '/health':
            self.serve_health()
        elif self.path.startswith('/progress/'):
            self.serve_progress()
        elif self.path.startswith('/download/'):
            self.serve_download()
        else:
            self.send_error(404, "Página no encontrada")
    
    def do_POST(self):
        """Manejar peticiones POST"""
        if self.path == '/transcribe':
            self.handle_transcribe()
        else:
            self.send_error(404, "Endpoint no encontrado")
    
    def serve_html(self):
        """Servir la página HTML principal con diseño moderno"""
        html_content = '''<!DOCTYPE html>
<html>
<head>
    <title>Whisper AI Transcriptor</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        }
        
        .header {
            text-align: center;
            margin-bottom: 40px;
        }
        
        .header h1 {
            color: #333;
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 300;
        }
        
        .header .subtitle {
            color: #666;
            font-size: 1.1em;
        }
        
        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .form-group {
            display: flex;
            flex-direction: column;
        }
        
        .form-group.full-width {
            grid-column: 1 / -1;
        }
        
        label {
            font-weight: 600;
            color: #333;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        input, select {
            padding: 12px 16px;
            border: 2px solid #e1e5e9;
            border-radius: 10px;
            font-size: 16px;
            transition: all 0.3s ease;
            background: #fff;
        }
        
        input:focus, select:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        input[type="file"] {
            border: 2px dashed #e1e5e9;
            background: #f8f9fa;
            cursor: pointer;
            padding: 20px;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        input[type="file"]:hover {
            border-color: #667eea;
            background: #f0f2ff;
        }
        
        .model-info {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 10px;
        }
        
        .model-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            border: 2px solid transparent;
            cursor: pointer;
            transition: all 0.3s ease;
            text-align: center;
        }
        
        .model-card:hover, .model-card.selected {
            border-color: #667eea;
            background: #f0f2ff;
        }
        
        .model-card h4 {
            color: #333;
            margin-bottom: 5px;
        }
        
        .model-card p {
            color: #666;
            font-size: 0.9em;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 10px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            margin-top: 20px;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }
        
        .btn-primary:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        
        .progress-section {
            display: none;
            margin-top: 30px;
            padding: 25px;
            background: #f8f9fa;
            border-radius: 15px;
            border-left: 5px solid #667eea;
        }
        
        .progress-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .progress-title {
            font-weight: 600;
            color: #333;
        }
        
        .progress-percentage {
            font-weight: 600;
            color: #667eea;
            font-size: 1.2em;
        }
        
        .progress-bar-container {
            width: 100%;
            height: 12px;
            background: #e1e5e9;
            border-radius: 6px;
            overflow: hidden;
            margin-bottom: 15px;
        }
        
        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            width: 0%;
            transition: width 0.5s ease;
            border-radius: 6px;
        }
        
        .progress-message {
            color: #666;
            font-style: italic;
        }
        
        .result-section {
            display: none;
            margin-top: 30px;
            padding: 25px;
            background: #e8f5e8;
            border-radius: 15px;
            border-left: 5px solid #28a745;
        }
        
        .result-info {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .info-item {
            display: flex;
            align-items: center;
            gap: 8px;
            color: #333;
            font-weight: 500;
        }
        
        .download-btn {
            background: #28a745;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .download-btn:hover {
            background: #218838;
            transform: translateY(-1px);
        }
        
        .log-section {
            display: none;
            margin-top: 20px;
            max-height: 200px;
            overflow-y: auto;
            background: #1a1a1a;
            border-radius: 10px;
            padding: 15px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
        }
        
        .log-entry {
            color: #00ff00;
            margin-bottom: 5px;
        }
        
        .log-entry.error { color: #ff4444; }
        .log-entry.warning { color: #ffaa00; }
        .log-entry.info { color: #44aaff; }
        
        @media (max-width: 768px) {
            .form-grid { grid-template-columns: 1fr; }
            .container { padding: 20px; }
            .header h1 { font-size: 2em; }
            .model-info { grid-template-columns: 1fr; }
        }
        
        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 1s ease-in-out infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-microphone"></i> Whisper AI Transcriptor</h1>
            <p class="subtitle">Convierte audio a subtítulos con inteligencia artificial avanzada</p>
        </div>
        
        <form id="uploadForm" enctype="multipart/form-data">
            <div class="form-grid">
                <div class="form-group full-width">
                    <label><i class="fas fa-file-audio"></i> Archivo de Audio</label>
                    <input type="file" id="audioFile" accept="audio/*,video/*" required>
                </div>
                
                <div class="form-group">
                    <label><i class="fas fa-list-ol"></i> Palabras por línea</label>
                    <input type="number" id="wordsPerLine" value="6" min="1" max="25">
                </div>
                
                <div class="form-group">
                    <label><i class="fas fa-language"></i> Idioma</label>
                    <select id="language">
                        <option value="auto">Auto-detectar</option>
                        <option value="es">Español</option>
                        <option value="en">Inglés</option>
                        <option value="fr">Francés</option>
                        <option value="de">Alemán</option>
                        <option value="it">Italiano</option>
                        <option value="pt">Portugués</option>
                        <option value="ru">Ruso</option>
                        <option value="ja">Japonés</option>
                        <option value="ko">Coreano</option>
                        <option value="zh">Chino</option>
                    </select>
                </div>
            </div>
            
            <div class="form-group">
                <label><i class="fas fa-brain"></i> Modelo de IA</label>
                <div class="model-info">
                    <div class="model-card" data-model="tiny">
                        <h4>Tiny</h4>
                        <p>Rápido • 39 MB</p>
                    </div>
                    <div class="model-card" data-model="base">
                        <h4>Base</h4>
                        <p>Balanceado • 74 MB</p>
                    </div>
                    <div class="model-card" data-model="small">
                        <h4>Small</h4>
                        <p>Mejor calidad • 244 MB</p>
                    </div>
                    <div class="model-card selected" data-model="medium">
                        <h4>Medium</h4>
                        <p>Muy bueno • 769 MB</p>
                    </div>
                    <div class="model-card" data-model="large-v3">
                        <h4>Large V3</h4>
                        <p>Mejor calidad • 1550 MB</p>
                    </div>
                    <div class="model-card" data-model="turbo">
                        <h4>Turbo</h4>
                        <p>Ultra rápido • 809 MB</p>
                    </div>
                </div>
                <input type="hidden" id="modelSize" value="medium">
            </div>
            
            <button type="submit" id="transcribeBtn" class="btn-primary">
                <i class="fas fa-rocket"></i> Comenzar Transcripción
            </button>
        </form>
        
        <div class="progress-section" id="progressSection">
            <div class="progress-header">
                <span class="progress-title" id="progressTitle">Procesando...</span>
                <span class="progress-percentage" id="progressPercentage">0%</span>
            </div>
            <div class="progress-bar-container">
                <div class="progress-bar" id="progressBar"></div>
            </div>
            <div class="progress-message" id="progressMessage">Iniciando proceso...</div>
        </div>
        
        <div class="result-section" id="resultSection">
            <div class="result-info">
                <div class="info-item">
                    <i class="fas fa-check-circle"></i>
                    <span>Transcripción completada</span>
                </div>
                <div class="info-item">
                    <i class="fas fa-closed-captioning"></i>
                    <span id="segmentsCount">0 segmentos</span>
                </div>
                <div class="info-item">
                    <i class="fas fa-globe"></i>
                    <span id="detectedLanguage">Idioma detectado</span>
                </div>
            </div>
            <button class="download-btn" id="downloadBtn">
                <i class="fas fa-download"></i> Descargar SRT
            </button>
        </div>
        
        <div class="log-section" id="logSection"></div>
    </div>

    <script>
        let currentTaskId = null;
        let progressInterval = null;
        let selectedModel = 'medium';
        
        // Elementos del DOM
        const form = document.getElementById('uploadForm');
        const progressSection = document.getElementById('progressSection');
        const resultSection = document.getElementById('resultSection');
        const transcribeBtn = document.getElementById('transcribeBtn');
        const logSection = document.getElementById('logSection');
        
        // Selección de modelo
        document.querySelectorAll('.model-card').forEach(card => {
            card.addEventListener('click', () => {
                document.querySelectorAll('.model-card').forEach(c => c.classList.remove('selected'));
                card.classList.add('selected');
                selectedModel = card.dataset.model;
                document.getElementById('modelSize').value = selectedModel;
            });
        });
        
        // Logging
        function log(message, type = 'info') {
            logSection.style.display = 'block';
            const entry = document.createElement('div');
            entry.className = `log-entry ${type}`;
            entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
            logSection.appendChild(entry);
            logSection.scrollTop = logSection.scrollHeight;
        }
        
        // Actualizar progreso
        function updateProgress(data) {
            const progressBar = document.getElementById('progressBar');
            const progressPercentage = document.getElementById('progressPercentage');
            const progressMessage = document.getElementById('progressMessage');
            
            progressBar.style.width = data.progress + '%';
            progressPercentage.textContent = data.progress + '%';
            progressMessage.textContent = data.message;
            
            log(data.message);
        }
        
        // Verificar progreso
        function checkProgress(taskId) {
            fetch(`/progress/${taskId}`)
                .then(response => response.json())
                .then(data => {
                    updateProgress(data);
                    
                    if (data.status === 'completed') {
                        clearInterval(progressInterval);
                        showResult(data);
                    } else if (data.status === 'error') {
                        clearInterval(progressInterval);
                        log(data.message, 'error');
                        resetForm();
                    }
                })
                .catch(error => {
                    log('Error verificando progreso: ' + error.message, 'error');
                });
        }
        
        // Mostrar resultado
        function showResult(data) {
            progressSection.style.display = 'none';
            resultSection.style.display = 'block';
            
            document.getElementById('segmentsCount').textContent = `${data.segments_count} segmentos`;
            document.getElementById('detectedLanguage').textContent = `Detectado: ${data.detected_language}`;
            
            const downloadBtn = document.getElementById('downloadBtn');
            downloadBtn.onclick = () => {
                const blob = new Blob([data.result], { type: 'text/plain' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                const filename = document.getElementById('audioFile').files[0].name.replace(/\\.[^/.]+$/, "") + '.srt';
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                log('Archivo descargado: ' + filename, 'info');
            };
            
            resetForm();
        }
        
        // Resetear formulario
        function resetForm() {
            transcribeBtn.disabled = false;
            transcribeBtn.innerHTML = '<i class="fas fa-rocket"></i> Comenzar Transcripción';
        }
        
        // Manejar envío del formulario
        form.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const file = document.getElementById('audioFile').files[0];
            if (!file) {
                log('Selecciona un archivo de audio', 'warning');
                return;
            }
            
            // Generar ID único para esta tarea
            currentTaskId = Date.now().toString();
            
            // Preparar interfaz
            transcribeBtn.disabled = true;
            transcribeBtn.innerHTML = '<div class="spinner"></div> Procesando...';
            progressSection.style.display = 'block';
            resultSection.style.display = 'none';
            logSection.innerHTML = '';
            logSection.style.display = 'block';
            
            log('Iniciando transcripción...', 'info');
            
            // Preparar datos
            const formData = new FormData();
            formData.append('audio', file);
            formData.append('words_per_line', document.getElementById('wordsPerLine').value);
            formData.append('model_size', selectedModel);
            formData.append('language', document.getElementById('language').value);
            formData.append('task_id', currentTaskId);
            
            try {
                // Enviar archivo
                const response = await fetch('/transcribe', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    throw new Error('Error en el servidor');
                }
                
                // Iniciar verificación de progreso
                progressInterval = setInterval(() => {
                    checkProgress(currentTaskId);
                }, 1000);
                
            } catch (error) {
                log('Error: ' + error.message, 'error');
                resetForm();
                progressSection.style.display = 'none';
            }
        });
        
        // Inicialización
        log('🌐 Servidor iniciado correctamente', 'info');
        log('📁 Selecciona un archivo de audio para comenzar', 'info');
    </script>
</body>
</html>'''
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html_content.encode('utf-8'))
    
    def serve_health(self):
        """Endpoint de salud"""
        response = {"status": "ok", "message": "Servidor funcionando"}
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode('utf-8'))
    
    def serve_progress(self):
        """Servir el progreso de una transcripción"""
        task_id = self.path.split('/')[-1]
        
        if task_id in progress_queue:
            progress_data = progress_queue[task_id]
        else:
            progress_data = {
                'status': 'not_found',
                'progress': 0,
                'message': 'Tarea no encontrada'
            }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(progress_data).encode('utf-8'))
    
    def handle_transcribe(self):
        """Manejar la transcripción de audio de forma asíncrona"""
        try:
            # Obtener el tamaño del contenido
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self.send_error(400, "No hay contenido")
                return
            
            # Leer los datos
            post_data = self.rfile.read(content_length)
            
            # Parsear multipart/form-data
            boundary = None
            content_type = self.headers.get('Content-Type', '')
            if 'boundary=' in content_type:
                boundary = content_type.split('boundary=')[1].encode()
            
            if not boundary:
                self.send_error(400, "Boundary no encontrado")
                return
            
            # Extraer archivo y parámetros
            parts = post_data.split(b'--' + boundary)
            
            audio_data = None
            words_per_line = 6
            model_size = 'medium'
            language = None
            task_id = str(uuid.uuid4())
            
            for part in parts:
                if b'Content-Disposition: form-data' in part:
                    if b'name="audio"' in part and b'filename=' in part:
                        header_end = part.find(b'\r\n\r\n')
                        if header_end != -1:
                            audio_data = part[header_end + 4:]
                            if audio_data.endswith(b'\r\n'):
                                audio_data = audio_data[:-2]
                    
                    elif b'name="words_per_line"' in part:
                        header_end = part.find(b'\r\n\r\n')
                        if header_end != -1:
                            value = part[header_end + 4:].decode().strip()
                            words_per_line = int(value) if value.isdigit() else 6
                    
                    elif b'name="model_size"' in part:
                        header_end = part.find(b'\r\n\r\n')
                        if header_end != -1:
                            model_size = part[header_end + 4:].decode().strip()
                    
                    elif b'name="language"' in part:
                        header_end = part.find(b'\r\n\r\n')
                        if header_end != -1:
                            lang = part[header_end + 4:].decode().strip()
                            if lang != 'auto':
                                language = lang
                    
                    elif b'name="task_id"' in part:
                        header_end = part.find(b'\r\n\r\n')
                        if header_end != -1:
                            task_id = part[header_end + 4:].decode().strip()
            
            if not audio_data:
                self.send_error(400, "Archivo de audio no encontrado")
                return
            
            print(f"📁 Procesando archivo ({len(audio_data)} bytes) - ID: {task_id}")
            print(f"⚙️ Modelo: {model_size}, Idioma: {language or 'auto'}")
            
            # Inicializar progreso
            progress_queue[task_id] = {
                'status': 'uploading',
                'progress': 10,
                'message': 'Archivo recibido, preparando...'
            }
            
            # Guardar archivo temporal
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name
            
            # Iniciar transcripción en hilo separado
            thread = threading.Thread(
                target=transcribe_audio_with_progress,
                args=(temp_path, model_size, language, words_per_line, task_id)
            )
            thread.daemon = True
            thread.start()
            
            # Limpiar archivo temporal después de un tiempo
            def cleanup_temp_file():
                time.sleep(300)  # 5 minutos
                try:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                    # Limpiar progreso después de 10 minutos
                    time.sleep(300)
                    if task_id in progress_queue:
                        del progress_queue[task_id]
                except:
                    pass
            
            cleanup_thread = threading.Thread(target=cleanup_temp_file)
            cleanup_thread.daemon = True
            cleanup_thread.start()
            
            # Responder inmediatamente con el task_id
            response = {
                'status': 'started',
                'task_id': task_id,
                'message': 'Transcripción iniciada'
            }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            self.send_error(500, f"Error interno: {str(e)}")
    
    def log_message(self, format, *args):
        """Personalizar logs del servidor"""
        print(f"🌐 {self.address_string()} - {format % args}")

def cleanup_old_tasks():
    """Limpiar tareas antiguas periódicamente"""
    while True:
        time.sleep(600)  # Cada 10 minutos
        current_time = time.time()
        tasks_to_remove = []
        
        for task_id in progress_queue:
            # Si la tarea tiene más de 30 minutos, eliminarla
            try:
                task_time = int(task_id.split('_')[0]) / 1000 if '_' in task_id else int(task_id) / 1000
                if current_time - task_time > 1800:  # 30 minutos
                    tasks_to_remove.append(task_id)
            except:
                continue
        
        for task_id in tasks_to_remove:
            del progress_queue[task_id]
            
        if tasks_to_remove:
            print(f"🧹 Limpieza: {len(tasks_to_remove)} tareas antiguas eliminadas")

def run_server():
    """Ejecutar el servidor"""
    try:
        # Iniciar hilo de limpieza
        cleanup_thread = threading.Thread(target=cleanup_old_tasks)
        cleanup_thread.daemon = True
        cleanup_thread.start()
        
        with socketserver.TCPServer(("", PORT), WhisperRequestHandler) as httpd:
            print(f"""
╔══════════════════════════════════════════════════════╗
║          🎵 WHISPER AI TRANSCRIPTOR AVANZADO         ║
╚══════════════════════════════════════════════════════╝

🌐 Servidor iniciado en: http://localhost:{PORT}
📱 Abre esa URL en tu navegador para comenzar

🚀 NUEVAS CARACTERÍSTICAS:
   ✨ Interfaz moderna y responsiva
   🧠 Modelos más potentes (Large-V3, Turbo)
   📊 Progreso en tiempo real
   🌍 Más idiomas soportados
   💾 Descarga automática de archivos
   🔄 Procesamiento asíncrono

🎯 MODELOS DISPONIBLES:
   • Tiny      - Ultra rápido (39 MB)
   • Base      - Balanceado (74 MB)
   • Small     - Buena calidad (244 MB)
   • Medium    - Muy buena calidad (769 MB)
   • Large-V3  - Máxima calidad (1550 MB) ⭐ NUEVO
   • Turbo     - Rápido y preciso (809 MB) ⭐ NUEVO

📋 FORMATOS SOPORTADOS:
   • Audio: MP3, WAV, M4A, FLAC, OGG
   • Video: MP4, AVI, MOV, MKV (extrae audio)

🛠️  ENDPOINTS:
   • /          - Interfaz web
   • /transcribe - Transcripción de audio
   • /progress/[id] - Estado del progreso
   • /health    - Estado del servidor

⏹️  Para detener: Ctrl+C
            """)
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print(f"\n⏹️  Servidor detenido por el usuario")
        print("🧹 Limpiando recursos...")
        # Limpiar cache de modelos si es necesario
        model_cache.clear()
        progress_queue.clear()
        print("✅ Limpieza completada")
        
    except Exception as e:
        print(f"❌ Error iniciando servidor: {e}")
        print("💡 Posibles soluciones:")
        print(f"   • Verificar que el puerto {PORT} esté libre")
        print("   • Ejecutar con permisos de administrador")
        print("   • Instalar dependencias: pip install openai-whisper")

if __name__ == "__main__":
    # Verificar que Whisper esté instalado
    try:
        import whisper
        print("✅ OpenAI Whisper detectado")
    except ImportError:
        print("❌ OpenAI Whisper no está instalado")
        print("📦 Instalar con: pip install openai-whisper")
        print("🔗 Más info: https://github.com/openai/whisper")
        exit(1)
    
    run_server()