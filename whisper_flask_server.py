#!/usr/bin/env python3
"""
Servidor web simple para Whisper usando solo bibliotecas estándar de Python
No requiere Flask, solo la biblioteca http.server incluida en Python
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

# Puerto del servidor
PORT = 8000

# Cache de modelos Whisper
model_cache = {}

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

class WhisperRequestHandler(http.server.BaseHTTPRequestHandler):
    
    def do_GET(self):
        """Manejar peticiones GET"""
        if self.path == '/':
            self.serve_html()
        elif self.path == '/health':
            self.serve_health()
        else:
            self.send_error(404, "Página no encontrada")
    
    def do_POST(self):
        """Manejar peticiones POST"""
        if self.path == '/transcribe':
            self.handle_transcribe()
        else:
            self.send_error(404, "Endpoint no encontrado")
    
    def serve_html(self):
        """Servir la página HTML principal"""
        html_content = '''<!DOCTYPE html>
<html>
<head>
    <title>Whisper Simple Server</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .container { background: #f5f5f5; padding: 20px; border-radius: 8px; }
        input, select, button { padding: 10px; margin: 5px; }
        button { background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        .progress { width: 100%; height: 20px; background: #ddd; border-radius: 10px; overflow: hidden; margin: 10px 0; }
        .progress-bar { height: 100%; background: #28a745; width: 0%; transition: width 0.3s ease; }
        .log { background: #222; color: #0f0; padding: 10px; border-radius: 4px; font-family: monospace; font-size: 12px; max-height: 200px; overflow-y: auto; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎵 Whisper Transcriptor</h1>
        <p>Servidor simple para transcripción de audio</p>
        
        <form id="uploadForm" enctype="multipart/form-data">
            <div>
                <label>Archivo de Audio:</label>
                <input type="file" id="audioFile" accept="audio/*" required>
            </div>
            
            <div>
                <label>Palabras por línea:</label>
                <input type="number" id="wordsPerLine" value="5" min="1" max="20">
            </div>
            
            <div>
                <label>Modelo:</label>
                <select id="modelSize">
                    <option value="tiny">Tiny (rápido)</option>
                    <option value="base" selected>Base (balanceado)</option>
                    <option value="small">Small (mejor)</option>
                    <option value="medium">Medium (muy bueno)</option>
                    <option value="large">Large (mejor calidad)</option>
                </select>
            </div>
            
            <div>
                <label>Idioma:</label>
                <select id="language">
                    <option value="auto">Auto-detectar</option>
                    <option value="es">Español</option>
                    <option value="en">Inglés</option>
                    <option value="fr">Francés</option>
                </select>
            </div>
            
            <button type="submit" id="transcribeBtn">🚀 Transcribir</button>
        </form>
        
        <div class="progress" id="progressContainer" style="display: none;">
            <div class="progress-bar" id="progressBar"></div>
        </div>
        <div id="progressText"></div>
        
        <div class="log" id="logArea" style="display: none;"></div>
    </div>

    <script>
        const form = document.getElementById('uploadForm');
        const progressContainer = document.getElementById('progressContainer');
        const progressBar = document.getElementById('progressBar');
        const progressText = document.getElementById('progressText');
        const logArea = document.getElementById('logArea');
        const transcribeBtn = document.getElementById('transcribeBtn');

        function log(message) {
            logArea.style.display = 'block';
            logArea.innerHTML += new Date().toLocaleTimeString() + ' - ' + message + '\\n';
            logArea.scrollTop = logArea.scrollHeight;
        }

        function updateProgress(percent, text) {
            progressContainer.style.display = 'block';
            progressBar.style.width = percent + '%';
            progressText.textContent = text;
        }

        form.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const file = document.getElementById('audioFile').files[0];
            if (!file) {
                alert('Selecciona un archivo');
                return;
            }

            transcribeBtn.disabled = true;
            transcribeBtn.textContent = '⏳ Procesando...';
            logArea.innerHTML = '';

            const formData = new FormData();
            formData.append('audio', file);
            formData.append('words_per_line', document.getElementById('wordsPerLine').value);
            formData.append('model_size', document.getElementById('modelSize').value);
            formData.append('language', document.getElementById('language').value);

            try {
                log('Enviando archivo...');
                updateProgress(10, 'Enviando archivo...');

                const response = await fetch('/transcribe', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error('Error en el servidor');
                }

                updateProgress(90, 'Descargando resultado...');
                log('Descargando resultado...');

                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = file.name.replace(/\\.[^/.]+$/, "") + '.srt';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);

                updateProgress(100, '¡Completado!');
                log('✅ Archivo descargado correctamente');

            } catch (error) {
                log('❌ Error: ' + error.message);
                updateProgress(0, 'Error');
            } finally {
                transcribeBtn.disabled = false;
                transcribeBtn.textContent = '🚀 Transcribir';
            }
        });

        // Verificar estado del servidor
        log('🌐 Servidor iniciado correctamente');
        log('📁 Selecciona un archivo de audio para comenzar');
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
    
    def handle_transcribe(self):
        """Manejar la transcripción de audio"""
        try:
            # Obtener el tamaño del contenido
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self.send_error(400, "No hay contenido")
                return
            
            # Leer los datos
            post_data = self.rfile.read(content_length)
            
            # Parsear multipart/form-data manualmente (simplificado)
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
            words_per_line = 5
            model_size = 'base'
            language = None
            
            for part in parts:
                if b'Content-Disposition: form-data' in part:
                    if b'name="audio"' in part and b'filename=' in part:
                        # Extraer archivo de audio
                        header_end = part.find(b'\r\n\r\n')
                        if header_end != -1:
                            audio_data = part[header_end + 4:]
                            # Remover el final si existe
                            if audio_data.endswith(b'\r\n'):
                                audio_data = audio_data[:-2]
                    
                    elif b'name="words_per_line"' in part:
                        header_end = part.find(b'\r\n\r\n')
                        if header_end != -1:
                            value = part[header_end + 4:].decode().strip()
                            words_per_line = int(value) if value.isdigit() else 5
                    
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
            
            if not audio_data:
                self.send_error(400, "Archivo de audio no encontrado")
                return
            
            print(f"📁 Procesando archivo de audio ({len(audio_data)} bytes)")
            print(f"⚙️ Configuración: {words_per_line} palabras/línea, modelo {model_size}")
            
            # Guardar archivo temporal
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name
            
            try:
                # Cargar modelo y transcribir
                print(f"🤖 Cargando modelo {model_size}...")
                model = get_model(model_size)
                
                print("🎙️ Iniciando transcripción...")
                result = model.transcribe(
                    temp_path,
                    language=language,
                    word_timestamps=True,
                    verbose=False
                )
                
                print("📄 Generando archivo SRT...")
                srt_content = create_srt_content(result['segments'], words_per_line)
                
                # Enviar respuesta
                self.send_response(200)
                self.send_header('Content-Type', 'text/plain; charset=utf-8')
                self.send_header('Content-Disposition', 'attachment; filename="transcription.srt"')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(srt_content.encode('utf-8'))
                
                print(f"✅ Transcripción completada - {len(result['segments'])} segmentos")
                
            finally:
                # Limpiar archivo temporal
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            self.send_error(500, f"Error interno: {str(e)}")
    
    def log_message(self, format, *args):
        """Personalizar logs del servidor"""
        print(f"🌐 {self.address_string()} - {format % args}")

def run_server():
    """Ejecutar el servidor"""
    try:
        with socketserver.TCPServer(("", PORT), WhisperRequestHandler) as httpd:
            print(f"""
╔══════════════════════════════════════╗
║     🎵 WHISPER SIMPLE SERVER        ║
╚══════════════════════════════════════╝

🌐 Servidor iniciado en: http://localhost:{PORT}
📱 Abre esa URL en tu navegador

🔧 Funcionalidades:
   - Transcripción de audio a SRT
   - Interfaz web integrada
   - Sin dependencias extra (solo Whisper)
   
🎯 Para usar:
   1. Abre http://localhost:{PORT}
   2. Selecciona tu archivo de audio
   3. Configura opciones
   4. ¡Transcribe!

⏹️  Para detener: Ctrl+C
            """)
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n⏹️  Servidor detenido por el usuario")
    except Exception as e:
        print(f"❌ Error iniciando servidor: {e}")

if __name__ == "__main__":
    run_server()