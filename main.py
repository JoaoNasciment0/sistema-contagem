import os
import sqlite3
from flask import Flask, render_template, request, redirect, Response, jsonify
import torch
from PIL import Image, ImageDraw
import base64
import numpy as np
import cv2

# Inicializar o Flask
app = Flask(__name__)

# Configurações de diretórios
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Carregar o modelo YOLOv5 treinado
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp/weights/best.pt', force_reload=True)

# Função para inicializar o banco de dados
def init_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT NOT NULL,
            image_path TEXT NOT NULL,
            count INTEGER NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Inicializar o banco de dados
init_db()

@app.route('/')
def index():
    return render_template('index.html')

# Função para gerar o feed de vídeo
def generate_video_feed():
    # Abrir a câmera
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Converter para RGB (OpenCV usa BGR por padrão)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)

        # Realizar a predição
        results = model([pil_img])
        detections = results.xyxy[0].cpu().numpy()  # Obter os resultados
        filtered_detections = [d for d in detections if d[4] >= 0.85]  # Filtrar predições com confiança >= 0.85

        # Desenhar as caixas na imagem
        for det in filtered_detections:
            x1, y1, x2, y2, conf, cls = det
            try:
                label = f"{results.names[int(cls)]} {conf:.2f}"
            except IndexError:
                label = f"Classe {int(cls)} {conf:.2f}"  # Se a classe não for encontrada, apenas exibe o número da classe

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Converter para JPEG e enviar via Response
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = jpeg.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict_camera', methods=['POST'])
def predict_camera():
    try:
        # Receber a imagem em base64
        data = request.get_json()
        image_data = data['image']
        
        # Converter base64 para imagem
        img_data = base64.b64decode(image_data.split(',')[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Converter para RGB (OpenCV usa BGR por padrão)
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Realizar a predição
        results = model([pil_img])
        detections = results.xyxy[0].cpu().numpy()  # Obter os resultados
        filtered_detections = [d for d in detections if d[4] >= 0.85]  # Filtrar predições com confiança >= 0.85

        # Contar e coletar as coordenadas dos BigBags detectados (classe 0, BigBag)
        bigbags = []
        bigbag_count = 0
        for det in filtered_detections:
            cls = int(det[5])
            if cls == 0:  # Classe 0 é BigBag
                bigbag_count += 1
                # Extrair as coordenadas dos BigBags detectados
                x1, y1, x2, y2, _, _ = det
                bigbags.append({
                    "x": int(x1),
                    "y": int(y1),
                    "width": int(x2 - x1),
                    "height": int(y2 - y1)
                })
        
        # Retornar a contagem e as coordenadas dos BigBags
        return jsonify({
            "status": "success", 
            "count": bigbag_count, 
            "bigbags": bigbags  # Incluindo as coordenadas dos BigBags
        })

    except Exception as e:
        print(f"Erro durante a predição: {e}")
        return jsonify({"status": "error", "message": str(e)})


@app.route('/history')
def history():
    try:
        # Recuperar os dados do banco de dados
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id, image_name, count FROM detections')
        records = cursor.fetchall()
        conn.close()

        return render_template('history.html', records=records)

    except Exception as e:
        print(f"Erro ao acessar o histórico: {e}")
        return "Ocorreu um erro ao acessar o histórico. Verifique o console para mais detalhes."

if __name__ == "__main__":
    port = int(os.getenv('PORT'), '5000')
    app.run(host='0.0.0.0', port=5000, debug=True)

