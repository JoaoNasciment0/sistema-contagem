import os
import sqlite3
from flask import Flask, render_template, request, redirect
import torch
from PIL import Image, ImageDraw

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

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Verificar se um arquivo foi enviado
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        # Salvar a imagem enviada
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(img_path)

        # Carregar a imagem e realizar a predição
        img = Image.open(img_path)
        results = model([img])
        detections = results.xyxy[0].cpu().numpy()  # Obter os resultados
        filtered_detections = [d for d in detections if d[4] >= 0.85]  # Filtrar predições com confiança >= 0.85

        # Adicionar caixas e rótulos na imagem
        img_with_boxes = img.copy()
        draw = ImageDraw.Draw(img_with_boxes)
        for det in filtered_detections:
            x1, y1, x2, y2, conf, cls = det
            label = f"{results.names[int(cls)]} {conf:.2f}"
            draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)
            draw.text((x1, y1), label, fill="blue")

        # Salvar a imagem processada
        img_output_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output.jpg')
        img_with_boxes.save(img_output_path)

        # Contar o número de big bags detectados
        bigbag_count = int(sum(det[5] == 0 for det in filtered_detections))  # Garantir que seja um número inteiro

        # Salvar os dados no banco de dados
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO detections (image_name, image_path, count)
            VALUES (?, ?, ?)
        ''', (file.filename, img_path, bigbag_count))
        conn.commit()
        conn.close()

        # Retornar os resultados ao usuário
        return render_template('result.html', img_url=img_output_path, bigbag_count=bigbag_count)

    except Exception as e:
        print(f"Erro durante a predição: {e}")
        return "Ocorreu um erro ao processar a imagem. Verifique o console para mais detalhes."

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
    app.run(debug=True)
