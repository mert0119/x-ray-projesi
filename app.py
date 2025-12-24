"""
MedScan AI - Flask Web Sunucusu
Tıbbi görüntü analiz sistemi için web arayüzü.
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import cv2
import base64
import numpy as np
from werkzeug.utils import secure_filename
from analyzers import MedicalReport
from treatments import get_treatment_recommendation

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

medical_report = MedicalReport()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'dcm'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def cv2_to_base64(img):
    """OpenCV görüntüsünü base64'e çevir"""
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')


@app.route('/')
def index():
    """Ana sayfa"""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """Görüntü analizi yap"""
    if 'image' not in request.files:
        return jsonify({'error': 'Görüntü dosyası bulunamadı'}), 400
    
    file = request.files['image']
    analysis_type = request.form.get('type', 'auto')
    
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Desteklenmeyen dosya formatı'}), 400
    
    try:
        import time
        timestamp = int(time.time() * 1000)
        original_filename = secure_filename(file.filename)
        name, ext = os.path.splitext(original_filename)
        filename = f"{name}_{timestamp}{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        
        result, detected_type = medical_report.analyze(filepath, analysis_type)
        verdict, severity = medical_report.get_verdict(result['score'])
        
        treatment = get_treatment_recommendation(detected_type, result['score'], result['findings'])
        
        response = {
            'success': True,
            'analysis_type': detected_type,
            'score': result['score'],
            'verdict': verdict,
            'severity': severity,
            'findings': result['findings'],
            'images': {
                'overlay': cv2_to_base64(result['overlay']),
                'heatmap': cv2_to_base64(result['heatmap']),
                'marked': cv2_to_base64(result['marked'])
            },
            'treatment': treatment
        }
        
        if 'suspicious_areas' in result:
            response['suspicious_areas'] = result['suspicious_areas']
        if 'tumor_candidates' in result:
            response['tumor_candidates'] = result['tumor_candidates']
        if 'fracture_lines' in result:
            response['fracture_lines'] = result['fracture_lines']
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Yüklenen dosyaları serve et"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    print("\n" + "="*50)
    print("🏥 MedScan AI - Tıbbi Görüntü Analiz Sistemi")
    print("="*50)
    print("\n🌐 Tarayıcıda açın: http://localhost:5000")
    print("\n⚠️  Bu sistem sadece eğitim amaçlıdır.")
    print("    Gerçek teşhis için doktora başvurun.")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)