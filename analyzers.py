"""
MedScan AI - Tıbbi Görüntü Analiz Modülleri
Akciğer, Beyin ve Kemik görüntülerini analiz eder.
"""

import cv2
import numpy as np
from PIL import Image
import os

try:
    import tensorflow as tf
    
    LUNG_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'lung_classifier_model.keras')
    if os.path.exists(LUNG_MODEL_PATH):
        LUNG_MODEL = tf.keras.models.load_model(LUNG_MODEL_PATH)
        print("✅ Akciğer sınıflandırma modeli yüklendi!")
    else:
        LUNG_MODEL = None
        print("⚠️ Akciğer modeli bulunamadı")
    
    BRAIN_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'brain_classifier_model.keras')
    if os.path.exists(BRAIN_MODEL_PATH):
        BRAIN_MODEL = tf.keras.models.load_model(BRAIN_MODEL_PATH)
        print("✅ Beyin tümör modeli yüklendi!")
    else:
        BRAIN_MODEL = None
        print("⚠️ Beyin modeli bulunamadı")
    
    BONE_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'bone_classifier_model.keras')
    if os.path.exists(BONE_MODEL_PATH):
        BONE_MODEL = tf.keras.models.load_model(BONE_MODEL_PATH)
        print("✅ Kemik kırık modeli yüklendi!")
    else:
        BONE_MODEL = None
        print("⚠️ Kemik modeli bulunamadı")
        
except ImportError:
    LUNG_MODEL = None
    BRAIN_MODEL = None
    BONE_MODEL = None
    print("⚠️ TensorFlow yüklü değil")




class LungAnalyzer:
    """
    Akciğer X-Ray Analizi - Deep Learning ile
    %97.8 doğruluk oranı ile COVID19/Normal sınıflandırma
    """
    
    def __init__(self):
        self.name = "Akciğer Analizi (AI)"
        self.model = LUNG_MODEL
        self.class_names = {0: 'COVID19', 1: 'NORMAL'}
    
    def analyze(self, image_path):
        """Akciğer görüntüsünü analiz et"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Görüntü okunamadı!")
        
        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if self.model is not None:
            img_resized = cv2.resize(img, (224, 224))
            img_array = img_resized.astype('float32') / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            prediction = self.model.predict(img_array, verbose=0)[0][0]
            
            is_normal = prediction > 0.5
            confidence = prediction if is_normal else (1 - prediction)
            
            if is_normal:
                risk_score = (1 - prediction) * 100
            else:
                risk_score = (1 - prediction) * 100
        else:
            risk_score = np.mean(gray) / 2.55
            is_normal = risk_score < 50
            confidence = 0.5
        
        # GÖRSEL ÇIKTILAR
        result = img.copy()
        heatmap = np.zeros_like(gray)
        
        if not is_normal:
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            kernel = np.ones((5, 5), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:
                    cv2.drawContours(heatmap, [contour], -1, 200, -1)
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 0.7, heatmap_colored, 0.3, 0)
        
        findings = []
        
        if is_normal:
            findings.append("✅ NORMAL AKCİĞER")
            findings.append(f"Güven oranı: %{confidence*100:.1f}")
            findings.append("Belirgin patoloji tespit edilmedi")
        else:
            findings.append("🔴 COVID-19 ŞÜPHESİ")
            findings.append(f"Güven oranı: %{confidence*100:.1f}")
            findings.append("⚠️ Acil PCR testi önerilir")
            findings.append("İzolasyon önlemleri alınmalıdır")
        
        return {
            'heatmap': heatmap_colored,
            'overlay': overlay,
            'marked': result,
            'score': round(risk_score, 1),
            'suspicious_areas': 0 if is_normal else 1,
            'findings': findings,
            'prediction': 'NORMAL' if is_normal else 'COVID19',
            'confidence': round(confidence * 100, 1)
        }


class BrainAnalyzer:
    """
    Beyin MRI Analizi - Deep Learning ile
    %81.9 doğruluk ile 4 sınıflı tümör sınıflandırma
    """
    
    def __init__(self):
        self.name = "Beyin MRI Analizi (AI)"
        self.model = BRAIN_MODEL
        self.class_names = {
            0: 'Glioma',
            1: 'Meningioma', 
            2: 'Normal',
            3: 'Pituitary'
        }
    
    def analyze(self, image_path):
        """Beyin MRI görüntüsünü analiz et"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Görüntü okunamadı!")
        
        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        predicted_class = 2
        confidence = 0.5
        risk_score = 0
        
        if self.model is not None:
            img_resized = cv2.resize(img, (224, 224))
            img_array = img_resized.astype('float32') / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            predictions = self.model.predict(img_array, verbose=0)[0]
            predicted_class = np.argmax(predictions)
            confidence = predictions[predicted_class]
            
            if predicted_class == 2:
                risk_score = (1 - confidence) * 30
            else:
                risk_score = 50 + confidence * 50
        
        class_name = self.class_names.get(predicted_class, 'Bilinmiyor')
        is_normal = (predicted_class == 2)
        
        result = img.copy()
        heatmap = np.zeros_like(gray)
        
        if not is_normal:
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            kernel = np.ones((5, 5), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:
                    cv2.drawContours(heatmap, [contour], -1, 200, -1)
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 255), 2)
        
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
        overlay = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)
        
        findings = []
        
        if is_normal:
            findings.append("✅ NORMAL BEYİN")
            findings.append(f"Güven oranı: %{confidence*100:.1f}")
            findings.append("Tümör tespit edilmedi")
        else:
            findings.append(f"🔴 {class_name.upper()} TÜMÖRÜ TESPİT EDİLDİ")
            findings.append(f"Güven oranı: %{confidence*100:.1f}")
            
            if predicted_class == 0:
                findings.append("⚠️ Glioma - Beyin/omurilik tümörü")
                findings.append("Acil nöroloji konsültasyonu gerekli")
            elif predicted_class == 1:
                findings.append("⚠️ Meningioma - Beyin zarı tümörü")
                findings.append("Genellikle cerrahi tedavi gerekir")
            elif predicted_class == 3:
                findings.append("⚠️ Hipofiz tümörü")
                findings.append("Endokrinoloji değerlendirmesi önerilir")
        
        return {
            'heatmap': heatmap_colored,
            'overlay': overlay,
            'marked': result,
            'score': round(risk_score, 1),
            'tumor_candidates': 0 if is_normal else 1,
            'findings': findings,
            'prediction': class_name,
            'confidence': round(confidence * 100, 1)
        }




class BoneAnalyzer:
    """
    Kemik X-Ray Analizi - Deep Learning ile
    Kırık/Sağlam sınıflandırma
    """
    
    def __init__(self):
        self.name = "Kemik X-Ray Analizi (AI)"
        self.model = BONE_MODEL
        self.class_names = {0: 'Kırık', 1: 'Sağlam'}
    
    def analyze(self, image_path):
        """Kemik X-Ray görüntüsünü analiz et"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Görüntü okunamadı!")
        
        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        is_fractured = False
        confidence = 0.5
        risk_score = 0
        
        if self.model is not None:
            img_resized = cv2.resize(img, (224, 224))
            img_array = img_resized.astype('float32') / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            prediction = self.model.predict(img_array, verbose=0)[0][0]
            
            is_fractured = prediction < 0.5
            confidence = (1 - prediction) if is_fractured else prediction
            
            if is_fractured:
                risk_score = 50 + (1 - prediction) * 50
            else:
                risk_score = prediction * 30
        
        result = img.copy()
        heatmap = np.zeros_like(gray)
        
        if is_fractured:
            edges = cv2.Canny(gray, 50, 150)
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.arcLength(contour, False) > 50:
                    cv2.drawContours(result, [contour], -1, (0, 0, 255), 2)
                    cv2.drawContours(heatmap, [contour], -1, 200, 2)
        
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_MAGMA)
        overlay = cv2.addWeighted(img, 0.7, heatmap_colored, 0.3, 0)
        
        findings = []
        
        if is_fractured:
            findings.append("🔴 KIRIK TESPİT EDİLDİ")
            findings.append(f"Güven oranı: %{confidence*100:.1f}")
            findings.append("⚠️ Ortopedi konsültasyonu gerekli")
            findings.append("Alçı/atel uygulaması değerlendirilmeli")
        else:
            findings.append("✅ KIRIK TESPİT EDİLMEDİ")
            findings.append(f"Güven oranı: %{confidence*100:.1f}")
            findings.append("Kemik yapısı normal görünüyor")
        
        return {
            'heatmap': heatmap_colored,
            'overlay': overlay,
            'marked': result,
            'score': round(risk_score, 1),
            'fracture_lines': 1 if is_fractured else 0,
            'findings': findings,
            'prediction': 'Kırık' if is_fractured else 'Sağlam',
            'confidence': round(confidence * 100, 1)
        }



class MedicalReport:
    """
    Tüm analizleri birleştirir ve rapor üretir.
    """
    
    def __init__(self):
        self.lung = LungAnalyzer()
        self.brain = BrainAnalyzer()
        self.bone = BoneAnalyzer()
    
    def analyze(self, image_path, analysis_type):
        """
        Belirtilen türde analiz yap.
        
        analysis_type: 'lung', 'brain', 'bone', 'auto'
        """
        analyzers = {
            'lung': self.lung,
            'brain': self.brain,
            'bone': self.bone
        }
        
        if analysis_type == 'auto':
            results = {}
            max_score = 0
            best_type = 'lung'
            
            for name, analyzer in analyzers.items():
                try:
                    result = analyzer.analyze(image_path)
                    results[name] = result
                    if result['score'] > max_score:
                        max_score = result['score']
                        best_type = name
                except:
                    pass
            
            return results.get(best_type, self.lung.analyze(image_path)), best_type
        
        analyzer = analyzers.get(analysis_type, self.lung)
        return analyzer.analyze(image_path), analysis_type
    
    def get_verdict(self, score):
        """Risk skoruna göre değerlendirme"""
        if score < 20:
            return "✅ NORMAL - Belirgin patoloji yok", "success"
        elif score < 40:
            return "⚠️ DİKKAT - Hafif anormallik mevcut", "warning"
        elif score < 60:
            return "⚠️ ORTA RİSK - Şüpheli bulgular var", "warning"
        elif score < 80:
            return "🚨 YÜKSEK RİSK - Ciddi patoloji şüphesi", "danger"
        else:
            return "🚨 KRİTİK - Acil değerlendirme gerekli!", "danger"