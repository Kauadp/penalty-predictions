#!/usr/bin/env python3
"""
Script de Inferência em Tempo Real - Predição de Pênaltis
=========================================================

Detecta jogador, extrai pose e prediz direção do chute em tempo real.

Uso:
    python predict_live.py --video path/to/video.mp4
    python predict_live.py --webcam  # Usar webcam
    python predict_live.py --video x.mp4 --debug  # Ver predições no terminal

Se o texto da predição não aparecer na janela (erro de fonte do Qt):
    pip install Pillow   # texto é desenhado com PIL e colado no frame
    Ou: QT_QPA_FONTDIR=/usr/share/fonts python predict_live.py --video x.mp4
"""

import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
import warnings

# Ajudar Qt a achar fontes (evita "Cannot find font directory" e texto que não aparece)
for _path in (
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/TTF",
    "/usr/share/fonts/truetype",
    "/usr/share/fonts",
):
    if os.path.isdir(_path):
        os.environ.setdefault("QT_QPA_FONTDIR", _path)
        break

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
    module="sklearn",
)
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
import joblib
import argparse
from pathlib import Path
from collections import deque
import time


class PenaltyPredictor:
    """Preditor de pênaltis em tempo real"""
    
    def __init__(self,
                 yolo_model="models/yolov8s.pt",
                 pose_model="models/pose_landmarker_heavy.task",
                 mlp_model="models/mlp_best_model.pkl",
                 scaler_model="models/scaler.pkl",
                 label_encoder="models/label_encoder.pkl",
                 debug=False):
        self.debug = bool(debug)
        self._predict_count = 0
        print("🚀 Inicializando Penalty Predictor...")
        
        # Carregar YOLO
        print(f"   📦 Carregando YOLO: {yolo_model}")
        self.yolo = YOLO(yolo_model)
        
        # Carregar MediaPipe Pose
        print(f"   📦 Carregando MediaPipe: {pose_model}")
        base_options = python.BaseOptions(model_asset_path=pose_model)
        self.pose_options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=0.3,
            min_pose_presence_confidence=0.3,
            num_poses=1
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(self.pose_options)
        
        # Carregar modelo MLP
        print(f"   📦 Carregando MLP: {mlp_model}")
        self.model = joblib.load(mlp_model)
        self.scaler = joblib.load(scaler_model)
        self.label_encoder = joblib.load(label_encoder)
        
        # Buffer para suavização temporal
        self.prediction_buffer = deque(maxlen=5)  # Últimas 5 predições
        
        # Tracking
        self.last_person_box = None
        # Buffer para velocidade das mãos (frame anterior)
        self._last_wrist_right = None  # (x, y, z)
        self._last_wrist_left = None
        # Última predição exibida (para não piscar quando pose falha 1 frame)
        classes = list(self.label_encoder.classes_)
        self._last_smoothed_proba = {c: 1.0 / len(classes) for c in classes}
        self._last_smoothed_label = classes[0]

        self.prob_history = []
        self.decision_made = False
        self.final_decision = None
        self.final_confidence = None
        self.decision_frame = None
        self.video_fps = None

        self.WINDOW = 15       # últimos 15 frames (~0.5s em 30fps)
        self.MIN_FRAMES = 15    # não decide antes disso
        self.THRESHOLD = 0.75   # confiança média mínima




        # Cores por classe
        self.colors = {
            'center': (0, 255, 255),    # Amarelo
            'left': (0, 165, 255),      # Laranja
            'right': (255, 0, 255)      # Magenta
        }
        
        print("✅ Inicialização completa!\n")
    
    def detect_person(self, frame, imgsz=320):
        """Detecta pessoa no frame com YOLO (imgsz menor = mais rápido)."""
        results = self.yolo(frame, imgsz=imgsz, verbose=False)[0]
        
        persons = []
        for box in results.boxes:
            if int(box.cls[0]) == 0:  # Classe pessoa
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                persons.append((x1, y1, x2, y2, confidence))
        
        if not persons:
            return None
        
        # Pegar pessoa mais confiável e maior
        best_person = max(persons, key=lambda p: p[4] * (p[2]-p[0]) * (p[3]-p[1]))
        return best_person[:4]  # (x1, y1, x2, y2)
    
    def extract_pose(self, frame, person_box):
        """Extrai pose igual get_data.py: crop da box, resize 256x256, sem padding."""
        x1, y1, x2, y2 = person_box
        h_frame, w_frame = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_frame, x2), min(h_frame, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None, None
        crop = cv2.resize(crop, (256, 256))
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # MediaPipe
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=crop_rgb
        )
        
        result = self.landmarker.detect(mp_image)
        
        if not result.pose_landmarks:
            return None, None
        
        landmarks = result.pose_landmarks[0]
        
        return landmarks, (x1, y1, x2, y2, crop.shape[1], crop.shape[0])
    
    def coords_norm(self, landmark_list):
        """Normalização idêntica a get_data.py: quadris 23/24 como referência."""
        x0 = (landmark_list[23].x + landmark_list[24].x) / 2
        y0 = (landmark_list[23].y + landmark_list[24].y) / 2
        z0 = (landmark_list[23].z + landmark_list[24].z) / 2
        scale = np.sqrt(
            (landmark_list[23].x - landmark_list[24].x) ** 2
            + (landmark_list[23].y - landmark_list[24].y) ** 2
            + (landmark_list[23].z - landmark_list[24].z) ** 2
        )
        if scale == 0:
            scale = 1e-6
        normalized = []
        for lm in landmark_list:
            normalized.append((lm.x - x0) / scale)
            normalized.append((lm.y - y0) / scale)
            normalized.append((lm.z - z0) / scale)
        return normalized

    def _create_features_from_coords(self, coords):
        """
        Features engineered a partir das coords normalizadas (f_0..f_98),
        igual ao modeling.ipynb criar_features_essenciais.
        coords: list de 99 valores (lm0.x, lm0.y, lm0.z, lm1.x, ...)
        """
        # Índices MediaPipe: 0=nose, 2=left_eye, 5=right_eye, 11=left_shoulder,
        # 12=right_shoulder, 15=left_wrist, 16=right_wrist, 24=right_hip, 26=right_knee, 28=right_ankle
        def pt(i):
            return np.array([coords[i * 3], coords[i * 3 + 1], coords[i * 3 + 2]])

        features = {}

        # 1. Assimetria das mãos (right_wrist - left_wrist)
        features["assimetria_maos_z"] = pt(16)[2] - pt(15)[2]
        features["assimetria_maos_x"] = pt(16)[0] - pt(15)[0]

        # 2. Posição média do corpo (média de todos os landmarks)
        all_z = [coords[i * 3 + 2] for i in range(33)]
        all_y = [coords[i * 3 + 1] for i in range(33)]
        features["corpo_z_medio"] = np.mean(all_z)
        features["corpo_y_medio"] = np.mean(all_y)

        # 3. Variância
        features["variancia_z"] = np.std(all_z)
        features["variancia_y"] = np.std(all_y)

        # 4. Direção do olhar (nose - centro dos olhos)
        eye_center_z = (pt(2)[2] + pt(5)[2]) / 2
        eye_center_x = (pt(2)[0] + pt(5)[0]) / 2
        features["direcao_olhar_z"] = pt(0)[2] - eye_center_z
        features["direcao_olhar_x"] = pt(0)[0] - eye_center_x

        # 5. Rotação do tronco (ombros)
        features["rotacao_tronco_z"] = pt(12)[2] - pt(11)[2]
        features["rotacao_tronco_x"] = pt(12)[0] - pt(11)[0]

        # 6. Velocidade das mãos (em coords normalizadas; 0 se não houver frame anterior)
        wr = pt(16)
        wl = pt(15)
        if self._last_wrist_right is not None:
            vel_r = float(np.linalg.norm(wr - self._last_wrist_right))
        else:
            vel_r = 0.0
        if self._last_wrist_left is not None:
            vel_l = float(np.linalg.norm(wl - self._last_wrist_left))
        else:
            vel_l = 0.0
        features["velocidade_mao_direita"] = vel_r
        features["velocidade_mao_esquerda"] = vel_l
        self._last_wrist_right = wr.copy()
        self._last_wrist_left = wl.copy()

        # 7. Ângulo joelho direito (hip 24 - knee 26 - ankle 28), em coords normalizadas
        p1, p2, p3 = pt(24), pt(26), pt(28)
        v1, v2 = p1 - p2, p3 - p2
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            features["angulo_joelho_direito"] = 0.0
        else:
            cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            features["angulo_joelho_direito"] = float(np.arccos(cos_a))

        return features
    
    def predict(self, landmarks):
        """Prediz direção do chute (pipeline igual get_data + modeling)."""
        # 1. Normalizar igual get_data (coords_norm) -> f_0..f_98
        coords = self.coords_norm(landmarks)
        # 2. Features engineered a partir das coords normalizadas (igual modeling)
        features_dict = self._create_features_from_coords(coords)
        features_eng = list(features_dict.values())
        # Ordem do treino: 13 engineered primeiro, depois 99 coords (total 112)
        our_names = list(features_dict.keys()) + [f"f_{i}" for i in range(99)]
        X = np.array(features_eng + coords, dtype=np.float64).reshape(1, -1)
        if hasattr(self.scaler, "feature_names_in_") and self.scaler.feature_names_in_ is not None:
            col_names = self.scaler.feature_names_in_
            if list(col_names) != our_names:
                idx = [our_names.index(c) for c in col_names]
                X = X[:, idx]
            X_df = pd.DataFrame(X, columns=col_names)
        else:
            X_df = pd.DataFrame(X, columns=our_names)

        X_scaled = self.scaler.transform(X_df)

        proba = self.model.predict_proba(X_scaled)[0]
        pred_class = self.model.predict(X_scaled)[0]
        pred_label = self.label_encoder.inverse_transform([pred_class])[0]

        proba_dict = {
            label: prob
            for label, prob in zip(self.label_encoder.classes_, proba)
        }

        self.prob_history.append(proba_dict)

        # manter apenas os últimos WINDOW frames
        if len(self.prob_history) > self.WINDOW:
            self.prob_history.pop(0)

        total_frames = len(self.prob_history)

        # só começa a considerar decisão após mínimo de frames
        if total_frames >= self.MIN_FRAMES and not self.decision_made:

            mean_probs = {
                label: np.mean([frame[label] for frame in self.prob_history])
                for label in proba_dict.keys()
            }

            best_label = max(mean_probs, key=mean_probs.get)

            if mean_probs[best_label] > self.THRESHOLD:
                self.final_decision = best_label
                self.final_confidence = mean_probs[best_label]   # <<< ADICIONE ISSO
                self.decision_made = True
                self.decision_frame = self.global_frame_count


                print(f"\n🎯 DECISÃO FINAL: {self.final_decision.upper()} "
                    f"(mean_recent={mean_probs[best_label]:.2%})\n")


        
        if self.debug:
            self._predict_count += 1
            probs_str = " | ".join(f"{k}: {v:.2%}" for k, v in proba_dict.items())
            print(f"[predict #{self._predict_count}] >>> {pred_label.upper()} <<<  ({probs_str})")


        return pred_label, proba_dict
    
    def smooth_prediction(self, proba_dict):
        """Suaviza predições usando buffer temporal"""
        self.prediction_buffer.append(proba_dict)
        
        if len(self.prediction_buffer) == 0:
            return proba_dict
        
        # Média das últimas N predições
        smoothed = {}
        for label in proba_dict.keys():
            smoothed[label] = np.mean([p[label] for p in self.prediction_buffer])
        
        # Classe com maior probabilidade suavizada
        best_label = max(smoothed.items(), key=lambda x: x[1])[0]
        
        return best_label, smoothed
    
    def draw_landmarks(self, frame, landmarks, bbox_info):
        """Desenha landmarks no frame"""
        x1, y1, x2, y2, w_crop, h_crop = bbox_info
        
        # Desenhar landmarks
        for lm in landmarks:
            if lm.visibility > 0.5:
                # Converter coordenadas normalizadas pra frame
                cx = int(lm.x * w_crop) + x1
                cy = int(lm.y * h_crop) + y1
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
        
        # Desenhar conexões principais (simplificado)
        connections = [
            (11, 12), (11, 13), (13, 15),  # Braço esquerdo
            (12, 14), (14, 16),             # Braço direito
            (11, 23), (12, 24), (23, 24),   # Tronco
            (23, 25), (25, 27), (27, 29),   # Perna esquerda
            (24, 26), (26, 28), (28, 30),   # Perna direita
        ]
        
        for idx1, idx2 in connections:
            lm1, lm2 = landmarks[idx1], landmarks[idx2]
            if lm1.visibility > 0.5 and lm2.visibility > 0.5:
                pt1 = (int(lm1.x * w_crop) + x1, int(lm1.y * h_crop) + y1)
                pt2 = (int(lm2.x * w_crop) + x1, int(lm2.y * h_crop) + y1)
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
    
    def _draw_block_letter(self, frame, letter, cx, cy, size, color_bgr, thickness=8):
        """Desenha letra R, L ou C só com retângulos e linhas (zero dependência de fonte)."""
        s = size
        x0 = cx - s // 2
        y0 = cy - s // 2
        # Todas as coordenadas relativas ao canto (x0, y0)
        if letter == "R":
            cv2.rectangle(frame, (x0, y0), (x0 + s // 4, y0 + s), color_bgr, -1)
            cv2.rectangle(frame, (x0, y0), (x0 + s, y0 + s // 4), color_bgr, -1)
            cv2.rectangle(frame, (x0, y0 + s // 2 - s // 8), (x0 + s, y0 + s // 2 + s // 8), color_bgr, -1)
            cv2.rectangle(frame, (x0 + 3 * s // 4, y0), (x0 + s, y0 + s // 2), color_bgr, -1)
            pts = np.array([[x0 + 3 * s // 4, y0 + s // 2], [x0 + s, y0 + s]], np.int32)
            cv2.fillPoly(frame, [pts], color_bgr)
        elif letter == "L":
            cv2.rectangle(frame, (x0, y0), (x0 + s // 4, y0 + s), color_bgr, -1)
            cv2.rectangle(frame, (x0, y0 + 3 * s // 4), (x0 + s, y0 + s), color_bgr, -1)
        else:
            # C (center)
            cv2.rectangle(frame, (x0, y0), (x0 + s // 4, y0 + s), color_bgr, -1)
            cv2.rectangle(frame, (x0, y0), (x0 + s, y0 + s // 4), color_bgr, -1)
            cv2.rectangle(frame, (x0, y0 + 3 * s // 4), (x0 + s, y0 + s), color_bgr, -1)

    def _draw_text_pixels(self, text, color_bgr=(255, 255, 255), bg_bgr=(0, 0, 0), scale=4):
        """Texto com PIL (fallback para 'PAUSADO')."""
        if not _PIL_AVAILABLE:
            return None
        try:
            font = ImageFont.load_default()
        except Exception:
            return None
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
        bg_rgb = (bg_bgr[2], bg_bgr[1], bg_bgr[0])
        w0, h0 = 120, 20
        img = Image.new("RGB", (w0, h0), bg_rgb)
        d = ImageDraw.Draw(img)
        d.text((4, 2), text, fill=color_rgb, font=font)
        arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        if scale != 1 and scale > 0:
            arr = cv2.resize(arr, (w0 * scale, h0 * scale), interpolation=cv2.INTER_NEAREST)
        return arr

    def draw_prediction(self, frame, pred_label, proba_dict):
        """Desenha a predição com texto pequeno no canto superior esquerdo"""
        h, w = frame.shape[:2]
        
        TEXT_COLOR = (255, 255, 255)  # branco
        
        # Posição: canto superior esquerdo
        x_start = 20               # margem da esquerda
        y_start = 20               # margem do topo
        
        # Bloco menor e compacto
        box_width = 180
        box_height = 120            # cabe as 4 linhas com folga pequena
        
        # Fundo semi-transparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_start, y_start), (x_start + box_width, y_start + box_height),
                    (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)
        
        # Texto principal (menor que antes)
        confidence = proba_dict.get(pred_label, 0.0)
        main_text = f"{pred_label.upper()} {confidence:.0%}"
        cv2.putText(
            frame,
            main_text,
            (x_start + 12, y_start + 38),   # alinhado dentro do bloco
            cv2.FONT_HERSHEY_DUPLEX,
            .7,                            # bem menor
            TEXT_COLOR,
            2,                              # espessura reduzida
            cv2.LINE_AA
        )
        
        # Probabilidades (ainda menores e mais apertadas)
        y_offset = y_start + 65
        for label in ["left", "center", "right"]:
            prob = proba_dict.get(label, 0.0)
            text = f"{label.upper():<6} {prob:.0%}"
            
            cv2.putText(
                frame,
                text,
                (x_start + 12, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,                       # tamanho bem reduzido
                TEXT_COLOR,
                1,                          # espessura fina
                cv2.LINE_AA
            )
            y_offset += 24     

        if self.decision_made:
            decision_text = f"DECISAO FINAL: {self.final_decision.upper()} ({self.final_confidence:.0%})"
            
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 1
            thickness = 2
            
            # calcular tamanho do texto para centralizar
            (text_width, text_height), _ = cv2.getTextSize(
                decision_text, font, font_scale, thickness
            )
            
            center_x = (w - text_width) // 2
            center_y = int(h * .85)   # meio inferior

            
            # fundo preto semi-transparente atrás do texto
            overlay = frame.copy()
            padding = 20
            
            cv2.rectangle(
                overlay,
                (center_x - padding, center_y - text_height - padding),
                (center_x + text_width + padding, center_y + padding),
                (0, 0, 0),
                -1
            )
            
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # desenhar texto
            cv2.putText(
                frame,
                decision_text,
                (center_x, center_y),
                font,
                font_scale,
                (0, 255, 255),  # amarelo
                thickness,
                cv2.LINE_AA
            )          
            


    def process_frame(self, frame, person_box=None):
        """Processa um frame completo. person_box opcional (reuso entre frames)."""
        if person_box is None:
            person_box = self.detect_person(frame)
        if person_box is None:
            self.draw_prediction(frame, self._last_smoothed_label, self._last_smoothed_proba)
            cv2.putText(frame, "Nenhuma pessoa detectada", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return frame
        
        # 2. Desenhar bounding box
        x1, y1, x2, y2 = person_box
        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # 3. Extrair pose
        landmarks, bbox_info = self.extract_pose(frame, person_box)

        if landmarks is None:
            # Mantém painel com última predição; aviso pequeno no canto
            self.draw_prediction(frame, self._last_smoothed_label, self._last_smoothed_proba)
            h, w = frame.shape[:2]
            cv2.putText(frame, "Pose nao detectada", (w - 220, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            return frame

        # 4. Desenhar landmarks
        #self.draw_landmarks(frame, landmarks, bbox_info)

        # 5. Predizer
        pred_label, proba_dict = self.predict(landmarks)

        # 6. Guardar última predição (SEM smoothing)
        self._last_smoothed_label = pred_label
        self._last_smoothed_proba = proba_dict

        # 7. Desenhar predição
        self.draw_prediction(frame, pred_label, proba_dict)


        return frame
    
    def run_video(self, video_source=0, output_path=None, detect_every=2):
        """
        Roda predição em vídeo.

        Args:
            video_source: 0 para webcam, ou caminho do arquivo
            output_path: Caminho pra salvar vídeo (opcional)
            detect_every: Rodar YOLO a cada N frames (maior = mais FPS, menos precisão do bbox)
        """
        cap = cv2.VideoCapture(video_source)
        if isinstance(video_source, str):
            filename = Path(video_source).stem.lower()
            if "left" in filename:
                self.y_real = "left"
            elif "right" in filename:
                self.y_real = "right"
            elif "center" in filename:
                self.y_real = "center"
            else:
                self.y_real = None
        else:
            self.y_real = None
        self._last_wrist_right = None
        self._last_wrist_left = None
        self.last_person_box = None
        self._predict_count = 0
        classes = list(self.label_encoder.classes_)
        self._last_smoothed_proba = {c: 1.0 / len(classes) for c in classes}
        self._last_smoothed_label = classes[0]
        if self.debug:
            print("[debug] Predições serão impressas abaixo (uma por frame com pose):\n")

        if not cap.isOpened():
            print(f"❌ Erro ao abrir vídeo: {video_source}")
            return

        detect_interval = detect_every
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"📹 Vídeo: {width}x{height} @ {fps}fps (YOLO a cada {detect_interval} frames)")
        
        # Writer (se output_path fornecido)
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"💾 Salvando em: {output_path}")
        
        # Processar frames
        frame_count = 0
        start_time = time.time()
        
        print(f"\n▶️  Processando... (Espaço = pausar | 'q' = sair)\n")

        paused = False
        last_processed = None
        self.global_frame_count = 0
        self.video_fps = fps

        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_count % detect_interval == 0:
                        self.last_person_box = self.detect_person(frame)
                    processed_frame = self.process_frame(frame, self.last_person_box)
                    frame_count += 1
                    self.global_frame_count += 1
                    elapsed = time.time() - start_time
                    current_fps = frame_count / elapsed if elapsed > 0 else 0
                    cv2.putText(processed_frame, f"FPS: {current_fps:.1f}",
                                (processed_frame.shape[1] - 150, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    last_processed = processed_frame
                    if writer:
                        writer.write(processed_frame)

                if last_processed is not None:
                    display = last_processed.copy()
                    if paused:
                        overlay = display.copy()
                        cv2.rectangle(overlay, (0, 0), (display.shape[1], display.shape[0]), (0, 0, 0), -1)
                        display = cv2.addWeighted(overlay, 0.4, display, 0.6, 0)
                        pausado_img = self._draw_text_pixels("PAUSADO", color_bgr=(0, 255, 255), bg_bgr=(0, 0, 0), scale=6)
                        if pausado_img is not None:
                            hd, wd = display.shape[:2]
                            ph, pw = pausado_img.shape[:2]
                            xp = (wd - pw) // 2
                            yp = (hd - ph) // 2
                            if xp >= 0 and yp >= 0 and xp + pw <= wd and yp + ph <= hd:
                                display[yp : yp + ph, xp : xp + pw] = pausado_img
                    cv2.imshow("Penalty Predictor", display)
                    #cv2.putText(frame, "DEBUG TEXT", (50,50),
                        #cv2.FONT_HERSHEY_SIMPLEX,
                       # 1, (0,0,255), 2)


                key = cv2.waitKey(50 if paused else 1) & 0xFF
                if key == ord("q"):
                    print("\n⏹️  Interrompido pelo usuário")
                    break
                if key == ord(" "):
                    paused = not paused
                    print("⏸️  Pausado" if paused else "▶️  Retomado")
        
        except KeyboardInterrupt:
            print("\n⏹️  Interrompido pelo usuário")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            # Estatísticas finais
            total_time = time.time() - start_time
            avg_fps = frame_count / total_time if total_time > 0 else 0

            total_frames = self.global_frame_count

            if self.decision_frame is not None:
                timestamp_predito = self.decision_frame / self.video_fps
                timestamp_total = total_frames / self.video_fps
                antecedencia = timestamp_total - timestamp_predito
            else:
                timestamp_predito = None
                timestamp_total = total_frames / self.video_fps
                antecedencia = None

            y_pred = self.final_decision

            # ===== SALVAR RESULTADO =====
            result = {
                "video": video_source,
                "y_real": getattr(self, "y_real", None),
                "y_pred": y_pred,
                "timestamp_predito": timestamp_predito,
                "timestamp_total": timestamp_total,
                "antecedencia": antecedencia,
                "fps_medio": avg_fps
            }

            df = pd.DataFrame([result])
            csv_path = "data/results_experimento.csv"

            if Path(csv_path).exists():
                df.to_csv(csv_path, mode="a", header=False, index=False)
            else:
                df.to_csv(csv_path, index=False)

            print(f"\n💾 Resultado salvo em {csv_path}")

            print(f"\n{'='*60}")
            print(f"📊 ESTATÍSTICAS")
            print(f"{'='*60}")
            print(f"Frames processados: {frame_count}")
            print(f"Tempo total: {total_time:.2f}s")
            print(f"FPS médio: {avg_fps:.1f}")
            print(f"{'='*60}\n")



def main():
    parser = argparse.ArgumentParser(
        description="Predição de Pênaltis em Tempo Real",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Usar webcam
  python predict_realtime.py --webcam
  
  # Processar vídeo
  python predict_realtime.py --video cuts-penalty/-left_01.mp4
  
  # Processar e salvar resultado
  python predict_realtime.py --video input.mp4 --output resultado.mp4
        """
    )
    
    parser.add_argument('--video', type=str, help='Caminho do vídeo de entrada')
    parser.add_argument('--webcam', action='store_true', help='Usar webcam')
    parser.add_argument('--output', type=str, help='Caminho do vídeo de saída (opcional)')
    
    parser.add_argument('--yolo', type=str, default='models/yolov8s.pt',
                       help='Modelo YOLO (padrão: models/yolov8s.pt)')
    parser.add_argument('--pose', type=str, default='models/pose_landmarker_heavy.task',
                       help='Modelo MediaPipe (padrão: models/pose_landmarker_heavy.task)')
    parser.add_argument('--mlp', type=str, default='models/mlp_best_model.pkl',
                       help='Modelo MLP (padrão: models/mlp_best_model.pkl)')
    parser.add_argument('--scaler', type=str, default='models/scaler.pkl',
                       help='Scaler (padrão: models/scaler.pkl)')
    parser.add_argument('--encoder', type=str, default='models/label_encoder.pkl',
                       help='Label Encoder (padrão: models/label_encoder.pkl)')
    parser.add_argument('--detect-every', type=int, default=3, metavar='N',
                        help='Rodar detecção YOLO a cada N frames (padrão: 3, maior = mais FPS)')
    parser.add_argument('--debug', action='store_true',
                        help='Imprimir no terminal cada predição (label e probabilidades)')

    args = parser.parse_args()
    
    # Validar argumentos
    if not args.webcam and not args.video:
        parser.error("Especifique --webcam ou --video")
    
    if args.webcam and args.video:
        parser.error("Use --webcam OU --video, não ambos")
    
    predictor = PenaltyPredictor(
        yolo_model=args.yolo,
        pose_model=args.pose,
        mlp_model=args.mlp,
        scaler_model=args.scaler,
        label_encoder=args.encoder,
        debug=args.debug,
    )
    
    video_source = 0 if args.webcam else args.video
    predictor.run_video(video_source, args.output, detect_every=args.detect_every)


if __name__ == "__main__":
    main()