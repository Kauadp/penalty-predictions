import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import logging
from collections import deque
from filterpy.kalman import KalmanFilter
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp_cpu

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pose_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PersonTracker:
    """Rastreador de pessoas com IDs persistentes"""
    
    def __init__(self, max_age=30, iou_threshold=0.3):
        self.tracks = {}
        self.next_id = 0
        self.max_age = max_age
        self.iou_threshold = iou_threshold
    
    def iou(self, box1, box2):
        """Calcula Intersection over Union entre duas bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return inter / (area1 + area2 - inter + 1e-6)
    
    def update(self, detections):
        """Atualiza tracks com novas detecções"""
        matched = set()
        
        # Match com tracks existentes
        for track_id, track_data in list(self.tracks.items()):
            best_iou = 0
            best_det = None
            
            for i, det in enumerate(detections):
                if i in matched:
                    continue
                iou_score = self.iou(track_data['box'], det)
                if iou_score > best_iou:
                    best_iou = iou_score
                    best_det = i
            
            if best_iou > self.iou_threshold:
                self.tracks[track_id]['box'] = detections[best_det]
                self.tracks[track_id]['age'] = 0
                self.tracks[track_id]['hits'] += 1
                matched.add(best_det)
            else:
                self.tracks[track_id]['age'] += 1
        
        # Remove tracks antigos
        self.tracks = {k: v for k, v in self.tracks.items() 
                      if v['age'] < self.max_age}
        
        # Adiciona novas detecções
        for i, det in enumerate(detections):
            if i not in matched:
                self.tracks[self.next_id] = {
                    'box': det, 
                    'age': 0,
                    'hits': 1
                }
                self.next_id += 1
        
        return self.tracks
    
    def get_best_track(self):
        """Retorna o track mais confiável (mais hits, menos idade)"""
        if not self.tracks:
            return None
        
        best_track = max(
            self.tracks.items(),
            key=lambda x: (x[1]['hits'], -x[1]['age'])
        )
        return best_track[1]['box']


class KalmanBoxTracker:
    """Filtro de Kalman para suavizar bounding boxes"""
    
    def __init__(self):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        # Estado: [x_center, y_center, vx, vy]
        self.kf.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        # Medição: [x_center, y_center]
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        self.kf.R *= 10
        self.kf.P *= 1000
        self.kf.Q *= 0.01
        self.initialized = False
    
    def update(self, box):
        """Atualiza com nova bounding box"""
        x1, y1, x2, y2 = box
        center = np.array([[(x1 + x2) / 2], [(y1 + y2) / 2]])
        
        if not self.initialized:
            self.kf.x = np.array([[center[0, 0]], [center[1, 0]], [0], [0]])
            self.initialized = True
        
        self.kf.predict()
        self.kf.update(center)
        
        # Retorna box suavizada
        w = x2 - x1
        h = y2 - y1
        cx, cy = self.kf.x[0, 0], self.kf.x[1, 0]
        
        return (
            int(cx - w/2),
            int(cy - h/2),
            int(cx + w/2),
            int(cy + h/2)
        )


def coords_norm(landmark_list):
    """Normaliza coordenadas da pose usando quadris como referência"""
    x0 = (landmark_list[23].x + landmark_list[24].x) / 2
    y0 = (landmark_list[23].y + landmark_list[24].y) / 2
    z0 = (landmark_list[23].z + landmark_list[24].z) / 2

    scale = np.sqrt(
        (landmark_list[23].x - landmark_list[24].x) ** 2 +
        (landmark_list[23].y - landmark_list[24].y) ** 2 +
        (landmark_list[23].z - landmark_list[24].z) ** 2
    )

    if scale == 0:
        scale = 1e-6

    normalized = []
    for lm in landmark_list:
        normalized.append((lm.x - x0) / scale)
        normalized.append((lm.y - y0) / scale)
        normalized.append((lm.z - z0) / scale)

    return normalized


def is_valid_pose(landmarks, min_visibility=0.5, min_landmarks=20):
    """Valida se a pose detectada é confiável"""
    visible_count = sum(1 for lm in landmarks 
                       if lm.visibility > min_visibility)
    
    if visible_count < min_landmarks:
        return False
    
    # Verificar se pontos-chave estão presentes
    key_points = [0, 11, 12, 23, 24]  # nose, shoulders, hips
    for idx in key_points:
        if landmarks[idx].visibility < 0.7:
            return False
    
    return True


def calculate_iou(box1, box2):
    """Calcula IoU entre duas boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return inter / (area1 + area2 - inter + 1e-6)


def select_main_person(persons, frame_shape, previous_box=None):
    """
    Seleciona pessoa principal baseado em múltiplos critérios
    """
    if not persons:
        return None
    
    h_frame, w_frame = frame_shape[:2]
    center_x = w_frame / 2
    
    best_score = -1
    chosen = None
    
    for p in persons:
        x1, y1, x2, y2 = p
        
        # Métricas
        area = (x2 - x1) * (y2 - y1)
        center_y = (y1 + y2) / 2
        box_center_x = (x1 + x2) / 2
        
        # Distância do centro horizontal
        dist_from_center = abs(box_center_x - center_x)
        
        # Score composto
        score = (
            area * 0.4 +                              # Tamanho
            center_y * 0.3 +                          # Posição vertical
            (w_frame - dist_from_center) * 0.2        # Centralidade
        )
        
        # Bonus por continuidade temporal
        if previous_box:
            iou = calculate_iou(p, previous_box)
            score += iou * area * 0.1
        
        if score > best_score:
            best_score = score
            chosen = p
    
    return chosen


class PoseExtractor:
    """Extrator de poses com tracking melhorado"""
    
    def __init__(self, yolo_model="models/yolov8s.pt", 
                 pose_model="models/pose_landmarker_heavy.task",
                 frame_skip=1, use_tracking=True, target_size=640):
        
        logger.info("Initializing PoseExtractor...")
        
        self.yolo = YOLO(yolo_model)
        self.frame_skip = frame_skip
        self.use_tracking = use_tracking
        self.target_size = target_size
        
        # Setup MediaPipe
        base_options = python.BaseOptions(model_asset_path=pose_model)
        self.pose_options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.7,
            num_poses=1
        )
        
        logger.info("PoseExtractor initialized successfully")
    
    def detect_persons(self, frame):
        """Detecta pessoas no frame com YOLO otimizado"""
        h, w = frame.shape[:2]
        scale = self.target_size / max(h, w)
        frame_small = cv2.resize(frame, None, fx=scale, fy=scale)
        
        results = self.yolo(frame_small, verbose=False)[0]
        
        persons = []
        for box in results.boxes:
            if int(box.cls[0]) == 0:  # pessoa
                x1, y1, x2, y2 = box.xyxy[0]
                persons.append((
                    int(x1/scale), int(y1/scale),
                    int(x2/scale), int(y2/scale)
                ))
        
        return persons
    
    def process_single_video(self, video_path, visualize=False):
        """Processa um único vídeo"""
        logger.info(f"Processing {video_path.name}")
        
        label = video_path.name.split("_")[0].strip("-")
        video = cv2.VideoCapture(str(video_path))
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup tracking
        tracker = PersonTracker() if self.use_tracking else None
        kalman = KalmanBoxTracker()
        
        # Criar landmarker para este vídeo
        landmarker = vision.PoseLandmarker.create_from_options(self.pose_options)
        
        data = []
        frame_id = 0
        last_chosen = None
        frames_processed = 0
        poses_detected = 0
        
        pbar = tqdm(total=total_frames, desc=video_path.name, leave=False)
        
        while True:
            success, frame = video.read()
            if not success:
                break
            
            pbar.update(1)
            
            # Frame skip para performance
            if frame_id % self.frame_skip != 0:
                frame_id += 1
                continue
            
            frames_processed += 1
            h_frame, w_frame, _ = frame.shape
            
            # Detectar pessoas
            persons = self.detect_persons(frame)
            
            # Atualizar tracker
            if self.use_tracking and tracker:
                tracker.update(persons)
                chosen = tracker.get_best_track()
            else:
                chosen = select_main_person(persons, frame.shape, last_chosen)
            
            # Suavização temporal básica
            if chosen is None and last_chosen is not None:
                chosen = last_chosen
            
            # Aplicar Kalman
            if chosen is not None:
                chosen = kalman.update(chosen)
                last_chosen = chosen
            
            # Visualização (opcional)
            if visualize:
                # Desenhar todas as pessoas (amarelo)
                for p in persons:
                    x1p, y1p, x2p, y2p = p
                    cv2.rectangle(frame, (x1p, y1p), (x2p, y2p), (0, 255, 255), 2)
                
                # Desenhar escolhido (verde grosso)
                if chosen:
                    x1c, y1c, x2c, y2c = chosen
                    cv2.rectangle(frame, (x1c, y1c), (x2c, y2c), (0, 255, 0), 4)
            
            # Extrair pose
            if chosen:
                x1c, y1c, x2c, y2c = chosen
                
                # Garantir que as coordenadas estão dentro dos limites
                x1c = max(0, x1c)
                y1c = max(0, y1c)
                x2c = min(w_frame, x2c)
                y2c = min(h_frame, y2c)
                
                crop = frame[y1c:y2c, x1c:x2c]
                
                if crop.size > 0:
                    crop = cv2.resize(crop, (256, 256))
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=crop_rgb
                    )
                    
                    result = landmarker.detect(mp_image)
                    
                    if result.pose_landmarks:
                        landmarks = result.pose_landmarks[0]
                        
                        # Validar qualidade da pose
                        if is_valid_pose(landmarks):
                            poses_detected += 1
                            
                            # Desenhar pontos no frame (se visualizando)
                            if visualize:
                                h_crop, w_crop, _ = crop.shape
                                for lm in landmarks:
                                    if lm.visibility > 0.5:
                                        cx = int(lm.x * w_crop) + x1c
                                        cy = int(lm.y * h_crop) + y1c
                                        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                            
                            # Normalizar coordenadas
                            coords = coords_norm(landmarks)
                            
                            row = {
                                "video": str(video_path),
                                "frame": frame_id,
                                "label": label,
                                "timestamp_ms": int((frame_id / fps) * 1000)
                            }
                            
                            for i, value in enumerate(coords):
                                row[f"f_{i}"] = value
                            
                            data.append(row)
            
            if visualize:
                cv2.imshow("Processing", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            
            frame_id += 1
        
        pbar.close()
        video.release()
        
        if visualize:
            cv2.destroyAllWindows()
        
        logger.info(f"Finished {video_path.name}: {frames_processed} frames, "
                   f"{poses_detected} valid poses")
        
        return data
    
    def process_videos(self, video_paths, use_parallel=False, max_workers=4, visualize=False):
        """Processa múltiplos vídeos"""
        logger.info(f"Processing {len(video_paths)} videos...")
        
        all_data = []
        checkpoint_interval = 500
        
        if use_parallel and not visualize:
            logger.info(f"Using parallel processing with {max_workers} workers")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self.process_single_video, video_path, False)
                    for video_path in video_paths
                ]
                
                for future in tqdm(futures, desc="Overall progress"):
                    video_data = future.result()
                    all_data.extend(video_data)
                    
                    # Checkpoint
                    if len(all_data) % checkpoint_interval == 0 and len(all_data) > 0:
                        self._save_checkpoint(all_data)
        else:
            for video_path in video_paths:
                video_data = self.process_single_video(video_path, visualize)
                all_data.extend(video_data)
                
                # Checkpoint
                if len(all_data) % checkpoint_interval == 0 and len(all_data) > 0:
                    self._save_checkpoint(all_data)
        
        logger.info(f"Processing complete: {len(all_data)} total pose samples")
        return pd.DataFrame(all_data)
    
    def _save_checkpoint(self, data):
        """Salva checkpoint do progresso"""
        df_temp = pd.DataFrame(data)
        checkpoint_path = f"data/checkpoint_{len(data)}.csv"
        df_temp.to_csv(checkpoint_path, index=False)
        logger.info(f"Checkpoint saved: {checkpoint_path}")


def main():
    """Função principal"""
    
    # Criar diretório de dados se não existir
    Path("data").mkdir(exist_ok=True)
    
    # Buscar vídeos
    video_paths = list(Path("cuts-penalty").glob("*.mp4"))
    logger.info(f"Found {len(video_paths)} videos")
    
    if not video_paths:
        logger.error("No videos found in 'cuts-penalty' directory!")
        return
    
    # Criar extrator
    extractor = PoseExtractor(
        yolo_model="models/yolov8s.pt",
        pose_model="models/pose_landmarker_heavy.task",
        frame_skip=1,  # Processar todos os frames (use 2 ou 3 para mais velocidade)
        use_tracking=True,
        target_size=640
    )
    
    # Processar vídeos
    df = extractor.process_videos(
        video_paths,
        use_parallel=False,  # True para processar em paralelo (desabilita visualização)
        max_workers=min(4, mp_cpu.cpu_count()),
        visualize=False  # True para visualizar (desabilita paralelização)
    )
    
    # Salvar dataset final
    output_path = "data/pose_dataset.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Dataset saved: {output_path}")
    logger.info(f"Final shape: {df.shape}")
    
    # Estatísticas
    logger.info("\n=== Dataset Statistics ===")
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Unique videos: {df['video'].nunique()}")
    logger.info(f"Labels distribution:\n{df['label'].value_counts()}")
    logger.info(f"Average samples per video: {len(df) / df['video'].nunique():.1f}")


if __name__ == "__main__":
    main()