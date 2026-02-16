# ⚽🤖 Predição de Direção de Pênaltis com Computer Vision e Deep Learning

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![PyTorch](https://img.shields.io/badge/Framework-Computer%20Vision-orange)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose%20Estimation-green)
![YOLO](https://img.shields.io/badge/YOLO-v8-red?logo=yolo)
![Accuracy](https://img.shields.io/badge/Accuracy-46.3%25-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 📌 Visão Geral

Este projeto implementa um sistema de **predição em tempo real da direção de pênaltis** usando técnicas avançadas de **Computer Vision** e **Machine Learning**. O sistema analisa a postura corporal do jogador durante a cobrança e prediz se o chute será para a **esquerda**, **direita** ou **centro** do gol.

A solução combina três tecnologias principais:
- **YOLOv8:** Detecção de pessoas no frame
- **MediaPipe Pose:** Extração de 33 pontos de landmarks da pose corporal
- **MLP Neural Network:** Classificação da direção baseada em features extraídas

Todo o pipeline foi implementado com foco em **inferência em tempo real**, permitindo predições durante a execução do pênalti com antecedência média de **0.3 segundos** antes do chute.

---

## 🏆 Resultados

### Métricas de Performance

- ✅ **Acurácia Global: 46.3%** (vs. baseline aleatório de 33.3%)
- ✅ **Taxa de Decisão: 63.3%** (95 de 150 vídeos)
- ✅ **Ganho sobre baseline: +13%**
- ✅ **Antecedência Média: 0.3s** antes do chute
- ✅ **FPS Médio: 20.5** frames por segundo

### Distribuição de Predições

| Direção | Precisão | Recall |
|---------|----------|---------|
| **Esquerda** | 71.6% | - |
| **Direita** | 28.4% | - |
| **Centro** | 0% | - |

> **Nota:** O modelo apresenta viés para classificação à esquerda e dificuldade em identificar chutes no centro devido ao desbalanceamento do dataset.

---

## 🖼️ Demonstração Visual

### Pipeline de Processamento

```
Vídeo Input → YOLO Detecção → MediaPipe Pose → Feature Engineering → MLP → Predição
```

### Predição em Tempo Real

O sistema exibe:
- Bounding box do jogador detectado
- Skeleton pose overlay (33 landmarks)
- Predição de direção com confidence score
- Antecedência temporal do chute

---

## 🧠 Arquitetura do Sistema

### 1. Pipeline de Extração de Dados

#### **get_data.py** - Processamento de Vídeos

**Componentes principais:**

- **PersonTracker:** Sistema de tracking multi-objeto com IoU
  ```python
  - Rastreamento persistente de IDs
  - IoU threshold: 0.3
  - Max age: 30 frames
  - Seleção do melhor track por hits e idade
  ```

- **KalmanBoxTracker:** Filtro de Kalman para suavização
  ```python
  - Estado: [x_center, y_center, vx, vy]
  - Suavização de bounding boxes
  - Redução de jitter temporal
  ```

- **Validação de Pose:**
  ```python
  - Mínimo 20 landmarks visíveis
  - Visibilidade > 0.5
  - Key points (nariz, ombros, quadris) > 0.7
  ```

- **Normalização Espacial:**
  - Origem: ponto médio entre quadris
  - Escala: distância entre quadris
  - Coordenadas: (x, y, z) normalizadas

**Saída:** `pose_dataset.csv` com 99 features (33 landmarks × 3 coordenadas)

### 2. Feature Engineering

#### **modeling.ipynb** - Criação de Features Avançadas

**Features extraídas (107 no total):**

1. **Coordenadas Normalizadas (99):** f_0 até f_98
   - 33 landmarks × 3 coordenadas (x, y, z)

2. **Velocidades (6):**
   - Pulso direito: vx, vy, vz
   - Pulso esquerdo: vx, vy, vz

3. **Centroides (1):**
   - Centro de massa corporal

4. **Ângulos Articulares (1):**
   - Ângulo do joelho direito (quadril-joelho-tornozelo)

**Processamento:**
```python
- Conversão wide → long format
- Cálculo de velocidades entre frames
- Ângulos usando produto vetorial
- Normalização com StandardScaler
```

### 3. Modelo de Classificação

#### **MLP Neural Network**

**Arquitetura:**
```
Input Layer (107 features)
    ↓
Hidden Layer 1 (128 neurons) + ReLU
    ↓
Hidden Layer 2 (64 neurons) + ReLU
    ↓
Hidden Layer 3 (32 neurons) + ReLU
    ↓
Output Layer (3 classes) + Softmax
```

**Hiperparâmetros otimizados:**
- Learning rate: 0.001
- Batch size: 32
- Alpha (L2): 0.0001
- Optimizer: Adam
- Epochs: 500 (early stopping)

**Tratamento de Desbalanceamento:**
- SMOTE para oversampling de classes minoritárias
- Validação cruzada estratificada (5-fold)

**Métricas de Treinamento:**
```python
GridSearchCV com 72 combinações
360 fits totais (5 folds × 72 configs)
Melhor score CV: ~0.46
```

### 4. Sistema de Inferência em Tempo Real

#### **predict_live.py** - Predição Live

**Componentes:**

1. **Detector YOLO:**
   - Modelo: YOLOv8s
   - Input size: 320×320 (otimizado para velocidade)
   - Confiança mínima: 0.3

2. **Extrator MediaPipe:**
   - Modelo: pose_landmarker_heavy.task
   - Min detection confidence: 0.3
   - Num poses: 1

3. **Suavização Temporal:**
   ```python
   - Buffer de 15 frames (~0.5s em 30fps)
   - Threshold de confiança: 0.75
   - Decisão final por média móvel
   ```

4. **Sistema de Decisão:**
   ```python
   if len(buffer) >= MIN_FRAMES:
       avg_confidence = mean(last_15_predictions)
       if avg_confidence > THRESHOLD:
           MAKE_DECISION()
   ```

**Visualização:**
- Overlay de skeleton pose
- Probabilidades por classe
- Barra de confiança
- Timestamp da decisão

---

## 📊 Feature Engineering Detalhado

### Normalização Espacial

A normalização utiliza os quadris como referência:

```python
x0 = (hip_left.x + hip_right.x) / 2
y0 = (hip_left.y + hip_right.y) / 2
z0 = (hip_left.z + hip_right.z) / 2

scale = sqrt((hip_left - hip_right)²)

x_norm = (x - x0) / scale
y_norm = (y - y0) / scale
z_norm = (z - z0) / scale
```

**Vantagens:**
- Invariância a escala e posição
- Foco em movimentos relativos
- Robustez a diferentes distâncias da câmera

### Cálculo de Velocidades

Velocidade estimada por diferença finita:

```python
velocity = (position_t - position_t-1) / Δt
```

**Landmarks rastreados:**
- Pulso direito (wrist_right)
- Pulso esquerdo (wrist_left)

### Ângulos Articulares

Ângulo do joelho calculado por produto vetorial:

```python
v1 = hip - knee
v2 = ankle - knee

cos(θ) = (v1 · v2) / (||v1|| × ||v2||)
θ = arccos(cos(θ))
```

---

## ⚙️ Tecnologias Utilizadas

### Core Libraries
- **Python 3.12**
- **OpenCV** (Processamento de vídeo)
- **MediaPipe** (Pose estimation)
- **Ultralytics YOLOv8** (Object detection)
- **scikit-learn** (Machine learning)
- **pandas** / **NumPy** (Data manipulation)
- **joblib** (Model persistence)

### Técnicas Avançadas
- **Kalman Filtering** (filterpy)
- **SMOTE** (imblearn)
- **Grid Search CV** (Hyperparameter tuning)
- **Stratified K-Fold** (Cross-validation)

### Visualization
- **Matplotlib** / **Seaborn**
- **PIL** (Text rendering)
- **tqdm** (Progress bars)

---

## 📁 Estrutura do Projeto

```
penalty-prediction/
├── data/
│   ├── pose_dataset.csv          # Dataset bruto extraído
│   ├── features_essenciais.csv   # Features engineered
│   └── results_experimento.csv   # Resultados de teste
│
├── models/
│   ├── yolov8s.pt                # Detector YOLO
│   ├── pose_landmarker_heavy.task # MediaPipe Pose
│   ├── mlp_best_model.pkl        # MLP treinado
│   ├── scaler.pkl                # StandardScaler
│   └── label_encoder.pkl         # Label Encoder
│
├── cuts-penalty/                  # Vídeos de entrada
│   ├── -left_01.mp4
│   ├── -right_01.mp4
│   └── center_01.mp4
│
├── scrapping.py                   # Download de vídeos (yt-dlp)
├── get_data.py                    # Extração de poses
├── modeling.ipynb                 # Training pipeline
├── result_analyses.ipynb          # Análise de resultados
├── predict_live.py                # Inferência em tempo real
└── README.md
```

---

## 🚀 Como Usar

### 1. Instalação

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/penalty-prediction.git
cd penalty-prediction

# Instale dependências
pip install opencv-python mediapipe ultralytics scikit-learn pandas numpy joblib filterpy imbalanced-learn tqdm pillow yt-dlp
```

### 2. Download de Vídeos (Opcional)

```bash
python scrapping.py
```

### 3. Extração de Poses

```bash
# Processar vídeos e extrair dataset
python get_data.py

# Output: data/pose_dataset.csv
```

**Configurações disponíveis:**
```python
extractor = PoseExtractor(
    yolo_model="models/yolov8s.pt",
    pose_model="models/pose_landmarker_heavy.task",
    frame_skip=1,           # Processar todos os frames
    use_tracking=True,      # Ativar tracking
    target_size=640         # Resolução YOLO
)
```

### 4. Treinamento do Modelo

Abra `modeling.ipynb` no Jupyter e execute todas as células:

```bash
jupyter notebook modeling.ipynb
```

**Processo:**
1. Load e análise exploratória
2. Feature engineering
3. Train/test split (80/20)
4. Grid Search CV
5. Treinamento final
6. Salvamento dos modelos

### 5. Predição em Tempo Real

```bash
# Usando vídeo
python predict_live.py --video cuts-penalty/-left_01.mp4

# Usando webcam
python predict_live.py --webcam

# Com debug no terminal
python predict_live.py --video test.mp4 --debug
```

**Saída:**
- Janela com visualização em tempo real
- Predições exibidas no frame
- Logs de confiança (se --debug)

---

## 📈 Análise de Resultados

### Experimento Completo

150 vídeos de teste foram processados:

```python
# Carregar resultados
df_results = pd.read_csv("data/results_experimento.csv")

# Métricas
acurácia = 0.463  # 46.3%
taxa_decisão = 0.633  # 63.3%
antecedência_média = 0.3  # segundos
```

### Distribuição Real vs Predita

| Label | Real | Predito |
|-------|------|---------|
| Left | 36.8% | **71.6%** |
| Right | 52.6% | 28.4% |
| Center | 10.5% | 0% |

### Desafios Identificados

1. **Desbalanceamento de Classes:**
   - Centro com apenas 10.5% dos samples
   - SMOTE aplicado, mas insuficiente

2. **Viés de Predição:**
   - Modelo favorece classificação à esquerda
   - Dificuldade em generalizar para centro

3. **Taxa de Não-Decisão:**
   - 36.7% dos casos sem decisão firme
   - Threshold de confiança conservador (0.75)

### Pontos Fortes

1. **Antecedência Temporal:**
   - Média de 0.3s antes do chute
   - Suficiente para reação humana

2. **Performance em Tempo Real:**
   - ~20 FPS em hardware comum
   - Latência aceitável para aplicações práticas

3. **Robustez:**
   - Tracking multi-frame
   - Filtro de Kalman reduz noise
   - Validação de poses

---

## 🔬 Detalhes Técnicos

### Formato do Dataset

**pose_dataset.csv:**
```
| video | frame | label | timestamp_ms | f_0 | f_1 | ... | f_98 |
|-------|-------|-------|--------------|-----|-----|-----|------|
| video1| 14    | left  | 466          | 0.12| -0.5| ... | 0.33 |
```

**features_essenciais.csv:**
```
| vel_wrist_r_x | vel_wrist_r_y | ... | angulo_joelho | f_0 | ... | label |
|---------------|---------------|-----|---------------|-----|-----|-------|
| 0.045         | -0.123        | ... | 2.34          | 0.12| ... | left  |
```

### Sistema de Logging

```python
# Configuração
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pose_extraction.log'),
        logging.StreamHandler()
    ]
)
```

**Logs incluem:**
- Progresso de processamento
- Poses válidas detectadas
- Estatísticas por vídeo
- Checkpoints de salvamento

### Checkpoints Automáticos

Durante extração de dados:
```python
# A cada 500 samples
if len(all_data) % 500 == 0:
    save_checkpoint(f"data/checkpoint_{len(all_data)}.csv")
```

---

## 🎯 Melhorias Futuras

### Curto Prazo

1. **Balanceamento de Dataset:**
   - Coletar mais vídeos de chutes no centro
   - Aplicar técnicas de augmentation temporal

2. **Feature Engineering:**
   - Ângulos adicionais (tornozelo, quadril)
   - Aceleração (segunda derivada)
   - Features de assimetria corporal

3. **Arquitetura do Modelo:**
   - Experimentar LSTM/GRU para sequências temporais
   - Atenção temporal nos últimos N frames
   - Ensemble de modelos

### Longo Prazo

1. **Deep Learning End-to-End:**
   - CNN 3D diretamente nos frames
   - Spatial-Temporal Graph CNN
   - Transformer para sequências de poses

2. **Dataset Expandido:**
   - Múltiplos ângulos de câmera
   - Diferentes níveis de competição
   - Dados de treino de goleiros

3. **Aplicação Prática:**
   - App mobile para análise em campo
   - Sistema de treinamento para goleiros
   - Análise estatística de jogadores

---

## 📚 Referências Técnicas

### Papers

1. **YOLO:**
   - Redmon et al. "You Only Look Once: Unified, Real-Time Object Detection"
   
2. **MediaPipe:**
   - Bazarevsky et al. "BlazePose: On-device Real-time Body Pose tracking"

3. **Pose Estimation:**
   - Cao et al. "OpenPose: Realtime Multi-Person 2D Pose Estimation"

### Frameworks

- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [MediaPipe](https://google.github.io/mediapipe/)
- [scikit-learn](https://scikit-learn.org/)

---

## 👤 Autor

**Kauã Dias**  
Estudante de Estatística e entusiasta de Computer Vision & Deep Learning

- 🐙 GitHub: [github.com/Kauadp](https://github.com/Kauadp)  
- 🔗 LinkedIn: [linkedin.com/in/kauad](https://www.linkedin.com/in/kauad/)

---

## 📞 Contato

Para dúvidas, sugestões ou colaborações:

- 📧 Email: [kauadp1405@example.com]
- 💬 Issues: [GitHub Issues](https://github.com/seu-usuario/penalty-prediction/issues)

---

<div align="center">

**⚽ Feito com paixão por futebol e tecnologia 🤖**

*"A melhor defesa é prever o ataque"*

</div>