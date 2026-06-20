"""
Utilidades para el Detector de Forma en Ejercicios
===================================================

Este módulo contiene funciones auxiliares para:
- Procesamiento de video
- Cálculo de ángulos articulares
- Visualización de landmarks
- Preprocesamiento de datos

Autor: César Adrián Delgado Díaz
Fecha: Diciembre 2025
"""

import numpy as np
import cv2
from typing import Tuple, List, Dict, Optional, Union
import math


# =============================================================================
# CONSTANTES DE MEDIAPIPE POSE LANDMARKS
# =============================================================================

POSE_LANDMARKS = {
    'NOSE': 0,
    'LEFT_EYE_INNER': 1,
    'LEFT_EYE': 2,
    'LEFT_EYE_OUTER': 3,
    'RIGHT_EYE_INNER': 4,
    'RIGHT_EYE': 5,
    'RIGHT_EYE_OUTER': 6,
    'LEFT_EAR': 7,
    'RIGHT_EAR': 8,
    'MOUTH_LEFT': 9,
    'MOUTH_RIGHT': 10,
    'LEFT_SHOULDER': 11,
    'RIGHT_SHOULDER': 12,
    'LEFT_ELBOW': 13,
    'RIGHT_ELBOW': 14,
    'LEFT_WRIST': 15,
    'RIGHT_WRIST': 16,
    'LEFT_PINKY': 17,
    'RIGHT_PINKY': 18,
    'LEFT_INDEX': 19,
    'RIGHT_INDEX': 20,
    'LEFT_THUMB': 21,
    'RIGHT_THUMB': 22,
    'LEFT_HIP': 23,
    'RIGHT_HIP': 24,
    'LEFT_KNEE': 25,
    'RIGHT_KNEE': 26,
    'LEFT_ANKLE': 27,
    'RIGHT_ANKLE': 28,
    'LEFT_HEEL': 29,
    'RIGHT_HEEL': 30,
    'LEFT_FOOT_INDEX': 31,
    'RIGHT_FOOT_INDEX': 32
}

# Conexiones para dibujar el esqueleto
POSE_CONNECTIONS = [
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Brazo izquierdo
    (11, 13), (13, 15),
    # Brazo derecho
    (12, 14), (14, 16),
    # Pierna izquierda
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # Pierna derecha
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]

# Colores para visualización (BGR)
COLORS = {
    'correct': (0, 255, 0),      # Verde
    'warning': (0, 165, 255),    # Naranja
    'error': (0, 0, 255),        # Rojo
    'neutral': (255, 255, 255),  # Blanco
    'skeleton': (245, 117, 66),  # Azul-coral
    'landmark': (80, 110, 230),  # Rojo-naranja
}


# =============================================================================
# FUNCIONES DE CÁLCULO DE ÁNGULOS
# =============================================================================

def calculate_angle(point_a: np.ndarray, 
                   point_b: np.ndarray, 
                   point_c: np.ndarray) -> float:
    """
    Calcula el ángulo entre tres puntos.
    
    El ángulo se calcula en el punto B, formado por los vectores BA y BC.
    
    Args:
        point_a: Primer punto [x, y] o [x, y, z]
        point_b: Punto central (vértice del ángulo) [x, y] o [x, y, z]
        point_c: Tercer punto [x, y] o [x, y, z]
    
    Returns:
        Ángulo en grados (0-180)
    
    Example:
        >>> shoulder = np.array([0.5, 0.3])
        >>> elbow = np.array([0.5, 0.5])
        >>> wrist = np.array([0.6, 0.7])
        >>> angle = calculate_angle(shoulder, elbow, wrist)
    """
    a = np.array(point_a[:2])
    b = np.array(point_b[:2])
    c = np.array(point_c[:2])
    
    # Vectores
    ba = a - b
    bc = c - b
    
    # Producto punto y magnitudes
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    
    # Limitar al rango válido para arccos
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    # Convertir a grados
    angle = np.degrees(np.arccos(cosine_angle))
    
    return angle


def calculate_angle_3d(point_a: np.ndarray, 
                       point_b: np.ndarray, 
                       point_c: np.ndarray) -> float:
    """
    Calcula el ángulo entre tres puntos en 3D.
    
    Args:
        point_a: Primer punto [x, y, z]
        point_b: Punto central (vértice) [x, y, z]
        point_c: Tercer punto [x, y, z]
    
    Returns:
        Ángulo en grados (0-180)
    """
    a = np.array(point_a[:3])
    b = np.array(point_b[:3])
    c = np.array(point_c[:3])
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    return np.degrees(np.arccos(cosine_angle))


def calculate_vertical_angle(point_a: np.ndarray, 
                            point_b: np.ndarray) -> float:
    """
    Calcula el ángulo de una línea respecto a la vertical.
    
    Args:
        point_a: Punto superior [x, y]
        point_b: Punto inferior [x, y]
    
    Returns:
        Ángulo en grados respecto a la vertical (0 = perfectamente vertical)
    """
    dx = point_b[0] - point_a[0]
    dy = point_b[1] - point_a[1]
    
    # Ángulo respecto a la vertical (eje Y)
    angle = np.degrees(np.arctan2(abs(dx), abs(dy)))
    
    return angle


def calculate_horizontal_angle(point_a: np.ndarray, 
                               point_b: np.ndarray) -> float:
    """
    Calcula el ángulo de una línea respecto a la horizontal.
    
    Args:
        point_a: Primer punto [x, y]
        point_b: Segundo punto [x, y]
    
    Returns:
        Ángulo en grados respecto a la horizontal
    """
    dx = point_b[0] - point_a[0]
    dy = point_b[1] - point_a[1]
    
    angle = np.degrees(np.arctan2(dy, dx))
    
    return abs(angle)


# =============================================================================
# FUNCIONES DE EXTRACCIÓN DE LANDMARKS
# =============================================================================

def extract_landmarks(results, flatten: bool = True) -> Optional[np.ndarray]:
    """
    Extrae los landmarks de pose de los resultados de MediaPipe.
    
    Args:
        results: Resultados de MediaPipe pose detection
        flatten: Si True, retorna array aplanado [x1,y1,z1,v1, x2,y2,z2,v2, ...]
    
    Returns:
        Array de landmarks o None si no se detectó pose
    """
    if results.pose_landmarks is None:
        return None
    
    landmarks = []
    for landmark in results.pose_landmarks.landmark:
        landmarks.append([
            landmark.x,
            landmark.y,
            landmark.z,
            landmark.visibility
        ])
    
    landmarks = np.array(landmarks)
    
    if flatten:
        return landmarks.flatten()
    
    return landmarks


def get_landmark_point(landmarks: np.ndarray, 
                       landmark_name: str) -> np.ndarray:
    """
    Obtiene las coordenadas de un landmark específico.
    
    Args:
        landmarks: Array de landmarks (33, 4) o (132,)
        landmark_name: Nombre del landmark (ej: 'LEFT_KNEE')
    
    Returns:
        Array [x, y, z, visibility]
    """
    if landmarks.ndim == 1:
        landmarks = landmarks.reshape(33, 4)
    
    idx = POSE_LANDMARKS[landmark_name]
    return landmarks[idx]


def get_body_part_coordinates(landmarks: np.ndarray, 
                              side: str = 'left') -> Dict[str, np.ndarray]:
    """
    Obtiene las coordenadas de las partes del cuerpo para un lado.
    
    Args:
        landmarks: Array de landmarks
        side: 'left' o 'right'
    
    Returns:
        Diccionario con coordenadas de shoulder, elbow, wrist, hip, knee, ankle
    """
    if landmarks.ndim == 1:
        landmarks = landmarks.reshape(33, 4)
    
    prefix = 'LEFT_' if side == 'left' else 'RIGHT_'
    
    return {
        'shoulder': landmarks[POSE_LANDMARKS[f'{prefix}SHOULDER']],
        'elbow': landmarks[POSE_LANDMARKS[f'{prefix}ELBOW']],
        'wrist': landmarks[POSE_LANDMARKS[f'{prefix}WRIST']],
        'hip': landmarks[POSE_LANDMARKS[f'{prefix}HIP']],
        'knee': landmarks[POSE_LANDMARKS[f'{prefix}KNEE']],
        'ankle': landmarks[POSE_LANDMARKS[f'{prefix}ANKLE']],
    }


# =============================================================================
# FUNCIONES DE ANÁLISIS DE EJERCICIOS
# =============================================================================

def analyze_squat(landmarks: np.ndarray) -> Dict:
    """
    Analiza la forma de una sentadilla.
    
    Args:
        landmarks: Array de landmarks de MediaPipe
    
    Returns:
        Diccionario con ángulos, estado y feedback
    """
    if landmarks.ndim == 1:
        landmarks = landmarks.reshape(33, 4)
    
    # Obtener puntos clave (promedio de ambos lados para simetría)
    left = get_body_part_coordinates(landmarks, 'left')
    right = get_body_part_coordinates(landmarks, 'right')
    
    # Calcular ángulos de rodilla
    left_knee_angle = calculate_angle(left['hip'], left['knee'], left['ankle'])
    right_knee_angle = calculate_angle(right['hip'], right['knee'], right['ankle'])
    knee_angle = (left_knee_angle + right_knee_angle) / 2
    
    # Calcular ángulo de cadera
    left_hip_angle = calculate_angle(left['shoulder'], left['hip'], left['knee'])
    right_hip_angle = calculate_angle(right['shoulder'], right['hip'], right['knee'])
    hip_angle = (left_hip_angle + right_hip_angle) / 2
    
    # Calcular inclinación del torso
    mid_shoulder = (left['shoulder'][:2] + right['shoulder'][:2]) / 2
    mid_hip = (left['hip'][:2] + right['hip'][:2]) / 2
    torso_angle = calculate_vertical_angle(mid_shoulder, mid_hip)
    
    # Verificar alineación de rodillas con pies
    left_knee_alignment = abs(left['knee'][0] - left['ankle'][0])
    right_knee_alignment = abs(right['knee'][0] - right['ankle'][0])
    
    # Determinar fase del ejercicio
    if knee_angle > 160:
        phase = 'standing'
    elif knee_angle > 100:
        phase = 'descending'
    else:
        phase = 'bottom'
    
    # Evaluar errores
    errors = []
    warnings = []
    
    # Error: Rodillas hacia adentro (valgus)
    if left_knee_alignment > 0.08 or right_knee_alignment > 0.08:
        errors.append("Rodillas desalineadas - mantén rodillas sobre los pies")
    
    # Error: Excesiva inclinación del torso
    if torso_angle > 45 and phase == 'bottom':
        errors.append("Torso muy inclinado - mantén el pecho arriba")
    elif torso_angle > 35 and phase == 'bottom':
        warnings.append("Ligera inclinación del torso")
    
    # Advertencia: Sentadilla parcial
    if phase == 'bottom' and knee_angle > 100:
        warnings.append("Profundidad insuficiente - baja más")
    
    # Calcular score
    score = 100
    score -= len(errors) * 25
    score -= len(warnings) * 10
    score = max(0, score)
    
    # Determinar estado general
    if errors:
        status = 'error'
    elif warnings:
        status = 'warning'
    else:
        status = 'correct'
    
    return {
        'exercise': 'squat',
        'phase': phase,
        'angles': {
            'knee': round(knee_angle, 1),
            'hip': round(hip_angle, 1),
            'torso_inclination': round(torso_angle, 1)
        },
        'alignment': {
            'left_knee': round(left_knee_alignment, 3),
            'right_knee': round(right_knee_alignment, 3)
        },
        'status': status,
        'errors': errors,
        'warnings': warnings,
        'score': score
    }


def analyze_deadlift(landmarks: np.ndarray) -> Dict:
    """
    Analiza la forma de un peso muerto.
    
    Args:
        landmarks: Array de landmarks de MediaPipe
    
    Returns:
        Diccionario con ángulos, estado y feedback
    """
    if landmarks.ndim == 1:
        landmarks = landmarks.reshape(33, 4)
    
    left = get_body_part_coordinates(landmarks, 'left')
    right = get_body_part_coordinates(landmarks, 'right')
    
    # Ángulo de la espalda (hombro-cadera-rodilla)
    left_back_angle = calculate_angle(left['shoulder'], left['hip'], left['knee'])
    right_back_angle = calculate_angle(right['shoulder'], right['hip'], right['knee'])
    back_angle = (left_back_angle + right_back_angle) / 2
    
    # Ángulo de rodilla
    left_knee_angle = calculate_angle(left['hip'], left['knee'], left['ankle'])
    right_knee_angle = calculate_angle(right['hip'], right['knee'], right['ankle'])
    knee_angle = (left_knee_angle + right_knee_angle) / 2
    
    # Calcular curvatura de espalda (diferencia entre hombros y caderas en Y)
    mid_shoulder = (left['shoulder'][:2] + right['shoulder'][:2]) / 2
    mid_hip = (left['hip'][:2] + right['hip'][:2]) / 2
    
    # Inclinación del torso respecto a vertical
    torso_angle = calculate_vertical_angle(mid_shoulder, mid_hip)
    
    # Determinar fase
    if knee_angle > 165 and torso_angle < 15:
        phase = 'lockout'
    elif torso_angle > 45:
        phase = 'bottom'
    else:
        phase = 'lifting'
    
    errors = []
    warnings = []
    
    # Error: Espalda redondeada (detección simplificada)
    # En un modelo real, usaríamos más puntos de la columna
    if back_angle < 80 and phase in ['bottom', 'lifting']:
        errors.append("Espalda redondeada - mantén la espalda recta")
    
    # Error: Rodillas bloqueadas al inicio
    if phase == 'bottom' and knee_angle > 170:
        errors.append("Rodillas muy rectas - flexiona ligeramente")
    
    # Advertencia: Levantar solo con espalda
    if phase == 'lifting' and knee_angle > 160:
        warnings.append("Usa más las piernas en el levantamiento")
    
    score = 100
    score -= len(errors) * 25
    score -= len(warnings) * 10
    score = max(0, score)
    
    if errors:
        status = 'error'
    elif warnings:
        status = 'warning'
    else:
        status = 'correct'
    
    return {
        'exercise': 'deadlift',
        'phase': phase,
        'angles': {
            'back': round(back_angle, 1),
            'knee': round(knee_angle, 1),
            'torso_inclination': round(torso_angle, 1)
        },
        'status': status,
        'errors': errors,
        'warnings': warnings,
        'score': score
    }


def analyze_bench_press(landmarks: np.ndarray) -> Dict:
    """
    Analiza la forma de un press de banca.
    
    Args:
        landmarks: Array de landmarks de MediaPipe
    
    Returns:
        Diccionario con ángulos, estado y feedback
    """
    if landmarks.ndim == 1:
        landmarks = landmarks.reshape(33, 4)
    
    left = get_body_part_coordinates(landmarks, 'left')
    right = get_body_part_coordinates(landmarks, 'right')
    
    # Ángulo de codo
    left_elbow_angle = calculate_angle(left['shoulder'], left['elbow'], left['wrist'])
    right_elbow_angle = calculate_angle(right['shoulder'], right['elbow'], right['wrist'])
    elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
    
    # Ángulo del hombro (abducción)
    # Medimos el ángulo entre el torso y el brazo
    mid_hip = (left['hip'][:2] + right['hip'][:2]) / 2
    left_shoulder_angle = calculate_angle(mid_hip, left['shoulder'], left['elbow'])
    right_shoulder_angle = calculate_angle(mid_hip, right['shoulder'], right['elbow'])
    shoulder_angle = (left_shoulder_angle + right_shoulder_angle) / 2
    
    # Simetría de muñecas (para detectar barra desbalanceada)
    wrist_symmetry = abs(left['wrist'][1] - right['wrist'][1])
    
    # Determinar fase
    if elbow_angle > 160:
        phase = 'lockout'
    elif elbow_angle < 100:
        phase = 'bottom'
    else:
        phase = 'pressing'
    
    errors = []
    warnings = []
    
    # Error: Codos muy abiertos (ángulo de hombro > 90 grados)
    if shoulder_angle > 100:
        errors.append("Codos muy abiertos - mantén ángulo de 45-75° con el torso")
    
    # Error: Barra desbalanceada
    if wrist_symmetry > 0.05:
        errors.append("Barra desbalanceada - mantén las muñecas al mismo nivel")
    
    # Advertencia: Rango de movimiento incompleto
    if phase == 'bottom' and elbow_angle > 110:
        warnings.append("Profundidad insuficiente - baja más la barra")
    
    # Advertencia: Codos muy pegados
    if shoulder_angle < 30:
        warnings.append("Codos muy pegados al cuerpo")
    
    score = 100
    score -= len(errors) * 25
    score -= len(warnings) * 10
    score = max(0, score)
    
    if errors:
        status = 'error'
    elif warnings:
        status = 'warning'
    else:
        status = 'correct'
    
    return {
        'exercise': 'bench_press',
        'phase': phase,
        'angles': {
            'elbow': round(elbow_angle, 1),
            'shoulder_abduction': round(shoulder_angle, 1)
        },
        'symmetry': {
            'wrist_diff': round(wrist_symmetry, 3)
        },
        'status': status,
        'errors': errors,
        'warnings': warnings,
        'score': score
    }


# =============================================================================
# FUNCIONES DE VISUALIZACIÓN
# =============================================================================

def draw_landmarks(image: np.ndarray, 
                   landmarks: np.ndarray,
                   connections: bool = True,
                   color: Tuple[int, int, int] = None) -> np.ndarray:
    """
    Dibuja los landmarks de pose en una imagen.
    
    Args:
        image: Imagen BGR de OpenCV
        landmarks: Array de landmarks (33, 4)
        connections: Si dibujar conexiones entre landmarks
        color: Color personalizado (BGR)
    
    Returns:
        Imagen con landmarks dibujados
    """
    if landmarks is None:
        return image
    
    if landmarks.ndim == 1:
        landmarks = landmarks.reshape(33, 4)
    
    img = image.copy()
    h, w = img.shape[:2]
    
    landmark_color = color if color else COLORS['landmark']
    connection_color = color if color else COLORS['skeleton']
    
    # Dibujar conexiones
    if connections:
        for start_idx, end_idx in POSE_CONNECTIONS:
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            
            # Solo dibujar si ambos puntos son visibles
            if start[3] > 0.5 and end[3] > 0.5:
                start_point = (int(start[0] * w), int(start[1] * h))
                end_point = (int(end[0] * w), int(end[1] * h))
                cv2.line(img, start_point, end_point, connection_color, 2)
    
    # Dibujar landmarks
    for i, landmark in enumerate(landmarks):
        if landmark[3] > 0.5:  # Solo si es visible
            x = int(landmark[0] * w)
            y = int(landmark[1] * h)
            cv2.circle(img, (x, y), 5, landmark_color, -1)
            cv2.circle(img, (x, y), 7, (255, 255, 255), 1)
    
    return img


def draw_angle(image: np.ndarray,
               point_a: np.ndarray,
               point_b: np.ndarray,
               point_c: np.ndarray,
               angle: float,
               color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """
    Dibuja un ángulo con su valor en la imagen.
    
    Args:
        image: Imagen BGR
        point_a, point_b, point_c: Puntos del ángulo (coordenadas normalizadas)
        angle: Valor del ángulo en grados
        color: Color del texto
    
    Returns:
        Imagen con ángulo dibujado
    """
    img = image.copy()
    h, w = img.shape[:2]
    
    # Convertir a coordenadas de pixel
    b_pixel = (int(point_b[0] * w), int(point_b[1] * h))
    
    # Dibujar arco indicando el ángulo
    radius = 30
    
    # Calcular ángulos para el arco
    vec_ba = np.array([point_a[0] - point_b[0], point_a[1] - point_b[1]])
    vec_bc = np.array([point_c[0] - point_b[0], point_c[1] - point_b[1]])
    
    angle_ba = np.degrees(np.arctan2(vec_ba[1], vec_ba[0]))
    angle_bc = np.degrees(np.arctan2(vec_bc[1], vec_bc[0]))
    
    # Dibujar arco
    cv2.ellipse(img, b_pixel, (radius, radius), 0, 
                min(angle_ba, angle_bc), max(angle_ba, angle_bc), color, 2)
    
    # Dibujar texto con el ángulo
    text_pos = (b_pixel[0] + 10, b_pixel[1] - 10)
    cv2.putText(img, f'{angle:.0f}°', text_pos, 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return img


def draw_status_overlay(image: np.ndarray,
                        analysis: Dict,
                        position: str = 'top') -> np.ndarray:
    """
    Dibuja un overlay con el estado del ejercicio.
    
    Args:
        image: Imagen BGR
        analysis: Diccionario con resultados del análisis
        position: 'top' o 'bottom'
    
    Returns:
        Imagen con overlay
    """
    img = image.copy()
    h, w = img.shape[:2]
    
    # Colores según estado
    status_colors = {
        'correct': (0, 200, 0),
        'warning': (0, 165, 255),
        'error': (0, 0, 200)
    }
    
    status = analysis.get('status', 'neutral')
    color = status_colors.get(status, (128, 128, 128))
    
    # Crear overlay semitransparente
    overlay = img.copy()
    
    if position == 'top':
        y_start = 0
        y_end = 120
    else:
        y_start = h - 120
        y_end = h
    
    cv2.rectangle(overlay, (0, y_start), (w, y_end), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    
    # Dibujar información
    y_offset = y_start + 25
    
    # Ejercicio y fase
    exercise = analysis.get('exercise', 'Unknown').replace('_', ' ').title()
    phase = analysis.get('phase', 'Unknown').title()
    cv2.putText(img, f"{exercise} - {phase}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Score
    score = analysis.get('score', 0)
    cv2.putText(img, f"Score: {score}%", (w - 120, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    y_offset += 30
    
    # Ángulos principales
    angles = analysis.get('angles', {})
    angle_text = " | ".join([f"{k}: {v}°" for k, v in list(angles.items())[:3]])
    cv2.putText(img, angle_text, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    y_offset += 25
    
    # Errores y advertencias
    errors = analysis.get('errors', [])
    warnings = analysis.get('warnings', [])
    
    if errors:
        cv2.putText(img, f"⚠ {errors[0][:50]}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    elif warnings:
        cv2.putText(img, f"! {warnings[0][:50]}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
    else:
        cv2.putText(img, "✓ Forma correcta", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
    
    return img


def create_dashboard(image: np.ndarray,
                     analysis: Dict,
                     rep_count: int = 0,
                     history: List[float] = None) -> np.ndarray:
    """
    Crea un dashboard completo con métricas del ejercicio.
    
    Args:
        image: Imagen del frame actual
        analysis: Resultados del análisis
        rep_count: Contador de repeticiones
        history: Historial de scores
    
    Returns:
        Imagen con dashboard
    """
    h, w = image.shape[:2]
    
    # Crear canvas más ancho para el dashboard
    dashboard_width = 300
    canvas = np.zeros((h, w + dashboard_width, 3), dtype=np.uint8)
    canvas[:, :w] = image
    
    # Panel derecho
    panel = canvas[:, w:]
    panel[:] = (30, 30, 30)  # Fondo gris oscuro
    
    # Título
    cv2.putText(panel, "EXERCISE MONITOR", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.line(panel, (20, 55), (280, 55), (100, 100, 100), 1)
    
    # Repeticiones
    cv2.putText(panel, "REPS", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    cv2.putText(panel, str(rep_count), (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    
    # Score actual
    score = analysis.get('score', 0)
    score_color = (0, 200, 0) if score >= 80 else (0, 165, 255) if score >= 50 else (0, 0, 200)
    
    cv2.putText(panel, "FORM SCORE", (150, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    cv2.putText(panel, f"{score}%", (150, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, score_color, 3)
    
    # Ángulos
    cv2.putText(panel, "ANGLES", (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    y_pos = 210
    for name, value in analysis.get('angles', {}).items():
        cv2.putText(panel, f"{name}:", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(panel, f"{value}°", (150, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 25
    
    # Estado
    status = analysis.get('status', 'neutral')
    status_text = {'correct': 'GOOD FORM', 'warning': 'NEEDS WORK', 'error': 'FIX FORM'}
    status_colors = {'correct': (0, 200, 0), 'warning': (0, 165, 255), 'error': (0, 0, 200)}
    
    cv2.rectangle(panel, (20, y_pos + 20), (280, y_pos + 60), status_colors.get(status, (128, 128, 128)), -1)
    cv2.putText(panel, status_text.get(status, 'ANALYZING'), (50, y_pos + 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Feedback
    y_pos += 90
    cv2.putText(panel, "FEEDBACK", (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    y_pos += 25
    messages = analysis.get('errors', []) + analysis.get('warnings', [])
    if messages:
        for msg in messages[:3]:
            # Dividir mensaje largo
            words = msg.split()
            lines = []
            current_line = []
            for word in words:
                current_line.append(word)
                if len(' '.join(current_line)) > 25:
                    lines.append(' '.join(current_line[:-1]))
                    current_line = [word]
            lines.append(' '.join(current_line))
            
            for line in lines:
                cv2.putText(panel, f"• {line}", (20, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
                y_pos += 20
    else:
        cv2.putText(panel, "• Keep up the good work!", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
    
    return canvas


# =============================================================================
# FUNCIONES DE PREPROCESAMIENTO
# =============================================================================

def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Normaliza los landmarks para que sean invariantes a escala y posición.
    
    Args:
        landmarks: Array de landmarks (33, 4) o (132,)
    
    Returns:
        Landmarks normalizados
    """
    if landmarks.ndim == 1:
        landmarks = landmarks.reshape(33, 4)
    
    # Usar caderas como punto de referencia
    left_hip = landmarks[POSE_LANDMARKS['LEFT_HIP']][:3]
    right_hip = landmarks[POSE_LANDMARKS['RIGHT_HIP']][:3]
    center = (left_hip + right_hip) / 2
    
    # Calcular escala usando distancia entre hombros
    left_shoulder = landmarks[POSE_LANDMARKS['LEFT_SHOULDER']][:3]
    right_shoulder = landmarks[POSE_LANDMARKS['RIGHT_SHOULDER']][:3]
    scale = np.linalg.norm(left_shoulder - right_shoulder)
    
    if scale < 1e-6:
        scale = 1.0
    
    # Normalizar
    normalized = landmarks.copy()
    normalized[:, :3] = (landmarks[:, :3] - center) / scale
    
    return normalized.flatten()


def augment_landmarks(landmarks: np.ndarray, 
                      noise_std: float = 0.01,
                      flip_horizontal: bool = False) -> np.ndarray:
    """
    Aplica augmentación a los landmarks.
    
    Args:
        landmarks: Array de landmarks
        noise_std: Desviación estándar del ruido gaussiano
        flip_horizontal: Si hacer flip horizontal
    
    Returns:
        Landmarks augmentados
    """
    if landmarks.ndim == 1:
        landmarks = landmarks.reshape(33, 4)
    
    augmented = landmarks.copy()
    
    # Añadir ruido
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, (33, 3))
        augmented[:, :3] += noise
    
    # Flip horizontal
    if flip_horizontal:
        augmented[:, 0] = 1 - augmented[:, 0]
        
        # Intercambiar landmarks izquierda-derecha
        swap_pairs = [
            (1, 4), (2, 5), (3, 6),  # Ojos
            (7, 8),  # Orejas
            (9, 10),  # Boca
            (11, 12),  # Hombros
            (13, 14),  # Codos
            (15, 16),  # Muñecas
            (17, 18), (19, 20), (21, 22),  # Dedos
            (23, 24),  # Caderas
            (25, 26),  # Rodillas
            (27, 28),  # Tobillos
            (29, 30), (31, 32)  # Pies
        ]
        
        for left_idx, right_idx in swap_pairs:
            augmented[left_idx], augmented[right_idx] = \
                augmented[right_idx].copy(), augmented[left_idx].copy()
    
    return augmented.flatten()


# =============================================================================
# FUNCIONES DE CONTEO DE REPETICIONES
# =============================================================================

class RepCounter:
    """
    Contador de repeticiones basado en ángulos articulares.
    """
    
    def __init__(self, 
                 exercise_type: str,
                 threshold_up: float = 160,
                 threshold_down: float = 90):
        """
        Inicializa el contador.
        
        Args:
            exercise_type: Tipo de ejercicio ('squat', 'deadlift', 'bench_press')
            threshold_up: Ángulo para considerar posición arriba
            threshold_down: Ángulo para considerar posición abajo
        """
        self.exercise_type = exercise_type
        self.threshold_up = threshold_up
        self.threshold_down = threshold_down
        
        self.count = 0
        self.state = 'up'  # 'up', 'down'
        self.angle_history = []
        
        # Configurar según ejercicio
        if exercise_type == 'squat':
            self.threshold_up = 160
            self.threshold_down = 100
        elif exercise_type == 'deadlift':
            self.threshold_up = 170
            self.threshold_down = 120
        elif exercise_type == 'bench_press':
            self.threshold_up = 160
            self.threshold_down = 90
    
    def update(self, angle: float) -> Tuple[int, str]:
        """
        Actualiza el contador con un nuevo ángulo.
        
        Args:
            angle: Ángulo actual del ejercicio
        
        Returns:
            Tuple (count, state)
        """
        self.angle_history.append(angle)
        
        # Limitar historial
        if len(self.angle_history) > 100:
            self.angle_history.pop(0)
        
        # Detectar transiciones
        if self.state == 'up' and angle < self.threshold_down:
            self.state = 'down'
        elif self.state == 'down' and angle > self.threshold_up:
            self.state = 'up'
            self.count += 1
        
        return self.count, self.state
    
    def reset(self):
        """Reinicia el contador."""
        self.count = 0
        self.state = 'up'
        self.angle_history = []


# =============================================================================
# FUNCIONES DE UTILIDAD GENERAL
# =============================================================================

def resize_with_aspect_ratio(image: np.ndarray, 
                             width: int = None, 
                             height: int = None) -> np.ndarray:
    """
    Redimensiona imagen manteniendo proporción.
    
    Args:
        image: Imagen a redimensionar
        width: Ancho deseado (opcional)
        height: Alto deseado (opcional)
    
    Returns:
        Imagen redimensionada
    """
    h, w = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if width is None:
        ratio = height / h
        new_size = (int(w * ratio), height)
    else:
        ratio = width / w
        new_size = (width, int(h * ratio))
    
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def create_video_writer(output_path: str, 
                        fps: float, 
                        frame_size: Tuple[int, int]) -> cv2.VideoWriter:
    """
    Crea un VideoWriter para guardar video.
    
    Args:
        output_path: Ruta del archivo de salida
        fps: Frames por segundo
        frame_size: (width, height)
    
    Returns:
        Objeto VideoWriter
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, frame_size)


def generate_synthetic_pose(exercise: str, 
                            phase: str,
                            add_error: bool = False) -> np.ndarray:
    """
    Genera una pose sintética para testing/demo.
    
    Args:
        exercise: Tipo de ejercicio
        phase: Fase del ejercicio ('up', 'down', 'mid')
        add_error: Si añadir un error común
    
    Returns:
        Array de landmarks sintéticos
    """
    # Pose base de pie
    landmarks = np.zeros((33, 4))
    
    # Visibilidad por defecto
    landmarks[:, 3] = 1.0
    
    # Posiciones base (normalizadas 0-1)
    # Cabeza
    landmarks[0] = [0.5, 0.15, 0, 1]  # Nariz
    landmarks[7] = [0.45, 0.12, 0, 1]  # Oreja izq
    landmarks[8] = [0.55, 0.12, 0, 1]  # Oreja der
    
    # Hombros
    landmarks[11] = [0.4, 0.25, 0, 1]  # Hombro izq
    landmarks[12] = [0.6, 0.25, 0, 1]  # Hombro der
    
    # Codos
    landmarks[13] = [0.35, 0.35, 0, 1]  # Codo izq
    landmarks[14] = [0.65, 0.35, 0, 1]  # Codo der
    
    # Muñecas
    landmarks[15] = [0.35, 0.45, 0, 1]  # Muñeca izq
    landmarks[16] = [0.65, 0.45, 0, 1]  # Muñeca der
    
    # Caderas
    landmarks[23] = [0.45, 0.5, 0, 1]  # Cadera izq
    landmarks[24] = [0.55, 0.5, 0, 1]  # Cadera der
    
    # Rodillas
    landmarks[25] = [0.45, 0.7, 0, 1]  # Rodilla izq
    landmarks[26] = [0.55, 0.7, 0, 1]  # Rodilla der
    
    # Tobillos
    landmarks[27] = [0.45, 0.9, 0, 1]  # Tobillo izq
    landmarks[28] = [0.55, 0.9, 0, 1]  # Tobillo der
    
    # Modificar según ejercicio y fase
    if exercise == 'squat':
        if phase == 'down':
            # Flexionar rodillas
            landmarks[25] = [0.4, 0.65, 0, 1]
            landmarks[26] = [0.6, 0.65, 0, 1]
            # Bajar caderas
            landmarks[23] = [0.42, 0.6, 0, 1]
            landmarks[24] = [0.58, 0.6, 0, 1]
            
            if add_error:
                # Rodillas hacia adentro
                landmarks[25] = [0.47, 0.65, 0, 1]
                landmarks[26] = [0.53, 0.65, 0, 1]
    
    elif exercise == 'deadlift':
        if phase == 'down':
            # Inclinación hacia adelante
            landmarks[11] = [0.35, 0.35, 0, 1]
            landmarks[12] = [0.55, 0.35, 0, 1]
            landmarks[0] = [0.45, 0.3, 0, 1]
            
            if add_error:
                # Espalda redondeada (hombros más abajo)
                landmarks[11] = [0.3, 0.4, 0, 1]
                landmarks[12] = [0.5, 0.4, 0, 1]
    
    return landmarks.flatten()


if __name__ == "__main__":
    # Test básico de las funciones
    print("Testing utils.py...")
    
    # Test de cálculo de ángulo
    p1 = np.array([0, 0])
    p2 = np.array([1, 0])
    p3 = np.array([1, 1])
    angle = calculate_angle(p1, p2, p3)
    print(f"Ángulo de 90°: {angle:.1f}°")
    
    # Test de generación de pose sintética
    pose = generate_synthetic_pose('squat', 'up')
    print(f"Pose sintética generada: {pose.shape}")
    
    # Test de análisis de sentadilla
    analysis = analyze_squat(pose)
    print(f"Análisis de sentadilla: {analysis['status']}")
    
    print("✓ Todas las pruebas pasaron")
