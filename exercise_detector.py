"""
Exercise Form Detector - Detector de Ejecución de Ejercicios
============================================================

Sistema de Computer Vision para detectar y analizar la ejecución
de ejercicios de gimnasio en tiempo real, utilizando MediaPipe
para estimación de pose y redes neuronales para clasificación.

Ejercicios soportados:
- Sentadilla (Squat)
- Peso Muerto (Deadlift)
- Press de Banca (Bench Press)

Autor: César Adrián Delgado Díaz
Fecha: Diciembre 2025
Licencia: MIT
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import os

# Importar utilidades locales
from utils import (
    extract_landmarks,
    analyze_squat,
    analyze_deadlift,
    analyze_bench_press,
    draw_landmarks,
    draw_status_overlay,
    create_dashboard,
    normalize_landmarks,
    RepCounter,
    resize_with_aspect_ratio,
    create_video_writer,
    COLORS
)


class ExerciseType(Enum):
    """Tipos de ejercicios soportados."""
    SQUAT = "squat"
    DEADLIFT = "deadlift"
    BENCH_PRESS = "bench_press"


@dataclass
class AnalysisResult:
    """Resultado del análisis de un frame."""
    exercise: str
    phase: str
    status: str
    score: float
    angles: Dict[str, float]
    errors: List[str]
    warnings: List[str]
    rep_count: int = 0
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario."""
        return {
            'exercise': self.exercise,
            'phase': self.phase,
            'status': self.status,
            'score': self.score,
            'angles': self.angles,
            'errors': self.errors,
            'warnings': self.warnings,
            'rep_count': self.rep_count,
            'timestamp': self.timestamp
        }


@dataclass
class SessionStats:
    """Estadísticas de una sesión de ejercicio."""
    exercise_type: str
    total_reps: int = 0
    correct_reps: int = 0
    avg_score: float = 0.0
    min_score: float = 100.0
    max_score: float = 0.0
    scores: List[float] = field(default_factory=list)
    errors_count: Dict[str, int] = field(default_factory=dict)
    duration_seconds: float = 0.0
    
    def update(self, analysis: Dict):
        """Actualiza estadísticas con nuevo análisis."""
        score = analysis.get('score', 0)
        self.scores.append(score)
        self.avg_score = np.mean(self.scores)
        self.min_score = min(self.min_score, score)
        self.max_score = max(self.max_score, score)
        
        # Contar errores
        for error in analysis.get('errors', []):
            self.errors_count[error] = self.errors_count.get(error, 0) + 1
    
    def summary(self) -> str:
        """Genera resumen de la sesión."""
        return f"""
╔══════════════════════════════════════════╗
║         SESSION SUMMARY                   ║
╠══════════════════════════════════════════╣
║  Exercise: {self.exercise_type:<28} ║
║  Total Reps: {self.total_reps:<26} ║
║  Correct Reps: {self.correct_reps:<24} ║
║  Avg Score: {self.avg_score:>6.1f}%                     ║
║  Best Score: {self.max_score:>5.1f}%                     ║
║  Lowest Score: {self.min_score:>5.1f}%                   ║
║  Duration: {self.duration_seconds:>6.1f}s                    ║
╚══════════════════════════════════════════╝
"""


class ExerciseFormDetector:
    """
    Detector de forma en ejercicios usando Computer Vision.
    
    Esta clase proporciona funcionalidades para:
    - Detectar pose en tiempo real usando MediaPipe
    - Analizar la forma del ejercicio
    - Contar repeticiones automáticamente
    - Proporcionar feedback visual
    - Exportar métricas y videos procesados
    
    Attributes:
        exercise_type (str): Tipo de ejercicio a analizar
        min_detection_confidence (float): Confianza mínima para detección
        min_tracking_confidence (float): Confianza mínima para tracking
    
    Example:
        >>> detector = ExerciseFormDetector()
        >>> detector.run_realtime(exercise_type='squat')
        
        >>> results = detector.analyze_video('workout.mp4', 'squat')
        >>> print(f"Total reps: {results['rep_count']}")
    """
    
    def __init__(self,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 model_complexity: int = 1):
        """
        Inicializa el detector de ejercicios.
        
        Args:
            min_detection_confidence: Confianza mínima para detección (0-1)
            min_tracking_confidence: Confianza mínima para tracking (0-1)
            model_complexity: Complejidad del modelo (0, 1, 2)
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity
        
        # Inicializar MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = None  # Se inicializa en run_realtime o analyze_video
        
        # Funciones de análisis por ejercicio
        self.analyzers = {
            'squat': analyze_squat,
            'deadlift': analyze_deadlift,
            'bench_press': analyze_bench_press
        }
        
        # Estado interno
        self.rep_counter = None
        self.session_stats = None
        self.is_running = False
        
        # Historial de análisis
        self.analysis_history: List[AnalysisResult] = []
        
    def _init_pose(self):
        """Inicializa el detector de pose de MediaPipe."""
        if self.pose is not None:
            self.pose.close()
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
    
    def _get_analyzer(self, exercise_type: str) -> Callable:
        """
        Obtiene la función de análisis para un tipo de ejercicio.
        
        Args:
            exercise_type: Tipo de ejercicio
            
        Returns:
            Función de análisis
            
        Raises:
            ValueError: Si el ejercicio no está soportado
        """
        if exercise_type not in self.analyzers:
            raise ValueError(
                f"Ejercicio '{exercise_type}' no soportado. "
                f"Opciones: {list(self.analyzers.keys())}"
            )
        return self.analyzers[exercise_type]
    
    def analyze_frame(self, 
                      frame: np.ndarray, 
                      exercise_type: str) -> Tuple[np.ndarray, Dict]:
        """
        Analiza un frame individual.
        
        Args:
            frame: Frame BGR de OpenCV
            exercise_type: Tipo de ejercicio
            
        Returns:
            Tuple de (frame procesado, diccionario de análisis)
        """
        # Convertir a RGB para MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detectar pose
        results = self.pose.process(rgb_frame)
        
        # Extraer landmarks
        landmarks = extract_landmarks(results, flatten=False)
        
        analysis = {
            'exercise': exercise_type,
            'phase': 'unknown',
            'status': 'no_detection',
            'score': 0,
            'angles': {},
            'errors': [],
            'warnings': [],
            'landmarks_detected': landmarks is not None
        }
        
        if landmarks is not None:
            # Obtener función de análisis
            analyzer = self._get_analyzer(exercise_type)
            analysis = analyzer(landmarks)
            
            # Actualizar contador de repeticiones
            if self.rep_counter is not None:
                # Usar el ángulo principal del ejercicio
                main_angle = analysis.get('angles', {}).get('knee', 180)
                if exercise_type == 'bench_press':
                    main_angle = analysis.get('angles', {}).get('elbow', 180)
                
                rep_count, state = self.rep_counter.update(main_angle)
                analysis['rep_count'] = rep_count
            
            # Actualizar estadísticas de sesión
            if self.session_stats is not None:
                self.session_stats.update(analysis)
            
            # Dibujar landmarks
            frame = draw_landmarks(frame, landmarks)
        
        # Dibujar overlay de estado
        frame = draw_status_overlay(frame, analysis)
        
        return frame, analysis
    
    def analyze_video(self,
                      video_path: str,
                      exercise_type: str,
                      output_path: Optional[str] = None,
                      show_preview: bool = False,
                      callback: Optional[Callable] = None) -> Dict:
        """
        Analiza un video completo de ejercicio.
        
        Args:
            video_path: Ruta al archivo de video
            exercise_type: Tipo de ejercicio
            output_path: Ruta para guardar video procesado (opcional)
            show_preview: Si mostrar preview en tiempo real
            callback: Función callback llamada por cada frame
            
        Returns:
            Diccionario con resultados completos del análisis
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video no encontrado: {video_path}")
        
        # Inicializar componentes
        self._init_pose()
        self.rep_counter = RepCounter(exercise_type)
        self.session_stats = SessionStats(exercise_type=exercise_type)
        self.analysis_history = []
        
        # Abrir video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise IOError(f"No se pudo abrir el video: {video_path}")
        
        # Obtener propiedades del video
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Preparar writer si se especificó output
        writer = None
        if output_path:
            writer = create_video_writer(output_path, fps, (width + 300, height))
        
        start_time = time.time()
        frame_count = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Analizar frame
                processed_frame, analysis = self.analyze_frame(frame, exercise_type)
                
                # Crear dashboard
                dashboard_frame = create_dashboard(
                    processed_frame, 
                    analysis,
                    self.rep_counter.count,
                    self.session_stats.scores
                )
                
                # Guardar análisis
                result = AnalysisResult(
                    exercise=analysis['exercise'],
                    phase=analysis['phase'],
                    status=analysis['status'],
                    score=analysis['score'],
                    angles=analysis.get('angles', {}),
                    errors=analysis.get('errors', []),
                    warnings=analysis.get('warnings', []),
                    rep_count=self.rep_counter.count,
                    timestamp=frame_count / fps
                )
                self.analysis_history.append(result)
                
                # Escribir frame procesado
                if writer:
                    writer.write(dashboard_frame)
                
                # Preview
                if show_preview:
                    preview = resize_with_aspect_ratio(dashboard_frame, width=900)
                    cv2.imshow('Exercise Analysis', preview)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Callback
                if callback:
                    callback(frame_count, total_frames, analysis)
                
                frame_count += 1
                
        finally:
            cap.release()
            if writer:
                writer.release()
            if show_preview:
                cv2.destroyAllWindows()
            self.pose.close()
        
        # Actualizar estadísticas finales
        self.session_stats.total_reps = self.rep_counter.count
        self.session_stats.correct_reps = sum(
            1 for r in self.analysis_history 
            if r.status == 'correct' and r.phase == 'bottom'
        )
        self.session_stats.duration_seconds = time.time() - start_time
        
        return {
            'video_path': video_path,
            'exercise_type': exercise_type,
            'rep_count': self.rep_counter.count,
            'avg_score': self.session_stats.avg_score,
            'min_score': self.session_stats.min_score,
            'max_score': self.session_stats.max_score,
            'total_frames': frame_count,
            'duration_seconds': self.session_stats.duration_seconds,
            'errors_summary': self.session_stats.errors_count,
            'output_path': output_path,
            'history': [r.to_dict() for r in self.analysis_history]
        }
    
    def run_realtime(self,
                     exercise_type: str,
                     camera_id: int = 0,
                     window_name: str = 'Exercise Form Detector',
                     output_path: Optional[str] = None,
                     mirror: bool = True) -> Dict:
        """
        Ejecuta análisis en tiempo real desde webcam.
        
        Args:
            exercise_type: Tipo de ejercicio
            camera_id: ID de la cámara (0 = webcam por defecto)
            window_name: Nombre de la ventana
            output_path: Ruta para guardar grabación (opcional)
            mirror: Si hacer efecto espejo
            
        Returns:
            Diccionario con estadísticas de la sesión
        
        Controls:
            - 'q': Salir
            - 'r': Reiniciar contador
            - 's': Capturar screenshot
            - 'p': Pausar/Reanudar
        """
        # Inicializar componentes
        self._init_pose()
        self.rep_counter = RepCounter(exercise_type)
        self.session_stats = SessionStats(exercise_type=exercise_type)
        self.analysis_history = []
        self.is_running = True
        
        # Abrir cámara
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise IOError(f"No se pudo abrir la cámara {camera_id}")
        
        # Configurar resolución
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Obtener propiedades
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Preparar writer
        writer = None
        if output_path:
            writer = create_video_writer(output_path, fps, (width + 300, height))
        
        start_time = time.time()
        frame_count = 0
        paused = False
        
        print(f"\n{'='*50}")
        print(f"Exercise Form Detector - {exercise_type.upper()}")
        print(f"{'='*50}")
        print("Controles:")
        print("  [Q] Salir")
        print("  [R] Reiniciar contador")
        print("  [S] Screenshot")
        print("  [P] Pausar/Reanudar")
        print(f"{'='*50}\n")
        
        try:
            while self.is_running and cap.isOpened():
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("Error leyendo frame de la cámara")
                        break
                    
                    # Efecto espejo
                    if mirror:
                        frame = cv2.flip(frame, 1)
                    
                    # Analizar frame
                    processed_frame, analysis = self.analyze_frame(frame, exercise_type)
                    
                    # Crear dashboard
                    dashboard_frame = create_dashboard(
                        processed_frame,
                        analysis,
                        self.rep_counter.count,
                        self.session_stats.scores[-50:] if self.session_stats.scores else []
                    )
                    
                    # Guardar
                    if writer:
                        writer.write(dashboard_frame)
                    
                    frame_count += 1
                
                # Mostrar
                cv2.imshow(window_name, dashboard_frame)
                
                # Manejar teclas
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.rep_counter.reset()
                    self.session_stats = SessionStats(exercise_type=exercise_type)
                    print("✓ Contador reiniciado")
                elif key == ord('s'):
                    screenshot_path = f"screenshot_{int(time.time())}.png"
                    cv2.imwrite(screenshot_path, dashboard_frame)
                    print(f"✓ Screenshot guardado: {screenshot_path}")
                elif key == ord('p'):
                    paused = not paused
                    print(f"{'⏸ Pausado' if paused else '▶ Reanudado'}")
                    
        finally:
            self.is_running = False
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            self.pose.close()
        
        # Estadísticas finales
        self.session_stats.total_reps = self.rep_counter.count
        self.session_stats.duration_seconds = time.time() - start_time
        
        print(self.session_stats.summary())
        
        return {
            'exercise_type': exercise_type,
            'rep_count': self.rep_counter.count,
            'avg_score': self.session_stats.avg_score,
            'duration_seconds': self.session_stats.duration_seconds,
            'errors_summary': self.session_stats.errors_count
        }
    
    def analyze_image(self,
                      image: Union[str, np.ndarray],
                      exercise_type: str,
                      draw_output: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Analiza una imagen individual.
        
        Args:
            image: Ruta a imagen o array numpy BGR
            exercise_type: Tipo de ejercicio
            draw_output: Si dibujar anotaciones en la imagen
            
        Returns:
            Tuple de (imagen procesada, análisis)
        """
        # Cargar imagen si es ruta
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Imagen no encontrada: {image}")
            image = cv2.imread(image)
        
        # Inicializar pose para imagen estática
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence
        )
        
        try:
            processed, analysis = self.analyze_frame(image, exercise_type)
            
            if draw_output:
                return processed, analysis
            return image, analysis
            
        finally:
            self.pose.close()
    
    def get_supported_exercises(self) -> List[str]:
        """Retorna lista de ejercicios soportados."""
        return list(self.analyzers.keys())
    
    def get_analysis_history(self) -> List[Dict]:
        """Retorna historial de análisis."""
        return [r.to_dict() for r in self.analysis_history]
    
    def export_metrics(self, output_path: str):
        """
        Exporta métricas a archivo CSV.
        
        Args:
            output_path: Ruta del archivo de salida
        """
        import csv
        
        if not self.analysis_history:
            print("No hay datos para exportar")
            return
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'timestamp', 'exercise', 'phase', 'status', 
                'score', 'rep_count', 'errors', 'warnings'
            ])
            writer.writeheader()
            
            for result in self.analysis_history:
                row = result.to_dict()
                row['errors'] = '; '.join(row['errors'])
                row['warnings'] = '; '.join(row['warnings'])
                del row['angles']
                writer.writerow(row)
        
        print(f"✓ Métricas exportadas a: {output_path}")


class ExerciseClassifierModel:
    """
    Modelo CNN para clasificación de errores en ejercicios.
    
    Este modelo toma los landmarks de MediaPipe como entrada
    y clasifica el estado del ejercicio en:
    - correct: Forma correcta
    - partial_error: Error menor/advertencia
    - major_error: Error significativo
    """
    
    def __init__(self, input_shape: int = 132, num_classes: int = 3):
        """
        Inicializa el modelo de clasificación.
        
        Args:
            input_shape: Número de features de entrada (33 landmarks * 4)
            num_classes: Número de clases de salida
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.class_names = ['correct', 'partial_error', 'major_error']
        
    def build_model(self):
        """Construye la arquitectura del modelo."""
        try:
            from tensorflow import keras
            from tensorflow.keras import layers
        except ImportError:
            raise ImportError(
                "TensorFlow no está instalado. "
                "Instalar con: pip install tensorflow"
            )
        
        self.model = keras.Sequential([
            # Capa de entrada
            layers.Input(shape=(self.input_shape,)),
            
            # Normalización
            layers.BatchNormalization(),
            
            # Capas densas
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            
            # Capa de salida
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: np.ndarray = None,
              y_val: np.ndarray = None,
              epochs: int = 50,
              batch_size: int = 32) -> Dict:
        """
        Entrena el modelo.
        
        Args:
            X_train: Datos de entrenamiento (N, 132)
            y_train: Etiquetas de entrenamiento (N, 3) one-hot
            X_val: Datos de validación (opcional)
            y_val: Etiquetas de validación (opcional)
            epochs: Número de épocas
            batch_size: Tamaño del batch
            
        Returns:
            Historial de entrenamiento
        """
        if self.model is None:
            self.build_model()
        
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        return history.history
    
    def predict(self, landmarks: np.ndarray) -> Tuple[str, float]:
        """
        Predice la clase para un conjunto de landmarks.
        
        Args:
            landmarks: Array de landmarks (132,) o (N, 132)
            
        Returns:
            Tuple de (clase predicha, confianza)
        """
        if self.model is None:
            raise ValueError("Modelo no construido. Llamar build_model() primero.")
        
        # Asegurar shape correcto
        if landmarks.ndim == 1:
            landmarks = landmarks.reshape(1, -1)
        
        # Normalizar
        landmarks_norm = normalize_landmarks(landmarks.reshape(-1, 33, 4))
        landmarks_norm = landmarks_norm.reshape(1, -1)
        
        # Predecir
        predictions = self.model.predict(landmarks_norm, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        
        return self.class_names[class_idx], float(confidence)
    
    def save(self, path: str):
        """Guarda el modelo."""
        if self.model is not None:
            self.model.save(path)
            print(f"✓ Modelo guardado en: {path}")
    
    def load(self, path: str):
        """Carga el modelo."""
        from tensorflow import keras
        self.model = keras.models.load_model(path)
        print(f"✓ Modelo cargado desde: {path}")
    
    def summary(self):
        """Muestra resumen del modelo."""
        if self.model is not None:
            self.model.summary()


def demo_synthetic():
    """
    Demostración con datos sintéticos.
    
    Útil para probar el sistema sin cámara ni videos.
    """
    from utils import generate_synthetic_pose
    
    print("\n" + "="*50)
    print("DEMO - Análisis de Poses Sintéticas")
    print("="*50)
    
    detector = ExerciseFormDetector()
    detector._init_pose()
    
    exercises = ['squat', 'deadlift', 'bench_press']
    phases = ['up', 'down']
    
    for exercise in exercises:
        print(f"\n--- {exercise.upper()} ---")
        for phase in phases:
            # Generar pose correcta
            pose = generate_synthetic_pose(exercise, phase, add_error=False)
            analysis = detector._get_analyzer(exercise)(pose)
            print(f"  {phase}: Score={analysis['score']}%, Status={analysis['status']}")
            
            # Generar pose con error
            pose_error = generate_synthetic_pose(exercise, phase, add_error=True)
            analysis_error = detector._get_analyzer(exercise)(pose_error)
            print(f"  {phase} (con error): Score={analysis_error['score']}%, "
                  f"Errors={analysis_error['errors'][:1]}")


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Exercise Form Detector - Detector de forma en ejercicios'
    )
    parser.add_argument(
        '--mode', 
        choices=['realtime', 'video', 'demo'],
        default='demo',
        help='Modo de ejecución'
    )
    parser.add_argument(
        '--exercise',
        choices=['squat', 'deadlift', 'bench_press'],
        default='squat',
        help='Tipo de ejercicio'
    )
    parser.add_argument(
        '--video',
        type=str,
        help='Ruta al video (modo video)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Ruta para guardar video procesado'
    )
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='ID de cámara (modo realtime)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        demo_synthetic()
        
    elif args.mode == 'realtime':
        detector = ExerciseFormDetector()
        results = detector.run_realtime(
            exercise_type=args.exercise,
            camera_id=args.camera,
            output_path=args.output
        )
        
    elif args.mode == 'video':
        if not args.video:
            print("Error: Se requiere --video para el modo video")
        else:
            detector = ExerciseFormDetector()
            results = detector.analyze_video(
                video_path=args.video,
                exercise_type=args.exercise,
                output_path=args.output,
                show_preview=True
            )
            print(f"\nResultados:")
            print(f"  Repeticiones: {results['rep_count']}")
            print(f"  Score promedio: {results['avg_score']:.1f}%")
