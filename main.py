from ultralytics import YOLO
import torch
import os
from datetime import datetime
import shutil
from pathlib import Path


def entrenar_detector():
    print("=" * 60)
    print(" ENTRENAMIENTO DE DETECTOR DE PLACAS")
    print("=" * 60)
    print("  SIN EARLY STOPPING - 100 ÉPOCAS COMPLETAS")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n Dispositivo: {device}")
    if torch.cuda.is_available():
        print(f" GPU: {torch.cuda.get_device_name(0)}")
        print(f" Memoria: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("️  Usando CPU (será más lento)")

    print("\n Cargando YOLOv8n pre-entrenado...")
    modelo = YOLO('yolov8n.pt')

    nombre_experimento = f"detector_placas_100ep_{datetime.now().strftime('%Y%m%d_%H%M')}"

    print(f"\n️ Configuración de entrenamiento:")
    print(f"    Experimento: {nombre_experimento}")
    print(f"    Épocas: 100 (completas)")
    print(f"    Batch size: 16")
    print(f"    Tamaño imagen: 640")
    print(f"    Early Stopping: DESACTIVADO")
    print(f"\n  Tiempo estimado:")
    print(f"   - CPU: 4-8 horas")
    print(f"   - GPU: 30-60 minutos")

    input("\n️  Presiona ENTER para comenzar el entrenamiento...")

    try:
        resultados = modelo.train(
            data='data.yaml',
            epochs=100,
            imgsz=640,
            batch=16,
            name=nombre_experimento,
            device=device,

            # Optimización
            lr0=0.001,
            weight_decay=0.0005,

            # Augmentación
            hsv_h=0.015,
            hsv_s=0.5,
            hsv_v=0.3,

            degrees=3.0,
            translate=0.1,
            scale=0.3,
            shear=1.0,

            perspective=0.0,

            flipud=0.0,
            fliplr=0.0,

            mosaic=0.3,
            mixup=0.0,

            patience=0
        )


        print("\n" + "=" * 60)
        print(" ENTRENAMIENTO COMPLETADO!")
        print("=" * 60)
        print(f" 100 épocas completadas exitosamente")

        from pathlib import Path

        posibles_rutas = list(Path('runs/detect').rglob(f'{nombre_experimento}/weights/best.pt'))

        if posibles_rutas:
            mejor_modelo = posibles_rutas[0]
            print(f"\n Mejor modelo: {mejor_modelo}")

            os.makedirs('modelo_final', exist_ok=True)

            destino = Path('runs/best.pt')
            shutil.copy2(mejor_modelo, destino)
            print(f" Modelo copiado a: {destino}")

            print("\n MÉTRICAS FINALES:")
            if hasattr(resultados, 'results_dict'):
                metrics = resultados.results_dict
                print(f"   mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
                print(f"   mAP50:    {metrics.get('metrics/mAP50(B)', 'N/A'):.4f}")
                print(f"   Precision: {metrics.get('metrics/precision(B)', 'N/A'):.4f}")
                print(f"   Recall:    {metrics.get('metrics/recall(B)', 'N/A'):.4f}")

            print("\n Para usar el modelo:")
            print("   1. Cierra y reinicia Streamlit")
            print("   2. El nuevo modelo se cargará automáticamente")
            print("   3. Prueba con tus imágenes")

        else:
            print("\n️  No se encontró el modelo best.pt")
            print("   Revisa la carpeta runs/detect manualmente")

        return resultados

    except KeyboardInterrupt:
        print("\n\n️  Entrenamiento interrumpido por el usuario")
        print("   El modelo guardado hasta la última época está en:")
        print("   runs/detect/{nombre_experimento}/weights/last.pt")

    except Exception as e:
        print(f"\n Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
if __name__ == "__main__":
    entrenar_detector()