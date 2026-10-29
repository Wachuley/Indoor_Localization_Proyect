"""Genera evidencias del reporte de avance de localización en interiores.

Uso:
  python generar_evidencias.py --datos datos_entrenamiento.csv --predicciones resultados/predicciones_prueba.csv

Si todavía no tienes predicciones_prueba.csv, omite --predicciones: obtendrás
las gráficas exploratorias y el baseline reproducido del mismo split.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES = ["acc_x", "acc_y", "acc_z", "mag_x", "mag_y", "mag_z", "azimut_z", "pitch_x", "roll_y"]
TARGETS = ["coordenada_x", "coordenada_y"]
MODELOS = {
    "Regresión lineal": "regresion_lineal",
    "Polinomial grado 2": "regresion_polinomial_grado_2",
    "Splines": "splines",
}


def guardar_figura(out: Path, stem: str):
    plt.tight_layout()
    plt.savefig(out / f"{stem}.png", dpi=180, bbox_inches="tight")
    plt.close()


def metricas(y, y_hat):
    return {
        "mae_x": mean_absolute_error(y[:, 0], y_hat[:, 0]),
        "mae_y": mean_absolute_error(y[:, 1], y_hat[:, 1]),
        "rmse_x": np.sqrt(mean_squared_error(y[:, 0], y_hat[:, 0])),
        "rmse_y": np.sqrt(mean_squared_error(y[:, 1], y_hat[:, 1])),
        "distancia_media": np.mean(np.linalg.norm(y-y_hat, axis=1)),
        "r2": r2_score(y, y_hat),
    }


def main():
    pa = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument('--datos', type=Path, default=Path('datos_entrenamiento.csv'))
    pa.add_argument('--predicciones', type=Path, default=Path('resultados/predicciones_prueba.csv'))
    pa.add_argument('--salida', type=Path, default=Path('evidencias_reporte'))
    args = pa.parse_args()
    if not args.datos.is_file():
        sys.exit(f"ERROR: No se encontró {args.datos}. Indica la ruta con --datos.")
    args.salida.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.datos)
    requeridas = FEATURES + TARGETS + ['sesion_inferida']
    faltan = [c for c in requeridas if c not in df.columns]
    if faltan:
        sys.exit(f"ERROR: Faltan columnas requeridas en el CSV: {faltan}")
    df = df.dropna(subset=TARGETS + ['sesion_inferida']).copy()
    for c in FEATURES + TARGETS:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=TARGETS)
    if df.empty:
        sys.exit('ERROR: No hay filas válidas con ambas coordenadas.')

    # Gráfica 1. Distribución espacial de lugares observados.
    fig, ax = plt.subplots(figsize=(7.2, 5))
    ax.scatter(df['coordenada_x'], df['coordenada_y'], s=7, alpha=.28, label='Mediciones')
    ax.set(title='Distribución espacial de las mediciones', xlabel='Coordenada X (unidad local)', ylabel='Coordenada Y (unidad local)')
    ax.grid(alpha=.2)
    guardar_figura(args.salida, '01_distribucion_coordenadas')

    # Gráfica 2. Cobertura por sesión.
    conteos = df.groupby('sesion_inferida').size().sort_index()
    conteos.rename('mediciones').to_csv(args.salida / 'conteo_por_sesion.csv', header=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([str(x) for x in conteos.index], conteos.values)
    ax.set(title='Cantidad de mediciones por sesión inferida', xlabel='Sesión inferida', ylabel='Número de mediciones')
    ax.grid(axis='y', alpha=.2)
    guardar_figura(args.salida, '02_mediciones_por_sesion')

    # Reproducimos la misma división usada en el entrenamiento.
    splitter = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=42)
    X, y, grupos = df[FEATURES], df[TARGETS], df['sesion_inferida']
    idx_train, idx_eval = next(splitter.split(X, y, groups=grupos))
    y_train = y.iloc[idx_train].to_numpy()
    y_eval = y.iloc[idx_eval].to_numpy()
    g_eval = grupos.iloc[idx_eval].to_numpy()
    media_train = np.mean(y_train, axis=0)
    baseline_pred = np.tile(media_train, (len(y_eval), 1))
    resultados = [{'modelo':'Baseline: media entrenamiento', **metricas(y_eval, baseline_pred)}]
    print(f"Mediciones totales: {len(df)}")
    print(f"Entrenamiento: {len(idx_train)}; evaluación: {len(idx_eval)}")
    print(f"Sesiones entrenamiento: {len(set(grupos.iloc[idx_train]))}; evaluación: {len(set(g_eval))}")
    print(f"Media entrenamiento X, Y: {media_train[0]:.5f}, {media_train[1]:.5f}")

    predicciones_disponibles = False
    if args.predicciones.is_file():
        pred = pd.read_csv(args.predicciones)
        real_x, real_y = 'coordenada_x_real', 'coordenada_y_real'
        if len(pred) != len(y_eval) or real_x not in pred or real_y not in pred:
            print('AVISO: Predicciones no coinciden en tamaño/columnas con el split recreado. Se omite comparación de modelos.')
        else:
            y_csv = pred[[real_x, real_y]].to_numpy(dtype=float)
            if not np.allclose(y_csv, y_eval, atol=1e-7, rtol=1e-8, equal_nan=True):
                print('AVISO: El orden de las predicciones no coincide con el split recreado. No se mezclan filas diferentes.')
            elif 'sesion_inferida' in pred and not np.array_equal(pred['sesion_inferida'].astype(str).to_numpy(), g_eval.astype(str)):
                print('AVISO: Las sesiones difieren de las del split recreado; se omite comparación.')
            else:
                predicciones_disponibles = True
                detalles_sesion = []
                for nombre, prefijo in MODELOS.items():
                    colx, coly = f'{prefijo}_x', f'{prefijo}_y'
                    if colx not in pred or coly not in pred:
                        print(f'AVISO: No se encontraron {colx}, {coly}; se omite {nombre}.')
                        continue
                    yh = pred[[colx, coly]].to_numpy(dtype=float)
                    resultados.append({'modelo':nombre, **metricas(y_eval, yh)})
                    dist = np.linalg.norm(y_eval - yh, axis=1)
                    for sesion, sub in pd.DataFrame({'sesion':g_eval, 'distancia':dist}).groupby('sesion'):
                        detalles_sesion.append({'modelo':nombre, 'sesion':sesion,
                                                'mediciones':len(sub), 'distancia_media':sub['distancia'].mean()})
                if detalles_sesion:
                    pd.DataFrame(detalles_sesion).to_csv(args.salida / 'errores_por_sesion.csv', index=False)
                    mejores = pd.DataFrame(detalles_sesion)
                    if 'Splines' in mejores['modelo'].values:
                        mejores = mejores[mejores['modelo'] == 'Splines']
                    else:
                        mejores = mejores[mejores['modelo'] == mejores['modelo'].iloc[0]]
                    fig, ax = plt.subplots(figsize=(8, 4.5))
                    ax.bar([str(v) for v in mejores['sesion']], mejores['distancia_media'])
                    ax.set(title=f"Error medio por sesión: {mejores['modelo'].iloc[0]}",
                           xlabel='Sesión de evaluación', ylabel='Distancia de error (unidad local)')
                    ax.grid(axis='y', alpha=.2)
                    guardar_figura(args.salida, '04_error_por_sesion')
    else:
        print(f'AVISO: No existe {args.predicciones}; se generará el baseline y la exploración sin comparar modelos.')

    res = pd.DataFrame(resultados)
    res.to_csv(args.salida / 'metricas_modelos_y_baseline.csv', index=False, float_format='%.6f')
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(res['modelo'], res['distancia_media'])
    ax.invert_yaxis()
    ax.set(title='Comparación con el modelo de referencia', xlabel='Distancia media de error (unidad local)')
    ax.grid(axis='x', alpha=.2)
    guardar_figura(args.salida, '03_comparacion_baseline')
    print('\nMétricas sobre las mismas sesiones de evaluación:')
    print(res.to_string(index=False, float_format=lambda n:f'{n:.4f}'))
    print(f'\nArchivos generados en: {args.salida.resolve()}')
    if not predicciones_disponibles:
        print('Revisa predicciones_prueba.csv para obtener también error por sesión y comparación de modelos.')

if __name__ == '__main__':
    main()
