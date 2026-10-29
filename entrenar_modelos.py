
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, SplineTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# 1. CONFIGURACION
# ============================================================

ARCHIVO_DATOS = Path("datos_entrenamiento.csv")
CARPETA_SALIDA = Path("resultados")

CARPETA_SALIDA.mkdir(exist_ok=True)

# Mediciones del smartphone que utilizaremos como entradas.
# Excluimos gyro_x, gyro_y y gyro_z porque son constantes.
VARIABLES_SENSOR = [
    "acc_x",
    "acc_y",
    "acc_z",
    "mag_x",
    "mag_y",
    "mag_z",
    "azimut_z",
    "pitch_x",
    "roll_y",
]

OBJETIVOS = ["coordenada_x", "coordenada_y"]

# ============================================================
# 2. CARGA Y VALIDACION
# ============================================================

if not ARCHIVO_DATOS.exists():
    raise FileNotFoundError(
        f"No se encontro el archivo: {ARCHIVO_DATOS.resolve()}"
    )

datos = pd.read_csv(ARCHIVO_DATOS)

columnas_requeridas = (
    VARIABLES_SENSOR
    + OBJETIVOS
    + ["sesion_inferida"]
)

faltantes = [
    columna
    for columna in columnas_requeridas
    if columna not in datos.columns
]

if faltantes:
    raise ValueError(
        f"Faltan columnas requeridas en el CSV: {faltantes}"
    )

# Conservamos solo filas con objetivos y sesión válidos.
datos = datos.dropna(
    subset=OBJETIVOS + ["sesion_inferida"]
).copy()

# Convertimos las entradas y objetivos a valores numéricos.
for columna in VARIABLES_SENSOR + OBJETIVOS:
    datos[columna] = pd.to_numeric(
        datos[columna], errors="coerce"
    )

# Las filas sin coordenadas válidas no sirven para entrenar.
datos = datos.dropna(subset=OBJETIVOS)

# X: entradas del modelo.
X = datos[VARIABLES_SENSOR]

# y: coordenadas que queremos predecir.
y = datos[OBJETIVOS]

# Grupos usados para separar sesiones completas.
grupos = datos["sesion_inferida"]

if grupos.nunique() < 2:
    raise ValueError(
        "Se necesitan al menos dos sesiones para separar "
        "entrenamiento y prueba."
    )

print("=== DATOS ===")
print(f"Mediciones disponibles: {len(datos)}")
print(f"Variables predictoras: {len(VARIABLES_SENSOR)}")
print(f"Sesiones disponibles: {grupos.nunique()}")
print(f"Coordenadas objetivo: {OBJETIVOS}")

# ============================================================
# 3. SEPARACION ENTRENAMIENTO / PRUEBA
# ============================================================

# Separamos sesiones completas, no filas aleatorias.
separador = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=42,
)

indices_entrenamiento, indices_prueba = next(
    separador.split(X, y, groups=grupos)
)

X_train = X.iloc[indices_entrenamiento]
X_test = X.iloc[indices_prueba]

y_train = y.iloc[indices_entrenamiento]
y_test = y.iloc[indices_prueba]

sesiones_train = set(
    grupos.iloc[indices_entrenamiento]
)
sesiones_test = set(
    grupos.iloc[indices_prueba]
)

assert sesiones_train.isdisjoint(sesiones_test)

print("\n=== SEPARACION ===")
print(f"Mediciones de entrenamiento: {len(X_train)}")
print(f"Mediciones de prueba: {len(X_test)}")
print(f"Sesiones de entrenamiento: {len(sesiones_train)}")
print(f"Sesiones de prueba: {len(sesiones_test)}")

# ============================================================
# 4. DEFINICION DE MODELOS
# ============================================================

modelos = {
    "Regresion lineal": Pipeline([
        ("imputacion", SimpleImputer(strategy="median")),
        ("escalado", StandardScaler()),
        ("modelo", LinearRegression()),
    ]),

    "Regresion polinomial grado 2": Pipeline([
        ("imputacion", SimpleImputer(strategy="median")),
        ("polinomios", PolynomialFeatures(
            degree=2,
            include_bias=False,
        )),
        ("escalado", StandardScaler()),
        ("modelo", Ridge(alpha=1.0)),
    ]),

    "Splines": Pipeline([
        ("imputacion", SimpleImputer(strategy="median")),
        ("splines", SplineTransformer(
            n_knots=5,
            degree=3,
            include_bias=False,
        )),
        ("escalado", StandardScaler()),
        ("modelo", Ridge(alpha=1.0)),
    ]),
}

# ============================================================
# 5. ENTRENAMIENTO Y EVALUACION
# ============================================================

resultados = []
predicciones = pd.DataFrame({
    "sesion_inferida": grupos.iloc[indices_prueba].to_numpy(),
    "coordenada_x_real": y_test["coordenada_x"].to_numpy(),
    "coordenada_y_real": y_test["coordenada_y"].to_numpy(),
})

# Agregamos metadatos útiles si existen.
for columna in ["timestamp", "visita_id", "place_id"]:
    if columna in datos.columns:
        predicciones[columna] = (
            datos.iloc[indices_prueba][columna].to_numpy()
        )

for nombre, modelo in modelos.items():
    print(f"\nEntrenando: {nombre}...")

    modelo.fit(X_train, y_train)

    pred = modelo.predict(X_test)

    # Error absoluto medio para cada coordenada.
    mae_x, mae_y = mean_absolute_error(
        y_test, pred, multioutput="raw_values"
    )

    # Raíz del error cuadrático medio para cada coordenada.
    rmse_x, rmse_y = np.sqrt(
        mean_squared_error(
            y_test, pred, multioutput="raw_values"
        )
    )

    # Distancia entre coordenadas reales y predichas.
    errores_x = y_test["coordenada_x"].to_numpy() - pred[:, 0]
    errores_y = y_test["coordenada_y"].to_numpy() - pred[:, 1]

    distancia_media = np.mean(
        np.sqrt(errores_x**2 + errores_y**2)
    )

    r2 = r2_score(y_test, pred)

    resultados.append({
        "modelo": nombre,
        "MAE_X": mae_x,
        "MAE_Y": mae_y,
        "RMSE_X": rmse_x,
        "RMSE_Y": rmse_y,
        "Error_distancia_media": distancia_media,
        "R2": r2,
    })

    # Guardamos las predicciones para analizarlas después.
    nombre_archivo = (
        nombre.lower()
        .replace(" ", "_")
        .replace("ó", "o")
    )

    predicciones[f"{nombre_archivo}_x"] = pred[:, 0]
    predicciones[f"{nombre_archivo}_y"] = pred[:, 1]

    joblib.dump(
        modelo,
        CARPETA_SALIDA / f"{nombre_archivo}.joblib",
    )

    print(f"MAE X: {mae_x:.4f}")
    print(f"MAE Y: {mae_y:.4f}")
    print(f"Error medio de distancia: {distancia_media:.4f}")
    print(f"R²: {r2:.4f}")

# ============================================================
# 6. GUARDAR Y COMPARAR RESULTADOS
# ============================================================

tabla_resultados = pd.DataFrame(resultados)

tabla_resultados = tabla_resultados.sort_values(
    "Error_distancia_media",
    ascending=True,
)

tabla_resultados.to_csv(
    CARPETA_SALIDA / "comparacion_modelos.csv",
    index=False,
)

predicciones.to_csv(
    CARPETA_SALIDA / "predicciones_prueba.csv",
    index=False,
)

print("\n=== COMPARACION FINAL ===")
print(
    tabla_resultados.to_string(
        index=False,
        float_format=lambda valor: f"{valor:.4f}",
    )
)

print("\nArchivos guardados en la carpeta 'resultados'.")