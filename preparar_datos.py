#!/usr/bin/env python3
"""
preparar_datos.py
Integra el dataset UCI "Geo-Magnetic field and WLAN dataset for indoor
localisation from wristband and smartphone" (measure1) en tablas listas para
regresion de coordenadas (X, Y).

Uso (rutas relativas o absolutas, todas configurables):
  python preparar_datos.py \
      --sens  measure1_smartphone_sens.csv \
      --ts    measure1_timestamp_id.csv \
      --map   PointsMapping.ods \
      --wifi  measure1_smartphone_wifi.csv \
      --out   salida

Requisitos: pandas, numpy y (solo si el mapa es .ods) odfpy.
  pip install pandas numpy odfpy

Reglas de integracion (resumen; el detalle queda en informe_integracion.txt):
  1. Timestamps = epoch Unix en milisegundos (se valida el rango de fechas).
  2. Medicion con timestamp t pertenece a la visita i si t_entrada <= t < t_salida
     (intervalo semiabierto: un limite compartido nunca asigna dos lugares).
  3. place_id se une directamente con la columna ID de PointsMapping (sin offsets).
  4. No se inventan coordenadas: un place_id sin coordenadas queda fuera del
     archivo de entrenamiento y se reporta.
  5. Wi-Fi: una fila por lugar, sin timestamp -> NO se une a las mediciones;
     se exporta como tabla independiente.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

SESSION_GAP_MS = 60_000  # hueco entre visitas que se interpreta como nueva sesion

SENSOR_RENAME = {
    "AccelerationX": "acc_x", "AccelerationY": "acc_y", "AccelerationZ": "acc_z",
    "MagneticFieldX": "mag_x", "MagneticFieldY": "mag_y", "MagneticFieldZ": "mag_z",
    "Z-AxisAgle(Azimuth)": "azimut_z", "X-AxisAngle(Pitch)": "pitch_x",
    "Y-AxisAngle(Roll)": "roll_y",
    "GyroX": "gyro_x", "GyroY": "gyro_y", "GyroZ": "gyro_z",
}


def log_factory(lines):
    def log(msg=""):
        print(msg)
        lines.append(msg)
    return log


def leer_mapa(path):
    if path.lower().endswith((".ods", ".xlsx", ".xls")):
        df = pd.read_excel(path)  # .ods requiere odfpy
    else:
        df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    faltan = {"ID", "X", "Y"} - set(df.columns)
    if faltan:
        sys.exit(f"El mapa de puntos debe tener columnas ID, X, Y; faltan {faltan}")
    return df[["ID", "X", "Y"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sens", required=True)
    ap.add_argument("--ts", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--wifi", default=None)
    ap.add_argument("--out", default="salida")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    L = []
    log = log_factory(L)

    # ------------------------------------------------------------------ carga
    sens = pd.read_csv(a.sens, skipinitialspace=True)
    sens.columns = [c.strip() for c in sens.columns]
    if "timestamp" not in sens.columns:
        sys.exit("smartphone_sens: falta la columna 'timestamp'")
    sens["fila_original_sens"] = np.arange(2, len(sens) + 2)  # linea en el CSV (1 = encabezado)

    ts = pd.read_csv(a.ts, header=None, names=["t_entrada", "t_salida", "place_id"])
    ts["visita_id"] = np.arange(1, len(ts) + 1)  # orden de aparicion en el archivo original

    mapa = leer_mapa(a.map)

    # ------------------------------------------------------------ validacion
    log("=== VALIDACION DE ENTRADAS ===")
    log(f"sensores: {sens.shape[0]} filas x {sens.shape[1]-1} columnas; NaN={int(sens.isna().sum().sum())}")
    log(f"tiempos : {ts.shape[0]} filas; NaN={int(ts[['t_entrada','t_salida','place_id']].isna().sum().sum())}")
    log(f"mapa    : {mapa.shape[0]} filas; IDs unicos={mapa.ID.nunique()}; NaN={int(mapa.isna().sum().sum())}")

    for nombre, serie in [("sens.timestamp", sens.timestamp), ("ts.t_entrada", ts.t_entrada), ("ts.t_salida", ts.t_salida)]:
        f = pd.to_datetime([serie.min(), serie.max()], unit="ms")
        log(f"{nombre}: {serie.min()}..{serie.max()} -> como epoch ms: {f[0]} .. {f[1]}")
        if not (pd.Timestamp("2000-01-01") < f[0] and f[1] < pd.Timestamp("2100-01-01")):
            sys.exit(f"{nombre} no parece epoch en milisegundos; no se une (revise unidades).")
    if not (ts.t_entrada < ts.t_salida).all():
        sys.exit("Hay intervalos con t_entrada >= t_salida.")

    dup_ts = int(sens.timestamp.duplicated().sum())
    log(f"timestamps duplicados en sensores: {dup_ts}")
    fuera_de_orden = int((sens.timestamp.diff() < 0).sum())
    log(f"saltos hacia atras en el orden del archivo de sensores: {fuera_de_orden} (se ordena por timestamp; la trazabilidad se conserva con fila_original_sens)")
    if mapa.ID.duplicated().any():
        sys.exit("IDs duplicados en el mapa de puntos.")

    # --------------------------------------------- intervalos y sus problemas
    ts_ord = ts.sort_values("t_entrada").reset_index(drop=True)
    solapes = int((ts_ord.t_entrada.values[1:] < ts_ord.t_salida.values[:-1]).sum())
    comparten = int((ts_ord.t_entrada.values[1:] == ts_ord.t_salida.values[:-1]).sum())
    log(f"intervalos solapados (tras ordenar): {solapes}; que comparten limite exacto: {comparten}")
    if solapes:
        sys.exit("Hay intervalos solapados: asignacion ambigua, revise antes de continuar.")
    ts_ord["duracion_visita_ms"] = ts_ord.t_salida - ts_ord.t_entrada
    ts_ord["sesion_inferida"] = (ts_ord.t_entrada.diff() > SESSION_GAP_MS).cumsum() + 1

    # ---------------------------------------- IDs de lugar vs coordenadas
    ids_ts = set(ts.place_id)
    sin_coord = sorted(ids_ts - set(mapa.ID))
    mapa_no_visitado = sorted(set(mapa.ID) - ids_ts)
    log("")
    log("=== LUGARES ===")
    log(f"IDs distintos en tiempos: {len(ids_ts)}; visitas totales: {len(ts)}; visitas repetidas al mismo lugar: {len(ts)-len(ids_ts)} (se conservan)")
    log(f"IDs de tiempos SIN coordenadas en el mapa: {sin_coord}")
    log(f"IDs del mapa no visitados en la secuencia: {mapa_no_visitado}")

    # ----------------------------------------------- union por intervalo
    s = sens.sort_values("timestamp").reset_index(drop=True)
    t = s.timestamp.values
    pos = np.searchsorted(ts_ord.t_entrada.values, t, side="right") - 1
    pos_c = np.clip(pos, 0, None)
    dentro = (pos >= 0) & (t < ts_ord.t_salida.values[pos_c])
    s["_pos"] = np.where(dentro, pos, -1)

    asignadas = s[dentro].copy()
    asignadas = asignadas.join(
        ts_ord[["visita_id", "place_id", "duracion_visita_ms", "sesion_inferida"]],
        on="_pos")
    asignadas = asignadas.merge(mapa, how="left", left_on="place_id", right_on="ID").drop(columns=["ID"])
    asignadas = asignadas.rename(columns={"X": "coordenada_x", "Y": "coordenada_y"})

    sin_asig = s[~dentro].copy()
    sin_asig["motivo"] = "fuera_de_todo_intervalo_de_visita"

    sin_xy = asignadas[asignadas.coordenada_x.isna() | asignadas.coordenada_y.isna()].copy()
    sin_xy["motivo"] = "lugar_sin_coordenadas_en_PointsMapping"
    entren = asignadas.drop(sin_xy.index).copy()

    # --------------------------------------------------- tabla de entrenamiento
    entren = entren.rename(columns={"timestamp": "timestamp", **SENSOR_RENAME})
    cols_sens = list(SENSOR_RENAME.values())
    meta = ["timestamp", "fila_original_sens", "visita_id", "sesion_inferida", "place_id",
            "duracion_visita_ms"]
    objetivo = ["coordenada_x", "coordenada_y"]
    entren = entren[meta + cols_sens + objetivo]
    entren["place_id"] = entren.place_id.astype(int)
    entren.to_csv(os.path.join(a.out, "datos_entrenamiento.csv"), index=False)

    # ----------------------------------------------------- sin asignacion
    def a_salida(df, extra_cols):
        d = df.rename(columns={"timestamp": "timestamp", **SENSOR_RENAME})
        return d[[c for c in ["timestamp", "fila_original_sens", *extra_cols, *cols_sens, "motivo"] if c in d.columns]]

    parte1 = a_salida(sin_asig, [])
    parte2 = a_salida(sin_xy, ["visita_id", "sesion_inferida", "place_id"])
    pd.concat([parte1, parte2], ignore_index=True).to_csv(
        os.path.join(a.out, "datos_sin_asignacion.csv"), index=False)

    # ------------------------------------------------------------- wifi
    wifi_info = "No se proporciono archivo Wi-Fi."
    if a.wifi:
        wf = pd.read_csv(a.wifi, header=None)
        n_ap = wf.shape[1] - 1
        wf.columns = ["place_id"] + [f"ap_{i:03d}" for i in range(1, n_ap + 1)]
        unico = wf.place_id.is_unique
        ids_wf = set(wf.place_id)
        wf = wf.merge(mapa.rename(columns={"ID": "place_id", "X": "coordenada_x", "Y": "coordenada_y"}),
                      how="left", on="place_id")
        wf = wf[["place_id", "coordenada_x", "coordenada_y"] + [c for c in wf.columns if c.startswith("ap_")]]
        wf.to_csv(os.path.join(a.out, "datos_wifi_independientes.csv"), index=False)
        vals = wf[[c for c in wf.columns if c.startswith("ap_")]].values
        wifi_info = (f"{len(wf)} filas, {n_ap} puntos de acceso (ap_001..ap_{n_ap:03d}, en el orden de columnas original); "
                     f"place_id unico por fila: {unico}; sin timestamp ni id de medicion; "
                     f"rango RSSI {vals.min()}..{vals.max()} dBm; "
                     f"{(vals == -100).mean()*100:.1f}% de valores = -100 (no detectado); "
                     f"IDs wifi sin coordenadas: {sorted(ids_wf - set(mapa.ID))}; "
                     f"IDs del mapa sin fila wifi: {sorted(set(mapa.ID) - ids_wf)}")

    # --------------------------------------------------------- estadisticas
    n = len(s)
    cuenta = asignadas.groupby("visita_id").size()
    visitas_sin_med = len(ts) - cuenta.size
    log("")
    log("=== VALIDACION DE LA INTEGRACION ===")
    log(f"filas originales de sensores           : {n}")
    log(f"mediciones dentro de algun intervalo   : {int(dentro.sum())}")
    log(f"mediciones SIN ubicacion (fuera)       : {len(sin_asig)} ({len(sin_asig)/n*100:.2f}%)")
    log(f"mediciones con lugar pero sin coordenadas: {len(sin_xy)} (place_id {sorted(set(sin_xy.place_id.astype(int)))})")
    log(f"filas finales en datos_entrenamiento   : {len(entren)} ({len(entren)/n*100:.2f}% de las originales)")
    log(f"filas en datos_sin_asignacion          : {len(sin_asig)+len(sin_xy)}")
    log(f"control: finales + sin asignacion = {len(entren)+len(sin_asig)+len(sin_xy)} (original {n})")
    log(f"lugares distintos en la tabla final    : {entren.place_id.nunique()}")
    log(f"visitas distintas en la tabla final    : {entren.visita_id.nunique()} de {len(ts)}")
    log(f"visitas sin ninguna medicion           : {visitas_sin_med}")
    log(f"mediciones por visita: min={cuenta.min()} mediana={cuenta.median():.0f} max={cuenta.max()}")
    log(f"visitas de exactamente 2000 ms          : {(ts_ord.duracion_visita_ms==2000).sum()} (ver nota en el informe)")
    log(f"sesiones inferidas (hueco > {SESSION_GAP_MS//1000}s)    : {ts_ord.sesion_inferida.nunique()}")
    log(f"NaN en tabla final: {int(entren.isna().sum().sum())}; timestamps duplicados: {int(entren.timestamp.duplicated().sum())}")
    const = [c for c in cols_sens if entren[c].nunique() == 1]
    log(f"columnas constantes en la tabla final: {const}")
    log("")
    log("=== WI-FI ===")
    log(wifi_info)

    log("")
    log("=== NOTAS Y LIMITACIONES ===")
    for nota in [
        "1) Los timestamps de sensores y tiempos son epoch ms y sus rangos de fechas coinciden; que compartan reloj es plausible pero no esta documentado en los archivos.",
        "2) place_id se une con ID del mapa sin desplazamiento. Evidencia: dentro de cada sesion casi todos los pasos entre lugares consecutivos son de ~0.6 m con esta union (solo 1 de 306 pasos > 1 m; con un desplazamiento +1, 21 de 308).",
        "3) place_id 0 aparece en tiempos y Wi-Fi pero NO en PointsMapping (que va de 1 a 325). No se le asigno coordenada; sus mediciones estan en datos_sin_asignacion.csv.",
        "4) Las mediciones fuera de intervalos son, por lo general, trayecto entre lugares o inicio/fin de sesion; no tienen lugar conocido y no se descartaron: estan en datos_sin_asignacion.csv.",
        "5) 18 visitas duran exactamente 2000 ms (inicio/fin de sesion). Se conservan; la columna duracion_visita_ms permite filtrarlas.",
        "6) gyro_x/y/z son constantes 0 en todo el archivo; no aportan informacion.",
        "7) sesion_inferida NO existe en los datos originales: es una agrupacion derivada por huecos > 60 s entre visitas. Util para separar train/test por sesion.",
        "8) Wi-Fi: una fila por lugar (huella), sin timestamp; copiarla a cada medicion crearia columnas constantes por lugar equivalentes a un identificador. Se exporta aparte.",
        "9) El reloj inteligente (smartwatch_sens) no se integro: el objetivo es el smartphone.",
    ]:
        log(nota)

    with open(os.path.join(a.out, "informe_integracion.txt"), "w", encoding="utf-8") as f:
        f.write("INFORME DE INTEGRACION (salida automatica de preparar_datos.py)\n\n")
        f.write("\n".join(L) + "\n")
    print(f"\nArchivos escritos en: {a.out}")


if __name__ == "__main__":
    main()
