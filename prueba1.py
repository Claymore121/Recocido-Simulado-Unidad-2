
import math
import os
import random
import re
import time
import unicodedata

import matplotlib

matplotlib.use("Agg")  # guarda las graficas en archivo sin abrir ventanas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Asegurar que el script se ejecute en el directorio del archivo
os.chdir(os.path.dirname(os.path.abspath(__file__)))

SEMILLA = 7            # cambiarla genera otro escenario reproducible
VELOCIDAD_KMH = 35.0   # velocidad urbana promedio en Culiacan


# ----------------------------------------------------------------------------
# Funciones de apoyo
# ----------------------------------------------------------------------------
def normalizar(texto):
  """Minusculas y sin acentos: 'Centro de Distribución' -> 'centro de distribucion'."""
  texto = unicodedata.normalize("NFKD", str(texto))
  return "".join(c for c in texto if not unicodedata.combining(c)).strip().lower()


def _es_secuencia(valores):
  """True si los valores son 0,1,2,... o 1,2,3,... (identificadores de nodo)."""
  v = np.asarray(valores, dtype=float)
  return len(v) > 1 and not np.isnan(v).any() and np.allclose(np.diff(v), 1)


def leer_matriz(archivo):
  """Lee una matriz cuadrada de Excel quitando encabezados de fila/columna.

  pd.read_excel(...).values toma la primera fila como encabezado y, si el
  Excel trae una columna con los nombres de los nodos, la mete en la matriz
  y desplaza todos los costos una columna. Aqui se detectan ambos casos.
  """
  crudo = pd.read_excel(archivo, header=None)
  num = crudo.apply(pd.to_numeric, errors="coerce")

  # Fila / columna de encabezados de texto (ej. "Nodo_0", "CEDIS 1")
  if num.iloc[0].isna().mean() > 0.5:
    num = num.iloc[1:, :]
  if num.iloc[:, 0].isna().mean() > 0.5:
    num = num.iloc[:, 1:]

  # Fila / columna de identificadores numericos (0, 1, 2, ...)
  filas, cols = num.shape
  if filas == cols + 1:
    num = num.iloc[1:, :]
  elif cols == filas + 1:
    num = num.iloc[:, 1:]
  elif _es_secuencia(num.iloc[0, 1:]) and _es_secuencia(num.iloc[1:, 0]):
    num = num.iloc[1:, 1:]

  return num.to_numpy(dtype=float)


# ----------------------------------------------------------------------------
# Optimizador
# ----------------------------------------------------------------------------
class OptimizadorRecocidoSimulado:

  def __init__(self, archivo_tiendas, archivo_distancias, archivo_costos):
    """Carga los datos y separa los nodos en CEDIS y tiendas."""
    self.df = pd.read_excel(archivo_tiendas)
    self.matriz_dist = leer_matriz(archivo_distancias)
    self.matriz_costos = leer_matriz(archivo_costos)
    self.n = len(self.matriz_costos)

    # Listas de Python: acceder a C[i][j] es mucho mas rapido que en numpy
    self._C = self.matriz_costos.tolist()
    self._D = self.matriz_dist.tolist()

    self.centros, self.tiendas = self._identificar_nodos()
    self.etiquetas = self._obtener_etiquetas()
    self.coords, self.coords_reales = self._obtener_coordenadas()

    print(
        f"[OK] Datos cargados: {self.n} nodos -> {len(self.centros)} CEDIS y"
        f" {len(self.tiendas)} tiendas."
    )

  # ------------------------------------------------------------------ datos
  def _columna(self, *opciones, prefijo=False):
    """Busca una columna del Excel de tiendas por nombre (sin acentos)."""
    for col in self.df.columns:
      nombre = normalizar(col)
      for op in opciones:
        if nombre == op or (prefijo and nombre.startswith(op)):
          return col
    return None

  def _identificar_nodos(self):
    """Usa la columna 'Tipo' para saber que nodos son CEDIS."""
    centros = []
    col_tipo = self._columna("tipo")
    if col_tipo is not None:
      tipos = self.df[col_tipo].map(normalizar)
      patron = r"centro|cedis|distribu"
      centros = [i for i in range(min(len(tipos), self.n))
                 if re.search(patron, tipos.iloc[i])]
    if not centros:
      print("[AVISO] No se reconocieron CEDIS en 'Tipo'; se usan los nodos 0-9.")
      centros = list(range(min(10, self.n)))
    tiendas = [i for i in range(self.n) if i not in set(centros)]
    return centros, tiendas

  def _obtener_etiquetas(self):
    """Nombre corto de cada nodo para tablas y graficas."""
    col = self._columna("nombre", "sucursal", "id", "clave", "nodo")
    etiquetas = []
    for i in range(self.n):
      if col is not None and i < len(self.df) and pd.notna(self.df[col].iloc[i]):
        etiquetas.append(str(self.df[col].iloc[i]))
      else:
        etiquetas.append(f"Nodo {i}")
    return etiquetas

  def _obtener_coordenadas(self):
    """Usa Latitud/Longitud si existen; si no, las reconstruye con MDS."""
    col_lat = self._columna("lat", prefijo=True)
    col_lon = self._columna("lon", "lng", prefijo=True)
    if col_lat is not None and col_lon is not None and len(self.df) >= self.n:
      xy = self.df[[col_lon, col_lat]].iloc[: self.n].to_numpy(dtype=float)
      if not np.isnan(xy).any():
        return xy, True

    # MDS clasico: posiciones 2D cuyas distancias se parecen a la matriz
    d = (self.matriz_dist + self.matriz_dist.T) / 2
    j = np.eye(self.n) - 1.0 / self.n
    b = -0.5 * j @ (d ** 2) @ j
    valores, vectores = np.linalg.eigh(b)
    idx = np.argsort(valores)[::-1][:2]
    xy = vectores[:, idx] * np.sqrt(np.maximum(valores[idx], 0))
    print("[AVISO] Sin columnas de latitud/longitud: el mapa usa posiciones"
          " aproximadas reconstruidas de la matriz de distancias.")
    return xy, False

  # ---------------------------------------------------------- funcion objetivo
  def costo_ruta(self, centro, ruta):
    """Combustible de un circuito: CEDIS -> tiendas en orden -> CEDIS."""
    if not ruta:
      return 0.0
    c = self._C
    total = c[centro][ruta[0]] + c[ruta[-1]][centro]
    for a, b in zip(ruta, ruta[1:]):
      total += c[a][b]
    return total

  def km_ruta(self, centro, ruta):
    """Kilometros del mismo circuito (usa la matriz de distancias)."""
    if not ruta:
      return 0.0
    d = self._D
    total = d[centro][ruta[0]] + d[ruta[-1]][centro]
    for a, b in zip(ruta, ruta[1:]):
      total += d[a][b]
    return total

  def costo_total(self, solucion):
    """Funcion objetivo z: suma del combustible de todas las rutas."""
    return sum(self.costo_ruta(c, r) for c, r in solucion.items())

  def km_total(self, solucion):
    return sum(self.km_ruta(c, r) for c, r in solucion.items())

  # --------------------------------------------------------- solucion inicial
  def solucion_inicial(self, max_tiendas=None):
    """Asigna cada tienda al CEDIS mas barato (ida + vuelta) con lugar libre.

    Se recorren los pares (CEDIS, tienda) del mas barato al mas caro; asi
    cada tienda cae en el CEDIS mas cercano que todavia tenga cupo. El orden
    de visita dentro de cada ruta es aleatorio (la referencia sin optimizar).
    """
    cupo = max_tiendas if max_tiendas is not None else len(self.tiendas)
    pares = sorted(
        (self._C[c][t] + self._C[t][c], c, t)
        for c in self.centros for t in self.tiendas
    )
    solucion = {c: [] for c in self.centros}
    asignadas = set()
    for _, c, t in pares:
      if t not in asignadas and len(solucion[c]) < cupo:
        solucion[c].append(t)
        asignadas.add(t)
    for ruta in solucion.values():
      random.shuffle(ruta)
    return solucion

  # ----------------------------------------------------------------- vecinos
  def generar_vecino(self, solucion, max_tiendas=None):
    """Aplica un movimiento al azar y devuelve solo las rutas que cambian.

    - swap:        intercambia 2 tiendas de la misma ruta
    - 2-opt:       invierte un tramo de la ruta (deshace cruces)
    - reubicar:    mueve una tienda a la ruta de otro CEDIS
    - intercambio: intercambia tiendas entre dos CEDIS
    """
    con_tiendas = [c for c, r in solucion.items() if r]
    largas = [c for c, r in solucion.items() if len(r) >= 2]
    p = random.random()

    if p < 0.35 and largas:  # swap
      c = random.choice(largas)
      ruta = list(solucion[c])
      i, j = random.sample(range(len(ruta)), 2)
      ruta[i], ruta[j] = ruta[j], ruta[i]
      return {c: ruta}

    if p < 0.70 and largas:  # 2-opt
      c = random.choice(largas)
      ruta = list(solucion[c])
      i, j = sorted(random.sample(range(len(ruta)), 2))
      ruta[i:j + 1] = reversed(ruta[i:j + 1])
      return {c: ruta}

    if len(self.centros) < 2 or not con_tiendas:
      return {}

    if p < 0.90:  # reubicar en otro CEDIS
      a = random.choice(con_tiendas)
      b = random.choice([c for c in self.centros if c != a])
      if max_tiendas is not None and len(solucion[b]) >= max_tiendas:
        return {}
      ruta_a, ruta_b = list(solucion[a]), list(solucion[b])
      tienda = ruta_a.pop(random.randrange(len(ruta_a)))
      ruta_b.insert(random.randint(0, len(ruta_b)), tienda)
      return {a: ruta_a, b: ruta_b}

    # intercambio entre CEDIS
    if len(con_tiendas) < 2:
      return {}
    a, b = random.sample(con_tiendas, 2)
    ruta_a, ruta_b = list(solucion[a]), list(solucion[b])
    i, j = random.randrange(len(ruta_a)), random.randrange(len(ruta_b))
    ruta_a[i], ruta_b[j] = ruta_b[j], ruta_a[i]
    return {a: ruta_a, b: ruta_b}

  def calibrar_temperatura(self, solucion, p_inicial=0.8, muestras=500):
    """T0 tal que, al inicio, un empeoramiento promedio se acepte con p_inicial.

    De P = exp(-delta/T0) se despeja T0 = -delta_promedio / ln(p_inicial).
    """
    deltas = []
    for _ in range(muestras):
      cambios = self.generar_vecino(solucion)
      if not cambios:
        continue
      delta = (sum(self.costo_ruta(c, r) for c, r in cambios.items())
               - sum(self.costo_ruta(c, solucion[c]) for c in cambios))
      if delta > 0:
        deltas.append(delta)
    if not deltas:
      return 1.0
    return -float(np.mean(deltas)) / math.log(p_inicial)

  # ---------------------------------------------------- recocido simulado
  def optimizar_recocido(self, alpha=0.995, iter_por_temp=100,
                         factor_t_min=1e-3, p_inicial=0.8,
                         max_tiendas="auto", semilla=SEMILLA):
    """Recocido Simulado sobre las rutas de todos los CEDIS a la vez.

    max_tiendas: tiendas maximas por camion/CEDIS. "auto" = 30 % arriba del
    promedio (90 / 10 = 9 -> 12); None = sin limite (el algoritmo tenderia a
    juntar todas las tiendas en pocos CEDIS para ahorrar viajes de salida).
    """
    random.seed(semilla)
    np.random.seed(semilla)
    inicio = time.time()

    if max_tiendas == "auto":
      max_tiendas = math.ceil(1.3 * len(self.tiendas) / len(self.centros))
    self.max_tiendas = max_tiendas

    solucion = self.solucion_inicial(max_tiendas)
    inicial = {c: list(r) for c, r in solucion.items()}
    costos = {c: self.costo_ruta(c, r) for c, r in solucion.items()}
    costo_actual = sum(costos.values())

    mejor = {c: list(r) for c, r in solucion.items()}
    mejor_costo = costo_actual

    t_inicial = self.calibrar_temperatura(solucion, p_inicial)
    t_min = t_inicial * factor_t_min
    t = t_inicial

    hist_actual, hist_mejor, hist_t = [costo_actual], [mejor_costo], [t]
    evaluados = aceptados = 0

    while t > t_min:
      for _ in range(iter_por_temp):
        cambios = self.generar_vecino(solucion, max_tiendas)
        if not cambios:
          continue
        nuevos = {c: self.costo_ruta(c, r) for c, r in cambios.items()}
        delta = sum(nuevos.values()) - sum(costos[c] for c in cambios)
        evaluados += 1

        # Criterio de Metropolis
        if delta <= 0 or random.random() < math.exp(-delta / t):
          for c, r in cambios.items():
            solucion[c] = r
            costos[c] = nuevos[c]
          costo_actual += delta
          aceptados += 1
          if costo_actual < mejor_costo - 1e-9:
            mejor = {c: list(r) for c, r in solucion.items()}
            mejor_costo = costo_actual

      t *= alpha  # enfriamiento geometrico
      hist_actual.append(costo_actual)
      hist_mejor.append(mejor_costo)
      hist_t.append(t)

    mejor = self.pulir_2opt(mejor)
    self.validar_solucion(mejor)

    return {
        "inicial": inicial,
        "final": mejor,
        "costo_inicial": self.costo_total(inicial),
        "costo_final": self.costo_total(mejor),
        "historial_actual": hist_actual,
        "historial_mejor": hist_mejor,
        "historial_t": hist_t,
        "t_inicial": t_inicial,
        "t_min": t_min,
        "niveles": len(hist_actual) - 1,
        "evaluados": evaluados,
        "aceptados": aceptados,
        "segundos": time.time() - inicio,
    }

  def pulir_2opt(self, solucion):
    """Busqueda local final: aplica 2-opt en cada ruta mientras mejore."""
    pulida = {}
    for c, ruta in solucion.items():
      ruta = list(ruta)
      mejora = True
      while mejora and len(ruta) >= 3:
        mejora = False
        costo = self.costo_ruta(c, ruta)
        for i in range(len(ruta) - 1):
          for j in range(i + 1, len(ruta)):
            nueva = ruta[:i] + ruta[i:j + 1][::-1] + ruta[j + 1:]
            nuevo_costo = self.costo_ruta(c, nueva)
            if nuevo_costo < costo - 1e-9:
              ruta, costo, mejora = nueva, nuevo_costo, True
      pulida[c] = ruta
    return pulida

  def validar_solucion(self, solucion):
    """Restriccion: cada tienda aparece exactamente una vez."""
    visitadas = [t for r in solucion.values() for t in r]
    assert sorted(visitadas) == sorted(self.tiendas), (
        "Error: hay tiendas repetidas o sin visitar.")

  # ---------------------------------------------------------------- reportes
  def tabla_tramos(self, solucion):
    """Rutas punto a punto: una fila por tramo."""
    filas = []
    for k, (c, ruta) in enumerate(solucion.items(), start=1):
      if not ruta:
        continue
      circuito = [c] + ruta + [c]
      for paso, (a, b) in enumerate(zip(circuito, circuito[1:]), start=1):
        filas.append({
            "CEDIS": f"CEDIS {k}",
            "Paso": paso,
            "Desde_nodo": a,
            "Desde": self.etiquetas[a],
            "Hasta_nodo": b,
            "Hasta": self.etiquetas[b],
            "Km": round(self._D[a][b], 2),
            "Costo_USD": round(self._C[a][b], 2),
        })
    return pd.DataFrame(filas)

  def tabla_resumen(self, solucion):
    filas = []
    for k, (c, ruta) in enumerate(solucion.items(), start=1):
      km = self.km_ruta(c, ruta)
      filas.append({
          "CEDIS": f"CEDIS {k}",
          "Nodo": c,
          "Nombre": self.etiquetas[c],
          "Tiendas": len(ruta),
          "Km": round(km, 2),
          "Costo_USD": round(self.costo_ruta(c, ruta), 2),
          "Horas": round(km / VELOCIDAD_KMH, 2),
          "Secuencia": " -> ".join(str(n) for n in [c] + ruta + [c]),
      })
    return pd.DataFrame(filas)

  def imprimir_reporte(self, res):
    ini, fin = res["inicial"], res["final"]
    km_ini, km_fin = self.km_total(ini), self.km_total(fin)
    c_ini, c_fin = res["costo_inicial"], res["costo_final"]

    print("\n" + "=" * 64)
    print(" REPORTE DE RESULTADOS - RECOCIDO SIMULADO (10 CEDIS)")
    print("=" * 64)
    print(f"T inicial calibrada: {res['t_inicial']:.4f}  |  T minima:"
          f" {res['t_min']:.6f}  |  Max. tiendas por CEDIS: {self.max_tiendas}")
    print(f"Niveles de temperatura: {res['niveles']}  |  Vecinos evaluados:"
          f" {res['evaluados']:,}  |  Aceptados: {res['aceptados']:,}")
    print(f"Tiempo de computo: {res['segundos']:.1f} s\n")

    print(f"{'':22}{'Inicial':>14}{'Optimizada':>14}{'Ahorro':>14}")
    print(f"{'Costo (USD)':22}{c_ini:>14.2f}{c_fin:>14.2f}{c_ini - c_fin:>14.2f}")
    print(f"{'Distancia (km)':22}{km_ini:>14.2f}{km_fin:>14.2f}"
          f"{km_ini - km_fin:>14.2f}")
    print(f"{'Tiempo (h)':22}{km_ini / VELOCIDAD_KMH:>14.2f}"
          f"{km_fin / VELOCIDAD_KMH:>14.2f}"
          f"{(km_ini - km_fin) / VELOCIDAD_KMH:>14.2f}")
    print(f"\nMejora en costo: {100 * (c_ini - c_fin) / c_ini:.2f} %\n")

    resumen = self.tabla_resumen(fin)
    print(resumen.drop(columns=["Secuencia", "Nombre"]).to_string(index=False))
    print("\nRutas optimizadas (nodo por nodo):")
    for _, fila in resumen.iterrows():
      print(f"  {fila['CEDIS']}: {fila['Secuencia']}")

    tramos_ini = self.tabla_tramos(ini)[["Km", "Costo_USD"]]
    tramos_fin = self.tabla_tramos(fin)[["Km", "Costo_USD"]]
    print("\nEstadisticas por tramo - inicial:")
    print(tramos_ini.describe().round(2).to_string())
    print("\nEstadisticas por tramo - optimizada:")
    print(tramos_fin.describe().round(2).to_string())

  def exportar_excel(self, res, archivo="rutas_optimizadas.xlsx"):
    with pd.ExcelWriter(archivo) as escritor:
      self.tabla_resumen(res["final"]).to_excel(
          escritor, sheet_name="Resumen", index=False)
      self.tabla_tramos(res["final"]).to_excel(
          escritor, sheet_name="Rutas punto a punto", index=False)
      self.tabla_tramos(res["inicial"]).to_excel(
          escritor, sheet_name="Rutas iniciales", index=False)
      pd.DataFrame({
          "Nivel": range(len(res["historial_actual"])),
          "Temperatura": res["historial_t"],
          "Costo_actual": res["historial_actual"],
          "Mejor_costo": res["historial_mejor"],
      }).to_excel(escritor, sheet_name="Historial", index=False)
    print(f"[OK] Rutas guardadas en '{archivo}'.")

  # ---------------------------------------------------------------- graficas
  def generar_visualizacion(self, res, archivo="evolucion_costo_recocido.png"):
    """Costo actual (sube y baja) y mejor costo por nivel de temperatura."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(res["historial_actual"], color="#d62728", linewidth=1.2,
            alpha=0.85, label="Costo actual (fluctuacion termica)")
    ax.plot(res["historial_mejor"], color="#1f4e79", linewidth=2,
            label="Mejor costo encontrado")
    ax.set_title("Evolucion termica del costo - Recocido Simulado (10 CEDIS)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Nivel de temperatura")
    ax.set_ylabel("Costo total de combustible (USD)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(archivo, dpi=300)
    plt.close(fig)
    print(f"[OK] Grafica de evolucion guardada en '{archivo}'.")

  def _dibujar_rutas(self, ax, solucion, titulo):
    colores = plt.get_cmap("tab10")
    xy = self.coords
    for k, (c, ruta) in enumerate(solucion.items()):
      color = colores(k % 10)
      circuito = [c] + ruta + [c]
      pts = xy[circuito]
      km = self.km_ruta(c, ruta)
      ax.plot(pts[:, 0], pts[:, 1], "-", color=color, linewidth=1.6,
              alpha=0.9, label=f"CEDIS {k + 1}: {len(ruta)} tiendas, {km:.1f} km")
      # flechas que indican el sentido del recorrido
      for a, b in zip(pts[:-1], pts[1:]):
        ax.annotate("", xy=(a + b) / 2 + (b - a) * 0.08, xytext=(a + b) / 2,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2))
      ax.scatter(xy[ruta, 0], xy[ruta, 1], s=28, color=color,
                 edgecolor="white", linewidth=0.6, zorder=3)
      ax.scatter(xy[c, 0], xy[c, 1], s=260, marker="s", color=color,
                 edgecolor="black", linewidth=1.4, zorder=4)
      ax.annotate(f"C{k + 1}", xy[c], ha="center", va="center", fontsize=7,
                  fontweight="bold", color="white", zorder=5)
    costo = self.costo_total(solucion)
    km = self.km_total(solucion)
    ax.set_title(f"{titulo}\n{km:,.1f} km  |  ${costo:,.2f} USD", fontsize=11,
                 fontweight="bold")
    if self.coords_reales:
      ax.set_xlabel("Longitud")
      ax.set_ylabel("Latitud")
      lat_media = np.mean(xy[:, 1])
      ax.set_aspect(1 / math.cos(math.radians(lat_media)), adjustable="datalim")
    else:
      ax.set_xlabel("Posicion aproximada (reconstruida de las distancias)")
      ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle="--", alpha=0.4)

  def generar_mapa_rutas(self, res, archivo="mapa_rutas_centros.png"):
    """Recorridos de cada CEDIS: ruta inicial vs. ruta optimizada."""
    fig, ejes = plt.subplots(1, 2, figsize=(18, 9))
    self._dibujar_rutas(ejes[0], res["inicial"], "Rutas iniciales (orden aleatorio)")
    self._dibujar_rutas(ejes[1], res["final"], "Rutas optimizadas (Recocido Simulado)")
    manejadores, etiquetas = ejes[1].get_legend_handles_labels()
    fig.legend(manejadores, etiquetas, loc="lower center", ncol=5, fontsize=9,
               frameon=False)
    fig.suptitle("Recorridos de los camiones por centro de distribucion -"
                 " Culiacan", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(archivo, dpi=300)
    plt.close(fig)
    print(f"[OK] Mapa de recorridos guardado en '{archivo}'.")

  # ----------------------------------------------------------------- pruebas
  def ejecutar_pruebas_funcionalidad(self):
    """Valida estructura y contenido de los datos antes de optimizar."""
    n = self.n
    assert self.matriz_dist.shape == (n, n), (
        f"La matriz de distancias no es cuadrada: {self.matriz_dist.shape}")
    assert self.matriz_costos.shape == (n, n), (
        f"La matriz de costos no es cuadrada: {self.matriz_costos.shape}")
    assert not np.isnan(self.matriz_dist).any(), "Hay vacios en distancias."
    assert not np.isnan(self.matriz_costos).any(), "Hay vacios en costos."
    assert (self.matriz_dist >= 0).all(), "Hay distancias negativas."
    assert (self.matriz_costos >= 0).all(), "Hay costos negativos."
    assert self._columna("tipo") is not None, "Falta la columna 'Tipo'."
    assert len(self.df) == n, (
        f"El archivo de tiendas tiene {len(self.df)} filas y las matrices {n}.")
    assert self.centros and self.tiendas, "No hay CEDIS o no hay tiendas."
    if not np.allclose(np.diag(self.matriz_costos), 0):
      print("[AVISO] La diagonal de costos no es 0: revisa el formato del Excel.")

    # Prueba de la funcion objetivo con una ruta de 2 tiendas
    c, t1, t2 = self.centros[0], self.tiendas[0], self.tiendas[1]
    esperado = self._C[c][t1] + self._C[t1][t2] + self._C[t2][c]
    assert abs(self.costo_ruta(c, [t1, t2]) - esperado) < 1e-9
    print("[OK] Pruebas de funcionalidad superadas.")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
  sistema = OptimizadorRecocidoSimulado(
      "datos_distribucion_tiendas.xlsx",
      "matriz_distancias.xlsx",
      "matriz_costos_combustible.xlsx",
  )
  sistema.ejecutar_pruebas_funcionalidad()

  resultado = sistema.optimizar_recocido()

  sistema.imprimir_reporte(resultado)
  sistema.exportar_excel(resultado)
  sistema.generar_visualizacion(resultado)
  sistema.generar_mapa_rutas(resultado)
