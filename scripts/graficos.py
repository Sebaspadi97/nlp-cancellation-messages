"""
¿Qué mensajes anuncian una cancelación? · SP Lab · Gráficos de la marca a partir de las tablas (carpeta tablas/)
Uso: python scripts/graficos.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(RAIZ, "tablas")
OUT = os.path.join(RAIZ, "graficos")
os.makedirs(OUT, exist_ok=True)

FUENTES = os.path.join(RAIZ, "fuentes")
for f in sorted(os.listdir(FUENTES)) if os.path.isdir(FUENTES) else []:
    if f.endswith(".ttf"):
        fm.fontManager.addfont(os.path.join(FUENTES, f))
_disp = {x.name for x in fm.fontManager.ttflist}
TITULO = "Montserrat" if "Montserrat" in _disp else "DejaVu Sans"
TEXTO = "Open Sans" if "Open Sans" in _disp else "DejaVu Sans"
plt.rcParams.update({"font.family": TEXTO, "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "#F8FAF7", "axes.facecolor": "#F8FAF7", "hatch.linewidth": 1.2})
VERDE, GRIS, CARBON, DORADO, NEUTRO, MENTA, ROJO = "#245B45", "#66736C", "#202824", "#D6A64F", "#B9C2BC", "#8CCFAD", "#A3332D"
DORADO_TXT = "#8A6420"
fmt = lambda v, d=2: f"{v:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")
PIE = "Datos sintéticos de un ejercicio académico de maestría (1.200 mensajes, 20 frases base) · Sebas Padilla Lab"


def encabezado(fig, t, s):
    fig.text(0.02, 0.965, t, fontsize=15, fontweight="bold", fontfamily=TITULO, color=CARBON, va="top")
    fig.text(0.02, 0.905, s, fontsize=10.5, color=GRIS, va="top")
    fig.text(0.02, 0.02, PIE, fontsize=7.5, color=GRIS)


# 1 · Torneo: F1 macro con frases nunca vistas
m = pd.read_csv(os.path.join(TAB, "matriz_comparativa.csv"))
nombres = {
    "TF-IDF palabras + caracteres": "TF-IDF palabras + caracteres",
    "TF-IDF caracteres": "TF-IDF n-gramas de caracteres",
    "TF-IDF lemas": "TF-IDF lemas",
    "TF-IDF palabras": "TF-IDF palabras",
    "fastText promedio (limpio, por pliegue)": "fastText, corpus limpio",
    "Word2Vec promedio (limpio, por pliegue)": "Word2Vec, corpus limpio",
    "fastText promedio (corpus original)": "fastText, corpus con etiqueta",
    "Word2Vec promedio (corpus original)": "Word2Vec, corpus con etiqueta",
}
m["nombre"] = m.representacion.map(nombres)
m = m.sort_values("F1_grupo").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(9.5, 6.4)); fig.subplots_adjust(left=0.3, right=0.95, top=0.8, bottom=0.15)
y = np.arange(len(m))
for i, r in m.iterrows():
    campeon = r.representacion == "TF-IDF caracteres"
    if not r.valida:
        ax.barh(i, r.F1_grupo, color="none", edgecolor=ROJO, hatch="///", lw=1.2, height=0.62)
    else:
        ax.barh(i, r.F1_grupo, color=VERDE if campeon else NEUTRO, height=0.62)
    etiqueta = fmt(r.F1_grupo) + ("  · fuga: no válida" if not r.valida else "  · campeón" if campeon else "")
    ax.text(r.F1_grupo + 0.012, i, etiqueta, va="center", fontsize=9.5,
            color=ROJO if not r.valida else CARBON, fontweight="bold" if campeon else "normal")
ax.axvline(0.20, color=DORADO, ls="--", lw=1.4)
ax.text(0.205, len(m) - 0.35, "Azar con 5 categorías: 0,20", color=DORADO_TXT, fontsize=9.5, fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels(m.nombre); ax.tick_params(axis="y", length=0, colors=CARBON)
ax.tick_params(axis="x", colors=GRIS); ax.set_xlim(0, 1.05)
ax.xaxis.set_major_formatter(lambda v, _: fmt(v, 1))
ax.set_xlabel("F1 macro con frases nunca vistas (validación agrupada, 4 pliegues) · 1 es perfecto", color=GRIS)
ax.spines["left"].set_visible(False); ax.spines["bottom"].set_color("#C9CFCB")
encabezado(fig, "Los caracteres duplican a las palabras; los embeddings solo ganaban con fuga",
           "F1 macro de 8 representaciones del texto, con el mismo clasificador y los mismos pliegues")
fig.savefig(os.path.join(OUT, "01_torneo_representaciones.png"), dpi=200); plt.close(fig)

# 2 · Resultado por categoría del campeón
c = pd.read_csv(os.path.join(TAB, "f1_por_categoria_campeon.csv")).sort_values("f1").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(9.5, 5.6)); fig.subplots_adjust(left=0.2, right=0.95, top=0.78, bottom=0.17)
y = np.arange(len(c))
colores = [DORADO if k == "Cancelación" else (VERDE if f >= 0.8 else NEUTRO) for k, f in zip(c.categoria, c.f1)]
ax.barh(y, c.f1, color=colores, height=0.6)
for i, r in c.iterrows():
    texto = fmt(r.f1) + (f"  · encuentra {r.aciertos} de {r.mensajes}" if r.categoria == "Cancelación"
                         else "  · no reconoce ningún mensaje" if r.f1 == 0 else "")
    ax.text(max(r.f1, 0) + 0.012, i, texto, va="center", fontsize=9.5, color=CARBON,
            fontweight="bold" if r.categoria == "Cancelación" else "normal")
ax.set_yticks(y); ax.set_yticklabels(c.categoria); ax.tick_params(axis="y", length=0, colors=CARBON)
ax.tick_params(axis="x", colors=GRIS); ax.set_xlim(0, 1.05); ax.xaxis.set_major_formatter(lambda v, _: fmt(v, 1))
ax.set_xlabel("F1 por categoría (predicciones fuera de pliegue sobre los 1.200 mensajes)", color=GRIS)
ax.spines["left"].set_visible(False); ax.spines["bottom"].set_color("#C9CFCB")
encabezado(fig, "El campeón domina dos categorías y no reconoce otras dos",
           "TF-IDF de n-gramas de caracteres; Cancelación es la categoría crítica")
fig.savefig(os.path.join(OUT, "02_resultado_por_categoria.png"), dpi=200); plt.close(fig)

# 3 · Umbral de alerta de Cancelación
a = pd.read_csv(os.path.join(TAB, "umbral_alerta_cancelacion.csv"))
fig, ax = plt.subplots(figsize=(9.5, 5.8)); fig.subplots_adjust(left=0.08, right=0.95, top=0.78, bottom=0.17)
ax.plot(a.umbral, a.recall, color=VERDE, lw=2.4, marker="o", ms=5, label="Recall: cancelaciones encontradas")
ax.plot(a.umbral, a.precision, color=GRIS, lw=2, marker="o", ms=4, ls="--", label="Precisión: alertas que eran ciertas")
ax.plot(a.umbral, a.mensajes_alertados, color=NEUTRO, lw=1.6, label="Mensajes alertados")
sel = a.loc[a.umbral.round(2).eq(0.15)].iloc[0]
ax.axvline(0.15, color=DORADO, ls="--", lw=1.4)
ax.scatter([0.15], [sel.recall], s=130, color=DORADO, zorder=5, edgecolor=CARBON, lw=0.8)
ax.annotate(f"Umbral elegido 0,15\nrecall {fmt(sel.recall)} · precisión {fmt(sel.precision)}",
            xy=(0.15, sel.recall), xytext=(0.2, 0.9), fontsize=9.5, color=DORADO_TXT, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=DORADO, lw=1))
ax.set_ylim(0, 1.08); ax.set_xlim(0.03, 0.52)
ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f} %"); ax.xaxis.set_major_formatter(lambda v, _: fmt(v))
ax.tick_params(colors=GRIS); ax.set_xlabel("Umbral: probabilidad mínima de Cancelación para alertar", color=GRIS)
ax.spines["left"].set_color("#C9CFCB"); ax.spines["bottom"].set_color("#C9CFCB")
ax.text(0.27, 0.56, "Recall: cancelaciones encontradas", color=VERDE, fontsize=9.5, fontweight="bold")
ax.text(0.27, 0.37, "Precisión: alertas que eran ciertas", color=GRIS, fontsize=9.5, fontweight="bold")
ax.text(0.27, 0.15, "Mensajes alertados", color=GRIS, fontsize=9.5)
encabezado(fig, "Con un umbral de 0,15, la alerta encuentra 8 de cada 10 cancelaciones",
           "Bajar el umbral recupera más casos a cambio de más falsas alarmas; cada alerta la revisa un agente")
fig.savefig(os.path.join(OUT, "03_umbral_cancelacion.png"), dpi=200); plt.close(fig)
print("Gráficos guardados en", OUT)
