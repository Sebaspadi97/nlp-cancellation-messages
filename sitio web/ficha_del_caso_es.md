---
titulo: "¿Qué mensajes anuncian una cancelación?"
pregunta: "¿Qué representación del texto detecta mejor a un cliente que quiere irse?"
hallazgo: "con frases nuevas, los n-gramas de caracteres duplican a las palabras (F1 0,47 frente a 0,21)."
descripcion: "Torneo de representaciones de texto para clasificar mensajes de clientes y alertar a tiempo las cancelaciones, con validación que no se deja engañar por frases repetidas."
pilar: "ml"
herramientas:
  - "Python"
  - "scikit-learn"
  - "pandas"
fecha: "nov 2026"
repo: "https://github.com/Sebaspadi97/nlp-cancellation-messages"
portada: "/img/casos/cancelacion/01_torneo_representaciones.webp"
destacado: false
orden: 5
publicar: true
etiquetaDatos: "datos sintéticos"
lineaEquipo: "Ejercicio académico en equipo con Juan David Gonzalez y Nelsón Fabian Manios."
simulador: false
ficha:
  - dato: "Equipo"
    valor: "Juan David Gonzalez, Nelsón Fabian Manios y Sebastián Padilla"
  - dato: "Mi rol"
    valor: "Ciencia de Datos: auditoría de los datos, control de fuga y embeddings"
  - dato: "Método"
    valor: "Clasificación de texto · TF-IDF, n-gramas y embeddings · validación agrupada · umbrales de decisión"
  - dato: "Herramientas"
    valor: "Python · scikit-learn · pandas · numpy"
  - dato: "Contexto"
    valor: "Ejercicio académico de la Maestría en Ciencia de Datos, 2026"
  - dato: "Estado"
    valor: "Terminado · aplicación interactiva disponible"
sobreLosDatos: "Ejercicio académico. Los 1.200 mensajes son sintéticos: se crearon como material de práctica a partir de 20 frases base con ruido agregado (mayúsculas, emojis, signos, correos y enlaces ficticios). Ninguno corresponde a una persona, un cliente o una empresa real, y no hay datos del empleador del autor. Las cifras son hallazgos del análisis, no impacto real."
licencia: "Ejercicio académico. Los datos son material de práctica y no se redistribuyen. Código (de los tres autores), tablas y gráficos con todos los derechos reservados; se pueden citar con atribución al equipo."
ingles:
  titulo: "Which messages signal a cancellation?"
  pregunta: "Which text representation best detects a customer who wants to leave?"
  hallazgo: "on unseen phrases, character n-grams double word features (F1 0.47 vs 0.21)."
---

## La pregunta de negocio

> ¿Qué representación del texto permite clasificar los mensajes de los clientes y detectar a tiempo los que anuncian una cancelación?

Los mensajes llegan por chat, correo, web y WhatsApp, y hay que enrutarlos a cinco equipos: Facturación, Servicio, Entrega, Producto y Cancelación. Cancelación es la categoría más pequeña y la más costosa de fallar: si no se detecta, el cliente se va sin que nadie intente retenerlo.

- **Quién decide:** el equipo de servicio al cliente, con un comité que fija qué tan automático puede ser el enrutamiento.
- **Costo del error:** perder una cancelación cuesta más que revisar una alerta de más.

## Los datos

El conjunto tiene 1.200 mensajes etiquetados, pero la auditoría mostró que son solo 20 frases repetidas con ruido.

```cifras
1.200 | mensajes
20 | frases base distintas
9,6 % | son cancelaciones
```

| Categoría | Mensajes | Participación |
| --- | --- | --- |
| Facturación | 314 | 26,2 % |
| Servicio | 290 | 24,2 % |
| Entrega | 256 | 21,3 % |
| Producto | 225 | 18,8 % |
| Cancelación | 115 | 9,6 % |

- **Ruido agregado:** mayúsculas al azar, emojis, signos repetidos (!!!, ??), errores de escritura (“paqte”, “x” por “por”), enlaces y correos de clientes ficticios.
- **Otras columnas:** canal, fecha, prioridad y satisfacción. No se usan para predecir (ver el enfoque).
- **Dato curioso:** las 1.200 filas son apenas 219 textos distintos. Con una partición al azar, el 91 % de los mensajes de prueba ya estaba, idéntico, en entrenamiento.

## El enfoque

Primero se auditó, luego se compararon 8 representaciones del texto con una evaluación que no premia la memoria.

1. **Auditoría con 10 pruebas A/B:** duplicados, datos personales, enlaces, emojis, mayúsculas y tildes, errores de escritura, negaciones, desbalance, metadatos y el corpus de embeddings. Cada limpieza se trató como una hipótesis y se midió.
2. **Validación agrupada por frase base:** 4 pliegues en los que ninguna frase aparece a la vez en entrenamiento y en prueba. Es la única forma de saber cómo le irá al modelo con mensajes nuevos.
3. **Torneo con el mismo clasificador:** regresión logística balanceada sobre TF-IDF de palabras, de lemas, de n-gramas de caracteres y de palabras + caracteres; y promedios de Word2Vec y fastText con dos corpus.
4. **Regla de selección declarada:** si dos representaciones válidas quedan a menos de 0,01 de F1, gana la más simple.
5. **Umbrales de decisión:** sobre las probabilidades del campeón, uno para alertar cancelaciones y otro para autoclasificar.

Tres cifras altas resultaron ser fugas y se descartaron: los duplicados (F1 de 1,00 con partición al azar), la prioridad (solo marca “alta” en Facturación y Cancelación, y se asigna después de leer el mensaje) y un corpus de embeddings que traía la categoría pegada al final de cada frase.

## Los hallazgos

> Con frases que el modelo nunca vio, los n-gramas de caracteres duplican a las palabras.

**El torneo.** TF-IDF de n-gramas de caracteres obtiene un F1 macro de 0,47; TF-IDF de palabras, 0,21, apenas por encima del 0,20 de adivinar al azar. Con tan pocas frases, una nueva casi no comparte palabras con las vistas, pero sí fragmentos: “cancel” en cancelar y cancelación, “cobr” en cobro y cobraron.

![Barras de F1 macro: fastText y Word2Vec con el corpus que traía la etiqueta (0,81 y 0,75) marcados como fuga; TF-IDF palabras + caracteres 0,47; n-gramas de caracteres 0,47, campeón; lemas 0,31; embeddings con corpus limpio 0,26 y 0,25; palabras 0,21.](https://sebaspadi97.github.io/img/casos/cancelacion/01_torneo_representaciones.webp "F1 macro con validación agrupada; las barras rayadas son fugas")

**Empate y desempate.** Palabras + caracteres saca 0,471 y caracteres solos 0,466. La diferencia es menor que la variación entre pliegues (0,32 a 0,60), así que gana la opción más simple: 961 rasgos frente a 1.141.

**El promedio esconde dos categorías en cero.** El campeón acierta casi siempre en Facturación y Entrega, encuentra la mitad de las cancelaciones y no reconoce ningún mensaje de Producto ni de Servicio, cuyas frases no comparten vocabulario.

![Barras de F1 por categoría del campeón: Facturación 0,90, Entrega 0,88, Cancelación 0,40 (encuentra 59 de 115), Servicio y Producto 0.](https://sebaspadi97.github.io/img/casos/cancelacion/02_resultado_por_categoria.webp "F1 por categoría con predicciones fuera de pliegue")

**Un umbral para no perder cancelaciones.** Si se alerta cuando la probabilidad de Cancelación llega a 0,15, aunque no sea la categoría más probable, se encuentran 8 de cada 10 cancelaciones. A cambio, solo 1 de cada 4 alertas es real.

![Curvas de recall, precisión y porcentaje de mensajes alertados según el umbral; con 0,15 el recall es 0,79 y la precisión 0,24.](https://sebaspadi97.github.io/img/casos/cancelacion/03_umbral_cancelacion.webp "Recall y precisión de la alerta de Cancelación según el umbral")

| Representación | F1 macro | F1 Cancelación | Rasgos | ¿Válida? |
| --- | --- | --- | --- | --- |
| TF-IDF palabras + caracteres | 0,47 | 0,50 | 1.141 | Sí |
| **TF-IDF n-gramas de caracteres** | **0,47** | **0,50** | **961** | **Sí · campeón** |
| TF-IDF lemas | 0,31 | 0,13 | 176 | Sí |
| fastText, corpus limpio | 0,26 | 0,19 | 50 | Sí |
| Word2Vec, corpus limpio | 0,25 | 0,58 | 50 | Sí |
| TF-IDF palabras | 0,21 | 0,00 | 180 | Sí |
| fastText, corpus con etiqueta | 0,81 | 0,75 | 50 | No: fuga |
| Word2Vec, corpus con etiqueta | 0,75 | 0,75 | 50 | No: fuga |

**Explóralo.** La [aplicación interactiva del torneo](https://sebaspadi97.github.io/interactivos/mensajes-de-cancelacion/) recorre el flujo completo, explica cada concepto (TF-IDF, n-gramas, F1 macro, umbrales, embeddings en 3D) y muestra el código de Python de cada paso.

## La decisión recomendada

> No automatizar todavía: un piloto asistido en el que el modelo sugiere y una persona decide.

- **Alerta de Cancelación** con probabilidad de 0,15 o más: va a una cola de retención con revisión humana.
- **Autoclasificación** solo con confianza de 0,70 o más: hoy cubre el 7 % de los mensajes, con exactitud del 100 % en ese grupo. El resto lo clasifica un agente con la sugerencia del modelo.
- **El modelo nunca cancela, renueva ni cobra.** Las correcciones de los agentes alimentan el reentrenamiento.
- **Para pasar a automático:** F1 macro de 0,80 o más con frases nuevas, recall de Cancelación de 0,90 o más y ningún canal más de 0,05 por debajo del promedio.

## Limitaciones y próximos pasos

Con 20 frases base, los resultados muestran el método, no el desempeño con lenguaje real.

- **Pocos datos:** el F1 del campeón varía entre 0,32 y 0,60 según el pliegue.
- **Dos categorías sin resolver:** Servicio y Producto quedan en cero; necesitan más frases con vocabulario propio.
- **Umbrales sin calibrar:** las probabilidades del modelo deben calibrarse con mensajes reales antes de fijar cortes definitivos.
- **Embeddings con un corpus pequeño:** fastText reconoce errores como “paqte” → paquete, pero necesita un corpus del dominio mucho más grande para competir.
- **Siguiente versión:** reunir 50 o más frases reales por categoría (primero Cancelación y Servicio), repetir el torneo con un conjunto de prueba nunca visto y compararlo con reglas simples y con fastText entrenado con texto del dominio.

## Equipo y mi rol

Ejercicio académico en equipo, organizado en tres roles. Mi parte fue la de Ciencia de Datos: la auditoría que encontró las 20 frases base y las tres fugas, las pruebas A/B de limpieza y el análisis de embeddings.

| Integrante | Rol | Aporte principal |
| --- | --- | --- |
| **Sebastián Padilla** | Ciencia de Datos | Auditoría de los datos, control de fuga, pruebas A/B y embeddings |
| Juan David Gonzalez | Especialista en Machine Learning | Pipelines, torneo de representaciones, métricas y umbrales |
| Nelsón Fabian Manios | Gerencia de IA | Criterios de aceptación, costo, riesgo y decisión del comité |

El análisis y los materiales se prepararon con apoyo de Claude (Anthropic). Todas las cifras se verificaron con las ejecuciones del código.
