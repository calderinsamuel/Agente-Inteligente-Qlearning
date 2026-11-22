# 🤖 Agente inteligente con Q-Learning

Proyecto de **aprendizaje por refuerzo tabular** en Python. Un agente aprende por prueba y error a navegar un laberinto 8×8, evitando obstáculos y trampas para llegar a la meta. Incluye notebook explicativo, ejecución por terminal, métricas y pruebas.

## Algoritmo
Se aplica una actualización de Q-learning a cada transición del entorno:

`Q(s,a) ← Q(s,a) + α [r + γ maxₐ' Q(s',a') − Q(s,a)]`

- **Estado:** posición del agente codificada como un entero (0–63).
- **Acciones:** ↑, →, ↓, ←.
- **Recompensa:** +10 por alcanzar meta; −5 por caer en trampa; −0.04 por paso; −0.25 por chocar.
- **Exploración:** estrategia epsilon-greedy con decaimiento gradual.
- **Terminación:** alcanzar la meta, caer en una trampa o superar 100 pasos.

## Cómo ejecutar (Windows / VS Code)
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train.py --episodes 4000 --seed 42
python -m unittest discover -s tests -v
jupyter lab notebooks/Agente_Qlearning.ipynb
```

En macOS/Linux activa con `source .venv/bin/activate`.

## Resultados
Tras `python train.py` se generan automáticamente en `outputs/`:
- `ruta_aprendida.png`: visualización de la trayectoria greedy.
- `curva_aprendizaje.png`: recompensas suavizadas.
- `metricas.json`: éxito, pasos y recompensa promedio.
- `entrenamiento.csv`: registro de los episodios.
- `q_table.npy`: tabla Q entrenada.

**Interpretación:** un éxito del 100 % en este entorno determinista y fijo no implica que un agente sea capaz de generalizar a escenarios distintos. La evaluación parte siempre desde la misma casilla; sería necesario probar mapas y condiciones nuevas para medir generalización.

## Estructura
`src/gridworld.py` define el entorno; `src/qlearning.py` implementa el algoritmo; `train.py` entrena y genera gráficos; `notebooks/` explica paso a paso; `tests/` valida el comportamiento.

> Proyecto educativo. Los resultados se calculan al ejecutar el código: no se afirma un desempeño antes de entrenar.
