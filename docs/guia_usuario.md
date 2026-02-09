# Guía de Usuario: Cómo Entrenar tu IA de Trading

Esta guía te explicará cómo enseñar a tu Inteligencia Artificial a operar en el mercado, paso a paso. No necesitas ser un experto en programación.

## Conceptos Básicos

Imagina que estás enseñando a un aprendiz.
1.  **Entrenamiento**: Le muestras al aprendiz gráficos del pasado (ej. últimos 2 años de Apple) y le dices: "Mira, aquí subió, aquí bajó". El aprendiz (la IA) busca patrones repetitivos.
2.  **Validación**: Le haces un examen con datos que no ha visto antes para asegurarte de que aprendió la lección y no solo memorizó las respuestas.
3.  **Backtesting**: Le dejas operar en un simulador con dinero falso para ver qué tal lo haría.

---

## Paso 1: Preparación (Instalar las herramientas)

Antes de empezar, necesitamos que tu computadora tenga las herramientas necesarias. Abre una terminal (o consola) en la carpeta del proyecto y escribe:

```bash
pip install -r requirements.txt
pip install pandas_ta tensorflow
```

Esto instala el "cerebro" (TensorFlow) y las "gafas analíticas" (pandas-ta) de tu IA.

---

## Paso 2: Entrenar a la IA

Para entrenar a la IA con una acción específica (por ejemplo, Apple - AAPL), tienes que ejecutar comandos en tu terminal.

### Opción A: Usando el sistema automático (Recomendado)
Hemos creado un script que verifica y entrena todo automáticamente.

1.  Abre la terminal.
2.  Ejecuta:
    ```bash
    python main.py train --symbols AAPL
    ```
    *Puedes cambiar AAPL por BTC-USD, TSLA, etc.*

**¿Qué está pasando internamente?**
- La IA descarga los precios de los últimos 2 años.
- Calcula indicadores matemáticos (RSI, MACD, etc.).
- Estudia los patrones y ajusta sus neuronas internas para minimizar sus errores.
- Guarda su "cerebro" entrenado en la carpeta `src/models/saved_models/`.

---

## Paso 3: Probar la IA (Backtesting)

Ahora que la IA ha estudiado, vamos a ver cómo le habría ido en el último año si le hubiéramos dado dinero real.

1.  En la terminal, escribe:
    ```bash
    python main.py backtest --symbols AAPL --start-date 2023-01-01
    ```

**Resultados a observar:**
Al finalizar, verás un reporte en pantalla. Fíjate en:
- **Total Return**: ¿Ganó o perdió dinero?
- **Win Rate**: ¿Qué porcentaje de veces acertó?
- **Max Drawdown**: ¿Cuál fue la peor racha de pérdidas? (Importante para saber si aguantarías emocionalmente).

---

## Paso 4: Obtener Señales (¿Qué hago hoy?)

Si quieres saber qué opina la IA sobre el mercado **HOY**, ejecuta:

```bash
python main.py signals --symbols AAPL
```

La IA analizará los datos hasta el último minuto y te dirá:
- **Señal**: COMPRAR (Buy), VENDER (Sell) o ESPERAR (Hold).
- **Confianza**: Qué tan segura está (ej. 85%).
- **Stop Loss**: A qué precio salir si se equivoca (para proteger tu dinero).

---

## Consejos para un Mejor Entrenamiento

- **Más Datos**: Si la IA no aprende bien, intenta darle más historia (modificando `training_period` en el código a '5y' o '10y').
- **Diversificación**: No entrenes solo con una acción. Prueba con varias para ver en cuáles funciona mejor tu estrategia.
- **Paciencia**: El entrenamiento puede tardar unos minutos dependiendo de la velocidad de tu computadora.
