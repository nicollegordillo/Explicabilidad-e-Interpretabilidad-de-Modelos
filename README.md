# Laboratorio 10 — Explicabilidad e Interpretabilidad (XAI)

**CC3092 Deep Learning — Universidad del Valle de Guatemala**
Nicolle Gordillo — 22246

Explicación de las decisiones de dos modelos con las técnicas vistas en la clase S15 (XAI):

- **Visión:** Grad-CAM implementado desde cero sobre ResNet-18 (ImageNet) y verificación numérica de su relación con CAM.
- **Datos tabulares (German Credit):** gradient boosting de scikit-learn explicado con SHAP (TreeSHAP, KernelSHAP y Shapley exacto escrito a mano), LIME desde cero y explicaciones contrafactuales.
- **Validación:** deletion/insertion y prueba de aleatorización de parámetros en cascada.

## Estructura

```
Explicabilidad-e-Interpretabilidad-de-Modelos/
├── Lab10_XAI_22246.ipynb      # notebook con todos los tasks (ejecutado, con salidas)
├── requirements.txt
├── imagenes/                  # imágenes propias del Task 1 y sus fuentes
│   ├── imagen_a.jpg
│   ├── imagen_b.jpg
│   ├── imagen_c.jpg
│   └── FUENTES.md
├── figuras/                   # figuras generadas por el notebook
├── resultados/                # números de cada entrega en JSON
    ├── resultados_parcial.json  # Tasks 1 y 2
    └── resultados_final.json    # Tasks 1 a 4
```

## Cómo ejecutar

En Google Colab:

```python
!git clone https://github.com/nicollegordillo/Explicabilidad-e-Interpretabilidad-de-Modelos.git
%cd Explicabilidad-e-Interpretabilidad-de-Modelos
```

Luego abra `Lab10_XAI_22246.ipynb` y ejecute *Runtime → Run all*. Todo corre en CPU con **semilla 42** (unos minutos en total; la parte más larga es la verificación de estabilidad de LIME del Task 3.5a).
El conjunto German Credit se descarga con `fetch_openml("credit-g", version=1)` y los pesos de ResNet-18 con torchvision.

## Entregas

| Entrega | Tasks |
|---|---|
| Parcial | Task 1 (Grad-CAM y CAM) y Task 2 (SHAP) |
| Final | Tasks 1 a 4 (agrega LIME, contrafactuales y validación de explicaciones) |

## Restricciones

Grad-CAM, CAM, LIME y el cálculo exacto de Shapley están escritos con PyTorch, NumPy y scikit-learn; no se usan `pytorch-grad-cam`, `captum` ni `lime`. La librería `shap` se usa solo para TreeSHAP, KernelSHAP y sus gráficos.
