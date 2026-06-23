# Pipeline de Limpieza de Datos Bancarios

Proyecto end-to-end de limpieza de datos: generación de datos sintéticos "sucios", limpieza en Python y Excel, y visualización en Power BI.

## Cómo funciona

1. **Generación de datos sintéticos:** se generan ~1,500 movimientos bancarios simulados (usando `Faker` con configuración regional México) con errores realistas intencionalmente inyectados:
   - Inconsistencias de formato (mayúsculas, espacios, errores de tipeo en categorías)
   - Montos en distintos formatos (con signo de dólar, texto "N/D", celdas vacías, formato MXN)
   - Fechas en múltiples formatos (`YYYY-MM-DD`, `DD/MM/YYYY`, `DD-MM-YYYY`, `MM/DD/YYYY`)
   - Duplicados exactos (~3%), valores nulos (~5%), filas completamente vacías, y outliers absurdos
2. **Limpieza en Python (pandas):** estandarización de categorías, parseo de fechas multi-formato, conversión de montos a numérico, eliminación de duplicados y filas vacías, tratamiento de outliers y nulos.
3. **Limpieza en Excel:** réplica del proceso de limpieza usando herramientas nativas de Excel (Power Query, fórmulas) con configuración regional en español/México, para practicar el mismo flujo en ambas herramientas.
4. **Dashboard en Power BI:** visualización interactiva de los datos limpios — gasto por categoría, tendencias en el tiempo, comparación por banco.

## Por qué importa

Los datos reales casi nunca llegan limpios. Este proyecto simula intencionalmente los problemas más comunes de datos del mundo real (formatos inconsistentes, duplicados, nulos) para practicar y demostrar un flujo de limpieza completo, no solo el análisis posterior asumiendo datos ya limpios.

## Stack técnico

`pandas`, `Faker`, Power Query / Excel, Power BI

## Cómo ejecutarlo

```bash
pip install pandas faker openpyxl
python generar_excel_sucio.py
```

Esto genera un archivo Excel con los datos "sucios" listos para limpiar.
