# 1. Configurar el directorio de trabajo (Working Directory)
# Esto le dice a R dónde buscar los archivos. Según tu imagen, esta es tu ruta:
setwd("C:/tfm_kodea/hedging_european_markets/scripts_garbia")

# 2. Ejecutar los scripts en orden secuencial
print("Ejecutando 01...")
source("01_carga_datos.R", encoding = "UTF-8")

print("Ejecutando 02...")
source("02_calculo_mv.R", encoding = "UTF-8")

print("Ejecutando 03...")
source("03_calculo_rendimientos.R", encoding = "UTF-8")

print("Ejecutando 04 y sub-scripts...")
source("04_visualizacion_datos.R", encoding = "UTF-8")
source("04.1_visualizacion_datos_sectores.R", encoding = "UTF-8")
source("4.02_posible_plot.R", encoding = "UTF-8")

print("Ejecutando 05...")
source("05_tabla_estadisticos.R", encoding = "UTF-8")
setwd("C:/tfm_kodea/hedging_european_markets")

print("¡Todos los scripts se han ejecutado correctamente!")