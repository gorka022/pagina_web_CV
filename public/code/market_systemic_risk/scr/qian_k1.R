# =========================================================================
# SCRIPT TFM - CAMINO B: RED DIRECCIONAL CON REZAGO (k > 0)
# EXTRACCIÓN DE MATRICES CQ (SUCIA) Y PCQ (LIMPIA)
# =========================================================================
library(MASS) # Necesaria para ginv()

# =========================================================================
# 1. PREPARACIÓN ROBUSTA DE LOS DATOS Y FILTRO TEMPORAL
# =========================================================================
auto_banks_europe_rend <- empresas_sectores_7_rend[, c("Name",   "Banks", "Real Estate Inv & Svs", "Alternative Energy", "Oil, Gas and Coal",  "Aerospace & Defense", "Ind. Metals & Mining", "Automobiles & Parts", "Pharm. & Biotech", "Chemicals","Electricity")]
comodities_auto_banks_europe_rend <- auto_banks_europe_rend %>%
  left_join(comodities_rend_inv, by = "Name")

comodities_auto_banks_europe_rend <- comodities_auto_banks_europe_rend %>%
  mutate(Name = as.Date(Name)) %>% 
  filter(Name >= as.Date("2015-01-01") & Name <= as.Date("2025-12-31"))

df_tradicional <- as.data.frame(comodities_auto_banks_europe_rend)







# 1. Preparación inicial de los datos
datos_numericos <- as.matrix(df_tradicional[, -1])
nombres <- colnames(datos_numericos)
n_empresas <- ncol(datos_numericos)

# Parámetros del modelo
tau <- 0.05       # Cuantil del 5% (riesgo extremo)
k <- 1            # Días de rezago (puedes cambiarlo a 5, 10, etc.)

cat("\nIniciando cálculo de matrices direccionales (CQ y PCQ) para k =", k, "...\n")

# 2. Construir el "Mega Vector" temporal (HOY y AYER)
# Recortamos los datos para que 'Hoy' y 'Ayer' coincidan en longitud temporal
datos_hoy  <- datos_numericos[(k + 1):nrow(datos_numericos), ]
datos_ayer <- datos_numericos[1:(nrow(datos_numericos) - k), ]

# Unimos todo: Columnas 1 a 30 (Hoy), Columnas 31 a 60 (Ayer)
Super_Datos <- cbind(datos_hoy, datos_ayer)
n_super <- ncol(Super_Datos)
T_dias <- nrow(Super_Datos)

# Nombres para la Súper Matriz
nombres_hoy <- paste0(nombres, "_t")
nombres_ayer <- paste0(nombres, "_t-", k)
colnames(Super_Datos) <- c(nombres_hoy, nombres_ayer)

# 3. Calcular la Súper Matriz de Hits
H_mat <- apply(Super_Datos, 2, function(x) {
  q_val <- quantile(x, probs = tau, na.rm = TRUE)
  ifelse(x <= q_val, 1 - tau, -tau)
})
H_mat[is.na(H_mat)] <- 0 # Limpieza de seguridad

# 4. Construir la Súper Matriz R de 60x60 (Covarianzas Crudas)
R_mat <- t(H_mat) %*% H_mat / T_dias


# =========================================================================
# FASE 1: MATRIZ CQ (RED SUCIA / CORRELACIÓN BRUTA)
# =========================================================================
CQ_mat_gigante <- matrix(0, nrow = n_super, ncol = n_super)

for(i in 1:n_super) {
  for(j in 1:n_super) {
    if(i == j) {
      CQ_mat_gigante[i, j] <- 1
    } else {
      # Normalización estándar sin invertir
      CQ_mat_gigante[i, j] <- R_mat[i, j] / sqrt(R_mat[i, i] * R_mat[j, j])
    }
  }
}

# RECORTE: Extraer el "Cuarto B" (Impacto de Ayer sobre Hoy)
CQ_direccional <- CQ_mat_gigante[1:n_empresas, (n_empresas + 1):n_super]
rownames(CQ_direccional) <- nombres # Filas: Receptores (Hoy)
colnames(CQ_direccional) <- nombres # Columnas: Emisores (Ayer)

cat("-> Matriz CQ (sucia y direccional) generada con éxito.\n")


# =========================================================================
# FASE 2: MATRIZ PCQ (RED LIMPIA / CORRELACIÓN PARCIAL DIRECCIONAL)
# =========================================================================
# Inversión de la Súper Matriz R (60x60)
P_mat <- ginv(R_mat)

PCQ_mat_gigante <- matrix(0, nrow = n_super, ncol = n_super)

for(i in 1:n_super) {
  for(j in 1:n_super) {
    if(i == j) {
      PCQ_mat_gigante[i, j] <- 1
    } else {
      # Fórmula PCQ (Con signo negativo)
      PCQ_mat_gigante[i, j] <- -P_mat[i, j] / sqrt(P_mat[i, i] * P_mat[j, j])
    }
  }
}

# RECORTE: Extraer el "Cuarto B" de la matriz limpia
PCQ_direccional <- PCQ_mat_gigante[1:n_empresas, (n_empresas + 1):n_super]
rownames(PCQ_direccional) <- nombres # Filas: Receptores (Hoy)
colnames(PCQ_direccional) <- nombres # Columnas: Emisores (Ayer)

cat("-> Matriz PCQ (limpia y direccional) generada con éxito.\n")
cat("\n¡Proceso finalizado! Ya tienes 'CQ_direccional' y 'PCQ_direccional' listas.\n")