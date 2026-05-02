# ============================================================
# Time Series Analysis — ARIMA & VAR Models
# Author: Gorka Crespo Bravo
# Description: Estimación de modelos ARIMA y VAR para
#              forecasting de series temporales financieras.
# ============================================================

library(tseries)
library(forecast)
library(vars)
library(ggplot2)
library(gridExtra)

# ============================================================
# 1. Carga y preparación de datos
# ============================================================
set.seed(42)

# Simulamos retornos de dos activos correlacionados
n <- 1000
e1 <- rnorm(n)
e2 <- 0.6 * e1 + 0.8 * rnorm(n)  # Correlación ~0.6

# Proceso AR(1) + ruido
y1 <- numeric(n)
y2 <- numeric(n)
y1[1] <- e1[1]; y2[1] <- e2[1]

for (t in 2:n) {
  y1[t] <- 0.3 * y1[t-1] + 0.1 * y2[t-1] + e1[t]
  y2[t] <- 0.2 * y1[t-1] + 0.4 * y2[t-1] + e2[t]
}

datos <- ts(cbind(Activo_A = y1, Activo_B = y2), frequency = 252)

cat("Dimensiones:", dim(datos), "\n")
cat("Correlación:", cor(y1, y2), "\n\n")

# ============================================================
# 2. Tests de estacionariedad
# ============================================================
cat("=== Tests ADF de Estacionariedad ===\n")
for (name in colnames(datos)) {
  test <- adf.test(datos[, name])
  cat(sprintf("  %s: ADF stat = %.4f, p-value = %.4f %s\n",
              name, test$statistic, test$p.value,
              ifelse(test$p.value < 0.05, "(Estacionaria)", "(No estacionaria)")))
}

# ============================================================
# 3. Modelo ARIMA — Serie univariante
# ============================================================
cat("\n=== ARIMA — Activo A ===\n")

# Auto ARIMA
fit_arima <- auto.arima(datos[, "Activo_A"], seasonal = FALSE,
                         stepwise = FALSE, approximation = FALSE)
cat("Modelo seleccionado:", capture.output(fit_arima)[2], "\n")
cat("AIC:", AIC(fit_arima), " BIC:", BIC(fit_arima), "\n")

# Diagnóstico de residuos
residuos <- residuals(fit_arima)
lb_test <- Box.test(residuos, lag = 20, type = "Ljung-Box")
cat(sprintf("Ljung-Box test: stat=%.4f, p-value=%.4f\n",
            lb_test$statistic, lb_test$p.value))

# Forecast
fc_arima <- forecast(fit_arima, h = 30)
cat("\nPredicciones ARIMA (próximos 5 periodos):\n")
print(head(as.data.frame(fc_arima), 5))

# ============================================================
# 4. Modelo VAR — Sistema multivariante
# ============================================================
cat("\n=== VAR — Sistema Bivariante ===\n")

# Selección de orden óptimo
lag_select <- VARselect(datos, lag.max = 10, type = "const")
cat("Orden óptimo (criterios):\n")
print(lag_select$selection)

optimal_lag <- lag_select$selection["AIC(n)"]
cat(sprintf("\nUsando p = %d (criterio AIC)\n", optimal_lag))

# Estimación del modelo VAR
fit_var <- VAR(datos, p = optimal_lag, type = "const")

# Resumen de coeficientes
cat("\nCoeficientes del VAR:\n")
for (name in names(fit_var$varresult)) {
  cat(sprintf("\n  Ecuación: %s\n", name))
  coefs <- coef(fit_var$varresult[[name]])
  for (i in 1:nrow(coefs)) {
    cat(sprintf("    %-15s: %8.4f (p=%.4f)\n",
                rownames(coefs)[i], coefs[i, 1], coefs[i, 4]))
  }
}

# Test de causalidad de Granger
cat("\n=== Causalidad de Granger ===\n")
g1 <- causality(fit_var, cause = "Activo_A")
g2 <- causality(fit_var, cause = "Activo_B")
cat(sprintf("  A → B: F=%.4f, p=%.4f %s\n",
            g1$Granger$statistic, g1$Granger$p.value,
            ifelse(g1$Granger$p.value < 0.05, "(**)", "")))
cat(sprintf("  B → A: F=%.4f, p=%.4f %s\n",
            g2$Granger$statistic, g2$Granger$p.value,
            ifelse(g2$Granger$p.value < 0.05, "(**)", "")))

# Impulso-Respuesta
irf_result <- irf(fit_var, impulse = "Activo_A",
                  response = "Activo_B", n.ahead = 20)

# Forecast VAR
fc_var <- predict(fit_var, n.ahead = 30)

# ============================================================
# 5. Visualización
# ============================================================
# ARIMA Forecast Plot
p1 <- autoplot(fc_arima) +
  labs(title = "ARIMA Forecast — Activo A", x = NULL, y = "Valor") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

# IRF Plot
irf_df <- data.frame(
  Periodo = 0:20,
  IRF = irf_result$irf$Activo_A,
  Lower = irf_result$Lower$Activo_A,
  Upper = irf_result$Upper$Activo_A
)
colnames(irf_df) <- c("Periodo", "IRF", "Lower", "Upper")

p2 <- ggplot(irf_df, aes(x = Periodo)) +
  geom_ribbon(aes(ymin = Lower, ymax = Upper), fill = "#3b82f6", alpha = 0.15) +
  geom_line(aes(y = IRF), color = "#3b82f6", linewidth = 1.2) +
  geom_hline(yintercept = 0, linestyle = "dashed", color = "gray50") +
  labs(title = "Impulso-Respuesta: Activo A → Activo B",
       x = "Periodos", y = "Respuesta") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

g <- arrangeGrob(p1, p2, ncol = 1)
ggsave("time_series_analysis.png", g, width = 12, height = 10, dpi = 150)
grid.arrange(p1, p2, ncol = 1)
