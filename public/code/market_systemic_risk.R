# ============================================================
# Market & Systemic Risk — VaR & CoVaR Estimation
# Author: Gorka Crespo Bravo
# Description: Estimación de Value at Risk y Conditional VaR
#              para análisis de riesgo sistémico.
# ============================================================

library(quantmod)
library(rugarch)
library(rmgarch)
library(PerformanceAnalytics)
library(ggplot2)

# ============================================================
# 1. Descarga de datos
# ============================================================
tickers <- c("^STOXX50E", "^GDAXI", "^FCHI", "^IBEX")
nombres <- c("EuroStoxx50", "DAX", "CAC40", "IBEX35")

getSymbols(tickers, src = "yahoo", from = "2015-01-01", to = "2024-01-01")

# Calcular retornos logarítmicos
returns_list <- lapply(tickers, function(t) {
  dailyReturn(Ad(get(gsub("\\^", "", t))), type = "log")
})
returns <- do.call(merge, returns_list)
colnames(returns) <- nombres
returns <- na.omit(returns)

cat("Dimensiones de los datos:", dim(returns), "\n")
cat("Periodo:", as.character(index(returns)[1]), "a",
    as.character(tail(index(returns), 1)), "\n\n")

# ============================================================
# 2. Value at Risk (VaR) — Métodos múltiples
# ============================================================
alpha <- 0.05  # Nivel de confianza 95%

# --- 2.1 VaR Histórico ---
var_historico <- apply(returns, 2, function(x) {
  quantile(x, probs = alpha)
})
cat("=== VaR Histórico (95%) ===\n")
print(round(var_historico * 100, 4))

# --- 2.2 VaR Paramétrico (Normal) ---
var_parametrico <- apply(returns, 2, function(x) {
  mu <- mean(x)
  sigma <- sd(x)
  mu + qnorm(alpha) * sigma
})
cat("\n=== VaR Paramétrico (95%) ===\n")
print(round(var_parametrico * 100, 4))

# --- 2.3 VaR con GARCH(1,1) ---
spec <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(0, 0)),
  distribution.model = "std"  # Distribución t-Student
)

var_garch <- sapply(nombres, function(name) {
  fit <- ugarchfit(spec, returns[, name], solver = "hybrid")
  sigma_t <- tail(sigma(fit), 1)
  mu_t <- tail(fitted(fit), 1)
  # VaR con distribución t-Student
  nu <- coef(fit)["shape"]
  mu_t + sigma_t * qt(alpha, df = nu)
})
cat("\n=== VaR GARCH(1,1) t-Student (95%) ===\n")
print(round(var_garch * 100, 4))

# ============================================================
# 3. Expected Shortfall (CVaR)
# ============================================================
cvar_historico <- apply(returns, 2, function(x) {
  var_level <- quantile(x, probs = alpha)
  mean(x[x <= var_level])
})
cat("\n=== CVaR Histórico (95%) ===\n")
print(round(cvar_historico * 100, 4))

# ============================================================
# 4. CoVaR — Adrian & Brunnermeier (2016)
# ============================================================
# CoVaR: VaR del sistema condicionado al estrés de una institución

estimate_covar <- function(system_ret, inst_ret, alpha = 0.05) {
  # Regresión cuantílica
  library(quantreg)

  df <- data.frame(system = as.numeric(system_ret),
                   inst = as.numeric(inst_ret))

  # VaR incondicional del sistema
  rq_uncond <- rq(system ~ 1, tau = alpha, data = df)
  VaR_uncond <- coef(rq_uncond)[1]

  # CoVaR: VaR del sistema | institución en su VaR
  rq_cond <- rq(system ~ inst, tau = alpha, data = df)
  beta <- coef(rq_cond)

  VaR_inst <- quantile(df$inst, probs = alpha)
  CoVaR <- beta[1] + beta[2] * VaR_inst

  # Delta CoVaR
  VaR_inst_median <- median(df$inst)
  CoVaR_median <- beta[1] + beta[2] * VaR_inst_median
  Delta_CoVaR <- CoVaR - CoVaR_median

  return(list(
    CoVaR = CoVaR,
    Delta_CoVaR = Delta_CoVaR,
    VaR_uncond = VaR_uncond,
    beta = beta
  ))
}

# Calcular CoVaR de cada índice sobre el EuroStoxx50
cat("\n=== CoVaR sobre EuroStoxx50 ===\n")
cat(sprintf("%-12s %10s %12s %12s\n",
            "Índice", "CoVaR", "ΔCoVaR", "VaR_uncond"))
cat(strrep("-", 50), "\n")

for (name in nombres[-1]) {
  result <- estimate_covar(returns[, "EuroStoxx50"], returns[, name])
  cat(sprintf("%-12s %10.4f%% %10.4f%% %10.4f%%\n",
              name,
              result$CoVaR * 100,
              result$Delta_CoVaR * 100,
              result$VaR_uncond * 100))
}

# ============================================================
# 5. Visualización — Rolling VaR
# ============================================================
window_size <- 252  # 1 año

rolling_var <- rollapply(returns[, "EuroStoxx50"], width = window_size,
                          FUN = function(x) quantile(x, probs = alpha),
                          align = "right", fill = NA)

df_plot <- data.frame(
  Fecha = index(returns),
  Retorno = as.numeric(returns[, "EuroStoxx50"]),
  VaR_Rolling = as.numeric(rolling_var)
)

p <- ggplot(df_plot, aes(x = Fecha)) +
  geom_line(aes(y = Retorno), color = "steelblue", alpha = 0.6, linewidth = 0.3) +
  geom_line(aes(y = VaR_Rolling), color = "red", linewidth = 0.8) +
  labs(
    title = "EuroStoxx 50 — Rolling VaR (95%, ventana 252 días)",
    x = NULL, y = "Retorno diario"
  ) +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

ggsave("rolling_var.png", p, width = 12, height = 6, dpi = 150)
print(p)
