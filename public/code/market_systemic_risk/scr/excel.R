library(writexl)

# Suponiendo que tienes estos tres dataframes:
# comodities_rend_acum
# otro_dataframe_2
# otro_dataframe_3

# 1. Creas una lista asignando el nombre que quieres para cada hoja en Excel
lista_dataframes <- list(
  "df1" = tabla_df1,
  "df2" = tabla_df2,
  "df3" = tabla_df3
)

# 2. Exportas la lista completa a un solo archivo Excel
write_xlsx(lista_dataframes, "mis_tres_dataframes.xlsx")