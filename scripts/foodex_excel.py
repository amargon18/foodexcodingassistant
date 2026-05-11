import pandas as pd

df = pd.read_excel('MTX_17.0.xlsx', sheet_name='term')

columnas = [
    'termCode',
    'termExtendedName', 
    'termShortName',
    'termScopeNote',
    'scientificNames',
    'commonNames',
    'allFacets',
    'status'
]

df_filtrado = df[columnas]

# Filtrar también los deprecated
df_filtrado = df_filtrado[df_filtrado['status'] != 'DEPRECATED']

df_filtrado.to_excel('foodex2_terms.xlsx', index=False)
print(f"{len(df_filtrado)} términos activos")