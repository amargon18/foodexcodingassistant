import duckdb
import pandas as pd
import time

print("Iniciando carga de FoodEx2 Terms en DuckDB...")

con = duckdb.connect('openfoodfacts.duckdb')
start = time.time()

try:
    # Leer Excel con pandas
    df = pd.read_excel('foodex2_terms.xlsx')

    # Filtrar columnas y deprecated
    columnas = [
        'termCode', 'termExtendedName', 'termShortName',
        'termScopeNote', 'scientificNames', 'commonNames',
        'allFacets', 'status'
    ]
    df = df[columnas]
    df = df[df['status'] != 'DEPRECATED']

    # Registrar el dataframe y crear la tabla en DuckDB
    con.register('df_temp', df)
    con.execute("DROP TABLE IF EXISTS foodex2_terms")
    con.execute("CREATE TABLE foodex2_terms AS SELECT * FROM df_temp")

    elapsed = time.time() - start
    count = con.execute("SELECT COUNT(*) FROM foodex2_terms").fetchone()[0]
    print(f"Carga completada en {elapsed:.1f}s — {count} términos cargados")

except Exception as e:
    print(f"Error durante la carga: {e}")
    import traceback
    traceback.print_exc()
finally:
    con.close()