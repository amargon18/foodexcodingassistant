import duckdb
import time

print("🔄 Iniciando carga de Open Food Facts en DuckDB...")
print("Archivo: en.openfoodfacts.org.products.csv (12GB)")
print("Esto puede tardar 10-30 minutos...\n")

# Conectar a DuckDB
con = duckdb.connect('openfoodfacts.duckdb')

start = time.time()

try:
    # Cargar CSV en DuckDB
    con.execute("""
        CREATE TABLE products AS
        SELECT
        product_name, generic_name, quantity, ingredients_text, categories, brands,
        labels, packaging, origins, manufacturing_places, traces, additives
        FROM read_csv(
        '/home/alexl/en.openfoodfacts.org.products.csv',
        delim='\\t',
        header=true,
        ignore_errors=true,
        parallel=true,
        all_varchar=true
        );
    """)

    
    elapsed = time.time() - start
    print(f"\n✅ Carga completada en {elapsed/60:.1f} minutos")
    
    # Verificar cuántos productos se cargaron
    result = con.execute("SELECT COUNT(*) FROM products").fetchone()
    print(f"📊 Total de productos cargados: {result[0]:,}")
    
    # Mostrar información de columnas
    columns = con.execute("PRAGMA table_info(products)").fetchall()
    print(f"\n📋 Total de columnas: {len(columns)}")
    
    print("\n🔍 Primeras 15 columnas:")
    for col in columns[:15]:
        print(f"  - {col[1]} ({col[2]})")
    
    # Mostrar muestra de datos
    print("\n📝 Muestra de 5 productos:")
    sample = con.execute("""
        SELECT *
        FROM products 
        WHERE product_name IS NOT NULL 
        LIMIT 5
    """).fetchall()
    
   
    
except Exception as e:
    print(f"\n❌ Error durante la carga: {e}")
    import traceback
    traceback.print_exc()
finally:
    con.close()
