import duckdb
import time
import urllib.request
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# =========================
# Ajustes (WSL + 12 cores + ~8GB RAM)
# =========================
DB_PATH     = os.path.abspath("openfoodfacts.duckdb")
MODEL_PATH  = os.path.abspath("lid.176.ftz")   # ← path absoluto, crítico para workers

BATCH_SIZE = 100_000
MAX_WORKERS = 10
CHUNK_SIZE  = 2_000

MIN_ING_LEN      = 20
MAX_CHARS_DETECT = 1000
MIN_CHARS_DETECT = 10

# =========================
# Descarga automática del modelo si no existe
# =========================
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
        print(f"Descargando modelo fasttext desde {url} ...")
        urllib.request.urlretrieve(url, MODEL_PATH)
        print(f"Modelo guardado en: {MODEL_PATH}")
    else:
        print(f"Modelo encontrado en: {MODEL_PATH}")

# =========================
# Test de detección antes de empezar
# =========================
def test_detection():
    import fasttext
    fasttext.FastText.eprint = lambda x: None
    model = fasttext.load_model(MODEL_PATH)
    samples = [
        ("es", "Ingredientes: agua, sal, harina de trigo, azúcar, aceite de oliva"),
        ("en", "Ingredients: water, salt, wheat flour, sugar, olive oil"),
        ("fr", "Ingrédients: eau, sel, farine de blé, sucre, huile d'olive"),
        ("de", "Zutaten: Wasser, Salz, Weizenmehl, Zucker, Olivenöl"),
    ]
    print("\n── Test de detección ──")
    all_ok = True
    for expected, text in samples:
        pred = model.predict(text.replace("\n", " "), k=1)
        detected = pred[0][0].replace("__label__", "")
        ok = "✅" if detected == expected else "❌"
        print(f"  {ok} esperado={expected} detectado={detected}  [{text[:50]}]")
        if detected != expected:
            all_ok = False
    if not all_ok:
        raise RuntimeError("El modelo no detecta correctamente. Revisa la instalación de fasttext.")
    print("── Test OK ──\n")

# =========================
# Worker: inicializa UNA VEZ por proceso
# =========================
def init_worker(model_path):
    import fasttext
    fasttext.FastText.eprint = lambda x: None
    global _model
    _model = fasttext.load_model(model_path)

def safe_detect(text: str):
    if not text:
        return None
    t = text.strip()
    if len(t) < MIN_CHARS_DETECT:
        return None
    t = t[:MAX_CHARS_DETECT].replace("\n", " ")
    pred = _model.predict(t, k=1)        # sin try/except → los errores reales se verán
    return pred[0][0].replace("__label__", "")

def detect_lang_batch(rows):
    return [
        (rid, safe_detect(pname) if pname else None, safe_detect(ing) if ing else None)
        for rid, pname, ing in rows
    ]

# =========================
# Main
# =========================
def main():
    ensure_model()
    test_detection()   # ← aborta si algo va mal antes de procesar millones de filas

    con = duckdb.connect(DB_PATH)
    con.execute("PRAGMA memory_limit='4GB'")

    con.execute("ALTER TABLE main.products_with_language1 ADD COLUMN IF NOT EXISTS lang_product_name VARCHAR")
    con.execute("ALTER TABLE main.products_with_language1 ADD COLUMN IF NOT EXISTS lang_ingredients_text VARCHAR")

    con.execute("""
        CREATE TEMP TABLE lang_updates (
            rowid                 BIGINT,
            lang_product_name     VARCHAR,
            lang_ingredients_text VARCHAR
        )
    """)

    total_rows = con.execute(f"""
        SELECT COUNT(*)
        FROM main.products_with_language1
        WHERE lang_product_name IS NULL
          AND lang_ingredients_text IS NULL
          AND ingredients_text IS NOT NULL
          AND length(trim(ingredients_text)) >= {MIN_ING_LEN}
          AND (product_name IS NOT NULL OR generic_name IS NOT NULL)
    """).fetchone()[0]

    workers = min(cpu_count(), MAX_WORKERS)
    print(f"Filas a procesar: {total_rows:,}")
    print(f"Workers: {workers} | Batch: {BATCH_SIZE:,} | Chunk: {CHUNK_SIZE:,}\n")

    last_rowid = -1
    processed  = 0
    t_total    = time.time()

    with tqdm(total=total_rows, unit="filas", dynamic_ncols=True) as pbar:
        with Pool(
            processes=workers,
            initializer=init_worker,
            initargs=(MODEL_PATH,),         # ← path absoluto pasado explícitamente
            maxtasksperchild=5_000
        ) as pool:
            while True:
                rows = con.execute(f"""
                    SELECT
                        rowid,
                        substr(coalesce(product_name, ''), 1, {MAX_CHARS_DETECT}),
                        substr(ingredients_text,           1, {MAX_CHARS_DETECT})
                    FROM main.products_with_language1
                    WHERE lang_product_name IS NULL
                      AND lang_ingredients_text IS NULL
                      AND ingredients_text IS NOT NULL
                      AND length(trim(ingredients_text)) >= {MIN_ING_LEN}
                      AND (product_name IS NOT NULL OR generic_name IS NOT NULL)
                      AND rowid > ?
                    ORDER BY rowid
                    LIMIT ?
                """, [last_rowid, BATCH_SIZE]).fetchall()

                if not rows:
                    break

                t_batch     = time.time()
                sub_batches = [rows[i:i+CHUNK_SIZE] for i in range(0, len(rows), CHUNK_SIZE)]
                results     = pool.map(detect_lang_batch, sub_batches)
                updates     = [item for batch in results for item in batch]

                con.executemany("INSERT INTO lang_updates VALUES (?, ?, ?)", updates)

                processed  += len(rows)
                last_rowid  = rows[-1][0]
                elapsed     = time.time() - t_batch

                pbar.update(len(rows))
                pbar.set_postfix({
                    "lote_s" : f"{elapsed:.1f}s",
                    "filas/s": f"{len(rows)/elapsed:,.0f}",
                    "total"  : f"{processed:,}",
                })

    print("\nAplicando UPDATE masivo en DuckDB...")
    t_upd = time.time()
    con.execute("""
        UPDATE main.products_with_language1 AS p
        SET lang_product_name     = u.lang_product_name,
            lang_ingredients_text = u.lang_ingredients_text
        FROM lang_updates AS u
        WHERE p.rowid = u.rowid
    """)
    print(f"UPDATE completado en {time.time()-t_upd:.1f}s")

    # Verificación rápida del resultado
    print("\n── Distribución de idiomas detectados (top 15) ──")
    result = con.execute("""
        SELECT lang_ingredients_text, COUNT(*) AS n
        FROM main.products_with_language1
        WHERE lang_ingredients_text IS NOT NULL
        GROUP BY 1
        ORDER BY 2 DESC
        LIMIT 15
    """).fetchall()
    for lang, n in result:
        print(f"  {lang:<8} {n:>10,}")

    con.execute("CHECKPOINT")
    con.close()

    total_time = time.time() - t_total
    print(f"\n✅ ¡Hecho! {processed:,} filas en {total_time/60:.1f} min "
          f"({processed/total_time:,.0f} filas/s promedio)")

if __name__ == "__main__":
    main()