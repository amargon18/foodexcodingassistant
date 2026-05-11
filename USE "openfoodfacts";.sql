USE "openfoodfacts";

CREATE OR REPLACE VIEW main.data AS
SELECT 'code', 'product_name', 'generic_name', COLUMNS('.*_tags$')
FROM main.products
