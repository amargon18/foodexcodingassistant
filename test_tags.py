from src.etl.cleaning import parse_ingredient_tags

# Test with mixed language tags
test_cases = [
    "en:sugar,fr:sucre,en:salt",  # Mixed
    "fr:farine,de:mehl",  # Only non-English
    "en:milk,en:butter",  # Only English
]

for tags in test_cases:
    result = parse_ingredient_tags(tags)
    print(f"Input: {tags}")
    print(f"Output: {result}\n")


import pandas as pd
df = pd.read_parquet("data/products_ingredients.parquet")
print(f"Total rows: {len(df)}")
print(df["ingredient"].head(20))