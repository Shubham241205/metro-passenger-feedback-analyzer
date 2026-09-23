import pandas as pd

raw_path = "data/raw/metro_passenger_feedback_dataset.csv"
df = pd.read_csv(raw_path)

print("--- WHITESPACE CHECK ---")
for col in df.select_dtypes(include='object').columns:
    has_leading_trailing = df[col].apply(lambda x: isinstance(x, str) and (x != x.strip())).sum()
    print(f"Column '{col}': {has_leading_trailing} rows with leading/trailing whitespace")

print("\n--- SENTIMENT SCORE RANGE ---")
print(f"Min sentiment score: {df['sentiment_score'].min()}, Max sentiment score: {df['sentiment_score'].max()}")

print("\n--- RATING VS SENTIMENT CROSS-TAB ---")
print(pd.crosstab(df['rating'], df['sentiment']))

print("\n--- DATE RANGE CHECK ---")
df['parsed_date'] = pd.to_datetime(df['feedback_date'], errors='coerce')
print(f"Min date: {df['parsed_date'].min()}, Max date: {df['parsed_date'].max()}")

print("\n--- FEEDBACK ID FORMAT CHECK ---")
invalid_id_format = (~df['feedback_id'].str.match(r'^MPF\d{5}$')).sum()
print(f"Invalid feedback_id format count: {invalid_id_format}")
