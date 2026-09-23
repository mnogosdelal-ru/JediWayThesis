import pandas as pd

# 1. Загружаем исходный файл
df = pd.read_csv('Результаты пульс-опроса-Q3-2026 - RawData.csv')

# 2. Создаем словарь для маппинга: каждый уникальный tg_id получает номер Student_N
unique_ids = df['tg_id'].dropna().unique()
mapping = {tg_id: f"Student_{i+1}" for i, tg_id in enumerate(unique_ids)}

# 3. Заменяем tg_id на новые идентификаторы
df['tg_id'] = df['tg_id'].map(mapping).fillna(df['tg_id'])

# 4. Сохраняем обезличенную версию в новый файл
df.to_csv('Результаты пульс-опроса-Q3-2026 - Anonymized.csv', index=False, encoding='utf-8')

print(f"✅ Готово! Файл сохранен.")
print(f"👥 Всего уникальных студентов в наборе: {len(mapping)}")