#!/usr/bin/env python3
"""
01_data_preprocessing.py

Загрузка данных из MySQL и предобработка для исследования джедайских практик.

Включает:
- Загрузку из MySQL
- Фильтрацию невалидных ответов
- Расчёт всех шкал
- Сохранение чистых данных для дальнейшего анализа

Использование:
    python 01_data_preprocessing.py
"""

import pandas as pd
import numpy as np
import mysql.connector
from datetime import datetime
from config import *

print("=" * 70)
print("ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ")
print("=" * 70)
print(f"Время начала: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()


def load_from_mysql() -> pd.DataFrame:
    """
    Загрузить данные из MySQL
    
    Returns:
        DataFrame с сырыми данными
    """
    print("Подключение к базе данных...")
    
    conn = mysql.connector.connect(**get_db_params())
    
    query = """
    SELECT 
        -- ID и метаданные
        id,
        created_at,
        completed_at,
        status,
        time_spent_seconds,
        attention_check_passed,
        
        -- Демография
        age,
        gender,
        children_count,
        work_experience_years,
        position,
        industry,
        remote_days,
        
        -- Страница 2: Самоощущения
        mindset_technical_humanitarian,
        social_style_intro_extro,
        order_minimalist_collector,
        deadline_early_last,
        task_single_multi,
        decision_rational_emotional,
        tool_preference,
        
        -- Страница 3-4: Срочное/важное
        personal_urgent_important,
        work_urgent_important,
        work_satisfaction,
        
        -- Страница 5: MIJS (сырые items)
        mijs_items,
        
        -- Страница 6: SWLS
        swls_items,
        
        -- Страница 7: MBI (22 items, Водопьянова 2007)
        mbi_item_1,
        mbi_item_2,
        mbi_item_3,
        mbi_item_4,
        mbi_item_5,
        mbi_item_6,
        mbi_item_7,
        mbi_item_8,
        mbi_item_9,
        mbi_item_10,
        mbi_item_11,
        mbi_item_12,
        mbi_item_13,
        mbi_item_14,
        mbi_item_15,
        mbi_item_16,
        mbi_item_17,
        mbi_item_18,
        mbi_item_19,
        mbi_item_20,
        mbi_item_21,
        mbi_item_22,
        
        -- Страница 8: Прокрастинация
        procrastination_items,
        
        -- Страница 9: Практики
        practices_frequency,
        practices_quality,
        
        -- Страница 10: Вакцины
        vaccines,
        
        -- Страница 12: Открытые вопросы
        open_most_useful_practice,
        open_other_practices
        
    FROM respondents
    WHERE status = 'completed'
      AND attention_check_passed = TRUE
    ORDER BY created_at
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    print(f"Загружено {len(df)} респондентов")
    return df


def filter_invalid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Отфильтровать невалидные ответы
    
    Критерии:
    - Время заполнения < 60 секунд
    - Пропуски в ключевых шкалах
    - Выбросы (3 sigma)
    """
    print("\nФильтрация невалидных ответов...")
    initial_n = len(df)
    
    # 1. Фильтр по времени заполнения
    df = df[df['time_spent_seconds'] >= MIN_TIME_SECONDS]
    print(f"  После фильтра по времени: {len(df)} (удалено {initial_n - len(df)})")
    
    # 2. Фильтр пропусков в ключевых шкалах
    key_columns = ['mijs_items', 'swls_items', 'practices_frequency']
    for col in key_columns:
        if col in df.columns:
            df = df[df[col].notna()]
    
    # 3. Фильтр выбросов (возраст)
    if 'age' in df.columns:
        mean_age = df['age'].mean()
        std_age = df['age'].std()
        df = df[(df['age'] >= mean_age - 3*std_age) & (df['age'] <= mean_age + 3*std_age)]
    
    print(f"  После фильтрации: {len(df)} респондентов")
    print(f"  Всего удалено: {initial_n - len(df)} ({(initial_n - len(df))/initial_n*100:.1f}%)")
    
    return df


def parse_json_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Распарсить JSON колонку в отдельные столбцы
    
    Args:
        df: DataFrame
        column: имя колонки с JSON
    
    Returns:
        DataFrame с добавленными столбцами
    """
    if column not in df.columns:
        return df
    
    # Пропустить если все NaN
    if df[column].isna().all():
        return df
    
    try:
        # Попытаться распарсить JSON
        parsed = df[column].apply(
            lambda x: pd.Series(x) if isinstance(x, (dict, list)) else pd.Series()
        )
        
        # Переименовать колонки
        parsed.columns = [f"{column}_{i+1}" for i in range(len(parsed.columns))]
        
        # Добавить к основному DataFrame
        df = pd.concat([df, parsed], axis=1)
        
    except Exception as e:
        print(f"  Warning: Не удалось распарсить {column}: {e}")
    
    return df


def calculate_scales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Рассчитать все шкалы из сырых items
    """
    print("\nРасчёт шкал...")
    
    # ========================================================================
    # MIJS (12 items, items 6-12 инвертируются)
    # ========================================================================
    print("  MIJS...")
    mijs_cols = [f'mijs_items_{i}' for i in range(1, 13)]
    if all(col in df.columns for col in mijs_cols):
        # Items 1-5 напрямую
        df['mijs_urgency'] = df[[f'mijs_items_{i}' for i in range(1, 6)]].sum(axis=1)
        
        # Items 6-12 инвертируются (1→5, 2→4, 3→3, 4→2, 5→1)
        agency_items = df[[f'mijs_items_{i}' for i in range(6, 13)]]
        df['mijs_agency'] = (6 - agency_items).sum(axis=1)
        
        df['mijs_total'] = df['mijs_urgency'] + df['mijs_agency']
    
    # ========================================================================
    # MBI (22 items, Водопьянова 2007)
    # ========================================================================
    print("  MBI (выгорание)...")
    
    # Эмоциональное истощение (9 items, все напрямую)
    exhaustion_cols = [f'mbi_item_{i}' for i in [1, 2, 3, 6, 8, 13, 14, 16, 20]]
    if all(col in df.columns for col in exhaustion_cols):
        df['mbi_exhaustion'] = df[exhaustion_cols].sum(axis=1)
    
    # Цинизм/Деперсонализация (5 items, все напрямую)
    cynicism_cols = [f'mbi_item_{i}' for i in [5, 10, 11, 15, 22]]
    if all(col in df.columns for col in cynicism_cols):
        df['mbi_cynicism'] = df[cynicism_cols].sum(axis=1)
    
    # Профессиональная эффективность (8 items, все инвертируются)
    efficacy_cols = [f'mbi_item_{i}' for i in [4, 7, 9, 12, 17, 18, 19, 21]]
    if all(col in df.columns for col in efficacy_cols):
        # Инверсия: 1→7, 2→6, 3→5, 4→4, 5→3, 6→2, 7→1
        df['mbi_efficacy'] = (8 - df[efficacy_cols]).sum(axis=1)
    
    # Общий балл выгорания
    if all(col in df.columns for col in ['mbi_exhaustion', 'mbi_cynicism', 'mbi_efficacy']):
        df['mbi_total'] = df['mbi_exhaustion'] + df['mbi_cynicism'] + df['mbi_efficacy']
    
    # ========================================================================
    # SWLS (5 items, все напрямую)
    # ========================================================================
    print("  SWLS...")
    swls_cols = [f'swls_items_{i}' for i in range(1, 6)]
    if all(col in df.columns for col in swls_cols):
        df['swls_total'] = df[swls_cols].sum(axis=1)
    
    # ========================================================================
    # Прокрастинация (8 items, item 1 инвертируется)
    # ========================================================================
    print("  Прокрастинация...")
    proc_cols = [f'procrastination_items_{i}' for i in range(1, 9)]
    if all(col in df.columns for col in proc_cols):
        # Item 1 инвертируется (4-балльная шкала: 1→4, 2→3, 3→2, 4→1)
        df.loc[:, f'procrastination_items_1'] = 5 - df[f'procrastination_items_1']
        df['procrastination_total'] = df[proc_cols].sum(axis=1)
    
    # ========================================================================
    # Практики (частота и качество)
    # ========================================================================
    print("  Практики...")
    
    # Частота (21 практика)
    if 'practices_frequency' in df.columns:
        # Распарсить JSON если ещё не распаршен
        if df['practices_frequency'].dtype == 'object':
            df = parse_json_column(df, 'practices_frequency')
        
        freq_cols = [f'practices_frequency_{i}' for i in range(1, 22)]
        available_cols = [c for c in freq_cols if c in df.columns]
        if available_cols:
            df['practices_freq_total'] = df[available_cols].sum(axis=1)
    
    # Качество (21 практика)
    if 'practices_quality' in df.columns:
        if df['practices_quality'].dtype == 'object':
            df = parse_json_column(df, 'practices_quality')
        
        qual_cols = [f'practices_quality_{i}' for i in range(1, 22)]
        available_cols = [c for c in qual_cols if c in df.columns]
        if available_cols:
            df['practices_quality_total'] = df[available_cols].sum(axis=1)
    
    # Индекс планирования (практики 6, 8, 9, 11)
    planning_items = [6, 8, 9, 11]
    planning_cols = [f'practices_frequency_{i}' for i in planning_items]
    available_planning = [c for c in planning_cols if c in df.columns]
    if available_planning:
        df['planning_index'] = df[available_planning].mean(axis=1)
    
    # Вакцины (5 items)
    # ========================================================================
    print("  Вакцины...")
    if 'vaccines' in df.columns:
        if df['vaccines'].dtype == 'object':
            df = parse_json_column(df, 'vaccines')
        
        vac_cols = [f'vaccines_{i}' for i in range(1, 6)]
        available_cols = [c for c in vac_cols if c in df.columns]
        if available_cols:
            df['vaccines_total'] = df[available_cols].sum(axis=1)
    
    return df


def save_clean_data(df: pd.DataFrame) -> None:
    """
    Сохранить чистые данные для дальнейшего анализа
    """
    print("\nСохранение данных...")
    
    # Сохранить полный чистый датасет
    output_file = EXPORT_DIR / 'clean_data.csv'
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"  Сохранено: {output_file}")
    
    # Сохранить только аналитические переменные (для удобства)
    analytic_cols = [
        'id', 'created_at', 'age', 'gender', 'children_count',
        'work_experience_years', 'position', 'industry', 'remote_days',
        'tool_preference',
        'personal_urgent_important', 'work_urgent_important', 'work_satisfaction',
        'mijs_total', 'mijs_urgency', 'mijs_agency',
        'mbi_total', 'mbi_exhaustion', 'mbi_cynicism', 'mbi_efficacy',
        'swls_total',
        'procrastination_total',
        'practices_freq_total', 'practices_quality_total', 'planning_index',
        'vaccines_total'
    ]
    
    available_cols = [c for c in analytic_cols if c in df.columns]
    df[available_cols].to_csv(EXPORT_DIR / 'analytic_data.csv', index=False, encoding='utf-8-sig')
    print(f"  Сохранено: {EXPORT_DIR / 'analytic_data.csv'}")


def print_descriptive_stats(df: pd.DataFrame) -> None:
    """
    Вывести описательную статистику
    """
    print("\n" + "=" * 70)
    print("ОПИСАТЕЛЬНАЯ СТАТИСТИКА")
    print("=" * 70)
    
    print(f"\nВсего респондентов: {len(df)}")
    
    # Демография
    print("\n--- Демография ---")
    if 'age' in df.columns:
        print(f"  Возраст: M={df['age'].mean():.1f}, SD={df['age'].std():.1f}")
    
    if 'gender' in df.columns:
        print(f"  Пол:")
        print(df['gender'].value_counts())
    
    if 'remote_days' in df.columns:
        print(f"  Удалённая работа:")
        print(df['remote_days'].value_counts())
    
    # Шкалы
    print("\n--- Шкалы ---")
    scale_cols = [
        'mijs_total', 'mbi_total', 'mbi_exhaustion', 'mbi_cynicism', 'mbi_efficacy',
        'swls_total', 'procrastination_total',
        'practices_freq_total', 'vaccines_total', 'social_desirability_total'
    ]
    
    for col in scale_cols:
        if col in df.columns:
            print(f"  {col}: M={df[col].mean():.2f}, SD={df[col].std():.2f}, "
                  f"Min={df[col].min()}, Max={df[col].max()}")


def main():
    """
    Основная функция предобработки
    """
    # 1. Загрузка из MySQL
    df = load_from_mysql()
    
    # 2. Фильтрация невалидных
    df = filter_invalid(df)
    
    # 3. Распарсить JSON колонки
    json_columns = [
        'mijs_items', 'swls_items', 'mbi_exhaustion_items', 'mbi_cynicism_items',
        'mbi_efficacy_items', 'procrastination_items',
        'practices_frequency', 'practices_quality', 'vaccines'
    ]
    
    for col in json_columns:
        if col in df.columns:
            df = parse_json_column(df, col)
    
    # 4. Расчёт шкал
    df = calculate_scales(df)
    
    # 5. Сохранение
    save_clean_data(df)
    
    # 6. Описательная статистика
    print_descriptive_stats(df)
    
    print("\n" + "=" * 70)
    print(f"Время завершения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == '__main__':
    main()
