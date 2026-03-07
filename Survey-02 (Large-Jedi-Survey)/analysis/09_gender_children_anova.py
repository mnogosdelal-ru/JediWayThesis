#!/usr/bin/env python3
"""
09_gender_differences.py

Гендерные различия и взаимодействие Пол × Дети → Выгорание
Проверка гипотез H12, H13, H14, H15, H19

H12: Женщины > Мужчины : Планирование
H13: Женщины > Мужчины : Стресс (MBI)
H14: Пол × Дети → Выгорание (2×2 ANOVA)
H15: Пол модерит связь Планирование → MIJS
H19: Гендерные различия в конкретных практиках

Использование:
    python 09_gender_differences.py
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import pingouin as pg
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from config import *

print("=" * 70)
print("ГЕНДЕРНЫЕ РАЗЛИЧИЯ И ВЗАИМОДЕЙСТВИЕ ПОЛ × ДЕТИ")
print("=" * 70)


def load_data() -> pd.DataFrame:
    """Загрузить чистые данные"""
    df = pd.read_csv(EXPORT_DIR / 'clean_data.csv')
    print(f"Загружено {len(df)} респондентов")
    return df


def preprocess_gender(df: pd.DataFrame) -> pd.DataFrame:
    """Подготовить переменную пола"""
    # Фильтр: только мужчины и женщины
    df = df[df['gender'].isin(['male', 'female'])].copy()
    
    # Кодирование: 0 = male, 1 = female
    df['gender_numeric'] = (df['gender'] == 'female').astype(int)
    
    print(f"Женщины: N = {len(df[df['gender'] == 'female'])}")
    print(f"Мужчины: N = {len(df[df['gender'] == 'male'])}")
    
    return df


def h12_planning_gender(df: pd.DataFrame) -> None:
    """
    H12: Женщины чаще мужчин выполняют практики планирования
    Тест: U-тест Манна-Уитни
    """
    print("\n" + "=" * 70)
    print("H12: Планирование (Женщины > Мужчины)")
    print("Тест: U-тест Манна-Уитни")
    print("=" * 70)
    
    women = df[df['gender_numeric'] == 1]['planning_index']
    men = df[df['gender_numeric'] == 0]['planning_index']
    
    # U-тест
    u_stat, p_value = stats.mannwhitneyu(women, men, alternative='two-sided')
    
    # Размер эффекта: rank-biserial correlation
    rbc = 1 - (2 * u_stat) / (len(women) * len(men))
    
    print(f"  Женщины: Median = {women.median():.2f}, IQR = {women.quantile(0.75) - women.quantile(0.25):.2f}")
    print(f"  Мужчины: Median = {men.median():.2f}, IQR = {men.quantile(0.75) - men.quantile(0.25):.2f}")
    print(f"  U = {u_stat:.0f}, p = {p_value:.4f}")
    print(f"  r (rank-biserial) = {rbc:.2f}")
    
    if p_value < 0.05 and rbc > 0:
        print("  ✓ H12 ПОДТВЕРЖДЕНА: Женщины чаще используют планирование")
    else:
        print("  ✗ H12 НЕ ПОДТВЕРЖДЕНА")
    
    # Визуализация
    plt.figure(figsize=(8, 6))
    plt.boxplot([women, men], labels=['Женщины', 'Мужчины'], showmeans=True)
    plt.ylabel('Индекс планирования (1-6)')
    plt.title('H12: Гендерные различия в планировании')
    plt.savefig(FIGURES_DIR / 'h12_planning_gender.png', dpi=300)
    plt.close()


def h13_stress_gender(df: pd.DataFrame) -> None:
    """
    H13: Женщины сообщают о более высоком выгорании
    Тест: t-тест Стьюдента
    """
    print("\n" + "=" * 70)
    print("H13: Выгорание MBI (Женщины > Мужчины)")
    print("Тест: t-тест Стьюдента")
    print("=" * 70)
    
    women = df[df['gender_numeric'] == 1]['mbi_exhaustion']
    men = df[df['gender_numeric'] == 0]['mbi_exhaustion']
    
    # Проверка гомогенности дисперсий
    levene_stat, levene_p = stats.levene(women, men)
    equal_var = levene_p > 0.05
    
    # t-тест
    t_stat, p_value = stats.ttest_ind(women, men, equal_var=equal_var)
    
    # d Коэна
    cohens_d = (women.mean() - men.mean()) / np.sqrt((women.std()**2 + men.std()**2) / 2)
    
    print(f"  Нормальность (Levene): p = {levene_p:.3f} → equal_var = {equal_var}")
    print(f"  Женщины: M = {women.mean():.2f}, SD = {women.std():.2f}")
    print(f"  Мужчины: M = {men.mean():.2f}, SD = {men.std():.2f}")
    print(f"  t({len(df)-2:.0f}) = {t_stat:.2f}, p = {p_value:.4f}")
    print(f"  d Коэна = {cohens_d:.2f}")
    
    if p_value < 0.05 and cohens_d > 0:
        print("  ✓ H13 ПОДТВЕРЖДЕНА: Женщины сообщают о более высоком выгорании")
    else:
        print("  ✗ H13 НЕ ПОДТВЕРЖДЕНА")
    
    # Визуализация
    plt.figure(figsize=(8, 6))
    plt.boxplot([women, men], labels=['Женщины', 'Мужчины'], showmeans=True)
    plt.ylabel('MBI: Эмоциональное истощение (9-63)')
    plt.title('H13: Гендерные различия в выгорании')
    plt.savefig(FIGURES_DIR / 'h13_stress_gender.png', dpi=300)
    plt.close()


def h14_gender_children_interaction(df: pd.DataFrame) -> None:
    """
    H14: Взаимодействие Пол × Дети → Выгорание
    Тест: 2×2 ANOVA
    """
    print("\n" + "=" * 70)
    print("H14: Взаимодействие Пол × Дети → Выгорание")
    print("Тест: Двухфакторный дисперсионный анализ (2×2 ANOVA)")
    print("=" * 70)
    
    # Создание переменной "Есть дети"
    df['has_children'] = (df['children_count'] > 0).astype(int)
    df['has_children_label'] = df['has_children'].map({0: 'Без детей', 1: 'С детьми'})
    df['gender_label'] = df['gender'].map({'male': 'Мужчины', 'female': 'Женщины'})
    
    # Групповые средние
    print("\n  Групповые средние (MBI: Эмоциональное истощение):")
    group_stats = df.groupby(['gender_label', 'has_children_label'])['mbi_exhaustion'].agg(['mean', 'std', 'count'])
    print(group_stats)
    
    # 2×2 ANOVA
    model = smf.ols('mbi_exhaustion ~ C(gender_numeric) * C(has_children)', data=df).fit()
    anova_table = pg.anova(model, detailed=True)
    
    print("\n  ANOVA таблица:")
    print(anova_table)
    
    # Проверка взаимодействия
    interaction_p = anova_table[anova_table['Source'] == 'C(gender_numeric):C(has_children)']['p-unc'].values[0]
    interaction_eta2 = anova_table[anova_table['Source'] == 'C(gender_numeric):C(has_children)']['np2'].values[0]
    
    print(f"\n  Взаимодействие Пол × Дети:")
    print(f"    p = {interaction_p:.4f}, η² = {interaction_eta2:.3f}")
    
    if interaction_p < 0.05:
        print("  ✓ H14 ПОДТВЕРЖДЕНА: Существует значимое взаимодействие")
        
        # Post-hoc тесты
        print("\n  Post-hoc сравнения:")
        posthoc = pg.pairwise_tests(dv='mbi_exhaustion', between=['gender_numeric', 'has_children'], 
                                    data=df, padjust='bonferroni')
        print(posthoc)
    else:
        print("  ✗ H14 НЕ ПОДТВЕРЖДЕНА: Взаимодействие не значимо")
    
    # Визуализация взаимодействия
    plt.figure(figsize=(10, 6))
    
    # Групповые средние
    group_means = df.groupby(['gender_label', 'has_children_label'])['mbi_exhaustion'].mean().unstack()
    
    x = np.arange(2)
    width = 0.35
    
    plt.bar(x - width/2, group_means['Без детей'], width, label='Без детей', color='#2E86AB')
    plt.bar(x + width/2, group_means['С детьми'], width, label='С детьми', color='#A23B72')
    
    plt.xlabel('Пол')
    plt.ylabel('MBI: Эмоциональное истощение (9-63)')
    plt.title('H14: Взаимодействие Пол × Дети → Выгорание')
    plt.xticks(x, group_means.index)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.savefig(FIGURES_DIR / 'h14_gender_children_interaction.png', dpi=300)
    plt.close()


def h15_moderation_gender(df: pd.DataFrame) -> None:
    """
    H15: Пол модерит связь Планирование → MIJS
    Тест: Регрессия с взаимодействием
    """
    print("\n" + "=" * 70)
    print("H15: Модерация гендером (Планирование → MIJS)")
    print("Тест: Регрессия с взаимодействием")
    print("=" * 70)
    
    # Модель с взаимодействием
    model = smf.ols('mijs_total ~ planning_index * C(gender_numeric)', data=df).fit()
    
    print(model.summary())
    
    # Проверка взаимодействия
    interaction_p = model.pvalues['planning_index:C(gender_numeric)[T.1]']
    interaction_coef = model.params['planning_index:C(gender_numeric)[T.1]']
    
    print(f"\n  Взаимодействие: β = {interaction_coef:.3f}, p = {interaction_p:.4f}")
    
    if interaction_p < 0.05:
        print("  ✓ H15 ПОДТВЕРЖДЕНА: Гендер модерит связь планирования с MIJS")
    else:
        print("  ✗ H15 НЕ ПОДТВЕРЖДЕНА")
    
    # Визуализация
    plt.figure(figsize=(10, 6))
    
    for gender, label in [(0, 'Мужчины'), (1, 'Женщины')]:
        subset = df[df['gender_numeric'] == gender]
        plt.scatter(subset['planning_index'], subset['mijs_total'], 
                   alpha=0.3, label=label, s=50)
        
        # Линия регрессии
        z = np.polyfit(subset['planning_index'], subset['mijs_total'], 1)
        p = np.poly1d(z)
        plt.plot(subset['planning_index'], p(subset['planning_index']), 
                'r-', linewidth=2 if gender == 1 else 1)
    
    plt.xlabel('Индекс планирования')
    plt.ylabel('MIJS (бремя срочности, 12-60)')
    plt.title('H15: Модерация гендером')
    plt.legend()
    plt.savefig(FIGURES_DIR / 'h15_moderation_gender.png', dpi=300)
    plt.close()


def h19_gender_practices(df: pd.DataFrame) -> None:
    """
    H19: Гендерные различия в конкретных практиках
    Тест: U-тест Манна-Уитни с поправкой Бонферрони
    """
    print("\n" + "=" * 70)
    print("H19: Гендерные различия в конкретных практиках")
    print("Тест: U-тест Манна-Уитни с поправкой Бонферрони")
    print("=" * 70)
    
    practice_cols = [f'practices_frequency_{i}' for i in range(1, 22)]
    available_cols = [c for c in practice_cols if c in df.columns]
    
    results = []
    alpha_corrected = 0.05 / len(available_cols)  # Поправка Бонферрони
    
    print(f"  Поправка Бонферрони: α = {alpha_corrected:.4f}")
    print()
    
    for col in available_cols:
        women = df[df['gender_numeric'] == 1][col]
        men = df[df['gender_numeric'] == 0][col]
        
        u_stat, p_value = stats.mannwhitneyu(women, men, alternative='two-sided')
        
        results.append({
            'practice': col.replace('practices_frequency_', 'P'),
            'women_median': women.median(),
            'men_median': men.median(),
            'u_stat': u_stat,
            'p_value': p_value,
            'significant': p_value < alpha_corrected
        })
    
    results_df = pd.DataFrame(results)
    print(results_df[['practice', 'women_median', 'men_median', 'p_value', 'significant']])
    
    # Сохранить результаты
    results_df.to_csv(TABLES_DIR / 'h19_gender_practices.csv', index=False)
    print(f"\n  Результаты сохранены в {TABLES_DIR / 'h19_gender_practices.csv'}")


def main():
    """Основная функция"""
    # Загрузка данных
    df = load_data()
    
    # Предобработка
    df = preprocess_gender(df)
    
    # Проверка гипотез
    h12_planning_gender(df)
    h13_stress_gender(df)
    h14_gender_children_interaction(df)
    h15_moderation_gender(df)
    h19_gender_practices(df)
    
    print("\n" + "=" * 70)
    print("АНАЛИЗ ЗАВЕРШЁН")
    print("=" * 70)


if __name__ == '__main__':
    main()
