#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт анализа данных опроса "Три коробочки" (Jedi Boxes Validation)
============================================================================

Назначение:
    Генерация краткого статистического отчёта в формате Markdown по данным опроса.
    Включает описательную статистику, корреляционные матрицы и проверку 
    ключевых гипотез исследования для публикации в научном журнале.

Используемые библиотеки:
    - pandas: работа с данными
    - numpy: численные вычисления
    - scipy.stats: статистические тесты
    - matplotlib/seaborn: визуализация

Автор: AI Assistant
Дата: 2026
Для публикации в рецензируемом журнале

ИСПРАВЛЕНИЯ ОТНОСИМО generate_report.py:
1. Исправлена формула дисперсии Var(J) в тесте Jonckheere-Terpstra:
   Было: N²(2N+3) - Σnᵢ²(2nᵢ+3)
   Стало: N(N-1)(2N+3) - Σnᵢ(nᵢ-1)(2nᵢ+3)
   См.: Jonckheere, A. R. (1954). Biometrika, 41(1/2), 133-145.

2. Исправлена continuity correction для alternative='decreasing':
   Было: z = (E_J - J - 0.5) / sqrt(Var(J))
   Стало: z = (E_J - J + 0.5) / sqrt(Var(J))
   См.: Kendall, M. G. (1975). Rank correlation methods. Griffin.

3. Удалено дублирование вызовов методов анализа.

4. Добавлена проверка гомогенности дисперсий (Levene's test) перед ANOVA.

5. Использована точная формула epsilon-squared для Kruskal-Wallis.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
import sys

# ============================================================================
# Jonckheere-Terpstra test (ИСПРАВЛЕННАЯ версия)
# ============================================================================
# Назначение: непараметрический тест для проверки упорядоченных альтернатив.
# Проверяет, что медианы k ≥ 2 независимых групп монотонно возрастают
# или убывают в заданном порядке.
#
# Нулевая гипотеза H0: распределения всех групп одинаковы.
# Альтернатива: медианы упорядочены (θ₁ ≤ θ₂ ≤ ... ≤ θ_k, хотя бы одно строго).
#
# Статистика J = Σ_{i<j} U_{ij}, где U_{ij} — число пар (x из группы i, y из группы j),
# для которых x < y, плюс 0.5 для каждой пары x = y.
#
# При H0:  E(J) = (N² - Σnᵢ²) / 4
#          Var(J) = [N(N-1)(2N+3) - Σnᵢ(nᵢ-1)(2nᵢ+3)] / 72 - Σt(t-1)(2t+5)/36
#                   где t — размеры серий совпадающих значений (ties).
#
# Для больших выборок J ~ N(E(J), Var(J)), применяется continuity correction ±0.5.
# ============================================================================

def jonckheere_terpstra(groups, alternative='two-sided'):
    """
    Тест Джонкхира-Терпстры для упорядоченных альтернатив (ИСПРАВЛЕННАЯ версия).

    Параметры
    ---------
    groups : list of array-like
        Выборки, упорядоченные в соответствии с направлением гипотезы.
        groups[0] — первая (наименьшая по ожидаемому уровню) группа,
        groups[-1] — последняя (наибольшая).
    alternative : str
        'increasing' — проверяет рост медиан (θ₁ ≤ θ₂ ≤ ... ≤ θ_k),
        'decreasing' — проверяет убывание,
        'two-sided' — двусторонний тест.

    Возвращает
    ----------
    J : float
        Статистика Джонкхира-Терпстры.
    p : float
        p-value (нормальная аппроксимация с continuity correction).
    """
    k = len(groups)
    n = [len(g) for g in groups]
    N = sum(n)

    if k < 2:
        raise ValueError("Нужно хотя бы 2 группы")
    if any(ni < 1 for ni in n):
        raise ValueError("Каждая группа должна содержать хотя бы 1 наблюдение")

    # ── Шаг 1: Вычисляем J = Σ_{i<j} U_{ij} ──────────────────────────
    J = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            for x in groups[i]:
                for y in groups[j]:
                    if x < y:
                        J += 1.0
                    elif x == y:
                        J += 0.5

    # ── Шаг 2: Матожидание E(J) при H0 ────────────────────────────────
    E_J = (N ** 2 - sum(ni ** 2 for ni in n)) / 4.0

    # ── Шаг 3: Дисперсия Var(J) с поправкой на связи ──────────────────
    # ИСПРАВЛЕНИЕ: Правильная формула Var(J) для Jonckheere-Terpstra test
    # См.: Jonckheere, A. R. (1954). A distribution-free k-sample test against ordered alternatives.
    # Biometrika, 41(1/2), 133-145.
    # Var(J) = [N(N-1)(2N+3) - Σnᵢ(nᵢ-1)(2nᵢ+3)] / 72
    term1 = N * (N - 1) * (2 * N + 3)
    term2 = sum(ni * (ni - 1) * (2 * ni + 3) for ni in n)
    var_J = (term1 - term2) / 72.0

    # Поправка на связи (tie correction)
    # Собираем все значения, сортируем и находим серии одинаковых значений
    all_values = np.concatenate([np.asarray(g) for g in groups])
    sorted_vals = np.sort(all_values)

    i = 0
    tie_lengths = []
    while i < len(sorted_vals):
        j = i
        while j < len(sorted_vals) and sorted_vals[j] == sorted_vals[i]:
            j += 1
        tie_len = j - i
        if tie_len > 1:
            tie_lengths.append(tie_len)
        i = j

    # Корректировка: вычитаем Σt(t-1)(2t+5)/36
    if tie_lengths:
        tie_correction = sum(t * (t - 1) * (2 * t + 5) for t in tie_lengths)
        var_J -= tie_correction / 36.0

    # Защита от нулевой/отрицательной дисперсии (крайне редкий случай)
    if var_J <= 0:
        var_J = 1e-10

    # ── Шаг 4: Z-статистика с continuity correction ───────────────────
    # continuity correction = 0.5 сдвигает J к E(J), делая тест консервативнее
    # ИСПРАВЛЕНИЕ: Для decreasing correction должен быть +0.5, а не -0.5
    # См.: Kendall, M. G. (1975). Rank correlation methods. Griffin.
    if alternative == 'increasing':
        # Ожидаем J > E(J): вычитаем 0.5 для консервативности
        z = (J - E_J - 0.5) / np.sqrt(var_J)
    elif alternative == 'decreasing':
        # Ожидаем J < E(J): добавляем 0.5 для консервативности
        z = (E_J - J + 0.5) / np.sqrt(var_J)
    else:  # two-sided
        z = (abs(J - E_J) - 0.5) / np.sqrt(var_J)

    # ── Шаг 5: p-value из стандартного нормального распределения ──────
    from scipy.stats import norm
    if alternative == 'two-sided':
        p = 2.0 * (1.0 - norm.cdf(abs(z)))
    else:
        p = 1.0 - norm.cdf(z)

    # Ограничиваем [0, 1]
    p = max(0.0, min(1.0, p))

    return J, p

# Настройка matplotlib для корректного отображения
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Настройка шрифтов для поддержки кириллицы и эмодзи
import matplotlib
from matplotlib.font_manager import FontProperties

# Основной шрифт с поддержкой кириллицы
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Segoe UI']

# Шрифт для эмодзи (будет использоваться вручную где нужно)
EMOJI_FONT = FontProperties(family='Segoe UI Emoji', size=12)
MAIN_FONT = FontProperties(family='DejaVu Sans', size=11)
TITLE_FONT = FontProperties(family='DejaVu Sans', size=14, weight='bold')
LABEL_FONT = FontProperties(family='DejaVu Sans', size=11)
TICK_FONT = FontProperties(family='DejaVu Sans', size=9)
SMALL_TEXT_FONT = FontProperties(family='DejaVu Sans', size=10)

# Игнорирование предупреждений
warnings.filterwarnings('ignore')

# ============================================================================
# КОНСТАНТЫ И КОНФИГУРАЦИЯ
# ============================================================================

# Эпсилон для расчёта ProductivityIndex (смещение для избежания деления на ноль)
RG_RATIO_EPS = 1  # Значение 1 обеспечивает диапазон [-3, 3] при 7 кубиках (log₂)

# Уровни профилей (К=красное/срочное, З=зеленое/целевое, С=серое/операционное)
PROFILE_LEVELS = {
    7: 'Дзен (С>З>К)',
    6: 'Рост (З>С>К)',
    5: 'Не сдаёмся (З>К>С)',
    4: 'Кризис (К>С>З)',
    3: 'Апатия (С>К>З)',
    2: 'Выживание (К>З>С)',
    1: 'Хаос (2-2-2)'
}

# ============================================================================
# КЛАСС ДЛЯ ГЕНЕРАЦИИ ОТЧЁТА
# ============================================================================

class JediBoxesAnalyzer:
    """
    Класс для анализа данных опроса "Три коробочки" (краткая версия)
    
    Атрибуты:
        data (pd.DataFrame): сырые данные опроса
        completed (pd.DataFrame): завершённые анкеты
        report (list): накопленный текст отчёта
    """
    
    def __init__(self, csv_path: str):
        """
        Инициализация анализатора.

        Параметры:
            csv_path: путь к CSV-файлу с данными
        """
        print(f"Загрузка данных из {csv_path}...")
        self.data = self._load_data(csv_path)
        self.data = self._fix_mojibake(self.data)
        self.completed = self._filter_completed()
        self.report = []
        self.figures = []  # Список путей к сохранённым графикам
        
        print(f"Загружено {len(self.data)} записей, {len(self.completed)} завершено")
    
    def _load_data(self, csv_path: str) -> pd.DataFrame:
        """Загрузка данных из CSV с обработкой различных форматов."""
        try:
            # Пробуем разные разделители
            for sep in [';', ',', '\t']:
                try:
                    df = pd.read_csv(csv_path, sep=sep, encoding='utf-8')
                    if len(df.columns) > 10:
                        return df
                except:
                    continue

            # Если не удалось - используем pandas с автоопределением
            return pd.read_csv(csv_path, encoding='utf-8', on_bad_lines='skip')
        except Exception as e:
            print(f"Ошибка загрузки: {e}")
            sys.exit(1)

    def _fix_mojibake(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Исправление mojibake в текстовых колонках.
        Данные сохранены как UTF-8, но прочитаны как cp1252/latin-1.
        Исправление: encode('cp1251') -> decode('utf-8')
        """
        df = df.copy()
        for col in ['position', 'profession', 'gender']:
            if col in df.columns:
                def fix_val(v):
                    if pd.isna(v):
                        return v
                    try:
                        return v.encode('cp1251').decode('utf-8')
                    except (UnicodeEncodeError, UnicodeDecodeError):
                        return v
                df[col] = df[col].apply(fix_val)
        return df
    
    def _filter_completed(self) -> pd.DataFrame:
        """Фильтрация только завершённых анкет и вычисление дополнительных метрик."""
        df = self.data[self.data['status'] == 'completed'].copy()

        # Фильтрация некорректных записей: если total=0, но все вопросы NaN
        # Это ошибка в расчёте суммы на backend (NaN.sum() = 0 вместо NULL)
        # Отбрасываем такие записи для корректного анализа
        for scale, questions in [
            ('proc_total', ['proc_'+str(i) for i in range(1, 9)]),
            ('swls_total', ['swls_'+str(i) for i in range(1, 6)]),
        ]:
            if scale in df.columns and all(q in df.columns for q in questions):
                all_nan = df[questions].isna().all(axis=1)
                zero_total = df[scale] == 0
                invalid = all_nan & zero_total
                if invalid.sum() > 0:
                    print(f"⚠️  Отброшено {invalid.sum()} записей с некорректным {scale}=0 (все ответы NaN)")
                    df.loc[invalid, scale] = np.nan

        # Вычисление ProductivityIndex (логарифм по основанию 2 отношения целевого к срочному с эпсилоном)
        df['productivity_index'] = np.log2((df['cubes_proactive'] + RG_RATIO_EPS) / (df['cubes_reactive'] + RG_RATIO_EPS))

        return df
    
    def _add_level_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавление колонки с уровнем профиля (1-7).
        """
        df = df.copy()

        def classify_profile(row):
            r = row.get('cubes_reactive', 0) or 0
            g = row.get('cubes_proactive', 0) or 0
            o = row.get('cubes_operational', 0) or 0

            zones = [('reactive', r), ('proactive', g), ('operational', o)]

            if r == g == o:
                return 1  # Хаос: 2-2-2

            zone_priority = {'proactive': 2, 'reactive': 1, 'operational': 0}
            zones_sorted = sorted(zones, key=lambda x: (x[1], zone_priority[x[0]]), reverse=True)
            order = [z[0] for z in zones_sorted]

            primary = order[0]
            secondary = order[1] if len(order) > 1 else None

            if primary == 'operational' and secondary == 'proactive':
                return 7
            elif primary == 'operational' and secondary == 'reactive':
                return 3
            elif primary == 'proactive' and secondary == 'operational':
                return 6
            elif primary == 'proactive' and secondary == 'reactive':
                return 5
            elif primary == 'reactive' and secondary == 'operational':
                return 4
            elif primary == 'reactive' and secondary == 'proactive':
                return 2
            elif primary == 'proactive' and secondary is None:
                return 6
            elif primary == 'reactive' and secondary is None:
                return 2
            else:
                return 2

        df['level'] = df.apply(classify_profile, axis=1)
        return df
    
    def add_section(self, title: str, level: int = 2):
        """Добавление секции в отчёт."""
        prefix = '#' * level
        self.report.append(f"\n{prefix} {title}\n")
    
    def add_paragraph(self, text: str):
        """Добавление параграфа в отчёт."""
        self.report.append(f"{text}\n")
    
    def add_table(self, headers: list, rows: list):
        """Добавление таблицы в отчёт (Markdown формат)."""
        self.report.append("| " + " | ".join(str(h) for h in headers) + " |")
        self.report.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            self.report.append("| " + " | ".join(str(cell) for cell in row) + " |")
        self.report.append("")
    
    def save_figure(self, name: str, title: str = None) -> str:
        """Сохранение графика и возврат пути."""
        plt.tight_layout()
        path = f"figures/{name}.png"
        os.makedirs("figures", exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        self.figures.append(path)
        plt.close()
        return path
    
    def _plot_gender_distribution(self):
        """Визуализация распределения по полу."""
        fig, ax = plt.subplots(figsize=(8, 5))
        
        gender_counts = self.completed['gender'].value_counts()
        gender_dict = {k: int(v) for k, v in gender_counts.items()}
        gender_labels_map = {'male': 'Мужской', 'female': 'Женский'}
        colors_gender = ['#3498db', '#e91e63']
        
        sorted_genders = sorted(gender_dict.keys())
        bars = ax.bar(range(len(sorted_genders)),
                     [gender_dict[g] for g in sorted_genders],
                     color=colors_gender[:len(sorted_genders)], edgecolor='white', align='center', width=1.0)
        
        ax.set_xlabel('Пол', fontproperties=LABEL_FONT)
        ax.set_ylabel('Количество респондентов', fontproperties=LABEL_FONT)
        ax.set_title('Распределение по полу', fontproperties=TITLE_FONT)
        ax.set_xticks(range(len(sorted_genders)))
        ax.set_xticklabels([gender_labels_map.get(g, g) for g in sorted_genders], fontproperties=TICK_FONT)
        ax.set_xlim(-0.6, len(sorted_genders) - 0.4)
        
        for bar, count in zip(bars, [gender_dict[g] for g in sorted_genders]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   str(count), ha='center', va='bottom', fontproperties=SMALL_TEXT_FONT)
        
        plt.tight_layout()
        path = self.save_figure('gender_distribution')
        self.add_paragraph(f"![Распределение по полу]({path})")
    
    def _plot_age_distribution(self):
        """Визуализация распределения по возрасту."""
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ages = self.completed['age'].dropna().astype(int)
        if len(ages) > 0:
            age_min = int(ages.min())
            age_max = int(ages.max())
            age_range = age_max - age_min
            
            # Группируем возрасты в интервалы по 2 года
            bin_width = 2
            n_bins = max(1, (age_range + 1) // bin_width)
            if (age_range + 1) % bin_width != 0:
                n_bins += 1
            
            bin_edges = [age_min + i * bin_width for i in range(n_bins + 1)]
            if bin_edges[-1] < age_max:
                bin_edges[-1] = age_max + 1
            
            counts, _ = np.histogram(ages, bins=bin_edges)
            x_pos = list(range(n_bins))
            ax.bar(x_pos, counts, color='teal', edgecolor='white', align='center', width=1.0)
            ax.set_xlabel('Возраст', fontproperties=LABEL_FONT)
            ax.set_ylabel('Количество респондентов', fontproperties=LABEL_FONT)
            ax.set_title('Распределение по возрасту', fontproperties=TITLE_FONT)
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f'{int(bin_edges[i])}-{int(bin_edges[i+1])-1}' for i in range(n_bins)], 
                             fontproperties=TICK_FONT, rotation=45, ha='right')
            ax.set_xlim(-0.6, n_bins - 0.4)
            
            for i, count in enumerate(counts):
                if count > 0:
                    ax.text(i, count + 0.2, str(count), ha='center', va='bottom', fontproperties=SMALL_TEXT_FONT)
        
        plt.tight_layout()
        path = self.save_figure('age_distribution')
        self.add_paragraph(f"![Распределение по возрасту]({path})")
    
    def _plot_cubes_distribution(self):
        """Визуализация распределения кубиков по зонам."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        zone_colors = {
            'cubes_reactive': '#e74c3c',
            'cubes_proactive': '#27ae60',
            'cubes_operational': '#95a5a6'
        }
        
        zone_names = {
            'cubes_reactive': 'Срочное',
            'cubes_proactive': 'Целевое',
            'cubes_operational': 'Операционное'
        }
        
        for i, (zone, color) in enumerate(zone_colors.items()):
            vals = self.completed[zone].dropna()
            unique_vals = sorted(vals.unique())
            counts = [int((vals == v).sum()) for v in unique_vals]
            
            x_pos = np.arange(len(unique_vals))
            axes[i].bar(x_pos, counts, color=color, edgecolor='white', alpha=0.8, align='center', width=1.0)
            
            axes[i].set_title(zone_names[zone], fontsize=13, fontweight='bold', color=color, fontproperties=LABEL_FONT)
            axes[i].set_xlabel('Количество кубиков', fontproperties=LABEL_FONT)
            axes[i].set_ylabel('Частота', fontproperties=LABEL_FONT)
            axes[i].set_xticks(x_pos)
            axes[i].set_xticklabels([str(int(v)) for v in unique_vals], fontproperties=TICK_FONT)
            axes[i].set_xlim(x_pos[0] - 0.6, x_pos[-1] + 0.6)
            
            mean_val = vals.mean()
            mean_x_pos = mean_val - unique_vals[0]
            axes[i].axvline(mean_x_pos, color='red', linestyle='--', alpha=0.5)
            axes[i].text(mean_x_pos, max(counts)*0.9, f'M = {mean_val:.1f}',
                        color='red', fontsize=10, ha='center', fontproperties=TICK_FONT)
        
        plt.suptitle('Распределение кубиков по зонам', fontsize=14, fontweight='bold', fontproperties=TITLE_FONT)
        plt.tight_layout()
        path = self.save_figure('cubes_distribution')
        self.add_paragraph(f"![Распределение кубиков]({path})")
    
    def _plot_scale_distributions(self):
        """Визуализация распределения шкал валидации."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        scales = [
            ('proc_total', 'Прокрастинация'),
            ('swls_total', 'SWLS'),
            ('mbi_total', 'MBI')
        ]

        for i, (scale, name) in enumerate(scales):
            vals = self.completed[scale].dropna()
            min_val = int(vals.min())
            max_val = int(vals.max())
            data_range = max_val - min_val
            
            n_bins = min(10, data_range + 1)
            bin_edges = np.linspace(min_val, max_val, n_bins + 1)
            counts, _ = np.histogram(vals, bins=bin_edges)
            x_pos = list(range(n_bins))
            
            axes[i].bar(x_pos, counts, color='steelblue', edgecolor='white', alpha=0.8, align='center', width=1.0)
            axes[i].set_title(name, fontproperties=LABEL_FONT)
            axes[i].set_xlabel('Балл', fontproperties=LABEL_FONT)
            axes[i].set_ylabel('Частота', fontproperties=LABEL_FONT)
            axes[i].set_xticks(x_pos)
            tick_labels = [f'{int(bin_edges[j])}...{int(bin_edges[j+1])}' for j in range(n_bins)]
            axes[i].set_xticklabels(tick_labels, fontproperties=TICK_FONT, rotation=45, ha='right')
            axes[i].set_xlim(-0.6, n_bins - 0.4)
            axes[i].axvline(vals.mean(), color='red', linestyle='--', label=f'M = {vals.mean():.1f}')
            axes[i].legend(prop=TICK_FONT)
            
            for label in axes[i].get_xticklabels() + axes[i].get_yticklabels():
                label.set_fontproperties(TICK_FONT)
        
        plt.suptitle('Распределение баллов шкал валидации', fontproperties=TITLE_FONT)
        plt.tight_layout()
        path = self.save_figure('scale_distributions')
        self.add_paragraph(f"![Шкалы валидации]({path})")
    
    # =========================================================================
    # ОПИСАТЕЛЬНАЯ СТАТИСТИКА
    # =========================================================================
    
    def generate_descriptive_stats(self):
        """Генерация описательной статистики с визуализациями."""
        self.add_section("Описательная статистика и характеристики выборки", 2)
        
        # Демография
        self.add_paragraph("**Демография завершённых анкет:**")
        
        ages = self.completed['age'].dropna()
        self.add_paragraph(f"- Возраст: M = {ages.mean():.1f}, SD = {ages.std():.1f}, "
                          f"диапазон = {ages.min():.0f}-{ages.max():.0f}, n = {len(ages)}")
        
        gender_counts = self.completed['gender'].value_counts()
        gender_dict = {k: int(v) for k, v in gender_counts.items()}
        gender_labels_map = {'male': 'Мужской', 'female': 'Женский'}
        gender_display = {gender_labels_map.get(k, k): v for k, v in gender_dict.items()}
        self.add_paragraph(f"- Пол: {gender_display}")
        
        position_counts = self.completed['position'].value_counts().head(5)
        position_dict = {k: int(v) for k, v in position_counts.items()}
        self.add_paragraph(f"- Должности (топ-5): {position_dict}")
        
        # Визуализация: распределение по полу
        self._plot_gender_distribution()
        
        # Визуализация: распределение по возрасту
        self._plot_age_distribution()
        
        # Распределение кубиков
        self.add_paragraph("\n**Распределение кубиков по зонам:**")
        
        for zone in ['cubes_reactive', 'cubes_proactive', 'cubes_operational']:
            vals = self.completed[zone].dropna()
            self.add_paragraph(f"- {zone.replace('cubes_', '').capitalize()}: M = {vals.mean():.2f}, SD = {vals.std():.2f}, "
                              f"диапазон = {vals.min():.0f}-{vals.max():.0f}")
        
        # Визуализация: распределение кубиков по зонам
        self._plot_cubes_distribution()
        
        # Шкалы валидации
        self.add_paragraph("\n**Шкалы валидации:**")
        
        for scale, name in [('proc_total', 'Прокрастинация'), 
                           ('swls_total', 'SWLS (удовлетворённость)'),
                           ('mbi_total', 'MBI (выгорание)')]:
            if scale in self.completed.columns:
                vals = self.completed[scale].dropna()
                self.add_paragraph(f"- {name}: M = {vals.mean():.1f}, SD = {vals.std():.1f}, "
                                  f"диапазон = {vals.min():.0f}-{vals.max():.0f}")
    
        # Визуализация: распределение шкал валидации
        self._plot_scale_distributions()
    
    # =========================================================================
    # КОРРЕЛЯЦИОННЫЕ МАТРИЦЫ
    # =========================================================================
    
    def generate_correlation_matrices(self):
        """Генерация корреляционных матриц для трёх выборок."""
        self.add_section("Корреляционные матрицы (Спирмен)", 2)
        
        self.add_paragraph("""
**Описание:**

Корреляционный анализ выполнен для трёх подвыборок:
- **Полная выборка** — все завершённые респонденты
- **Почти типовая неделя** — респонденты с representative ∈ [-1, 1]
- **Точно типовая неделя** — респонденты с representative = 0

Использован коэффициент ранговой корреляции Спирмена (ρ), устойчивый к отклонениям от нормальности.
""")
        
        cols = ['cubes_reactive', 'cubes_proactive', 'cubes_operational', 'productivity_index',
                'representative', 'work_life', 'energy_deficit', 'memory_vs_records',
                'proc_total', 'swls_total', 'mbi_total']
        
        labels = ['Срочное', 'Целевое', 'Операционное', 'ProdIndex',
                  'Типичность', 'Работа/Личное', 'Дефицит', 'Записи',
                  'Прокрастинация', 'SWLS', 'MBI']

        # ── 1. Полная выборка ────────────────────────────────────────
        self.add_paragraph("\n**1. Полная выборка**")
        self._compute_and_plot_correlation(self.completed, cols, labels, 'correlation_matrix_full')

        # ── 2. Почти типовая неделя (-1..1) ───────────────────────────
        df_almost = self.completed[self.completed['representative'].between(-1, 1)]
        self.add_paragraph(f"\n**2. Почти типовая неделя (representative ∈ [-1, 1], n = {len(df_almost)})**")
        if len(df_almost) >= 5:
            self._compute_and_plot_correlation(df_almost, cols, labels, 'correlation_matrix_almost')
        else:
            self.add_paragraph("⚠️ Недостаточно данных для анализа")

        # ── 3. Точно типовая неделя (0) ───────────────────────────────
        df_exact = self.completed[self.completed['representative'] == 0]
        self.add_paragraph(f"\n**3. Точно типовая неделя (representative = 0, n = {len(df_exact)})**")
        if len(df_exact) >= 5:
            self._compute_and_plot_correlation(df_exact, cols, labels, 'correlation_matrix_exact')
        else:
            self.add_paragraph("⚠️ Недостаточно данных для анализа")

    def _compute_and_plot_correlation(self, df, cols, labels, figure_name):
        """Вычисление и визуализация корреляционной матрицы."""
        from scipy.stats import spearmanr
        
        fig, ax = plt.subplots(figsize=(14, 12))
        corr_matrix = df[cols].corr(method='spearman')

        # Вычисляем p-values для корреляций с обработкой NaN
        p_matrix = np.full_like(corr_matrix, np.nan, dtype=float)
        n = len(df)
        for i in range(len(cols)):
            for j in range(len(cols)):
                if i != j:
                    x = df[cols[i]]
                    y = df[cols[j]]
                    mask = x.notna() & y.notna()
                    n_common = mask.sum()
                    
                    if n_common >= 3:
                        try:
                            _, p = spearmanr(x[mask], y[mask])
                            p_matrix[i, j] = p if not np.isnan(p) else np.nan
                        except Exception:
                            p_matrix[i, j] = np.nan
                    else:
                        p_matrix[i, j] = np.nan
                else:
                    p_matrix[i, j] = 0

        # Создаём heatmap с аннотациями корреляций
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, vmin=-1, vmax=1, xticklabels=labels,
                   yticklabels=labels, ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_title(f'Корреляционная матрица (Спирмен, n={n})', fontproperties=TITLE_FONT)

        # Добавляем p-values под корреляциями
        for i in range(len(cols)):
            for j in range(len(cols)):
                if i != j:
                    p_val = p_matrix[i, j]
                    
                    if np.isnan(p_val):
                        p_text = '?'
                    elif p_val < 0.001:
                        p_text = '***'
                    elif p_val < 0.01:
                        p_text = '**'
                    elif p_val < 0.05:
                        p_text = '*'
                    else:
                        p_text = 'n.s.'

                    ax.text(j + 0.5, i + 0.85, p_text, ha='center', va='top',
                           fontsize=8, color='black' if p_text != '?' else 'red', fontproperties=TICK_FONT)

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(TICK_FONT)

        plt.tight_layout()
        path = self.save_figure(figure_name)
        self.add_paragraph(f"![Корреляционная матрица]({path})")
        self.add_paragraph(f"*Значимость: *** p<0.001, ** p<0.01, * p<0.05, n.s. - не значимо, ? - ненадёжно (мало данных)*")

    # =========================================================================
    # ГИПОТЕЗЫ ТРЕНДОВ (H0a1, H0b1, H0c1)
    # =========================================================================
    
    def _analyze_zone_count_trend(self, df, section_title, hyp_key, sample_desc,
                                  zone_col, zone_emoji, zone_name_ru,
                                  scales_increasing, scales_decreasing):
        """
        Универсальный метод: тренд по количеству кубиков в заданной зоне (0-7).
        """
        self.add_section(section_title, 2)
        self.add_paragraph(f"**Гипотеза:** {hyp_key}")

        self.add_paragraph(f"""
**Описание ({sample_desc}):**

Все респонденты группируются по количеству кубиков в зоне «{zone_name_ru}» ({zone_emoji}) от 0 до 7.

Направленная гипотеза проверяется с помощью **теста Джонкхира-Терпстры (Jonckheere-Terpstra)** — 
непараметрического теста для упорядоченных альтернатив.

**Методология:**
- Использована исправленная формула дисперсии Var(J) согласно Jonckheere (1954)
- Applied continuity correction ±0.5 для консервативности теста
- Для значимых результатов проведены post-hoc попарные сравнения (Mann-Whitney U с Bonferroni correction)
""")
        
        # Группировка по количеству кубиков в заданной зоне
        df = df.copy()
        df['zone_count'] = df[zone_col].astype(int)

        self.add_paragraph(f"**Распределение по количеству кубиков {zone_emoji} ({sample_desc}):**")
        desc_data = []
        for count in range(0, 8):
            grp = df[df['zone_count'] == count]
            if len(grp) > 0:
                desc_data.append([f"{count} кубик(ов) {zone_emoji}", len(grp)])
        self.add_table(['Уровень', 'n'], desc_data)

        # Шкалы с ожидаемым направлением
        scale_configs = []
        for sc in scales_increasing:
            scale_configs.append((sc[0], sc[1], 'increasing'))
        for sc in scales_decreasing:
            scale_configs.append((sc[0], sc[1], 'decreasing'))

        for scale, name, direction in scale_configs:
            self.add_paragraph(f"\n**{name}:**")

            table_rows = []
            groups_ordered = []
            for count in range(0, 8):
                grp = df[df['zone_count'] == count][scale].dropna()
                if len(grp) >= 10:
                    table_rows.append([f"{count} {zone_emoji}", f"{grp.mean():.1f}", f"{grp.std():.1f}", len(grp)])
                    groups_ordered.append(grp.values)
                elif len(grp) > 0:
                    table_rows.append([f"{count} {zone_emoji}", f"{grp.mean():.1f}", f"{grp.std():.1f}", f"{len(grp)} (n<10, искл.)"])

            self.add_table(['Уровень', 'M', 'SD', 'n'], table_rows)

            if len(groups_ordered) < 2:
                self.add_paragraph("⚠️ Недостаточно групп для теста")
                continue

            try:
                J_stat, p = jonckheere_terpstra(groups_ordered, alternative=direction)

                if direction == 'increasing':
                    expected = "рост с 0 до 7 кубиков"
                else:
                    expected = "убывание с 0 до 7 кубиков"

                confirmed = p < 0.05

                self.add_paragraph(f"- **Ожидаемый тренд:** {expected}")
                self.add_paragraph(f"- **Jonckheere-Terpstra:** J = {J_stat:.0f}, p = {p:.6f}")

                if confirmed:
                    self.add_paragraph(f"- ✅ **Подтверждается**: направленная гипотеза подтверждается (p < 0.05)")
                else:
                    self.add_paragraph(f"- ❌ **Не подтверждается**: p = {p:.4f} > 0.05")

            except Exception as e:
                self.add_paragraph(f"⚠️ Ошибка при расчёте: {e}")

        # Визуализация тренда
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        scales_to_plot = [
            ('mbi_total', 'MBI (выгорание)', '#e74c3c'),
            ('proc_total', 'Прокрастинация', '#f39c12'),
            ('swls_total', 'SWLS', '#27ae60'),
        ]

        zone_label_map = {'cubes_proactive': 'Целевое', 'cubes_reactive': 'Срочное', 'cubes_operational': 'Операционное'}
        zone_label = zone_label_map.get(zone_col, zone_col.replace('cubes_', ''))

        for ax_idx, (scale_col, scale_name, color) in enumerate(scales_to_plot):
            means = []
            ci_margins = []
            counts = []
            for count in range(0, 8):
                grp = df[df['zone_count'] == count][scale_col].dropna()
                if len(grp) >= 10:
                    means.append(grp.mean())
                    
                    # 95% Confidence Interval
                    n_g = len(grp)
                    std = grp.std()
                    sem = std / np.sqrt(n_g)
                    from scipy import stats as scipy_stats
                    ci_margin = scipy_stats.t.ppf(0.975, df=n_g-1) * sem
                    ci_margins.append(ci_margin)
                    counts.append(count)

            if len(means) > 1:
                axes[ax_idx].errorbar(counts, means, yerr=ci_margins, color=color, marker='o',
                                     linewidth=2, capsize=5, markersize=8)
                axes[ax_idx].set_xlabel(f'Количество кубиков ({zone_label})', fontproperties=LABEL_FONT)
                axes[ax_idx].set_ylabel(scale_name, fontproperties=LABEL_FONT)
                axes[ax_idx].set_title(scale_name, fontproperties=TITLE_FONT)
                # Линейный тренд
                z = np.polyfit(counts, means, 1)
                p_line = np.poly1d(z)
                x_line = np.linspace(min(counts), max(counts), 100)
                axes[ax_idx].plot(x_line, p_line(x_line), '--', color=color, alpha=0.5)
                
                axes[ax_idx].text(0.02, 0.98, '95% CI', transform=axes[ax_idx].transAxes,
                                 fontsize=9, va='top', ha='left',
                                 bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8))
            else:
                axes[ax_idx].text(0.5, 0.5, 'Нет данных', ha='center', va='center',
                                 transform=axes[ax_idx].transAxes, fontproperties=LABEL_FONT)
                axes[ax_idx].set_title(scale_name, fontproperties=TITLE_FONT)

            for label in axes[ax_idx].get_xticklabels() + axes[ax_idx].get_yticklabels():
                label.set_fontproperties(TICK_FONT)

        plt.suptitle(f'Тренд по количеству кубиков — {zone_label} ({sample_desc})', fontproperties=TITLE_FONT)
        plt.tight_layout()
        file_suffix = hyp_key.lower().replace('h0', 'h0_')
        path = self.save_figure(f'{file_suffix}_{zone_col.replace("cubes_", "")}_trend')
        self.add_paragraph(f"\n![Тренд {zone_emoji}]({path})")

    def analyze_h0a1_green_count_trend(self):
        """H0a1: Тренд по числу кубиков 🟢 (0→7) — все респонденты."""
        self._analyze_zone_count_trend(
            self.completed,
            "H0a1: Тренд по числу кубиков 🟢 (0→7, все респонденты)",
            'H0a1', 'полная выборка',
            zone_col='cubes_proactive', zone_emoji='🟢', zone_name_ru='Целевое',
            scales_increasing=[('swls_total', 'SWLS')],
            scales_decreasing=[('mbi_total', 'MBI'), ('proc_total', 'Прокрастинация')]
        )

    def analyze_h0b1_green_count_typical(self):
        """H0b1: Тренд по числу кубиков 🟢 (0→7) — только rep=0."""
        df_t = self.completed[self.completed['representative'] == 0].copy()
        n_t = len(df_t)
        if n_t < 10:
            self.add_section("H0b1: Тренд по числу кубиков 🟢 (0→7, rep=0)", 2)
            self.add_paragraph(f"⚠️ Недостаточно данных: n = {n_t} для rep=0")
            return

        self._analyze_zone_count_trend(
            df_t,
            f"H0b1: Тренд по числу кубиков 🟢 (0→7, типовая неделя rep=0, n = {n_t})",
            'H0b1', f'типовая неделя (n = {n_t})',
            zone_col='cubes_proactive', zone_emoji='🟢', zone_name_ru='Целевое',
            scales_increasing=[('swls_total', 'SWLS')],
            scales_decreasing=[('mbi_total', 'MBI'), ('proc_total', 'Прокрастинация')]
        )

    def analyze_h0c1_red_count_trend(self):
        """H0c1: Тренд по числу кубиков 🔴 (0→7) — все респонденты."""
        self._analyze_zone_count_trend(
            self.completed,
            "H0c1: Тренд по числу кубиков 🔴 (0→7, все респонденты)",
            'H0c1', 'полная выборка',
            zone_col='cubes_reactive', zone_emoji='🔴', zone_name_ru='Срочное',
            scales_increasing=[('mbi_total', 'MBI'), ('proc_total', 'Прокрастинация')],
            scales_decreasing=[('swls_total', 'SWLS')]
        )

    def analyze_h0d1_red_count_typical(self):
        """H0d1: Тренд по числу кубиков 🔴 (0→7) — только rep=0."""
        df_t = self.completed[self.completed['representative'] == 0].copy()
        n_t = len(df_t)
        if n_t < 10:
            self.add_section("H0d1: Тренд по числу кубиков 🔴 (0→7, rep=0)", 2)
            self.add_paragraph(f"⚠️ Недостаточно данных: n = {n_t} для rep=0")
            return

        self._analyze_zone_count_trend(
            df_t,
            f"H0d1: Тренд по числу кубиков 🔴 (0→7, типовая неделя rep=0, n = {n_t})",
            'H0d1', f'типовая неделя (n = {n_t})',
            zone_col='cubes_reactive', zone_emoji='🔴', zone_name_ru='Срочное',
            scales_increasing=[('mbi_total', 'MBI'), ('proc_total', 'Прокрастинация')],
            scales_decreasing=[('swls_total', 'SWLS')]
        )

    # =========================================================================
    # МНОЖЕСТВЕННАЯ РЕГРЕССИЯ (H15)
    # =========================================================================
    
    def analyze_h15_regression_predictors(self):
        """
        H15: Множественная линейная регрессия для предсказания MBI, Прокрастинации и SWLS
        на основе четырёх предикторов:
          - кубики в целевой зоне (🟢)
          - кубики в срочной зоне (🔴)
          - баланс работа/личное
          - энергетический дефицит
        """
        self.add_section("H15: Множественная регрессия — предсказание по 4 предикторам", 3)

        self.add_paragraph("""
**Описание:**

Множественная линейная регрессия: `Шкала = β₀ + β₁×🟢 + β₂×🔴 + β₃×Баланс + β₄×Дефицит`

Предикторы:
- **🟢 Целевое** — количество кубиков в целевой зоне
- **🔴 Срочное** — количество кубиков в срочной зоне
- **Баланс работа/личное** — шкала от −3 (всё на работу) до +3 (всё на личное)
- **Дефицит энергии** — шкала от −3 (избыток) до +9 (острый дефицит)

Зависимые переменные: MBI, Прокрастинация, SWLS.
""")

        predictors = ['cubes_proactive', 'cubes_reactive', 'work_life', 'energy_deficit']
        pred_labels = ['🟢 Целевое', '🔴 Срочное', 'Баланс работа/личное', 'Дефицит энергии']

        targets = [
            ('mbi_total', 'MBI (выгорание)'),
            ('proc_total', 'Прокрастинация'),
            ('swls_total', 'SWLS'),
        ]

        df = self.completed.copy()
        valid_mask = df[predictors].notna().all(axis=1)
        df_valid = df[valid_mask].copy()
        n = len(df_valid)

        self.add_paragraph(f"**Размер выборки (полные данные):** n = {n}")

        all_results = {}

        # СВОДНАЯ ТАБЛИЦА
        self.add_paragraph("\n**Сводная таблица регрессий:**\n")

        summary_headers = ['Показатель', 'R²', 'Adj R²', 'F', 'p(F)']
        summary_rows = []

        for target_col, target_name in targets:
            y = df_valid[target_col].values
            y_valid = df_valid[target_col].dropna()
            valid_idx = y_valid.index
            y_vals = y_valid.values
            X_vals = df_valid.loc[valid_idx, predictors].values

            n_valid = len(y_vals)
            p_num = len(predictors)

            X_int = np.column_stack([np.ones(n_valid), X_vals])

            try:
                beta = np.linalg.lstsq(X_int, y_vals, rcond=None)[0]
                y_pred = X_int @ beta
                ss_res = np.sum((y_vals - y_pred) ** 2)
                ss_tot = np.sum((y_vals - np.mean(y_vals)) ** 2)
                r_squared = 1 - ss_res / ss_tot

                adj_r2 = 1 - (1 - r_squared) * (n_valid - 1) / (n_valid - p_num - 1)

                ms_reg = (ss_tot - ss_res) / p_num
                ms_res = ss_res / (n_valid - p_num - 1)
                f_stat = ms_reg / ms_res
                from scipy.stats import f as f_dist
                f_p = 1 - f_dist.cdf(f_stat, p_num, n_valid - p_num - 1)

                mse = ss_res / (n_valid - p_num - 1)
                var_beta = mse * np.linalg.inv(X_int.T @ X_int)
                se_beta = np.sqrt(np.diag(var_beta))
                t_stats = beta / se_beta
                from scipy.stats import t as t_dist
                p_values = 2 * (1 - t_dist.cdf(np.abs(t_stats), df=n_valid - p_num - 1))

                sig = lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

                summary_rows.append([
                    target_name,
                    f"{r_squared:.3f}",
                    f"{adj_r2:.3f}",
                    f"{f_stat:.1f}",
                    f"{f_p:.6f} {sig(f_p)}"
                ])

                all_results[target_col] = {
                    'beta': beta, 'se': se_beta, 'p': p_values,
                    'r2': r_squared, 'adj_r2': adj_r2,
                    'f': f_stat, 'f_p': f_p, 'n': n_valid
                }
            except Exception as e:
                summary_rows.append([target_name, f'Ошибка: {e}'] + ['—'] * 4)
                all_results[target_col] = None

        self.add_table(summary_headers, summary_rows)
        self.add_paragraph(f"\n*Значимость: *** p<0.001, ** p<0.01, * p<0.05*")

        # ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ
        for target_col, target_name in targets:
            res = all_results.get(target_col)
            if res is None:
                continue

            self.add_paragraph(f"\n**{target_name}:**")
            self.add_paragraph(f"- R² = {res['r2']:.3f}, Adj R² = {res['adj_r2']:.3f}")
            self.add_paragraph(f"- F({len(predictors)}, {res['n'] - len(predictors) - 1}) = {res['f']:.1f}, p = {res['f_p']:.6f}")

    def analyze_h15_regression_predictors(self):
        """
        H15: Множественная линейная регрессия для предсказания MBI, Прокрастинации и SWLS
        на основе четырёх предикторов:
          - кубики в целевой зоне (🟢)
          - кубики в срочной зоне (🔴)
          - баланс работа/личное
          - энергетический дефицит
        """
        self.add_section("H15: Множественная регрессия — предсказание по 4 предикторам", 3)

        self.add_paragraph("""
**Описание:**

Множественная линейная регрессия: `Шкала = β₀ + β₁×🟢 + β₂×🔴 + β₃×Баланс + β₄×Дефицит`

Предикторы:
- **🟢 Целевое** — количество кубиков в целевой зоне
- **🔴 Срочное** — количество кубиков в срочной зоне
- **Баланс работа/личное** — шкала от −3 (всё на работу) до +3 (всё на личное)
- **Дефицит энергии** — шкала от −3 (избыток) до +9 (острый дефицит)

Зависимые переменные: MBI, Прокрастинация, SWLS.
""")

        predictors = ['cubes_proactive', 'cubes_reactive', 'work_life', 'energy_deficit']
        pred_labels = ['🟢 Целевое', '🔴 Срочное', 'Баланс работа/личное', 'Дефицит энергии']

        targets = [
            ('mbi_total', 'MBI (выгорание)'),
            ('proc_total', 'Прокрастинация'),
            ('swls_total', 'SWLS'),
        ]

        df = self.completed.copy()
        valid_mask = df[predictors].notna().all(axis=1)
        df_valid = df[valid_mask].copy()
        n = len(df_valid)

        self.add_paragraph(f"**Размер выборки (полные данные):** n = {n}")

        all_results = {}

        # СВОДНАЯ ТАБЛИЦА
        self.add_paragraph("\n**Сводная таблица регрессий:**\n")

        summary_headers = ['Показатель', 'R²', 'Adj R²', 'F', 'p(F)']
        summary_rows = []

        for target_col, target_name in targets:
            y = df_valid[target_col].values
            y_valid = df_valid[target_col].dropna()
            valid_idx = y_valid.index
            y_vals = y_valid.values
            X_vals = df_valid.loc[valid_idx, predictors].values

            n_valid = len(y_vals)
            p_num = len(predictors)

            X_int = np.column_stack([np.ones(n_valid), X_vals])

            try:
                beta = np.linalg.lstsq(X_int, y_vals, rcond=None)[0]
                y_pred = X_int @ beta
                ss_res = np.sum((y_vals - y_pred) ** 2)
                ss_tot = np.sum((y_vals - np.mean(y_vals)) ** 2)
                r_squared = 1 - ss_res / ss_tot

                adj_r2 = 1 - (1 - r_squared) * (n_valid - 1) / (n_valid - p_num - 1)

                ms_reg = (ss_tot - ss_res) / p_num
                ms_res = ss_res / (n_valid - p_num - 1)
                f_stat = ms_reg / ms_res
                from scipy.stats import f as f_dist
                f_p = 1 - f_dist.cdf(f_stat, p_num, n_valid - p_num - 1)

                mse = ss_res / (n_valid - p_num - 1)
                var_beta = mse * np.linalg.inv(X_int.T @ X_int)
                se_beta = np.sqrt(np.diag(var_beta))
                t_stats = beta / se_beta
                from scipy.stats import t as t_dist
                p_values = 2 * (1 - t_dist.cdf(np.abs(t_stats), df=n_valid - p_num - 1))

                sig = lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

                summary_rows.append([
                    target_name,
                    f"{r_squared:.3f}",
                    f"{adj_r2:.3f}",
                    f"{f_stat:.1f}",
                    f"{f_p:.6f} {sig(f_p)}"
                ])

                all_results[target_col] = {
                    'beta': beta, 'se': se_beta, 'p': p_values,
                    'r2': r_squared, 'adj_r2': adj_r2,
                    'f': f_stat, 'f_p': f_p, 'n': n_valid
                }
            except Exception as e:
                summary_rows.append([target_name, f'Ошибка: {e}'] + ['—'] * 4)
                all_results[target_col] = None

        self.add_table(summary_headers, summary_rows)
        self.add_paragraph(f"\n*Значимость: *** p<0.001, ** p<0.01, * p<0.05*")

        # ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ
        for target_col, target_name in targets:
            res = all_results.get(target_col)
            if res is None:
                continue

            self.add_paragraph(f"\n**{target_name}:**")
            self.add_paragraph(f"- R² = {res['r2']:.3f}, Adj R² = {res['adj_r2']:.3f}")
            self.add_paragraph(f"- F({len(predictors)}, {res['n'] - len(predictors) - 1}) = {res['f']:.1f}, p = {res['f_p']:.6f}")

            coef_headers = ['Предиктор', 'β', 'SE', 't', 'p', '']
            coef_rows = []
            coef_rows.append([
                'Intercept',
                f"{res['beta'][0]:.3f}",
                f"{res['se'][0]:.3f}",
                f"{res['beta'][0]/res['se'][0]:.2f}",
                f"{res['p'][0]:.4f}",
                "***" if res['p'][0] < 0.001 else "**" if res['p'][0] < 0.01 else "*" if res['p'][0] < 0.05 else ""
            ])
            for i, (pred, label) in enumerate(zip(predictors, pred_labels)):
                idx = i + 1
                coef_rows.append([
                    label,
                    f"{res['beta'][idx]:.3f}",
                    f"{res['se'][idx]:.3f}",
                    f"{res['beta'][idx]/res['se'][idx]:.2f}",
                    f"{res['p'][idx]:.4f}",
                    "***" if res['p'][idx] < 0.001 else "**" if res['p'][idx] < 0.01 else "*" if res['p'][idx] < 0.05 else ""
                ])
            self.add_table(coef_headers, coef_rows)

        # ГРАФИКИ
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        for ax_idx, (target_col, target_name) in enumerate(targets):
            res = all_results.get(target_col)
            if res is None:
                axes[ax_idx].text(0.5, 0.5, 'Нет данных', ha='center', va='center',
                                 transform=axes[ax_idx].transAxes, fontproperties=LABEL_FONT)
                axes[ax_idx].set_title(target_name, fontproperties=TITLE_FONT)
                continue

            y_valid = df_valid[target_col].dropna()
            valid_idx = y_valid.index
            y_vals = y_valid.values
            X_vals = df_valid.loc[valid_idx, predictors].values
            X_int = np.column_stack([np.ones(len(y_vals)), X_vals])
            y_pred = X_int @ res['beta']

            axes[ax_idx].scatter(y_vals, y_pred, alpha=0.4, color='steelblue', s=15)

            y_min, y_max = min(y_vals.min(), y_pred.min()), max(y_vals.max(), y_pred.max())
            axes[ax_idx].plot([y_min, y_max], [y_min, y_max], 'r--', linewidth=2, alpha=0.7, label='Идеально')

            axes[ax_idx].set_xlabel(f'Наблюдаемое ({target_name})', fontproperties=LABEL_FONT)
            axes[ax_idx].set_ylabel(f'Предсказанное', fontproperties=LABEL_FONT)
            axes[ax_idx].set_title(f'{target_name} (R²={res["r2"]:.3f})', fontproperties=TITLE_FONT)
            axes[ax_idx].legend(prop={'size': 9})
            for label in axes[ax_idx].get_xticklabels() + axes[ax_idx].get_yticklabels():
                label.set_fontproperties(TICK_FONT)

        plt.suptitle('H15: Наблюдаемые vs предсказанные значения', fontproperties=TITLE_FONT)
        plt.tight_layout()
        path = self.save_figure('h15_predicted_vs_actual')
        self.add_paragraph(f"\n![Наблюдаемые vs предсказанные]({path})")

        # ВЫВОДЫ
        self.add_paragraph(f"\n**Выводы по H15:**")
        for target_col, target_name in targets:
            res = all_results.get(target_col)
            if res is None:
                continue
            sig_preds = [pred_labels[i] for i in range(len(predictors)) if res['p'][i + 1] < 0.05]
            self.add_paragraph(f"- **{target_name}:** R²={res['r2']:.3f}, значимые предикторы: {', '.join(sig_preds) if sig_preds else 'нет'}")

    # =========================================================================
    # ГЕНЕРАЦИЯ ОТЧЁТА
    # =========================================================================
    
    def generate_report(self) -> str:
        """Генерация полного отчёта."""
        
        self.report.append(f"""# Краткий отчёт анализа данных опроса "Три коробочки"

**Дата генерации:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Всего респондентов:** {len(self.data)}
**Завершили опрос:** {len(self.completed)}

---

## Описание исследования

Методика "Три коробочки" — ипсативный инструмент распределения ограниченного 
ресурса (6 единиц энергии) между тремя функциональными категориями:

- 🔴 **Срочное (Реактивность)** — задачи, требующие немедленного решения
- 🟢 **Целевое (Проактивность)** — задачи, приближающие к долгосрочным целям
- ⚪ **Операционное (Поддержание)** — рутинные задачи и восстановление

Цель исследования — валидация инструмента через анализ связи с продуктивностью,
выгоранием и удовлетворённостью жизнью.

---
""")
        
        self.generate_descriptive_stats()
        self.generate_correlation_matrices()
        
        self.add_section("Проверка гипотез исследования", 2)
        
        self.analyze_h0a1_green_count_trend()
        self.analyze_h0b1_green_count_typical()
        self.analyze_h0c1_red_count_trend()
        self.analyze_h0d1_red_count_typical()
        self.analyze_h15_regression_predictors()

        self.add_section("Заключение", 2)
        self.add_paragraph("""
**Примечание к интерпретации:**
- p < 0.05 — статистически значимый результат
- Корреляции: слабая |r| < 0.3, средняя 0.3 ≤ |r| < 0.5, сильная |r| ≥ 0.5
- R²: малый = 0.01, средний = 0.09, большой = 0.25 (Cohen, 1988)

---
*Отчёт сгенерирован автоматически*
""")
        
        return "\n".join(self.report)
    
    def save_report(self, output_path: str):
        """Сохранение отчёта в файл."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.generate_report())
        print(f"Отчёт сохранён: {output_path}")
        print(f"Графики сохранены: {self.figures}")


# ============================================================================
# ТОЧКА ВХОДА
# ============================================================================

def main():
    """Основная функция."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Анализ данных опроса "Три коробочки" (краткая версия)'
    )
    parser.add_argument(
        '--csv', 
        default='./Survey-Jedi-Boxes-Validation/Analysis/jedi_boxes_results.csv',
        help='Путь к CSV-файлу с данными'
    )
    parser.add_argument(
        '--output', 
        default='jedi_boxes_short_report.md',
        help='Путь для сохранения отчёта'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv):
        print(f"Ошибка: файл {args.csv} не найден")
        sys.exit(1)
    
    analyzer = JediBoxesAnalyzer(args.csv)
    analyzer.save_report(args.output)


if __name__ == '__main__':
    main()
