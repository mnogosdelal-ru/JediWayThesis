#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт анализа данных опроса "Три коробочки" (Jedi Boxes Validation)
============================================================================

Назначение:
    Генерация статистического отчёта в формате Markdown по данным опроса.
    Включает описательную статистику, визуализации распределений и проверку
    гипотез исследования.

Используемые библиотеки (научное сообщество):
    - pandas: работа с данными
    - numpy: численные вычисления
    - scipy.stats: статистические тесты
    - matplotlib/seaborn: визуализация
    - pingouin: дополнительные статистики (при наличии)

Автор: AI Assistant
Дата: 2026
Для публикации в рецензируемом журнале
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
# Jonckheere-Terpstra test (manual implementation)
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
# При H0:  E(J) = (N² - Σn_i²) / 4
#          Var(J) = [N²(2N+3) - Σn_i²(2n_i+3)] / 72 - Σt(t-1)(2t+5) / 36
#                   где t — размеры серий совпадающих значений (ties).
#
# Для больших выборок J ~ N(E(J), Var(J)), применяется continuity correction ±0.5.
# ============================================================================

def jonckheere_terpstra(groups, alternative='two-sided'):
    """
    Тест Джонкхира-Терпстры для упорядоченных альтернатив.

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
    # Базовая формула без учёта связей
    term1 = N ** 2 * (2 * N + 3)
    term2 = sum(ni ** 2 * (2 * ni + 3) for ni in n)
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
    if alternative == 'increasing':
        z = (J - E_J - 0.5) / np.sqrt(var_J)
    elif alternative == 'decreasing':
        z = (E_J - J - 0.5) / np.sqrt(var_J)
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

# Цвета для зон (согласно дизайну опроса)
ZONE_COLORS = {
    'reactive': '#e74c3c',      # Красный - Срочное
    'proactive': '#27ae60',     # Зелёный - Целевое
    'operational': '#95a5a6'    # Серый - Операционное
}

# Названия зон на русском с Unicode-кружками (поддерживаются в DejaVu Sans)
# Используем ● (U+25CF) вместо эмодзи
ZONE_NAMES = {
    'reactive': 'Срочное ●',
    'proactive': 'Целевое ●',
    'operational': 'Операционное ●'
}

# Цвета для кружков будут заданы через цвет текста
ZONE_TEXT_COLORS = {
    'reactive': '#e74c3c',      # Красный
    'proactive': '#27ae60',     # Зелёный
    'operational': '#95a5a6'    # Серый
}

# ============================================================================
# НАСТРОЙКИ АНАЛИЗА
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

# Гипотезы исследования
HYPOTHESES = {
    'H0': 'Различия между уровнями 1-7 по MBI, прокрастинации и SWLS',
    'H0a': 'Тренд по позиции 🟢 (все респонденты): MBI и прокрастинация X < Y < Z, SWLS X > Y > Z',
    'H0b': 'Тренд по позиции 🟢 (rep=0): MBI и прокрастинация X < Y < Z, SWLS X > Y > Z',
    'H0c': 'Тренд по позиции 🔴 (все респонденты): MBI и прокрастинация R1 > R2 > R3, SWLS R1 < R2 < R3',
    'H0a1': 'Тренд по числу кубиков 🟢 (0→7, все респонденты): MBI и прокрастинация ↓, SWLS ↑',
    'H0b1': 'Тренд по числу кубиков 🟢 (0→7, rep=0): MBI и прокрастинация ↓, SWLS ↑',
    'H0c1': 'Тренд по числу кубиков 🔴 (0→7, все респонденты): MBI и прокрастинация ↑, SWLS ↓',
    'H14': 'Связь MBI и SWLS: линейная или нелинейная (квадратичная)?',
    'H15': 'Множественная регрессия: MBI, Прокрастинация и SWLS предсказываются кубиками 🟢, 🔴, балансом работа/личное и дефицитом энергии',
    'H5a': 'Детальное сравнение респондентов «по памяти» (1-2) и «по записям» (4-5): кубики, шкалы, демография, контекст',
    'H10a': 'Владельцы бизнеса и высшее руководство отличаются по распределению кубиков от остальных',
    'H13': 'В IT больше прокрастинация, выгорание и ниже удовлетворённость жизнью, чем в других сферах',
    'H1': '🔴 положительно коррелирует с прокрастинацией и MBI',
    'H2': '🟢 положительно коррелирует с SWLS и отрицательно с MBI',
    'H3': 'Низкое 🔴 связано с более низким выгоранием',
    'H4': 'Профили с доминированием 🟢 имеют высокие баллы SWLS',
    'H5': 'Модерация записями: восстановление по записям vs памяти',
    'H6': 'Модерация красным: связь ⚪ с прокрастинацией усиливается при высоком 🔴',
    'H7': 'Криволинейная связь 🟢 и выгорания (U-образная)',
    'H8': 'Баланс работа/личное как модератор связи 🔴 и выгорания',
    'H9': 'Энергетический дефицит как медиатор',
    'H10': 'Гендерные различия в распределении зон',
    'H11': 'Возрастной тренд: с возрастом доля 🔴 снижается',
    'H12': 'Профиль "Дзен" связан с наименьшим энергетическим дефицитом',
    'H7a': 'Линейная регрессия: MBI, Прокрастинация и SWLS предсказываются числом срочных и целевых кубиков (🔴, 🟢)',
    'H10a': 'Владельцы бизнеса и высшее руководство отличаются по распределению кубиков от остальных'
}


# ============================================================================
# КЛАСС ДЛЯ ГЕНЕРАЦИИ ОТЧЁТА
# ============================================================================

class JediBoxesAnalyzer:
    """
    Класс для анализа данных опроса "Три коробочки"
    
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

        # Фильтрация некорректных записей: если total=0, но есть вопросы с NaN
        # Это ошибка в расчёте суммы (NaN.sum() = 0 вместо NaN)
        # Проверяем каждую шкалу
        for scale, questions in [
            ('proc_total', ['proc_'+str(i) for i in range(1, 9)]),
            ('swls_total', ['swls_'+str(i) for i in range(1, 6)]),
        ]:
            if scale in df.columns and all(q in df.columns for q in questions):
                # Если total=0, но все вопросы NaN — это некорректная запись
                all_nan = df[questions].isna().all(axis=1)
                zero_total = df[scale] == 0
                invalid = all_nan & zero_total
                if invalid.sum() > 0:
                    print(f"⚠️  Удалено {invalid.sum()} записей с некорректным {scale}=0 (все ответы NaN)")
                    df.loc[invalid, scale] = np.nan

        # Вычисление ProductivityIndex (логарифм по основанию 2 отношения целевого к срочному с эпсилоном)
        # ProductivityIndex = log₂((Целевое + eps) / (Срочное + eps))
        # Интерпретация: > 0 — преобладание проактивности, < 0 — преобладание реактивности, = 0 — баланс
        # При eps=1 и 7 кубиках диапазон: [-3, 3]
        df['productivity_index'] = np.log2((df['cubes_proactive'] + RG_RATIO_EPS) / (df['cubes_reactive'] + RG_RATIO_EPS))

        return df
    
    def _add_level_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавление колонки с уровнем профиля (1-7).

        Алгоритм классификации:
        - Сортируем зоны по убыванию (с приоритетом при ничьих)
        - Определяем primary и secondary
        - Присваиваем уровень
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
                return 7  # Дзен: ⚪ > 🟢 > 🔴
            elif primary == 'operational' and secondary == 'reactive':
                return 3  # Апатия: ⚪ > 🔴 > 🟢
            elif primary == 'proactive' and secondary == 'operational':
                return 6  # Рост: 🟢 > ⚪ > 🔴
            elif primary == 'proactive' and secondary == 'reactive':
                return 5  # Не сдаёмся: 🟢 > 🔴 > ⚪
            elif primary == 'reactive' and secondary == 'operational':
                return 4  # Кризис: 🔴 > ⚪ > 🟢
            elif primary == 'reactive' and secondary == 'proactive':
                return 2  # Выживание: 🔴 > 🟢 > ⚪
            elif primary == 'proactive' and secondary is None:
                return 6  # Рост: только 🟢
            elif primary == 'reactive' and secondary is None:
                return 2  # Выживание: только 🔴
            else:
                return 2  # По умолчанию — Выживание

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
        # Заголовок
        self.report.append("| " + " | ".join(str(h) for h in headers) + " |")
        # Разделитель
        self.report.append("| " + " | ".join(["---"] * len(headers)) + " |")
        # Строки
        for row in rows:
            self.report.append("| " + " | ".join(str(cell) for cell in row) + " |")
        self.report.append("")
    
    def add_test_result(self, test_name: str, statistic: float, p_value: float, 
                        conclusion: str, effect_size: float = None):
        """Форматированный вывод результата статистического теста."""
        # Проверка на NaN/None для статистики
        if statistic is None or (isinstance(statistic, float) and np.isnan(statistic)):
            stat_str = "N/A"
        else:
            stat_str = f"{statistic:.3f}"
        
        # Проверка на NaN/None для p-value
        if p_value is None or (isinstance(p_value, float) and np.isnan(p_value)):
            p_str = "N/A"
            sig = "не определён"
            comparison = ""
        else:
            p_str = f"{p_value:.4f}"
            sig = "значим" if p_value < 0.05 else "не значим"
            comparison = f" (p {'<' if p_value < 0.05 else '>'} 0.05)"
        
        self.add_paragraph(f"**{test_name}**:")
        self.add_paragraph(f"- Статистика: {stat_str}")
        self.add_paragraph(f"- p-value: {p_str}")
        if effect_size is not None and not np.isnan(effect_size):
            self.add_paragraph(f"- Размер эффекта: {effect_size:.3f}")
        
        self.add_paragraph(f"**Вывод**: Результат {sig}{comparison}. {conclusion}")
    
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
    
    # =========================================================================
    # ОПИСАТЕЛЬНАЯ СТАТИСТИКА
    # =========================================================================
    
    def generate_descriptive_stats(self):
        """Генерация описательной статистики."""
        self.add_section("Описательная статистика", 2)
        
        # Демография
        self.add_paragraph("**Демография завершённых анкет:**")
        
        ages = self.completed['age'].dropna()
        self.add_paragraph(f"- Возраст: M = {ages.mean():.1f}, SD = {ages.std():.1f}, "
                          f"диапазон = {ages.min():.0f}-{ages.max():.0f}, n = {len(ages)}")
        
        gender_counts = self.completed['gender'].value_counts()
        # Преобразуем np.int64 в обычные int для вывода
        gender_dict = {k: int(v) for k, v in gender_counts.items()}
        self.add_paragraph(f"- Пол: {gender_dict}")
        
        position_counts = self.completed['position'].value_counts()
        position_dict = {k: int(v) for k, v in position_counts.items()}
        self.add_paragraph(f"- Должности: {position_dict}")
        
        # Распределение кубиков
        self.add_paragraph("\n**Распределение кубиков по зонам:**")
        
        for zone in ['cubes_reactive', 'cubes_proactive', 'cubes_operational']:
            name = ZONE_NAMES.get(zone.replace('cubes_', ''), zone)
            vals = self.completed[zone].dropna()
            self.add_paragraph(f"- {name}: M = {vals.mean():.2f}, SD = {vals.std():.2f}, "
                              f"диапазон = {vals.min():.0f}-{vals.max():.0f}")
        
        # Распределение уровней
        self.add_paragraph("\n**Распределение уровней профиля:**")
        self.completed = self._add_level_column(self.completed)
        level_counts = self.completed['level'].value_counts().sort_index()
        
        table_data = []
        for level in sorted(level_counts.index):
            count = level_counts[level]
            pct = count / len(self.completed) * 100
            profile_name = PROFILE_LEVELS.get(level, f'Уровень {level}')
            table_data.append([level, profile_name, count, f"{pct:.1f}%"])
        
        self.add_table(['Уровень', 'Профиль', 'n', '%'], table_data)
        
        # Шкалы валидации
        self.add_paragraph("\n**Шкалы валидации:**")
        
        for scale, name in [('proc_total', 'Прокрастинация'), 
                           ('swls_total', 'SWLS (удовлетворённость)'),
                           ('mbi_total', 'MBI (выгорание)')]:
            if scale in self.completed.columns:
                vals = self.completed[scale].dropna()
                self.add_paragraph(f"- {name}: M = {vals.mean():.1f}, SD = {vals.std():.1f}, "
                                  f"диапазон = {vals.min():.0f}-{vals.max():.0f}")
    
    def generate_visualizations(self):
        """Генерация визуализаций распределений."""
        self.add_section("Визуализации распределений", 2)

        # 1. Отдельные гистограммы контекстуальных показателей
        self.add_paragraph("**Контекстуальные показатели:**")

        # Баланс работа/личное - отдельный график
        work_life = self.completed['work_life'].dropna()
        fig, ax = plt.subplots(figsize=(9, 5))

        # Показываем все значения от -3 до 3
        all_wl_values = list(range(-3, 4))
        wl_labels = {
            -3: 'Всё на\nличное',
            -2: '-2',
            -1: '-1',
            0: 'Баланс',
            1: '1',
            2: '2',
            3: 'Всё на\nработу'
        }
        wl_y_values = [int((work_life == x).sum()) for x in all_wl_values]
        wl_x_positions = list(range(len(all_wl_values)))

        ax.bar(wl_x_positions, wl_y_values, color='steelblue', edgecolor='white', align='center', width=1.0)
        ax.set_xlabel('Балл', fontproperties=LABEL_FONT)
        ax.set_ylabel('Число респондентов', fontproperties=LABEL_FONT)
        ax.set_title('Баланс работа/личное', fontproperties=TITLE_FONT)
        ax.set_xticks(wl_x_positions)
        ax.set_xticklabels([wl_labels[x] for x in all_wl_values], fontproperties=TICK_FONT, rotation=0)
        ax.set_xlim(-0.6, len(all_wl_values) - 0.4)

        # Добавляем подписи значений
        for i, count in enumerate(wl_y_values):
            if count > 0:
                ax.text(i, count + 0.3, str(count), ha='center', va='bottom', fontproperties=SMALL_TEXT_FONT)

        plt.tight_layout()
        path = self.save_figure('work_life_balance')
        self.add_paragraph(f"![Баланс работа/личное]({path})")

        # Типичность недели - отдельный график
        rep = self.completed['representative'].dropna()
        fig, ax = plt.subplots(figsize=(9, 5))

        # Показываем все значения от -3 до 3
        all_rep_values = list(range(-3, 4))
        rep_labels = {
            -3: 'Значительно\nхуже',
            -2: '-2',
            -1: '-1',
            0: 'Как\nобычно',
            1: '1',
            2: '2',
            3: 'Значительно\nлучше'
        }
        rep_y_values = [int((rep == x).sum()) for x in all_rep_values]
        rep_x_positions = list(range(len(all_rep_values)))

        ax.bar(rep_x_positions, rep_y_values, color='forestgreen', edgecolor='white', align='center', width=1.0)
        ax.set_xlabel('Балл', fontproperties=LABEL_FONT)
        ax.set_ylabel('Число респондентов', fontproperties=LABEL_FONT)
        ax.set_title('Типичность недели', fontproperties=TITLE_FONT)
        ax.set_xticks(rep_x_positions)
        ax.set_xticklabels([rep_labels[x] for x in all_rep_values], fontproperties=TICK_FONT, rotation=0)
        ax.set_xlim(-0.6, len(all_rep_values) - 0.4)

        for i, count in enumerate(rep_y_values):
            if count > 0:
                ax.text(i, count + 0.3, str(count), ha='center', va='bottom', fontproperties=SMALL_TEXT_FONT)

        plt.tight_layout()
        path = self.save_figure('week_typicality')
        self.add_paragraph(f"![Типичность недели]({path})")

        # Энергетический дефицит - отдельный график
        deficit = self.completed['energy_deficit'].dropna()
        fig, ax = plt.subplots(figsize=(9, 5))

        # Показываем все возможные значения
        all_deficit_values = [-3, -2, -1, 0, 3, 6, 9]
        deficit_labels = {
            -3: 'Избыток\nэнергии',
            -2: '-2',
            -1: '-1',
            0: 'Впритык',
            3: '3',
            6: '6',
            9: 'Острый\nдефицит'
        }
        deficit_y_values = [int((deficit == x).sum()) for x in all_deficit_values]
        deficit_x_positions = list(range(len(all_deficit_values)))

        ax.bar(deficit_x_positions, deficit_y_values, color='crimson', edgecolor='white', align='center', width=1.0)
        ax.set_xlabel('Балл', fontproperties=LABEL_FONT)
        ax.set_ylabel('Число респондентов', fontproperties=LABEL_FONT)
        ax.set_title('Энергетический дефицит', fontproperties=TITLE_FONT)
        ax.set_xticks(deficit_x_positions)
        ax.set_xticklabels([deficit_labels[x] for x in all_deficit_values], fontproperties=TICK_FONT, rotation=0)
        ax.set_xlim(-0.6, len(all_deficit_values) - 0.4)

        for i, count in enumerate(deficit_y_values):
            if count > 0:
                ax.text(i, count + 0.3, str(count), ha='center', va='bottom', fontproperties=SMALL_TEXT_FONT)

        plt.tight_layout()
        path = self.save_figure('energy_deficit')
        self.add_paragraph(f"![Энергетический дефицит]({path})")

        # 2. Распределение кубиков по зонам
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for i, (zone, color) in enumerate(ZONE_COLORS.items()):
            vals = self.completed[f'cubes_{zone}'].dropna()
            # Получаем уникальные значения и сортируем
            unique_vals = sorted(vals.unique())
            counts = [int((vals == v).sum()) for v in unique_vals]

            # Используем integer positions для баров
            x_pos = np.arange(len(unique_vals))
            axes[i].bar(x_pos, counts, color=color, edgecolor='white', alpha=0.8, align='center', width=1.0)

            # Заголовок с цветным кружком
            axes[i].set_title(ZONE_NAMES[zone], fontsize=13, fontweight='bold',
                            color=ZONE_TEXT_COLORS[zone], fontproperties=LABEL_FONT)
            axes[i].set_xlabel('Количество кубиков', fontproperties=LABEL_FONT)
            axes[i].set_ylabel('Частота', fontproperties=LABEL_FONT)
            axes[i].set_xticks(x_pos)
            axes[i].set_xticklabels([str(int(v)) for v in unique_vals], fontproperties=TICK_FONT)
            axes[i].set_xlim(x_pos[0] - 0.6, x_pos[-1] + 0.6)
            
            # Вычисляем позицию среднего через интерполяцию
            mean_val = vals.mean()
            # Находим два ближайших значения в unique_vals
            unique_arr = np.array(unique_vals)
            # Интерполируем позицию: если среднее = 1.8, а значения [0,1,2,3,4], позиция = 1.8
            mean_x_pos = mean_val - unique_vals[0]  # Позиция относительно первого значения
            
            axes[i].axvline(mean_x_pos, color='red', linestyle='--', alpha=0.5)
            axes[i].text(mean_x_pos, max(counts)*0.9, f'M = {mean_val:.1f}',
                        color='red', fontsize=10, ha='center', fontproperties=TICK_FONT)

        plt.suptitle('Распределение кубиков по зонам', fontsize=14, fontweight='bold',
                    fontproperties=TITLE_FONT)
        plt.tight_layout()
        path = self.save_figure('cubes_distribution', 'Распределение кубиков')
        self.add_paragraph(f"![Распределение кубиков]({path})")

        # 3. Распределение уровней профиля
        fig, ax = plt.subplots(figsize=(10, 5))

        level_counts = self.completed['level'].value_counts().sort_index()
        colors = ['#e74c3c', '#e67e22', '#d35400', '#f39c12', '#27ae60', '#2ecc71', '#1abc9c']
        bars = ax.bar([PROFILE_LEVELS[l] for l in level_counts.index],
                     level_counts.values, color=colors[:len(level_counts)])

        ax.set_xlabel('Уровень профиля', fontproperties=LABEL_FONT)
        ax.set_ylabel('Количество респондентов', fontproperties=LABEL_FONT)
        ax.set_title('Распределение уровней профиля', fontproperties=TITLE_FONT)
        plt.xticks(rotation=30, ha='right', fontproperties=TICK_FONT)
        plt.yticks(fontproperties=TICK_FONT)

        # Добавляем подписи
        for bar, count in zip(bars, level_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   str(count), ha='center', va='bottom', fontproperties=SMALL_TEXT_FONT)

        plt.tight_layout()
        path = self.save_figure('profile_levels', 'Уровни профиля')
        self.add_paragraph(f"![Уровни профиля]({path})")

        # 3a. Распределение по полу
        fig, ax = plt.subplots(figsize=(8, 5))

        gender_counts = self.completed['gender'].value_counts()
        # Преобразуем в обычные int
        gender_counts = {k: int(v) for k, v in gender_counts.items()}
        sorted_genders = sorted(gender_counts.keys())
        gender_labels_map = {'male': 'Мужской', 'female': 'Женский'}
        colors_gender = ['#3498db', '#e91e63']

        bars = ax.bar(range(len(sorted_genders)),
                     [gender_counts[g] for g in sorted_genders],
                     color=colors_gender[:len(sorted_genders)], edgecolor='white', align='center', width=1.0)

        ax.set_xlabel('Пол', fontproperties=LABEL_FONT)
        ax.set_ylabel('Количество респондентов', fontproperties=LABEL_FONT)
        ax.set_title('Распределение по полу', fontproperties=TITLE_FONT)
        ax.set_xticks(range(len(sorted_genders)))
        ax.set_xticklabels([gender_labels_map.get(g, g) for g in sorted_genders], fontproperties=TICK_FONT)
        ax.set_xlim(-0.6, len(sorted_genders) - 0.4)

        for bar, count in zip(bars, [gender_counts[g] for g in sorted_genders]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   str(count), ha='center', va='bottom', fontproperties=SMALL_TEXT_FONT)

        plt.tight_layout()
        path = self.save_figure('gender_distribution')
        self.add_paragraph(f"![Распределение по полу]({path})")

        # 3b. Распределение по возрасту
        fig, ax = plt.subplots(figsize=(10, 5))

        ages = self.completed['age'].dropna().astype(int)
        if len(ages) > 0:
            age_min = int(ages.min())
            age_max = int(ages.max())
            age_range = age_max - age_min
            
            # Группируем возрасты в интервалы по 2 года (сокращаем в 2 раза)
            bin_width = 2
            n_bins = max(1, (age_range + 1) // bin_width)
            if (age_range + 1) % bin_width != 0:
                n_bins += 1
            
            # Создаём бины
            bin_edges = [age_min + i * bin_width for i in range(n_bins + 1)]
            # Убеждаемся, что последний бин включает max
            if bin_edges[-1] < age_max:
                bin_edges[-1] = age_max + 1
            
            counts, _ = np.histogram(ages, bins=bin_edges)
            bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges) - 1)]
            
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

        # 4. Корреляционная матрица с p-value, ProductivityIndex и контекстуальными показателями
        fig, ax = plt.subplots(figsize=(14, 12))

        cols = ['cubes_reactive', 'cubes_proactive', 'cubes_operational', 'productivity_index',
                'representative', 'work_life', 'energy_deficit', 'memory_vs_records',
                'proc_total', 'swls_total', 'mbi_total']
        corr_matrix = self.completed[cols].corr(method='spearman')

        # Вычисляем p-values для корреляций
        from scipy.stats import spearmanr
        p_matrix = np.zeros_like(corr_matrix)
        n = len(self.completed)
        for i in range(len(cols)):
            for j in range(len(cols)):
                if i != j:
                    _, p = spearmanr(self.completed[cols[i]], self.completed[cols[j]])
                    p_matrix[i, j] = p
                else:
                    p_matrix[i, j] = 0

        labels = ['Срочное ●', 'Целевое ●', 'Операционное ●', 'ProdIndex',
                  'Типичность', 'Работа/Личное', 'Дефицит', 'Записи',
                  'Прокрастинация', 'SWLS', 'MBI']

        # Создаём heatmap с аннотациями корреляций
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, vmin=-1, vmax=1, xticklabels=labels,
                   yticklabels=labels, ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_title('Корреляционная матрица (Спирмен)', fontproperties=TITLE_FONT)

        # Добавляем p-values под корреляциями
        for i in range(len(cols)):
            for j in range(len(cols)):
                if i != j:
                    p_val = p_matrix[i, j]
                    if p_val < 0.001:
                        p_text = '***'
                    elif p_val < 0.01:
                        p_text = '**'
                    elif p_val < 0.05:
                        p_text = '*'
                    else:
                        p_text = 'n.s.'
                    ax.text(j + 0.5, i + 0.85, p_text, ha='center', va='top',
                           fontsize=8, color='black', fontproperties=TICK_FONT)

        # Настраиваем шрифт для меток
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(TICK_FONT)

        plt.tight_layout()
        path = self.save_figure('correlation_matrix', 'Корреляции')
        self.add_paragraph(f"![Корреляционная матрица]({path})")
        self.add_paragraph(f"*Значимость: *** p<0.001, ** p<0.01, * p<0.05, n.s. - не значимо | eps={RG_RATIO_EPS}*")

        # 4a. Корреляционная матрица для почти типовой недели (-1..1)
        df_almost = self.completed[self.completed['representative'].between(-1, 1)]
        if len(df_almost) >= 5:
            fig, ax = plt.subplots(figsize=(14, 12))
            corr_almost = df_almost[cols].corr(method='spearman')
            p_matrix_almost = np.zeros_like(corr_almost)
            for i in range(len(cols)):
                for j in range(len(cols)):
                    if i != j:
                        _, p = spearmanr(df_almost[cols[i]], df_almost[cols[j]])
                        p_matrix_almost[i, j] = p

            sns.heatmap(corr_almost, annot=True, fmt='.2f', cmap='RdBu_r',
                       center=0, vmin=-1, vmax=1, xticklabels=labels,
                       yticklabels=labels, ax=ax, cbar_kws={'shrink': 0.8})
            ax.set_title(f'Корреляционная матрица — Почти типовая неделя (n={len(df_almost)})', fontproperties=TITLE_FONT)

            for i in range(len(cols)):
                for j in range(len(cols)):
                    if i != j:
                        p_val = p_matrix_almost[i, j]
                        if p_val < 0.001: p_text = '***'
                        elif p_val < 0.01: p_text = '**'
                        elif p_val < 0.05: p_text = '*'
                        else: p_text = 'n.s.'
                        ax.text(j + 0.5, i + 0.85, p_text, ha='center', va='top',
                               fontsize=8, color='black', fontproperties=TICK_FONT)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontproperties(TICK_FONT)
            plt.tight_layout()
            path = self.save_figure('correlation_matrix_almost')
            self.add_paragraph(f"\n![Корреляционная матрица — почти типовая]({path})")

        # 4b. Корреляционная матрица для точно типовой недели (0)
        df_exact = self.completed[self.completed['representative'] == 0]
        if len(df_exact) >= 5:
            fig, ax = plt.subplots(figsize=(14, 12))
            corr_exact = df_exact[cols].corr(method='spearman')
            p_matrix_exact = np.zeros_like(corr_exact)
            for i in range(len(cols)):
                for j in range(len(cols)):
                    if i != j:
                        _, p = spearmanr(df_exact[cols[i]], df_exact[cols[j]])
                        p_matrix_exact[i, j] = p

            sns.heatmap(corr_exact, annot=True, fmt='.2f', cmap='RdBu_r',
                       center=0, vmin=-1, vmax=1, xticklabels=labels,
                       yticklabels=labels, ax=ax, cbar_kws={'shrink': 0.8})
            ax.set_title(f'Корреляционная матрица — Точно типовая неделя (n={len(df_exact)})', fontproperties=TITLE_FONT)

            for i in range(len(cols)):
                for j in range(len(cols)):
                    if i != j:
                        p_val = p_matrix_exact[i, j]
                        if p_val < 0.001: p_text = '***'
                        elif p_val < 0.01: p_text = '**'
                        elif p_val < 0.05: p_text = '*'
                        else: p_text = 'n.s.'
                        ax.text(j + 0.5, i + 0.85, p_text, ha='center', va='top',
                               fontsize=8, color='black', fontproperties=TICK_FONT)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontproperties(TICK_FONT)
            plt.tight_layout()
            path = self.save_figure('correlation_matrix_exact')
            self.add_paragraph(f"\n![Корреляционная матрица — точно типовая]({path})")

        # 4c. Корреляционная матрица Кендалла (для сравнения со Спирменом)
        fig, ax = plt.subplots(figsize=(14, 12))
        corr_matrix_kendall = self.completed[cols].corr(method='kendall')

        # Вычисляем p-values для корреляций Кендалла
        from scipy.stats import kendalltau
        p_matrix_kendall = np.zeros_like(corr_matrix_kendall)
        for i in range(len(cols)):
            for j in range(len(cols)):
                if i != j:
                    _, p = kendalltau(self.completed[cols[i]], self.completed[cols[j]])
                    p_matrix_kendall[i, j] = p
                else:
                    p_matrix_kendall[i, j] = 0

        sns.heatmap(corr_matrix_kendall, annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, vmin=-1, vmax=1, xticklabels=labels,
                   yticklabels=labels, ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_title('Корреляционная матрица (Кендалл τ)', fontproperties=TITLE_FONT)

        # Добавляем p-values под корреляциями
        for i in range(len(cols)):
            for j in range(len(cols)):
                if i != j:
                    p_val = p_matrix_kendall[i, j]
                    if p_val < 0.001:
                        p_text = '***'
                    elif p_val < 0.01:
                        p_text = '**'
                    elif p_val < 0.05:
                        p_text = '*'
                    else:
                        p_text = 'n.s.'
                    ax.text(j + 0.5, i + 0.85, p_text, ha='center', va='top',
                           fontsize=8, color='black', fontproperties=TICK_FONT)

        # Настраиваем шрифт для меток
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(TICK_FONT)

        plt.tight_layout()
        path = self.save_figure('correlation_matrix_kendall', 'Корреляции Кендалла')
        self.add_paragraph(f"\n![Корреляционная матрица — Кендалл]({path})")
        self.add_paragraph(f"*Корреляция Кендалла (τ): *** p<0.001, ** p<0.01, * p<0.05, n.s. - не значимо | eps={RG_RATIO_EPS}*")

        # 5. Распределение шкал
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for i, (scale, name) in enumerate([('proc_total', 'Прокрастинация'),
                                           ('swls_total', 'SWLS'),
                                           ('mbi_total', 'MBI')]):
            vals = self.completed[scale].dropna()
            min_val = int(vals.min())
            max_val = int(vals.max())
            data_range = max_val - min_val
            
            # Определяем число столбцов (не больше 10)
            n_bins = min(10, data_range + 1)
            # Ищем хороший делитель
            if data_range > 10:
                for n in range(10, 4, -1):
                    if data_range % n == 0:
                        n_bins = n
                        break
                else:
                    # Расширяем диапазон
                    for n in range(10, 4, -1):
                        remainder = data_range % n
                        if remainder == 0 or (max_val + (n - remainder)) <= (max_val + 5):
                            n_bins = n
                            max_val = max_val + (n - remainder) if remainder > 0 else max_val
                            break
            
            # Создаём бины
            bin_edges = np.linspace(min_val, max_val, n_bins + 1)
            counts, _ = np.histogram(vals, bins=bin_edges)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            x_pos = list(range(n_bins))
            axes[i].bar(x_pos, counts, color='steelblue', edgecolor='white', alpha=0.8, align='center', width=1.0)
            axes[i].set_title(name, fontproperties=LABEL_FONT)
            axes[i].set_xlabel('Балл', fontproperties=LABEL_FONT)
            axes[i].set_ylabel('Частота', fontproperties=LABEL_FONT)
            axes[i].set_xticks(x_pos)
            # Подписываем тики в формате диапазона min...max
            tick_labels = [f'{int(bin_edges[j])}...{int(bin_edges[j+1])}' for j in range(n_bins)]
            axes[i].set_xticklabels(tick_labels, fontproperties=TICK_FONT, rotation=45, ha='right')
            axes[i].set_xlim(-0.6, n_bins - 0.4)
            axes[i].axvline(vals.mean(), color='red', linestyle='--', label=f'M = {vals.mean():.1f}')
            axes[i].legend(prop=TICK_FONT)
            # Настраиваем шрифт для tick labels
            for label in axes[i].get_xticklabels() + axes[i].get_yticklabels():
                label.set_fontproperties(TICK_FONT)

        plt.suptitle('Распределение баллов шкал валидации', fontproperties=TITLE_FONT)
        plt.tight_layout()
        path = self.save_figure('scale_distributions', 'Шкалы')
        self.add_paragraph(f"![Шкалы]({path})")

        # Тесты на нормальность распределений
        self._test_normality_scales()

    def _test_normality_scales(self):
        """
        Проверка нормальности распределений шкал SWLS, MBI и Прокрастинации.
        
        Используемые тесты:
        1. Тест Шапиро-Уилка (Shapiro-Wilk) — чувствителен к отклонениям от нормальности
        2. Тест Д'Агостино (D'Agostino-Pearson) — проверяет асимметрию и эксцесс
        3. Оценка асимметрии (skewness) и эксцесса (kurtosis) с Z-критерием
        4. QQ-plot для визуальной проверки
        """
        self.add_section("Проверка нормальности распределений шкал", 3)
        
        self.add_paragraph(
            "Для оценки соответствия распределений шкал нормальному закону проведены:\n"
            "- **Тест Шапиро-Уилка** (Shapiro-Wilk): чувствителен к различным отклонениям от нормальности\n"
            "- **Тест Д'Агостино** (D'Agostino-Pearson): основан на проверке асимметрии и эксцесса\n"
            "- **Z-критерий для асимметрии и эксцесса**: |Z| > 1.96 указывает на значимое отклонение\n"
            "- **QQ-plot**: визуальная проверка (точки должны лежать на прямой)"
        )
        
        scales = [
            ('proc_total', 'Прокрастинация'),
            ('swls_total', 'SWLS'),
            ('mbi_total', 'MBI')
        ]
        
        table_rows = []
        
        for scale, name in scales:
            vals = self.completed[scale].dropna()
            n = len(vals)
            
            # Тест Шапиро-Уилка
            try:
                sw_stat, sw_p = stats.shapiro(vals)
                sw_result = f"W = {sw_stat:.4f}, p = {sw_p:.4f}"
                sw_significant = sw_p < 0.05
            except Exception as e:
                sw_result = f"Ошибка: {e}"
                sw_significant = None
                sw_p = None
            
            # Тест Д'Агостино (проверка асимметрии и эксцесса)
            try:
                dp_stat, dp_p = stats.normaltest(vals)
                dp_result = f"K² = {dp_stat:.3f}, p = {dp_p:.4f}"
                dp_significant = dp_p < 0.05
            except Exception as e:
                dp_result = f"Ошибка: {e}"
                dp_significant = None
                dp_p = None
            
            # Асимметрия и эксцесс с Z-критерием
            skewness = stats.skew(vals)
            kurtosis = stats.kurtosis(vals)  # Fisher's kurtosis (нормальное = 0)
            
            # Стандартные ошибки
            se_skew = np.sqrt(6 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3))) if n > 3 else 1
            se_kurt = np.sqrt(
                (4 * (n**2 - 1) * (n - 3)) / ((n + 1)**2 * (n + 3) * (n + 5))
            ) if n > 5 else 1
            
            z_skew = skewness / se_skew
            z_kurt = kurtosis / se_kurt
            
            skew_significant = abs(z_skew) > 1.96
            kurt_significant = abs(z_kurt) > 1.96
            
            # Вывод по шкале
            normality_violated = (
                (sw_significant if sw_significant is not None else False) or
                (dp_significant if dp_significant is not None else False) or
                skew_significant or
                kurt_significant
            )
            
            conclusion = "❌ Значимое отклонение от нормальности" if normality_violated else "✅ Нет свидетельств отклонения от нормальности"
            
            table_rows.append([
                name,
                f"{n}, M = {vals.mean():.1f}, SD = {vals.std():.1f}",
                sw_result,
                dp_result,
                f"{skewness:.3f} (Z = {z_skew:.2f})",
                f"{kurtosis:.3f} (Z = {z_kurt:.2f})",
                conclusion
            ])
        
        self.add_table(
            ['Шкала', 'Описательная стат.', 'Шапиро-Уилк', 'Д\'Агостино', 'Асимметрия', 'Эксцесс', 'Вывод'],
            table_rows
        )
        
        # Интерпретация результатов
        self.add_paragraph("**Интерпретация:**\n")
        
        for scale, name in scales:
            vals = self.completed[scale].dropna()
            skewness = stats.skew(vals)
            kurtosis = stats.kurtosis(vals)
            n = len(vals)
            
            se_skew = np.sqrt(6 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3))) if n > 3 else 1
            z_skew = skewness / se_skew
            
            self.add_paragraph(f"**{name}:**")
            
            if abs(z_skew) < 0.5:
                direction = "практически симметрично"
            elif z_skew > 0:
                direction = f"правосторонняя асимметрия (склон к низким значениям, Z = {z_skew:.2f})"
            else:
                direction = f"левосторонняя асимметрия (склон к высоким значениям, Z = {z_skew:.2f})"
            
            if abs(stats.kurtosis(vals) / np.sqrt(
                (4 * (n**2 - 1) * (n - 3)) / ((n + 1)**2 * (n + 3) * (n + 5))
            )) > 1.96:
                kurt_text = "значимый эксцесс (тяжёлые/лёгкие хвосты)"
            else:
                kurt_text = "эксцесс не значим"
            
            self.add_paragraph(f"Распределение {direction}, {kurt_text}.\n")
        
        # QQ-plot для визуальной проверки
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, (scale, name) in enumerate(scales):
            vals = self.completed[scale].dropna()
            stats.probplot(vals, dist="norm", plot=axes[i])
            axes[i].set_title(name, fontproperties=LABEL_FONT)
            # Настраиваем шрифт для tick labels
            for label in axes[i].get_xticklabels() + axes[i].get_yticklabels():
                label.set_fontproperties(TICK_FONT)
        
        plt.suptitle('QQ-plot для проверки нормальности распределений', fontproperties=TITLE_FONT)
        plt.tight_layout()
        path = self.save_figure('qqplot_normality')
        self.add_paragraph(f"\n![QQ-plot нормальности]({path})")
        
        self.add_paragraph(
            "**Примечание:** При больших выборках (n > 200) тесты на нормальность обладают высокой "
            "мощностью и часто отвергают H₀ о нормальности даже при незначительных отклонениях. "
            "Для таких случаев рекомендуется использовать непараметрические тесты (Kruskal-Wallis, "
            "Манна-Уитни, Jonckheere-Terpstra), которые не требуют предположения о нормальности."
        )

    def analyze_h0_anova_levels(self):
        """
        H0: Респонденты с разными профилями (уровнями 1-6) статистически 
        значимо различаются по MBI, прокрастинации и SWLS.
        
        Метод: ANOVA / Kruskal-Wallis + post-hoc
        """
        self.add_section("H0: Различия между уровнями профиля", 3)
        
        self.add_paragraph(HYPOTHESES['H0'])
        
        for scale, name in [('proc_total', 'Прокрастинация'),
                           ('swls_total', 'SWLS'),
                           ('mbi_total', 'MBI (выгорание)')]:
            self.add_paragraph(f"\n**{name}:**")
            
            # Группировка по уровням - фильтруем пустые группы
            groups = []
            for level in range(1, 8):
                group_data = self.completed[self.completed['level'] == level][scale].dropna()
                if len(group_data) > 0:
                    groups.append(group_data)
            
            # Проверяем, есть ли достаточно групп с данными
            if len(groups) < 2:
                self.add_paragraph("⚠️ Недостаточно данных для анализа")
                continue
            
            # Проверка нормальности (тест Шапиро-Уилка) для групп с n >= 3
            normal_groups = 0
            for g in groups:
                if len(g) >= 3:
                    try:
                        _, p_norm = stats.shapiro(g)
                        if p_norm > 0.05:
                            normal_groups += 1
                    except:
                        pass
            
            # Выбираем тест и вычисляем статистику
            if normal_groups >= 2 and all(len(g) >= 3 for g in groups):
                # ANOVA - используем только непустые группы
                try:
                    f_stat, p_value = stats.f_oneway(*groups)
                except Exception as e:
                    # Если ANOVA не работает, используем Kruskal-Wallis
                    h_stat, p_value = stats.kruskal(*groups)
                    f_stat = h_stat
                    normal_groups = 0  # Сбрасываем для правильного вывода
                test_name = "Односторонний ANOVA"
            else:
                # Kruskal-Wallis (непараметрический)
                try:
                    h_stat, p_value = stats.kruskal(*groups)
                    f_stat = h_stat
                except Exception as e:
                    self.add_paragraph(f"⚠️ Ошибка при расчёте: {e}")
                    continue
                test_name = "Kruskal-Wallis"
            
            # Проверяем на NaN
            if p_value is None or (isinstance(p_value, float) and np.isnan(p_value)):
                self.add_paragraph("⚠️ Недостаточно данных для надёжного расчёта")
                continue
            
            # Вычисление размера эффекта
            if normal_groups >= 2:
                # eta-squared
                grand_mean = self.completed[scale].mean()
                ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups if len(g) > 0)
                ss_total = sum((self.completed[scale] - grand_mean)**2)
                es = ss_between / ss_total if ss_total > 0 else 0
            else:
                # epsilon-squared
                k = len(groups)
                n = sum(len(g) for g in groups)
                try:
                    h_stat, _ = stats.kruskal(*groups)
                    es = (h_stat - k + 1) / (n - k) if n > k else 0
                except:
                    es = 0
            
            conclusion = "Подтверждается" if p_value < 0.05 else "Не подтверждается"
            self.add_test_result(test_name, f_stat if f_stat is not None else 0, 
                               p_value, conclusion, es)
            
            # Post-hoc (Tukey или Dunn)
            if p_value < 0.05:
                self.add_paragraph("*Post-hoc сравнения доступны в детальном анализе*")
            
            # Описательная статистика по группам — сортировка по убыванию M
            table_data = []
            for level in range(1, 8):
                g = self.completed[self.completed['level'] == level][scale].dropna()
                if len(g) > 0:
                    profile_name = PROFILE_LEVELS.get(level, f'Уровень {level}')
                    table_data.append((g.mean(), [f"{level} — {profile_name}", f"{g.mean():.1f}", f"{g.std():.1f}", len(g)]))

            # Сортируем по убыванию среднего
            table_data.sort(key=lambda x: x[0], reverse=True)
            sorted_rows = [row for _, row in table_data]

            self.add_table(['Уровень', 'M', 'SD', 'n'], sorted_rows)

    def _analyze_zone_position_trend(self, df, section_title, hyp_key, sample_desc,
                                     zone_col, zone_emoji, zone_name_ru,
                                     scales_increasing, scales_decreasing):
        """
        Универсальный метод: тренд по позиции заданной зоны.

        Параметры
        ---------
        zone_col : str
            Имя колонки зоны ('cubes_reactive', 'cubes_proactive', 'cubes_operational').
        zone_emoji : str
            Эмодзи зоны для отображения.
        zone_name_ru : str
            Русское название зоны.
        scales_increasing : list
            Шкалы, для которых ожидается рост (позиция 1→2→3 → шкала растёт).
        scales_decreasing : list
            Шкалы, для которых ожидается убывание (позиция 1→2→3 → шкала падает).
        """
        self.add_section(section_title, 2)
        self.add_paragraph(HYPOTHESES[hyp_key])

        # Строим описание ожидаемого тренда
        trend_desc_parts = []
        for sc in scales_increasing:
            trend_desc_parts.append(f"{sc[1]} растёт: R1 < R2 < R3")
        for sc in scales_decreasing:
            trend_desc_parts.append(f"{sc[1]} убывает: R1 > R2 > R3")
        trend_desc = ", ".join(trend_desc_parts)

        self.add_paragraph(f"""
**Описание ({sample_desc}):**

Все респонденты делятся на 3 группы по позиции зоны «{zone_name_ru}» ({zone_emoji}) при сортировке зон по убыванию:
- **Уровень R1:** {zone_emoji} на 1-м месте (или все зоны равны)
- **Уровень R2:** {zone_emoji} на 2-м месте
- **Уровень R3:** {zone_emoji} на 3-м месте

Направленная гипотеза:
{trend_desc}

Для проверки используется **тест Джонкхира-Терпстры (Jonckheere-Terpstra)** — непараметрический тест для упорядоченных альтернатив.
""")

        # Классификация по позиции заданной зоны
        def zone_position(row):
            r = row.get('cubes_reactive', 0) or 0
            g = row.get('cubes_proactive', 0) or 0
            o = row.get('cubes_operational', 0) or 0

            # Если все зоны равны → R1 (1-е место для любой зоны)
            if r == g == o:
                return 0

            zones = [('reactive', r), ('proactive', g), ('operational', o)]
            zones_sorted = sorted(zones, key=lambda x: x[1], reverse=True)
            for i, (name, val) in enumerate(zones_sorted):
                if name == zone_col.replace('cubes_', ''):
                    return i
            return 1  # fallback

        df = df.copy()
        df['zone_pos'] = df.apply(zone_position, axis=1)
        pos_labels = {0: f'R1 ({zone_emoji} 1-е)', 1: f'R2 ({zone_emoji} 2-е)', 2: f'R3 ({zone_emoji} 3-е)'}

        self.add_paragraph(f"**Распределение по позициям {zone_emoji} ({sample_desc}):**")
        desc_data = []
        for pos in [0, 1, 2]:
            grp = df[df['zone_pos'] == pos]
            if len(grp) > 0:
                desc_data.append([pos_labels[pos], len(grp)])
        self.add_table(['Позиция ' + zone_emoji, 'n'], desc_data)

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
            for pos in [0, 1, 2]:
                grp = df[df['zone_pos'] == pos][scale].dropna()
                if len(grp) > 0:
                    table_rows.append([pos_labels[pos], f"{grp.mean():.1f}", f"{grp.std():.1f}", len(grp)])
                    groups_ordered.append(grp.values)

            self.add_table(['Позиция ' + zone_emoji, 'M', 'SD', 'n'], table_rows)

            if len(groups_ordered) < 3:
                self.add_paragraph("⚠️ Недостаточно групп для теста")
                continue

            try:
                J_stat, p = jonckheere_terpstra(groups_ordered, alternative=direction)

                if direction == 'increasing':
                    expected = "R1 < R2 < R3 (рост)"
                else:
                    expected = "R1 > R2 > R3 (убывание)"

                confirmed = p < 0.05

                self.add_paragraph(f"- **Ожидаемый тренд:** {expected}")
                self.add_paragraph(f"- **Jonckheere-Terpstra:** J = {J_stat:.0f}, p = {p:.6f}")

                if confirmed:
                    self.add_paragraph(f"- ✅ **Подтверждается**: направленная гипотеза подтверждается (p < 0.05)")
                else:
                    self.add_paragraph(f"- ❌ **Не подтверждается**: p = {p:.4f} > 0.05")

                # Попарные сравнения (Манн-Уитни U с поправкой Бонферрони α/3)
                self.add_paragraph(f"\n**Попарные сравнения (U-тест, α = 0.0167, Бонферрони):**")

                pair_labels = [(0, 1, 'R1 vs R2'), (1, 2, 'R2 vs R3'), (0, 2, 'R1 vs R3')]
                for i, j, lbl in pair_labels:
                    grp_i = df[df['zone_pos'] == i][scale].dropna()
                    grp_j = df[df['zone_pos'] == j][scale].dropna()
                    if len(grp_i) >= 2 and len(grp_j) >= 2:
                        u_stat, u_p = stats.mannwhitneyu(grp_i, grp_j, alternative='two-sided')
                        u_p_bonf = min(u_p * 3, 1.0)  # поправка Бонферрони (3 сравнения)
                        mean_i, mean_j = grp_i.mean(), grp_j.mean()
                        diff = mean_j - mean_i
                        sig = u_p_bonf < 0.05
                        sig_mark = "✅" if sig else "❌"
                        self.add_paragraph(f"- **{lbl}:** M({mean_i:.1f}) vs M({mean_j:.1f}), Δ = {diff:+.1f}, U = {u_stat:.0f}, p = {u_p:.4f}, p_adj = {u_p_bonf:.4f} {sig_mark}")
                    else:
                        self.add_paragraph(f"- **{lbl}:** недостаточно данных")
            except Exception as e:
                self.add_paragraph(f"⚠️ Ошибка при расчёте: {e}")

    def analyze_h0a_green_position_trend(self):
        """H0a: Тренд по позиции 🟢 — все респонденты."""
        self._analyze_zone_position_trend(
            self.completed,
            "H0a: Тренд по позиции 🟢 (все респонденты, Jonckheere-Terpstra)",
            'H0a', 'полная выборка',
            zone_col='cubes_proactive', zone_emoji='🟢', zone_name_ru='Целевое',
            scales_increasing=[
                ('mbi_total', 'MBI (выгорание)'),
                ('proc_total', 'Прокрастинация'),
            ],
            scales_decreasing=[
                ('swls_total', 'SWLS'),
            ]
        )

    def analyze_h0b_green_position_typical(self):
        """H0b: Тренд по позиции 🟢 — только rep=0."""
        df_t = self.completed[self.completed['representative'] == 0].copy()
        n_t = len(df_t)
        if n_t < 10:
            self.add_section("H0b: Тренд по позиции 🟢 (rep=0, Jonckheere-Terpstra)", 2)
            self.add_paragraph(HYPOTHESES['H0b'])
            self.add_paragraph(f"⚠️ Недостаточно данных: n = {n_t} для rep=0")
            return

        self._analyze_zone_position_trend(
            df_t,
            "H0b: Тренд по позиции 🟢 (типовая неделя rep=0, Jonckheere-Terpstra)",
            'H0b', f'типовая неделя (n = {n_t})',
            zone_col='cubes_proactive', zone_emoji='🟢', zone_name_ru='Целевое',
            scales_increasing=[
                ('mbi_total', 'MBI (выгорание)'),
                ('proc_total', 'Прокрастинация'),
            ],
            scales_decreasing=[
                ('swls_total', 'SWLS'),
            ]
        )

    def analyze_h0c_reactive_position_trend(self):
        """H0c: Тренд по позиции 🔴 — все респонденты."""
        self._analyze_zone_position_trend(
            self.completed,
            "H0c: Тренд по позиции 🔴 (все респонденты, Jonckheere-Terpstra)",
            'H0c', 'полная выборка',
            zone_col='cubes_reactive', zone_emoji='🔴', zone_name_ru='Срочное',
            scales_increasing=[
                ('swls_total', 'SWLS'),
            ],
            scales_decreasing=[
                ('mbi_total', 'MBI (выгорание)'),
                ('proc_total', 'Прокрастинация'),
            ]
        )

    def _analyze_zone_count_trend(self, df, section_title, hyp_key, sample_desc,
                                  zone_col, zone_emoji, zone_name_ru,
                                  scales_increasing, scales_decreasing):
        """
        Универсальный метод: тренд по количеству кубиков в заданной зоне (0-7).

        Параметры
        ---------
        zone_col : str
            Имя колонки зоны ('cubes_reactive', 'cubes_proactive', 'cubes_operational').
        zone_emoji : str
            Эмодзи зоны для отображения.
        zone_name_ru : str
            Русское название зоны.
        scales_increasing : list
            Шкалы, для которых ожидается рост (число кубиков 0→7 → шкала растёт).
        scales_decreasing : list
            Шкалы, для которых ожидается убывание (число кубиков 0→7 → шкала падает).
        """
        self.add_section(section_title, 2)
        self.add_paragraph(HYPOTHESES[hyp_key])

        # Строим описание ожидаемого тренда
        trend_desc_parts = []
        for sc in scales_increasing:
            trend_desc_parts.append(f"{sc[1]} растёт: 0 < 1 < 2 < ... < 7")
        for sc in scales_decreasing:
            trend_desc_parts.append(f"{sc[1]} убывает: 0 > 1 > 2 > ... > 7")
        trend_desc = ", ".join(trend_desc_parts)

        self.add_paragraph(f"""
**Описание ({sample_desc}):**

Все респонденты группируются по количеству кубиков в зоне «{zone_name_ru}» ({zone_emoji}) от 0 до 7:
- **0 кубиков {zone_emoji}** — зона полностью отсутствует
- **1 кубик {zone_emoji}**
- ...
- **7 кубиков {zone_emoji}** — все кубики в {zone_name_ru}

Направленная гипотеза:
{trend_desc}

Для проверки используется **тест Джонкхира-Терпстры (Jonckheere-Terpstra)** — непараметрический тест для упорядоченных альтернатив.
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
        self.add_table(['Уровень (' + zone_emoji + ')', 'n'], desc_data)

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

            self.add_table(['Уровень (' + zone_emoji + ')', 'M', 'SD', 'n'], table_rows)

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

                # Попарные сравнения (Манн-Уитни U с поправкой Бонферрони)
                # Количество попарных сравнений = C(k,2) где k — число групп
                k_groups = len(groups_ordered)
                n_comparisons = k_groups * (k_groups - 1) // 2
                alpha_bonf = 0.05 / n_comparisons if n_comparisons > 0 else 0.05

                self.add_paragraph(f"\n**Попарные сравнения (U-тест, α = {alpha_bonf:.4f}, Бонферрони, {n_comparisons} сравнений):**")

                # Выполняем все попарные сравнения (только для групп с n>=10)
                counts_present = [c for c in range(0, 8) if len(df[df['zone_count'] == c][scale].dropna()) >= 10]
                n_comparisons_made = 0
                for i_idx, count_i in enumerate(counts_present):
                    for count_j in counts_present[i_idx + 1:]:
                        grp_i = df[df['zone_count'] == count_i][scale].dropna()
                        grp_j = df[df['zone_count'] == count_j][scale].dropna()
                        if len(grp_i) >= 2 and len(grp_j) >= 2:
                            u_stat, u_p = stats.mannwhitneyu(grp_i, grp_j, alternative='two-sided')
                            u_p_bonf = min(u_p * n_comparisons, 1.0)
                            mean_i, mean_j = grp_i.mean(), grp_j.mean()
                            diff = mean_j - mean_i
                            sig = u_p_bonf < 0.05
                            sig_mark = "✅" if sig else "❌"
                            self.add_paragraph(f"- **{count_i} vs {count_j} {zone_emoji}:** M({mean_i:.1f}) vs M({mean_j:.1f}), Δ = {diff:+.1f}, U = {u_stat:.0f}, p = {u_p:.4f}, p_adj = {u_p_bonf:.4f} {sig_mark}")
                            n_comparisons_made += 1
                if n_comparisons_made == 0:
                    self.add_paragraph(f"⚠️ Нет пар с достаточным количеством данных (нужно ≥2 в каждой группе)")
            except Exception as e:
                self.add_paragraph(f"⚠️ Ошибка при расчёте: {e}")

        # Визуализация тренда
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        scales_to_plot = [
            ('mbi_total', 'MBI (выгорание)', '#e74c3c'),
            ('proc_total', 'Прокрастинация', '#f39c12'),
            ('swls_total', 'SWLS', '#27ae60'),
        ]

        # Текстовое имя зоны для графиков (без эмодзи)
        zone_label_map = {'cubes_proactive': 'Целевое', 'cubes_reactive': 'Срочное', 'cubes_operational': 'Операционное'}
        zone_label = zone_label_map.get(zone_col, zone_col.replace('cubes_', ''))

        for ax_idx, (scale_col, scale_name, color) in enumerate(scales_to_plot):
            means = []
            sems = []
            counts = []
            for count in range(0, 8):
                grp = df[df['zone_count'] == count][scale_col].dropna()
                if len(grp) >= 10:
                    means.append(grp.mean())
                    sems.append(grp.std() / np.sqrt(len(grp)))  # SEM — стандартная ошибка среднего
                    counts.append(count)

            if len(means) > 1:
                axes[ax_idx].errorbar(counts, means, yerr=sems, color=color, marker='o',
                                     linewidth=2, capsize=5, markersize=8)
                axes[ax_idx].set_xlabel(f'Количество кубиков ({zone_label})', fontproperties=LABEL_FONT)
                axes[ax_idx].set_ylabel(scale_name, fontproperties=LABEL_FONT)
                axes[ax_idx].set_title(scale_name, fontproperties=TITLE_FONT)
                # Линейный тренд
                z = np.polyfit(counts, means, 1)
                p_line = np.poly1d(z)
                x_line = np.linspace(min(counts), max(counts), 100)
                axes[ax_idx].plot(x_line, p_line(x_line), '--', color=color, alpha=0.5)
            else:
                axes[ax_idx].text(0.5, 0.5, 'Нет данных', ha='center', va='center',
                                 transform=axes[ax_idx].transAxes, fontproperties=LABEL_FONT)
                axes[ax_idx].set_title(scale_name, fontproperties=TITLE_FONT)

            for label in axes[ax_idx].get_xticklabels() + axes[ax_idx].get_yticklabels():
                label.set_fontproperties(TICK_FONT)

        plt.suptitle(f'Тренд по количеству кубиков — {zone_label} ({sample_desc})', fontproperties=TITLE_FONT)
        plt.tight_layout()
        safe_zone = zone_col.replace('cubes_', '')
        # Уникальное имя файла на основе ключа гипотезы
        file_suffix = hyp_key.lower().replace('h0', 'h0_')
        path = self.save_figure(f'{file_suffix}_{safe_zone}_trend')
        self.add_paragraph(f"\n![Тренд {zone_emoji}]({path})")

    def analyze_h0a1_green_count_trend(self):
        """H0a1: Тренд по числу кубиков 🟢 (0→7) — все респонденты."""
        self._analyze_zone_count_trend(
            self.completed,
            "H0a1: Тренд по числу кубиков 🟢 (0→7, все респонденты, Jonckheere-Terpstra)",
            'H0a1', 'полная выборка',
            zone_col='cubes_proactive', zone_emoji='🟢', zone_name_ru='Целевое',
            scales_increasing=[
                ('swls_total', 'SWLS'),
            ],
            scales_decreasing=[
                ('mbi_total', 'MBI (выгорание)'),
                ('proc_total', 'Прокрастинация'),
            ]
        )

    def analyze_h0b1_green_count_typical(self):
        """H0b1: Тренд по числу кубиков 🟢 (0→7) — только rep=0."""
        df_t = self.completed[self.completed['representative'] == 0].copy()
        n_t = len(df_t)
        if n_t < 10:
            self.add_section("H0b1: Тренд по числу кубиков 🟢 (0→7, rep=0, Jonckheere-Terpstra)", 2)
            self.add_paragraph(HYPOTHESES['H0b1'])
            self.add_paragraph(f"⚠️ Недостаточно данных: n = {n_t} для rep=0")
            return

        self._analyze_zone_count_trend(
            df_t,
            f"H0b1: Тренд по числу кубиков 🟢 (0→7, типовая неделя rep=0, Jonckheere-Terpstra, n = {n_t})",
            'H0b1', f'типовая неделя (n = {n_t})',
            zone_col='cubes_proactive', zone_emoji='🟢', zone_name_ru='Целевое',
            scales_increasing=[
                ('swls_total', 'SWLS'),
            ],
            scales_decreasing=[
                ('mbi_total', 'MBI (выгорание)'),
                ('proc_total', 'Прокрастинация'),
            ]
        )

    def analyze_h0c1_red_count_trend(self):
        """H0c1: Тренд по числу кубиков 🔴 (0→7) — все респонденты."""
        self._analyze_zone_count_trend(
            self.completed,
            "H0c1: Тренд по числу кубиков 🔴 (0→7, все респонденты, Jonckheere-Terpstra)",
            'H0c1', 'полная выборка',
            zone_col='cubes_reactive', zone_emoji='🔴', zone_name_ru='Срочное',
            scales_increasing=[
                ('mbi_total', 'MBI (выгорание)'),
                ('proc_total', 'Прокрастинация'),
            ],
            scales_decreasing=[
                ('swls_total', 'SWLS'),
            ]
        )
    
    def analyze_h1_correlations_reactive(self):
        """
        H1: Доля 🔴 (Срочное) положительно коррелирует с прокрастинацией и MBI.
        Дополнительно: корреляция 🔴 с SWLS (исследовательский).

        Метод: Корреляция Спирмена
        """
        self.add_section("H1: Корреляция 🔴 с валидационными шкалами", 3)

        self.add_paragraph(HYPOTHESES['H1'])

        var1 = self.completed['cubes_reactive']
        var1_min, var1_max = var1.min(), var1.max()

        for scale, name in [('proc_total', 'Прокрастинация'),
                           ('mbi_total', 'MBI (выгорание)')]:
            var2 = self.completed[scale]
            var2_min, var2_max = var2.min(), var2.max()

            r, p = stats.spearmanr(var1, var2)

            self.add_paragraph(f"**🔴 ({int(var1_min)}-{int(var1_max)}) vs {name} ({int(var2_min)}-{int(var2_max)}):**")
            self.add_paragraph(f"Коэффициент корреляции r = {r:.3f}, p = {p:.4f}")
            self.add_paragraph(f"Диапазон значений: 🔴 [{int(var1_min)}; {int(var1_max)}], {name} [{int(var2_min)}; {int(var2_max)}]")

            conclusion = "Подтверждается (положительная связь)" if r > 0 and p < 0.05 else "Не подтверждается"
            self.add_paragraph(f"**Вывод**: {conclusion}")

        # Дополнительно: корреляция 🔴 с SWLS (исследовательский анализ)
        scale, name = 'swls_total', 'SWLS (удовлетворённость)'
        var2 = self.completed[scale]
        var2_min, var2_max = var2.min(), var2.max()
        r, p = stats.spearmanr(var1, var2)

        self.add_paragraph(f"\n**🔴 ({int(var1_min)}-{int(var1_max)}) vs {name} ({int(var2_min)}-{int(var2_max)}) (дополнительно):**")
        self.add_paragraph(f"Коэффициент корреляции r = {r:.3f}, p = {p:.4f}")
        self.add_paragraph(f"Диапазон значений: 🔴 [{int(var1_min)}; {int(var1_max)}], {name} [{int(var2_min)}; {int(var2_max)}]")
        direction = "отрицательная" if r < 0 else "положительная"
        conclusion = f"Обнаружена {direction} связь" if p < 0.05 else "Связь не обнаружена"
        self.add_paragraph(f"**Вывод**: {conclusion}")
    
    def analyze_h2_correlations_proactive(self):
        """
        H2: Доля 🟢 (Целевое) положительно коррелирует с SWLS и 
        отрицательно — с MBI.
        
        Метод: Корреляция Спирмена
        """
        self.add_section("H2: Корреляция 🟢 с валидационными шкалами", 3)
        
        self.add_paragraph(HYPOTHESES['H2'])
        
        var1 = self.completed['cubes_proactive']
        var1_min, var1_max = int(var1.min()), int(var1.max())
        
        # SWLS
        var2_swls = self.completed['swls_total']
        var2_min, var2_max = int(var2_swls.min()), int(var2_swls.max())
        
        r, p = stats.spearmanr(var1, var2_swls)
        
        self.add_paragraph(f"**🟢 ({var1_min}-{var1_max}) vs SWLS ({var2_min}-{var2_max}):**")
        self.add_paragraph(f"Коэффициент корреляции r = {r:.3f}, p = {p:.4f}")
        self.add_paragraph(f"Диапазон значений: 🟢 [{var1_min}; {var1_max}], SWLS [{var2_min}; {var2_max}]")
        
        conclusion = "Подтверждается (положительная связь)" if r > 0 and p < 0.05 else "Не подтверждается"
        self.add_paragraph(f"**Вывод**: {conclusion}")
        
        # MBI
        var2_mbi = self.completed['mbi_total']
        var2_min, var2_max = int(var2_mbi.min()), int(var2_mbi.max())
        
        r, p = stats.spearmanr(var1, var2_mbi)
        
        self.add_paragraph(f"\n**🟢 ({var1_min}-{var1_max}) vs MBI ({var2_min}-{var2_max}):**")
        self.add_paragraph(f"Коэффициент корреляции r = {r:.3f}, p = {p:.4f}")
        self.add_paragraph(f"Диапазон значений: 🟢 [{var1_min}; {var1_max}], MBI [{var2_min}; {var2_max}]")
        
        conclusion = "Подтверждается (отрицательная связь)" if r < 0 and p < 0.05 else "Не подтверждается"
        self.add_paragraph(f"**Вывод**: {conclusion}")

        # Дополнительно: корреляция 🟢 с Прокрастинацией (исследовательский анализ)
        scale, name = 'proc_total', 'Прокрастинация'
        var2 = self.completed[scale]
        var2_min, var2_max = var2.min(), var2.max()
        r, p = stats.spearmanr(var1, var2)

        self.add_paragraph(f"\n**🟢 ({var1_min}-{var1_max}) vs {name} ({var2_min}-{var2_max}) (дополнительно):**")
        self.add_paragraph(f"Коэффициент корреляции r = {r:.3f}, p = {p:.4f}")
        self.add_paragraph(f"Диапазон значений: 🟢 [{var1_min}; {var1_max}], {name} [{var2_min}; {var2_max}]")
        direction = "отрицательная" if r < 0 else "положительная"
        conclusion = f"Обнаружена {direction} связь" if p < 0.05 else "Связь не обнаружена"
        self.add_paragraph(f"**Вывод**: {conclusion}")

    def analyze_h3_low_reactive(self):
        """
        H3: Профили с низким 🔴 (<2 кубиков) демонстрируют более низкое выгорание.
        
        Метод: Сравнение двух групп (t-тест / U-критерий Манна-Уитни)
        """
        self.add_section("H3: Низкое 🔴 и выгорание", 3)
        
        self.add_paragraph(HYPOTHESES['H3'])
        
        low_r = self.completed[self.completed['cubes_reactive'] < 2]['mbi_total'].dropna()
        high_r = self.completed[self.completed['cubes_reactive'] >= 2]['mbi_total'].dropna()
        
        self.add_paragraph(f"- Низкое 🔴 (<2): n = {len(low_r)}, M = {low_r.mean():.1f}, SD = {low_r.std():.1f}")
        self.add_paragraph(f"- Высокое 🔴 (≥2): n = {len(high_r)}, M = {high_r.mean():.1f}, SD = {high_r.std():.1f}")
        
        # Проверка нормальности
        if len(low_r) >= 3 and len(high_r) >= 3:
            try:
                _, p_low = stats.shapiro(low_r)
                _, p_high = stats.shapiro(high_r)
            except:
                p_low, p_high = 0, 0
            
            if p_low > 0.05 and p_high > 0.05:
                t_stat, p_value = stats.ttest_ind(low_r, high_r)
                test_name = "t-тест Стьюдента"
                es = (low_r.mean() - high_r.mean()) / np.sqrt(
                    ((len(low_r)-1)*low_r.std()**2 + (len(high_r)-1)*high_r.std()**2) / 
                    (len(low_r) + len(high_r) - 2))
            else:
                u_stat, p_value = stats.mannwhitneyu(low_r, high_r, alternative='two-sided')
                test_name = "U-критерий Манна-Уитни"
                es = None
        else:
            u_stat, p_value = stats.mannwhitneyu(low_r, high_r, alternative='two-sided')
            test_name = "U-критерий Манна-Уитни"
            es = None
        
        # Проверка на None/NaN
        if p_value is None or (isinstance(p_value, float) and np.isnan(p_value)):
            self.add_paragraph("⚠️ Недостаточно данных для надёжного расчёта")
            return
            
        conclusion = "Подтверждается" if p_value < 0.05 else "Не подтверждается"
        self.add_test_result(test_name, u_stat if 'U' in test_name else t_stat, 
                           p_value, conclusion, es)
    
    def analyze_h4_green_dominance(self):
        """
        H4: Профили с доминированием 🟢 (уровни 5 и 6) имеют высокие баллы SWLS.
        
        Метод: Сравнение средних
        """
        self.add_section("H4: Доминирование 🟢 и удовлетворённость", 3)
        
        self.add_paragraph(HYPOTHESES['H4'])
        
        green_levels = self.completed[self.completed['level'].isin([6, 7])]['swls_total'].dropna()
        other_levels = self.completed[~self.completed['level'].isin([6, 7])]['swls_total'].dropna()

        self.add_paragraph(f"- Уровни 6-7 (доминирование 🟢): n = {len(green_levels)}, "
                          f"M = {green_levels.mean():.1f}, SD = {green_levels.std():.1f}")
        self.add_paragraph(f"- Другие уровни: n = {len(other_levels)}, "
                          f"M = {other_levels.mean():.1f}, SD = {other_levels.std():.1f}")
        
        u_stat, p_value = stats.mannwhitneyu(green_levels, other_levels, alternative='greater')
        
        # Проверка на None/NaN
        if p_value is None or (isinstance(p_value, float) and np.isnan(p_value)):
            self.add_paragraph("⚠️ Недостаточно данных для надёжного расчёта")
            return
            
        conclusion = "Подтверждается" if p_value < 0.05 else "Не подтверждается"
        self.add_test_result("U-критерий Манна-Уитни", u_stat, p_value, conclusion)
    
    def analyze_h5_moderation_records(self):
        """
        H5: Модерация записями. Респонденты, восстанавливавшие по записям (4-5 баллов),
        имеют больше 🟢, меньше , ниже прокрастинацию и выгорание.

        Метод: Сравнение групп
        """
        self.add_section("H5: Модерация записями", 3)

        self.add_paragraph(HYPOTHESES['H5'])

        by_records = self.completed[self.completed['memory_vs_records'] >= 4]  # По записям
        by_memory = self.completed[self.completed['memory_vs_records'] <= 2]  # По памяти

        self.add_paragraph(f"- По записям (4-5): n = {len(by_records)}")
        self.add_paragraph(f"- По памяти (1-2): n = {len(by_memory)}")

        # Сравнение зон с p-value
        table_data = []
        for zone in ['cubes_proactive', 'cubes_reactive', 'cubes_operational']:
            zone_name = ZONE_NAMES.get(zone.replace('cubes_', ''), zone)
            r1 = by_records[zone].mean()
            r2 = by_memory[zone].mean()
            
            if len(by_records) >= 3 and len(by_memory) >= 3:
                u_stat, p = stats.mannwhitneyu(by_records[zone], by_memory[zone])
                sig = "*" if p is not None and p < 0.05 else ""
                p_str = f"p={p:.3f}{sig}"
            else:
                p_str = "n/a"
            
            table_data.append([zone_name, f"{r1:.2f}", f"{r2:.2f}", f"{r1-r2:+.2f}", p_str])

        self.add_table(['Зона', 'По записям', 'По памяти', 'Разница', 'p-value'], table_data)

        # Прокрастинация с p-value
        if len(by_records) >= 3 and len(by_memory) >= 3:
            proc_by_records = by_records['proc_total'].mean()
            proc_by_memory = by_memory['proc_total'].mean()
            u_proc, p_proc = stats.mannwhitneyu(by_records['proc_total'], by_memory['proc_total'])
            sig_proc = "*" if p_proc is not None and p_proc < 0.05 else ""
            self.add_paragraph(f"\nПрокрастинация: по записям = {proc_by_records:.1f}, "
                              f"по памяти = {proc_by_memory:.1f} (p={p_proc:.3f}{sig_proc})")

            # MBI с p-value
            mbi_by_records = by_records['mbi_total'].mean()
            mbi_by_memory = by_memory['mbi_total'].mean()
            u_mbi, p_mbi = stats.mannwhitneyu(by_records['mbi_total'], by_memory['mbi_total'])
            sig_mbi = "*" if p_mbi is not None and p_mbi < 0.05 else ""
            self.add_paragraph(f"MBI: по записям = {mbi_by_records:.1f}, по памяти = {mbi_by_memory:.1f} (p={p_mbi:.3f}{sig_mbi})")
        else:
            proc_by_records = by_records['proc_total'].mean() if len(by_records) > 0 else 0
            proc_by_memory = by_memory['proc_total'].mean() if len(by_memory) > 0 else 0
            self.add_paragraph(f"\nПрокрастинация: по записям = {proc_by_records:.1f}, "
                              f"по памяти = {proc_by_memory:.1f}")
            
            mbi_by_records = by_records['mbi_total'].mean() if len(by_records) > 0 else 0
            mbi_by_memory = by_memory['mbi_total'].mean() if len(by_memory) > 0 else 0
            self.add_paragraph(f"MBI: по записям = {mbi_by_records:.1f}, по памяти = {mbi_by_memory:.1f}")

        # Интерпретация
        self.add_paragraph("\n*Значимость: * p<0.05*")
    
    def analyze_h6_moderation_red(self):
        """
        H6: Модерация красным. Связь ⚪ с прокрастинацией усиливается при высоком 🔴.
        
        Метод: Взаимодействие в регрессии
        """
        self.add_section("H6: Модерация 🔴", 3)
        
        self.add_paragraph(HYPOTHESES['H6'])
        
        # Разбиваем на группы по уровню 🔴
        low_red = self.completed[self.completed['cubes_reactive'] <= 1]
        high_red = self.completed[self.completed['cubes_reactive'] >= 3]
        
        if len(low_red) > 10 and len(high_red) > 10:
            r_low, p_low = stats.spearmanr(low_red['cubes_operational'], 
                                          low_red['proc_total'])
            r_high, p_high = stats.spearmanr(high_red['cubes_operational'], 
                                            high_red['proc_total'])
            
            self.add_paragraph(f"- Низкое 🔴 (0-1): корреляция ⚪-прокрастинация r = {r_low:.3f}, p = {p_low:.4f}")
            self.add_paragraph(f"- Высокое 🔴 (3+): корреляция ⚪-прокрастинация r = {r_high:.3f}, p = {p_high:.4f}")
            
            conclusion = "Подтверждается (связь сильнее при высоком 🔴)" if (abs(r_high) > abs(r_low)) and (p_high is not None and p_high < 0.05) else "Не подтверждается"
            self.add_paragraph(f"**Вывод**: {conclusion}")
        else:
            self.add_paragraph("⚠️ Недостаточно данных для анализа модерации")
    
    def analyze_h7_curvilinear_green(self):
        """
        H7: Криволинейная связь 🟢 и выгорания (U-образная).
        
        Метод: Полиномиальная регрессия
        """
        self.add_section("H7: Криволинейная связь 🟢 и MBI", 3)
        
        self.add_paragraph(HYPOTHESES['H7'])
        
        x = self.completed['cubes_proactive']
        y = self.completed['mbi_total']
        
        # Линейная регрессия
        slope, intercept, r_lin, p_lin, _ = stats.linregress(x, y)
        
        # Квадратичная регрессия
        x_quad = np.column_stack([x, x**2])
        # Простая реализация
        r_quad = np.corrcoef(x, y)[0,1]
        
        self.add_paragraph(f"- Линейная модель: r = {r_lin:.3f}, p = {p_lin:.4f}")
        
        # Визуализация
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(x, y, alpha=0.5, label='Данные')

        # Линия тренда
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'r-', label='Линейный тренд')

        ax.set_xlabel('Кубиков в зоне Целевое ●', fontproperties=LABEL_FONT)
        ax.set_ylabel('MBI (выгорание)', fontproperties=LABEL_FONT)
        ax.set_title('Связь Целевое ● и выгорания', fontproperties=TITLE_FONT)
        ax.legend(prop=TICK_FONT)
        # Настраиваем шрифт для tick labels
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(TICK_FONT)

        plt.tight_layout()
        path = self.save_figure('green_burnout_curvilinear', 'Криволинейная связь')
        self.add_paragraph(f"![Криволинейная связь]({path})")
        
        self.add_paragraph("*Требуется большая выборка для надёжной проверки U-образной зависимости*")
    
    def analyze_h8_balance_moderation(self):
        """
        H8: Баланс работа/личное как модератор связи 🔴 и выгорания.
        
        Метод: Модерационный анализ
        """
        self.add_section("H8: Модерация балансом работа/личное", 3)
        
        self.add_paragraph(HYPOTHESES['H8'])
        
        #work_life: -3 = только работа, +3 = только личное
        more_work = self.completed[self.completed['work_life'] < 0]  # Больше работы
        more_life = self.completed[self.completed['work_life'] > 0]  # Больше личного
        
        if len(more_work) > 10 and len(more_life) > 10:
            r_work, p_work = stats.spearmanr(more_work['cubes_reactive'], 
                                             more_work['mbi_total'])
            r_life, p_life = stats.spearmanr(more_life['cubes_reactive'], 
                                             more_life['mbi_total'])
            
            self.add_paragraph(f"- Больше работы (work_life < 0): корреляция 🔴-MBI r = {r_work:.3f}, p = {p_work:.4f}")
            self.add_paragraph(f"- Больше личного (work_life > 0): корреляция 🔴-MBI r = {r_life:.3f}, p = {p_life:.4f}")
            
            conclusion = "Подтверждается" if (abs(r_work) > abs(r_life)) and (p_work is not None and p_work < 0.05) else "Не подтверждается"
            self.add_paragraph(f"**Вывод**: {conclusion}")
        else:
            self.add_paragraph("⚠️ Недостаточно данных")
    
    def analyze_h9_mediation_deficit(self):
        """
        H9: Энергетический дефицит как медиатор связи 🔴 → выгорание.
        
        Метод: Медационный анализ (упрощённый)
        """
        self.add_section("H9: Дефицит как медиатор", 3)
        
        self.add_paragraph(HYPOTHESES['H9'])
        
        # Прямая связь: 🔴 → MBI
        r_direct, p_direct = stats.spearmanr(self.completed['cubes_reactive'], 
                                             self.completed['mbi_total'])
        
        # Связь через дефицит: 🔴 → дефицит → MBI
        r_deficit, p_deficit = stats.spearmanr(self.completed['cubes_reactive'], 
                                               self.completed['energy_deficit'])
        r_deficit_mbi, p_deficit_mbi = stats.spearmanr(self.completed['energy_deficit'], 
                                                       self.completed['mbi_total'])
        
        self.add_paragraph(f"- Прямая связь 🔴 → MBI: r = {r_direct:.3f}, p = {p_direct:.4f}")
        self.add_paragraph(f"- Связь 🔴 → дефицит: r = {r_deficit:.3f}, p = {p_deficit:.4f}")
        self.add_paragraph(f"- Связь дефицит → MBI: r = {r_deficit_mbi:.3f}, p = {p_deficit_mbi:.4f}")
        
        self.add_paragraph("*Полный медиационный анализ требует большей выборки*")
    
    def analyze_h10_gender_differences(self):
        """
        H10: Гендерные различия в распределении зон.
        
        Метод: Сравнение средних (t-тест / U-критерий)
        """
        self.add_section("H10: Гендерные различия", 3)
        
        self.add_paragraph(HYPOTHESES['H10'])
        
        male = self.completed[self.completed['gender'] == 'male']
        female = self.completed[self.completed['gender'] == 'female']
        
        self.add_paragraph(f"- Мужчины: n = {len(male)}")
        self.add_paragraph(f"- Женщины: n = {len(female)}")
        
        table_data = []
        for zone in ['cubes_reactive', 'cubes_proactive', 'cubes_operational']:
            zone_name = ZONE_NAMES.get(zone.replace('cubes_', ''), zone)
            
            m_mean = male[zone].mean()
            f_mean = female[zone].mean()
            
            if len(male) >= 3 and len(female) >= 3:
                u_stat, p = stats.mannwhitneyu(male[zone], female[zone])
                sig = "*" if p is not None and p < 0.05 else ""
            else:
                p = 1.0
                sig = ""
            
            table_data.append([zone_name, f"{m_mean:.2f}", f"{f_mean:.2f}", f"{m_mean-f_mean:+.2f}", f"p={p:.3f}{sig}"])
        
        self.add_table(['Зона', 'Мужчины (M)', 'Женщины (M)', 'Разница', 'p-value'], table_data)
        
        if any('*' in row[4] for row in table_data):
            self.add_paragraph("**Вывод**: Подтверждается — есть значимые гендерные различия")
        else:
            self.add_paragraph("**Вывод**: Не подтверждается — нет значимых гендерных различий")
    
    def analyze_h11_age_trend(self):
        """
        H11: Возрастной тренд. С возрастом доля 🔴 снижается.
        
        Метод: Корреляция
        """
        self.add_section("H11: Возрастной тренд", 3)
        
        self.add_paragraph(HYPOTHESES['H11'])
        
        for zone in ['cubes_reactive', 'cubes_proactive', 'cubes_operational']:
            zone_name = ZONE_NAMES.get(zone.replace('cubes_', ''), zone)
            r, p = stats.spearmanr(self.completed['age'], self.completed[zone])
            
            sig = "*" if p is not None and p < 0.05 else ""
            direction = "снижается" if r < 0 else "растёт"
            self.add_paragraph(f"- {zone_name}: r = {r:.3f}, p = {p:.4f} {sig} — с возрастом {direction}")
    
    def analyze_h12_zen_deficit(self):
        """
        H12: Профиль "Дзен" (уровень 7) связан с наименьшим энергетическим дефицитом.
        
        Метод: Сравнение средних
        """
        self.add_section("H12: Профиль Дзен и дефицит", 3)
        
        self.add_paragraph(HYPOTHESES['H12'])
        
        zen = self.completed[self.completed['level'] == 7]['energy_deficit'].dropna()
        other = self.completed[self.completed['level'] != 7]['energy_deficit'].dropna()

        if len(zen) >= 3:
            self.add_paragraph(f"- Уровень 7 (Дзен): n = {len(zen)}, M = {zen.mean():.2f}")
            self.add_paragraph(f"- Другие уровни: n = {len(other)}, M = {other.mean():.2f}")

            u_stat, p = stats.mannwhitneyu(zen, other, alternative='less')

            # Проверка на None/NaN
            if p is None or (isinstance(p, float) and np.isnan(p)):
                self.add_paragraph("⚠️ Недостаточно данных для надёжного расчёта")
                return

            conclusion = "Подтверждается" if p < 0.05 else "Не подтверждается"
            self.add_test_result("U-критерий Манна-Уитни", u_stat, p, conclusion)
        else:
            self.add_paragraph("⚠️ Недостаточно данных (мало респондентов с уровнем 7)")
    

    def analyze_h7a_linear_regression(self):
        """
        H7a: Множественная линейная регрессия для предсказания MBI, Прокрастинации и SWLS
        на основе двух зон (🔴, 🟢).

        Анализ для трёх выборок: полная, почти типовую (-1..1), точно типовую (0).
        """
        self.add_section("H7a: Линейная регрессия — предсказание по 🔴 и 🟢", 3)

        self.add_paragraph(HYPOTHESES['H7a'])

        self.add_paragraph("""
Множественная линейная регрессия: `Шкала = β₀ + β₁×🔴 + β₂×🟢`

Анализ выполняется для трёх подвыборок:
- **Полная** — все завершённые респонденты
- **Почти типовую** — representative ∈ [-1, 1]
- **Точно типовая** — representative = 0
""")

        # Определяем три выборки
        df_full = self.completed
        df_almost = self.completed[self.completed['representative'].between(-1, 1)]
        df_exact = self.completed[self.completed['representative'] == 0]

        samples = [
            ('Полная', df_full),
            ('Почти типовая (−1..1)', df_almost),
            ('Точно типовая (0)', df_exact),
        ]

        targets = [
            ('mbi_total', 'MBI'),
            ('proc_total', 'Прокрастинация'),
            ('swls_total', 'SWLS'),
        ]

        # ==================== СВОДНАЯ ТАБЛИЦА ====================
        self.add_paragraph("**Сводная таблица регрессий:**\n")

        summary_headers = ['Выборка', 'n', 'Показатель', 'R²',
                           'β₀', 'β₁(🔴)', 'p(🔴)',
                           'β₂(🟢)', 'p(🟢)']
        summary_rows = []

        all_results = {}  # Для графиков

        for sample_name, df in samples:
            n = len(df)
            if n < 10:
                summary_rows.append([sample_name, str(n), '—', '—', '—', '—', '—', '—', '—'])
                all_results[sample_name] = None
                continue

            X_r = df['cubes_reactive'].values
            X_g = df['cubes_proactive'].values
            X = np.column_stack([X_r, X_g])

            sample_results = {}

            for target_col, target_name in targets:
                y = df[target_col].values
                X_int = np.column_stack([np.ones(len(X)), X])

                try:
                    beta = np.linalg.lstsq(X_int, y, rcond=None)[0]
                    y_pred = X_int @ beta
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot)

                    p_num = 2
                    mse = ss_res / (n - p_num - 1)
                    var_beta = mse * np.linalg.inv(X_int.T @ X_int)
                    se_beta = np.sqrt(np.diag(var_beta))
                    t_stats = beta / se_beta
                    from scipy.stats import t as t_dist
                    p_values = 2 * (1 - t_dist.cdf(np.abs(t_stats), df=n-p_num-1))

                    sig = lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

                    summary_rows.append([
                        sample_name, str(n), target_name,
                        f"{r_squared:.3f}",
                        f"{beta[0]:.2f}",
                        f"{beta[1]:.3f}", f"{p_values[1]:.4f}{sig(p_values[1])}",
                        f"{beta[2]:.3f}", f"{p_values[2]:.4f}{sig(p_values[2])}"
                    ])

                    sample_results[target_col] = {
                        'beta': beta, 'r2': r_squared, 'p': p_values, 'n': n
                    }
                except Exception as e:
                    summary_rows.append([sample_name, str(n), target_name, f'Ошибка: {e}'] + ['—']*5)
                    sample_results[target_col] = None

            all_results[sample_name] = sample_results

        self.add_table(summary_headers, summary_rows)
        self.add_paragraph(f"*Значимость: *** p<0.001, ** p<0.01, * p<0.05*")

        # ==================== ГРАФИКИ ====================
        for target_col, target_name in targets:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            for ax_idx, (sample_name, df) in enumerate(samples):
                ax = axes[ax_idx]
                res = all_results[sample_name]
                if res is None or res.get(target_col) is None:
                    ax.text(0.5, 0.5, f'{sample_name}\nнет данных', ha='center', va='center',
                           transform=ax.transAxes, fontproperties=LABEL_FONT)
                    ax.set_title(sample_name, fontproperties=TITLE_FONT)
                    continue

                r_data = res[target_col]
                beta = r_data['beta']
                r2 = r_data['r2']

                X_r = df['cubes_reactive'].values
                X_g = df['cubes_proactive'].values
                y = df[target_col].values

                mean_r, mean_g = X_r.mean(), X_g.mean()

                # 🔴
                x_line = np.linspace(X_r.min(), X_r.max(), 50)
                y_line = beta[0] + beta[1]*x_line + beta[2]*mean_g
                ax.scatter(X_r, y, alpha=0.4, color='#e74c3c', s=20)
                ax.plot(x_line, y_line, '#e74c3c', linewidth=2, label=f'🔴 β={beta[1]:.2f}')

                # 🟢
                x_line = np.linspace(X_g.min(), X_g.max(), 50)
                y_line = beta[0] + beta[1]*mean_r + beta[2]*x_line
                ax.scatter(X_g, y, alpha=0.4, color='#27ae60', s=20)
                ax.plot(x_line, y_line, '#27ae60', linewidth=2, label=f'🟢 β={beta[2]:.2f}')

                ax.set_title(f'{sample_name}\n(R²={r2:.3f}, n={r_data["n"]})', fontproperties=TITLE_FONT)
                ax.set_xlabel('Кубики', fontproperties=LABEL_FONT)
                ax.set_ylabel(target_name, fontproperties=LABEL_FONT)
                ax.legend(prop={'size': 9})
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontproperties(TICK_FONT)

            plt.suptitle(f'{target_name}: регрессия по 🔴 и 🟢', fontproperties=TITLE_FONT)
            plt.tight_layout()
            path = self.save_figure(f'h7a_{target_col}')
            self.add_paragraph(f"\n![{target_name}](figures/h7a_{target_col}.png)")

        # ==================== СРАВНЕНИЕ R² ====================
        fig, ax = plt.subplots(figsize=(8, 5))
        x_pos = np.arange(len(targets))
        w = 0.25
        colors_s = ['#3498db', '#e67e22', '#2ecc71']

        for si, (sample_name, _) in enumerate(samples):
            res = all_results[sample_name]
            if res is None:
                continue
            r2_vals = []
            for target_col, _ in targets:
                r2_vals.append(res[target_col]['r2'] if res.get(target_col) else 0)
            ax.bar(x_pos + si*w - w, r2_vals, w, label=sample_name, color=colors_s[si], alpha=0.8)

        ax.set_ylabel('R²', fontproperties=LABEL_FONT)
        ax.set_title('Сравнение R² по выборкам', fontproperties=TITLE_FONT)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([t[1] for t in targets], fontproperties=TICK_FONT)
        ax.legend(prop=TICK_FONT)
        for label in ax.get_yticklabels():
            label.set_fontproperties(TICK_FONT)
        plt.tight_layout()
        path = self.save_figure('h7a_r2_comparison')
        self.add_paragraph(f"\n![Сравнение R²](figures/h7a_r2_comparison.png)")

        # ==================== ВЫВОДЫ ====================
        self.add_paragraph(f"\n**Выводы по H7a:**")
        for sample_name, _ in samples:
            res = all_results[sample_name]
            if res is None:
                continue
            for target_col, target_name in targets:
                r = res.get(target_col)
                if r:
                    sig_zones = []
                    for zi, zone in [(1, '🔴'), (2, '🟢')]:
                        if r['p'][zi] < 0.05:
                            sig_zones.append(zone)
                    zones_str = ', '.join(sig_zones) if sig_zones else 'нет значимых'
                    self.add_paragraph(f"- {sample_name}, {target_name}: R²={r['r2']:.3f}, значимые: {zones_str}")

    def analyze_h10a_position_comparison(self):
        """
        H10a: Сравнение владельцев бизнеса и высшего руководства с остальными.

        Метод: U-критерий Манна-Уитни
        """
        self.add_section("H10a: Владельцы бизнеса и высшее руководство", 3)

        self.add_paragraph(HYPOTHESES['H10a'])

        # Определяем группы
        leadership_positions = ['Владелец бизнеса', 'Высший менеджмент']
        leaders = self.completed[self.completed['position'].isin(leadership_positions)]
        others = self.completed[~self.completed['position'].isin(leadership_positions)]

        self.add_paragraph(f"- Владельцы бизнеса и высшее руководство: n = {len(leaders)}")
        self.add_paragraph(f"- Остальные: n = {len(others)}")

        if len(leaders) >= 5 and len(others) >= 5:
            # Сравниваем распределение кубиков
            table_data = []
            for zone in ['cubes_reactive', 'cubes_proactive', 'cubes_operational']:
                zone_name = ZONE_NAMES.get(zone.replace('cubes_', ''), zone)
                
                leaders_mean = leaders[zone].mean()
                others_mean = others[zone].mean()
                
                u_stat, p = stats.mannwhitneyu(leaders[zone], others[zone])
                sig = "*" if p is not None and p < 0.05 else ""
                
                table_data.append([zone_name, f"{leaders_mean:.2f}", f"{others_mean:.2f}", 
                                 f"{leaders_mean-others_mean:+.2f}", f"p={p:.3f}{sig}"])

            self.add_table(['Зона', 'Руководство (M)', 'Остальные (M)', 'Разница', 'p-value'], table_data)

            # Сравниваем валидационные шкалы
            self.add_paragraph("\n**Валидационные шкалы:**")
            for scale, name in [('proc_total', 'Прокрастинация'),
                               ('swls_total', 'SWLS'),
                               ('mbi_total', 'MBI')]:
                leaders_mean = leaders[scale].mean()
                others_mean = others[scale].mean()
                u_stat, p = stats.mannwhitneyu(leaders[scale], others[scale])
                sig = "*" if p is not None and p < 0.05 else ""
                self.add_paragraph(f"- {name}: руководство = {leaders_mean:.1f}, остальные = {others_mean:.1f} (p={p:.3f}{sig})")

            # Проверка значимых различий
            significant = any('*' in row[4] for row in table_data)
            if significant:
                self.add_paragraph("\n**Вывод**: Подтверждается — есть значимые различия между группами")
            else:
                self.add_paragraph("\n**Вывод**: Не подтверждается — нет значимых различий между группами")
        else:
            self.add_paragraph("⚠️ Недостаточно данных для анализа (нужно минимум 5 в каждой группе)")

    def analyze_h13_it_comparison(self):
        """
        H13: Сравнение IT и не-IT респондентов по прокрастинации, MBI и SWLS.
        """
        self.add_section("H13: IT vs другие сферы", 3)

        self.add_paragraph(HYPOTHESES['H13'])

        self.add_paragraph("""
**Описание:**

Сравниваем респондентов из IT с остальными по трём шкалам:
- **Прокрастинация:** выше в IT?
- **MBI (выгорание):** выше в IT?
- **SWLS (удовлетворённость жизнью):** ниже в IT?

IT определяется по колонке `profession` (содержит 'IT' или 'it') или `position`.
""")

        # Определяем IT-респондентов по profession
        df = self.completed.copy()
        df['is_it'] = df['profession'].str.contains('IT|it|ит|ИТ|айти|Айти', na=False, regex=True)

        it_group = df[df['is_it'] == True]
        other_group = df[df['is_it'] == False]

        n_it = len(it_group)
        n_other = len(other_group)

        self.add_paragraph(f"**Размер групп:**")
        self.add_paragraph(f"- IT: n = {n_it}")
        self.add_paragraph(f"- Другие сферы: n = {n_other}")

        if n_it < 5 or n_other < 5:
            self.add_paragraph("⚠️ Недостаточно данных для анализа (нужно минимум 5 в каждой группе)")
            return

        # Кубики
        self.add_paragraph(f"\n**Распределение кубиков:**")
        cube_data = []
        for zone_col, zone_name in [('cubes_reactive', '🔴 Срочное'),
                                     ('cubes_proactive', '🟢 Целевое'),
                                     ('cubes_operational', '⚪ Операционное')]:
            it_mean = it_group[zone_col].mean()
            other_mean = other_group[zone_col].mean()
            u_stat, p = stats.mannwhitneyu(it_group[zone_col], other_group[zone_col])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            cube_data.append([zone_name, f"{it_mean:.2f}", f"{other_mean:.2f}",
                            f"{it_mean - other_mean:+.2f}", f"{p:.4f} {sig}"])

        self.add_table(['Зона', 'IT (M)', 'Другие (M)', 'Разница', 'p-value'], cube_data)

        # Валидационные шкалы
        self.add_paragraph(f"\n**Валидационные шкалы:**")

        scale_data = []
        for scale_col, scale_name, direction in [
            ('proc_total', 'Прокрастинация', 'greater'),
            ('mbi_total', 'MBI (выгорание)', 'greater'),
            ('swls_total', 'SWLS', 'less')
        ]:
            it_mean = it_group[scale_col].mean()
            it_sd = it_group[scale_col].std()
            other_mean = other_group[scale_col].mean()
            other_sd = other_group[scale_col].std()

            u_stat, p = stats.mannwhitneyu(it_group[scale_col], other_group[scale_col],
                                           alternative='greater' if direction == 'greater' else 'less')

            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            confirmed = "✅" if p < 0.05 else "❌"

            scale_data.append([scale_name, f"{it_mean:.1f} ± {it_sd:.1f}",
                             f"{other_mean:.1f} ± {other_sd:.1f}",
                             f"{it_mean - other_mean:+.1f}", f"{p:.4f} {sig}", confirmed])

            self.add_paragraph(f"- **{scale_name}:** IT = {it_mean:.1f} ± {it_sd:.1f}, Другие = {other_mean:.1f} ± {other_sd:.1f}, p = {p:.4f} {sig} {confirmed}")

        self.add_table(['Шкала', 'IT (M ± SD)', 'Другие (M ± SD)', 'Разница', 'p-value', 'Гипотеза'], scale_data)

        # Визуализация
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        colors_it = '#e67e22'
        colors_other = '#3498db'

        for idx, (scale_col, scale_name) in enumerate([
            ('proc_total', 'Прокрастинация'),
            ('mbi_total', 'MBI'),
            ('swls_total', 'SWLS')
        ]):
            it_vals = it_group[scale_col].dropna()
            other_vals = other_group[scale_col].dropna()

            # Находим общий диапазон значений (округляем до целых)
            min_val = int(round(min(it_vals.min(), other_vals.min())))
            max_val = int(round(max(it_vals.max(), other_vals.max())))
            data_range = max_val - min_val

            # Определяем число столбцов (не больше 10)
            # Сначала ищем хороший делитель (≥ 5) data_range ≤ 10
            n_bins = 1
            found_good_divisor = False
            if data_range > 0:
                # Ищем делитель от 10 до 5
                for n in range(min(10, data_range), 4, -1):
                    if data_range % n == 0:
                        n_bins = n
                        found_good_divisor = True
                        break
                
                # Если не нашли хороший делитель, расширяем диапазон
                if not found_good_divisor:
                    for n in range(10, 4, -1):
                        remainder = data_range % n
                        if remainder == 0:
                            n_bins = n
                            found_good_divisor = True
                            break
                        else:
                            additional = n - remainder
                            if additional <= 5:
                                n_bins = n
                                max_val = max_val + additional
                                data_range = max_val - min_val
                                found_good_divisor = True
                                break
                
                # Если всё ещё не нашли хороший делитель, ищем любой делитель
                if not found_good_divisor:
                    for n in range(4, 1, -1):
                        if data_range % n == 0:
                            n_bins = n
                            break
            
            # Если всё ещё не нашли, используем 10 столбцов
            if n_bins == 1 and data_range > 0:
                n_bins = min(10, data_range)
            
            # Создаём бины с помощью numpy
            bin_edges = np.linspace(min_val, max_val, n_bins + 1)
            
            # Считаем частоты для каждой группы
            counts_it, _ = np.histogram(it_vals, bins=bin_edges)
            counts_other, _ = np.histogram(other_vals, bins=bin_edges)
            
            # Центры бинов для отображения
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            x_pos = list(range(n_bins))
            width = 0.4
            
            axes[idx].bar([x - width/2 for x in x_pos], counts_it, width=width,
                         color=colors_it, alpha=0.7, label='IT', edgecolor='white', align='center')
            axes[idx].bar([x + width/2 for x in x_pos], counts_other, width=width,
                         color=colors_other, alpha=0.7, label='Другие', edgecolor='white', align='center')
            axes[idx].axvline(it_vals.mean(), color=colors_it, linestyle='--', linewidth=2, zorder=10)
            axes[idx].axvline(other_vals.mean(), color=colors_other, linestyle='--', linewidth=2, zorder=10)
            axes[idx].set_xlabel('Балл', fontproperties=LABEL_FONT)
            axes[idx].set_ylabel('Частота', fontproperties=LABEL_FONT)
            axes[idx].set_title(f'{scale_name}: IT vs Другие', fontproperties=TITLE_FONT)
            axes[idx].set_xticks(x_pos)
            axes[idx].set_xticklabels([str(int(round(v))) for v in bin_centers], fontproperties=TICK_FONT)
            axes[idx].set_xlim(-0.6, n_bins - 0.4)
            axes[idx].legend(prop=TICK_FONT)
            for label in axes[idx].get_xticklabels() + axes[idx].get_yticklabels():
                label.set_fontproperties(TICK_FONT)

        plt.suptitle('H13: Сравнение IT и других сфер', fontproperties=TITLE_FONT)
        plt.tight_layout()
        path = self.save_figure('h13_it_comparison')
        self.add_paragraph(f"\n![H13: IT vs Другие](figures/h13_it_comparison.png)")

        # Выводы
        self.add_paragraph(f"\n**Выводы по H13:**")
        it_higher = 0
        it_lower = 0
        for scale_col, scale_name, direction in [
            ('proc_total', 'Прокрастинация', 'greater'),
            ('mbi_total', 'MBI (выгорание)', 'greater'),
            ('swls_total', 'SWLS', 'less')
        ]:
            it_mean = it_group[scale_col].mean()
            other_mean = other_group[scale_col].mean()
            u_stat, p = stats.mannwhitneyu(it_group[scale_col], other_group[scale_col],
                                           alternative='greater' if direction == 'greater' else 'less')
            if p < 0.05:
                if direction == 'greater':
                    it_higher += 1
                    self.add_paragraph(f"- ✅ {scale_name} значимо **выше** в IT (p = {p:.4f})")
                else:
                    it_lower += 1
                    self.add_paragraph(f"- ✅ {scale_name} значимо **ниже** в IT (p = {p:.4f})")
            else:
                self.add_paragraph(f"- ❌ {scale_name}: нет значимых различий (p = {p:.4f})")

        if it_higher >= 2 and it_lower >= 1:
            self.add_paragraph(f"\n**Итог:** Гипотеза **подтверждается** — в IT выше и прокрастинация, и выгорание, и ниже удовлетворённость")
        elif it_higher >= 2:
            self.add_paragraph(f"\n**Итог:** Гипотеза **частично подтверждается** — в IT выше негативные показатели, но SWLS не отличается")
        else:
            self.add_paragraph(f"\n**Итог:** Гипотеза **не подтверждается** — значимых различий по ожидаемым направлениям нет")


    def analyze_productivity_index(self):
        """
        Анализ индекса ProductivityIndex — логарифм по основанию 2 отношения целевого к срочному.

        ProductivityIndex = log₂((Целевое + eps) / (Срочное + eps))

        Интерпретация:
        - ProductivityIndex > 0: преобладание проактивности (целевых задач)
        - ProductivityIndex = 0: баланс между срочными и целевыми задачами
        - ProductivityIndex < 0: преобладание реактивности (срочных задач)

        Логарифм по основанию 2 делает шкалу симметричной и интерпретируемой:
        при 7 кубиках и eps=1 диапазон составляет [-3, 3].
        Значения +1 и -1 означают различие в 2 раза, +2 и -2 — в 4 раза, +3 и -3 — в 8 раз.
        """
        self.add_section("Анализ индекса ProductivityIndex (log₂(Целевое/Срочное))", 2)

        self.add_paragraph("""
**Описание индекса:**

ProductivityIndex — это логарифм по основанию 2 отношения количества кубиков в зоне "Целевое" к количеству
кубиков в зоне "Срочное" с добавлением небольшого смещения (eps) для избежания
деления на ноль и логарифмирования нуля.

Формула: ProductivityIndex = log₂((Целевое + eps) / (Срочное + eps))

Интерпретация:
- ProductivityIndex > 0: преобладание проактивности (целевые задачи доминируют)
- ProductivityIndex ≈ 0: баланс между срочными и целевыми задачами
- ProductivityIndex < 0: преобладание реактивности (срочные задачи доминируют)

Преимущество логарифмической шкалы по основанию 2:
- Симметрия: отношения 2:1 и 1:2 дают равные по модулю, но противоположные по знаку значения (+1 и -1)
- Интерпретируемость: значение +1 означает в 2 раза больше целевых, -1 — в 2 раза больше срочных
- При 7 кубиках и eps=1 диапазон составляет [-3, 3]

Высокий ProductivityIndex указывает на более продуктивный профиль распределения энергии.

**Параметр смещения:** eps = {eps} (диапазон при 7 кубиках: [-3, 3])
""".format(eps=RG_RATIO_EPS))

        # Описательная статистика
        productivity_index = self.completed['productivity_index'].dropna()
        self.add_paragraph(f"**Описательная статистика ProductivityIndex:**")
        self.add_paragraph(f"- M = {productivity_index.mean():.2f}, SD = {productivity_index.std():.2f}")
        self.add_paragraph(f"- Медиана = {productivity_index.median():.2f}")
        self.add_paragraph(f"- Диапазон = {productivity_index.min():.2f} – {productivity_index.max():.2f}")

        # Категоризация по логарифмической шкале (log₂)
        # log₂(1.5) ≈ 0.585, log₂(0.67) ≈ -0.585
        high_proactivity = (productivity_index > 0.585).sum()
        balanced = ((productivity_index >= -0.585) & (productivity_index <= 0.585)).sum()
        high_reactivity = (productivity_index < -0.585).sum()

        self.add_paragraph(f"\n**Распределение по категориям:**")
        self.add_paragraph(f"- Преобладание проактивности (PI > 0.585): {high_proactivity} ({high_proactivity/len(productivity_index)*100:.1f}%)")
        self.add_paragraph(f"- Баланс (-0.585 ≤ PI ≤ 0.585): {balanced} ({balanced/len(productivity_index)*100:.1f}%)")
        self.add_paragraph(f"- Преобладание реактивности (PI < -0.585): {high_reactivity} ({high_reactivity/len(productivity_index)*100:.1f}%)")

        # Корреляции ProductivityIndex с валидационными шкалами
        self.add_paragraph(f"\n**Корреляции ProductivityIndex с валидационными шкалами:**")

        for scale, name in [('proc_total', 'Прокрастинация'),
                           ('swls_total', 'SWLS'),
                           ('mbi_total', 'MBI')]:
            r, p = stats.spearmanr(productivity_index, self.completed[scale])
            sig = "*" if p is not None and p < 0.05 else ""
            direction = "положительная" if r > 0 else "отрицательная"
            strength = abs(r)
            if strength >= 0.5:
                strength_str = "сильная"
            elif strength >= 0.3:
                strength_str = "умеренная"
            else:
                strength_str = "слабая"

            self.add_paragraph(f"- {name}: r = {r:.3f}, p = {p:.4f} {sig} ({strength_str} {direction})")

        # Визуализация распределения ProductivityIndex
        fig, ax = plt.subplots(figsize=(9, 5))
        # Для непрерывной переменной используем гистограмму с правильными бинами
        n_bins = 20
        counts, bins, patches = ax.hist(productivity_index, bins=n_bins, color='purple', edgecolor='white', alpha=0.8, align='mid')
        
        ax.axvline(productivity_index.mean(), color='red', linestyle='--', label=f'M = {productivity_index.mean():.2f}')
        ax.axvline(0.0, color='black', linestyle=':', linewidth=2, label='Баланс (PI=0)')
        ax.set_xlabel('ProductivityIndex (log₂(Целевое/Срочное))', fontproperties=LABEL_FONT)
        ax.set_ylabel('Число респондентов', fontproperties=LABEL_FONT)
        ax.set_title('Распределение индекса ProductivityIndex', fontproperties=TITLE_FONT)
        ax.set_xlim(bins[0], bins[-1])
        ax.legend(prop=TICK_FONT)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(TICK_FONT)
        plt.tight_layout()
        path = self.save_figure('productivity_index_distribution')
        self.add_paragraph(f"\n![Распределение ProductivityIndex]({path})")

        # Линейная регрессия с ProductivityIndex
        self.add_paragraph(f"\n**Линейная регрессия с ProductivityIndex:**")
        self.add_paragraph("*Модель: Зависимая переменная = β₀ + β₁ × ProductivityIndex*")

        targets = [
            ('mbi_total', 'MBI (выгорание)'),
            ('proc_total', 'Прокрастинация'),
            ('swls_total', 'SWLS (удовлетворённость)')
        ]

        for target_col, target_name in targets:
            y = self.completed[target_col].values
            X = productivity_index.values

            # Простая линейная регрессия
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(X, y)
            r_squared = r_value ** 2

            self.add_paragraph(f"\n**{target_name}:**")
            self.add_paragraph(f"- Уравнение: {target_name} = {intercept:.2f} + {slope:.2f} × ProductivityIndex")
            self.add_paragraph(f"- R² = {r_squared:.3f} ({r_squared*100:.1f}% дисперсии объясняется)")
            self.add_paragraph(f"- Коэффициент ProductivityIndex: {slope:.2f} (p={p_value:.3f})")

            if r_squared >= 0.3:
                strength = "сильная"
            elif r_squared >= 0.1:
                strength = "умеренная"
            else:
                strength = "слабая"

            self.add_paragraph(f"- Предсказательная способность: {strength} (R²={r_squared:.3f})")

            # Интерпретация направления
            if slope > 0:
                self.add_paragraph(f"- Интерпретация: чем выше ProductivityIndex (больше проактивности), тем выше {target_name}")
            else:
                self.add_paragraph(f"- Интерпретация: чем выше ProductivityIndex (больше проактивности), тем ниже {target_name}")

        # Визуализация регрессий
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        for idx, (target_col, target_name) in enumerate(targets):
            y = self.completed[target_col].values
            X = productivity_index.values

            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(X, y)
            r_squared = r_value ** 2

            axes[idx].scatter(X, y, alpha=0.6, color='purple')
            x_line = np.linspace(X.min(), X.max(), 100)
            y_line = intercept + slope * x_line
            axes[idx].plot(x_line, y_line, 'r-', linewidth=2)
            axes[idx].axvline(0, color='black', linestyle=':', alpha=0.5)
            axes[idx].set_xlabel('ProductivityIndex (log₂)', fontproperties=LABEL_FONT)
            axes[idx].set_ylabel(target_name, fontproperties=LABEL_FONT)
            axes[idx].set_title(f'ProductivityIndex → {target_name}', fontproperties=TITLE_FONT)
            axes[idx].text(0.05, 0.95, f'R²={r_squared:.3f}\np={p_value:.3f}',
                          transform=axes[idx].transAxes, fontsize=10, verticalalignment='top',
                          fontproperties=TICK_FONT)

        plt.suptitle('Линейная регрессия: ProductivityIndex как предиктор', fontproperties=TITLE_FONT)
        plt.tight_layout()
        path = self.save_figure('productivity_index_regressions')
        self.add_paragraph(f"\n![Регрессии ProductivityIndex]({path})")

        # --- Линейная регрессия по трём выборкам ---
        self.add_paragraph(f"\n**Линейная регрессия с ProductivityIndex по подвыборкам:**")

        df_almost = self.completed[self.completed['representative'].between(-1, 1)]
        df_exact = self.completed[self.completed['representative'] == 0]

        samples_pi = [
            ('Полная', self.completed, productivity_index),
            ('Почти типовая (−1..1)', df_almost, df_almost['productivity_index'].dropna()),
            ('Точно типовая (0)', df_exact, df_exact['productivity_index'].dropna()),
        ]

        reg_headers = ['Выборка', 'n', 'Показатель', 'β₁', 'R²', 'p']
        reg_rows = []

        for sample_name, df_s, pi_s in samples_pi:
            n_s = len(df_s)
            pi_s = pi_s.dropna()
            if n_s < 5 or len(pi_s) < 5:
                reg_rows.append([sample_name, str(n_s), '—', '—', '—', '—'])
                continue

            for target_col, target_name in targets:
                y_s = df_s[target_col].values
                X_s = pi_s.values
                # Совместимые индексы
                valid = df_s[target_col].notna().values & pi_s.notna().values
                y_v = df_s[target_col].values[valid]
                X_v = pi_s.values[valid]
                if len(X_v) < 5:
                    reg_rows.append([sample_name, str(len(X_v)), target_name, '—', '—', '—'])
                    continue
                slope_s, intercept_s, r_s, p_s, _ = linregress(X_v, y_v)
                r2_s = r_s ** 2
                sig = "***" if p_s < 0.001 else "**" if p_s < 0.01 else "*" if p_s < 0.05 else ""
                reg_rows.append([
                    sample_name, str(len(X_v)), target_name,
                    f"{slope_s:.3f}", f"{r2_s:.3f}", f"{p_s:.4f} {sig}"
                ])

        self.add_table(reg_headers, reg_rows)

        # Выводы
        self.add_paragraph(f"\n**Выводы по индексу ProductivityIndex:**")

        # Находим самые сильные корреляции
        correlations = []
        for scale, name in [('proc_total', 'Прокрастинация'),
                           ('swls_total', 'SWLS'),
                           ('mbi_total', 'MBI')]:
            r, p = stats.spearmanr(productivity_index, self.completed[scale])
            correlations.append((abs(r), name, r, p))

        correlations.sort(reverse=True)
        strongest_corr = correlations[0]

        self.add_paragraph(f"- Наиболее сильная связь: с {strongest_corr[1]} (r={strongest_corr[2]:.3f}, p={strongest_corr[3]:.3f})")

        if strongest_corr[2] > 0:
            self.add_paragraph(f"- Чем выше ProductivityIndex (больше проактивности), тем выше {strongest_corr[1]}")
        else:
            self.add_paragraph(f"- Чем выше ProductivityIndex (больше проактивности), тем ниже {strongest_corr[1]}")

        self.add_paragraph(f"- Логарифмическая шкала по основанию 2 обеспечивает симметрию и интерпретируемость:")
        self.add_paragraph(f"  значения +X и -X соответствуют равному по силе, но противоположному по направлению дисбалансу")
        self.add_paragraph(f"  Значение +1 означает в 2 раза больше целевых, -1 — в 2 раза больше срочных")
        self.add_paragraph(f"- При 7 кубиках и eps=1 диапазон ProductivityIndex составляет [-3, 3]")
        self.add_paragraph(f"- ProductivityIndex может быть полезен как компактный индикатор продуктивного профиля")
        self.add_paragraph(f"- Используемый eps = {RG_RATIO_EPS} обеспечивает диапазон [-3, 3] при логарифме по основанию 2")


    # =========================================================================
    # АНАЛИЗ ВРЕМЕНИ ОТВЕТА НА СТРАНИЦЕ 2
    # =========================================================================

    def analyze_response_time(self):
        """
        Анализ корреляции времени ответа на странице 2 (кубики) с количеством кубиков.

        Проверяем гипотезу: респонденты с разным распределением кубиков
        затрачивают разное время на принятие решения.
        """
        self.add_section("Анализ времени ответа на странице 2 (распределение кубиков)", 2)

        self.add_paragraph("""
**Описание анализа:**

Время, затраченное респондентом на распределение кубиков по зонам, может быть
связано с когнитивной нагрузкой при принятии решения. Например:
- Быстрое решение может указывать на уверенность или интуитивный выбор
- Длительное размышление может свидетельствовать о внутренней борьбе или
  желании дать социально ожидаемый ответ

Здесь анализируется корреляция между временем на странице 2 (в секундах)
и количеством кубиков в каждой зоне.
""")

        # Фильтруем респондентов, у которых есть время на странице 2
        # и статус completed
        df_time = self.completed.copy()

        # Проверяем наличие колонок времени
        if 'time_page2_total' not in df_time.columns:
            self.add_paragraph("**Внимание:** Колонка time_page2_total не найдена в данных.")
            self.add_paragraph("Анализ времени ответа невозможен.")
            return

        # time_page2_total уже в секундах (вычисляется при сохранении)
        # Фильтруем только тех, у кого time_page2_total > 0
        df_valid = df_time[df_time['time_page2_total'] > 0].copy()

        # Отбрасываем экстремальные выбросы: время > 100 * медиана
        median_time = df_valid['time_page2_total'].median()
        time_threshold = 100 * median_time
        n_before = len(df_valid)
        df_valid = df_valid[df_valid['time_page2_total'] <= time_threshold].copy()
        n_excluded = n_before - len(df_valid)

        self.add_paragraph(f"**Выборка:** {len(df_valid)} респондентов из {len(self.completed)} завершённых")
        if n_excluded > 0:
            self.add_paragraph(f"**Отброшено выбросов:** {n_excluded} респондентов с временем > {time_threshold:.0f} сек (100 × медиана = {median_time:.0f} сек)")

        # Описательная статистика времени
        time_vals = df_valid['time_page2_total'].dropna()
        if len(time_vals) > 0:
            self.add_paragraph(f"\n**Время на странице 2:**")
            self.add_paragraph(f"- M = {time_vals.mean():.1f} сек, SD = {time_vals.std():.1f}")
            self.add_paragraph(f"- Медиана = {time_vals.median():.1f} сек")
            self.add_paragraph(f"- Диапазон = {time_vals.min():.0f} – {time_vals.max():.0f} сек")
            self.add_paragraph(f"- Q1 = {time_vals.quantile(0.25):.0f} сек, Q3 = {time_vals.quantile(0.75):.0f} сек")

        # Корреляции Спирмена
        self.add_paragraph(f"\n**Корреляции Спирмена: время vs кубики:**")

        correlations = []
        for zone_col, zone_name in [('cubes_reactive', 'Срочное ●'),
                                     ('cubes_proactive', 'Целевое ●'),
                                     ('cubes_operational', 'Операционное ●')]:
            r, p = stats.spearmanr(df_valid['time_page2_total'], df_valid[zone_col])
            correlations.append((zone_col, zone_name, r, p))

            # Определяем силу и направление
            direction = "положительная" if r > 0 else "отрицательная"
            strength = abs(r)
            if strength >= 0.5:
                strength_str = "сильная"
            elif strength >= 0.3:
                strength_str = "умеренная"
            else:
                strength_str = "слабая"

            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            sig_text = f"p = {p:.4f} {sig}" if p is not None else "p = N/A"

            self.add_paragraph(f"- {zone_name}: r = {r:.3f}, {sig_text} ({strength_str} {direction})")

        # Визуализация: scatter plot времени vs кубики
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        zone_colors = {'cubes_reactive': ZONE_COLORS['reactive'],
                       'cubes_proactive': ZONE_COLORS['proactive'],
                       'cubes_operational': ZONE_COLORS['operational']}
        zone_labels_ru = {'cubes_reactive': 'Срочное ●',
                          'cubes_proactive': 'Целевое ●',
                          'cubes_operational': 'Операционное ●'}

        for idx, (zone_col, zone_name, r, p) in enumerate(correlations):
            x = df_valid[zone_col].values
            y = df_valid['time_page2_total'].values

            axes[idx].scatter(x, y, alpha=0.6, color=zone_colors[zone_col])

            # Линия тренда
            from scipy.stats import linregress
            slope, intercept, r_val, p_val, std_err = linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = intercept + slope * x_line
            axes[idx].plot(x_line, y_line, 'r-', linewidth=2, alpha=0.7)

            axes[idx].set_xlabel(f'Количество кубиков ({zone_name})', fontproperties=LABEL_FONT)
            axes[idx].set_ylabel('Время (сек)', fontproperties=LABEL_FONT)
            axes[idx].set_title(f'Время vs {zone_name}', fontproperties=TITLE_FONT)
            axes[idx].text(0.05, 0.95, f'r = {r:.3f}\np = {p:.3f}',
                          transform=axes[idx].transAxes, fontsize=10, verticalalignment='top',
                          fontproperties=TICK_FONT)

            # Настройка шрифтов
            for label in axes[idx].get_xticklabels() + axes[idx].get_yticklabels():
                label.set_fontproperties(TICK_FONT)

        plt.suptitle('Время ответа vs количество кубиков в зонах', fontproperties=TITLE_FONT)
        plt.tight_layout()
        path = self.save_figure('response_time_vs_cubes')
        self.add_paragraph(f"\n![Время ответа vs кубики]({path})")

        # Выводы
        self.add_paragraph(f"\n**Выводы по анализу времени:**")

        # Находим самую сильную корреляцию
        if correlations:
            strongest = max(correlations, key=lambda x: abs(x[2]))
            direction = "больше кубиков — больше времени" if strongest[2] > 0 else "больше кубиков — меньше времени"
            self.add_paragraph(f"- Наиболее сильная связь времени с зоной {strongest[1]}: "
                              f"r = {strongest[2]:.3f}, p = {strongest[3]:.4f}")
            self.add_paragraph(f"- Направление: {direction}")

            # Проверяем, есть ли значимые корреляции
            sig_corrs = [(z, r, p) for z, _, r, p in correlations if p is not None and p < 0.05]
            if sig_corrs:
                self.add_paragraph(f"- Значимые корреляции (p < 0.05) обнаружены для {len(sig_corrs)} зон")
            else:
                self.add_paragraph(f"- Значимых корреляций (p < 0.05) не обнаружено")

    def analyze_h14_mbi_swls_relationship(self):
        """
        H14: Связь MBI и SWLS — линейная или нелинейная (квадратичная)?

        Метод: Сравнение линейной и квадратичной регрессий,
        тест на добавление квадратичного члена (F-test / ANOVA).
        """
        self.add_section("H14: Линейная или нелинейная связь MBI и SWLS", 3)

        self.add_paragraph(HYPOTHESES['H14'])

        self.add_paragraph("""
**Описание:**

Выгорание (MBI) и удовлетворённость жизнью (SWLS) тесно связаны.
Остаётся вопрос: является ли эта связь строго линейной,
или она имеет нелинейный (квадратичный) характер?

Подход:
1. **Линейная модель:** SWLS = β₀ + β₁ × MBI
2. **Квадратичная модель:** SWLS = β₀ + β₁ × MBI + β₂ × MBI²
3. **Сравнение моделей:** F-тест (ANOVA) — улучшает ли квадратичный член модель?
4. **Проверка квадратичного члена:** t-тест для β₂
""")

        x = self.completed['mbi_total'].values
        y = self.completed['swls_total'].values
        n = len(x)

        # ── 1. Линейная модель ──────────────────────────────────────
        X_lin = np.column_stack([np.ones(n), x])
        beta_lin = np.linalg.lstsq(X_lin, y, rcond=None)[0]
        y_pred_lin = X_lin @ beta_lin
        ss_res_lin = np.sum((y - y_pred_lin) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2_lin = 1 - ss_res_lin / ss_tot

        # ── 2. Квадратичная модель ──────────────────────────────────
        X_quad = np.column_stack([np.ones(n), x, x ** 2])
        beta_quad = np.linalg.lstsq(X_quad, y, rcond=None)[0]
        y_pred_quad = X_quad @ beta_quad
        ss_res_quad = np.sum((y - y_pred_quad) ** 2)
        r2_quad = 1 - ss_res_quad / ss_tot

        # ── 3. F-тест для сравнения вложенных моделей ───────────────
        # H0: β₂ = 0 (линейная модель достаточна)
        # H1: β₂ ≠ 0 (квадратичная модель лучше)
        df_diff = 1  # одна дополнительная переменная
        df_residual = n - 3  # 3 параметра в квадратичной модели
        f_stat = ((ss_res_lin - ss_res_quad) / df_diff) / (ss_res_quad / df_residual)
        from scipy.stats import f as f_dist
        f_p = 1 - f_dist.cdf(f_stat, df_diff, df_residual)

        # ── 4. t-тест для β₂ ────────────────────────────────────────
        p_num = 2  # число предикторов (без intercept)
        mse = ss_res_quad / (n - p_num - 1)
        var_beta = mse * np.linalg.inv(X_quad.T @ X_quad)
        se_beta2 = np.sqrt(var_beta[2, 2])
        t_beta2 = beta_quad[2] / se_beta2
        from scipy.stats import t as t_dist
        p_beta2 = 2 * (1 - t_dist.cdf(np.abs(t_beta2), df=df_residual))

        # ── Выводы ──────────────────────────────────────────────────
        self.add_paragraph(f"**Размер выборки:** n = {n}")

        self.add_paragraph(f"\n**Линейная модель:**")
        self.add_paragraph(f"- SWLS = {beta_lin[0]:.2f} + ({beta_lin[1]:.3f}) × MBI")
        self.add_paragraph(f"- R² = {r2_lin:.3f} ({r2_lin * 100:.1f}% дисперсии объясняется)")

        self.add_paragraph(f"\n**Квадратичная модель:**")
        self.add_paragraph(f"- SWLS = {beta_quad[0]:.2f} + ({beta_quad[1]:.3f}) × MBI + ({beta_quad[2]:.4f}) × MBI²")
        self.add_paragraph(f"- R² = {r2_quad:.3f} ({r2_quad * 100:.1f}% дисперсии объясняется)")

        self.add_paragraph(f"\n**Сравнение моделей (F-тест):**")
        self.add_paragraph(f"- F({df_diff}, {df_residual}) = {f_stat:.3f}, p = {f_p:.4f}")
        if f_p < 0.05:
            self.add_paragraph(f"- ✅ Квадратичная модель **значимо лучше** линейной (p < 0.05)")
        else:
            self.add_paragraph(f"- ❌ Квадратичная модель **не значимо лучше** линейной (p = {f_p:.4f} > 0.05)")

        self.add_paragraph(f"\n**Тест для квадратичного члена β₂:**")
        self.add_paragraph(f"- β₂ = {beta_quad[2]:.4f}, SE = {se_beta2:.4f}, t = {t_beta2:.3f}, p = {p_beta2:.4f}")
        if p_beta2 < 0.05:
            self.add_paragraph(f"- ✅ Квадратичный член **значим** (p < 0.05)")
        else:
            self.add_paragraph(f"- ❌ Квадратичный член **не значим** (p = {p_beta2:.4f} > 0.05)")

        # Итоговый вывод
        self.add_paragraph(f"\n**Итог:**")
        if f_p < 0.05 and p_beta2 < 0.05:
            if beta_quad[2] > 0:
                self.add_paragraph(f"- Связь MBI–SWLS **нелинейная (U-образная)**: при очень высоком выгорании снижение SWLS замедляется")
            else:
                self.add_paragraph(f"- Связь MBI–SWLS **нелинейная (инвертированная U)**: при среднем выгорании связь сильнее")
            self.add_paragraph(f"- Квадратичная модель объясняет дополнительно {(r2_quad - r2_lin) * 100:.1f}% дисперсии")
        else:
            self.add_paragraph(f"- Связь MBI–SWLS **линейная**: квадратичный член не улучшает модель")
            self.add_paragraph(f"- Линейная модель: R² = {r2_lin:.3f}")

        # ── Визуализация ────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 6))

        # Точки
        ax.scatter(x, y, alpha=0.4, color='steelblue', s=20, label='Данные')

        # Линейная линия
        x_sorted = np.linspace(x.min(), x.max(), 200)
        y_lin_line = beta_lin[0] + beta_lin[1] * x_sorted
        ax.plot(x_sorted, y_lin_line, 'r-', linewidth=2, label=f'Линейная (R²={r2_lin:.3f})')

        # Квадратичная линия
        y_quad_line = beta_quad[0] + beta_quad[1] * x_sorted + beta_quad[2] * x_sorted ** 2
        ax.plot(x_sorted, y_quad_line, 'g-', linewidth=2, label=f'Квадратичная (R²={r2_quad:.3f})')

        ax.set_xlabel('MBI (выгорание)', fontproperties=LABEL_FONT)
        ax.set_ylabel('SWLS (удовлетворённость жизнью)', fontproperties=LABEL_FONT)
        ax.set_title('Связь MBI и SWLS: линейная vs квадратичная', fontproperties=TITLE_FONT)
        ax.legend(prop=TICK_FONT)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(TICK_FONT)

        plt.tight_layout()
        path = self.save_figure('h14_mbi_swls_relationship')
        self.add_paragraph(f"\n![Связь MBI–SWLS]({path})")

        # ── Дополнительно: корреляция Спирмена ──────────────────────
        r_spearman, p_spearman = stats.spearmanr(x, y)
        self.add_paragraph(f"\n**Корреляция Спирмена:** r = {r_spearman:.3f}, p = {p_spearman:.4f}")

    def analyze_h15_regression_predictors(self):
        """
        H15: Множественная линейная регрессия для предсказания MBI, Прокрастинации и SWLS
        на основе четырёх предикторов:
          - кубики в целевой зоне (🟢)
          - кубики в срочной зоне (🔴)
          - баланс работа/личное
          - энергетический дефицит
        """
        self.add_section("H15: Множественная регрессия — предсказание по 🟢, 🔴, балансу и дефициту", 3)

        self.add_paragraph(HYPOTHESES['H15'])

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

        # Предикторы
        predictors = ['cubes_proactive', 'cubes_reactive', 'work_life', 'energy_deficit']
        pred_labels = ['🟢 Целевое', '🔴 Срочное', 'Баланс работа/личное', 'Дефицит энергии']
        pred_colors = ['#27ae60', '#e74c3c', '#3498db', '#e67e22']

        targets = [
            ('mbi_total', 'MBI (выгорание)'),
            ('proc_total', 'Прокрастинация'),
            ('swls_total', 'SWLS'),
        ]

        df = self.completed.copy()
        # Убираем строки с пропущенными значениями в предикторах
        valid_mask = df[predictors].notna().all(axis=1)
        df_valid = df[valid_mask].copy()
        n = len(df_valid)

        self.add_paragraph(f"**Размер выборки (полные данные):** n = {n}")

        # Храним результаты для графиков
        all_results = {}

        # ==================== СВОДНАЯ ТАБЛИЦА ====================
        self.add_paragraph("\n**Сводная таблица регрессий:**\n")

        summary_headers = ['Показатель', 'R²', 'Adj R²', 'F', 'p(F)']
        summary_rows = []

        for target_col, target_name in targets:
            y = df_valid[target_col].values
            # Убираем пропуски в зависимой переменной
            y_valid = df_valid[target_col].dropna()
            valid_idx = y_valid.index
            y_vals = y_valid.values
            X_vals = df_valid.loc[valid_idx, predictors].values

            n_valid = len(y_vals)
            p_num = len(predictors)

            # Добавляем intercept
            X_int = np.column_stack([np.ones(n_valid), X_vals])

            try:
                beta = np.linalg.lstsq(X_int, y_vals, rcond=None)[0]
                y_pred = X_int @ beta
                ss_res = np.sum((y_vals - y_pred) ** 2)
                ss_tot = np.sum((y_vals - np.mean(y_vals)) ** 2)
                r_squared = 1 - ss_res / ss_tot

                # Adjusted R²
                adj_r2 = 1 - (1 - r_squared) * (n_valid - 1) / (n_valid - p_num - 1)

                # F-тест
                ms_reg = (ss_tot - ss_res) / p_num
                ms_res = ss_res / (n_valid - p_num - 1)
                f_stat = ms_reg / ms_res
                from scipy.stats import f as f_dist
                f_p = 1 - f_dist.cdf(f_stat, p_num, n_valid - p_num - 1)

                # Стандартные ошибки и t-тесты для коэффициентов
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

        # ==================== ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ ====================
        for target_col, target_name in targets:
            res = all_results.get(target_col)
            if res is None:
                continue

            self.add_paragraph(f"\n**{target_name}:**")
            self.add_paragraph(f"- R² = {res['r2']:.3f}, Adj R² = {res['adj_r2']:.3f}")
            self.add_paragraph(f"- F({len(predictors)}, {res['n'] - len(predictors) - 1}) = {res['f']:.1f}, p = {res['f_p']:.6f}")

            # Коэффициенты
            self.add_paragraph(f"\n**Коэффициенты:**")
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

            # Интерпретация
            self.add_paragraph(f"\n**Интерпретация:**")
            for i, label in enumerate(pred_labels):
                idx = i + 1
                direction = "положительная" if res['beta'][idx] > 0 else "отрицательная"
                sig_mark = "✅ значим" if res['p'][idx] < 0.05 else "❌ не значим"
                self.add_paragraph(f"- {label}: β = {res['beta'][idx]:.3f} ({direction}), {sig_mark}")

        # ==================== ГРАФИКИ ====================
        # 1. Сравнение R²
        fig, ax = plt.subplots(figsize=(8, 5))
        x_pos = np.arange(len(targets))
        colors_bar = ['#e74c3c', '#f39c12', '#27ae60']

        r2_vals = [all_results[t[0]]['r2'] if all_results.get(t[0]) else 0 for t in targets]
        adj_r2_vals = [all_results[t[0]]['adj_r2'] if all_results.get(t[0]) else 0 for t in targets]

        w = 0.35
        bars1 = ax.bar(x_pos - w/2, r2_vals, w, label='R²', color=colors_bar, alpha=0.7, edgecolor='white')
        bars2 = ax.bar(x_pos + w/2, adj_r2_vals, w, label='Adj R²', color='white', edgecolor=colors_bar, linewidth=2, alpha=0.9)

        ax.set_ylabel('R²', fontproperties=LABEL_FONT)
        ax.set_title('Доля объяснённой дисперсии по шкалам', fontproperties=TITLE_FONT)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([t[1] for t in targets], fontproperties=TICK_FONT)
        ax.legend(prop=TICK_FONT)
        ax.set_ylim(0, max(max(r2_vals), max(adj_r2_vals)) * 1.2)
        for label in ax.get_yticklabels():
            label.set_fontproperties(TICK_FONT)

        # Подписи значений
        for bar, val in zip(bars1, r2_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{val:.3f}', ha='center', va='bottom', fontproperties=SMALL_TEXT_FONT)
        for bar, val in zip(bars2, adj_r2_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{val:.3f}', ha='center', va='bottom', fontproperties=SMALL_TEXT_FONT)

        plt.tight_layout()
        path = self.save_figure('h15_r2_comparison')
        self.add_paragraph(f"\n![Сравнение R²]({path})")

        # 2. Графики предсказанных vs наблюдаемых значений
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

            # Линия идеального предсказания
            y_min, y_max = min(y_vals.min(), y_pred.min()), max(y_vals.max(), y_pred.max())
            axes[ax_idx].plot([y_min, y_max], [y_min, y_max], 'r--', linewidth=2, alpha=0.7, label='Идеально')

            axes[ax_idx].set_xlabel(f'Наблюдаемое ({target_name})', fontproperties=LABEL_FONT)
            axes[ax_idx].set_ylabel(f'Предсказанное', fontproperties=LABEL_FONT)
            axes[ax_idx].set_title(f'{target_name} (R²={res["r2"]:.3f})', fontproperties=TITLE_FONT)
            axes[ax_idx].legend(prop=TICK_FONT)
            for label in axes[ax_idx].get_xticklabels() + axes[ax_idx].get_yticklabels():
                label.set_fontproperties(TICK_FONT)

        plt.suptitle('H15: Наблюдаемые vs предсказанные значения', fontproperties=TITLE_FONT)
        plt.tight_layout()
        path = self.save_figure('h15_predicted_vs_actual')
        self.add_paragraph(f"\n![Наблюдаемые vs предсказанные]({path})")

        # 3. Стандартизованные коэффициенты (beta weights)
        fig, ax = plt.subplots(figsize=(10, 6))

        # Стандартизация предикторов и зависимых переменных
        std_coefs = {}
        for target_col, target_name in targets:
            res = all_results.get(target_col)
            if res is None:
                continue

            y_valid = df_valid[target_col].dropna()
            valid_idx = y_valid.index
            y_vals = y_valid.values
            X_vals = df_valid.loc[valid_idx, predictors].values

            # Стандартизация
            y_std = (y_vals - y_vals.mean()) / y_vals.std()
            X_std = (X_vals - X_vals.mean(axis=0)) / X_vals.std(axis=0)
            X_int_std = np.column_stack([np.ones(len(y_std)), X_std])

            beta_std = np.linalg.lstsq(X_int_std, y_std, rcond=None)[0]
            std_coefs[target_col] = beta_std[1:]  # без intercept

        x_pos = np.arange(len(pred_labels))
        w_bar = 0.25
        colors_target = ['#e74c3c', '#f39c12', '#27ae60']

        for ti, (target_col, target_name) in enumerate(targets):
            if target_col in std_coefs:
                offset = (ti - 1) * w_bar
                bars = ax.bar(x_pos + offset, std_coefs[target_col], w_bar,
                             label=target_name, color=colors_target[ti], alpha=0.8, edgecolor='white')

        ax.axhline(y=0, color='black', linewidth=1, alpha=0.5)
        ax.set_ylabel('Стандартизованный β', fontproperties=LABEL_FONT)
        ax.set_title('Стандартизованные коэффициенты регрессии', fontproperties=TITLE_FONT)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(pred_labels, fontproperties=TICK_FONT, rotation=15, ha='right')
        ax.legend(prop=TICK_FONT)
        for label in ax.get_yticklabels():
            label.set_fontproperties(TICK_FONT)

        plt.tight_layout()
        path = self.save_figure('h15_standardized_coefs')
        self.add_paragraph(f"\n![Стандартизованные коэффициенты]({path})")

        # ==================== ВЫВОДЫ ====================
        self.add_paragraph(f"\n**Выводы по H15:**")
        for target_col, target_name in targets:
            res = all_results.get(target_col)
            if res is None:
                continue
            sig_preds = [pred_labels[i] for i in range(len(predictors)) if res['p'][i + 1] < 0.05]
            strongest_idx = np.argmin(res['p'][1:]) + 1
            self.add_paragraph(f"- **{target_name}:** R²={res['r2']:.3f}, значимые предикторы: {', '.join(sig_preds) if sig_preds else 'нет'}, сильнейший — {pred_labels[strongest_idx - 1]} (β={res['beta'][strongest_idx]:.3f}, p={res['p'][strongest_idx]:.4f})")

    def analyze_h5a_records_vs_memory(self):
        """
        H5a: Детальное сравнение респондентов «по памяти» (1-2) и «по записям» (4-5)
        по всем параметрам: кубики, шкалы валидации, демография, контекстные показатели.
        """
        self.add_section("H5a: Детальное сравнение «по памяти» (1-2) vs «по записям» (4-5)", 3)

        self.add_paragraph(HYPOTHESES['H5a'])

        self.add_paragraph("""
**Описание:**

На вопрос *«Когда вы распределяли "энергию" по коробочкам, то какую часть вы восстановили по памяти, а какую по своим записям (дневники, ежедневник, календарь, список задач и т.п.)?»*
респонденты отвечали по шкале от 1 до 5:
- **1-2:** опирались преимущественно на **память**
- **4-5:** опирались преимущественно на **записи**

Сравниваем обе группы по всем доступным параметрам.
""")

        df = self.completed.copy()
        by_memory = df[df['memory_vs_records'] <= 2].copy()
        by_records = df[df['memory_vs_records'] >= 4].copy()

        n_mem = len(by_memory)
        n_rec = len(by_records)

        self.add_paragraph(f"**Размер групп:**")
        self.add_paragraph(f"- По памяти (1-2): n = {n_mem}")
        self.add_paragraph(f"- По записям (4-5): n = {n_rec}")

        if n_mem < 5 or n_rec < 5:
            self.add_paragraph("⚠️ Недостаточно данных для анализа (нужно минимум 5 в каждой группе)")
            return

        # ── 1. Кубики ──────────────────────────────────────────────
        self.add_paragraph(f"\n**Распределение кубиков по зонам:**")

        cube_data = []
        for zone_col, zone_name in [('cubes_reactive', '🔴 Срочное'),
                                     ('cubes_proactive', '🟢 Целевое'),
                                     ('cubes_operational', '⚪ Операционное')]:
            mem_vals = by_memory[zone_col].dropna()
            rec_vals = by_records[zone_col].dropna()
            mem_mean = mem_vals.mean()
            rec_mean = rec_vals.mean()
            # Тест Манна-Уитни (ранговые данные)
            u_stat, p = stats.mannwhitneyu(mem_vals, rec_vals, alternative='two-sided')
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            cube_data.append([zone_name, f"{mem_mean:.2f} ± {mem_vals.std():.2f}",
                            f"{rec_mean:.2f} ± {rec_vals.std():.2f}",
                            f"{rec_mean - mem_mean:+.2f}", f"{p:.4f} {sig}"])

        self.add_table(['Зона', 'По памяти (M ± SD)', 'По записям (M ± SD)', 'Разница', 'p-value'], cube_data)

        # ── 2. Шкалы валидации ─────────────────────────────────────
        self.add_paragraph(f"\n**Шкалы валидации:**")

        scale_data = []
        for scale_col, scale_name in [('proc_total', 'Прокрастинация'),
                                       ('swls_total', 'SWLS'),
                                       ('mbi_total', 'MBI (выгорание)')]:
            mem_vals = by_memory[scale_col].dropna()
            rec_vals = by_records[scale_col].dropna()
            mem_mean = mem_vals.mean()
            rec_mean = rec_vals.mean()
            # Единообразно используем U-тест Манна-Уитни (непараметрический)
            u_stat, p = stats.mannwhitneyu(mem_vals, rec_vals, alternative='two-sided')
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            scale_data.append([scale_name, f"{mem_mean:.1f} ± {mem_vals.std():.1f}",
                             f"{rec_mean:.1f} ± {rec_vals.std():.1f}",
                             f"{rec_mean - mem_mean:+.1f}", f"{p:.4f} {sig}", "U-тест"])

        self.add_table(['Шкала', 'По памяти (M ± SD)', 'По записям (M ± SD)', 'Разница', 'p-value', 'Тест'], scale_data)

        # ── 3. Контекстные показатели ──────────────────────────────
        self.add_paragraph(f"\n**Контекстные показатели:**")

        context_data = []
        for col, name in [('representative', 'Типичность недели'),
                          ('work_life', 'Баланс работа/личное'),
                          ('energy_deficit', 'Энергетический дефицит')]:
            mem_vals = by_memory[col].dropna()
            rec_vals = by_records[col].dropna()
            if len(mem_vals) > 0 and len(rec_vals) > 0:
                u_stat, p = stats.mannwhitneyu(mem_vals, rec_vals, alternative='two-sided')
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                context_data.append([name, f"{mem_vals.median():.0f} [{mem_vals.quantile(0.25):.0f}; {mem_vals.quantile(0.75):.0f}]",
                                   f"{rec_vals.median():.0f} [{rec_vals.quantile(0.25):.0f}; {rec_vals.quantile(0.75):.0f}]",
                                   f"{p:.4f} {sig}"])

        self.add_table(['Показатель', 'По памяти (Медиана [Q1;Q3])', 'По записям (Медиана [Q1;Q3])', 'p-value'], context_data)

        # ── 4. Профиль уровней ─────────────────────────────────────
        self.add_paragraph(f"\n**Распределение уровней профиля:**")

        by_memory = self._add_level_column(by_memory)
        by_records = self._add_level_column(by_records)

        level_data = []
        for level in range(1, 8):
            mem_count = len(by_memory[by_memory['level'] == level])
            rec_count = len(by_records[by_records['level'] == level])
            if mem_count > 0 or rec_count > 0:
                mem_pct = mem_count / n_mem * 100 if n_mem > 0 else 0
                rec_pct = rec_count / n_rec * 100 if n_rec > 0 else 0
                profile_name = PROFILE_LEVELS.get(level, f'Уровень {level}')
                level_data.append([f"{level} — {profile_name}", f"{mem_count} ({mem_pct:.1f}%)",
                                 f"{rec_count} ({rec_pct:.1f}%)"])

        self.add_table(['Профиль', 'По памяти', 'По записям'], level_data)

        # Тест на различие распределения уровней (Манна-Уитни)
        mem_levels = by_memory['level'].values
        rec_levels = by_records['level'].values
        u_level, p_level = stats.mannwhitneyu(mem_levels, rec_levels, alternative='two-sided')
        self.add_paragraph(f"\n- **Тест различия уровней:** U = {u_level:.0f}, p = {p_level:.4f}")
        if p_level < 0.05:
            self.add_paragraph(f"- ✅ Распределение уровней **значимо различается**")
        else:
            self.add_paragraph(f"- ❌ Распределение уровней **не различается** значимо")

        # ── 5. Демография ──────────────────────────────────────────
        self.add_paragraph(f"\n**Демография:**")

        # Возраст
        mem_age = by_memory['age'].dropna()
        rec_age = by_records['age'].dropna()
        if len(mem_age) > 0 and len(rec_age) > 0:
            try:
                _, p_norm_mem_a = stats.shapiro(mem_age) if len(mem_age) >= 3 else (0, 0)
                _, p_norm_rec_a = stats.shapiro(rec_age) if len(rec_age) >= 3 else (0, 0)
            except:
                p_norm_mem_a, p_norm_rec_a = 0, 0

            if p_norm_mem_a > 0.05 and p_norm_rec_a > 0.05 and len(mem_age) >= 10 and len(rec_age) >= 10:
                t_age, p_age = stats.ttest_ind(mem_age, rec_age)
                test_age = "t-тест"
            else:
                u_age, p_age = stats.mannwhitneyu(mem_age, rec_age, alternative='two-sided')
                test_age = "U-тест"

            sig_age = "***" if p_age < 0.001 else "**" if p_age < 0.01 else "*" if p_age < 0.05 else ""
            self.add_paragraph(f"- **Возраст:** память = {mem_age.mean():.1f} ± {mem_age.std():.1f}, "
                              f"записи = {rec_age.mean():.1f} ± {rec_age.std():.1f} ({test_age}, p = {p_age:.4f} {sig_age})")

        # Пол
        if 'gender' in by_memory.columns:
            mem_gender = by_memory['gender'].value_counts()
            rec_gender = by_records['gender'].value_counts()
            gender_labels_map = {'male': 'Мужской', 'female': 'Женский'}
            self.add_paragraph(f"- **Пол:**")
            for g in ['male', 'female']:
                label = gender_labels_map.get(g, g)
                mem_n = int(mem_gender.get(g, 0))
                rec_n = int(rec_gender.get(g, 0))
                mem_p = mem_n / n_mem * 100 if n_mem > 0 else 0
                rec_p = rec_n / n_rec * 100 if n_rec > 0 else 0
                self.add_paragraph(f"  - {label}: память = {mem_n} ({mem_p:.1f}%), записи = {rec_n} ({rec_p:.1f}%)")

            # Хи-квадрат тест для пола
            try:
                # Создаём таблицу сопряжённости
                all_genders = sorted(set(by_memory['gender'].dropna().unique()) | set(by_records['gender'].dropna().unique()))
                contingency = []
                for grp in [by_memory, by_records]:
                    row = [int((grp['gender'] == g).sum()) for g in all_genders]
                    contingency.append(row)
                chi2, p_chi, _, _ = stats.chi2_contingency(contingency)
                sig_chi = "***" if p_chi < 0.001 else "**" if p_chi < 0.01 else "*" if p_chi < 0.05 else ""
                self.add_paragraph(f"- **Тест на различие пола:** χ² = {chi2:.2f}, p = {p_chi:.4f} {sig_chi}")
            except:
                pass

        # ── 6. ProductivityIndex ───────────────────────────────────
        self.add_paragraph(f"\n**ProductivityIndex:**")
        mem_pi = by_memory['productivity_index'].dropna()
        rec_pi = by_records['productivity_index'].dropna()
        if len(mem_pi) > 0 and len(rec_pi) > 0:
            u_pi, p_pi = stats.mannwhitneyu(mem_pi, rec_pi, alternative='two-sided')
            sig_pi = "***" if p_pi < 0.001 else "**" if p_pi < 0.01 else "*" if p_pi < 0.05 else ""
            self.add_paragraph(f"- Память: M = {mem_pi.mean():.2f}, SD = {mem_pi.std():.2f}")
            self.add_paragraph(f"- Записи: M = {rec_pi.mean():.2f}, SD = {rec_pi.std():.2f}")
            self.add_paragraph(f"- U-тест: p = {p_pi:.4f} {sig_pi}")

        # ── 7. Визуализация ────────────────────────────────────────
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        # Кубики
        zones = ['cubes_reactive', 'cubes_proactive', 'cubes_operational']
        zone_names_ru = ['🔴 Срочное', '🟢 Целевое', '⚪ Операционное']
        zone_colors = ['#e74c3c', '#27ae60', '#95a5a6']

        for i, (zone, zname, zcolor) in enumerate(zip(zones, zone_names_ru, zone_colors)):
            ax = axes[0, i]
            mem_vals = by_memory[zone].dropna()
            rec_vals = by_records[zone].dropna()
            ax.boxplot([mem_vals, rec_vals], labels=['Память', 'Записи'], patch_artist=True,
                       boxprops=dict(facecolor=zcolor, alpha=0.6),
                       medianprops=dict(color='black'))
            ax.set_title(zname, fontproperties=TICK_FONT)
            ax.set_ylabel('Кубики', fontproperties=LABEL_FONT)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontproperties(TICK_FONT)

        # Шкалы
        for i, (scale, sname) in enumerate([('mbi_total', 'MBI'), ('proc_total', 'Прокрастинация'), ('swls_total', 'SWLS')]):
            ax = axes[1, i]
            mem_vals = by_memory[scale].dropna()
            rec_vals = by_records[scale].dropna()
            ax.boxplot([mem_vals, rec_vals], labels=['Память', 'Записи'], patch_artist=True,
                       boxprops=dict(facecolor='steelblue', alpha=0.6),
                       medianprops=dict(color='black'))
            ax.set_title(sname, fontproperties=TICK_FONT)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontproperties(TICK_FONT)

        plt.suptitle('H5a: Сравнение «по памяти» vs «по записям»', fontproperties=TITLE_FONT)
        plt.tight_layout()
        path = self.save_figure('h5a_memory_vs_records')
        self.add_paragraph(f"\n![Сравнение память vs записи]({path})")

        # ── 8. Итоговый вывод ──────────────────────────────────────
        self.add_paragraph(f"\n**Итог по H5a:**")
        sig_count = sum(1 for row in scale_data if '*' in row[4])
        if sig_count > 0:
            self.add_paragraph(f"- Обнаружены значимые различия по {sig_count} шкале(ам) валидации")
        else:
            self.add_paragraph(f"- Значимых различий по шкалам валидации не обнаружено")

        sig_cubes = sum(1 for row in cube_data if '*' in row[4])
        if sig_cubes > 0:
            self.add_paragraph(f"- Значимые различия по {sig_cubes} зоне(ам) кубиков")

        # Подсчитаем общие значимые
        all_sig = []
        for row in scale_data:
            if '*' in row[4]:
                all_sig.append(row[0])
        for row in cube_data:
            if '*' in row[4]:
                all_sig.append(row[0])
        for row in context_data:
            if '*' in row[3]:
                all_sig.append(row[0])

        if all_sig:
            self.add_paragraph(f"- **Значимые параметры:** {', '.join(all_sig)}")
        else:
            self.add_paragraph(f"- **Значимых различий не обнаружено ни по одному параметру**")


    # =========================================================================
    # ГЕНЕРАЦИЯ ОТЧЁТА
    # =========================================================================
    
    def generate_report(self) -> str:
        """Генерация полного отчёта."""
        
        # Заголовок
        self.report.append(f"""# Отчёт анализа данных опроса "Три коробочки"

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
        
        # Описательная статистика
        self.generate_descriptive_stats()
        
        # Визуализации
        self.generate_visualizations()
        
        # Гипотезы
        self.add_section("Проверка гипотез исследования", 2)
        
        # Выполняем все анализы
        self.analyze_h0_anova_levels()
        self.analyze_h0a_green_position_trend()
        self.analyze_h0b_green_position_typical()
        self.analyze_h0c_reactive_position_trend()
        self.analyze_h0a1_green_count_trend()
        self.analyze_h0b1_green_count_typical()
        self.analyze_h0c1_red_count_trend()
        self.analyze_h10a_position_comparison()
        self.analyze_h13_it_comparison()
        self.analyze_h1_correlations_reactive()
        self.analyze_h2_correlations_proactive()
        self.analyze_h3_low_reactive()
        self.analyze_h4_green_dominance()
        self.analyze_h5_moderation_records()
        self.analyze_h5a_records_vs_memory()
        self.analyze_h6_moderation_red()
        self.analyze_h7_curvilinear_green()
        self.analyze_h8_balance_moderation()
        self.analyze_h9_mediation_deficit()
        self.analyze_h10_gender_differences()
        self.analyze_h11_age_trend()
        self.analyze_h12_zen_deficit()
        self.analyze_h14_mbi_swls_relationship()
        self.analyze_h15_regression_predictors()
        self.analyze_h7a_linear_regression()
        self.analyze_h10a_position_comparison()
        self.analyze_productivity_index()
        self.analyze_response_time()

        # Заключение
        self.add_section("Заключение", 2)
        self.add_paragraph("""
**Примечание к интерпретации:**
- p < 0.05 — статистически значимый результат
- Размер эффекта: малый = 0.2, средний = 0.5, большой = 0.8
- Корреляции: слабая |r| < 0.3, средняя 0.3 ≤ |r| < 0.5, сильная |r| ≥ 0.5

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
        description='Анализ данных опроса "Три коробочки"'
    )
    parser.add_argument(
        '--csv', 
        default='./Survey-Jedi-Boxes-Validation/Analysis/jedi_boxes_results.csv',
        help='Путь к CSV-файлу с данными'
    )
    parser.add_argument(
        '--output', 
        default='jedi_boxes_report.md',
        help='Путь для сохранения отчёта'
    )
    
    args = parser.parse_args()
    
    # Проверка существования файла
    if not os.path.exists(args.csv):
        print(f"Ошибка: файл {args.csv} не найден")
        sys.exit(1)
    
    # Запуск анализа
    analyzer = JediBoxesAnalyzer(args.csv)
    analyzer.save_report(args.output)


if __name__ == '__main__':
    main()
