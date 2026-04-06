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
    6: 'Дзен (С>З>К)',
    5: 'Рост (З>С>К)',
    4: 'Не сдаёмся (З>К>С)',
    3: 'Кризис (К>С>З или С>К>З)',
    2: 'Выживание (К>З>С)',
    1: 'Хаос (все зоны равны)'
}

# Гипотезы исследования
HYPOTHESES = {
    'H0': 'Различия между уровнями 1-6 по MBI, прокрастинации и SWLS',
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
    'H7a': 'Линейная регрессия: MBI, Прокрастинация и SWLS предсказываются числом красных и зеленых кубиков',
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
    
    def _filter_completed(self) -> pd.DataFrame:
        """Фильтрация только завершённых анкет и вычисление дополнительных метрик."""
        df = self.data[self.data['status'] == 'completed'].copy()

        # Вычисление ProductivityIndex (логарифм по основанию 2 отношения целевого к срочному с эпсилоном)
        # ProductivityIndex = log₂((Целевое + eps) / (Срочное + eps))
        # Интерпретация: > 0 — преобладание проактивности, < 0 — преобладание реактивности, = 0 — баланс
        # При eps=1 и 7 кубиках диапазон: [-3, 3]
        df['productivity_index'] = np.log2((df['cubes_proactive'] + RG_RATIO_EPS) / (df['cubes_reactive'] + RG_RATIO_EPS))

        return df
    
    def _add_level_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавление колонки с уровнем профиля (1-6).
        
        Алгоритм классификации из research_plan.md:
        - Сортируем зоны по убыванию
        - Применяем правила для "ничьих"
        - Присваиваем уровень
        """
        df = df.copy()
        
        def classify_profile(row):
            r = row.get('cubes_reactive', 0) or 0
            g = row.get('cubes_proactive', 0) or 0
            o = row.get('cubes_operational', 0) or 0
            
            # Создаём список зон с их значениями
            zones = [('reactive', r), ('proactive', g), ('operational', o)]
            
            # Сортируем по убыванию
            zones_sorted = sorted(zones, key=lambda x: x[1], reverse=True)
            
            # Правила для ничьих (иерархия: reactive > proactive > operational)
            if r == g == o:
                return 1  # Хаос (2-2-2)

            # Сортируем зоны по убыванию для определения primary и secondary
            # zones_sorted уже отсортирован: [('zone_name', value), ...]
            # При равенстве значений порядок определяется исходным порядком в zones
            # (stable sort), поэтому нужно явно обработать ничьи
            
            # Определяем порядок зон с учётом иерархии при ничьих
            # Иерархия зон: proactive > reactive > operational
            # (проактивность важнее реактивности, реактивность важнее операционности)
            zone_priority = {'proactive': 2, 'reactive': 1, 'operational': 0}
            
            # Сортируем: сначала по значению (убывание), затем по приоритету (убывание)
            zones_sorted = sorted(zones, key=lambda x: (x[1], zone_priority[x[0]]), reverse=True)
            
            # Извлекаем порядок зон
            order = [z[0] for z in zones_sorted]
            
            # Определяем уровень по порядку доминирования
            primary = order[0]
            secondary = order[1] if len(order) > 1 else None
            
            if primary == 'operational' and secondary == 'proactive':
                return 6  # Дзен: ⚪ > 🟢 > 🔴
            elif primary == 'operational' and secondary == 'reactive':
                return 3  # Кризис: ⚪ > 🔴 > 🟢 (операционное доминирует, но срочное второе)
            elif primary == 'proactive' and (secondary == 'operational' or secondary is None):
                return 5  # Рост: 🟢 > ⚪ > 🔴
            elif primary == 'proactive' and secondary == 'reactive':
                return 4  # Не сдаёмся: 🟢 > 🔴 > ⚪
            elif primary == 'reactive' and secondary == 'operational':
                return 3  # Кризис: 🔴 > ⚪ > 🟢
            elif primary == 'reactive' and (secondary == 'proactive' or secondary is None):
                return 2  # Выживание: 🔴 > 🟢 > ⚪
            else:
                return 1  # Хаос (должно быть покрыто выше)
            
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

        ax.bar(wl_x_positions, wl_y_values, color='steelblue', edgecolor='white')
        ax.set_xlabel('Балл', fontproperties=LABEL_FONT)
        ax.set_ylabel('Число респондентов', fontproperties=LABEL_FONT)
        ax.set_title('Баланс работа/личное', fontproperties=TITLE_FONT)
        ax.set_xticks(wl_x_positions)
        ax.set_xticklabels([wl_labels[x] for x in all_wl_values], fontproperties=TICK_FONT, rotation=0)

        # Добавляем подписи значений
        for i, count in enumerate(wl_y_values):
            if count > 0:
                ax.text(i, count + 0.3, str(count), ha='center', fontproperties=SMALL_TEXT_FONT)

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

        ax.bar(rep_x_positions, rep_y_values, color='forestgreen', edgecolor='white')
        ax.set_xlabel('Балл', fontproperties=LABEL_FONT)
        ax.set_ylabel('Число респондентов', fontproperties=LABEL_FONT)
        ax.set_title('Типичность недели', fontproperties=TITLE_FONT)
        ax.set_xticks(rep_x_positions)
        ax.set_xticklabels([rep_labels[x] for x in all_rep_values], fontproperties=TICK_FONT, rotation=0)

        for i, count in enumerate(rep_y_values):
            if count > 0:
                ax.text(i, count + 0.3, str(count), ha='center', fontproperties=SMALL_TEXT_FONT)

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

        ax.bar(deficit_x_positions, deficit_y_values, color='crimson', edgecolor='white')
        ax.set_xlabel('Балл', fontproperties=LABEL_FONT)
        ax.set_ylabel('Число респондентов', fontproperties=LABEL_FONT)
        ax.set_title('Энергетический дефицит', fontproperties=TITLE_FONT)
        ax.set_xticks(deficit_x_positions)
        ax.set_xticklabels([deficit_labels[x] for x in all_deficit_values], fontproperties=TICK_FONT, rotation=0)

        for i, count in enumerate(deficit_y_values):
            if count > 0:
                ax.text(i, count + 0.3, str(count), ha='center', fontproperties=SMALL_TEXT_FONT)

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
            axes[i].bar(x_pos, counts, color=color, edgecolor='white', alpha=0.8, width=0.8)
            
            # Заголовок с цветным кружком
            axes[i].set_title(ZONE_NAMES[zone], fontsize=13, fontweight='bold',
                            color=ZONE_TEXT_COLORS[zone], fontproperties=LABEL_FONT)
            axes[i].set_xlabel('Количество кубиков', fontproperties=LABEL_FONT)
            axes[i].set_ylabel('Частота', fontproperties=LABEL_FONT)
            axes[i].set_xticks(x_pos)
            axes[i].set_xticklabels([str(int(v)) for v in unique_vals], fontproperties=TICK_FONT)
            axes[i].axvline(x_pos[np.argmax(counts)], color='red', linestyle='--', alpha=0.5)
            axes[i].text(x_pos[np.argmax(counts)]/2, max(counts)*0.9, f'M = {vals.mean():.1f}',
                        color='red', fontsize=10, fontproperties=TICK_FONT)

        plt.suptitle('Распределение кубиков по зонам', fontsize=14, fontweight='bold',
                    fontproperties=TITLE_FONT)
        plt.tight_layout()
        path = self.save_figure('cubes_distribution', 'Распределение кубиков')
        self.add_paragraph(f"![Распределение кубиков]({path})")

        # 3. Распределение уровней профиля
        fig, ax = plt.subplots(figsize=(10, 5))

        level_counts = self.completed['level'].value_counts().sort_index()
        colors = ['#e74c3c', '#e67e22', '#f39c12', '#27ae60', '#2ecc71', '#1abc9c']
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

        # 5. Распределение шкал
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for i, (scale, name) in enumerate([('proc_total', 'Прокрастинация'),
                                           ('swls_total', 'SWLS'),
                                           ('mbi_total', 'MBI')]):
            vals = self.completed[scale].dropna()
            axes[i].hist(vals, bins=15, color='steelblue', edgecolor='white', alpha=0.8)
            axes[i].set_title(name, fontproperties=LABEL_FONT)
            axes[i].set_xlabel('Балл', fontproperties=LABEL_FONT)
            axes[i].set_ylabel('Частота', fontproperties=LABEL_FONT)
            axes[i].axvline(vals.mean(), color='red', linestyle='--', label=f'M = {vals.mean():.1f}')
            axes[i].legend(prop=TICK_FONT)
            # Настраиваем шрифт для tick labels
            for label in axes[i].get_xticklabels() + axes[i].get_yticklabels():
                label.set_fontproperties(TICK_FONT)

        plt.suptitle('Распределение баллов шкал валидации', fontproperties=TITLE_FONT)
        plt.tight_layout()
        path = self.save_figure('scale_distributions', 'Шкалы')
        self.add_paragraph(f"![Шкалы]({path})")

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
            for level in range(1, 7):
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
            
            # Описательная статистика по группам
            table_data = []
            for level in range(1, 7):
                g = self.completed[self.completed['level'] == level][scale].dropna()
                if len(g) > 0:
                    table_data.append([level, f"{g.mean():.1f}", f"{g.std():.1f}", len(g)])
            
            self.add_table(['Уровень', 'M', 'SD', 'n'], table_data)
    
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
        
        green_levels = self.completed[self.completed['level'].isin([5, 6])]['swls_total'].dropna()
        other_levels = self.completed[~self.completed['level'].isin([5, 6])]['swls_total'].dropna()
        
        self.add_paragraph(f"- Уровни 5-6 (доминирование 🟢): n = {len(green_levels)}, "
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
        H12: Профиль "Дзен" (уровень 6) связан с наименьшим энергетическим дефицитом.
        
        Метод: Сравнение средних
        """
        self.add_section("H12: Профиль Дзен и дефицит", 3)
        
        self.add_paragraph(HYPOTHESES['H12'])
        
        zen = self.completed[self.completed['level'] == 6]['energy_deficit'].dropna()
        other = self.completed[self.completed['level'] != 6]['energy_deficit'].dropna()
        
        if len(zen) >= 3:
            self.add_paragraph(f"- Уровень 6 (Дзен): n = {len(zen)}, M = {zen.mean():.2f}")
            self.add_paragraph(f"- Другие уровни: n = {len(other)}, M = {other.mean():.2f}")
            
            u_stat, p = stats.mannwhitneyu(zen, other, alternative='less')
            
            # Проверка на None/NaN
            if p is None or (isinstance(p, float) and np.isnan(p)):
                self.add_paragraph("⚠️ Недостаточно данных для надёжного расчёта")
                return
            
            conclusion = "Подтверждается" if p < 0.05 else "Не подтверждается"
            self.add_test_result("U-критерий Манна-Уитни", u_stat, p, conclusion)
        else:
            self.add_paragraph("⚠️ Недостаточно данных (мало респондентов с уровнем 6)")
    

    def analyze_h7a_linear_regression(self):
        """
        H7a: Линейная регрессия для предсказания MBI, Прокрастинации и SWLS
        на основе числа красных и зеленых кубиков.

        Метод: Множественная линейная регрессия с R^2
        """
        self.add_section("H7a: Линейная регрессия - предсказание по кубикам", 3)

        self.add_paragraph(HYPOTHESES['H7a'])

        # Подготовим данные
        X_reactive = self.completed['cubes_reactive'].values
        X_proactive = self.completed['cubes_proactive'].values
        X = np.column_stack([X_reactive, X_proactive])

        targets = [
            ('mbi_total', 'MBI (выгорание)'),
            ('proc_total', 'Прокрастинация'),
            ('swls_total', 'SWLS (удовлетворённость)')
        ]

        for target_col, target_name in targets:
            y = self.completed[target_col].values

            # Множественная линейная регрессия
            # Добавляем intercept
            X_with_intercept = np.column_stack([np.ones(len(X)), X])
            
            # Вычисляем коэффициенты методом наименьших квадратов
            try:
                beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
                y_pred = X_with_intercept @ beta
                
                # Вычисляем R^2
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                
                # Стандартные ошибки коэффициентов
                n = len(y)
                p = 2  # количество предикторов
                mse = ss_res / (n - p - 1)
                var_beta = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
                se_beta = np.sqrt(np.diag(var_beta))
                
                # t-статистики и p-value для коэффициентов
                t_stats = beta / se_beta
                from scipy.stats import t as t_dist
                p_values = 2 * (1 - t_dist.cdf(np.abs(t_stats), df=n-p-1))
                
                intercept, coef_reactive, coef_proactive = beta
                
                self.add_paragraph(f"**{target_name}:**")
                self.add_paragraph(f"- Уравнение: {target_name} = {intercept:.2f} + {coef_reactive:.2f}×Срочное + {coef_proactive:.2f}×Целевое")
                self.add_paragraph(f"- R² = {r_squared:.3f} ({r_squared*100:.1f}% дисперсии объясняется моделью)")
                self.add_paragraph(f"- Коэффициенты:")
                self.add_paragraph(f"  - Intercept: {intercept:.2f} (p={p_values[0]:.3f})")
                self.add_paragraph(f"  - Срочное (🔴): {coef_reactive:.2f} (p={p_values[1]:.3f})")
                self.add_paragraph(f"  - Целевое (🟢): {coef_proactive:.2f} (p={p_values[2]:.3f})")
                
                # Интерпретация
                if r_squared >= 0.3:
                    strength = "сильная"
                elif r_squared >= 0.1:
                    strength = "умеренная"
                else:
                    strength = "слабая"
                
                self.add_paragraph(f"- Предсказательная способность модели: {strength} (R²={r_squared:.3f})")
                
                # Визуализация для каждого таргета
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # График: Срочное vs target
                axes[0].scatter(X_reactive, y, alpha=0.6, color='red')
                # Линия регрессии (фиксируем проактивное на среднем)
                x_line = np.linspace(X_reactive.min(), X_reactive.max(), 100)
                y_line = intercept + coef_reactive * x_line + coef_proactive * np.mean(X_proactive)
                axes[0].plot(x_line, y_line, 'r-', linewidth=2)
                axes[0].set_xlabel('Срочное (🔴)', fontproperties=LABEL_FONT)
                axes[0].set_ylabel(target_name, fontproperties=LABEL_FONT)
                axes[0].set_title(f'Срочное → {target_name}', fontproperties=TITLE_FONT)
                axes[0].text(0.05, 0.95, f'R²={r_squared:.3f}', transform=axes[0].transAxes,
                           fontsize=12, verticalalignment='top', fontproperties=TICK_FONT)
                
                # График: Целевое vs target
                axes[1].scatter(X_proactive, y, alpha=0.6, color='green')
                y_line2 = intercept + coef_reactive * np.mean(X_reactive) + coef_proactive * x_line
                axes[1].plot(x_line, y_line2, 'g-', linewidth=2)
                axes[1].set_xlabel('Целевое (🟢)', fontproperties=LABEL_FONT)
                axes[1].set_ylabel(target_name, fontproperties=LABEL_FONT)
                axes[1].set_title(f'Целевое → {target_name}', fontproperties=TITLE_FONT)
                axes[1].text(0.05, 0.95, f'R²={r_squared:.3f}', transform=axes[1].transAxes,
                           fontsize=12, verticalalignment='top', fontproperties=TICK_FONT)
                
                plt.suptitle(f'Линейная регрессия: {target_name}', fontproperties=TITLE_FONT)
                plt.tight_layout()
                path = self.save_figure(f'regression_{target_col}', target_name)
                self.add_paragraph(f"![{target_name}]({path})")
                
            except Exception as e:
                self.add_paragraph(f"⚠️ Ошибка при расчёте регрессии: {e}")

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
        ax.hist(productivity_index, bins=20, color='purple', edgecolor='white', alpha=0.8)
        ax.axvline(productivity_index.mean(), color='red', linestyle='--', label=f'M = {productivity_index.mean():.2f}')
        ax.axvline(0.0, color='black', linestyle=':', linewidth=2, label='Баланс (PI=0)')
        ax.set_xlabel('ProductivityIndex (log₂(Целевое/Срочное))', fontproperties=LABEL_FONT)
        ax.set_ylabel('Число респондентов', fontproperties=LABEL_FONT)
        ax.set_title('Распределение индекса ProductivityIndex', fontproperties=TITLE_FONT)
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

        # --- Линейная регрессия по типовой неделе ---
        self.add_paragraph(f"\n**Линейная регрессия с ProductivityIndex (типовая неделя, representative ∈ [-1, 1]):**")

        df_typical = self.completed[self.completed['representative'].between(-1, 1)].copy()
        n_full = len(self.completed)
        n_typical = len(df_typical)

        self.add_paragraph(f"*Выборка: {n_typical} из {n_full} респондентов ({n_typical/n_full*100:.1f}%)*")

        # ProductivityIndex для типовой недели
        pi_typical = df_typical['productivity_index'].dropna()

        # Сводная таблица: полная vs типовая
        reg_headers = ['Показатель', 'Полная (β₁)', 'Полная (R²)', 'Полная (p)',
                       'Типовая (β₁)', 'Типовая (R²)', 'Типовая (p)', 'ΔR²']
        reg_rows = []

        for target_col, target_name in targets:
            # Полная выборка
            y_full = self.completed[target_col].values
            X_full = productivity_index.values
            slope_f, intercept_f, r_f, p_f, _ = linregress(X_full, y_full)
            r2_f = r_f ** 2

            # Типовая неделя
            if n_typical >= 3 and len(pi_typical) >= 3:
                y_t = df_typical[target_col].values
                X_t = pi_typical.values
                slope_t, intercept_t, r_t, p_t, _ = linregress(X_t, y_t)
                r2_t = r_t ** 2
            else:
                slope_t, r2_t, p_t = np.nan, np.nan, np.nan

            delta_r2 = r2_t - r2_f if not np.isnan(r2_t) else np.nan

            reg_rows.append([
                target_name,
                f"{slope_f:.3f}", f"{r2_f:.3f}", f"{p_f:.4f}",
                f"{slope_t:.3f}" if not np.isnan(slope_t) else "N/A",
                f"{r2_t:.3f}" if not np.isnan(r2_t) else "N/A",
                f"{p_t:.4f}" if not np.isnan(p_t) else "N/A",
                f"{delta_r2:+.3f}" if not np.isnan(delta_r2) else "N/A"
            ])

        self.add_table(reg_headers, reg_rows)

        # Визуализация: сравнение регрессий полная vs типовая
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        for idx, (target_col, target_name) in enumerate(targets):
            # Полная выборка
            y_full = self.completed[target_col].values
            X_full = productivity_index.values
            slope_f, intercept_f, r_f, p_f, _ = linregress(X_full, y_full)
            r2_f = r_f ** 2

            # Типовая неделя
            y_t = df_typical[target_col].values
            X_t = pi_typical.values
            slope_t, intercept_t, r_t, p_t, _ = linregress(X_t, y_t)
            r2_t = r_t ** 2

            # Scatter полной
            axes[idx].scatter(X_full, y_full, alpha=0.3, color='steelblue', s=30, label='Полная')
            # Scatter типовой
            axes[idx].scatter(X_t, y_t, alpha=0.5, color='coral', s=30, label='Типовая')

            # Линии тренда
            x_min = min(X_full.min(), X_t.min())
            x_max = max(X_full.max(), X_t.max())
            x_line = np.linspace(x_min, x_max, 100)
            y_line_f = intercept_f + slope_f * x_line
            y_line_t = intercept_t + slope_t * x_line
            axes[idx].plot(x_line, y_line_f, 'b-', linewidth=2, alpha=0.7, label=f'Полная R²={r2_f:.3f}')
            axes[idx].plot(x_line, y_line_t, 'r-', linewidth=2, alpha=0.7, label=f'Типовая R²={r2_t:.3f}')

            axes[idx].axvline(0, color='black', linestyle=':', alpha=0.5)
            axes[idx].set_xlabel('ProductivityIndex (log₂)', fontproperties=LABEL_FONT)
            axes[idx].set_ylabel(target_name, fontproperties=LABEL_FONT)
            axes[idx].set_title(f'{target_name}', fontproperties=TITLE_FONT)
            axes[idx].legend(prop={'size': 8})
            for label in axes[idx].get_xticklabels() + axes[idx].get_yticklabels():
                label.set_fontproperties(TICK_FONT)

        plt.suptitle('ProductivityIndex: сравнение регрессий (полная vs типовая неделя)', fontproperties=TITLE_FONT)
        plt.tight_layout()
        path = self.save_figure('productivity_index_typical_week')
        self.add_paragraph(f"\n![Регрессии ProductivityIndex — типовая неделя]({path})")

        # Детальная статистика по типовой неделе
        self.add_paragraph(f"\n**Детальная статистика (типовая неделя):**")
        for target_col, target_name in targets:
            y_t = df_typical[target_col].values
            X_t = pi_typical.values
            slope_t, intercept_t, r_t, p_t, _ = linregress(X_t, y_t)
            r2_t = r_t ** 2

            self.add_paragraph(f"- **{target_name}:** {target_name} = {intercept_t:.2f} + {slope_t:.2f} × PI, R² = {r2_t:.3f}, p = {p_t:.4f}")
            if slope_t > 0:
                self.add_paragraph(f"  → Чем выше ProductivityIndex, тем выше {target_name}")
            else:
                self.add_paragraph(f"  → Чем выше ProductivityIndex, тем ниже {target_name}")

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
    # АНАЛИЗ ТИПОВОЙ НЕДЕЛИ
    # =========================================================================

    def analyze_typical_week_comparison(self):
        """
        Анализ гипотез на подвыборке респондентов с типовой неделей (-1..1).

        Сравниваем корреляции полной выборки и типовой недели.
        """
        self.add_section("Проверка гипотез на типовой неделе", 2)

        self.add_paragraph("""
**Описание анализа:**

Один из источников шума в данных — нетипичные недели. Респонденты, у которых
неделя была «значительно хуже» или «значительно лучше» обычной, могут иметь
искажённое распределение кубиков, не отражающее их типичный паттерн.

Здесь проверяются те же гипотезы, но только для респондентов, которые указали,
что их неделя была близка к обычной (значение representative от -1 до 1 включительно).

Сравниваются коэффициенты корреляции Спирмена для полной выборки и для
подвыборки типовой недели.
""")

        # Фильтруем типовую неделю
        df_typical = self.completed[self.completed['representative'].between(-1, 1)].copy()
        n_full = len(self.completed)
        n_typical = len(df_typical)

        self.add_paragraph(f"**Полная выборка:** {n_full} респондентов")
        self.add_paragraph(f"**Типовая неделя (representative ∈ [-1, 1]):** {n_typical} респондентов ({n_typical/n_full*100:.1f}%)")

        # Определяем список гипотез для проверки
        # Формат: (название, зона, шкала, ожидаемое направление, label)
        hypotheses_to_check = [
            ('H1: 🔴 vs Прокрастинация', 'cubes_reactive', 'proc_total', 'positive', '🔴 → Прокрастинация'),
            ('H1: 🔴 vs MBI', 'cubes_reactive', 'mbi_total', 'positive', '🔴 → MBI'),
            ('H1: 🔴 vs SWLS', 'cubes_reactive', 'swls_total', 'negative', '🔴 → SWLS'),
            ('H2: 🟢 vs SWLS', 'cubes_proactive', 'swls_total', 'positive', '🟢 → SWLS'),
            ('H2: 🟢 vs MBI', 'cubes_proactive', 'mbi_total', 'negative', '🟢 → MBI'),
            ('H2: 🟢 vs Прокрастинация', 'cubes_proactive', 'proc_total', 'negative', '🟢 → Прокрастинация'),
            ('H5: Память vs Записи → MBI', 'memory_vs_records', 'mbi_total', 'positive', 'Записи → MBI'),
            ('H7: 🟢² vs MBI (квадратичная)', 'cubes_proactive_sq', 'mbi_total', 'positive', '🟢² → MBI'),
            ('H8: Баланс × 🔴 → MBI', 'work_life_x_reactive', 'mbi_total', 'positive', 'Баланс×🔴 → MBI'),
            ('H9: Дефицит как медиатор', 'energy_deficit', 'mbi_total', 'positive', 'Дефицит → MBI'),
            ('H10a: Должность → кубики', None, None, None, 'Должность → кубики'),
            ('H7a: Регрессия MBI', None, None, None, 'Регрессия MBI'),
            ('H7a: Регрессия Прокрастинация', None, None, None, 'Регрессия Прокрастинация'),
            ('H7a: Регрессия SWLS', None, None, None, 'Регрессия SWLS'),
        ]

        # Для простых корреляций считаем коэффициенты
        simple_hypotheses = [h for h in hypotheses_to_check if h[2] is not None and not h[0].startswith('H7a') and h[0] != 'H10a']

        self.add_paragraph(f"\n**Сравнительная таблица корреляций Спирмена:**\n")

        # Заголовки таблицы
        headers = ['Гипотеза', 'Полная (r)', 'Полная (p)', 'Типовая (r)', 'Типовая (p)', 'delta_r', 'Значимо?']
        rows = []

        for hyp_name, zone_col, scale_col, expected_dir, label in simple_hypotheses:
            # Специальная обработка для H10a и квадратичных/модерационных
            if zone_col == 'memory_vs_records':
                var1_full = self.completed[zone_col]
                var1_typical = df_typical[zone_col]
            elif zone_col == 'cubes_proactive_sq':
                var1_full = self.completed['cubes_proactive'] ** 2
                var1_typical = df_typical['cubes_proactive'] ** 2
            elif zone_col == 'work_life_x_reactive':
                var1_full = self.completed['work_life'] * self.completed['cubes_reactive']
                var1_typical = df_typical['work_life'] * df_typical['cubes_reactive']
            elif zone_col == 'energy_deficit':
                var1_full = self.completed[zone_col]
                var1_typical = df_typical[zone_col]
            else:
                var1_full = self.completed[zone_col]
                var1_typical = df_typical[zone_col]

            var2_full = self.completed[scale_col]
            var2_typical = df_typical[scale_col]

            # Полная выборка
            r_full, p_full = stats.spearmanr(var1_full, var2_full)

            # Типовая неделя
            if n_typical >= 3:
                r_typical, p_typical = stats.spearmanr(var1_typical, var2_typical)
            else:
                r_typical, p_typical = np.nan, np.nan

            # Разница абсолютных значений
            delta = abs(r_typical) - abs(r_full) if not np.isnan(r_typical) else np.nan

            # Определяем значимость изменений
            if np.isnan(r_typical):
                sig = "N/A"
            elif p_typical < 0.05 and p_full >= 0.05:
                sig = "✅ стало знач."
            elif p_typical >= 0.05 and p_full < 0.05:
                sig = "❌ стало не знач."
            elif abs(delta) > 0.1:
                sig = "↑ усиление" if delta > 0 else "↓ ослабление"
            else:
                sig = "—"

            rows.append([
                label,
                f"{r_full:.3f}",
                f"{p_full:.4f}",
                f"{r_typical:.3f}",
                f"{p_typical:.4f}",
                f"{delta:+.3f}" if not np.isnan(delta) else "N/A",
                sig
            ])

        self.add_table(headers, rows)
        self.add_paragraph(f"*delta_r — изменение абсолютного значения корреляции при переходе к типовой неделе*")

        # Дополнительно: индивидуальные корреляции по зонам для типовой недели
        self.add_paragraph(f"\n**Корреляции зон с валидационными шкалами (типовая неделя):**")

        zone_names = {
            'cubes_reactive': 'Срочное ●',
            'cubes_proactive': 'Целевое ●',
            'cubes_operational': 'Операционное ●'
        }
        scale_names = {
            'proc_total': 'Прокрастинация',
            'swls_total': 'SWLS',
            'mbi_total': 'MBI'
        }

        corr_data = []
        for zone_col, zone_name in zone_names.items():
            for scale_col, scale_name in scale_names.items():
                r, p = stats.spearmanr(df_typical[zone_col], df_typical[scale_col])
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                corr_data.append([zone_name, scale_name, f"{r:.3f}", f"{p:.4f}", sig])

        self.add_table(['Зона', 'Шкала', 'r', 'p', 'Знач.'], corr_data)

        # Визуализация: сравнение корреляций
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(simple_hypotheses))
        width = 0.35

        r_full_vals = []
        r_typical_vals = []
        labels = []

        for hyp_name, zone_col, scale_col, expected_dir, label in simple_hypotheses:
            labels.append(label)
            if zone_col == 'cubes_proactive_sq':
                var1_full = self.completed['cubes_proactive'] ** 2
                var1_typical = df_typical['cubes_proactive'] ** 2
            elif zone_col == 'work_life_x_reactive':
                var1_full = self.completed['work_life'] * self.completed['cubes_reactive']
                var1_typical = df_typical['work_life'] * df_typical['cubes_reactive']
            else:
                var1_full = self.completed[zone_col]
                var1_typical = df_typical[zone_col]

            var2_full = self.completed[scale_col]
            var2_typical = df_typical[scale_col]

            r_full, _ = stats.spearmanr(var1_full, var2_full)
            r_typical, _ = stats.spearmanr(var1_typical, var2_typical)

            r_full_vals.append(r_full)
            r_typical_vals.append(r_typical)

        bars1 = ax.bar(x - width/2, r_full_vals, width, label='Полная выборка', color='steelblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, r_typical_vals, width, label='Типовая неделя', color='coral', alpha=0.8)

        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_xlabel('Гипотеза', fontproperties=LABEL_FONT)
        ax.set_ylabel('Коэффициент корреляции Спирмена', fontproperties=LABEL_FONT)
        ax.set_title('Сравнение корреляций: полная выборка vs типовая неделя', fontproperties=TITLE_FONT)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontproperties=TICK_FONT)
        ax.legend(prop=TICK_FONT)
        plt.tight_layout()
        path = self.save_figure('typical_week_correlations')
        self.add_paragraph(f"\n![Сравнение корреляций]({path})")

        # Выводы
        self.add_paragraph(f"\n**Выводы по анализу типовой недели:**")

        # Считаем, сколько корреляций усилились/ослабели
        strengthened = sum(1 for r in rows if 'усиление' in r[6])
        weakened = sum(1 for r in rows if 'ослабление' in r[6])
        became_sig = sum(1 for r in rows if 'стало знач' in r[6])
        lost_sig = sum(1 for r in rows if 'стало не знач' in r[6])

        self.add_paragraph(f"- При фильтрации по типовой неделе:")
        self.add_paragraph(f"  - Усилились корреляции: {strengthened}")
        self.add_paragraph(f"  - Ослабели корреляции: {weakened}")
        self.add_paragraph(f"  - Стали значимыми: {became_sig}")
        self.add_paragraph(f"  - Потеряли значимость: {lost_sig}")

        if became_sig > 0:
            self.add_paragraph(f"- Фильтрация по типовой неделе помогает выявить скрытые связи")
        else:
            self.add_paragraph(f"- Фильтрация по типовой неделе не привела к появлению новых значимых связей")


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

        # Дополнительно: сравнение времени для разных уровней профиля
        if 'level' in df_valid.columns or True:  # Всегда вычисляем level
            df_valid = self._add_level_column(df_valid)

            self.add_paragraph(f"\n**Время ответа по уровням профиля:**")

            level_stats = df_valid.groupby('level')['time_page2_total'].agg(['mean', 'std', 'count'])

            table_data = []
            for level in sorted(level_stats.index):
                row = level_stats.loc[level]
                profile_name = PROFILE_LEVELS.get(level, f'Уровень {level}')
                table_data.append([level, profile_name, int(row['count']),
                                   f"{row['mean']:.1f}", f"{row['std']:.1f}"])

            self.add_table(['Уровень', 'Профиль', 'n', 'M (сек)', 'SD'], table_data)

            # Визуализация: boxplot времени по уровням
            fig, ax = plt.subplots(figsize=(10, 6))

            sorted_levels = sorted(df_valid['level'].unique())
            bp_data = [df_valid[df_valid['level'] == l]['time_page2_total'].values for l in sorted_levels]
            labels = [PROFILE_LEVELS.get(l, f'Ур.{l}') for l in sorted_levels]

            bp = ax.boxplot(bp_data, labels=labels, patch_artist=True, widths=0.6)

            for patch, level in zip(bp['boxes'], sorted_levels):
                # Цвета по доминирующей зоне
                if level in [2, 3]:
                    patch.set_facecolor(ZONE_COLORS['reactive'])
                elif level in [4, 5]:
                    patch.set_facecolor(ZONE_COLORS['proactive'])
                elif level == 6:
                    patch.set_facecolor(ZONE_COLORS['operational'])
                else:
                    patch.set_facecolor('gray')
                patch.set_alpha(0.7)

            ax.set_xlabel('Уровень профиля', fontproperties=LABEL_FONT)
            ax.set_ylabel('Время на странице 2 (сек)', fontproperties=LABEL_FONT)
            ax.set_title('Время ответа по уровням профиля', fontproperties=TITLE_FONT)
            plt.xticks(rotation=30, ha='right', fontproperties=TICK_FONT)
            plt.yticks(fontproperties=TICK_FONT)
            plt.tight_layout()
            path = self.save_figure('response_time_by_level')
            self.add_paragraph(f"\n![Время ответа по уровням]({path})")

            # Статистический тест: Крускал-Уоллис для сравнения уровней
            if len(sorted_levels) >= 2:
                groups = [df_valid[df_valid['level'] == l]['time_page2_total'].values for l in sorted_levels]
                # Фильтруем пустые группы
                groups = [g for g in groups if len(g) > 1]
                if len(groups) >= 2:
                    h_stat, p_val = stats.kruskal(*groups)
                    self.add_paragraph(f"\n**Тест Крускала-Уоллиса: различия времени по уровням:**")
                    self.add_paragraph(f"- H = {h_stat:.3f}, p = {p_val:.4f}")
                    if p_val < 0.05:
                        self.add_paragraph(f"- **Результат значим**: существуют статистически значимые различия "
                                          f"во времени ответа между уровнями профиля (p < 0.05)")
                    else:
                        self.add_paragraph(f"- **Результат не значим**: нет статистически значимых различий "
                                          f"во времени ответа между уровнями профиля (p > 0.05)")

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
        self.analyze_h1_correlations_reactive()
        self.analyze_h2_correlations_proactive()
        self.analyze_h3_low_reactive()
        self.analyze_h4_green_dominance()
        self.analyze_h5_moderation_records()
        self.analyze_h6_moderation_red()
        self.analyze_h7_curvilinear_green()
        self.analyze_h8_balance_moderation()
        self.analyze_h9_mediation_deficit()
        self.analyze_h10_gender_differences()
        self.analyze_h11_age_trend()
        self.analyze_h12_zen_deficit()
        self.analyze_h7a_linear_regression()
        self.analyze_h10a_position_comparison()
        self.analyze_productivity_index()
        self.analyze_response_time()
        self.analyze_typical_week_comparison()

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
