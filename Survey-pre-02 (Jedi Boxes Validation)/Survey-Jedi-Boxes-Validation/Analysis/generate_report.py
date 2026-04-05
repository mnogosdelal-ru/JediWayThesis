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
RG_RATIO_EPS = 0.1  # Рекомендуемое значение: 0.1

# Уровни профилей (К=красное/срочное, З=зеленое/целевое, С=серое/операционное)
PROFILE_LEVELS = {
    6: 'Дзен (С>З>К)',
    5: 'Рост (З>С>К)',
    4: 'Не сдаёмся (З>К>С)',
    3: 'Кризис (К>С>З)',
    2: 'Выживание (К>З>С)',
    1: 'Хаос (2-2-2)'
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

        # Вычисление ProductivityIndex (логарифм отношения целевого к срочному с эпсилоном)
        # ProductivityIndex = log((Целевое + eps) / (Срочное + eps))
        # Интерпретация: > 0 — преобладание проактивности, < 0 — преобладание реактивности, = 0 — баланс
        df['productivity_index'] = np.log((df['cubes_proactive'] + RG_RATIO_EPS) / (df['cubes_reactive'] + RG_RATIO_EPS))

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
            
            # Проверяем равенства и применяем иерархию
            order = []
            if r > g and r > o:
                order = ['reactive']
            elif g > r and g > o:
                order = ['proactive']
            elif o > r and o > g:
                order = ['operational']
            elif r == g and r > o:
                # r = g > o: proactive доминирует (рост важнее реактивности)
                order = ['proactive', 'reactive']
            elif r == o and r > g:
                # r = o > g: reactive доминирует (срочное сильнее влияет)
                order = ['reactive', 'operational']
            elif g == o and g > r:
                # g = o > r: proactive доминирует
                order = ['proactive', 'operational']
            else:
                order = [zones_sorted[0][0]]
            
            # Определяем уровень по порядку доминирования
            primary = order[0]
            secondary = order[1] if len(order) > 1 else None
            
            if primary == 'operational' and secondary == 'proactive':
                return 6  # Дзен: ⚪ > 🟢 > 🔴
            elif primary == 'proactive' and (secondary == 'operational' or secondary is None):
                return 5  # Рост: 🟢 > ⚪ > 🔴
            elif primary == 'proactive' and secondary == 'reactive':
                return 4  # Не сдаёмся: 🟢 > 🔴 > ⚪
            elif primary == 'reactive' and secondary == 'operational':
                return 3  # Кризис: 🔴 > ⚪ > 🟢
            elif primary == 'reactive' and (secondary == 'proactive' or secondary is None):
                return 2  # Выживание: 🔴 > 🟢 > ⚪
            else:
                return 1  # Хаос
            
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
        Анализ индекса ProductivityIndex — логарифм отношения целевого к срочному.

        ProductivityIndex = log((Целевое + eps) / (Срочное + eps))

        Интерпретация:
        - ProductivityIndex > 0: преобладание проактивности (целевых задач)
        - ProductivityIndex = 0: баланс между срочными и целевыми задачами
        - ProductivityIndex < 0: преобладание реактивности (срочных задач)

        Логарифм делает шкалу симметричной: например, отношение 2:1 и 1:2
        дают значения +0.69 и -0.69 соответственно.
        """
        self.add_section("Анализ индекса ProductivityIndex (log(Целевое/Срочное))", 2)

        self.add_paragraph("""
**Описание индекса:**

ProductivityIndex — это логарифм отношения количества кубиков в зоне "Целевое" к количеству
кубиков в зоне "Срочное" с добавлением небольшого смещения (eps) для избежания
деления на ноль и логарифмирования нуля.

Формула: ProductivityIndex = log((Целевое + eps) / (Срочное + eps))

Интерпретация:
- ProductivityIndex > 0: преобладание проактивности (целевые задачи доминируют)
- ProductivityIndex ≈ 0: баланс между срочными и целевыми задачами
- ProductivityIndex < 0: преобладание реактивности (срочные задачи доминируют)

Преимущество логарифмической шкалы: симметрия. Отношения 2:1 и 1:2 дают
равные по модулю, но противоположные по знаку значения (+0.69 и -0.69).

Высокий ProductivityIndex указывает на более продуктивный профиль распределения энергии.

**Параметр смещения:** eps = {eps}
""".format(eps=RG_RATIO_EPS))

        # Описательная статистика
        productivity_index = self.completed['productivity_index'].dropna()
        self.add_paragraph(f"**Описательная статистика ProductivityIndex:**")
        self.add_paragraph(f"- M = {productivity_index.mean():.2f}, SD = {productivity_index.std():.2f}")
        self.add_paragraph(f"- Медиана = {productivity_index.median():.2f}")
        self.add_paragraph(f"- Диапазон = {productivity_index.min():.2f} – {productivity_index.max():.2f}")

        # Категоризация по логарифмической шкале
        # log(1.5) ≈ 0.405, log(0.67) ≈ -0.405
        high_proactivity = (productivity_index > 0.4).sum()
        balanced = ((productivity_index >= -0.4) & (productivity_index <= 0.4)).sum()
        high_reactivity = (productivity_index < -0.4).sum()

        self.add_paragraph(f"\n**Распределение по категориям:**")
        self.add_paragraph(f"- Преобладание проактивности (PI > 0.4): {high_proactivity} ({high_proactivity/len(productivity_index)*100:.1f}%)")
        self.add_paragraph(f"- Баланс (-0.4 ≤ PI ≤ 0.4): {balanced} ({balanced/len(productivity_index)*100:.1f}%)")
        self.add_paragraph(f"- Преобладание реактивности (PI < -0.4): {high_reactivity} ({high_reactivity/len(productivity_index)*100:.1f}%)")

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
        ax.set_xlabel('ProductivityIndex (log(Целевое/Срочное))', fontproperties=LABEL_FONT)
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
            axes[idx].set_xlabel('ProductivityIndex (log)', fontproperties=LABEL_FONT)
            axes[idx].set_ylabel(target_name, fontproperties=LABEL_FONT)
            axes[idx].set_title(f'ProductivityIndex → {target_name}', fontproperties=TITLE_FONT)
            axes[idx].text(0.05, 0.95, f'R²={r_squared:.3f}\np={p_value:.3f}',
                          transform=axes[idx].transAxes, fontsize=10, verticalalignment='top',
                          fontproperties=TICK_FONT)

        plt.suptitle('Линейная регрессия: ProductivityIndex как предиктор', fontproperties=TITLE_FONT)
        plt.tight_layout()
        path = self.save_figure('productivity_index_regressions')
        self.add_paragraph(f"\n![Регрессии ProductivityIndex]({path})")

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

        self.add_paragraph(f"- Логарифмическая шкала обеспечивает симметрию: значения +X и -X соответствуют")
        self.add_paragraph(f"  равному по силе, но противоположному по направлению дисбалансу")
        self.add_paragraph(f"- ProductivityIndex может быть полезен как компактный индикатор продуктивного профиля")
        self.add_paragraph(f"- Рекомендуемый eps = {RG_RATIO_EPS} для сглаживания крайних значений")


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
