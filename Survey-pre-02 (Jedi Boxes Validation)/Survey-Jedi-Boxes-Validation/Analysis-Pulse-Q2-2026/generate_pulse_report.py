#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Генерация отчета по лонгитюдному пульс-опросу (Q2 2026)

Проверка гипотез с временным лагом:
- H1: cubes_proactive[week N] → representative[week N+1]
- H2: cubes_reactive[week N] → energy_deficit[week N+1]

Статистические методы:
- Корреляция Спирмена (ранговые данные)
- Крест-лаговый анализ с контролем автокорреляции
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os
import sys
from datetime import datetime

# Настройка шрифтов для кириллицы
FONT_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FONT_DIR, exist_ok=True)

TITLE_FONT = font_manager.FontProperties(family='DejaVu Sans', size=14)
LABEL_FONT = font_manager.FontProperties(family='DejaVu Sans', size=12)
TICK_FONT = font_manager.FontProperties(family='DejaVu Sans', size=10)


class PulseAnalyzer:
    """Анализатор лонгитюдных данных пульс-опроса."""
    
    def __init__(self, csv_path: str):
        """
        Инициализация анализатора.
        
        Args:
            csv_path: Путь к CSV-файлу с данными
        """
        self.data = pd.read_csv(csv_path, encoding='utf-8')
        self.valid_weeks = self.data.dropna(subset=['week', 'tg_id'])
        self.figures = []
        
        print(f"Загружено наблюдений: {len(self.data)}")
        print(f"Уникальных участников: {self.data['tg_id'].nunique()}")
        print(f"Уникальных недель: {sorted(self.data['week'].dropna().unique())}")
    
    def save_figure(self, name: str) -> str:
        """Сохранение графика."""
        path = os.path.join(FONT_DIR, f"{name}.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        self.figures.append(name)
        return f"figures/{name}.png"
    
    def add_section(self, title: str, level: int = 1):
        """Добавление заголовка."""
        prefix = "#" * level
        self.report.append(f"\n{prefix} {title}\n")
    
    def add_paragraph(self, text: str):
        """Добавление абзаца."""
        self.report.append(text + "\n")
    
    def add_table(self, headers: list, rows: list):
        """Добавление Markdown-таблицы."""
        if not rows:
            self.report.append("*Нет данных для таблицы*\n")
            return
        
        # Ширина колонок
        col_widths = [max(len(str(h)), max(len(str(r[i])) for r in rows) if rows else 0) 
                      for i, h in enumerate(headers)]
        
        # Заголовок
        header_row = "| " + " | ".join(str(h) for h in headers) + " |"
        self.report.append(header_row)
        
        # Разделитель с выравниванием по левому краю
        sep_row = "| " + " | ".join(":" + "-" * (col_widths[i] - 1) for i in range(len(headers))) + " |"
        self.report.append(sep_row)
        
        # Данные
        for row in rows:
            data_row = "| " + " | ".join(str(row[i]) for i in range(len(headers))) + " |"
            self.report.append(data_row)
    
    def create_lagged_dataset(self) -> pd.DataFrame:
        """
        Создание датасета с лаговыми переменными.
        
        Создает пары наблюдений (week N, week N+1) для каждого участника.
        """
        # Сортировка по участнику и неделе
        sorted_data = self.data.sort_values(['tg_id', 'week'])
        
        # Создание лаговых переменных
        lagged_rows = []
        
        for tg_id, group in sorted_data.groupby('tg_id'):
            group = group.sort_values('week').reset_index(drop=True)
            
            if len(group) < 2:
                continue
            
            for i in range(len(group) - 1):
                current = group.iloc[i]
                next_week = group.iloc[i + 1]
                
                lagged_rows.append({
                    'tg_id': tg_id,
                    'week_N': current['week'],
                    'week_N+1': next_week['week'],
                    # Предикторы на неделе N
                    'cubes_proactive_N': current['cubes_proactive'],
                    'cubes_reactive_N': current['cubes_reactive'],
                    'cubes_operational_N': current['cubes_operational'],
                    'representative_N': current['representative'],
                    'work_life_N': current['work_life'],
                    'energy_deficit_N': current['energy_deficit'],
                    # Исходы на неделе N+1
                    'representative_N+1': next_week['representative'],
                    'energy_deficit_N+1': next_week['energy_deficit'],
                    'cubes_proactive_N+1': next_week['cubes_proactive'],
                    'cubes_reactive_N+1': next_week['cubes_reactive'],
                    'work_life_N+1': next_week['work_life'],
                })
        
        lagged_df = pd.DataFrame(lagged_rows)
        print(f"Создано пар наблюдений (N → N+1): {len(lagged_df)}")
        
        return lagged_df
    
    def analyze_lagged_hypotheses(self):
        """
        Проверка гипотез с временным лагом.
        
        H1: cubes_proactive[week N] → representative[week N+1]
        H2: cubes_reactive[week N] → energy_deficit[week N+1]
        """
        self.add_section("Проверка гипотез с временным лагом", 2)
        
        self.add_paragraph("""
**Методология:**

Для проверки гипотез с лагом используется корреляционный анализ Спирмена:
- Корреляция Спирмена подходит для ранговых данных (ипсативные кубики, Likert-шкалы)
- Проверяется связь предиктора на неделе N с исходом на неделе N+1
- Контроль автокорреляции через включение базового уровня исхода

**Гипотезы:**

1. **H1:** `cubes_proactive[week N]` положительно коррелирует с `representative[week N+1]`
   - Ожидание: проактивное планирование улучшает типичность следующей недели

2. **H2:** `cubes_reactive[week N]` положительно коррелирует с `energy_deficit[week N+1]`
   - Ожидание: реактивность приводит к накоплению дефицита энергии

3. **H3:** `cubes_proactive[week N]` отрицательно коррелирует с `energy_deficit[week N+1]`
   - Ожидание: проактивность снижает накопление дефицита энергии

4. **H4:** `cubes_proactive[week N]` положительно коррелирует с `cubes_proactive[week N+1]`
   - Ожидание: проактивность — устойчивая привычка (автокорреляция)

5. **H5:** `cubes_proactive[week N]` положительно коррелирует с `work_life[week N]`
   - Ожидание: проактивность связана с лучшим балансом работа/личное

6. **H6:** `cubes_proactive[week N]` положительно коррелирует с `work_life[week N+1]`
   - Ожидание: проактивность улучшает баланс на следующей неделе
""")
        
        lagged = self.create_lagged_dataset()
        
        # Отфильтровываем полные наблюдения
        lagged_clean = lagged.dropna(subset=[
            'cubes_proactive_N', 'representative_N+1',
            'cubes_reactive_N', 'energy_deficit_N+1',
            'representative_N', 'energy_deficit_N'
        ]).copy()
        
        n_pairs = len(lagged_clean)
        self.add_paragraph(f"**Размер выборки (пары наблюдений):** n = {n_pairs}")
        
        if n_pairs < 30:
            self.add_paragraph("⚠️ Недостаточно данных для статистически значимых выводов (нужно минимум 30 пар)")
            return
        
        # =========================================================================
        # H1: cubes_proactive[N] → representative[N+1]
        # =========================================================================
        self.add_section("H1: Влияние проактивности на типичность следующей недели", 3)
        
        x = lagged_clean['cubes_proactive_N'].values
        y = lagged_clean['representative_N+1'].values
        
        # Базовая корреляция Спирмена
        rho, p_value = stats.spearmanr(x, y)
        
        # Контроль автокорреляции: частичная корреляция
        # Убираем влияние representative_N на representative_N+1
        z = lagged_clean['representative_N'].values
        valid_idx = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        
        if np.sum(valid_idx) >= 30:
            # Ручная вычисление частичной корреляции Спирмена
            r_xy, _ = stats.spearmanr(x[valid_idx], y[valid_idx])
            r_xz, _ = stats.spearmanr(x[valid_idx], z[valid_idx])
            r_yz, _ = stats.spearmanr(y[valid_idx], z[valid_idx])
            n_valid = np.sum(valid_idx)
            numerator = r_xy - r_xz * r_yz
            denominator = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
            partial_rho = numerator / denominator if denominator != 0 else 0
            # Приближённый p-value через t-распределение
            t_stat = partial_rho * np.sqrt((n_valid - 3) / (1 - partial_rho**2)) if abs(partial_rho) < 1 else 0
            partial_p = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_valid-3))
        else:
            partial_rho, partial_p = np.nan, np.nan
        
        # Интерпретация
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."
        direction = "✅" if rho > 0 and p_value < 0.05 else "❌" if rho <= 0 and p_value < 0.05 else "⚪"
        
        self.add_paragraph(f"\n**Результаты:**")
        self.add_paragraph(f"- Корреляция Спирмена: ρ = {rho:.3f}, p = {p_value:.4f} {sig}")
        self.add_paragraph(f"- Частичная корреляция (контроль автокорреляции): ρ = {partial_rho:.3f}, p = {partial_p:.4f}")
        self.add_paragraph(f"- Направление связи: {'Положительное' if rho > 0 else 'Отрицательное' if rho < 0 else 'Нет связи'}")
        
        if rho > 0 and p_value < 0.05:
            self.add_paragraph(f"- {direction} **Гипотеза H1 ПОДТВЕРЖДАЕТСЯ:** проактивность на неделе N предсказывает лучшую типичность на неделе N+1")
        elif rho <= 0 and p_value < 0.05:
            self.add_paragraph(f"- {direction} **Гипотеза H1 Опровергается:** наблюдается отрицательная связь")
        else:
            self.add_paragraph(f"- {direction} **Гипотеза H1 НЕ подтверждается:** связь не статистически значима")
        
        # Визуализация
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter plot с линией тренда
        ax1 = axes[0]
        valid = ~(np.isnan(x) | np.isnan(y))
        ax1.scatter(x[valid], y[valid], alpha=0.5, s=40, color='#27ae60')
        
        # Линия тренда
        if np.sum(valid) >= 3:
            z_fit = np.polyfit(x[valid], y[valid], 1)
            p_fit = np.poly1d(z_fit)
            x_line = np.linspace(x[valid].min(), x[valid].max(), 50)
            ax1.plot(x_line, p_fit(x_line), 'r--', linewidth=2, alpha=0.7)
        
        ax1.set_xlabel('cubes_proactive[week N]', fontproperties=LABEL_FONT)
        ax1.set_ylabel('representative[week N+1]', fontproperties=LABEL_FONT)
        ax1.set_title(f'H1: Проактивность → Типичность\n(ρ = {rho:.3f}, p = {p_value:.4f})', fontproperties=TITLE_FONT)
        
        for label in ax1.get_xticklabels() + ax1.get_yticklabels():
            label.set_fontproperties(TICK_FONT)
        
        # Распределение разниц
        ax2 = axes[1]
        if 'representative_N' in lagged_clean.columns:
            diff = lagged_clean['representative_N+1'] - lagged_clean['representative_N']
            valid_diff = diff.dropna()
            ax2.hist(valid_diff.values, bins=15, alpha=0.7, color='#3498db', edgecolor='black')
            ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax2.set_xlabel('Изменение representative (N+1 - N)', fontproperties=LABEL_FONT)
            ax2.set_ylabel('Частота', fontproperties=LABEL_FONT)
            ax2.set_title('Распределение изменений типичности', fontproperties=TITLE_FONT)
            
            for label in ax2.get_xticklabels() + ax2.get_yticklabels():
                label.set_fontproperties(TICK_FONT)
        
        plt.suptitle('H1: Анализ с лагом', fontproperties=TITLE_FONT, size=16)
        plt.tight_layout()
        path = self.save_figure('h1_lagged_analysis')
        self.add_paragraph(f"\n![H1 анализ]({path})")
        
        # =========================================================================
        # H2: cubes_reactive[N] → energy_deficit[N+1]
        # =========================================================================
        self.add_section("H2: Влияние реактивности на дефицит энергии следующей недели", 3)
        
        x2 = lagged_clean['cubes_reactive_N'].values
        y2 = lagged_clean['energy_deficit_N+1'].values
        
        # Базовая корреляция Спирмена
        rho2, p_value2 = stats.spearmanr(x2, y2)
        
        # Контроль автокорреляции
        z2 = lagged_clean['energy_deficit_N'].values
        valid_idx2 = ~(np.isnan(x2) | np.isnan(y2) | np.isnan(z2))
        
        if np.sum(valid_idx2) >= 30:
            # Ручная вычисление частичной корреляции Спирмена
            r_xy2, _ = stats.spearmanr(x2[valid_idx2], y2[valid_idx2])
            r_xz2, _ = stats.spearmanr(x2[valid_idx2], z2[valid_idx2])
            r_yz2, _ = stats.spearmanr(y2[valid_idx2], z2[valid_idx2])
            n_valid2 = np.sum(valid_idx2)
            numerator2 = r_xy2 - r_xz2 * r_yz2
            denominator2 = np.sqrt((1 - r_xz2**2) * (1 - r_yz2**2))
            partial_rho2 = numerator2 / denominator2 if denominator2 != 0 else 0
            t_stat2 = partial_rho2 * np.sqrt((n_valid2 - 3) / (1 - partial_rho2**2)) if abs(partial_rho2) < 1 else 0
            partial_p2 = 2 * (1 - stats.t.cdf(abs(t_stat2), df=n_valid2-3))
        else:
            partial_rho2, partial_p2 = np.nan, np.nan
        
        # Интерпретация
        sig2 = "***" if p_value2 < 0.001 else "**" if p_value2 < 0.01 else "*" if p_value2 < 0.05 else "n.s."
        direction2 = "✅" if rho2 > 0 and p_value2 < 0.05 else "❌" if rho2 <= 0 and p_value2 < 0.05 else "⚪"
        
        self.add_paragraph(f"\n**Результаты:**")
        self.add_paragraph(f"- Корреляция Спирмена: ρ = {rho2:.3f}, p = {p_value2:.4f} {sig2}")
        self.add_paragraph(f"- Частичная корреляция (контроль автокорреляции): ρ = {partial_rho2:.3f}, p = {partial_p2:.4f}")
        self.add_paragraph(f"- Направление связи: {'Положительное' if rho2 > 0 else 'Отрицательное' if rho2 < 0 else 'Нет связи'}")
        
        if rho2 > 0 and p_value2 < 0.05:
            self.add_paragraph(f"- {direction2} **Гипотеза H2 ПОДТВЕРЖДАЕТСЯ:** реактивность на неделе N предсказывает больший дефицит энергии на неделе N+1")
        elif rho2 <= 0 and p_value2 < 0.05:
            self.add_paragraph(f"- {direction2} **Гипотеза H2 Опровергается:** наблюдается отрицательная связь")
        else:
            self.add_paragraph(f"- {direction2} **Гипотеза H2 НЕ подтверждается:** связь не статистически значима")
        
        # Визуализация
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter plot
        ax1 = axes[0]
        valid2 = ~(np.isnan(x2) | np.isnan(y2))
        ax1.scatter(x2[valid2], y2[valid2], alpha=0.5, s=40, color='#e74c3c')
        
        if np.sum(valid2) >= 3:
            z_fit2 = np.polyfit(x2[valid2], y2[valid2], 1)
            p_fit2 = np.poly1d(z_fit2)
            x_line2 = np.linspace(x2[valid2].min(), x2[valid2].max(), 50)
            ax1.plot(x_line2, p_fit2(x_line2), 'r--', linewidth=2, alpha=0.7)
        
        ax1.set_xlabel('cubes_reactive[week N]', fontproperties=LABEL_FONT)
        ax1.set_ylabel('energy_deficit[week N+1]', fontproperties=LABEL_FONT)
        ax1.set_title(f'H2: Реактивность → Дефицит энергии\n(ρ = {rho2:.3f}, p = {p_value2:.4f})', fontproperties=TITLE_FONT)
        
        for label in ax1.get_xticklabels() + ax1.get_yticklabels():
            label.set_fontproperties(TICK_FONT)
        
        # Распределение разниц
        ax2 = axes[1]
        if 'energy_deficit_N' in lagged_clean.columns:
            diff2 = lagged_clean['energy_deficit_N+1'] - lagged_clean['energy_deficit_N']
            valid_diff2 = diff2.dropna()
            ax2.hist(valid_diff2.values, bins=15, alpha=0.7, color='#e67e22', edgecolor='black')
            ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax2.set_xlabel('Изменение energy_deficit (N+1 - N)', fontproperties=LABEL_FONT)
            ax2.set_ylabel('Частота', fontproperties=LABEL_FONT)
            ax2.set_title('Распределение изменений дефицита энергии', fontproperties=TITLE_FONT)
            
            for label in ax2.get_xticklabels() + ax2.get_yticklabels():
                label.set_fontproperties(TICK_FONT)
        
        plt.suptitle('H2: Анализ с лагом', fontproperties=TITLE_FONT, size=16)
        plt.tight_layout()
        path = self.save_figure('h2_lagged_analysis')
        self.add_paragraph(f"\n![H2 анализ]({path})")
        
        # =========================================================================
        # H3: cubes_proactive[N] → energy_deficit[N+1] (отрицательная корреляция)
        # =========================================================================
        self.add_section("H3: Влияние проактивности на дефицит энергии следующей недели", 3)
        
        x3 = lagged_clean['cubes_proactive_N'].values
        y3 = lagged_clean['energy_deficit_N+1'].values
        
        # Базовая корреляция Спирмена
        rho3, p_value3 = stats.spearmanr(x3, y3)
        
        # Контроль автокорреляции
        z3 = lagged_clean['energy_deficit_N'].values
        valid_idx3 = ~(np.isnan(x3) | np.isnan(y3) | np.isnan(z3))
        
        if np.sum(valid_idx3) >= 30:
            # Ручная вычисление частичной корреляции Спирмена
            r_xy3, _ = stats.spearmanr(x3[valid_idx3], y3[valid_idx3])
            r_xz3, _ = stats.spearmanr(x3[valid_idx3], z3[valid_idx3])
            r_yz3, _ = stats.spearmanr(y3[valid_idx3], z3[valid_idx3])
            n_valid3 = np.sum(valid_idx3)
            numerator3 = r_xy3 - r_xz3 * r_yz3
            denominator3 = np.sqrt((1 - r_xz3**2) * (1 - r_yz3**2))
            partial_rho3 = numerator3 / denominator3 if denominator3 != 0 else 0
            t_stat3 = partial_rho3 * np.sqrt((n_valid3 - 3) / (1 - partial_rho3**2)) if abs(partial_rho3) < 1 else 0
            partial_p3 = 2 * (1 - stats.t.cdf(abs(t_stat3), df=n_valid3-3))
        else:
            partial_rho3, partial_p3 = np.nan, np.nan
        
        # Интерпретация (ожидаем ОТРИЦАТЕЛЬНую корреляцию)
        sig3 = "***" if p_value3 < 0.001 else "**" if p_value3 < 0.01 else "*" if p_value3 < 0.05 else "n.s."
        # Для отрицательной корреляции: подтверждается если rho < 0 и p < 0.05
        direction3 = "✅" if rho3 < 0 and p_value3 < 0.05 else "❌" if rho3 >= 0 and p_value3 < 0.05 else "⚪"
        
        self.add_paragraph(f"\n**Результаты:**")
        self.add_paragraph(f"- Корреляция Спирмена: ρ = {rho3:.3f}, p = {p_value3:.4f} {sig3}")
        self.add_paragraph(f"- Частичная корреляция (контроль автокорреляции): ρ = {partial_rho3:.3f}, p = {partial_p3:.4f}")
        self.add_paragraph(f"- Направление связи: {'Отрицательное' if rho3 < 0 else 'Положительное' if rho3 > 0 else 'Нет связи'}")
        
        if rho3 < 0 and p_value3 < 0.05:
            self.add_paragraph(f"- {direction3} **Гипотеза H3 ПОДТВЕРЖДАЕТСЯ:** проактивность на неделе N предсказывает меньший дефицит энергии на неделе N+1")
        elif rho3 >= 0 and p_value3 < 0.05:
            self.add_paragraph(f"- {direction3} **Гипотеза H3 Опровергается:** наблюдается положительная связь")
        else:
            self.add_paragraph(f"- {direction3} **Гипотеза H3 НЕ подтверждается:** связь не статистически значима")
        
        # Визуализация
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter plot
        ax1 = axes[0]
        valid3 = ~(np.isnan(x3) | np.isnan(y3))
        ax1.scatter(x3[valid3], y3[valid3], alpha=0.5, s=40, color='#27ae60')
        
        if np.sum(valid3) >= 3:
            z_fit3 = np.polyfit(x3[valid3], y3[valid3], 1)
            p_fit3 = np.poly1d(z_fit3)
            x_line3 = np.linspace(x3[valid3].min(), x3[valid3].max(), 50)
            ax1.plot(x_line3, p_fit3(x_line3), 'r--', linewidth=2, alpha=0.7)
        
        ax1.set_xlabel('cubes_proactive[week N]', fontproperties=LABEL_FONT)
        ax1.set_ylabel('energy_deficit[week N+1]', fontproperties=LABEL_FONT)
        ax1.set_title(f'H3: Проактивность → Дефицит энергии\n(ρ = {rho3:.3f}, p = {p_value3:.4f})', fontproperties=TITLE_FONT)
        
        for label in ax1.get_xticklabels() + ax1.get_yticklabels():
            label.set_fontproperties(TICK_FONT)
        
        # Распределение разниц
        ax2 = axes[1]
        if 'energy_deficit_N' in lagged_clean.columns:
            diff3 = lagged_clean['energy_deficit_N+1'] - lagged_clean['energy_deficit_N']
            valid_diff3 = diff3.dropna()
            ax2.hist(valid_diff3.values, bins=15, alpha=0.7, color='#9b59b6', edgecolor='black')
            ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax2.set_xlabel('Изменение energy_deficit (N+1 - N)', fontproperties=LABEL_FONT)
            ax2.set_ylabel('Частота', fontproperties=LABEL_FONT)
            ax2.set_title('Распределение изменений дефицита энергии', fontproperties=TITLE_FONT)
            
            for label in ax2.get_xticklabels() + ax2.get_yticklabels():
                label.set_fontproperties(TICK_FONT)
        
        plt.suptitle('H3: Анализ с лагом', fontproperties=TITLE_FONT, size=16)
        plt.tight_layout()
        path = self.save_figure('h3_lagged_analysis')
        self.add_paragraph(f"\n![H3 анализ]({path})")
        
        # =========================================================================
        # H4: cubes_proactive[N] → cubes_proactive[N+1] (автокорреляция)
        # =========================================================================
        self.add_section("H4: Автокорреляция проактивности (устойчивость привычки)", 3)
        
        x4 = lagged_clean['cubes_proactive_N'].values
        y4 = lagged_clean['cubes_proactive_N+1'].values
        
        # Базовая корреляция Спирмена
        rho4, p_value4 = stats.spearmanr(x4, y4)
        
        # Интерпретация
        sig4 = "***" if p_value4 < 0.001 else "**" if p_value4 < 0.01 else "*" if p_value4 < 0.05 else "n.s."
        direction4 = "✅" if rho4 > 0 and p_value4 < 0.05 else "❌" if rho4 <= 0 and p_value4 < 0.05 else "⚪"
        
        self.add_paragraph(f"\n**Результаты:**")
        self.add_paragraph(f"- Корреляция Спирмена: ρ = {rho4:.3f}, p = {p_value4:.4f} {sig4}")
        self.add_paragraph(f"- Направление связи: {'Положительное' if rho4 > 0 else 'Отрицательное' if rho4 < 0 else 'Нет связи'}")
        
        if rho4 > 0 and p_value4 < 0.05:
            self.add_paragraph(f"- {direction4} **Гипотеза H4 ПОДТВЕРЖДАЕТСЯ:** проактивность — устойчивая привычка (значимая автокорреляция)")
        elif rho4 <= 0 and p_value4 < 0.05:
            self.add_paragraph(f"- {direction4} **Гипотеза H4 Опровергается:** наблюдается отрицательная автокорреляция")
        else:
            self.add_paragraph(f"- {direction4} **Гипотеза H4 НЕ подтверждается:** автокорреляция не значима")
        
        # Визуализация
        fig, ax = plt.subplots(figsize=(8, 6))
        valid4 = ~(np.isnan(x4) | np.isnan(y4))
        ax.scatter(x4[valid4], y4[valid4], alpha=0.5, s=40, color='#9b59b6')
        
        if np.sum(valid4) >= 3:
            z_fit4 = np.polyfit(x4[valid4], y4[valid4], 1)
            p_fit4 = np.poly1d(z_fit4)
            x_line4 = np.linspace(x4[valid4].min(), x4[valid4].max(), 50)
            ax.plot(x_line4, p_fit4(x_line4), 'r--', linewidth=2, alpha=0.7)
            ax.plot([x4[valid4].min(), x4[valid4].max()], 
                   [x4[valid4].min(), x4[valid4].max()], 'k:', alpha=0.5)
        
        ax.set_xlabel('cubes_proactive[week N]', fontproperties=LABEL_FONT)
        ax.set_ylabel('cubes_proactive[week N+1]', fontproperties=LABEL_FONT)
        ax.set_title(f'H4: Автокорреляция проактивности\n(ρ = {rho4:.3f}, p = {p_value4:.4f})', fontproperties=TITLE_FONT)
        
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(TICK_FONT)
        
        plt.tight_layout()
        path = self.save_figure('h4_autocorrelation')
        self.add_paragraph(f"\n![H4 автокорреляция]({path})")
        
        # =========================================================================
        # H5: cubes_proactive[N] → work_life[N] (одновременная связь)
        # =========================================================================
        self.add_section("H5: Связь проактивности с балансом работа/личное (одновременная)", 3)
        
        x5 = lagged_clean['cubes_proactive_N'].values
        y5 = lagged_clean['work_life_N'].values
        
        # Базовая корреляция Спирмена
        rho5, p_value5 = stats.spearmanr(x5, y5)
        
        # Интерпретация
        sig5 = "***" if p_value5 < 0.001 else "**" if p_value5 < 0.01 else "*" if p_value5 < 0.05 else "n.s."
        direction5 = "✅" if rho5 > 0 and p_value5 < 0.05 else "❌" if rho5 <= 0 and p_value5 < 0.05 else "⚪"
        
        self.add_paragraph(f"\n**Результаты:**")
        self.add_paragraph(f"- Корреляция Спирмена: ρ = {rho5:.3f}, p = {p_value5:.4f} {sig5}")
        self.add_paragraph(f"- Направление связи: {'Положительное' if rho5 > 0 else 'Отрицательное' if rho5 < 0 else 'Нет связи'}")
        
        if rho5 > 0 and p_value5 < 0.05:
            self.add_paragraph(f"- {direction5} **Гипотеза H5 ПОДТВЕРЖДАЕТСЯ:** проактивность связана с лучшим балансом работа/личное")
        elif rho5 <= 0 and p_value5 < 0.05:
            self.add_paragraph(f"- {direction5} **Гипотеза H5 Опровергается:** наблюдается отрицательная связь")
        else:
            self.add_paragraph(f"- {direction5} **Гипотеза H5 НЕ подтверждается:** связь не значима")
        
        # Визуализация
        fig, ax = plt.subplots(figsize=(8, 6))
        valid5 = ~(np.isnan(x5) | np.isnan(y5))
        ax.scatter(x5[valid5], y5[valid5], alpha=0.5, s=40, color='#3498db')
        
        if np.sum(valid5) >= 3:
            z_fit5 = np.polyfit(x5[valid5], y5[valid5], 1)
            p_fit5 = np.poly1d(z_fit5)
            x_line5 = np.linspace(x5[valid5].min(), x5[valid5].max(), 50)
            ax.plot(x_line5, p_fit5(x_line5), 'r--', linewidth=2, alpha=0.7)
        
        ax.set_xlabel('cubes_proactive[week N]', fontproperties=LABEL_FONT)
        ax.set_ylabel('work_life[week N]', fontproperties=LABEL_FONT)
        ax.set_title(f'H5: Проактивность vs Баланс (одновременная)\n(ρ = {rho5:.3f}, p = {p_value5:.4f})', fontproperties=TITLE_FONT)
        
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(TICK_FONT)
        
        plt.tight_layout()
        path = self.save_figure('h5_concurrent_work_life')
        self.add_paragraph(f"\n![H5 одновременная связь]({path})")
        
        # =========================================================================
        # H6: cubes_proactive[N] → work_life[N+1] (лаговая связь)
        # =========================================================================
        self.add_section("H6: Влияние проактивности на баланс следующей недели", 3)
        
        x6 = lagged_clean['cubes_proactive_N'].values
        y6 = lagged_clean['work_life_N+1'].values
        
        # Базовая корреляция Спирмена
        rho6, p_value6 = stats.spearmanr(x6, y6)
        
        # Контроль автокорреляции
        z6 = lagged_clean['work_life_N'].values
        valid_idx6 = ~(np.isnan(x6) | np.isnan(y6) | np.isnan(z6))
        
        if np.sum(valid_idx6) >= 30:
            # Ручная вычисление частичной корреляции Спирмена
            r_xy6, _ = stats.spearmanr(x6[valid_idx6], y6[valid_idx6])
            r_xz6, _ = stats.spearmanr(x6[valid_idx6], z6[valid_idx6])
            r_yz6, _ = stats.spearmanr(y6[valid_idx6], z6[valid_idx6])
            n_valid6 = np.sum(valid_idx6)
            numerator6 = r_xy6 - r_xz6 * r_yz6
            denominator6 = np.sqrt((1 - r_xz6**2) * (1 - r_yz6**2))
            partial_rho6 = numerator6 / denominator6 if denominator6 != 0 else 0
            t_stat6 = partial_rho6 * np.sqrt((n_valid6 - 3) / (1 - partial_rho6**2)) if abs(partial_rho6) < 1 else 0
            partial_p6 = 2 * (1 - stats.t.cdf(abs(t_stat6), df=n_valid6-3))
        else:
            partial_rho6, partial_p6 = np.nan, np.nan
        
        # Интерпретация
        sig6 = "***" if p_value6 < 0.001 else "**" if p_value6 < 0.01 else "*" if p_value6 < 0.05 else "n.s."
        direction6 = "✅" if rho6 > 0 and p_value6 < 0.05 else "❌" if rho6 <= 0 and p_value6 < 0.05 else "⚪"
        
        self.add_paragraph(f"\n**Результаты:**")
        self.add_paragraph(f"- Корреляция Спирмена: ρ = {rho6:.3f}, p = {p_value6:.4f} {sig6}")
        self.add_paragraph(f"- Частичная корреляция (контроль автокорреляции): ρ = {partial_rho6:.3f}, p = {partial_p6:.4f}")
        self.add_paragraph(f"- Направление связи: {'Положительное' if rho6 > 0 else 'Отрицательное' if rho6 < 0 else 'Нет связи'}")
        
        if rho6 > 0 and p_value6 < 0.05:
            self.add_paragraph(f"- {direction6} **Гипотеза H6 ПОДТВЕРЖДАЕТСЯ:** проактивность на неделе N предсказывает лучший баланс на неделе N+1")
        elif rho6 <= 0 and p_value6 < 0.05:
            self.add_paragraph(f"- {direction6} **Гипотеза H6 Опровергается:** наблюдается отрицательная связь")
        else:
            self.add_paragraph(f"- {direction6} **Гипотеза H6 НЕ подтверждается:** связь не значима")
        
        # Визуализация
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter plot
        ax1 = axes[0]
        valid6 = ~(np.isnan(x6) | np.isnan(y6))
        ax1.scatter(x6[valid6], y6[valid6], alpha=0.5, s=40, color='#1abc9c')
        
        if np.sum(valid6) >= 3:
            z_fit6 = np.polyfit(x6[valid6], y6[valid6], 1)
            p_fit6 = np.poly1d(z_fit6)
            x_line6 = np.linspace(x6[valid6].min(), x6[valid6].max(), 50)
            ax1.plot(x_line6, p_fit6(x_line6), 'r--', linewidth=2, alpha=0.7)
        
        ax1.set_xlabel('cubes_proactive[week N]', fontproperties=LABEL_FONT)
        ax1.set_ylabel('work_life[week N+1]', fontproperties=LABEL_FONT)
        ax1.set_title(f'H6: Проактивность → Баланс (лаг)\n(ρ = {rho6:.3f}, p = {p_value6:.4f})', fontproperties=TITLE_FONT)
        
        for label in ax1.get_xticklabels() + ax1.get_yticklabels():
            label.set_fontproperties(TICK_FONT)
        
        # Распределение разниц
        ax2 = axes[1]
        if 'work_life_N' in lagged_clean.columns:
            diff6 = lagged_clean['work_life_N+1'] - lagged_clean['work_life_N']
            valid_diff6 = diff6.dropna()
            ax2.hist(valid_diff6.values, bins=15, alpha=0.7, color='#16a085', edgecolor='black')
            ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax2.set_xlabel('Изменение work_life (N+1 - N)', fontproperties=LABEL_FONT)
            ax2.set_ylabel('Частота', fontproperties=LABEL_FONT)
            ax2.set_title('Распределение изменений баланса', fontproperties=TITLE_FONT)
            
            for label in ax2.get_xticklabels() + ax2.get_yticklabels():
                label.set_fontproperties(TICK_FONT)
        
        plt.suptitle('H6: Анализ с лагом', fontproperties=TITLE_FONT, size=16)
        plt.tight_layout()
        path = self.save_figure('h6_lagged_analysis')
        self.add_paragraph(f"\n![H6 анализ]({path})")
        
        # =========================================================================
        # СВОДНАЯ ТАБЛИЦА
        # =========================================================================
        self.add_section("Сводные результаты", 3)
        
        summary_headers = ['Гипотеза', 'Предиктор[week N]', 'Исход[week N+1]', 
                          'Ожидание', 'ρ (Спирмен)', 'p-value', 'Частичная ρ', 'p (частичная)', 'Вывод']
        summary_rows = [
            ['H1', 'cubes_proactive', 'representative', 'Положительная (+)',
             f'{rho:.3f}', f'{p_value:.4f}', f'{partial_rho:.3f}', f'{partial_p:.4f}',
             'Подтверждается' if rho > 0 and p_value < 0.05 else 'Не подтверждается'],
            ['H2', 'cubes_reactive', 'energy_deficit', 'Положительная (+)',
             f'{rho2:.3f}', f'{p_value2:.4f}', f'{partial_rho2:.3f}', f'{partial_p2:.4f}',
             'Подтверждается' if rho2 > 0 and p_value2 < 0.05 else 'Не подтверждается'],
            ['H3', 'cubes_proactive', 'energy_deficit', 'Отрицательная (-)',
             f'{rho3:.3f}', f'{p_value3:.4f}', f'{partial_rho3:.3f}', f'{partial_p3:.4f}',
             'Подтверждается' if rho3 < 0 and p_value3 < 0.05 else 'Не подтверждается'],
            ['H4', 'cubes_proactive', 'cubes_proactive', 'Положительная (+)',
             f'{rho4:.3f}', f'{p_value4:.4f}', '—', '—',
             'Подтверждается' if rho4 > 0 and p_value4 < 0.05 else 'Не подтверждается'],
            ['H5', 'cubes_proactive', 'work_life', 'Положительная (+)',
             f'{rho5:.3f}', f'{p_value5:.4f}', '—', '—',
             'Подтверждается' if rho5 > 0 and p_value5 < 0.05 else 'Не подтверждается'],
            ['H6', 'cubes_proactive', 'work_life', 'Положительная (+)',
             f'{rho6:.3f}', f'{p_value6:.4f}', f'{partial_rho6:.3f}', f'{partial_p6:.4f}',
             'Подтверждается' if rho6 > 0 and p_value6 < 0.05 else 'Не подтверждается'],
        ]
        
        self.add_table(summary_headers, summary_rows)
    
    def analyze_regression_energy_deficit(self):
        """
        Множественная линейная регрессия для предсказания energy_deficit[N+1].
        
        Предикторы:
        - cubes_proactive[N]
        - cubes_reactive[N]
        - energy_deficit[N] (контроль автокорреляции)
        """
        self.add_section("Множественная регрессия: предсказание дефицита энергии", 2)
        
        self.add_paragraph("""
**Цель:** Оценить вклад проактивности и реактивности в дефицит энергии следующей недели,
контролируя базовый уровень дефицита.

**Модель:** `energy_deficit[N+1] = β₀ + β₁×cubes_proactive[N] + β₂×cubes_reactive[N] + β₃×energy_deficit[N]`

**Интерпретация:**
- β₁ < 0: проактивность снижает дефицит энергии
- β₂ > 0: реактивность увеличивает дефицит энергии
- β₃ > 0: автокорреляция дефицита энергии
""")
        
        # Подготовка данных
        lagged = self.create_lagged_dataset()
        lagged_clean = lagged.dropna(subset=[
            'cubes_proactive_N', 'cubes_reactive_N',
            'energy_deficit_N', 'energy_deficit_N+1'
        ]).copy()
        
        n = len(lagged_clean)
        self.add_paragraph(f"**Размер выборки:** n = {n}")
        
        if n < 30:
            self.add_paragraph("⚠️ Недостаточно данных для регрессионного анализа (нужно минимум 30)")
            return
        
        # Предикторы
        X = lagged_clean[['cubes_proactive_N', 'cubes_reactive_N', 'energy_deficit_N']].values
        y = lagged_clean['energy_deficit_N+1'].values
        
        # Добавляем константу
        X_int = np.column_stack([np.ones(n), X])
        
        # Линейная регрессия (MLE через метод наименьших квадратов)
        try:
            beta = np.linalg.lstsq(X_int, y, rcond=None)[0]
            y_pred = X_int @ beta
            
            # R²
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - ss_res / ss_tot
            
            # Adjusted R²
            p_num = 3  # количество предикторов
            adj_r2 = 1 - (1 - r_squared) * (n - 1) / (n - p_num - 1)
            
            # F-статистика
            ms_reg = (ss_tot - ss_res) / p_num
            ms_res = ss_res / (n - p_num - 1)
            f_stat = ms_reg / ms_res
            f_p = 1 - stats.f.cdf(f_stat, p_num, n - p_num - 1)
            
            # Стандартные ошибки коэффициентов
            mse = ss_res / (n - p_num - 1)
            var_beta = mse * np.linalg.inv(X_int.T @ X_int)
            se_beta = np.sqrt(np.diag(var_beta))
            
            # t-статистики и p-значения
            t_stats = beta / se_beta
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - p_num - 1))
            
            # Интерпретация
            self.add_paragraph(f"\n**Результаты регрессии:**")
            self.add_paragraph(f"- R² = {r_squared:.3f}")
            self.add_paragraph(f"- Adjusted R² = {adj_r2:.3f}")
            self.add_paragraph(f"- F({p_num}, {n - p_num - 1}) = {f_stat:.2f}, p = {f_p:.6f}")
            
            # Таблица коэффициентов
            coef_headers = ['Предиктор', 'β', 'SE', 't', 'p', 'Значимость']
            coef_rows = []
            
            pred_names = ['Intercept', 'cubes_proactive[N]', 'cubes_reactive[N]', 'energy_deficit[N]']
            sig_labels = ['***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.' for p in p_values]
            
            for i, (name, b, se, t, p, sig) in enumerate(zip(pred_names, beta, se_beta, t_stats, p_values, sig_labels)):
                coef_rows.append([name, f'{b:.4f}', f'{se:.4f}', f'{t:.2f}', f'{p:.4f}', sig])
            
            self.add_table(coef_headers, coef_rows)
            
            # Выводы по коэффициентам
            self.add_paragraph(f"\n**Интерпретация коэффициентов:**")
            
            # β₁: cubes_proactive
            if p_values[1] < 0.05:
                direction_proactive = 'снижает' if beta[1] < 0 else 'увеличивает'
                self.add_paragraph(f"- **cubes_proactive[N]:** β = {beta[1]:.4f}, p = {p_values[1]:.4f} — проактивность {direction_proactive} дефицит энергии на следующей неделе ✅")
            else:
                self.add_paragraph(f"- **cubes_proactive[N]:** β = {beta[1]:.4f}, p = {p_values[1]:.4f} — не значимый предиктор")
            
            # β₂: cubes_reactive
            if p_values[2] < 0.05:
                direction_reactive = 'снижает' if beta[2] < 0 else 'увеличивает'
                self.add_paragraph(f"- **cubes_reactive[N]:** β = {beta[2]:.4f}, p = {p_values[2]:.4f} — реактивность {direction_reactive} дефицит энергии на следующей неделе ✅")
            else:
                self.add_paragraph(f"- **cubes_reactive[N]:** β = {beta[2]:.4f}, p = {p_values[2]:.4f} — не значимый предиктор")
            
            # β₃: energy_deficit (автокорреляция)
            if p_values[3] < 0.05:
                self.add_paragraph(f"- **energy_deficit[N]:** β = {beta[3]:.4f}, p = {p_values[3]:.4f} — значимая автокорреляция дефицита энергии")
            else:
                self.add_paragraph(f"- **energy_deficit[N]:** β = {beta[3]:.4f}, p = {p_values[3]:.4f} — автокорреляция не значима")
            
            # Визуализация
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Наблюдаемые vs предсказанные
            ax1 = axes[0]
            ax1.scatter(y, y_pred, alpha=0.5, s=40, color='#3498db')
            
            y_min, y_max = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
            ax1.plot([y_min, y_max], [y_min, y_max], 'r--', linewidth=2, alpha=0.7, label='Идеально')
            
            ax1.set_xlabel('Наблюдаемый energy_deficit[N+1]', fontproperties=LABEL_FONT)
            ax1.set_ylabel('Предсказанный', fontproperties=LABEL_FONT)
            ax1.set_title(f'Наблюдаемые vs предсказанные\n(R² = {r_squared:.3f})', fontproperties=TITLE_FONT)
            ax1.legend(prop={'size': 10})
            
            for label in ax1.get_xticklabels() + ax1.get_yticklabels():
                label.set_fontproperties(TICK_FONT)
            
            # Остатки
            ax2 = axes[1]
            residuals = y - y_pred
            ax2.hist(residuals, bins=15, alpha=0.7, color='#2ecc71', edgecolor='black')
            ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax2.set_xlabel('Остатки', fontproperties=LABEL_FONT)
            ax2.set_ylabel('Частота', fontproperties=LABEL_FONT)
            ax2.set_title('Распределение остатков', fontproperties=TITLE_FONT)
            
            for label in ax2.get_xticklabels() + ax2.get_yticklabels():
                label.set_fontproperties(TICK_FONT)
            
            plt.suptitle('Множественная регрессия: energy_deficit[N+1]', fontproperties=TITLE_FONT, size=16)
            plt.tight_layout()
            path = self.save_figure('regression_energy_deficit')
            self.add_paragraph(f"\n![Регрессия]({path})")
            
            # Сравнение с корреляциями
            self.add_paragraph(f"\n**Сравнение с корреляционным анализом:**")
            
            # Вычисляем корреляции для сравнения
            x_proactive = lagged_clean['cubes_proactive_N'].values
            x_reactive = lagged_clean['cubes_reactive_N'].values
            y_energy = lagged_clean['energy_deficit_N+1'].values
            
            rho_corr_proactive, p_corr_proactive = stats.spearmanr(x_proactive, y_energy)
            rho_corr_reactive, p_corr_reactive = stats.spearmanr(x_reactive, y_energy)
            
            self.add_paragraph(f"- В корреляционном анализе без контроля автокорреляции:")
            self.add_paragraph(f"  - cubes_proactive → energy_deficit[N+1]: ρ = {rho_corr_proactive:.3f}, p = {p_corr_proactive:.4f}")
            self.add_paragraph(f"  - cubes_reactive → energy_deficit[N+1]: ρ = {rho_corr_reactive:.3f}, p = {p_corr_reactive:.4f}")
            self.add_paragraph(f"- В регрессии с контролем energy_deficit[N]:")
            self.add_paragraph(f"  - cubes_proactive: β = {beta[1]:.4f}, p = {p_values[1]:.4f}")
            self.add_paragraph(f"  - cubes_reactive: β = {beta[2]:.4f}, p = {p_values[2]:.4f}")
            
            if abs(beta[1]) > 0 and p_values[1] < 0.05:
                self.add_paragraph(f"\n✅ **Вывод:** Проактивность имеет независимый эффект на дефицит энергии после контроля автокорреляции")
            if abs(beta[2]) > 0 and p_values[2] < 0.05:
                self.add_paragraph(f"✅ **Вывод:** Реактивность имеет независимый эффект на дефицит энергии после контроля автокорреляции")
                
        except Exception as e:
            self.add_paragraph(f"⚠️ Ошибка при выполнении регрессии: {e}")
                
    def analyze_regression_work_life(self):
        """
        Множественная линейная регрессия для предсказания work_life[N+1].
        
        Предикторы:
        - cubes_proactive[N]
        - cubes_reactive[N]
        - energy_deficit[N]
        """
        self.add_section("Множественная регрессия: предсказание баланса работа/личное", 2)
        
        self.add_paragraph("""
**Цель:** Оценить вклад проактивности, реактивности и дефицита энергии в баланс работа/личное на следующей неделе.

**Модель:** `work_life[N+1] = β₀ + β₁×cubes_proactive[N] + β₂×cubes_reactive[N] + β₃×energy_deficit[N]`

**Интерпретация:**
- β₁ > 0: проактивность улучшает баланс
- β₂ < 0: реактивность ухудшает баланс
- β₃ < 0: дефицит энергии ухудшает баланс
""")
        
        # Подготовка данных
        lagged = self.create_lagged_dataset()
        lagged_clean = lagged.dropna(subset=[
            'cubes_proactive_N', 'cubes_reactive_N',
            'energy_deficit_N', 'work_life_N+1'
        ]).copy()
        
        n = len(lagged_clean)
        self.add_paragraph(f"**Размер выборки:** n = {n}")
        
        if n < 30:
            self.add_paragraph("⚠️ Недостаточно данных для регрессионного анализа (нужно минимум 30)")
            return
        
        # Предикторы
        X = lagged_clean[['cubes_proactive_N', 'cubes_reactive_N', 'energy_deficit_N']].values
        y = lagged_clean['work_life_N+1'].values
        
        # Добавляем константу
        X_int = np.column_stack([np.ones(n), X])
        
        # Линейная регрессия
        try:
            beta = np.linalg.lstsq(X_int, y, rcond=None)[0]
            y_pred = X_int @ beta
            
            # R²
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - ss_res / ss_tot
            
            # Adjusted R²
            p_num = 3
            adj_r2 = 1 - (1 - r_squared) * (n - 1) / (n - p_num - 1)
            
            # F-статистика
            ms_reg = (ss_tot - ss_res) / p_num
            ms_res = ss_res / (n - p_num - 1)
            f_stat = ms_reg / ms_res
            f_p = 1 - stats.f.cdf(f_stat, p_num, n - p_num - 1)
            
            # Стандартные ошибки
            mse = ss_res / (n - p_num - 1)
            var_beta = mse * np.linalg.inv(X_int.T @ X_int)
            se_beta = np.sqrt(np.diag(var_beta))
            
            # t-статистики и p-значения
            t_stats = beta / se_beta
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - p_num - 1))
            
            # Вывод результатов
            self.add_paragraph(f"\n**Результаты регрессии:**")
            self.add_paragraph(f"- R² = {r_squared:.3f}")
            self.add_paragraph(f"- Adjusted R² = {adj_r2:.3f}")
            self.add_paragraph(f"- F({p_num}, {n - p_num - 1}) = {f_stat:.2f}, p = {f_p:.6f}")
            
            # Таблица коэффициентов
            coef_headers = ['Предиктор', 'β', 'SE', 't', 'p', 'Значимость']
            coef_rows = []
            
            pred_names = ['Intercept', 'cubes_proactive[N]', 'cubes_reactive[N]', 'energy_deficit[N]']
            sig_labels = ['***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.' for p in p_values]
            
            for i, (name, b, se, t, p, sig) in enumerate(zip(pred_names, beta, se_beta, t_stats, p_values, sig_labels)):
                coef_rows.append([name, f'{b:.4f}', f'{se:.4f}', f'{t:.2f}', f'{p:.4f}', sig])
            
            self.add_table(coef_headers, coef_rows)
            
            # Выводы
            self.add_paragraph(f"\n**Интерпретация коэффициентов:**")
            
            if p_values[1] < 0.05:
                direction = 'улучшает' if beta[1] > 0 else 'ухудшает'
                self.add_paragraph(f"- **cubes_proactive[N]:** β = {beta[1]:.4f}, p = {p_values[1]:.4f} — проактивность {direction} баланс ✅")
            else:
                self.add_paragraph(f"- **cubes_proactive[N]:** β = {beta[1]:.4f}, p = {p_values[1]:.4f} — не значимый")
            
            if p_values[2] < 0.05:
                direction = 'улучшает' if beta[2] > 0 else 'ухудшает'
                self.add_paragraph(f"- **cubes_reactive[N]:** β = {beta[2]:.4f}, p = {p_values[2]:.4f} — реактивность {direction} баланс")
            else:
                self.add_paragraph(f"- **cubes_reactive[N]:** β = {beta[2]:.4f}, p = {p_values[2]:.4f} — не значимый")
            
            if p_values[3] < 0.05:
                direction = 'улучшает' if beta[3] > 0 else 'ухудшает'
                self.add_paragraph(f"- **energy_deficit[N]:** β = {beta[3]:.4f}, p = {p_values[3]:.4f} — дефицит энергии {direction} баланс")
            else:
                self.add_paragraph(f"- **energy_deficit[N]:** β = {beta[3]:.4f}, p = {p_values[3]:.4f} — не значимый")
            
            # Визуализация
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            ax1 = axes[0]
            ax1.scatter(y, y_pred, alpha=0.5, s=40, color='#1abc9c')
            
            y_min, y_max = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
            ax1.plot([y_min, y_max], [y_min, y_max], 'r--', linewidth=2, alpha=0.7, label='Идеально')
            
            ax1.set_xlabel('Наблюдаемый work_life[N+1]', fontproperties=LABEL_FONT)
            ax1.set_ylabel('Предсказанный', fontproperties=LABEL_FONT)
            ax1.set_title(f'Наблюдаемые vs предсказанные\n(R² = {r_squared:.3f})', fontproperties=TITLE_FONT)
            ax1.legend(prop={'size': 10})
            
            for label in ax1.get_xticklabels() + ax1.get_yticklabels():
                label.set_fontproperties(TICK_FONT)
            
            ax2 = axes[1]
            residuals = y - y_pred
            ax2.hist(residuals, bins=15, alpha=0.7, color='#3498db', edgecolor='black')
            ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax2.set_xlabel('Остатки', fontproperties=LABEL_FONT)
            ax2.set_ylabel('Частота', fontproperties=LABEL_FONT)
            ax2.set_title('Распределение остатков', fontproperties=TITLE_FONT)
            
            for label in ax2.get_xticklabels() + ax2.get_yticklabels():
                label.set_fontproperties(TICK_FONT)
            
            plt.suptitle('Множественная регрессия: work_life[N+1]', fontproperties=TITLE_FONT, size=16)
            plt.tight_layout()
            path = self.save_figure('regression_work_life')
            self.add_paragraph(f"\n![Регрессия]({path})")
            
            # Итоговые выводы
            self.add_paragraph(f"\n**Итоговые выводы:**")
            sig_proactive = p_values[1] < 0.05 and beta[1] > 0
            sig_reactive = p_values[2] < 0.05 and beta[2] < 0
            sig_deficit = p_values[3] < 0.05 and beta[3] < 0
            
            if sig_proactive:
                self.add_paragraph(f"✅ Проактивность положительно влияет на будущий баланс")
            if sig_reactive:
                self.add_paragraph(f"✅ Реактивность отрицательно влияет на будущий баланс")
            if sig_deficit:
                self.add_paragraph(f"✅ Дефицит энергии отрицательно влияет на будущий баланс")
            
            if not any([sig_proactive, sig_reactive, sig_deficit]):
                self.add_paragraph(f"⚪ Ни один предиктор не показал значимого влияния на баланс")
                
        except Exception as e:
            self.add_paragraph(f"⚠️ Ошибка при выполнении регрессии: {e}")
                
    def generate_descriptive_stats(self):
        """Описательная статистика."""
        self.add_section("Описательная статистика", 2)
        
        self.add_paragraph(f"**Всего наблюдений:** {len(self.data)}")
        self.add_paragraph(f"**Уникальных участников:** {self.data['tg_id'].nunique()}")
        self.add_paragraph(f"**Недели:** {sorted(self.data['week'].dropna().unique())}")
        
        # Статистика по переменным
        self.add_paragraph("\n**Распределение переменных:**\n")
        
        var_stats = []
        for col in ['cubes_reactive', 'cubes_proactive', 'cubes_operational',
                    'representative', 'work_life', 'energy_deficit']:
            if col in self.data.columns:
                vals = self.data[col].dropna()
                var_stats.append([
                    col,
                    f"{vals.mean():.1f}",
                    f"{vals.median():.1f}",
                    f"{vals.std():.2f}",
                    f"[{vals.min():.1f}, {vals.max():.1f}]"
                ])
        
        self.add_table(['Переменная', 'Среднее', 'Медиана', 'Стд. откл.', 'Диапазон'], var_stats)
    
    def generate_report(self) -> str:
        """Генерация полного отчёта."""
        self.report = []
        
        self.report.append(f"""# Отчёт по лонгитюдному пульс-опросу (Q2 2026)

**Дата генерации:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Всего наблюдений:** {len(self.data)}
**Участников:** {self.data['tg_id'].nunique()}
**Недель:** {len(self.data['week'].dropna().unique())}

---

## Введение

Это лонгитюдное исследование с еженедельными пульс-опросами. 
Каждый участник заполнял опрос раз в неделю в течение нескольких недель.

**Цель анализа:** проверить гипотезы с временным лагом — влияние поведения на одной неделе 
на показатели следующей недели.

---
""")
        
        self.generate_descriptive_stats()
        self.analyze_lagged_hypotheses()
        self.analyze_regression_energy_deficit()
        self.analyze_regression_work_life()
        
        self.add_section("Методология", 2)
        self.add_paragraph("""
**Статистические методы:**

1. **Корреляция Спирмена** — непараметрическая мера монотонной связи для ранговых данных
2. **Частичная корреляция** — контроль автокорреляции (влияние базового уровня исхода)
3. **Лаговый анализ** — создание пар наблюдений (week N, week N+1) для каждого участника

**Обоснование выбора методов:**

- Все переменные имеют ранговую природу (ипсативные кубики, шкалы Likert)
- Корреляция Спирмена не требует предположений о нормальности распределения
- Контроль автокорреляции важен для корректной оценки лаговых эффектов

---

**Примечание к интерпретации:**
- ρ (Спирмен): слабая |ρ| < 0.3, средняя 0.3 ≤ |ρ| < 0.5, сильная |ρ| ≥ 0.5
- p < 0.05 — статистически значимый результат
- Частичная корреляция контролирует влияние базового уровня зависимой переменной

---
*Отчёт сгенерирован автоматически*
""")
        
        return "\n".join(self.report)
    
    def save_report(self, output_path: str):
        """Сохранение отчёта в файл."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.generate_report())
        print(f"\nОтчёт сохранён: {output_path}")
        print(f"Графики сохранены в: {FONT_DIR}")


# ============================================================================
# ТОЧКА ВХОДА
# ============================================================================

def main():
    """Основная функция."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Анализ лонгитюдных данных пульс-опроса (Q2 2026)'
    )
    parser.add_argument(
        '--csv', 
        default='./Результаты пульс-опроса - RawData.csv',
        help='Путь к CSV-файлу с данными'
    )
    parser.add_argument(
        '--output', 
        default='pulse_lagged_report.md',
        help='Путь для сохранения отчёта'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv):
        print(f"Ошибка: файл {args.csv} не найден")
        sys.exit(1)
    
    analyzer = PulseAnalyzer(args.csv)
    analyzer.save_report(args.output)


if __name__ == '__main__':
    main()
