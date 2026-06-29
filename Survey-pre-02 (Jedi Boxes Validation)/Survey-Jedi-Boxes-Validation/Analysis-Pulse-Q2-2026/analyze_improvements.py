#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Анализ улучшений по целевым метрикам пульс-опроса.

Целевые метрики:
- cubes_proactive
- satisfaction
- representative
- work_life
- energy_deficit

Методология:
- Для каждого респондента сравниваем первое и последнее измерение
- Рассчитываем абсолютную и относительную разницу
- Классифицируем степень улучшения
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os
from datetime import datetime

# Настройка шрифтов для кириллицы
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ImprovementAnalyzer:
    """Анализатор улучшений по целевым метрикам."""
    
    TARGET_METRICS = ['cubes_proactive', 'satisfaction', 'representative', 
                      'work_life', 'energy_deficit']
    
    # Направления улучшения (положительное изменение = улучшение)
    # Для energy_deficit отрицательное изменение = улучшение (меньше дефицит)
    IMPROVEMENT_DIRECTION = {
        'cubes_proactive': 'positive',
        'satisfaction': 'positive',
        'representative': 'positive',
        'work_life': 'positive',
        'energy_deficit': 'negative'  # меньше дефицит = лучше
    }
    
    def __init__(self, csv_path: str, exclude_weeks: list = None):
        """
        Инициализация анализатора.
        
        Args:
            csv_path: Путь к CSV-файлу с данными
            exclude_weeks: Список недель для исключения (например, [0, 1] если satisfaction не собирался)
        """
        self.data = pd.read_csv(csv_path, encoding='utf-8')
        self.exclude_weeks = exclude_weeks or []
        
        # Фильтрация исключенных недель
        if self.exclude_weeks:
            self.data = self.data[~self.data['week'].isin(self.exclude_weeks)]
            print(f"Исключены недели: {self.exclude_weeks}")
        
        # Удаление строк без tg_id или week
        self.data = self.data.dropna(subset=['tg_id', 'week'])
        
        print(f"Загружено наблюдений: {len(self.data)}")
        print(f"Уникальных участников: {self.data['tg_id'].nunique()}")
        print(f"Уникальных недель: {sorted(self.data['week'].unique())}")
        
        # Доступные целевые метрики
        available_metrics = [m for m in self.TARGET_METRICS if m in self.data.columns]
        print(f"Доступные целевые метрики: {available_metrics}")
        
    def calculate_improvements(self) -> pd.DataFrame:
        """
        Расчет улучшений для каждого респондента по каждой метрике.
        
        Сравнивает первое и последнее измерение для каждого участника.
        
        Returns:
            DataFrame с результатами улучшений
        """
        improvements = []
        
        for tg_id, group in self.data.groupby('tg_id'):
            # Сортировка по неделе
            group = group.sort_values('week').reset_index(drop=True)
            
            if len(group) < 2:
                continue  # Пропускаем участников с одним измерением
            
            first_week = group.iloc[0]['week']
            last_week = group.iloc[-1]['week']
            
            # Проверяем, есть ли хотя бы 2 разные недели
            if first_week == last_week:
                continue
            
            participant_data = {'tg_id': tg_id, 'n_measurements': len(group),
                               'first_week': first_week, 'last_week': last_week}
            
            for metric in self.TARGET_METRICS:
                if metric not in group.columns:
                    continue
                
                first_value = group.iloc[0][metric]
                last_value = group.iloc[-1][metric]
                
                # Пропускаем если есть пропуски
                if pd.isna(first_value) or pd.isna(last_value):
                    participant_data[f'{metric}_improved'] = np.nan
                    participant_data[f'{metric}_delta'] = np.nan
                    participant_data[f'{metric}_category'] = np.nan
                    continue
                
                delta = last_value - first_value
                
                # Определяем направление улучшения
                direction = self.IMPROVEMENT_DIRECTION.get(metric, 'positive')
                if direction == 'positive':
                    improved = delta > 0
                else:  # negative
                    improved = delta < 0
                
                # Классификация степени улучшения
                # Используем процентное изменение относительно диапазона шкалы
                if metric == 'energy_deficit':
                    # Для energy_deficit: отрицательное delta = улучшение
                    if delta < -1:
                        category = 'strong_improvement'
                    elif delta < -0.3:
                        category = 'moderate_improvement'
                    elif abs(delta) <= 0.3:
                        category = 'no_change'
                    elif delta <= 1:
                        category = 'moderate_decline'
                    else:
                        category = 'strong_decline'
                else:
                    # Для остальных метрик: положительное delta = улучшение
                    if delta > 1:
                        category = 'strong_improvement'
                    elif delta > 0.3:
                        category = 'moderate_improvement'
                    elif abs(delta) <= 0.3:
                        category = 'no_change'
                    elif delta >= -1:
                        category = 'moderate_decline'
                    else:
                        category = 'strong_decline'
                
                participant_data[f'{metric}_improved'] = improved
                participant_data[f'{metric}_delta'] = delta
                participant_data[f'{metric}_category'] = category
            
            improvements.append(participant_data)
        
        improvements_df = pd.DataFrame(improvements)
        print(f"\nРеспондентов с >= 2 измерениями: {len(improvements_df)}")
        
        return improvements_df
    
    def generate_summary_stats(self, improvements_df: pd.DataFrame):
        """Генерация сводной статистики по улучшениям."""
        print("\n" + "="*80)
        print("СВОДНАЯ СТАТИСТИКА УЛУЧШЕНИЙ")
        print("="*80)
        
        for metric in self.TARGET_METRICS:
            if f'{metric}_category' not in improvements_df.columns:
                print(f"\n{metric}: ДАННЫЕ ОТСУТСТВУЮТ")
                continue
            
            categories = improvements_df[f'{metric}_category'].dropna()
            if len(categories) == 0:
                print(f"\n{metric}: НЕТ ВАЛИДНЫХ ДАННЫХ")
                continue
            
            total = len(categories)
            
            # Подсчет по категориям
            strong_imp = (categories == 'strong_improvement').sum()
            mod_imp = (categories == 'moderate_improvement').sum()
            no_change = (categories == 'no_change').sum()
            mod_decl = (categories == 'moderate_decline').sum()
            strong_decl = (categories == 'strong_decline').sum()
            
            # Процент улучшившихся (умеренно + сильно)
            improved_pct = ((strong_imp + mod_imp) / total * 100) if total > 0 else 0
            
            # Среднее изменение
            avg_delta = improvements_df[f'{metric}_delta'].mean()
            
            print(f"\n{metric}:")
            print(f"  Всего респондентов: {total}")
            print(f"  Сильно улучшились: {strong_imp} ({strong_imp/total*100:.1f}%)")
            print(f"  Умеренно улучшились: {mod_imp} ({mod_imp/total*100:.1f}%)")
            print(f"  Без изменений: {no_change} ({no_change/total*100:.1f}%)")
            print(f"  Умеренно ухудшились: {mod_decl} ({mod_decl/total*100:.1f}%)")
            print(f"  Сильно ухудшились: {strong_decl} ({strong_decl/total*100:.1f}%)")
            print(f"  ИТОГО улучшились: {strong_imp + mod_imp} ({improved_pct:.1f}%)")
            print(f"  Среднее изменение: {avg_delta:.3f}")
    
    def create_visualizations(self, improvements_df: pd.DataFrame):
        """Создание визуализаций улучшений."""
        figures_dir = 'improvement_figures'
        os.makedirs(figures_dir, exist_ok=True)
        
        # 1. Сводная диаграмма по всем метрикам
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        category_order = ['strong_improvement', 'moderate_improvement', 'no_change', 
                         'moderate_decline', 'strong_decline']
        category_labels = {
            'strong_improvement': 'Сильно улучшились',
            'moderate_improvement': 'Умеренно улучшились',
            'no_change': 'Без изменений',
            'moderate_decline': 'Умеренно ухудшились',
            'strong_decline': 'Сильно ухудшились'
        }
        
        colors = {
            'strong_improvement': '#27ae60',
            'moderate_improvement': '#2ecc71',
            'no_change': '#95a5a6',
            'moderate_decline': '#e74c3c',
            'strong_decline': '#c0392b'
        }
        
        for idx, metric in enumerate(self.TARGET_METRICS):
            if f'{metric}_category' not in improvements_df.columns:
                continue
            
            categories = improvements_df[f'{metric}_category'].dropna()
            if len(categories) == 0:
                continue
            
            ax = axes[idx]
            
            # Подсчет категорий
            counts = [((categories == cat).sum()) for cat in category_order]
            
            # Сортированный bar chart
            x = range(len(category_order))
            bars = ax.bar(x, counts, color=[colors[cat] for cat in category_order])
            
            ax.set_xticks(x)
            ax.set_xticklabels([category_labels[cat] for cat in category_order], 
                              rotation=45, ha='right')
            ax.set_title(f'{metric}\n(n={len(categories)})', fontsize=12)
            ax.set_ylabel('Количество респондентов')
            
            # Добавление значений на столбцы
            for bar, count in zip(bars, counts):
                if count > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           str(count), ha='center', va='bottom', fontsize=9)
        
        # Последняя пустая ячейка
        axes[-1].axis('off')
        
        plt.suptitle('Распределение улучшений по целевым метрикам', fontsize=14, y=1.02)
        plt.tight_layout()
        path = os.path.join(figures_dir, 'improvement_distribution.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"\nСохранено: {path}")
        
        # 2. Корреляция между улучшениями разных метрик
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Выбираем метрики с данными
        available_deltas = [f'{m}_delta' for m in self.TARGET_METRICS 
                           if f'{m}_delta' in improvements_df.columns]
        available_deltas = [d for d in available_deltas 
                           if improvements_df[d].notna().sum() > 10]
        
        if len(available_deltas) >= 2:
            corr_data = improvements_df[available_deltas].dropna()
            if len(corr_data) > 10:
                corr_matrix = corr_data.corr()
                
                im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
                
                # Подписи
                ax.set_xticks(range(len(available_deltas)))
                ax.set_yticks(range(len(available_deltas)))
                ax.set_xticklabels([d.replace('_delta', '') for d in available_deltas], 
                                  rotation=45, ha='right')
                ax.set_yticklabels([d.replace('_delta', '') for d in available_deltas])
                
                # Добавление значений в клетки
                for i in range(len(available_deltas)):
                    for j in range(len(available_deltas)):
                        text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                     ha='center', va='center', color='black')
                
                plt.colorbar(im, ax=ax, label='Корреляция Пирсона')
                ax.set_title('Корреляция между улучшениями метрик', fontsize=12)
                
                path = os.path.join(figures_dir, 'improvement_correlations.png')
                plt.savefig(path, dpi=150, bbox_inches='tight')
                print(f"Сохранено: {path}")
        
        plt.close('all')
    
    def generate_detailed_report(self, improvements_df: pd.DataFrame):
        """Генерация детального отчета в Markdown."""
        report = []
        
        report.append(f"# Анализ улучшений по целевым метрикам\n\n")
        report.append(f"**Дата генерации:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        report.append(f"**Всего наблюдений:** {len(self.data)}\n")
        report.append(f"**Участников с >= 2 измерениями:** {len(improvements_df)}\n")
        if self.exclude_weeks:
            report.append(f"**Исключены недели:** {self.exclude_weeks}\n")
        report.append("\n---\n\n")
        
        # Методология
        report.append("## Методология\n\n")
        report.append("**Сравнение:** Первое vs Последнее измерение для каждого участника\n\n")
        report.append("**Категории улучшений:**\n")
        report.append("- **Сильно улучшились:** delta > 1 (или delta < -1 для energy_deficit)\n")
        report.append("- **Умеренно улучшились:** 0.3 < delta <= 1 (или -1 <= delta < -0.3)\n")
        report.append("- **Без изменений:** |delta| <= 0.3\n")
        report.append("- **Умеренно ухудшились:** -1 <= delta < -0.3 (или 0.3 > delta >= -1)\n")
        report.append("- **Сильно ухудшились:** delta < -1 (или delta > 1)\n\n")
        
        report.append("**Направление улучшения:**\n")
        for metric, direction in self.IMPROVEMENT_DIRECTION.items():
            if direction == 'positive':
                report.append(f"- `{metric}`: увеличение = улучшение\n")
            else:
                report.append(f"- `{metric}`: уменьшение = улучшение (меньше дефицит)\n")
        
        # Сводная таблица
        report.append("\n## Сводные результаты\n\n")
        
        headers = ['Метрика', 'Всего', 'Сильно улучшились', 'Умеренно улучшились', 
                  'Итого улучшились', 'Без изменений', 'Ухудшились', 'Среднее изменение']
        
        rows = []
        for metric in self.TARGET_METRICS:
            if f'{metric}_category' not in improvements_df.columns:
                continue
            
            categories = improvements_df[f'{metric}_category'].dropna()
            if len(categories) == 0:
                continue
            
            total = len(categories)
            strong_imp = (categories == 'strong_improvement').sum()
            mod_imp = (categories == 'moderate_improvement').sum()
            no_change = (categories == 'no_change').sum()
            mod_decl = (categories == 'moderate_decline').sum()
            strong_decl = (categories == 'strong_decline').sum()
            total_improved = strong_imp + mod_imp
            total_declined = mod_decl + strong_decl
            avg_delta = improvements_df[f'{metric}_delta'].mean()
            
            rows.append([
                metric,
                total,
                f"{strong_imp} ({strong_imp/total*100:.1f}%)",
                f"{mod_imp} ({mod_imp/total*100:.1f}%)",
                f"{total_improved} ({total_improved/total*100:.1f}%)",
                f"{no_change} ({no_change/total*100:.1f}%)",
                f"{total_declined} ({total_declined/total*100:.1f}%)",
                f"{avg_delta:.3f}"
            ])
        
        # Markdown таблица
        header_row = "| " + " | ".join(headers) + " |"
        report.append(header_row)
        sep_row = "| " + " | ".join([":" + "-" * (len(h) + 1) for h in headers]) + " |"
        report.append(sep_row)
        for row in rows:
            data_row = "| " + " | ".join(str(cell) for cell in row) + " |"
            report.append(data_row)
        
        # Детальный анализ по каждой метрике
        for metric in self.TARGET_METRICS:
            if f'{metric}_delta' not in improvements_df.columns:
                continue
            
            deltas = improvements_df[f'{metric}_delta'].dropna()
            if len(deltas) == 0:
                continue
            
            report.append(f"\n## Детальный анализ: {metric}\n\n")
            
            # Статистика распределения
            report.append(f"**Количество респондентов:** {len(deltas)}\n\n")
            report.append(f"- Среднее изменение: {deltas.mean():.3f}\n")
            report.append(f"- Медиана: {deltas.median():.3f}\n")
            report.append(f"- Стд. отклонение: {deltas.std():.3f}\n")
            report.append(f"- Минимум: {deltas.min():.3f}\n")
            report.append(f"- Максимум: {deltas.max():.3f}\n")
            
            # Проверка значимости (t-тест против 0)
            from scipy import stats
            t_stat, p_value = stats.ttest_rel(deltas, np.zeros(len(deltas)))
            report.append(f"- t-статистика: {t_stat:.3f}\n")
            report.append(f"- p-value: {p_value:.4f}\n")
            
            if p_value < 0.05:
                if deltas.mean() > 0:
                    report.append(f"\n✅ **Значимое положительное изменение** (p < 0.05)\n")
                else:
                    report.append(f"\n⚠️ **Значимое отрицательное изменение** (p < 0.05)\n")
            else:
                report.append(f"\n⚪ **Изменение не статистически значимо** (p >= 0.05)\n")
        
        return "\n".join(report)
    
    def run_full_analysis(self, output_md: str = 'improvement_report.md'):
        """Запуск полного анализа."""
        print("\n" + "="*80)
        print("ЗАПУСК ПОЛНОГО АНАЛИЗА УЛУЧШЕНИЙ")
        print("="*80)
        
        # Расчет улучшений
        improvements_df = self.calculate_improvements()
        
        # Сводная статистика
        self.generate_summary_stats(improvements_df)
        
        # Визуализации
        self.create_visualizations(improvements_df)
        
        # Отчет
        report = self.generate_detailed_report(improvements_df)
        
        # Сохранение отчета
        with open(output_md, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nОтчет сохранен: {output_md}")
        
        return improvements_df


def main():
    """Основная функция."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Анализ улучшений по целевым метрикам пульс-опроса'
    )
    parser.add_argument(
        '--csv',
        default='./Результаты пульс-опроса - RawData.csv',
        help='Путь к CSV-файлу с данными'
    )
    parser.add_argument(
        '--output',
        default='improvement_report.md',
        help='Путь для сохранения отчета'
    )
    parser.add_argument(
        '--exclude-weeks',
        type=int,
        nargs='+',
        default=[0, 1],
        help='Недели для исключения (например, --exclude-weeks 0 1)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv):
        print(f"Ошибка: файл {args.csv} не найден")
        return
    
    # Запуск анализа
    analyzer = ImprovementAnalyzer(args.csv, exclude_weeks=args.exclude_weeks)
    analyzer.run_full_analysis(args.output)


if __name__ == '__main__':
    main()
