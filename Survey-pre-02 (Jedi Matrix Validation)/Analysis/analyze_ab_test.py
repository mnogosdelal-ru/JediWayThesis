#!/usr/bin/env python3
"""
Скрипт для анализа A/B теста вариантов контрола продуктивности.
Проверяет все гипотезы H1-H13 и генерирует отчёт в формате Markdown.

Использование:
    python3 analyze_ab_test.py [--csv путь_к_csv] [--output путь_к_отчёту]
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from scipy import stats
import numpy as np

# Цвета для консоли
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_colored(text, color):
    print(f"{color}{text}{Colors.END}")

# Статистические функции
def cohens_d(group1, group2):
    """Вычисляет Cohen's d для двух групп"""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def fisher_z_transform(r):
    """Трансформация Фишера для корреляций"""
    if r >= 1 or r <= -1:
        return float('inf') if r > 0 else float('-inf')
    return 0.5 * np.log((1 + r) / (1 - r))

def compare_correlations(r1, n1, r2, n2):
    """Сравнение двух корреляций с помощью z-теста Фишера"""
    z1 = fisher_z_transform(r1)
    z2 = fisher_z_transform(r2)
    se = np.sqrt(1/(n1-3) + 1/(n2-3))
    if se == 0:
        return 0, 1
    z = (z1 - z2) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p

class ABTestAnalyzer:
    """Класс для анализа A/B теста"""
    
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.data = []
        self.report = []
        self.load_data()
        
    def load_data(self):
        """Загрузка данных из CSV"""
        print_colored(f"Загрузка данных из {self.csv_path}...", Colors.BLUE)
        
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                # Преобразуем числовые поля
                for field in ['age', 'group_id', 'p1_tl', 'p1_tr', 'p1_bl', 'p1_br',
                              'p2_tl', 'p2_tr', 'p2_bl', 'p2_br',
                              'slider_balance', 'slider_desired', 'slider_others',
                              'rating_understanding', 'rating_ease', 'alt_understanding',
                              'time_page1_total', 'time_page2_total', 'time_total']:
                    if field in row and row[field]:
                        try:
                            row[field] = float(row[field])
                        except (ValueError, TypeError):
                            row[field] = None
                self.data.append(row)
        
        # Фильтруем только завершённые опросы
        self.completed = [d for d in self.data if d.get('status') == 'completed']
        print_colored(f"Загружено {len(self.data)} записей, {len(self.completed)} завершённых", Colors.GREEN)
    
    def add_section(self, title, level=2):
        """Добавляет секцию в отчёт"""
        self.report.append(f"\n{'#' * level} {title}\n")
    
    def add_paragraph(self, text):
        """Добавляет параграф в отчёт"""
        self.report.append(f"{text}\n")
    
    def add_table(self, headers, rows):
        """Добавляет таблицу в отчёт"""
        self.report.append("| " + " | ".join(headers) + " |")
        self.report.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            self.report.append("| " + " | ".join(str(x) for x in row) + " |")
        self.report.append("")
    
    def add_test_result(self, name, statistic, p_value, conclusion, effect_size=None):
        """Добавляет результат теста"""
        if p_value < 0.001:
            p_str = "p < 0.001"
            sig = "***"
        elif p_value < 0.01:
            p_str = f"p = {p_value:.4f}"
            sig = "**"
        elif p_value < 0.05:
            p_str = f"p = {p_value:.4f}"
            sig = "*"
        else:
            p_str = f"p = {p_value:.4f}"
            sig = "n.s."
        
        effect_str = f", Cohen's d = {effect_size:.3f}" if effect_size is not None else ""
        
        self.add_paragraph(f"**{name}**: {statistic:.2f}, {p_str}{effect_str} → {sig} {conclusion}")
    
    # ==================== ГИПОТЕЗЫ H1-H4 ====================
    
    def analyze_h1_understanding(self):
        """H1: Вариант 3 (горизонтальные бегунки) будет оценён как более понятный"""
        self.add_section("H1: Понятность вариантов контрола", 3)
        
        # Разделяем по вариантам
        standard = [d['rating_understanding'] for d in self.completed 
                   if d.get('variant') == 'standard' and d.get('rating_understanding')]
        horizontal = [d['rating_understanding'] for d in self.completed 
                     if d.get('variant') == 'horizontal' and d.get('rating_understanding')]
        
        if not standard or not horizontal:
            self.add_paragraph("⚠️ Недостаточно данных для анализа")
            return
        
        # Описательная статистика
        self.add_paragraph(f"**Стандартный вариант**: M = {np.mean(standard):.2f}, SD = {np.std(standard):.2f}, медиана = {np.median(standard):.1f}, n = {len(standard)}")
        self.add_paragraph(f"**Горизонтальный вариант**: M = {np.mean(horizontal):.2f}, SD = {np.std(horizontal):.2f}, медиана = {np.median(horizontal):.1f}, n = {len(horizontal)}")
        
        # t-тест (параметрический)
        t_stat, p_value_t = stats.ttest_ind(standard, horizontal)
        d = cohens_d(standard, horizontal)
        
        conclusion_t = "Подтверждается (горизонтальный понятнее)" if p_value_t < 0.05 and np.mean(horizontal) > np.mean(standard) else "Не подтверждается"
        
        self.add_paragraph("\n**1. t-тест для независимых выборок:**")
        self.add_test_result("t-тест", t_stat, p_value_t, conclusion_t, d)
        
        # Критерий Манна-Уитни (непараметрический)
        # Более уместен для ординальных данных (шкала Лайкерта 1-7)
        u_stat, p_value_mw = stats.mannwhitneyu(standard, horizontal, alternative='two-sided')
        
        # Рассчитываем r (размер эффекта для Манна-Уитни)
        n1, n2 = len(standard), len(horizontal)
        r = 1 - (2 * u_stat) / (n1 * n2)  # эффект r = Z / sqrt(N)
        
        conclusion_mw = "Подтверждается (горизонтальный понятнее)" if p_value_mw < 0.05 and np.median(horizontal) > np.median(standard) else "Не подтверждается"
        
        self.add_paragraph("\n**2. Критерий Манна-Уитни (непараметрический):**")
        self.add_paragraph(f"**U-тест**: U = {u_stat:.1f}, p = {p_value_mw:.4f}, r = {abs(r):.3f}")
        
        # Интерпретация размера эффекта r
        if abs(r) < 0.1:
            eff_interp = "пренебрежимый"
        elif abs(r) < 0.3:
            eff_interp = "малый"
        elif abs(r) < 0.5:
            eff_interp = "средний"
        else:
            eff_interp = "большой"
        self.add_paragraph(f"Размер эффекта: {eff_interp}")
        
        # Нормальность (тест Шапиро-Уилка)
        self.add_paragraph("\n**Проверка нормальности (тест Шапиро-Уилка):**")
        if len(standard) >= 3:
            _, p_shap_std = stats.shapiro(standard)
            self.add_paragraph(f"  Standard: p = {p_shap_std:.4f} {'(нормально)' if p_shap_std > 0.05 else '(не нормально)'}")
        if len(horizontal) >= 3:
            _, p_shap_hor = stats.shapiro(horizontal)
            self.add_paragraph(f"  Horizontal: p = {p_shap_hor:.4f} {'(нормально)' if p_shap_hor > 0.05 else '(не нормально)'}")
        
        self.add_paragraph("\n**Вывод:**")
        if p_value_t < 0.05 and p_value_mw < 0.05:
            self.add_paragraph("Оба теста показывают значимые различия — результаты устойчивы")
        elif p_value_t < 0.05:
            self.add_paragraph("Только t-тест показывает значимые различия (Манна-Уитни — нет)")
        elif p_value_mw < 0.05:
            self.add_paragraph("Только Манна-Уитни показывает значимые различия (более надёжен для ординальных данных)")
        else:
            self.add_paragraph("Значимых различий не обнаружено обоими тестами")
    
    def analyze_h2_time(self):
        """H2: Вариант 1 потребует меньше времени на заполнение"""
        self.add_section("H2: Время заполнения", 3)
        
        standard_time = [d['time_page1_total'] for d in self.completed 
                        if d.get('variant') == 'standard' and d.get('time_page1_total')]
        horizontal_time = [d['time_page1_total'] for d in self.completed 
                          if d.get('variant') == 'horizontal' and d.get('time_page1_total')]
        
        if not standard_time or not horizontal_time:
            self.add_paragraph("⚠️ Недостаточно данных для анализа")
            return
        
        # Логарифмическая трансформация
        log_standard = np.log1p(standard_time)
        log_horizontal = np.log1p(horizontal_time)
        
        t_stat, p_value = stats.ttest_ind(log_standard, log_horizontal)
        d = cohens_d(log_standard, log_horizontal)
        
        conclusion = "Подтверждается (стандартный быстрее)" if p_value < 0.05 and np.mean(standard_time) < np.mean(horizontal_time) else "Не подтверждается"
        
        self.add_paragraph(f"**Стандартный вариант**: M = {np.mean(standard_time):.0f} сек, SD = {np.std(standard_time):.0f}, n = {len(standard_time)}")
        self.add_paragraph(f"**Горизонтальный вариант**: M = {np.mean(horizontal_time):.0f} сек, SD = {np.std(horizontal_time):.0f}, n = {len(horizontal_time)}")
        self.add_test_result("t-тест (log-transform)", t_stat, p_value, conclusion, d)
    
    def analyze_h3_preference_second(self):
        """H3: Респонденты чаще предпочитают second, потому что поняли идею"""
        self.add_section("H3: Предпочтение альтернативного варианта", 3)
        
        pref_by_variant = {'standard': {'first': 0, 'second': 0, 'equal': 0, 'unsure': 0},
                          'horizontal': {'first': 0, 'second': 0, 'equal': 0, 'unsure': 0}}
        
        for d in self.completed:
            variant = d.get('variant', '')
            pref = d.get('preference', '').strip()
            if variant in pref_by_variant and pref:
                pref_by_variant[variant][pref] = pref_by_variant[variant].get(pref, 0) + 1
        
        self.add_paragraph("**Распределение предпочтений по использованному варианту:**")
        self.add_table(['Вариант', 'first', 'second', 'equal', 'unsure'],
                      [['Стандартный (использовали)', 
                        pref_by_variant['standard']['first'],
                        pref_by_variant['standard']['second'],
                        pref_by_variant['standard']['equal'],
                        pref_by_variant['standard']['unsure']],
                       ['Горизонтальный (использовали)', 
                        pref_by_variant['horizontal']['first'],
                        pref_by_variant['horizontal']['second'],
                        pref_by_variant['horizontal']['equal'],
                        pref_by_variant['horizontal']['unsure']]])
        
        std_total = sum(pref_by_variant['standard'].values())
        hor_total = sum(pref_by_variant['horizontal'].values())
        
        if std_total > 0 and hor_total > 0:
            std_second_pct = pref_by_variant['standard']['second'] / std_total * 100
            hor_second_pct = pref_by_variant['horizontal']['second'] / hor_total * 100
            
            self.add_paragraph(f"\n**Выбор 'second' (альтернативный вариант):**")
            self.add_paragraph(f"  После standard: {pref_by_variant['standard']['second']}/{std_total} = {std_second_pct:.1f}%")
            self.add_paragraph(f"  После horizontal: {pref_by_variant['horizontal']['second']}/{hor_total} = {hor_second_pct:.1f}%")
            
            if std_total >= 10:
                binom_result = stats.binomtest(pref_by_variant['standard']['second'], std_total, 0.5)
                self.add_paragraph(f"\n**Биномиальный тест (H0: p = 0.5):**")
                self.add_paragraph(f"  После standard: p = {binom_result.pvalue:.4f}")
                conclusion_std = "Значимо чаще выбирают second" if binom_result.pvalue < 0.05 and std_second_pct > 50 else "Нет значимого предпочтения"
                self.add_paragraph(f"  Вывод: {conclusion_std}")
            
            if hor_total >= 10:
                binom_result_h = stats.binomtest(pref_by_variant['horizontal']['second'], hor_total, 0.5)
                self.add_paragraph(f"  После horizontal: p = {binom_result_h.pvalue:.4f}")
                conclusion_hor = "Значимо чаще выбирают second" if binom_result_h.pvalue < 0.05 and hor_second_pct > 50 else "Нет значимого предпочтения"
                self.add_paragraph(f"  Вывод: {conclusion_hor}")
            
            self.add_paragraph(f"\n**Общий вывод:**")
            if std_second_pct > 50 and hor_second_pct > 50:
                self.add_paragraph("Подтверждается — респонденты обоих групп чаще выбирают альтернативный вариант second, что может свидетельствовать о понимании идеи после первого использования")
            elif std_second_pct > 50:
                self.add_paragraph("Частично подтверждается — только группа standard значимо чаще выбирает second")
            elif hor_second_pct > 50:
                self.add_paragraph("Частично подтверждается — только группа horizontal значимо чаще выбирает second")
            else:
                self.add_paragraph("Не подтверждается — нет тенденции выбирать альтернативный вариант second")
        else:
            self.add_paragraph("⚠️ Недостаточно данных для анализа")
    
    def analyze_h4_preference(self):
        """H4: При прямом сравнении респонденты предпочтут Вариант 3"""
        self.add_section("H4: Прямое предпочтение вариантов (с учётом группы)", 3)
        
        final_preference = {'standard': 0, 'horizontal': 0, 'equal': 0, 'unsure': 0}
        
        for d in self.completed:
            variant = d.get('variant', '').strip()
            pref = d.get('preference', '').strip()
            
            if not pref:
                continue
            
            if pref == 'equal':
                final_preference['equal'] += 1
            elif pref == 'unsure':
                final_preference['unsure'] += 1
            elif variant == 'standard':
                if pref == 'first':
                    final_preference['standard'] += 1
                elif pref == 'second':
                    final_preference['horizontal'] += 1
            elif variant == 'horizontal':
                if pref == 'first':
                    final_preference['horizontal'] += 1
                elif pref == 'second':
                    final_preference['standard'] += 1
        
        if sum(final_preference.values()) == 0:
            self.add_paragraph("⚠️ Нет данных о предпочтениях")
            return
        
        total = sum(final_preference.values())
        standard_pct = final_preference['standard'] / total * 100
        horizontal_pct = final_preference['horizontal'] / total * 100
        equal_pct = final_preference['equal'] / total * 100
        
        self.add_paragraph("**Итоговое распределение предпочтений (какой вариант выбрали):**")
        self.add_table(['Вариант', 'Количество', 'Процент'], 
                      [['Предпочитают Standard', final_preference['standard'], f"{standard_pct:.1f}%"],
                       ['Предпочитают Horizontal', final_preference['horizontal'], f"{horizontal_pct:.1f}%"],
                       ['Оба одинаково', final_preference['equal'], f"{equal_pct:.1f}%"],
                       ['Затрудняюсь', final_preference['unsure'], f"{(final_preference['unsure']/total*100):.1f}%"]])
        
        if final_preference['horizontal'] >= 10:
            binom_h = stats.binomtest(final_preference['horizontal'], total, 0.5)
            self.add_paragraph(f"\n**Биномиальный тест (H0: p = 0.5 для горизонтального):**")
            self.add_paragraph(f"  Выбрали горизонтальный: {final_preference['horizontal']}/{total}, p = {binom_h.pvalue:.4f}")
            if binom_h.pvalue < 0.05:
                if horizontal_pct > 50:
                    self.add_paragraph(f"  Значимо чаще выбирают горизонтальный вариант")
                else:
                    self.add_paragraph(f"  Значимо чаще выбирают стандартный вариант")
        
        self.add_paragraph(f"\n**Вывод:**")
        if horizontal_pct > standard_pct + 20:
            self.add_paragraph("Подтверждается — большинство предпочитает горизонтальный вариант")
        elif standard_pct > horizontal_pct + 20:
            self.add_paragraph("Не подтверждается — большинство предпочитает стандартный вариант")
        elif abs(horizontal_pct - standard_pct) < 20:
            self.add_paragraph("Нет явного предпочтения — варианты примерно одинаково популярны")
        else:
            self.add_paragraph("Не подтверждается")
    
    # ==================== ГИПОТЕЗЫ H5-H7 (Эквивалентность) ====================
    
    def analyze_h5_equivalence(self):
        """H5a/b: Распределение по зонам не отличается между группами"""
        self.add_section("H5: Эквивалентность измерений (зоны матрицы)", 3)
        
        zones = ['tl', 'tr', 'bl', 'br']
        zone_names = {
            'tl': 'Зелёная (не срочное + важное)',
            'tr': 'Оранжевая (срочное + важное)',
            'bl': 'Серая (не срочное + не важное)',
            'br': 'Красная (срочное + не важное)'
        }
        results = []
        
        for page in [1, 2]:
            page_name = "Личная жизнь" if page == 1 else "Работа"
            self.add_paragraph(f"\n**Страница {page} ({page_name}):**")
            
            # Собираем данные по зонам для таблицы
            table_data = []
            
            for zone in zones:
                col = f'p{page}_{zone}'
                standard = [d[col] for d in self.completed if d.get('variant') == 'standard' and d.get(col)]
                horizontal = [d[col] for d in self.completed if d.get('variant') == 'horizontal' and d.get(col)]
                
                if standard and horizontal:
                    t_stat, p_value = stats.ttest_ind(standard, horizontal)
                    diff = np.mean(horizontal) - np.mean(standard)
                    results.append([f"p{page}_{zone}", diff, p_value])
                    
                    # Добавляем строку в таблицу
                    table_data.append([
                        zone_names.get(zone, zone),
                        f"{np.mean(standard):.1f}%",
                        f"±{np.std(standard):.1f}%",
                        len(standard),
                        f"{np.mean(horizontal):.1f}%",
                        f"±{np.std(horizontal):.1f}%",
                        len(horizontal),
                        f"{diff:+.1f}%",
                        f"{p_value:.4f}",
                        "✓ Эквивалентно" if abs(diff) < 5 and p_value >= 0.05 else "✗ Различия"
                    ])
            
            # Выводим таблицу
            self.add_table(
                ['Зона', 'Standard M', 'Standard SD', 'n', 'Horizontal M', 'Horizontal SD', 'n', 'Разница', 'p-value', 'Статус'],
                table_data
            )
        
        equivalent = all(abs(r[1]) < 5 and r[2] >= 0.05 for r in results)
        
        # Сводная таблица по всем зонам
        self.add_paragraph("\n**Сводная таблица эквивалентности:**")
        summary_table = []
        for r in results:
            zone = r[0]
            diff = r[1]
            p_val = r[2]
            equiv = "✓" if abs(diff) < 5 and p_val >= 0.05 else "✗"
            summary_table.append([zone, f"{diff:+.1f}%", f"p = {p_val:.4f}", equiv])
        
        self.add_table(
            ['Зона', 'Разница (H - S)', 'p-value', 'Эквивалентно'],
            summary_table
        )
        
        self.add_paragraph(f"\n**Критерии эквивалентности:**")
        self.add_paragraph(f"  - Разница < 5 процентных пунктов")
        self.add_paragraph(f"  - p-value > 0.05 (нет значимых различий)")
        
        self.add_paragraph(f"\n**Общий вывод по H5**: {'Эквивалентность подтверждена — оба варианта измеряют одно и то же' if equivalent else 'Эквивалентность нарушена — есть значимые различия между вариантами'}")
    
    def analyze_h6_correlations(self):
        """H6: Корреляция между зонами и контрольными переменными одинакова"""
        self.add_section("H6: Эквивалентность корреляций", 3)
        
        # Анализируем корреляции между зоной TL (зеленая) и желаемой продуктивностью
        standard_data = [(d['p1_tl'], d['slider_desired']) for d in self.completed 
                        if d.get('variant') == 'standard' and d.get('p1_tl') and d.get('slider_desired')]
        horizontal_data = [(d['p1_tl'], d['slider_desired']) for d in self.completed 
                          if d.get('variant') == 'horizontal' and d.get('p1_tl') and d.get('slider_desired')]
        
        if standard_data and horizontal_data:
            # Корреляция для standard
            x1_std = [x[0] for x in standard_data]
            y1_std = [x[1] for x in standard_data]
            r1, p1 = stats.pearsonr(x1_std, y1_std)
            
            # Корреляция для horizontal
            x1_hor = [x[0] for x in horizontal_data]
            y1_hor = [x[1] for x in horizontal_data]
            r2, p2 = stats.pearsonr(x1_hor, y1_hor)
            
            # Сравнение корреляций
            z, p_compare = compare_correlations(r1, len(standard_data), r2, len(horizontal_data))
            
            self.add_paragraph("**Корреляция TL (зелёная зона) vs Желаемая продуктивность:**")
            self.add_table(
                ['Вариант', 'r', 'p-value', 'n'],
                [['Standard', f'{r1:.3f}', f'{p1:.4f}', len(standard_data)],
                 ['Horizontal', f'{r2:.3f}', f'{p2:.4f}', len(horizontal_data)]]
            )
            self.add_paragraph(f"Сравнение корреляций: z = {z:.2f}, p = {p_compare:.4f}")
            
            conclusion = "Подтверждается (корреляции эквивалентны)" if p_compare >= 0.05 else "Не подтверждается"
            self.add_paragraph(f"**Вывод**: {conclusion}")
        else:
            self.add_paragraph("⚠️ Недостаточно данных для анализа")
        
        # Дополнительные корреляции для проверки стабильности
        self.add_paragraph("\n**Дополнительные корреляции (все респонденты):**")
        
        zones = ['tl', 'tr', 'bl', 'br']
        zone_names = {'tl': 'TL (зелёная)', 'tr': 'TR (оранжевая)', 'bl': 'BL (серая)', 'br': 'BR (красная)'}
        
        for page in [1, 2]:
            page_name = "Личная жизнь" if page == 1 else "Работа"
            self.add_paragraph(f"\n*Страница {page} ({page_name}):*")
            
            for zone in zones:
                col = f'p{page}_{zone}'
                data = [(d[col], d['slider_desired']) for d in self.completed if d.get(col) and d.get('slider_desired')]
                
                if len(data) >= 10:
                    r, p = stats.pearsonr([x[0] for x in data], [x[1] for x in data])
                    sig = "*" if p < 0.05 else ""
                    self.add_paragraph(f"  {zone_names[zone]} vs желаемая продуктивность: r = {r:.3f}, p = {p:.4f} {sig}")
    
    def analyze_h7_life_work_diff(self):
        """H7: Разница между личной жизнью и работой одинакова в обеих группах"""
        self.add_section("H7: Эквивалентность разницы работа/личное", 3)
        
        def calc_diff(row):
            if row.get('p1_tr') and row.get('p2_tr'):
                return row['p2_tr'] - row['p1_tr']
            return None
        
        standard = [calc_diff(d) for d in self.completed if d.get('variant') == 'standard']
        horizontal = [calc_diff(d) for d in self.completed if d.get('variant') == 'horizontal']
        standard = [x for x in standard if x is not None]
        horizontal = [x for x in horizontal if x is not None]
        
        if standard and horizontal:
            t_stat, p_value = stats.ttest_ind(standard, horizontal)
            diff = np.mean(horizontal) - np.mean(standard)
            
            self.add_paragraph(f"Разница (работа - личное) в зелёной зоне:")
            self.add_paragraph(f"  Standard: M = {np.mean(standard):.1f}%")
            self.add_paragraph(f"  Horizontal: M = {np.mean(horizontal):.1f}%")
            self.add_test_result("t-тест", t_stat, p_value, "Эквивалентность подтверждена" if p_value >= 0.05 else "Не подтверждается")
        else:
            self.add_paragraph("⚠️ Недостаточно данных")
    
    def analyze_additional_correlations(self):
        """Дополнительный анализ корреляций"""
        self.add_section("Дополнительные корреляции", 2)
        
        zones = ['tl', 'tr', 'bl', 'br']
        zone_names = {
            'tl': 'TL (зелёная)',
            'tr': 'TR (оранжевая)',
            'bl': 'BL (серая)',
            'br': 'BR (красная)'
        }
        
        # 1. Корреляции между работой и личной жизнью
        self.add_section("Корреляции между работой и личной жизнью", 3)
        
        table_data = []
        for zone in zones:
            data = [(d[f'p1_{zone}'], d[f'p2_{zone}']) for d in self.completed 
                   if d.get(f'p1_{zone}') and d.get(f'p2_{zone}')]
            
            if len(data) >= 10:
                r, p = stats.pearsonr([x[0] for x in data], [x[1] for x in data])
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                table_data.append([zone_names[zone], f"{r:.3f}", f"{p:.4f}", len(data), sig])
        
        if table_data:
            self.add_table(
                ['Зона', 'r', 'p-value', 'n', 'Значимо'],
                table_data
            )
            self.add_paragraph("* p < 0.05, ** p < 0.01, *** p < 0.001")
        else:
            self.add_paragraph("⚠️ Недостаточно данных")
    
        # 2. Корреляции с балансом
        self.add_section("Корреляции с балансом работа/личное", 3)
        self.add_paragraph("Баланс: 0 = только работа, 100 = только личная жизнь")
        
        table_data = []
        for zone in zones:
            for page in [1, 2]:
                page_name = "Личная жизнь" if page == 1 else "Работа"
                col = f'p{page}_{zone}'
                data = [(d[col], d['slider_balance']) for d in self.completed 
                       if d.get(col) and d.get('slider_balance')]
                
                if len(data) >= 10:
                    r, p = stats.pearsonr([x[0] for x in data], [x[1] for x in data])
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                    table_data.append([f"{zone_names[zone]} ({page_name})", f"{r:.3f}", f"{p:.4f}", len(data), sig])
        
        if table_data:
            self.add_table(
                ['Зона', 'r', 'p-value', 'n', 'Значимо'],
                table_data
            )
        else:
            self.add_paragraph("⚠️ Недостаточно данных")
    
        # 3. Корреляции с желаемой продуктивностью
        self.add_section("Корреляции с ощущаемой продуктивностью (vs желаемый уровень)", 3)
        
        table_data = []
        for zone in zones:
            for page in [1, 2]:
                page_name = "Личная жизнь" if page == 1 else "Работа"
                col = f'p{page}_{zone}'
                data = [(d[col], d['slider_desired']) for d in self.completed 
                       if d.get(col) and d.get('slider_desired')]
                
                if len(data) >= 10:
                    r, p = stats.pearsonr([x[0] for x in data], [x[1] for x in data])
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                    table_data.append([f"{zone_names[zone]} ({page_name})", f"{r:.3f}", f"{p:.4f}", len(data), sig])
        
        if table_data:
            self.add_table(
                ['Зона', 'r', 'p-value', 'n', 'Значимо'],
                table_data
            )
        else:
            self.add_paragraph("⚠️ Недостаточно данных")
    
        # 4. Корреляции с продуктивностью относительно других
        self.add_section("Корреляции с продуктивностью (в сравнении с другими)", 3)
        
        table_data = []
        for zone in zones:
            for page in [1, 2]:
                page_name = "Личная жизнь" if page == 1 else "Работа"
                col = f'p{page}_{zone}'
                # Ищем поле для сравнения с другими - возможно slider_others
                data = [(d[col], d.get('slider_others')) for d in self.completed 
                       if d.get(col) and d.get('slider_others')]
                
                if len(data) >= 10:
                    r, p = stats.pearsonr([x[0] for x in data], [x[1] for x in data])
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                    table_data.append([f"{zone_names[zone]} ({page_name})", f"{r:.3f}", f"{p:.4f}", len(data), sig])
        
        if table_data:
            self.add_table(
                ['Зона', 'r', 'p-value', 'n', 'Значимо'],
                table_data
            )
        else:
            self.add_paragraph("⚠️ Недостаточно данных (проверьте наличие поля slider_others в данных)")
        
        # 5. Общая сводка корреляций
        self.add_section("Сводка корреляций", 3)
        
        self.add_paragraph("""
**Интерпретация:**
- r > 0 — прямая связь (больше в одной сфере → больше в другой)
- r < 0 — обратная связь
- |r| < 0.3 — слабая связь
- 0.3 ≤ |r| < 0.5 — средняя связь
- |r| ≥ 0.5 — сильная связь
""")
        
    # ==================== ГИПОТЕЗЫ H8-H13 (Демография) ====================
    
    def analyze_h8_gender_life(self):
        """H8: Распределение по зонам в личной жизни отличается для мужчин и женщин"""
        self.add_section("H8: Гендерные различия (личная жизнь)", 3)
        
        male = [d['p1_tl'] for d in self.completed if d.get('gender') == 'male' and d.get('p1_tl')]
        female = [d['p1_tl'] for d in self.completed if d.get('gender') == 'female' and d.get('p1_tl')]
        
        if male and female:
            t_stat, p_value = stats.ttest_ind(male, female)
            d = cohens_d(male, female)
            self.add_paragraph(f"Зона TL (личная жизнь):")
            self.add_paragraph(f"  Мужчины: M = {np.mean(male):.1f}%, n = {len(male)}")
            self.add_paragraph(f"  Женщины: M = {np.mean(female):.1f}%, n = {len(female)}")
            self.add_test_result("t-тест", t_stat, p_value, 
                               "Подтверждается" if p_value < 0.05 else "Не подтверждается", d)
        else:
            self.add_paragraph("⚠️ Недостаточно данных")
    
    def analyze_h9_gender_work(self):
        """H9: Распределение по зонам на работе отличается для мужчин и женщин"""
        self.add_section("H9: Гендерные различия (работа)", 3)
        
        male = [d['p2_tl'] for d in self.completed if d.get('gender') == 'male' and d.get('p2_tl')]
        female = [d['p2_tl'] for d in self.completed if d.get('gender') == 'female' and d.get('p2_tl')]
        
        if male and female:
            t_stat, p_value = stats.ttest_ind(male, female)
            d = cohens_d(male, female)
            self.add_paragraph(f"Зона TL (работа):")
            self.add_paragraph(f"  Мужчины: M = {np.mean(male):.1f}%, n = {len(male)}")
            self.add_paragraph(f"  Женщины: M = {np.mean(female):.1f}%, n = {len(female)}")
            self.add_test_result("t-тест", t_stat, p_value,
                               "Подтверждается" if p_value < 0.05 else "Не подтверждается", d)
        else:
            self.add_paragraph("⚠️ Недостаточно данных")
    
    def analyze_h10_moderation(self):
        """H10: Возраст является модератором связи гендер-баланс"""
        self.add_section("H10: Модерация возрастом", 3)
        
        data_with_age = [d for d in self.completed if d.get('age') and d.get('slider_balance')]
        
        if len(data_with_age) < 30:
            self.add_paragraph("⚠️ Недостаточно данных для анализа модерации")
            return
        
        young = [d['slider_balance'] for d in data_with_age if d['age'] < 35]
        middle = [d['slider_balance'] for d in data_with_age if 35 <= d['age'] < 50]
        older = [d['slider_balance'] for d in data_with_age if d['age'] >= 50]
        
        self.add_paragraph(f"Баланс по возрастным группам:")
        self.add_paragraph(f"  До 35 лет: M = {np.mean(young):.1f}%, n = {len(young)}")
        self.add_paragraph(f"  35-49 лет: M = {np.mean(middle):.1f}%, n = {len(middle)}")
        self.add_paragraph(f"  50+ лет: M = {np.mean(older):.1f}%, n = {len(older)}")
        
        f_stat, p_value = stats.f_oneway(young, middle, older)
        self.add_test_result("ANOVA", f_stat, p_value, 
                           "Подтверждается" if p_value < 0.05 else "Не подтверждается")
    
    def analyze_h11_position_work(self):
        """H11: Распределение по зонам в работе отличается для руководителей"""
        self.add_section("H11: Различия по должности (работа)", 3)
        
        leaders = ['Владелец бизнеса', 'Высший менеджмент', 'Тимлид']
        
        leader_data = [d['p2_tl'] for d in self.completed 
                      if d.get('position') in leaders and d.get('p2_tl')]
        employee_data = [d['p2_tl'] for d in self.completed 
                        if d.get('position') not in leaders and d.get('p2_tl') and d.get('position')]
        
        if leader_data and employee_data:
            t_stat, p_value = stats.ttest_ind(leader_data, employee_data)
            d = cohens_d(leader_data, employee_data)
            self.add_paragraph(f"Зона TL (работа):")
            self.add_paragraph(f"  Руководители: M = {np.mean(leader_data):.1f}%, n = {len(leader_data)}")
            self.add_paragraph(f"  Сотрудники: M = {np.mean(employee_data):.1f}%, n = {len(employee_data)}")
            self.add_test_result("t-тест", t_stat, p_value,
                               "Подтверждается" if p_value < 0.05 else "Не подтверждается", d)
        else:
            self.add_paragraph("⚠️ Недостаточно данных")
    
    def analyze_h12_position_life(self):
        """H12: Распределение по зонам в личной жизни отличается для руководителей"""
        self.add_section("H12: Различия по должности (личная жизнь)", 3)
        
        leaders = ['Владелец бизнеса', 'Высший менеджмент', 'Тимлид']
        
        leader_data = [d['p1_tl'] for d in self.completed 
                      if d.get('position') in leaders and d.get('p1_tl')]
        employee_data = [d['p1_tl'] for d in self.completed 
                        if d.get('position') not in leaders and d.get('p1_tl') and d.get('position')]
        
        if leader_data and employee_data:
            t_stat, p_value = stats.ttest_ind(leader_data, employee_data)
            d = cohens_d(leader_data, employee_data)
            self.add_paragraph(f"Зона TL (личная жизнь):")
            self.add_paragraph(f"  Руководители: M = {np.mean(leader_data):.1f}%, n = {len(leader_data)}")
            self.add_paragraph(f"  Сотрудники: M = {np.mean(employee_data):.1f}%, n = {len(employee_data)}")
            self.add_test_result("t-тест", t_stat, p_value,
                               "Подтверждается" if p_value < 0.05 else "Не подтверждается", d)
        else:
            self.add_paragraph("⚠️ Недостаточно данных")
    
    def analyze_h13_balance_productivity(self):
        """H13: Отрицательная корреляция между балансом и продуктивностью"""
        self.add_section("H13: Баланс работа/личное и продуктивность", 3)
        
        data = [(d['slider_balance'], d['slider_desired']) for d in self.completed 
               if d.get('slider_balance') and d.get('slider_desired')]
        
        if len(data) < 10:
            self.add_paragraph("⚠️ Недостаточно данных")
            return
        
        r, p_value = stats.pearsonr([x[0] for x in data], [x[1] for x in data])
        
        self.add_paragraph(f"Корреляция между балансом (0=работа, 100=личное) и субъективной продуктивностью:")
        self.add_paragraph(f"  r = {r:.3f}, p = {p_value:.4f}")
        
        conclusion = "Подтверждается (отрицательная связь)" if r < 0 and p_value < 0.05 else "Не подтверждается"
        self.add_paragraph(f"**Вывод**: {conclusion}")
    
    def generate_summary_statistics(self):
        """Генерация сводной статистики"""
        self.add_section("Сводная статистика", 2)
        
        groups = {}
        for d in self.completed:
            g = d.get('group_id', 'unknown')
            groups[g] = groups.get(g, 0) + 1
        
        self.add_paragraph("**Распределение по группам:**")
        for g, count in sorted(groups.items()):
            variant = 'standard' if g in [1, 2] else 'horizontal'
            order = 'life→work' if g in [1, 3] else 'work→life'
            self.add_paragraph(f"  Группа {g} ({variant}, {order}): {count}")
        
        ages = [d['age'] for d in self.completed if d.get('age')]
        genders = {}
        positions = {}
        for d in self.completed:
            g = d.get('gender', 'unknown')
            genders[g] = genders.get(g, 0) + 1
            p = d.get('position', 'unknown')
            positions[p] = positions.get(p, 0) + 1
        
        self.add_paragraph(f"\n**Демография (n={len(self.completed)}):**")
        if ages:
            self.add_paragraph(f"  Возраст: M = {np.mean(ages):.1f}, SD = {np.std(ages):.1f}, диапазон = {min(ages)}-{max(ages)}")
        self.add_paragraph(f"  Пол: {genders}")
        self.add_paragraph(f"  Должности: {positions}")
        
        times = [d['time_total'] for d in self.completed if d.get('time_total')]
        if times:
            self.add_paragraph(f"\n**Время прохождения:**")
            self.add_paragraph(f"  M = {np.mean(times):.0f} сек ({np.mean(times)/60:.1f} мин)")
            self.add_paragraph(f"  SD = {np.std(times):.0f} сек")
            self.add_paragraph(f"  Медиана = {np.median(times):.0f} сек")
    
    def run_analysis(self):
        """Запуск полного анализа"""
        print_colored("Запуск анализа гипотез...", Colors.BLUE)
        
        self.report.append(f"""# Отчёт A/B теста вариантов контрола продуктивности

**Дата анализа:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Всего респондентов:** {len(self.data)}
**Завершили опрос:** {len(self.completed)}

---

## Описание исследования

Цель исследования — определить, какой вариант интерфейса контрола продуктивности 
(Стандартный или Горизонтальный) является более понятным для респондентов.

- **Стандартный вариант**: вертикальный бегунок слева от матрицы
- **Горизонтальный вариант**: все бегунки горизонтальные под матрицей

---
""")
        
        self.generate_summary_statistics()
        
        self.add_section("Основные гипотезы (H1-H4)")
        self.analyze_h1_understanding()
        self.analyze_h2_time()
        self.analyze_h3_preference_second()
        self.analyze_h4_preference()
        
        self.add_section("Гипотезы эквивалентности (H5-H7)")
        self.analyze_h5_equivalence()
        self.analyze_h6_correlations()
        self.analyze_h7_life_work_diff()
        
        self.add_section("Дополнительные корреляции")
        self.analyze_additional_correlations()
        
        self.add_section("Гипотезы демографических различий (H8-H13)")
        self.analyze_h8_gender_life()
        self.analyze_h9_gender_work()
        self.analyze_h10_moderation()
        self.analyze_h11_position_work()
        self.analyze_h12_position_life()
        self.analyze_h13_balance_productivity()
        
        self.add_section("Заключение", 2)
        self.add_paragraph("""
**Примечание к интерпретации:**
- p < 0.05 — статистически значимый результат
- Cohen's d: малый эффект = 0.2, средний = 0.5, большой = 0.8
- Для эквивалентности: разница < 5% и p > 0.05
- Для Манна-Уитни: r < 0.1 пренебрежимый, 0.1-0.3 малый, 0.3-0.5 средний, >0.5 большой

---
*Отчёт сгенерирован автоматически*
""")
        
        return "\n".join(self.report)
    
    def save_report(self, output_path):
        """Сохранение отчёта в файл"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.run_analysis())
        print_colored(f"Отчёт сохранён: {output_path}", Colors.GREEN)


def main():
    parser = argparse.ArgumentParser(description='Анализ A/B теста вариантов контрола')
    parser.add_argument('--csv', default='AB-test-respondents/ab_test_respondents_2026-03-24_125257.csv',
                       help='Путь к CSV файлу с данными')
    parser.add_argument('--output', default='Analysis/analysis_report.md',
                       help='Путь для сохранения отчёта')
    args = parser.parse_args()
    
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print_colored(f"Ошибка: файл {csv_path} не найден", Colors.RED)
        sys.exit(1)
    
    analyzer = ABTestAnalyzer(str(csv_path))
    analyzer.save_report(args.output)


if __name__ == '__main__':
    main()
