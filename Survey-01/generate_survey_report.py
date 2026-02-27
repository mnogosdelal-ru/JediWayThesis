import pandas as pd
import numpy as np
import pingouin as pg
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
import os
import sklearn.utils.validation
import factor_analyzer.factor_analyzer
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, f_oneway, mannwhitneyu, pearsonr
from statsmodels.stats.multitest import multipletests

# --- Настройки ---
INPUT_FILE = 'Большое исследование джедайских приемов (Responses).csv'
REPORT_DIR = 'C:\\Users\\maxim\\OneDrive\\Obsidian\\MyBrain\\Мои исследования\\Джедайская шкала\\'
#REPORT_DIR = ''
OUTPUT_FILE = REPORT_DIR + 'survey_report.md'
IMAGES_DIR = REPORT_DIR + 'images'

# Целевая шкала для корреляционного анализа (можно менять)
# TARGET_SCALE = 'MIJS-2+'
#TARGET_SCALE = 'MIJS-3+'
TARGET_SCALE = 'single_item'
# TARGET_SCALE = 'MIJS-2'
#TARGET_SCALE = 'MIJS'

if not os.path.exists(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)

# Исправление совместимости factor_analyzer с новыми версиями scikit-learn
_original_check_array = sklearn.utils.validation.check_array
def _patched_check_array(*args, **kwargs):
    if 'force_all_finite' in kwargs:
        kwargs['ensure_all_finite'] = kwargs.pop('force_all_finite')
    return _original_check_array(*args, **kwargs)

# Патчим и в sklearn, и внутри factor_analyzer
sklearn.utils.validation.check_array = _patched_check_array
factor_analyzer.factor_analyzer.check_array = _patched_check_array

class HeaderManager:
    """Утилита для управления многоуровневой нумерацией заголовков."""
    def __init__(self):
        self.levels = [0] * 5

    def get_header(self, level, text):
        self.levels[level - 1] += 1
        for i in range(level, 5):
            self.levels[i] = 0
        
        prefix = ".".join(map(str, self.levels[:level]))
        return f"{'#' * level} {prefix}. {text}"

def calculate_omega_from_data(data_subset):
    """Вычисляет омегу МакДональда на основе 1-факторной модели EFA."""
    if data_subset.empty or data_subset.shape[1] < 2:
        return np.nan
    
    fa = FactorAnalyzer(n_factors=1, rotation=None)
    fa.fit(data_subset)
    loadings = fa.loadings_.flatten()
    
    # Сумма нагрузок в квадрате / (Сумма нагрузок в квадрате + Сумма уникальностей)
    # Уникальность = 1 - нагрузка^2
    sum_loadings = np.sum(loadings)
    sum_uniqueness = np.sum(1 - loadings**2)
    
    omega = (sum_loadings**2) / (sum_loadings**2 + sum_uniqueness)
    return omega

def save_plot(filename):
    path = os.path.join(IMAGES_DIR, filename)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path

def generate_report():
    print(f"Загрузка данных из {INPUT_FILE}...")
    hm = HeaderManager()
    
    # Загружаем данные
    df = pd.read_csv(INPUT_FILE)

    # Колонки для анализа шкал + Демография
    # 1: Возраст, 2: Пол, 4: Джедайская (текст), 55: Хронотип, 59: Удаленка
    COL_MAPPING = {
        1: 'age',
        2: 'gender',
        4: 'jedi_single_raw',
        5: 'mijs_q1',
        6: 'mijs_q2',
        7: 'mijs_q3',
        8: 'mijs_q4',
        9: 'mijs_q5',
        10: 'mijs_q6',
        
        # Группа 1: Частота использования приемов (11-41)
        11: 'prac_screen_bed', 12: 'prac_two_min_nothing', 13: 'prac_unload_memory',
        14: 'prac_separate_tasks', 15: 'prac_only_listed', 16: 'prac_monkey_style',
        17: 'prac_later_list', 18: 'prac_silence_blocks', 19: 'prac_morning_no_phone',
        20: 'prac_task_before_email', 21: 'prac_daily_plan', 22: 'prac_workout',
        23: 'prac_daily_results', 24: 'prac_hide_phone', 25: 'prac_weekly_plan',
        26: 'prac_close_tabs', 27: 'prac_adjust_weekly_plan', 28: 'prac_internet_shabbat',
        29: 'prac_weekly_results', 30: 'prac_please_monkey', 31: 'prac_inbox_zero',
        32: 'prac_task_before_world', 33: 'prac_align_goals', 34: 'prac_green_task',
        35: 'prac_green_before_email', 36: 'prac_toilet_phone', 37: 'prac_idea_incubator',
        38: 'prac_clean_list', 39: 'prac_15min_thoughts', 40: 'prac_daily_log',
        41: 'prac_shy_smartphone',

        # Группа 2: Уровень внедрения (42-52)
        42: 'setup_blue_filter', 43: 'setup_sleep_audit', 44: 'setup_dnd_mode',
        45: 'setup_remove_social', 46: 'setup_no_red_dots', 47: 'setup_adblock',
        48: 'setup_chat_notif_off', 49: 'setup_email_notif_off', 50: 'setup_app_push_off',
        51: 'setup_watch_notif_off', 52: 'setup_auto_read_email',

        53: 'work_with_tasks', 54: 'task_tools',
        55: 'chronotype',
        56: 'luckiness',
        59: 'remote_work'
    }
    
    data = df.iloc[:, list(COL_MAPPING.keys())].copy()
    data.columns = [COL_MAPPING[i] for i in COL_MAPPING.keys()]

    # Функция для маппинга по ключевым словам (более надежно)
    def map_jedi(val):
        if not isinstance(val, str): return val
        val = val.lower()
        if "практически" in val: return 1
        if "стабильно" in val: return 2
        if "поровну" in val: return 3
        if "не много" in val: return 4
        if "избытком" in val: return 5
        return np.nan

    def map_frequency(val):
        if not isinstance(val, str): return val
        val = val.lower()
        if "никогда" in val or "крайне редко" in val: return 0
        if "редко" in val and "месяц" in val: return 1
        if "иногда" in val and "неделю" in val: return 2
        if "часто" in val and "неделю" in val: return 3
        if "практически всегда" in val: return 4
        if "редко" in val and "месяца" in val: return 1
        if "иногда" in val and "месяц" in val: return 2
        if "часто" in val and "месяц" in val: return 3
        if "постоянно" in val and "неделю" in val: return 4
        if "постоянно" in val and "день" in val: return 4
        return np.nan

    def map_implementation(val):
        if not isinstance(val, str): return val
        val = val.lower()
        if "не применил" in val: return 0
        if "но не полностью" in val: return 1
        if "небольшими исключениями" in val: return 2
        if "по максимуму" in val: return 3
        if "у меня этого нет" in val: return 3
        return np.nan

    def map_remote(val):
        if not isinstance(val, str): return val
        val = val.lower()
        if "в офисе" in val and "удаленная работа - это исключение" in val: return 0
        if "1 день" in val: return 1
        if "2 дня" in val: return 2
        if "3 день" in val: return 3
        if "4 день" in val: return 4
        if "удаленно" in val and "выезд в офис - это исключение" in val: return 5
        return np.nan

    # Применяем маппинг
    data['jedi_single'] = data['jedi_single_raw'].apply(map_jedi)
    data['jedi_inv'] = 6 - data['jedi_single']

    # Маппинг практик
    prac_cols = [c for c in data.columns if c.startswith('prac_')]
    for col in prac_cols:
        data[col] = data[col].apply(map_frequency)

    setup_cols = [c for c in data.columns if c.startswith('setup_')]
    for col in setup_cols:
        data[col] = data[col].apply(map_implementation)

    data['remote_work_num'] = data['remote_work'].apply(map_remote)

    # Для всех числовых колонок убеждаемся, что они числовые
    numeric_cols = ['age', 'mijs_q1', 'mijs_q2', 'mijs_q3', 'mijs_q4', 'mijs_q5', 'mijs_q6', 'jedi_single', 'luckiness', 'remote_work_num'] + prac_cols + setup_cols
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Сохраняем полную версию для демографии (до dropna по шкалам)
    demo_data = data.copy()
    
    # Очистка для анализа шкал (только те, кто ответил на все пункты MIJS и Джедайскую)
    scale_items_all = ['mijs_q1', 'mijs_q2', 'mijs_q3', 'mijs_q4', 'mijs_q5', 'mijs_q6', 'jedi_single']
    data_scales = data.dropna(subset=scale_items_all).copy()
    
    final_count = len(data_scales)
    print(f"Очистка: было {len(df)}, для анализа шкал пригодно {final_count} анкет.")

    report_content = "# Отчет по исследованию джедайских приемов\n\n"
    
    # --- Раздел 1: Информация о выборке ---
    report_content += hm.get_header(1, "Информация о выборке") + "\n\n"
    report_content += f"Всего в опросе приняло участие **{len(df)}** человек. После очистки данных для анализа психометрики шкал использовано **{final_count}** анкет.\n\n"
    
    # Настройка визуализаций
    sns.set_theme(style="whitegrid", palette="muted")
    
    # 1.1 Демографические характеристики
    report_content += hm.get_header(2, "Демографические характеристики") + "\n\n"
    
    plt.figure(figsize=(10, 5))
    sns.histplot(demo_data['age'].dropna(), bins=10, kde=True, color='teal')
    plt.title("Распределение респондентов по возрасту")
    plt.xlabel("Возраст")
    plt.ylabel("Количество")
    save_plot("demo_age.png")
    report_content += hm.get_header(3, "Распределение по возрасту") + "\n"
    report_content += f"![Распределение по возрасту](images/demo_age.png)\n\n"
    
    # ... (other demo plots)
    plt.figure(figsize=(7, 7))
    gender_counts = demo_data['gender'].value_counts()
    plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
    plt.title("Распределение по полу")
    save_plot("demo_gender.png")
    report_content += hm.get_header(3, "Распределение по полу") + "\n"
    report_content += f"![Распределение по полу](images/demo_gender.png)\n\n"

    plt.figure(figsize=(10, 6))
    jedi_counts = demo_data['jedi_single'].value_counts().sort_index()
    sns.barplot(x=jedi_counts.index, y=jedi_counts.values, hue=jedi_counts.index, palette="viridis", legend=False)
    plt.title("Распределение по однопунктовой 'Джедайской шкале'")
    plt.xlabel("Балл (1 - плохо, 5 - отлично)")
    plt.ylabel("Количество")
    save_plot("demo_jedi.png")
    report_content += hm.get_header(3, "Однопунктовая 'Джедайская шкала'") + "\n"
    report_content += f"![Джедайская шкала](images/demo_jedi.png)\n\n"

    plt.figure(figsize=(10, 6))
    chrono_counts = demo_data['chronotype'].value_counts()
    sns.barplot(x=chrono_counts.values, y=chrono_counts.index, hue=chrono_counts.index, palette="magma", legend=False)
    plt.title("Распределение по хронотипу")
    save_plot("demo_chronotype.png")
    report_content += hm.get_header(3, "Хронотип") + "\n"
    report_content += f"![Хронотип](images/demo_chronotype.png)\n\n"

    plt.figure(figsize=(10, 6))
    remote_counts = demo_data['remote_work'].value_counts()
    sns.barplot(x=remote_counts.values, y=remote_counts.index, hue=remote_counts.index, palette="coolwarm", legend=False)
    plt.title("Режим работы (офис / удаленка)")
    save_plot("demo_remote.png")
    report_content += hm.get_header(3, "Режим работы") + "\n"
    report_content += f"![Удаленка](images/demo_remote.png)\n\n"

    # --- Раздел 2: Анализ используемых шкал ---
    report_content += hm.get_header(1, "Анализ используемых шкал") + "\n\n"

    QUESTION_TEXTS = {
        'jedi_single': "Джедайская шкала (распределение времени и сил)",
        'mijs_q1': "Каждый день появляются незапланированные срочные задачи, которые я не могу отложить",
        'mijs_q2': "Я не могу отвести время на важное, потому что оно полностью занято срочным",
        'mijs_q3': "Я в большей степени тушу кризисы, чем их предотвращаю",
        'mijs_q4': "Мне трудно видеть прогресс в важных проектах, потому что до них редко доходит очередь",
        'mijs_q5': "К концу дня у меня почти нет энергии на важное",
        'mijs_q6': "Я чувствую, что мои ресурсы (время и энергия) исчерпываются на срочное"
    }

    scales = {
        "MIJS": ['mijs_q1', 'mijs_q2', 'mijs_q3', 'mijs_q4', 'mijs_q5', 'mijs_q6'],
        "MIJS-2": ['mijs_q2', 'mijs_q3', 'mijs_q4', 'mijs_q6'],
        "MIJS-2+": ['mijs_q2', 'mijs_q3', 'mijs_q4', 'mijs_q6', 'jedi_inv'],
        "MIJS-3+": ['mijs_q2', 'mijs_q4', 'mijs_q6', 'jedi_inv'],
        "single_item": ['jedi_inv']
    }

    summary_rows = []

    for scale_name, items in scales.items():
        report_content += hm.get_header(2, f"Шкала: {scale_name}") + "\n\n"
        
        # Расчет суммарного балла (сумма)
        data_scales[scale_name + '_total'] = data_scales[items].sum(axis=1)
        
        # График распределения баллов по шкале
        plt.figure(figsize=(8, 5))
        sns.histplot(data_scales[scale_name + '_total'], bins="auto", kde=True, color='indigo')
        plt.title(f"Распределение баллов по шкале {scale_name}")
        plt.xlabel("Суммарный балл")
        plt.ylabel("Количество")
        save_plot(f"dist_{scale_name}.png")
        report_content += f"![Распределение {scale_name}](images/dist_{scale_name}.png)\n\n"

        report_content += "Состав пунктов:\n"
        for it in items:
            txt = QUESTION_TEXTS.get(it, "Инвертированная Джедайская шкала")
            report_content += f"- {it}: {txt}\n"
        report_content += "\n"

        if len(items) == 1:
            continue

        # 2.x.1 Надежность
        report_content += hm.get_header(3, "Надежность (Reliability)") + "\n\n"
        alpha = pg.cronbach_alpha(data_scales[items])[0]
        omega = calculate_omega_from_data(data_scales[items])
        report_content += f"- **Альфа Кронбаха:** {alpha:.3f}\n"
        report_content += f"- **Омега МакДональда:** {omega:.3f}\n\n"

        report_content += "#### Коэффициенты при исключении пункта:\n\n"
        report_content += "| Исключенный пункт | Альфа (if deleted) | Омега (if deleted) | Корреляция пункт-итог |\n"
        report_content += "| :--- | :---: | :---: | :---: |\n"

        for item in items:
            rem = [x for x in items if x != item]
            c_alpha = pg.cronbach_alpha(data_scales[rem])[0] if len(rem) > 1 else np.nan
            c_omega = calculate_omega_from_data(data_scales[rem]) if len(rem) > 1 else np.nan
            item_total_corr = data_scales[item].corr(data_scales[rem].sum(axis=1))
            report_content += f"| {item} | {c_alpha:.3f} | {c_omega:.3f} | {item_total_corr:.3f} |\n"
        report_content += "\n"

        # 2.x.2 EFA
        report_content += hm.get_header(3, "Эксплораторный факторный анализ (EFA)") + "\n\n"
        kmo_model = calculate_kmo(data_scales[items])[1]
        chi_square, p_value = calculate_bartlett_sphericity(data_scales[items])
        report_content += f"- **Критерий KMO:** {kmo_model:.3f}\n"
        report_content += f"- **Тест Бартлетта:** chi2={chi_square:.2f}, p={p_value:.4f}\n\n"

        # EFA для 1 фактора
        fa = FactorAnalyzer(n_factors=1, rotation=None)
        fa.fit(data_scales[items])
        
        # Собственные значения (все)
        ev, _ = fa.get_eigenvalues()
        report_content += "#### Собственные значения факторов (Eigenvalues):\n"
        report_content += ", ".join([f"{v:.3f}" for v in ev]) + "\n\n"

        var_info = fa.get_factor_variance()
        report_content += "#### Факторные нагрузки (1 фактор):\n\n"
        report_content += "| Пункт | Нагрузка |\n"
        report_content += "| :--- | :---: |\n"
        for i, item in enumerate(items):
            report_content += f"| {item} | {fa.loadings_[i][0]:.3f} |\n"
        report_content += "\n"
        
        report_content += f"- **Объясненная дисперсия (1 фактор):** {var_info[1][0]*100:.2f}%\n"
        report_content += f"- **Собственное число основного фактора:** {var_info[0][0]:.3f}\n\n"

        # 2.x.3 Резюме
        report_content += hm.get_header(3, "Резюме по применимости") + "\n\n"
        verdict = "Применима"
        if alpha < 0.7: verdict = "Низкая надежность"
        if kmo_model < 0.6: verdict = "Структура не выражена"

        report_content += "| Параметр | Значение | Комментарий |\n"
        report_content += "| :--- | :---: | :--- |\n"
        report_content += f"| Alpha | {alpha:.3f} | {'ОК' if alpha >= 0.7 else 'Слабо'} |\n"
        report_content += f"| KMO | {kmo_model:.3f} | {'ОК' if kmo_model >= 0.6 else 'Плохо'} |\n"
        report_content += f"| Variance | {var_info[1][0]*100:.1f}% | {'Хорошо' if var_info[1][0] > 0.4 else 'Средне'} |\n"
        report_content += f"| **ИТОГ** | **{verdict}** | |\n\n"
        report_content += "---\n\n"

        # Собираем данные для итоговой таблицы
        summary_rows.append({
            'Scale': scale_name,
            'Alpha': alpha,
            'Omega': omega,
            'Variance': var_info[1][0] * 100,
            'KMO': kmo_model,
            'Bartlett_Chi2': chi_square,
            'Bartlett_p': p_value
        })

    # --- Раздел 2.5: Итого по шкалам ---
    report_content += hm.get_header(2, "Итого по шкалам") + "\n\n"
    report_content += "| Шкала | Alpha | Omega | % Variance | KMO | Bartlett χ² | Bartlett p |\n"
    report_content += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    for row in summary_rows:
        if row['Scale'] == TARGET_SCALE:
            report_content += f"| **{row['Scale']}** | **{row['Alpha']:.3f}** | **{row['Omega']:.3f}** | **{row['Variance']:.1f}%** | **{row['KMO']:.3f}** | **{row['Bartlett_Chi2']:.2f}** | **{row['Bartlett_p']:.4f}** |\n"
        else:
            report_content += f"| {row['Scale']} | {row['Alpha']:.3f} | {row['Omega']:.3f} | {row['Variance']:.1f}% | {row['KMO']:.3f} | {row['Bartlett_Chi2']:.2f} | {row['Bartlett_p']:.4f} |\n"
    report_content += "\n"
    report_content += f"==Дальнейший анализ основан на шкале  **{TARGET_SCALE}**==."
    report_content += "\n"

    # --- Раздел 3: Выбираем лучшие приемы и практики ---
    report_content += hm.get_header(1, "Выбираем лучшие приемы и практики") + "\n\n"
    # --- Раздел 3.1: Корреляционный анализ ---
    report_content += hm.get_header(2, "Корреляционный анализ (ранговая корреляция)") + "\n\n"
    report_content += f"В данном разделе представлена связь различных практик с целевой шкалой **{TARGET_SCALE}**. "
    report_content += "Для анализа используется коэффициент ранговой корреляции Спирмена (ρ).\n\n"

    target_data = data_scales[TARGET_SCALE + '_total']
    
    # Список признаков для анализа (исключая саму целевую шкалу, пункты MIJS и общую джедайской шкалу)
    features_to_analyze = prac_cols + setup_cols + ['luckiness', 'remote_work_num']
    
    # Названия признаков дословно из CSV
    # Собираем обратный маппинг: программное_имя -> заголовок_в_csv
    FEATURE_LABELS = {}
    for i, prog_name in COL_MAPPING.items():
        FEATURE_LABELS[prog_name] = df.columns[i]
    
    # Дополнительные настройки для специальных полей
    FEATURE_LABELS['remote_work_num'] = df.columns[59] # В среднем, сколько дней в неделю...
    FEATURE_LABELS['luckiness'] = df.columns[56] # На сколько удачливым человеком...

    corr_results = []
    for feat in features_to_analyze:
        # Убираем NaN для корректного расчета пары
        valid_idx = data_scales[feat].notna() & target_data.notna()
        if valid_idx.sum() > 30: # Минимум 30 наблюдений
            rho, pval = spearmanr(data_scales.loc[valid_idx, feat], target_data[valid_idx])
            
            corr_results.append({
                'Label': FEATURE_LABELS.get(feat, feat),
                'Rho': rho,
                'PVal': pval
            })

    # Коррекция p-значений (FDR) для корреляций
    if corr_results:
        p_vals = [r['PVal'] for r in corr_results]
        reject, p_adj, _, _ = multipletests(p_vals, method='fdr_bh')
        for i, r in enumerate(corr_results):
            r['PVal_adj'] = p_adj[i]
            
            # Звездочки на основе скорректированного p-value
            sig = ""
            if p_adj[i] < 0.0001: sig = "****"
            elif p_adj[i] < 0.001: sig = "***"
            elif p_adj[i] < 0.01: sig = "**"
            elif p_adj[i] < 0.05: sig = "*"
            r['Sig'] = sig

    # Сортировка по скорректированному p-value
    corr_results.sort(key=lambda x: (x.get('PVal_adj', 1.0), -abs(x['Rho'])))

    report_content += "| № | Прием | ρ (Rho) | p-value | p (adj) | Знач. |\n"
    report_content += "| :--- | :--- | :---: | :---: | :---: | :---: |\n"
    for i, res in enumerate(corr_results, 1):
        report_content += f"| {i} | {res['Label']} | {res['Rho']:.3f} | {res['PVal']:.4f} | {res['PVal_adj']:.4f} | {res['Sig']} |\n"
    report_content += f"\n*Примечание: Значимость (*) указана на основе скорректированного p-value (FDR). p < 0.05, ** p < 0.01, *** p < 0.001. Отрицательная корреляция означает, что использование приема связано с более низким баллом по шкале {TARGET_SCALE} (меньше стресса/завала).* \n\n"

    # --- Раздел 3.2: Сравнение средних (ANOVA) ---
    report_content += hm.get_header(2, "Сравнение средних (ANOVA)") + "\n\n"
    report_content += "В данном разделе анализируется, как частота использования или уровень внедрения приема связаны со средним значением продуктивности. "
    report_content += "Для проверки значимости различий между группами используется однофакторный дисперсионный анализ (One-way ANOVA).\n\n"

    # --- 3.2.1. Частота использования приемов ---
    report_content += hm.get_header(3, "Частота использования приемов") + "\n\n"
    report_content += "Респонденты разделены на группы по частоте использования (0 — никогда, 4 — всегда).\n\n"

    anova_results_prac = []
    for feat in prac_cols:
        groups = []
        means = {}
        for level in range(5):
            group_data = data_scales[data_scales[feat] == level][TARGET_SCALE + '_total'].dropna()
            if len(group_data) > 10:
                groups.append(group_data)
                means[level] = group_data.mean()
            else:
                means[level] = np.nan
        
        if len(groups) >= 2:
            f_stat, pval = f_oneway(*groups)
            anova_results_prac.append({
                'Label': FEATURE_LABELS.get(feat, feat),
                'Means': [means.get(l, np.nan) for l in range(5)],
                'F': f_stat, 'PVal': pval
            })

    # Коррекция p-значений (FDR) для ANOVA (частота)
    if anova_results_prac:
        p_vals = [r['PVal'] for r in anova_results_prac]
        reject, p_adj, _, _ = multipletests(p_vals, method='fdr_bh')
        for i, r in enumerate(anova_results_prac):
            r['PVal_adj'] = p_adj[i]
            sig = "****" if p_adj[i] < 0.0001 else "***" if p_adj[i] < 0.001 else "**" if p_adj[i] < 0.01 else "*" if p_adj[i] < 0.05 else ""
            r['Sig'] = sig

    anova_results_prac.sort(key=lambda x: x['PVal_adj'])
    report_content += "| № | Прием | Ср. (0) | Ср. (1) | Ср. (2) | Ср. (3) | Ср. (4) | F | p-value | p (adj) | Знач. |\n"
    report_content += "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    for i, res in enumerate(anova_results_prac, 1):
        m_str = " | ".join([f"{m:.1f}" if not np.isnan(m) else "-" for m in res['Means']])
        report_content += f"| {i} | {res['Label']} | {m_str} | {res['F']:.2f} | {res['PVal']:.4f} | {res['PVal_adj']:.4f} | {res['Sig']} |\n"
    report_content += f"\n*Примечание: Значимость (*) указана на основе скорректированного p-value (FDR). 0 — никогда/редко, 4 — всегда. В ячейках указан средний балл по шкале {TARGET_SCALE}. Чем ниже балл, тем выше продуктивность.* \n\n"

    # --- 3.2.2. Уровень внедрения приемов ---
    report_content += hm.get_header(3, "Уровень внедрения приемов") + "\n\n"
    report_content += "Респонденты разделены на группы по уровню внедрения (0 — не применил(а), 3 — применил(а) по максимуму).\n\n"

    anova_results_setup = []
    for feat in setup_cols:
        groups = []
        means = {}
        for level in range(4): # 0, 1, 2, 3
            group_data = data_scales[data_scales[feat] == level][TARGET_SCALE + '_total'].dropna()
            if len(group_data) > 10:
                groups.append(group_data)
                means[level] = group_data.mean()
            else:
                means[level] = np.nan
        
        if len(groups) >= 2:
            f_stat, pval = f_oneway(*groups)
            anova_results_setup.append({
                'Label': FEATURE_LABELS.get(feat, feat),
                'Means': [means.get(l, np.nan) for l in range(4)],
                'F': f_stat, 'PVal': pval
            })

    # Коррекция p-значений (FDR) для ANOVA (внедрение)
    if anova_results_setup:
        p_vals = [r['PVal'] for r in anova_results_setup]
        reject, p_adj, _, _ = multipletests(p_vals, method='fdr_bh')
        for i, r in enumerate(anova_results_setup):
            r['PVal_adj'] = p_adj[i]
            sig = "***" if p_adj[i] < 0.001 else "**" if p_adj[i] < 0.01 else "*" if p_adj[i] < 0.05 else ""
            r['Sig'] = sig

    anova_results_setup.sort(key=lambda x: x['PVal_adj'])
    report_content += "| № | Прием | Ср. (0) | Ср. (1) | Ср. (2) | Ср. (3) | F | p-value | p (adj) | Знач. |\n"
    report_content += "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    for i, res in enumerate(anova_results_setup, 1):
        m_str = " | ".join([f"{m:.1f}" if not np.isnan(m) else "-" for m in res['Means']])
        report_content += f"| {i} | {res['Label']} | {m_str} | {res['F']:.2f} | {res['PVal']:.4f} | {res['PVal_adj']:.4f} | {res['Sig']} |\n"
    report_content += f"\n*Примечание: Значимость (*) указана на основе скорректированного p-value (FDR). 0 — не применил(а), 3 — по максимуму. В ячейках указан средний балл по шкале {TARGET_SCALE}.*\n\n"


    # --- Раздел 3.3: Сравнение трёх групп продуктивности (Kruskal-Wallis + Dunn) ---
    report_content += hm.get_header(2, "Сравнение трёх групп продуктивности (Kruskal-Wallis)") + "\n\n"
    report_content += "Респонденты разделены на три группы по терцилям (20-й и 80-й перцентили) суммарного балла продуктивности. "
    report_content += f"Низкая группа – наиболее продуктивные (низкие баллы {TARGET_SCALE}), высокая группа – наименее продуктивные (высокие баллы {TARGET_SCALE}). "
    report_content += "Для каждой практики проверяется гипотеза о различиях распределений частоты использования между тремя группами с помощью критерия Краскела-Уоллиса. "
    report_content += "В случае значимых различий (p < 0.05 после FDR‑коррекции) выполняются попарные пост‑хок тесты Данна.\n\n"

    target_col = TARGET_SCALE + '_total'
    target = data_scales[target_col].dropna()

    # Определяем границы групп (терцили)
    p20 = target.quantile(0.15)
    p80 = target.quantile(0.75)

    # Создаём переменную группы
    data_scales['prod_group'] = pd.cut(data_scales[target_col], 
                                        bins=[-np.inf, p20, p80, np.inf], 
                                        labels=['Продуктивные', 'Средняки', 'Непродуктивные'])

    # Подсчёт числа респондентов в группах
    group_counts = data_scales['prod_group'].value_counts().sort_index()
    for label, count in group_counts.items():
        report_content += f"- **Группа {label}:** {count} респондентов\n"
    report_content += "\n"

    # Список практик
    all_practices = prac_cols + setup_cols

    # Собираем результаты
    kw_results = []
    for feat in all_practices:
        # Формируем списки значений по группам, удаляя пропуски
        low_vals = data_scales.loc[data_scales['prod_group'] == 'Продуктивные', feat].dropna()
        mid_vals = data_scales.loc[data_scales['prod_group'] == 'Средняки', feat].dropna()
        high_vals = data_scales.loc[data_scales['prod_group'] == 'Непродуктивные', feat].dropna()
        
        # Проверяем, что в каждой группе есть данные
        if len(low_vals) < 2 or len(mid_vals) < 2 or len(high_vals) < 2:
            continue
        
        # Вычисляем средние значения
        mean_low = low_vals.mean()
        mean_mid = mid_vals.mean()
        mean_high = high_vals.mean()
        
        from scipy.stats import kruskal
        h_stat, p_val = kruskal(low_vals, mid_vals, high_vals)
        kw_results.append({
            'label': FEATURE_LABELS.get(feat, feat),
            'feat': feat,
            'mean_low': mean_low,
            'mean_mid': mean_mid,
            'mean_high': mean_high,
            'h': h_stat,
            'p_val': p_val
        })

    # Коррекция p-значений (FDR)
    if kw_results:
        p_vals = [r['p_val'] for r in kw_results]
        reject, p_adj, _, _ = multipletests(p_vals, method='fdr_bh')
        for i, r in enumerate(kw_results):
            r['p_adj'] = p_adj[i]
            r['reject'] = reject[i]
        kw_results.sort(key=lambda x: x['p_adj'])

        # Таблица результатов Краскела-Уоллиса со средними
        report_content += "| № | Практика | Ср. Продуктивные | Ср. Средняки | Ср. Непродуктивные | H-статистика | p (исх.) | p (скорр.) | Значима |\n"
        report_content += "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
        for i, r in enumerate(kw_results, 1):
            sig_marker = "✅" if r['reject'] else ""
            report_content += f"| {i} | {r['label']} | {r['mean_low']:.2f} | {r['mean_mid']:.2f} | {r['mean_high']:.2f} | {r['h']:.2f} | {r['p_val']:.4f} | {r['p_adj']:.4f} | {sig_marker} |\n"

        # Выделим значимые
        sig_kw = [r for r in kw_results if r['reject']]
        report_content += f"\n**Значимых различий после коррекции (p<sub>adj</sub> < 0.05): {len(sig_kw)}**\n\n"

        # Для значимых практик проводим post-hoc тест Данна и строим ящичные диаграммы
        if sig_kw:
            # Ограничим число практик для детального анализа (например, первые 15)
            top_sig = sig_kw[:15]
            for r in top_sig:
                feat = r['feat']
                report_content += f"#### {r['label']}\n\n"
                # Собираем данные по группам
                groups = ['Продуктивные', 'Средняки', 'Непродуктивные']
                data_groups = [data_scales.loc[data_scales['prod_group'] == g, feat].dropna() for g in groups]
                
                # Post-hoc тест Данна (пакет scikit-posthocs)
                try:
                    import scikit_posthocs as sp
                    # Собираем данные для теста, избегая дублирования индексов
                    all_values = pd.concat(data_groups, ignore_index=True)
                    all_groups = pd.concat([pd.Series([g]*len(dg)) for g, dg in zip(groups, data_groups)], ignore_index=True)
                    dunn_df = pd.DataFrame({'values': all_values, 'group': all_groups})
                    
                    dunn_results = sp.posthoc_dunn(dunn_df, val_col='values', group_col='group', p_adjust='fdr_bh')
                    
                    # Переупорядочиваем для наглядности
                    order = ['Непродуктивные', 'Средняки', 'Продуктивные']
                    if all(name in dunn_results.index for name in order):
                        dunn_results = dunn_results.loc[order, order]
                    
                    report_content += "**Попарные сравнения (p-скорр.):**\n\n"
                    # Используем to_markdown для красивого вывода (требуется tabulate)
                    try:
                        from tabulate import tabulate
                        report_content += dunn_results.round(4).to_markdown(floatfmt=".4f") + "\n\n"
                    except ImportError:
                        # Fallback на HTML
                        report_content += dunn_results.round(4).to_html(classes='table table-striped') + "\n\n"
                except ImportError:
                    report_content += "*Для post-hoc анализа требуется установить `scikit-posthocs`.*\n\n"                

                # Ящичная диаграмма
                plt.figure(figsize=(8, 5))
                data_to_plot = [data_scales.loc[data_scales['prod_group'] == g, feat].dropna() for g in groups]
                bp = plt.boxplot(data_to_plot, tick_labels=groups, patch_artist=True,
                                boxprops=dict(facecolor='lightblue'),
                                medianprops=dict(color='red'))
                plt.ylabel('Частота использования')
                plt.title(f'Распределение частоты: {r["label"]}')
                plt.xticks(rotation=15)

                # Добавляем средние значения в виде зелёных треугольников
                for j, group_data in enumerate(data_to_plot, start=1):
                    mean_val = group_data.mean()
                    # Рисуем большую точку
                    plt.plot(j, mean_val, 'o', markersize=16, color='green', label='Среднее' if j == 1 else "")
                    # Добавляем текстовую подпись справа от точки с небольшим смещением
                    plt.text(j-0.1, mean_val + 0.2, f'{mean_val:.2f}', 
                             verticalalignment='center', fontsize=12, color='darkgreen')

                # Добавляем легенду (только один элемент для среднего)
                if any(len(g) > 0 for g in data_to_plot):
                    plt.legend(loc='upper right')

                plt.tight_layout()
                plot_filename = f"kruskal_box_{feat}.png"
                save_plot(plot_filename)
                report_content += f"![Boxplot](images/{plot_filename})\n\n"
        else:
            report_content += "*Значимых различий не обнаружено.*\n\n"
    else:
        report_content += "*Недостаточно данных для анализа.*\n\n"



    # --- Раздел 3.4: Сводная таблица лучших практик по версии разных методов ---
    report_content += hm.get_header(2, "Сводная таблица лучших практик") + "\n\n"
    report_content += "В этой таблице представлены практики, которые вошли в топ-15 по каждому из использованных методов анализа. "
    report_content += "✅ означает, что практика входит в топ-15 по версии данного метода.\n\n"

    # Собираем топ-15 из каждого метода
    # Корреляционный анализ (Spearman)
    top15_corr = [r['Label'] for r in corr_results[:15]] if len(corr_results) >= 15 else [r['Label'] for r in corr_results]
    
    # ANOVA частота
    top15_anova_prac = [r['Label'] for r in anova_results_prac[:15]] if len(anova_results_prac) >= 15 else [r['Label'] for r in anova_results_prac]
    
    # ANOVA внедрение
    top15_anova_setup = [r['Label'] for r in anova_results_setup[:15]] if len(anova_results_setup) >= 15 else [r['Label'] for r in anova_results_setup]
    
    # Kruskal-Wallis
    top15_kw = [r['label'] for r in kw_results[:15]] if len(kw_results) >= 15 else [r['label'] for r in kw_results]

    # Все практики (только приемы, без дополнительных переменных)
    all_practice_labels = []
    for feat in (prac_cols + setup_cols):
        label = FEATURE_LABELS.get(feat, feat)
        all_practice_labels.append((feat, label))  # сохраняем и внутреннее имя, и метку для отладки

    # Сортируем практики по алфавиту для удобства чтения
    all_practice_labels.sort(key=lambda x: x[1])

    # Формируем строки таблицы
    table_rows = []
    for feat, label in all_practice_labels:
        in_corr = "✅" if label in top15_corr else ""
        in_anova_prac = "✅" if label in top15_anova_prac else ""
        in_kw = "✅" if label in top15_kw else ""
        
        # Сумма галочек (считаем количество "✅")
        sum_checks = sum([1 for x in [in_corr, in_anova_prac, in_kw] if x == "✅"])
        
        table_rows.append({
            'Практика': label,
            'Корреляция (Spearman)': in_corr,
            'ANOVA (частота)': in_anova_prac,
            'Kruskal-Wallis': in_kw,
            'Сумма': sum_checks
        })

    # Сортируем по убыванию суммы, затем по алфавиту
    table_rows.sort(key=lambda x: (-x['Сумма'], x['Практика']))

    # Вывод таблицы в Markdown
    report_content += "| № | Практика | Корреляция (Spearman) | ANOVA (частота) | Kruskal-Wallis | Сумма |\n"
    report_content += "| :---: | :--- | :---: | :---: | :---: | :---: |\n"
    n = 0
    for row in table_rows:
        n = n + 1
        report_content += f"| {n} | {row['Практика']} | {row['Корреляция (Spearman)']} | {row['ANOVA (частота)']} | {row['Kruskal-Wallis']} | {row['Сумма']} |\n"
    report_content += "\n"





    # --- Раздел 4: Анализ шкалы, составленной из практик, одобренных большинством методов ---
    report_content += hm.get_header(1, "Анализ шкалы из практик, получивших наибольший консенсус") + "\n\n"
    report_content += "В этом разделе мы отбираем практики, которые вошли в топ‑15 хотя бы трёх из четырёх использованных методов "
    report_content += "(корреляционный анализ, ANOVA для частоты, критерий Краскела‑Уоллиса). "
    report_content += "Из этих практик формируется новая шкала, и проводится её детальный психометрический анализ.\n\n"

    # Определяем, какие практики попали в топ‑15 по каждому методу (используем уже вычисленные списки)
    top15_corr_feats = [k for k, v in FEATURE_LABELS.items() if v in top15_corr]  # обратное сопоставление метки -> внутреннее имя
    top15_anova_prac_feats = [k for k, v in FEATURE_LABELS.items() if v in top15_anova_prac]
    top15_kw_feats = [k for k, v in FEATURE_LABELS.items() if v in top15_kw]

    # Для каждой практики считаем количество попаданий
    consensus_counts = {}
    for feat, label in all_practice_labels:
        count = 0
        if feat in top15_corr_feats: count += 1
        if feat in top15_anova_prac_feats: count += 1
        if feat in top15_kw_feats: count += 1
        consensus_counts[feat] = count

    # Отбираем практики с количеством попаданий >= 2
    selected_feats = [feat for feat, cnt in consensus_counts.items() if cnt >= 2]
    selected_labels = [FEATURE_LABELS.get(feat, feat) for feat in selected_feats]

    if len(selected_feats) < 2:
        report_content += "Недостаточно практик для построения надёжной шкалы (требуется хотя бы две).\n\n"
    else:
        report_content += f"**Отобрано практик:** {len(selected_feats)}\n\n"
        report_content += "Состав шкалы:\n"
        n = 0
        for label in selected_labels:
            n = n + 1
            report_content += f"{n}. {label}\n"
        report_content += "\n"

        # Создаём датафрейм только с этими практиками и удаляем строки с пропусками
        scale_data = data_scales[selected_feats].dropna()
        n_respondents = len(scale_data)
        report_content += f"**Число респондентов для анализа:** {n_respondents}\n\n"

        if n_respondents < 50:
            report_content += "⚠️ **Предупреждение:** выборка меньше 50 человек, результаты могут быть неустойчивыми.\n\n"

        # --- Надёжность всей шкалы ---
        report_content += hm.get_header(2, "Надёжность общей шкалы") + "\n\n"
        alpha = pg.cronbach_alpha(scale_data)[0]
        omega = calculate_omega_from_data(scale_data)
        report_content += f"- **Альфа Кронбаха:** {alpha:.3f}\n"
        report_content += f"- **Омега МакДональда:** {omega:.3f}\n\n"


        report_content += "#### Коэффициенты при исключении пункта:\n\n"
        report_content += "| Исключенный пункт | Альфа (if deleted) | Омега (if deleted) |\n"
        report_content += "| :--- | :---: | :---: |\n"

        for feat in selected_feats:
            rem = [x for x in selected_feats if x != feat]
            if len(rem) > 1:
                c_alpha = pg.cronbach_alpha(scale_data[rem])[0]
                c_omega = calculate_omega_from_data(scale_data[rem])
            else:
                c_alpha = np.nan
                c_omega = np.nan

            label = FEATURE_LABELS.get(feat, feat)
            report_content += f"| {label} | {c_alpha:.3f} | {c_omega:.3f} |\n"
        report_content += "\n"


        # --- Корреляция суммарного балла всей шкалы с целевой шкалой ---
        report_content += hm.get_header(2, "Валидность по отношению к целевой шкале") + "\n\n"
        # Корреляция суммарного балла шкалы с целевой шкалой продуктивности
        total_score = scale_data.mean(axis=1)  # или scale_data.sum(axis=1)
        target = data_scales.loc[scale_data.index, TARGET_SCALE + '_total']
        r_pearson, p_pearson = pearsonr(total_score, target)
        r_spearman, p_spearman = spearmanr(total_score, target)
        
        report_content += f"- **Корреляция с {TARGET_SCALE} (Пирсон r):** {r_pearson:.3f}, p = {p_pearson:.4f}\n"
        report_content += f"- **Корреляция с {TARGET_SCALE} (Спирмен ρ):** {r_spearman:.3f}, p = {p_spearman:.4f}\n\n"

        # Описательные статистики суммарного балла (среднее)
        report_content += hm.get_header(2, "Описательные характеристики") + "\n\n"
        total_score = scale_data.sum(axis=1)
        report_content += "**Описательные статистики суммарного балла (среднее по пунктам):**\n\n"
        report_content += f"- Среднее: {total_score.mean():.3f}\n"
        report_content += f"- Стандартное отклонение: {total_score.std():.3f}\n"
        report_content += f"- Минимум: {total_score.min():.3f}\n"
        report_content += f"- Максимум: {total_score.max():.3f}\n\n"

        # Гистограмма распределения
        plt.figure(figsize=(8, 5))
        sns.histplot(total_score, bins=8, kde=True, color='skyblue')
        plt.title('Распределение суммарного балла по шкале')
        plt.xlabel('Средний балл по шкале')
        plt.ylabel('Частота')
        plot_filename = f"scale_{i}_hist.png"
        save_plot(plot_filename)
        report_content += f"![Гистограмма](images/{plot_filename})\n\n"


        # --- Пригодность для факторного анализа ---
        report_content += hm.get_header(2, "Пригодность для факторного анализа") + "\n\n"
        kmo_all, kmo_model = calculate_kmo(scale_data)
        chi_square, p_value = calculate_bartlett_sphericity(scale_data)
        report_content += f"- **KMO (мера выборочной адекватности):** {kmo_model:.3f}\n"
        report_content += f"- **Тест сферичности Бартлетта:** χ² = {chi_square:.2f}, p = {p_value:.4f}\n\n"

        # --- Факторный анализ с oblimin вращением ---
        report_content += hm.get_header(2, "Эксплораторный факторный анализ (вращение oblimin)") + "\n\n"

        # Сначала определяем количество факторов (собственные числа > 1)
        fa_temp = FactorAnalyzer(rotation=None)
        fa_temp.fit(scale_data)
        ev, _ = fa_temp.get_eigenvalues()
        n_factors = sum(ev > 1)

        # Таблица объяснённой дисперсии до вращения
        total_variance = ev.sum()  # общая дисперсия (равна числу переменных)
        report_content += "**Объяснённая дисперсия (до вращения):**\n\n"
        report_content += "| Фактор | Собственное значение | % дисперсии | Накопленный % |\n"
        report_content += "| :--- | :---: | :---: | :---: |\n"
        cumul = 0
        for i, val in enumerate(ev, 1):
            percent = (val / total_variance) * 100
            cumul += percent
            report_content += f"| {i} | {val:.3f} | {percent:.2f}% | {cumul:.2f}% |\n"
        report_content += "\n"

        report_content += f"**Факторов с собственным значением > 1:** {n_factors}\n\n"

        if n_factors == 0:
            n_factors = 1  # минимум один фактор

        # Выполняем EFA с oblimin
        fa = FactorAnalyzer(n_factors=n_factors, rotation='oblimin')
        fa.fit(scale_data)

        # Информация о дисперсии после вращения
        var_info = fa.get_factor_variance()
        report_content += "**Объяснённая дисперсия после вращения (суммы квадратов нагрузок):**\n\n"
        report_content += "| Фактор | Дисперсия после вращения | % дисперсии | Накопленный % |\n"
        report_content += "| :--- | :---: | :---: | :---: |\n"
        for i in range(n_factors):
            report_content += f"| {i+1} | {var_info[0][i]:.3f} | {var_info[1][i]*100:.2f}% | {var_info[2][i]*100:.2f}% |\n"
        report_content += "\n"

        # Факторные нагрузки после вращения
        threshold = 0.4
        loadings = fa.loadings_
        report_content += "**Факторные нагрузки (после oblimin):**\n\n"
        # Создаём заголовки столбцов
        header = "| Практика | " + " | ".join([f"Фактор {i+1}" for i in range(n_factors)]) + " |\n"
        separator = "| :--- |" + " :---: |" * n_factors + "\n"
        report_content += header + separator
        for j, feat in enumerate(selected_feats):
            row = f"| {selected_labels[j]} |"
            for i in range(n_factors):
                if loadings[j, i] >= threshold:
                    row += f" 👉{loadings[j, i]:.3f} |"
                else:
                    row += f" {loadings[j, i]:.3f} |"
            report_content += row + "\n"
        report_content += "\n"

        # Корреляции факторов (при косоугольном вращении)
        if n_factors > 1 and hasattr(fa, 'phi_'):
            corr_factors = fa.phi_
            report_content += "**Корреляционная матрица факторов:**\n\n"
            # Создаём матрицу в виде таблицы
            corr_header = "| | " + " | ".join([f"Фактор {i+1}" for i in range(n_factors)]) + " |\n"
            corr_sep = "| :--- |" + " :---: |" * n_factors + "\n"
            report_content += corr_header + corr_sep
            for i in range(n_factors):
                row = f"| Фактор {i+1} |"
                for j in range(n_factors):
                    row += f" {corr_factors[i, j]:.3f} |"
                report_content += row + "\n"
            report_content += "\n"

        # --- Формирование подшкал на основе факторной структуры ---
        if n_factors > 1:
            report_content += hm.get_header(2, "Анализ подшкал, соответствующих факторам") + "\n\n"
            # Для каждого фактора определим практики с нагрузкой > 0.4 (или > 0.3, если слабые)
            subscales = []
            for i in range(n_factors):
                items = [selected_feats[j] for j in range(len(selected_feats)) if abs(loadings[j, i]) > threshold]
                if len(items) >= 2:
                    subscales.append({
                        'factor': i+1,
                        'items': items,
                        'labels': [selected_labels[j] for j in range(len(selected_feats)) if abs(loadings[j, i]) > threshold]
                    })
                else:
                    # Если ни одна практика не превышает порог, берём максимум (хотя бы одну)
                    max_idx = np.argmax(np.abs(loadings[:, i]))
                    items = [selected_feats[max_idx]]
                    subscales.append({
                        'factor': i+1,
                        'items': items,
                        'labels': [selected_labels[max_idx]]
                    })

            # Для каждой подшкалы вычисляем надёжность и корреляцию с целевой шкалой
            for sub in subscales:
                factor_num = sub['factor']
                items = sub['items']
                labels = sub['labels']
                report_content += f"#### Подшкала фактора {factor_num}\n\n"
                report_content += "Состав:\n"
                for lbl in labels:
                    report_content += f"- {lbl}\n"
                report_content += "\n"

                report_content += "**Надёжность подшкалы:**\n\n"
                if len(items) >= 2:
                    sub_data = scale_data[items]
                    alpha_sub = pg.cronbach_alpha(sub_data)[0]
                    omega_sub = calculate_omega_from_data(sub_data)
                    report_content += f"- **Альфа Кронбаха:** {alpha_sub:.3f}\n"
                    report_content += f"- **Омега МакДональда:** {omega_sub:.3f}\n"
                else:
                    report_content += "*Подшкала состоит из одной практики – надёжность не вычисляется.*\n"

                # Корреляция подшкалы с целевой шкалой продуктивности
                # Создаём суммарный балл подшкалы (среднее или сумма)
                sub_score = scale_data[items].sum(axis=1)
                target = data_scales.loc[scale_data.index, TARGET_SCALE + '_total']
                rho, pval = spearmanr(sub_score, target)
                report_content += f"- **Корреляция с целевой шкалой (Spearman ρ):** {rho:.3f}, p = {pval:.4f}\n\n"

                # Описательные статистики суммарного балла (среднее)
                report_content += "**Описательные статистики суммарного балла (среднее по пунктам):**\n\n"
                report_content += f"- Среднее: {sub_score.mean():.3f}\n"
                report_content += f"- Стандартное отклонение: {sub_score.std():.3f}\n"
                report_content += f"- Минимум: {sub_score.min():.3f}\n"
                report_content += f"- Максимум: {sub_score.max():.3f}\n\n"

                # Гистограмма распределения
                plt.figure(figsize=(8, 5))
                sns.histplot(sub_score, bins=8, kde=True, color='skyblue')
                plt.title(f'Распределение суммарного балла подшкалы фактора {factor_num}')
                plt.xlabel('Средний балл по подшкале')
                plt.ylabel('Частота')
                plot_filename = f"subscale_{factor_num}_hist.png"
                save_plot(plot_filename)
                report_content += f"![Гистограмма](images/{plot_filename})\n\n"




    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"Отчет успешно сохранен в {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_report()
