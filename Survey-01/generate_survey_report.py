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
from scipy.stats import spearmanr, f_oneway

# --- Настройки ---
INPUT_FILE = 'Большое исследование джедайских приемов (Responses).csv'
REPORT_DIR = 'C:\\Users\\maxim\\OneDrive\\Obsidian\\MyBrain\\Мои исследования\\Джедайская шкала\\'
REPORT_DIR = ''
OUTPUT_FILE = REPORT_DIR + 'survey_report.md'
IMAGES_DIR = REPORT_DIR + 'images'

# Целевая шкала для корреляционного анализа (можно менять)
# TARGET_SCALE = 'MIJS-2+'
TARGET_SCALE = 'MIJS-3+'
# TARGET_SCALE = 'MIJS-2'
# TARGET_SCALE = 'MIJS'

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
        "MIJS-3+": ['mijs_q2', 'mijs_q4', 'mijs_q6', 'jedi_inv']
    }

    summary_rows = []

    for scale_name, items in scales.items():
        report_content += hm.get_header(2, f"Шкала: {scale_name}") + "\n\n"
        
        # Расчет суммарного балла (сумма)
        data_scales[scale_name + '_total'] = data_scales[items].sum(axis=1)
        
        # График распределения баллов по шкале
        plt.figure(figsize=(8, 5))
        sns.histplot(data_scales[scale_name + '_total'], bins=8, kde=True, color='indigo')
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

    # --- Раздел 3: Корреляционный анализ ---
    report_content += hm.get_header(1, "Корреляционный анализ (ранговая корреляция)") + "\n\n"
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
            
            sig = ""
            if pval < 0.0001: sig = "****"
            elif pval < 0.001: sig = "***"
            elif pval < 0.01: sig = "**"
            elif pval < 0.05: sig = "*"
            
            corr_results.append({
                'Label': FEATURE_LABELS.get(feat, feat),
                'Rho': rho,
                'PVal': pval,
                'Sig': sig
            })

    # Сортировка по p-value (по возрастанию) и затем по Rho (по возрастанию)
    corr_results.sort(key=lambda x: (x['PVal'], x['Rho']))

    report_content += "| № | Прием | ρ (Rho) | p-value | Знач. |\n"
    report_content += "| :--- | :--- | :---: | :---: | :---: |\n"
    for i, res in enumerate(corr_results, 1):
        report_content += f"| {i} | {res['Label']} | {res['Rho']:.3f} | {res['PVal']:.4f} | {res['Sig']} |\n"
    report_content += "\n*Примечание: * p < 0.05, ** p < 0.01, *** p < 0.001. Отрицательная корреляция означает, что использование приема связано с более низким баллом по шкале MIJS (меньше стресса/завала).* \n\n"

    # --- Раздел 4: Сравнение средних (ANOVA) ---
    report_content += hm.get_header(1, "Сравнение средних (ANOVA)") + "\n\n"
    report_content += "В данном разделе анализируется, как частота использования или уровень внедрения приема связаны со средним значением продуктивности. "
    report_content += "Для проверки значимости различий между группами используется однофакторный дисперсионный анализ (One-way ANOVA).\n\n"

    # --- 4.1. Частота использования приемов ---
    report_content += hm.get_header(2, "Частота использования приемов") + "\n\n"
    report_content += "Респонденты разделены на группы по частоте использования (0 — никогда, 4 — всегда).\n\n"

    anova_results_prac = []
    for feat in prac_cols:
        groups = []
        means = {}
        for level in range(5):
            group_data = data_scales[data_scales[feat] == level][TARGET_SCALE + '_total'].dropna()
            if len(group_data) > 3:
                groups.append(group_data)
                means[level] = group_data.mean()
            else:
                means[level] = np.nan
        
        if len(groups) >= 2:
            f_stat, pval = f_oneway(*groups)
            sig = "****" if pval < 0.0001 else "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            anova_results_prac.append({
                'Label': FEATURE_LABELS.get(feat, feat),
                'Means': [means.get(l, np.nan) for l in range(5)],
                'F': f_stat, 'PVal': pval, 'Sig': sig
            })

    anova_results_prac.sort(key=lambda x: x['PVal'])
    report_content += "| № | Прием | Ср. (0) | Ср. (1) | Ср. (2) | Ср. (3) | Ср. (4) | F | p-value | Знач. |\n"
    report_content += "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    for i, res in enumerate(anova_results_prac, 1):
        m_str = " | ".join([f"{m:.1f}" if not np.isnan(m) else "-" for m in res['Means']])
        report_content += f"| {i} | {res['Label']} | {m_str} | {res['F']:.2f} | {res['PVal']:.4f} | {res['Sig']} |\n"
    report_content += f"\n*Примечание: 0 — никогда/редко, 4 — всегда. В ячейках указан средний балл по шкале {TARGET_SCALE}. Чем ниже балл, тем выше продуктивность.* \n\n"

    # --- 4.2. Уровень внедрения приемов ---
    report_content += hm.get_header(2, "Уровень внедрения приемов") + "\n\n"
    report_content += "Респонденты разделены на группы по уровню внедрения (0 — не применил(а), 3 — применил(а) по максимуму).\n\n"

    anova_results_setup = []
    for feat in setup_cols:
        groups = []
        means = {}
        for level in range(4): # 0, 1, 2, 3
            group_data = data_scales[data_scales[feat] == level][TARGET_SCALE + '_total'].dropna()
            if len(group_data) > 3:
                groups.append(group_data)
                means[level] = group_data.mean()
            else:
                means[level] = np.nan
        
        if len(groups) >= 2:
            f_stat, pval = f_oneway(*groups)
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            anova_results_setup.append({
                'Label': FEATURE_LABELS.get(feat, feat),
                'Means': [means.get(l, np.nan) for l in range(4)],
                'F': f_stat, 'PVal': pval, 'Sig': sig
            })

    anova_results_setup.sort(key=lambda x: x['PVal'])
    report_content += "| № | Прием | Ср. (0) | Ср. (1) | Ср. (2) | Ср. (3) | F | p-value | Знач. |\n"
    report_content += "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    for i, res in enumerate(anova_results_setup, 1):
        m_str = " | ".join([f"{m:.1f}" if not np.isnan(m) else "-" for m in res['Means']])
        report_content += f"| {i} | {res['Label']} | {m_str} | {res['F']:.2f} | {res['PVal']:.4f} | {res['Sig']} |\n"
    report_content += f"\n*Примечание: 0 — не применил(а), 3 — по максимуму. В ячейках указан средний балл по шкале {TARGET_SCALE}.*\n\n"

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"Отчет успешно сохранен в {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_report()
