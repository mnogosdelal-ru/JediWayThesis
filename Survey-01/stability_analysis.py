import pandas as pd
import numpy as np
import pingouin as pg
from factor_analyzer import FactorAnalyzer
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.utils.validation
import factor_analyzer.factor_analyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from scipy.stats import spearmanr, f_oneway, kruskal, pearsonr
from statsmodels.stats.multitest import multipletests
import textwrap

# --- Настройки (из основного скрипта) ---
INPUT_FILE = 'Большое исследование джедайских приемов (Responses).csv'
REPORT_DIR = os.path.dirname(os.path.abspath(__file__))
#REPORT_DIR = 'C:\\Users\\maxim\\OneDrive\\Obsidian\\MyBrain\\Мои исследования\\Джедайская шкала\\'
IMAGES_DIR = os.path.join(REPORT_DIR, 'images')
ROBUSTNESS_REPORT = os.path.join(REPORT_DIR, 'robustness_report.md')

# Целевая шкала для анализа (можно менять)
# Доступные: 'MIJS-2+', 'MIJS-3+', 'single_item', 'MIJS-2', 'MIJS'
TARGET_SCALE = 'MIJS-2+'

# Определения составов шкал
SCALES = {
    "MIJS": ['mijs_q1', 'mijs_q2', 'mijs_q3', 'mijs_q4', 'mijs_q5', 'mijs_q6'],
    "MIJS-2": ['mijs_q2', 'mijs_q3', 'mijs_q4', 'mijs_q6'],
    "MIJS-2+": ['mijs_q2', 'mijs_q3', 'mijs_q4', 'mijs_q6', 'jedi_inv'],
    "MIJS-3+": ['mijs_q2', 'mijs_q4', 'mijs_q6', 'jedi_inv'],
    "single_item": ['jedi_inv']
}

# Настройки симуляции
N_ITERATIONS = 1000 # Установлено 10 для отладки
SAMPLE_SIZE = 180  # Уменьшено пользователем для большей жесткости отбора
TOP_K = 15         
CORE_THRESHOLD = 75 # Порог для включения в "Ядро"
CONSENSUS_THRESHOLD = 3 # Порог для включения в "Консенсус"

if not os.path.exists(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)

# Исправление совместимости factor_analyzer с новыми версиями scikit-learn
_original_check_array = sklearn.utils.validation.check_array
def _patched_check_array(*args, **kwargs):
    if 'force_all_finite' in kwargs:
        kwargs['ensure_all_finite'] = kwargs.pop('force_all_finite')
    return _original_check_array(*args, **kwargs)

sklearn.utils.validation.check_array = _patched_check_array
factor_analyzer.factor_analyzer.check_array = _patched_check_array

def calculate_omega_from_data(data_subset):
    """Вычисляет омегу МакДональда на основе 1-факторной модели EFA."""
    if data_subset.empty or data_subset.shape[1] < 2:
        return np.nan
    fa = FactorAnalyzer(n_factors=1, rotation=None)
    try:
        fa.fit(data_subset)
        loadings = fa.loadings_.flatten()
        sum_loadings = np.sum(loadings)
        sum_uniqueness = np.sum(1 - loadings**2)
        omega = (sum_loadings**2) / (sum_loadings**2 + sum_uniqueness)
        return omega
    except:
        return np.nan

def load_and_preprocess():
    """Загрузка и маппинг данных (аналогично generate_survey_report.py)"""
    df = pd.read_csv(INPUT_FILE)
    
    COL_MAPPING = {
        1: 'age', 2: 'gender', 4: 'jedi_single_raw',
        5: 'mijs_q1', 6: 'mijs_q2', 7: 'mijs_q3', 8: 'mijs_q4', 9: 'mijs_q5', 10: 'mijs_q6',
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
        42: 'setup_blue_filter', 43: 'setup_sleep_audit', 44: 'setup_dnd_mode',
        45: 'setup_remove_social', 46: 'setup_no_red_dots', 47: 'setup_adblock',
        48: 'setup_chat_notif_off', 49: 'setup_email_notif_off', 50: 'setup_app_push_off',
        51: 'setup_watch_notif_off', 52: 'setup_auto_read_email',
        56: 'luckiness', 59: 'remote_work'
    }
    
    data = df.iloc[:, list(COL_MAPPING.keys())].copy()
    data.columns = [COL_MAPPING[i] for i in COL_MAPPING.keys()]

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
        if "редко" in val and ("месяц" in val or "месяца" in val): return 1
        if "иногда" in val and ("неделю" in val or "месяц" in val): return 2
        if "часто" in val and ("неделю" in val or "месяц" in val): return 3
        if "практически всегда" in val or "постоянно" in val: return 4
        return np.nan

    def map_implementation(val):
        if not isinstance(val, str): return val
        val = val.lower()
        if "не применил" in val: return 0
        if "но не полностью" in val: return 1
        if "небольшими исключениями" in val: return 2
        if "по максимуму" in val or "у меня этого нет" in val: return 3
        return np.nan

    data['jedi_single'] = data['jedi_single_raw'].apply(map_jedi)
    data['jedi_inv'] = 6 - data['jedi_single']
    
    # Пункты MIJS (убеждаемся, что они числовые)
    mijs_items = ['mijs_q1', 'mijs_q2', 'mijs_q3', 'mijs_q4', 'mijs_q5', 'mijs_q6']
    for col in mijs_items:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    prac_cols = [c for c in data.columns if c.startswith('prac_')]
    for col in prac_cols: data[col] = data[col].apply(map_frequency)
    
    setup_cols = [c for c in data.columns if c.startswith('setup_')]
    for col in setup_cols: data[col] = data[col].apply(map_implementation)

    # Расчет целевой шкалы
    if TARGET_SCALE not in SCALES:
        raise ValueError(f"Unknown TARGET_SCALE: {TARGET_SCALE}. Available: {list(SCALES.keys())}")
    
    target_items = SCALES[TARGET_SCALE]
    data[TARGET_SCALE + '_total'] = data[target_items].sum(axis=1)
    
    # Очистка (по пунктам выбранной шкалы)
    data = data.dropna(subset=target_items).copy()
    
    # Метки для отчета
    FEATURE_LABELS = {name: df.columns[i] for i, name in COL_MAPPING.items()}
    
    return data, prac_cols, setup_cols, FEATURE_LABELS

def get_top_consensus_practices(df_scales, prac_cols, setup_cols):
    """Ядро алгоритма отбора (топ-15 консенсус)"""
    all_practices = prac_cols + setup_cols
    target_data = df_scales[TARGET_SCALE + '_total']
    
    # 1. Spearman Correlation
    corr_pvals = []
    corr_feats = []
    for feat in all_practices:
        valid = df_scales[feat].notna() & target_data.notna()
        if valid.sum() > 30:
            rho, p = spearmanr(df_scales.loc[valid, feat], target_data[valid])
            corr_pvals.append(p)
            corr_feats.append((feat, p, rho))
    
    if corr_pvals:
        _, p_adj, _, _ = multipletests(corr_pvals, method='fdr_bh')
        corr_results = [(f[0], p_adj[i], f[2]) for i, f in enumerate(corr_feats)]
        corr_results.sort(key=lambda x: (x[1], -abs(x[2])))
        top_corr = [x[0] for x in corr_results[:TOP_K]]
    else:
        top_corr = []

    # 2. ANOVA (Frequency Only as in original)
    anova_pvals = []
    anova_feats = []
    for feat in prac_cols:
        groups = [df_scales[df_scales[feat] == l][TARGET_SCALE + '_total'].dropna() for l in range(5)]
        groups = [g for g in groups if len(g) > 10]
        if len(groups) >= 2:
            _, p = f_oneway(*groups)
            anova_pvals.append(p)
            anova_feats.append((feat, p))
    
    if anova_pvals:
        _, p_adj, _, _ = multipletests(anova_pvals, method='fdr_bh')
        anova_results = [(f[0], p_adj[i]) for i, f in enumerate(anova_feats)]
        anova_results.sort(key=lambda x: x[1])
        top_anova = [x[0] for x in anova_results[:TOP_K]]
    else:
        top_anova = []

    # 3. Kruskal-Wallis (3 groups: 15% - 75%)
    p15 = target_data.quantile(0.15)
    p75 = target_data.quantile(0.75)
    df_scales['prod_group'] = pd.cut(target_data, bins=[-np.inf, p15, p75, np.inf], labels=['P', 'S', 'N'])
    
    kw_pvals = []
    kw_feats = []
    for feat in all_practices:
        low = df_scales.loc[df_scales['prod_group'] == 'P', feat].dropna()
        mid = df_scales.loc[df_scales['prod_group'] == 'S', feat].dropna()
        high = df_scales.loc[df_scales['prod_group'] == 'N', feat].dropna()
        if len(low) >= 2 and len(mid) >= 2 and len(high) >= 2:
            _, p = kruskal(low, mid, high)
            kw_pvals.append(p)
            kw_feats.append((feat, p))
            
    if kw_pvals:
        _, p_adj, _, _ = multipletests(kw_pvals, method='fdr_bh')
        kw_results = [(f[0], p_adj[i]) for i, f in enumerate(kw_feats)]
        kw_results.sort(key=lambda x: x[1])
        top_kw = [x[0] for x in kw_results[:TOP_K]]
    else:
        top_kw = []

    # Consensus (appears in at least 2 top-15 lists)
    counts = {}
    for f in (top_corr + top_anova + top_kw):
        counts[f] = counts.get(f, 0) + 1
    
    consensus = [f for f, count in counts.items() if count >= CONSENSUS_THRESHOLD]
    return set(consensus)

def run_stability_analysis():
    print("Загрузка данных...")
    data, prac_cols, setup_cols, FEATURE_LABELS = load_and_preprocess()
    
    stability_counts = {f: 0 for f in (prac_cols + setup_cols)}
    
    print(f"Запуск симуляции ({N_ITERATIONS} итераций)...")
    for i in range(1, N_ITERATIONS + 1):
        if i % 10 == 0:
            print(f"Итерация {i}/{N_ITERATIONS}...")
        subset = data.sample(n=SAMPLE_SIZE, replace=False)
        top_set = get_top_consensus_practices(subset, prac_cols, setup_cols)
        for feat in top_set:
            stability_counts[feat] += 1
            
    # Результаты
    results = []
    for feat, count in stability_counts.items():
        score = (count / N_ITERATIONS) * 100
        results.append({
            'Feature': feat,
            'Label': FEATURE_LABELS.get(feat, feat),
            'Stability': score
        })
    
    results.sort(key=lambda x: x['Stability'], reverse=True)
    df_res = pd.DataFrame(results)
    
    # Отчет
    print(f"Генерация отчета в {ROBUSTNESS_REPORT}...")
    with open(ROBUSTNESS_REPORT, 'w', encoding='utf-8') as f:
        f.write("# Анализ устойчивости (Stability Selection)\n\n")
        f.write(f"Параметры: {N_ITERATIONS} итераций, подвыборка {SAMPLE_SIZE} человек (~80%).\n")
        f.write(f"Целевая шкала: {TARGET_SCALE}\n")
        f.write(f"Stability Score показывает, в каком проценте случаев практика попадала в 'Топ-15 консенсуса' в {CONSENSUS_THRESHOLD} из 3 алгоритмов при случайных изменениях в выборке.\n\n")

        f.write("![Stability Selection Graph](images/stability_selection.png)\n\n")
        
        f.write("| № | Практика | Stability Score (%) | Статус |\n")
        f.write("| :--- | :--- | :---: | :--- |\n")
        for i, row in df_res.iterrows():
            status = "🔥 Ядро (Strong)" if row['Stability'] >= CORE_THRESHOLD else "✅ Устойчивая" if row['Stability'] >= 50 else "⚠️ Нестабильная" if row['Stability'] > 20 else "❌ Шумовая"
            f.write(f"| {i+1} | {row['Label']} | {row['Stability']:.1f}% | {status} |\n")
            
    # Визуализация
    plt.figure(figsize=(10, 14))
    df_plot = df_res[df_res['Stability'] > 5].copy() # Только те, кто хоть раз попал
    
    # Перенос длинных подписей
    df_plot['Label_Wrapped'] = df_plot['Label'].apply(lambda x: "\n".join(textwrap.wrap(x, width=40)))
    
    sns.barplot(data=df_plot, x='Stability', y='Label_Wrapped', hue='Stability', palette='viridis', legend=False)
    plt.axvline(CORE_THRESHOLD, color='red', linestyle='--', label=f'{CORE_THRESHOLD}% Threshold')
    plt.title(f"Stability Selection: Частота попадания в ТОП-15 ({TARGET_SCALE})")
    plt.xlabel("Stability Score (%)")
    plt.ylabel("Практика")
    plt.tight_layout()
    plot_path = os.path.join(IMAGES_DIR, "stability_selection.png")
    plt.savefig(plot_path)
    plt.close()
    # --- Дополнительный анализ Ядра ---
    core_feats = [r['Feature'] for r in results if r['Stability'] >= CORE_THRESHOLD]
    
    if len(core_feats) >= 2:
        print(f"Анализ ядра ({len(core_feats)} практик)...")
        core_data = data[core_feats].dropna()
        
        with open(ROBUSTNESS_REPORT, 'a', encoding='utf-8') as f:
            f.write("\n---\n\n")
            f.write("# Психометрический анализ 'Ядра' практик\n\n")
            f.write(f"В ядро вошли практики с устойчивостью >= {CORE_THRESHOLD}%.\n\n")
            
            # Надежность
            alpha = pg.cronbach_alpha(core_data)[0]
            omega = calculate_omega_from_data(core_data)
            f.write(f"- **Альфа Кронбаха:** {alpha:.3f}\n")
            f.write(f"- **Омега МакДональда:** {omega:.3f}\n\n")
            
            # Валидность
            total_core = core_data.mean(axis=1)
            target = data.loc[core_data.index, TARGET_SCALE + '_total']
            r_pears, _ = pearsonr(total_core, target)
            r_spear, _ = spearmanr(total_core, target)
            f.write(f"- **Корреляция с целевой шкалой (Pearson r):** {r_pears:.3f}\n")
            f.write(f"- **Корреляция с целевой шкалой (Spearman ρ):** {r_spear:.3f}\n\n")
            
            # EFA
            kmo = calculate_kmo(core_data)[1]
            chi, pval = calculate_bartlett_sphericity(core_data)
            f.write("### Факторный анализ ядра\n\n")
            f.write(f"- **KMO:** {kmo:.3f}\n")
            f.write(f"- **Тест Бартлетта:** p={pval:.4f}\n\n")
            
            # Определяем количество факторов по Кайзеру (EV > 1)
            fa_init = FactorAnalyzer(n_factors=1, rotation=None)
            fa_init.fit(core_data)
            ev, _ = fa_init.get_eigenvalues()
            n_factors = sum(ev > 1)
            if n_factors < 1: n_factors = 1

            # Ре-ран с нужным количеством факторов
            # Если фактор один - без ротации, если больше - с varimax для удобства интерпретации
            fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax' if n_factors > 1 else None)
            fa.fit(core_data)
            loadings = fa.loadings_
            var_info = fa.get_factor_variance()
            
            if n_factors == 1:
                f.write(f"**Дисперсия, объясненная фактором (EFA):** {var_info[1][0]*100:.1f}%\n")
            else:
                total_var = var_info[2][-1] * 100
                f.write(f"**Количество выделенных факторов (EV > 1):** {n_factors}\n")
                f.write(f"**Суммарная объясненная дисперсия (EFA, {n_factors} факт.):** {total_var:.1f}%\n")
            
            f.write(f"**Собственные числа (Eigenvalues) и полная дисперсия (PCA):**\n\n")
            
            f.write("| Фактор | Собств. число | % Полной дисперсии |\n")
            f.write("| :--- | :---: | :---: |\n")
            total_items = len(ev)
            for i, val in enumerate(ev):
                var_pct = (val / total_items) * 100
                f.write(f"| {i+1} | {val:.3f} | {var_pct:.1f}% |\n")
            f.write("\n")
            
            headers = ["Практика"] + [f"Фактор {j+1}" for j in range(n_factors)]
            header_row = "| " + " | ".join(headers) + " |"
            sep_row = "| :--- | " + " | ".join([":---:" for _ in range(n_factors)]) + " |"
            f.write(header_row + "\n")
            f.write(sep_row + "\n")
            
            for i, feat in enumerate(core_feats):
                label = FEATURE_LABELS.get(feat, feat)
                lds = loadings[i]
                if n_factors == 1:
                    lds_str = f"{lds[0]:.3f}"
                else:
                    lds_str = " | ".join([f"{l:.3f}" for l in lds])
                f.write(f"| {label} | {lds_str} |\n")
            
            # --- Сравнение Ядра с Целевой шкалой ---
            print("Валидация ядра (Ядро vs Целевая шкала)...")
            target_col = TARGET_SCALE + '_total'
            # Оцениваем предсказательную силу через линейную регрессию
            # Обучаем простую модель: Целевая шкала ~ Среднее по Ядру
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score
            
            X = total_core.values.reshape(-1, 1)
            y = target.values
            
            reg = LinearRegression().fit(X, y)
            y_pred = reg.predict(X)
            r2 = r2_score(y, y_pred)
            
            f.write("\n### Прогностическая ценность 'Ядра'\n\n")
            f.write(f"Оценка связи между средним баллом по практикам 'Ядра' и целевой шкалой {TARGET_SCALE}.\n\n")
            f.write(f"- **Коэффициент детерминации (R²):** {r2:.3f}\n")
            f.write(f"- **Интерпретация:** Ядро практик объясняет **{r2*100:.1f}%** дисперсии целевой шкалы.\n\n")
            
            f.write("![Core vs Target](images/core_vs_target.png)\n\n")
            
            # Построение графика
            plt.figure(figsize=(10, 6))
            sns.regplot(x=total_core, y=target, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
            plt.title(f"Связь Ядра практик и целевой шкалы {TARGET_SCALE} (R² = {r2:.3f})")
            plt.xlabel("Средний балл по практикам Ядра")
            plt.ylabel(f"Суммарный балл по шкале {TARGET_SCALE}")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            core_plot_path = os.path.join(IMAGES_DIR, "core_vs_target.png")
            plt.savefig(core_plot_path)
            plt.close()
    
    print("Готово!")

if __name__ == "__main__":
    run_stability_analysis()
