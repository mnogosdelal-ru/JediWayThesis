#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Monkeypatch for scikit-learn 1.4+ / 1.8+ compatibility with factor_analyzer
try:
    import sklearn.utils
    import sklearn.utils.validation
    
    def patched_check_array(array, *args, **kwargs):
        if 'force_all_finite' in kwargs:
            kwargs['ensure_all_finite'] = kwargs.pop('force_all_finite')
        return original_check_array(array, *args, **kwargs)

    # Patch in all possible locations where factor_analyzer might have imported it from
    original_check_array = sklearn.utils.validation.check_array
    sklearn.utils.validation.check_array = patched_check_array
    sklearn.utils.check_array = patched_check_array
    
    # Some versions of factor_analyzer might import it very early or from internal sklearn paths
    import sklearn
    if hasattr(sklearn, 'check_array'):
        sklearn.check_array = patched_check_array

except Exception:
    pass

import sys
import io
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.decomposition import PCA
from factor_analyzer import FactorAnalyzer
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import os

# Setup encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(SCRIPT_DIR, 'Тест многопунктовой шкалы - RawResponses (1).csv')
OUTPUT_MD = os.path.join(SCRIPT_DIR, 'mis_full_psychometric_report.md')
#OUTPUT_MD = r'C:\Users\maxim\OneDrive\Obsidian\MyBrain\Мои исследования\Джедайская шкала\mis_full_psychometric_report.md'

MIS_TEXTS = {
    1: "Каждый день появляются незапланированные срочные задачи, которые я не могу отложить",
    2: "Я не могу предсказать, какие срочные дела потребуют моего внимания завтра",
    3: "Мне часто приходится менять свои планы из-за внезапно появившихся срочных задач",
    4: "Срочные дела часто прерывают мою работу над важным проектом",
    5: "Я часто теряю контроль над своим расписанием из-за непредвиденных срочных дел",
    6: "За один день больше 3-х раз что-то срочное требует моей немедленной реакции",
    7: "Я не могу планировать свой день, потому что могут прилететь срочные дела",
    8: "Срочные задачи приходят в самые неудобные моменты, когда я был в потоке работы",
    9: "Я не могу отвести время на важное, потому что оно полностью занято срочным",
    10: "Мне не хватает ресурсов (времени/энергии) одновременно на срочное и на важное",
    11: "Если бы было меньше срочных дел, я мог бы лучше развиваться профессионально",
    12: "Из-за срочных дел я постоянно откладываю работу над своими важными проектами",
    13: "Я начинаю работу над важным, но часто срочное заставляет меня её бросить",
    14: "На важные проекты остаётся только то время, когда я уже уставший",
    15: "Срочные дела забирают мою лучшую энергию, важное получает остатки",
    16: "Я живу в режиме \"тушения пожаров\" большую часть времени",
    17: "Мои действия в основном реактивные, а не проактивные",
    18: "Я редко действую согласно плану; чаще реагирую на обстоятельства",
    19: "Моя работа — это в основном решение проблем, которые уже произошли",
    20: "Я в большей степени тушу кризисы, чем их предотвращаю",
    21: "У меня есть ощущение, что я всегда спешу, чтобы справиться со срочным",
    22: "Из-за срочных дел я отстаю от своих долгосрочных целей",
    23: "Мне трудно видеть прогресс в важных проектах, потому что до них редко доходит очередь",
    24: "Срочные дела мешают мне развиваться в нужном мне направлении",
    25: "Я не могу набрать нужную скорость в важном проекте из-за постоянных прерываний",
    26: "Мой потенциал не реализуется из-за необходимости постоянно реагировать на срочное",
    27: "К концу дня у меня почти нет энергии на важное",
    28: "Я чувствую, что мои ресурсы (время и энергия) исчерпываются на срочное",
    29: "Усталость вызвана у меня в первую очередь с режимом \"тушения пожаров\"",
    30: "Если бы было меньше срочных дел, я мог бы работать более эффективно и спокойнее"
}

MIS_BLOCKS = {
    "Block_1_Disruptions": list(range(1, 9)),
    "Block_2_Resource_Conflict": list(range(9, 16)),
    "Block_3_Reactivity": list(range(16, 22)),
    "Block_4_LongTerm_Impact": list(range(22, 27)),
    "Block_5_Exhaustion": list(range(27, 31)),
}

BLOCK_NAMES_RU = {
    "Block_1_Disruptions": "Вторжения и нарушения планов",
    "Block_2_Resource_Conflict": "Дефицит ресурсов для важного",
    "Block_3_Reactivity": "Реактивность / 'Пожарный режим'",
    "Block_4_LongTerm_Impact": "Влияние на долгосрочные цели",
    "Block_5_Exhaustion": "Истощение ресурсов и усталость",
    "MIS_Total": "Общий балл MIS (30 пунктов)"
}

OSD_MAPPING = {
    'OSD_planomernost': 'Планомерность',
    'OSD_celeustremlonnost': 'Целеустремленность',
    'OSD_nastoichivost': 'Настойчивость',
    'OSD_fiksacia': 'Фиксация на структурировании',
    'OSD_samoorganizacia': 'Самоорганизация',
    'OSD_orientacia_nastoyaschee': 'Ориентация на настоящее',
    'OSD_total': 'ОСД Итого'
}

# ============================================================
# UTILS
# ============================================================
def cronbach_alpha(df):
    itemvars = df.var(axis=0, ddof=1)
    tvar = df.sum(axis=1).var(ddof=1)
    n = df.shape[1]
    if tvar == 0 or n <= 1: return 0
    return (n / (n - 1)) * (1 - itemvars.sum() / tvar)

def format_p(p):
    if p < 0.001: return "< 0.001***"
    if p < 0.01: return f"{p:.3f}**"
    if p < 0.05: return f"{p:.3f}*"
    return f"{p:.3f}"

def mcdonald_omega(loadings):
    # loadings should be a pandas Series or 1D array of loadings on one factor
    sums_loadings = np.sum(loadings)**2
    sum_unique_vars = np.sum(1 - loadings**2)
    return sums_loadings / (sums_loadings + sum_unique_vars)

# ============================================================
# ANALYSIS CLASS
# ============================================================
class MISAnalyzer:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.clean_data()
        
    def clean_data(self):
        # Numeric conversion
        mis_cols = [f'MIS_q_{i}' for i in range(1, 31)]
        for col in mis_cols + ['single_item', 'age']:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Calculate MIS blocks
        for bname, items in MIS_BLOCKS.items():
            cols = [f'MIS_q_{i}' for i in items]
            self.df[bname] = self.df[cols].sum(axis=1)
        
        self.df['MIS_Total'] = self.df[[f'MIS_q_{i}' for i in range(1, 31)]].sum(axis=1)
        
        # MIS-5 Selection (Top loadings on F1)
        self.mis_5_cols = ['MIS_q_28', 'MIS_q_24', 'MIS_q_16', 'MIS_q_26', 'MIS_q_15']
        self.df['MIS_5'] = self.df[self.mis_5_cols].sum(axis=1)
        
        self.df = self.df.dropna(subset=['MIS_Total', 'single_item'])

    def get_psychometrics(self, cols=None):
        if cols is None:
            cols = [f'MIS_q_{i}' for i in range(1, 31)]
        
        alpha = cronbach_alpha(self.df[cols])
        
        block_alphas = {}
        if len(cols) > 6: # Only for full scale
            for bname, items in MIS_BLOCKS.items():
                b_cols = [f'MIS_q_{i}' for i in items]
                block_alphas[bname] = cronbach_alpha(self.df[b_cols])
            
        item_stats = []
        subset = self.df[cols]
        total_score = subset.sum(axis=1)
        
        for col in cols:
            q_num = int(col.split('_')[-1])
            desc = subset[col].describe()
            other_sum = total_score - subset[col]
            r_it, _ = stats.pearsonr(subset[col], other_sum)
            alpha_del = cronbach_alpha(subset.drop(columns=[col]))
            
            item_stats.append({
                'id': q_num,
                'mean': desc['mean'],
                'sd': desc['std'],
                'rit': r_it,
                'alpha_del': alpha_del,
                'text': MIS_TEXTS[q_num]
            })
            
        return {
            'alpha': alpha,
            'block_alphas': block_alphas,
            'item_stats': item_stats,
            'n': len(self.df)
        }

    def get_factor_structure(self, cols=None, n_factors=None):
        if cols is None:
            cols = [f'MIS_q_{i}' for i in range(1, 31)]
        data = self.df[cols]
        
        # PCA for eigenvalues and total variance
        pca = PCA().fit(data)
        evs = pca.explained_variance_
        pca_var = pca.explained_variance_ratio_
        
        # EFA for loadings
        loadings = None
        efa_var = None
        if n_factors is None:
            n_factors = 1 if len(cols) == 5 else 5
            
        try:
            fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax' if n_factors > 1 else None)
            fa.fit(data)
            loadings = pd.DataFrame(fa.loadings_, index=cols, columns=[f'Factor {i+1}' for i in range(n_factors)])
            efa_var = fa.get_factor_variance()[1]
        except Exception as e:
            print(f"    EFA Error for {len(cols)} items: {e}")
            # PCA Loadings as fallback (scaled by sqrt of eigenvalues)
            pca_f = PCA(n_components=n_factors)
            std_data = (data - data.mean()) / data.std()
            pca_f.fit(std_data)
            loadings_values = pca_f.components_.T * np.sqrt(pca_f.explained_variance_)
            loadings = pd.DataFrame(loadings_values, index=cols, columns=[f'Factor {i+1}' for i in range(n_factors)])
            efa_var = pca_f.explained_variance_ratio_
        
        # EFA metrics
        kmo = None
        bartlett_p = None
        try:
            from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
            kmo_all, kmo_model = calculate_kmo(data)
            kmo = kmo_model
            chi_sq, bartlett_p = calculate_bartlett_sphericity(data)
        except:
            pass
            
        return {
            'eigenvalues': evs,
            'var_explained_pca': pca_var,
            'loadings': loadings,
            'var_explained_efa': efa_var,
            'kmo': kmo,
            'bartlett_p': bartlett_p
        }

    def get_correlations(self):
        mis_scales = list(MIS_BLOCKS.keys()) + ['MIS_Total', 'MIS_5']
        osd_scales = list(OSD_MAPPING.keys())
        criteria = osd_scales + ['single_item']
        
        corr_matrix = pd.DataFrame(index=mis_scales, columns=criteria)
        p_matrix = pd.DataFrame(index=mis_scales, columns=criteria)
        
        for m in mis_scales:
            for c in criteria:
                if c in self.df.columns:
                    r, p = stats.pearsonr(self.df[m], self.df[c])
                    corr_matrix.loc[m, c] = r
                    p_matrix.loc[m, c] = p
        
        return corr_matrix, p_matrix

    def get_demographics(self):
        results = {}
        target = 'MIS_Total'
        
        # Gender
        males = self.df[self.df['gender'] == 'Мужской'][target]
        females = self.df[self.df['gender'] == 'Женский'][target]
        if len(males) > 1 and len(females) > 1:
            t_stat, p_val = stats.ttest_ind(males, females, equal_var=False)
            results['gender'] = {
                'male_mean': males.mean(),
                'female_mean': females.mean(),
                't': t_stat,
                'p': p_val
            }
            
        # Age
        if 'age' in self.df.columns:
            r, p = stats.pearsonr(self.df['age'].dropna(), self.df.loc[self.df['age'].notna(), target])
            results['age'] = {'r': r, 'p': p}
            
        # Position
        pos_counts = self.df['position'].value_counts()
        valid_pos = pos_counts[pos_counts >= 5].index.tolist()
        if len(valid_pos) > 1:
            groups = [self.df[self.df['position'] == p][target] for p in valid_pos]
            f_stat, p_val = stats.f_oneway(*groups)
            pos_means = self.df[self.df['position'].isin(valid_pos)].groupby('position')[target].mean()
            results['position'] = {
                'f': f_stat,
                'p': p_val,
                'means': pos_means,
                'counts': pos_counts[valid_pos]
            }
            
        return results

# ============================================================
# REPORT GENERATOR
# ============================================================
def run_report():
    print("Loading analyzer...")
    analyzer = MISAnalyzer(FILE_PATH)
    
    print("Computing metrics (Full Scale)...")
    psy_full = analyzer.get_psychometrics()
    factors_full = analyzer.get_factor_structure()
    
    print("Computing metrics (MIS-5)...")
    psy_5 = analyzer.get_psychometrics(analyzer.mis_5_cols)
    factors_5 = analyzer.get_factor_structure(analyzer.mis_5_cols)
    omega_5 = mcdonald_omega(factors_5['loadings'].iloc[:, 0])
    
    corr_m, p_m = analyzer.get_correlations()
    
    with open(OUTPUT_MD, 'w', encoding='utf-8') as f:
        f.write("# Психометрический отчет по шкале MIS (30 пунктов)\n\n")
        f.write(f"**N = {psy_full['n']}**\n\n")
        
        f.write("## 1. Надежность полной шкалы\n\n")
        f.write(f"- **Общая Альфа Кронбаха (MIS Total): {psy_full['alpha']:.3f}**\n")
        f.write("\n### Надежность субшкал (блоков):\n\n")
        f.write("| Субшкала | Альфа |\n|---|---|\n")
        for bname, alpha in psy_full['block_alphas'].items():
            f.write(f"| {BLOCK_NAMES_RU[bname]} | {alpha:.3f} |\n")
        f.write("\n")
        
        f.write("## 2. Факторная структура (PCA)\n\n")
        f.write("### Собственные числа (Eigenvalues):\n")
        evs = factors_full['eigenvalues'][:10]
        f.write(", ".join([f"{v:.2f}" for v in evs]) + " ...\n\n")
        
        f.write("### Дисперсия (PCA):\n\n")
        var_exp = factors_full['var_explained_pca']
        f.write("| Показатель | F1 | F2 | F3 | F4 | F5 |\n|---|---|---|---|---|---|\n")
        f.write("| Доля дисперсии | " + " | ".join([f"{v:.1%}" for v in var_exp[:5]]) + " |\n")
        f.write("| Накопленная дисп. | " + " | ".join([f"{np.sum(var_exp[:i+1]):.1%}" for i in range(5)]) + " |\n\n")
        
        f.write("### Нагрузки факторов (PCA):\n\n")
        loadings_full = factors_full['loadings']
        # Sort by Factor 1 descending
        loadings_full_sorted = loadings_full.sort_values(by='Factor 1', ascending=False)
        
        f.write("| Пункт | F1 | F2 | F3 | F4 | F5 |\n|---|---|---|---|---|---|\n")
        for idx, row in loadings_full_sorted.iterrows():
            f.write(f"| {idx} | " + " | ".join([f"{v:.3f}" for v in row]) + " |\n")
        f.write("\n")
        
        f.write("---")
        f.write("\n\n# Краткая версия шкалы (MIS-5)\n\n")
        f.write("На основе анализа факторных нагрузок и unidimensional-структуры была выделена краткая версия шкалы из 5 пунктов.\n\n")
        
        f.write("## 1. Психометрические свойства MIS-5\n\n")
        f.write(f"- **Альфа Кронбаха (MIS-5): {psy_5['alpha']:.3f}**\n")
        f.write(f"- **Омега Макдональда (MIS-5): {omega_5:.3f}**\n")
        f.write(f"- **Доля дисперсии (PCA F1): {factors_5['var_explained_pca'][0]:.1%}**\n")
        
        if factors_5['kmo'] is not None:
            f.write(f"- **Мера адекватности выборки KMO: {factors_5['kmo']:.3f}**\n")
        if factors_5['bartlett_p'] is not None:
            f.write(f"- **Критерий сферичности Барлетта: p {format_p(factors_5['bartlett_p'])}**\n")
        
        f.write("\n### Собственные числа (MIS-5):\n")
        evs_5 = factors_5['eigenvalues']
        f.write(", ".join([f"{v:.2f}" for v in evs_5]) + "\n\n")
        
        # Generation of histogram
        try:
            plt.figure(figsize=(10, 6))
            sns.set_theme(style="whitegrid")
            ax = sns.histplot(analyzer.df['MIS_5'], kde=True, color='royalblue', bins=10)
            plt.title('Распределение баллов по шкале MIS-5', fontsize=15)
            plt.xlabel('Суммарный балл (5-25)', fontsize=12)
            plt.ylabel('Частота (N)', fontsize=12)
            
            # Save plot to the same directory as OUTPUT_MD
            img_path = os.path.join(os.path.dirname(OUTPUT_MD), 'mis_5_distribution.png')
            plt.savefig(img_path, dpi=300, bbox_inches='tight')
            plt.close()
            f.write("### Распределение баллов:\n\n")
            f.write(f"![Распределение MIS-5](mis_5_distribution.png)\n\n")
            
            # Regression plot: MIS-5 vs single_item
            plt.figure(figsize=(10, 6))
            r_val, _ = stats.pearsonr(analyzer.df['single_item'], analyzer.df['MIS_5'])
            r2 = r_val**2
            
            sns.regplot(x=analyzer.df['single_item'], y=analyzer.df['MIS_5'], 
                        scatter_kws={'alpha':0.5, 'color':'royalblue'}, 
                        line_kws={'color':'darkorange'})
            plt.title('Связь MIS-5 с однопунктовым критерием', fontsize=15)
            plt.xlabel('Оценка по single_item (1-10)', fontsize=12)
            plt.ylabel('Суммарный балл MIS-5', fontsize=12)
            
            # Add R^2 text
            plt.text(0.05, 0.95, f'$R^2 = {r2:.3f}$', transform=plt.gca().transAxes, 
                     fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            
            reg_img_path = os.path.join(os.path.dirname(OUTPUT_MD), 'mis_5_vs_single.png')
            plt.savefig(reg_img_path, dpi=300, bbox_inches='tight')
            plt.close()
            f.write("### Валидность (визуализация связи):\n\n")
            f.write(f"![Связь MIS-5 с single_item](mis_5_vs_single.png)\n\n")
            
        except Exception as plot_err:
            print(f"    Plotting error: {plot_err}")

        f.write("### Факторные нагрузки MIS-5 (EFA):\n\n")
        loadings_5 = factors_5['loadings']
        f.write("| Пункт | Нагрузка (Factor 1) |\n|---|---|\n")
        for idx, row in loadings_5.iterrows():
            f.write(f"| {idx} | {row.iloc[0]:.3f} |\n")
        f.write("\n")
        
        f.write("### Пункты MIS-5 и их характеристики:\n\n")
        f.write("| # | Текст пункта | Mean | SD | r-total | α-deleted |\n")
        f.write("|---|---|---|---|---|---|\n")
        for s in psy_5['item_stats']:
            f.write(f"| {s['id']} | {s['text']} | {s['mean']:.2f} | {s['sd']:.2f} | {s['rit']:.3f} | {s['alpha_del']:.3f} |\n")
        f.write("\n")
        
        f.write("## 2. Валидность и Сравнение\n\n")
        f.write("Сравнение корреляций полной (MIS-30) и краткой (MIS-5) версий с внешними критериями:\n\n")
        
        f.write("| Критерий | MIS-Total (30) | MIS-5 (Краткая) | Сохранение связи |\n")
        f.write("|---|---|---|---|\n")
        
        full_corrs = corr_m.loc['MIS_Total']
        short_corrs = corr_m.loc['MIS_5']
        
        for c in corr_m.columns:
            r_f = full_corrs[c]
            r_s = short_corrs[c]
            ratio = (r_s / r_f * 100) if abs(r_f) > 0.01 else 0
            label = OSD_MAPPING.get(c, c)
            f.write(f"| {label} | {r_f:.3f} | {r_s:.3f} | {ratio:.1f}% |\n")
        
        f.write("\n\n## 3. Таблица всех корреляций\n\n")
        f.write("| Шкала MIS | " + " | ".join([OSD_MAPPING.get(c, c) for c in corr_m.columns]) + " |\n")
        f.write("|---|" + "---|"*len(corr_m.columns) + "\n")
        for m_idx, m_row in corr_m.iterrows():
            row_str = f"| {BLOCK_NAMES_RU.get(m_idx, m_idx)} | "
            cells = []
            for c in corr_m.columns:
                r = m_row[c]
                p = p_m.loc[m_idx, c]
                star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                cells.append(f"{r:.3f}{star}")
            f.write(row_str + " | ".join(cells) + " |\n")
            
    print(f"Report generated: {OUTPUT_MD}")

if __name__ == "__main__":
    run_report()

    print(f"Report generated: {OUTPUT_MD}")

if __name__ == "__main__":
    run_report()
