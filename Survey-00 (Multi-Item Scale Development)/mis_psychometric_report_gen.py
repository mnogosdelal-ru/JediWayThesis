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
SHORT_SCALE_SIZE = 6
#MIS_CUSTOM_COLS = ['MIS_q_28', 'MIS_q_24', 'MIS_q_16', 'MIS_q_26', 'MIS_q_13']
MIS_CUSTOM_COLS = ['MIS_q_15', 'MIS_q_16', 'MIS_q_24', 'MIS_q_26', 'MIS_q_28']
MAX_ITTERATIONS = 1000
TARGET_FOR_R_SCALE = 'single_item'

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
        self.auto_select_mis_short()
        self.auto_select_mis_5r()
        
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
        
        if 'MIS_CUSTOM_COLS' in globals() and MIS_CUSTOM_COLS:
            self.df['MIS_custom'] = self.df[MIS_CUSTOM_COLS].sum(axis=1)
            
        self.df = self.df.dropna(subset=['MIS_Total', 'single_item'])

    def auto_select_mis_short(self, n=None):
        if n is None:
            n = SHORT_SCALE_SIZE
        print(f"Automatically selecting top {n} items for MIS-short...")
        cols = [f'MIS_q_{i}' for i in range(1, 31)]
        data = self.df[cols]
        
        # Use FactorAnalyzer to get loadings on the first factor
        try:
            fa = FactorAnalyzer(n_factors=1, rotation=None)
            fa.fit(data)
            loadings = fa.loadings_[:, 0]
        except Exception as e:
            print(f"    Auto-selection EFA Error: {e}. Falling back to PCA.")
            pca = PCA(n_components=1)
            # Standardize before PCA
            std_data = (data - data.mean()) / data.std()
            pca.fit(std_data)
            loadings = pca.components_[0]
            
        loadings_series = pd.Series(np.abs(loadings), index=cols)
        self.mis_short_cols = loadings_series.sort_values(ascending=False).head(n).index.tolist()
        
        print(f"    Selected items: {self.mis_short_cols}")
        self.df['MIS_short'] = self.df[self.mis_short_cols].sum(axis=1)

    def auto_select_mis_5r(self, n=None):
        if n is None:
            n = SHORT_SCALE_SIZE
        print(f"Automatically selecting top {n} items for MIS-{n}R (Correlation with {TARGET_FOR_R_SCALE})...")
        cols = [f'MIS_q_{i}' for i in range(1, 31)]
        corrs = self.df[cols].corrwith(self.df[TARGET_FOR_R_SCALE]).abs()
        self.mis_5r_cols = corrs.sort_values(ascending=False).head(n).index.tolist()
        print(f"    Selected items (Criterion): {self.mis_5r_cols}")
        self.df['MIS_5R'] = self.df[self.mis_5r_cols].sum(axis=1)

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

    def get_factor_structure(self, cols=None, n_factors=None, rotation='varimax'):
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
            # If columns == 30, use 5 factors. For short scales (like MIS-4/5/6), use 1 factor.
            n_factors = 5 if len(cols) == 30 else 1
            
        # Ensure n_factors doesn't exceed number of items
        n_factors = min(n_factors, len(cols))
            
        try:
            fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation if n_factors > 1 else None)
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

    def get_inter_item_correlations(self, cols):
        inter_corr = self.df[cols].corr()
        # Average inter-item correlation (excluding diagonal)
        mask = np.ones(inter_corr.shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        avg_corr = inter_corr.where(mask).stack().mean()
        return inter_corr, avg_corr

    def bootstrap_selection_stability(self, n_iterations=100, n_top=None, method='factor'):
        if n_top is None:
            n_top = SHORT_SCALE_SIZE
            
        print(f"Running Bootstrap Stability Analysis ({method}, {n_iterations} iterations)...")
        cols = [f'MIS_q_{i}' for i in range(1, 31)]
        selection_counts = {c: 0 for c in cols}
        
        for i in range(n_iterations):
            if (i + 1) % 100 == 0:
                print(f"    Iteration {i + 1}/{n_iterations}...")
            
            # Resample with 80% fraction as per user's modification
            sample = self.df[cols + ([TARGET_FOR_R_SCALE] if method == 'criterion' else [])].sample(frac=0.8, replace=False)
            
            if method == 'factor':
                try:
                    # FA 1-factor unrotated
                    fa = FactorAnalyzer(n_factors=1, rotation=None)
                    fa.fit(sample[cols])
                    loadings = np.abs(fa.loadings_[:, 0])
                except:
                    # Fallback to PCA
                    pca = PCA(n_components=1)
                    std_sample = (sample[cols] - sample[cols].mean()) / sample[cols].std()
                    pca.fit(std_sample)
                    loadings = np.abs(pca.components_[0])
                metric_series = pd.Series(loadings, index=cols)
            else: # criterion
                corrs = sample[cols].corrwith(sample[TARGET_FOR_R_SCALE]).abs()
                metric_series = corrs
                
            top_items = metric_series.sort_values(ascending=False).head(n_top).index
            for item in top_items:
                selection_counts[item] += 1
                
        # Convert to percentage
        stability = pd.Series(selection_counts) / n_iterations
        return stability.sort_values(ascending=False)

    def get_correlations(self):
        mis_scales = list(MIS_BLOCKS.keys()) + ['MIS_Total', 'MIS_short']
        if 'MIS_5R' in self.df.columns:
            mis_scales.append('MIS_5R')
        if 'MIS_custom' in self.df.columns:
            mis_scales.append('MIS_custom')
            
        osd_scales = list(OSD_MAPPING.keys())
        criteria = osd_scales + ['single_item']
        
        corr_matrix = pd.DataFrame(index=mis_scales, columns=criteria)
        p_matrix = pd.DataFrame(index=mis_scales, columns=criteria)
        
        for m in mis_scales:
            for c in criteria:
                if c in self.df.columns and m in self.df.columns:
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
    factors_unrotated = analyzer.get_factor_structure(rotation=None)
    factors_full = analyzer.get_factor_structure(rotation='varimax')
    
    # Bootstrap Stability Analysis
    stability_data_short = analyzer.bootstrap_selection_stability(n_iterations=MAX_ITTERATIONS, method='factor')
    stability_data_5r = analyzer.bootstrap_selection_stability(n_iterations=MAX_ITTERATIONS, method='criterion')
    
    corr_m, p_m = analyzer.get_correlations()
    
    def write_scale_analysis(f, analyzer, title, scale_id, cols, stability_series=None):
        print(f"Computing metrics for {title}...")
        psy = analyzer.get_psychometrics(cols)
        factors = analyzer.get_factor_structure(cols)
        omega = mcdonald_omega(factors['loadings'].iloc[:, 0])
        inter_corr_mtx, avg_inter_corr = analyzer.get_inter_item_correlations(cols)
        
        f.write(f"\n\n# {title}\n\n")
        f.write(f"Размер шкалы: {len(cols)} пунктов.\n\n")
        
        if scale_id == 'short':
             f.write("> [!TIP]\n")
             f.write(f"> Для выбора пунктов использовалось **однофакторное невращаемое решение**. Порядок пунктов в таблице полной шкалы может отличаться, так как там применено вращение Varimax для 5 факторов, перераспределяющее нагрузку. Для краткой версии мы выбираем пункты, максимально нагруженные на общий 'General Factor'.\n\n")

        f.write(f"## 1. Психометрические свойства {scale_id}\n\n")
        f.write(f"- **Альфа Кронбаха: {psy['alpha']:.3f}**\n")
        f.write(f"- **Омега Макдональда: {omega:.3f}**\n")
        f.write(f"- **Средняя межпунктовая корреляция: {avg_inter_corr:.3f}**\n")
        f.write(f"- **Доля дисперсии (PCA F1): {factors['var_explained_pca'][0]:.1%}**\n")
        
        if factors['kmo'] is not None:
            f.write(f"- **Мера адекватности выборки KMO: {factors['kmo']:.3f}**\n")
        if factors['bartlett_p'] is not None:
            f.write(f"- **Критерий сферичности Барлетта: p {format_p(factors['bartlett_p'])}**\n")
        
        f.write(f"\n### Собственные числа ({scale_id}):\n")
        evs = factors['eigenvalues']
        f.write(", ".join([f"{v:.2f}" for v in evs]) + "\n\n")
        
        # Plots
        try:
            plt.figure(figsize=(10, 6))
            sns.set_theme(style="whitegrid")
            # Use columns sum from dataframe directly if possible
            scale_sum_col = 'MIS_short' if scale_id == 'short' else ('MIS_5R' if scale_id == '5r' else 'MIS_custom')
            ax = sns.histplot(analyzer.df[scale_sum_col], kde=False, color='royalblue', bins=5)
            plt.title(f'Распределение баллов по шкале {title}', fontsize=15)
            plt.xlabel(f'Суммарный балл ({len(cols)}-{len(cols)*5})', fontsize=12)
            plt.ylabel('Частота (N)', fontsize=12)
            
            img_name = f'mis_{scale_id}_distribution.png'
            img_path = os.path.join(os.path.dirname(OUTPUT_MD), img_name)
            plt.savefig(img_path, dpi=300, bbox_inches='tight')
            plt.close()
            f.write("### Распределение баллов:\n\n")
            f.write(f"![Распределение {scale_id}]({img_name})\n\n")
            
            # Regression
            plt.figure(figsize=(10, 6))
            r_val, _ = stats.pearsonr(analyzer.df['single_item'], analyzer.df[scale_sum_col])
            r2 = r_val**2
            
            sns.regplot(x=analyzer.df['single_item'], y=analyzer.df[scale_sum_col], 
                        scatter_kws={'alpha':0.5, 'color':'royalblue'}, 
                        line_kws={'color':'darkorange'})
            plt.title(f'Связь {title} с однопунктовым критерием', fontsize=15)
            plt.xlabel('Оценка по single_item (1-10)', fontsize=12)
            plt.ylabel(f'Суммарный балл {scale_id}', fontsize=12)
            
            plt.text(0.05, 0.95, f'$R^2 = {r2:.3f}$', transform=plt.gca().transAxes, 
                     fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            
            reg_img_name = f'mis_{scale_id}_vs_single.png'
            reg_img_path = os.path.join(os.path.dirname(OUTPUT_MD), reg_img_name)
            plt.savefig(reg_img_path, dpi=300, bbox_inches='tight')
            plt.close()
            f.write("### Валидность (визуализация связи):\n\n")
            f.write(f"![Связь {scale_id} с single_item]({reg_img_name})\n\n")
            
            if scale_id in ['short', '5r'] and stability_series is not None:
                # Stability plot
                plt.figure(figsize=(10, 6))
                top_15_stability = stability_series.head(15)
                # Ensure we represent IDs correctly for x-axis
                x_labels = [str(int(idx.split('_')[-1])) for idx in top_15_stability.index]
                
                ax = sns.barplot(x=x_labels, y=top_15_stability.values, color='seagreen' if scale_id == 'short' else 'darkcyan')

                # Add text labels on top of bars
                for p in ax.patches:
                    ax.annotate(f'{p.get_height():.1%}', 
                                (p.get_x() + p.get_width() / 2., p.get_height()), 
                                ha = 'center', va = 'center', 
                                xytext = (0, 9), 
                                textcoords = 'offset points', 
                                fontsize=9)

                method_name = "Bootstrap" if scale_id == 'short' else "Criterion Stablity"
                plt.title(f'Устойчивость выбора пунктов MIS-{SHORT_SCALE_SIZE} ({method_name})', fontsize=15)
                plt.xlabel('ID пункта (MIS_q_x)', fontsize=12)
                plt.ylabel(f'Частота попадания в ТОП-{SHORT_SCALE_SIZE}', fontsize=12)
                plt.ylim(0, 1.1)
                
                stab_img_name = f'mis_{scale_id}_stability.png'
                stab_img_path = os.path.join(os.path.dirname(OUTPUT_MD), stab_img_name)
                plt.savefig(stab_img_path, dpi=300, bbox_inches='tight')
                plt.close()
                f.write("### Анализ устойчивости выбора (Bootstrap):\n\n")
                f.write(f"![Устойчивость {scale_id}]({stab_img_name})\n\n")
                
                f.write("> [!NOTE]\n")
                if scale_id == 'short':
                    f.write(f"> График показывает, как часто каждый пункт попадал в ТОП при {MAX_ITTERATIONS} повторных расчетах на случайных подвыборках (методом Факторного анализа). ")
                else:
                    f.write(f"> График показывает, как часто каждый пункт попадал в ТОП при {MAX_ITTERATIONS} повторных расчетах на случайных подвыборках (по корреляции с критерием {TARGET_FOR_R_SCALE}). ")
                f.write("Пункты с частотой выше 0.8-0.9 считаются абсолютно стабильными для этой шкалы.\n\n")
                
                f.write("| Пункт | Частота попадания в ТОП |\n|---|---|\n")
                for idx, val in top_15_stability.items():
                    bold = "**" if idx in cols else ""
                    f.write(f"| {bold}{idx}{bold} | {val:.1%} |\n")
                f.write("\n")
                
        except Exception as plot_err:
            print(f"    Plotting error for {scale_id}: {plot_err}")

        f.write(f"### Факторные нагрузки {scale_id} (EFA):\n\n")
        loadings = factors['loadings']
        f.write("| Пункт | Нагрузка (Factor 1) |\n|---|---|\n")
        for idx, row in loadings.iterrows():
            f.write(f"| {idx} | {row.iloc[0]:.3f} |\n")
        f.write("\n")
        
        f.write(f"### Матрица межпунктовых корреляций {scale_id}:\n\n")
        f.write("| | " + " | ".join(inter_corr_mtx.columns) + " |\n")
        f.write("|---|" + "---|"*len(inter_corr_mtx.columns) + "\n")
        for idx, row in inter_corr_mtx.iterrows():
            f.write(f"| **{idx}** | " + " | ".join([f"{v:.3f}" for v in row]) + " |\n")
        f.write("\n")
        
        f.write(f"### Пункты {scale_id} и их характеристики:\n\n")
        f.write("| # | Текст пункта | Mean | SD | r-total | α-deleted |\n")
        f.write("|---|---|---|---|---|---|\n")
        for s in psy['item_stats']:
            f.write(f"| {s['id']} | {s['text']} | {s['mean']:.2f} | {s['sd']:.2f} | {s['rit']:.3f} | {s['alpha_del']:.3f} |\n")
        f.write("\n")
        
        f.write("## 2. Валидность и Сравнение\n\n")
        f.write(f"Сравнение корреляций полной (MIS-30) и исследуемой ({scale_id}) версий с внешними критериями:\n\n")
        f.write(f"| Критерий | MIS-Total (30) | {scale_id} | Сохранение связи |\n")
        f.write("|---|---|---|---|\n")
        
        full_corrs = corr_m.loc['MIS_Total']
        test_corrs = corr_m.loc[scale_sum_col]
        
        for c in corr_m.columns:
            if c not in OSD_MAPPING and c != 'single_item': continue
            r_f = full_corrs[c]
            r_s = test_corrs[c]
            ratio = (r_s / r_f * 100) if abs(r_f) > 0.01 else 0
            label = OSD_MAPPING.get(c, c)
            f.write(f"| {label} | {r_f:.3f} | {r_s:.3f} | {ratio:.1f}% |\n")
    
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
        
        f.write("### Накопленная дисперсия (PCA):\n\n")
        var_exp = factors_full['var_explained_pca']
        f.write("| Показатель | F1 | F2 | F3 | F4 | F5 |\n|---|---|---|---|---|---|\n")
        f.write("| Доля дисперсии | " + " | ".join([f"{v:.1%}" for v in var_exp[:5]]) + " |\n")
        f.write("| Накопленная дисп. | " + " | ".join([f"{np.sum(var_exp[:i+1]):.1%}" for i in range(5)]) + " |\n\n")
        
        f.write("### Нагрузки факторов (без вращения):\n\n")
        f.write("> [!NOTE]\n")
        f.write("> В этой таблице представлены нагрузки на первые 5 факторов до применения вращения. ")
        f.write(f"Первый фактор (F1) здесь отражает общее ядро шкалы (General Factor), по которому отбирались пункты для MIS-{SHORT_SCALE_SIZE}.\n\n")
        
        loadings_unrot = factors_unrotated['loadings']
        loadings_unrot_sorted = loadings_unrot.sort_values(by='Factor 1', ascending=False)
        
        f.write("| Пункт | F1 | F2 | F3 | F4 | F5 |\n|---|---|---|---|---|---|\n")
        for idx, row in loadings_unrot_sorted.iterrows():
            f.write(f"| {idx} | " + " | ".join([f"{v:.3f}" for v in row]) + " |\n")
        f.write("\n")

        f.write("### Нагрузки факторов (EFA Varimax):\n\n")
        f.write("> [!NOTE]\n")
        f.write("> Использовано вращение **Varimax** для лучшей интерпретируемости пятифакторной структуры. ")
        f.write("Нагрузки перераспределены для выделения независимых субшкал (блоков).\n\n")
        loadings_full = factors_full['loadings']
        # Sort by Factor 1 descending
        loadings_full_sorted = loadings_full.sort_values(by='Factor 1', ascending=False)
        
        f.write("| Пункт | F1 | F2 | F3 | F4 | F5 |\n|---|---|---|---|---|---|\n")
        for idx, row in loadings_full_sorted.iterrows():
            f.write(f"| {idx} | " + " | ".join([f"{v:.3f}" for v in row]) + " |\n")
        f.write("\n")
        
        f.write("---")
        
        # Section for Short Scale
        write_scale_analysis(f, analyzer, f"Краткая версия шкалы (MIS-{SHORT_SCALE_SIZE})", "short", analyzer.mis_short_cols, stability_data_short)
        
        f.write("\n---\n")
        
        # Section for MIS-5R
        write_scale_analysis(f, analyzer, f"Краткая версия шкалы (MIS-{SHORT_SCALE_SIZE}R)", "5r", analyzer.mis_5r_cols, stability_data_5r)

        f.write("\n---\n")
        
        # Section for Custom Scale
        if hasattr(analyzer, 'mis_custom_cols') or ('MIS_CUSTOM_COLS' in globals() and MIS_CUSTOM_COLS):
             custom_cols = MIS_CUSTOM_COLS
             write_scale_analysis(f, analyzer, "Произвольная версия шкалы (MIS_CUSTOM)", "custom", custom_cols)

        f.write("\n\n## 3. Таблица всех корреляций (Все версии)\n\n")
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
