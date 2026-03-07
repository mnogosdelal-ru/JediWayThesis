"""
Конфигурация для анализа данных исследования джедайских практик

Установка зависимостей:
pip install pandas numpy scipy statsmodels pingouin matplotlib seaborn python-dotenv mysql-connector-python

Использование:
from config import *
df = load_data()
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Загрузить переменные окружения из .env файла
load_dotenv()

# ============================================================================
# НАСТРОЙКИ БАЗЫ ДАННЫХ
# ============================================================================

DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_NAME = os.getenv('DB_NAME', 'jedi_survey')
DB_USER = os.getenv('DB_USER', 'root')
DB_PASS = os.getenv('DB_PASS', '')
DB_PORT = os.getenv('DB_PORT', '3306')

# ============================================================================
# ПУТИ К ФАЙЛАМ И ПАПКАМ
# ============================================================================

# Базовая директория проекта
BASE_DIR = Path(__file__).parent.parent

# Директории для результатов
EXPORT_DIR = BASE_DIR / 'analysis' / 'export'
TABLES_DIR = BASE_DIR / 'analysis' / 'tables'
FIGURES_DIR = BASE_DIR / 'analysis' / 'figures'

# Создать директории если не существуют
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# НАСТРОЙКИ АНАЛИЗА
# ============================================================================

# Уровень значимости
ALPHA = 0.05

# Поправка Бонферрони для множественных сравнений (21 практика)
BONFERRONI_ALPHA = ALPHA / 21  # ≈ 0.0024

# Минимальный размер выборки для включения в анализ
MIN_SAMPLE_SIZE = 50

# Максимальное время заполнения (сек) для фильтрации невалидных ответов
MAX_TIME_SECONDS = 60  # 1 минута

# ============================================================================
# СПИСОК ШКАЛ И ИХ ПАРАМЕТРЫ
# ============================================================================

SCALES = {
    'mijs': {
        'items': 12,
        'range': (12, 60),
        'inverted': [6, 7, 8, 9, 10, 11, 12],  # items для инверсии
        'subscale_urgency': [1, 2, 3, 4, 5],
        'subscale_agency': [6, 7, 8, 9, 10, 11, 12]
    },
    'mbi': {
        'items': 22,
        'range': (22, 154),
        'subscale_exhaustion': [1, 2, 3, 6, 8, 13, 14, 16, 20],  # 9 items напрямую
        'subscale_cynicism': [5, 10, 11, 15, 22],  # 5 items напрямую
        'subscale_efficacy': [4, 7, 9, 12, 17, 18, 19, 21],  # 8 items все инвертируются
        'inverted': [4, 7, 9, 12, 17, 18, 19, 21]
    },
    'swls': {
        'items': 5,
        'range': (5, 35),
        'inverted': []
    },
    'procrastination': {
        'items': 8,
        'range': (8, 32),
        'inverted': [1]  # только первый item инвертируется
    },
    'practices_freq': {
        'items': 21,
        'range': (21, 126),
        'weekly_planning': [9, 10, 11]  # практики недельного планирования
    },
    'vaccines': {
        'items': 5,
        'range': (0, 15)
    }
}

# ============================================================================
# ГИПОТЕЗЫ ДЛЯ ПРОВЕРКИ
# ============================================================================

HYPOTHESES = {
    'H1': {'type': 'correlation', 'variables': ['personal_urgent_important', 'mijs_total'], 'expected': 'negative'},
    'H2': {'type': 'correlation', 'variables': ['mijs_total', 'mbi_exhaustion'], 'expected': 'positive'},
    'H3': {'type': 'cfa', 'variables': 'mijs', 'expected': 'two_factor'},
    'H4': {'type': 'regression', 'predictors': 'practices', 'outcome': 'mijs_total', 'expected_r2': 0.20},
    'H5': {'type': 'regression', 'predictors': 'weekly_planning', 'outcome': 'mijs_total', 'expected_beta': -0.15},
    'H6': {'type': 'correlation', 'variables': ['practices_freq_total', 'mbi_exhaustion'], 'expected': 'negative'},
    'H7': {'type': 'correlation', 'variables': ['practices_freq_total', 'swls_total'], 'expected': 'positive'},
    'H8': {'type': 'correlation', 'variables': ['practices_freq_total', 'procrastination_total'], 'expected': 'negative'},
    'H9': {'type': 'regression', 'predictors': 'practices', 'outcome': 'mbi_exhaustion', 'expected_r2': 0.20},
    'H10': {'type': 'regression', 'predictors': 'practices', 'outcome': 'swls_total', 'expected_r2': 0.20},
    'H11': {'type': 'moderation', 'predictor': 'practices', 'outcome': 'urgent_important', 'moderator': 'domain'},
    'H12': {'type': 'ancova', 'predictor': 'practices', 'outcome': 'mijs_total', 'covariate': 'social_desirability'},
    'H13': {'type': 'mann_whitney', 'variable': 'planning_index', 'groups': 'gender', 'expected': 'female > male'},
    'H14': {'type': 'ttest', 'variable': 'mbi_exhaustion', 'groups': 'gender', 'expected': 'female > male'},
    'H15': {'type': 'moderation', 'predictor': 'planning', 'outcome': 'mijs_total', 'moderator': 'gender'},
    'H16': {'type': 'ttest', 'variable': 'work_urgent_important', 'groups': 'remote_work', 'expected': 'remote > office'},
    'H17': {'type': 'ttest', 'variable': 'mijs_total', 'groups': 'remote_work', 'expected': 'remote < office'},
    'H18': {'type': 'correlation', 'variables': ['tool_preference', 'mijs_total'], 'expected': 'null'},
    'H19': {'type': 'mann_whitney', 'variable': 'practices', 'groups': 'gender', 'correction': 'bonferroni'},
    'H20': {'type': 'correlation', 'variables': ['mbi_exhaustion', 'practices_freq_total'], 'expected': 'negative'},
    'H21': {'type': 'correlation', 'variables': ['practices_freq_total', 'work_urgent_important'], 'expected': 'positive'},
    'H22': {'type': 'correlation', 'variables': ['work_urgent_important', 'personal_urgent_important'], 'expected': 'positive'},
    'H23': {'type': 'correlation', 'variables': ['personal_urgent_important', 'mbi_exhaustion'], 'expected': 'negative'},
    'H24': {'type': 'mediation', 'chain': ['mbi_exhaustion', 'practices_freq_total', 'work_urgent_important', 'personal_urgent_important']}
}

# ============================================================================
# СТИЛИ ДЛЯ ВИЗУАЛИЗАЦИИ
# ============================================================================

PLOT_STYLE = {
    'figure.figsize': (10, 8),
    'figure.dpi': 300,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
}

# Цветовая палитра
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#28A745',
    'warning': '#FFC107',
    'danger': '#DC3545',
    'info': '#17A2B8',
}

# ============================================================================
# ФУНКЦИИ ДЛЯ ЗАГРУЗКИ ДАННЫХ
# ============================================================================

def get_db_connection_string():
    """Вернуть строку подключения к MySQL"""
    return f"mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def get_db_params():
    """Вернуть параметры для подключения"""
    return {
        'host': DB_HOST,
        'database': DB_NAME,
        'user': DB_USER,
        'password': DB_PASS,
        'port': DB_PORT
    }
