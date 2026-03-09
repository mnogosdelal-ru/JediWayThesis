# Техническая документация

## Дата обновления: 2026-03-10

---

## Содержание

1. [Структура базы данных](#структура-базы-данных)
2. [Структура резервных копий](#структура-резервных-копий)
3. [Загрузка данных](#загрузка-данных)
4. [Python-скрипты для анализа](#python-скрипты-для-анализа)

---

## Структура базы данных

### Таблица: `respondents`

#### Основные поля

| Поле | Тип | Описание |
|------|-----|----------|
| `id` | CHAR(32) | UUID респондента (MD5 hash) |
| `code` | CHAR(6) | Уникальный код (3 буквы + 3 цифры) |
| `session_id` | VARCHAR(64) | ID сессии |
| `created_at` | TIMESTAMP | Дата создания записи |
| `updated_at` | TIMESTAMP | Дата последнего обновления |
| `status` | ENUM | Статус: 'in_progress', 'completed', 'abandoned' |
| `current_page` | INT | Текущая страница (0-10) |
| `completed_at` | TIMESTAMP | Дата завершения опроса |

#### Демография (Страница 1)

| Поле | Тип | Диапазон | Описание |
|------|-----|----------|----------|
| `age` | INT | 18-100 | Возраст |
| `gender` | ENUM | - | Пол: 'male', 'female', 'other', 'prefer_not' |
| `children_count` | INT | 0+ | Количество детей |
| `work_experience_years` | INT | 0+ | Стаж работы |
| `position` | ENUM | - | Позиция (owner, top_manager, ..., unemployed) |
| `industry` | ENUM | - | Отрасль (it, science, ..., unemployed) |
| `industry_other` | TEXT | - | Другая отрасль |
| `remote_days` | ENUM | - | Удалёнка: 'office', '1', '2', '3', '4', 'full_remote' |
| `mindset_technical_humanitarian` | INT | 1-5 | Склад ума (1=технарь, 5=гуманитарий) |
| `tool_preference` | INT | 1-5 | Предпочтение инструментов (1=электронные, 5=бумажные) |

#### Шкалы

**Страница 2: Срочное/важное (личная жизнь)**
- `personal_urgent_important` (INT, 1-5)

**Страница 3: Срочное/важное (работа)**
- `work_urgent_important` (INT, 1-5)
- `work_satisfaction` (INT, 1-7)

**Страница 4: MIJS**
- `mijs_items` (JSON) — ответы с ключами
- `mijs_urgency_score` (INT, 5-25)
- `mijs_agency_score` (INT, 7-35)
- `mijs_total` (INT, 12-60)

**Страница 5: SWLS**
- `swls_items` (JSON) — ответы с ключами
- `swls_total` (INT, 5-35)

**Страница 6: MBI**
- `mbi_exhaustion_items` (JSON) — ответы с ключами
- `mbi_exhaustion_score` (INT, 0-54)
- `mbi_cynicism_items` (JSON) — ответы с ключами
- `mbi_cynicism_score` (INT, 0-30)
- `mbi_efficacy_items` (JSON) — ответы с ключами
- `mbi_efficacy_score` (INT, 0-48)
- `mbi_total` (INT, 0-132)

**Страница 7: Прокрастинация**
- `procrastination_items` (JSON) — ответы с ключами
- `procrastination_total` (INT, 8-40)

**Страница 8: Практики**
- `practices_frequency` (JSON) — ответы с ключами (_freq)
- `practices_quality` (JSON) — ответы с ключами (_qual)
- `practices_freq_total` (INT, 20-120)
- `practices_quality_total` (INT, 20-80)

**Страница 9: Вакцины**
- `vaccines` (JSON) — ответы с ключами
- `vaccines_total` (INT, 0-15)

**Страница 10: Открытые вопросы**
- `open_most_useful_practice` (TEXT)
- `open_other_practices` (TEXT)

#### Контроль качества

| Поле | Тип | Описание |
|------|-----|----------|
| `attention_check_passed` | BOOLEAN | Пройдена ли проверка (практика 16) |
| `attention_check_freq_answer` | INT | Ответ на частоту (должен быть 6) |
| `attention_check_quality_answer` | INT | Ответ на качество (должен быть 4) |

#### Технические поля

| Поле | Тип | Описание |
|------|-----|----------|
| `ip_address` | VARCHAR(45) | IP адрес |
| `user_agent` | TEXT | User agent браузера |
| `time_spent_seconds` | INT | Время прохождения (секунды) |

#### Индексы

```sql
INDEX idx_status (status)
INDEX idx_completed_at (completed_at)
INDEX idx_code (code)
INDEX idx_mijs_total (mijs_total)
INDEX idx_mbi_exhaustion (mbi_exhaustion_score)
INDEX idx_mbi_cynicism (mbi_cynicism_score)
INDEX idx_mbi_efficacy (mbi_efficacy_score)
INDEX idx_swls_total (swls_total)
INDEX idx_procrastination (procrastination_total)
INDEX idx_practices_freq (practices_freq_total)
INDEX idx_practices_quality (practices_quality_total)
INDEX idx_vaccines (vaccines_total)
INDEX idx_personal_urgent (personal_urgent_important)
INDEX idx_work_urgent (work_urgent_important)
INDEX idx_work_satisfaction (work_satisfaction)
INDEX idx_age (age)
INDEX idx_children_count (children_count)
INDEX idx_remote_days (remote_days)
INDEX idx_status_completed (status, completed_at)
INDEX idx_mindset (mindset_technical_humanitarian)
```

---

## Структура резервных копий

### Расположение

**Папка:** `responses/{respondent_id}.json`

### Структура JSON файла

```json
{
  "respondent_id": "7cec08e7e860a00d82c3d550e3c8b2e7",
  "timestamp": "2026-03-10 12:34:56",
  "answers": {
    "open_most_useful_practice": "...",
    "open_other_practices": "...",
    "age": 35,
    "gender": "male",
    "children_count": 2,
    "position": "middle_manager",
    "work_experience_years": 12,
    "industry": "it",
    "remote_days": "3",
    "mindset_technical_humanitarian": 3,
    "tool_preference": 2,
    "personal_urgent_important": 4,
    "work_urgent_important": 4,
    "work_satisfaction": 5,
    
    "mijs_items": "{\"mijs_q1_...\": 1, ...}",
    "mijs_total": 38,
    "swls_items": "{\"swls_q1_...\": 5, ...}",
    "swls_total": 26,
    "mbi_exhaustion_items": "{...}",
    "mbi_exhaustion_score": 28,
    "mbi_cynicism_items": "{...}",
    "mbi_cynicism_score": 12,
    "mbi_efficacy_items": "{...}",
    "mbi_efficacy_score": 35,
    "mbi_total": 75,
    "procrastination_items": "{...}",
    "procrastination_total": 22,
    "practices_frequency": "{...}",
    "practices_quality": "{...}",
    "practices_freq_total": 85,
    "practices_quality_total": 65,
    "vaccines": "{...}",
    "vaccines_total": 8,
    
    "attention_check_passed": 1,
    "attention_check_freq_answer": 6,
    "attention_check_quality_answer": 4,
    "time_spent_seconds": 915
  }
}
```

### Ключи переменных

**Формат:** `{префикс}_q{номер}_{первые 4 слова транслитом}`

**Примеры:**
- `mijs_q1_ya_chuvstvuyu_chto_moi_resursy`
- `swls_q1_v_osnovnom_moya_zhizn_blizka_k_idealu`
- `mbi_exh_q1_k_kontsu_rabochey_nedeli_ya_chuvstvuyu_sebya_emotsionalno_opustoshennym`
- `proc_q1_ya_chasto_zamechayu_chto_zaranee_vypolnyayu_zadachi`
- `prac_q1_lezha_v_posteli_ne_polzovatsya_ustroistvami_freq`
- `prac_q1_lezha_v_posteli_ne_polzovatsya_ustroistvami_qual`
- `vacc_q1_udalit_prilozheniya_sotsialnykh_setey`

**Полный список ключей:** См. `config/variable_keys.php`

---

## Загрузка данных

### Вариант 1: Из базы данных MySQL

```python
import mysql.connector
import pandas as pd
import json

# Подключение к БД
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='',
    database='jedi_survey'
)

# Загрузка данных
query = """
SELECT 
    id, code, status, created_at, completed_at,
    age, gender, children_count, position, industry, remote_days,
    mindset_technical_humanitarian, tool_preference,
    personal_urgent_important, work_urgent_important, work_satisfaction,
    mijs_items, mijs_total,
    swls_items, swls_total,
    mbi_exhaustion_items, mbi_exhaustion_score,
    mbi_cynicism_items, mbi_cynicism_score,
    mbi_efficacy_items, mbi_efficacy_score,
    procrastination_items, procrastination_total,
    practices_frequency, practices_quality,
    vaccines, vaccines_total,
    time_spent_seconds
FROM respondents
WHERE status = 'completed'
  AND attention_check_passed = TRUE
ORDER BY created_at
"""

df = pd.read_sql(query, conn)

# Распарсить JSON поля
json_columns = [
    'mijs_items', 'swls_items',
    'mbi_exhaustion_items', 'mbi_cynicism_items', 'mbi_efficacy_items',
    'procrastination_items',
    'practices_frequency', 'practices_quality',
    'vaccines'
]

for col in json_columns:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: json.loads(x) if pd.notna(x) else {})

conn.close()
```

### Вариант 2: Из резервных копий (JSON файлы)

```python
import json
import pandas as pd
from pathlib import Path

# Путь к папке с бэкапами
backup_dir = Path('responses')

# Загрузка всех файлов
data = []
for json_file in backup_dir.glob('*.json'):
    with open(json_file, 'r', encoding='utf-8') as f:
        backup = json.load(f)
        data.append(backup)

# Преобразование в DataFrame
df = pd.DataFrame(data)

# Распарсить вложенные JSON
if 'answers' in df.columns:
    answers_df = pd.json_normalize(df['answers'])
    df = pd.concat([df.drop('answers', axis=1), answers_df], axis=1)

# Распарсить JSON строки внутри answers
json_columns = [
    'mijs_items', 'swls_items',
    'mbi_exhaustion_items', 'mbi_cynicism_items', 'mbi_efficacy_items',
    'procrastination_items',
    'practices_frequency', 'practices_quality',
    'vaccines'
]

for col in json_columns:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: json.loads(x) if pd.notna(x) else {})
```

---

## Python-скрипты для анализа

### 1. `analysis/config.py`

Конфигурация для подключения к БД и путей к файлам.

```python
# База данных
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'jedi_survey'
}

# Пути
BASE_DIR = Path(__file__).parent.parent
RESPONSES_DIR = BASE_DIR / 'responses'
EXPORT_DIR = BASE_DIR / 'analysis' / 'export'
```

### 2. `analysis/01_data_preprocessing.py`

Предобработка данных: загрузка, фильтрация, расчёт шкал.

**Использование:**
```bash
cd analysis
python 01_data_preprocessing.py
```

**Выходные данные:**
- `analysis/export/clean_data.csv` — очищенные данные
- `analysis/export/analytic_data.csv` — данные для анализа

### 3. `analysis/09_gender_children_anova.py`

ANOVA для анализа гендерных различий и различий по количеству детей.

**Использование:**
```bash
cd analysis
python 09_gender_children_anova.py
```


## Контакты

Исследователь: maxim.dorofeev@mnogosdelal.ru
