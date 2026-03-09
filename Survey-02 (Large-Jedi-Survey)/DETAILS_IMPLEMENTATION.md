# Детали реализации

**Дата обновления:** 2026-03-10

Этот документ описывает техническую реализацию веб-приложения для исследования джедайских практик.

---

## Содержание

1. [Архитектура приложения](#архитектура-приложения)
2. [Структура проекта](#структура-проекта)
3. [База данных](#база-данных)
4. [Резервное копирование](#резервное-копирование)
5. [Логирование](#логирование)

---

## Архитектура приложения

### Компоненты

| Компонент | Файлы | Назначение |
|-----------|-------|------------|
| **Frontend** | `templates/page_*.php` | 11 страниц опросника (0-10) |
| **Backend** | `public/*.php` | Обработка запросов, сохранение данных |
| **База данных** | MySQL | Хранение ответов респондентов |
| **Бэкапы** | `responses/*.json` | Резервные копии ответов |

### Поток данных

```
┌──────────────┐
│  Браузер     │
│  (Клиент)    │
└──────┬───────┘
       │
       │ 1. GET /public/
       ▼
┌──────────────┐
│  index.php   │───> Отображение страницы опроса
└──────────────┘

       │
       │ 2. POST ответы
       ▼
┌──────────────┐
│  save.php    │───> Сохранение в БД
└──────────────┘

       │
       │ 3. Завершение (стр. 10)
       ▼
┌──────────────┐
│  index.php   │───> Сохранение бэкапа в responses/
│  (page 10)   │
└──────────────┘
```

---

## Структура проекта

```
Survey-02 (Large-Jedi-Survey)/
│
├── config/
│   ├── config.php              # Конфигурация (БД, логирование)
│   └── variable_keys.php       # Ключи переменных для всех шкал
│
├── public/
│   ├── index.php               # Главный маршрутизатор
│   ├── save.php                # AJAX сохранение ответов
│   ├── results.php             # Страница результатов
│   ├── backup.php              # Функция резервного копирования
│   ├── api.php                 # API для получения данных
│   └── styles.css              # Стили приложения
│
├── src/
│   ├── Database.php            # PDO wrapper для работы с БД
│   ├── Survey.php              # Логика опросника
│   └── Calculator.php          # Расчёт баллов шкал
│
├── templates/
│   ├── page_0.php              # Информированное согласие
│   ├── page_1.php              # Демография
│   ├── page_2.php              # Личная жизнь
│   ├── page_3.php              # Работа
│   ├── page_4.php              # MIJS
│   ├── page_5.php              # SWLS
│   ├── page_6.php              # MBI
│   ├── page_7.php              # Прокрастинация
│   ├── page_8.php              # Практики
│   ├── page_9.php              # Вакцины
│   ├── page_10.php             # Открытые вопросы + Спасибо
│   ├── header.php              # Шапка страниц
│   └── footer.php              # Подвал страниц
│
├── responses/                  # Резервные копии (JSON)
│   └── {respondent_id}.json
│
├── logs/                       # Логи приложения
│   └── app_YYYY-MM-DD.log
│
├── analysis/                   # Python-скрипты анализа
│   ├── config.py
│   ├── 01_data_preprocessing.py
│   └── 09_gender_children_anova.py
│
├── sql/
│   ├── create_database_full.sql    # Создание БД с нуля
│   └── 011_add_indexes_for_percentiles.sql  # Индексы
│
└── documentation/
    ├── TECHNICAL_DOCS.md       # Полная техническая документация
    ├── RESPONSES_BACKUP.md     # Описание системы бэкапов
    └── DETAILS_IMPLEMENTATION.md # Этот файл
```

---

## База данных

### Таблица `respondents`

**Файл миграции:** [`sql/create_database_full.sql`](sql/create_database_full.sql)

#### Группы полей

**Идентификаторы:**
- `id` (CHAR(32)) — UUID респондента
- `code` (CHAR(6)) — Публичный код (3 буквы + 3 цифры)
- `session_id` (VARCHAR(64)) — ID сессии PHP

**Статус:**
- `status` (ENUM) — 'in_progress', 'completed', 'abandoned'
- `current_page` (INT) — Текущая страница (0-10)
- `completed_at` (TIMESTAMP) — Дата завершения

**Демография:**
- `age`, `gender`, `children_count`
- `work_experience_years`, `position`, `industry`
- `remote_days`, `mindset_technical_humanitarian`, `tool_preference`

**Шкалы (баллы):**
- `mijs_total`, `mijs_urgency_score`, `mijs_agency_score`
- `swls_total`
- `mbi_exhaustion_score`, `mbi_cynicism_score`, `mbi_efficacy_score`, `mbi_total`
- `procrastination_total`
- `practices_freq_total`, `practices_quality_total`
- `vaccines_total`
- `personal_urgent_important`, `work_urgent_important`, `work_satisfaction`

**Шкалы (JSON с ключами):**
- `mijs_items`, `swls_items`
- `mbi_exhaustion_items`, `mbi_cynicism_items`, `mbi_efficacy_items`
- `procrastination_items`
- `practices_frequency`, `practices_quality`
- `vaccines`

**Открытые вопросы:**
- `open_most_useful_practice` (TEXT)
- `open_other_practices` (TEXT)

**Контроль качества:**
- `attention_check_passed` (BOOLEAN)
- `attention_check_freq_answer`, `attention_check_quality_answer` (INT)

**Технические поля:**
- `ip_address`, `user_agent`, `time_spent_seconds`

#### Индексы

**Файл:** [`sql/011_add_indexes_for_percentiles.sql`](sql/011_add_indexes_for_percentiles.sql)

```sql
-- Основные индексы
INDEX idx_status (status)
INDEX idx_completed_at (completed_at)
INDEX idx_code (code)

-- Индексы для шкал
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
INDEX idx_mindset (mindset_technical_humanitarian)

-- Составной индекс для фильтрации
INDEX idx_status_completed (status, completed_at)
```

---

## Резервное копирование

### Механизм

**Файл функции:** [`public/backup.php`](public/backup.php)

**Процесс:**
1. Респондент завершает опрос (страница 10)
2. `index.php` вызывает `saveRespondentBackup()`
3. Создаётся JSON файл в папке `responses/`
4. Имя файла: `{respondent_id}.json`

### Структура JSON бэкапа

**Пример файла:** [`responses/{respondent_id}.json`](responses/)

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
    ...
    "mijs_items": "{\"mijs_q1_...\": 1, ...}",
    "mijs_total": 38,
    ...
    "time_spent_seconds": 915
  }
}
```

**Полное описание:** См. [`RESPONSES_BACKUP.md`](RESPONSES_BACKUP.md)

### Восстановление из бэкапа

**Вариант 1: Python-скрипт**

```python
import json
from pathlib import Path

backup_dir = Path('responses')
for json_file in backup_dir.glob('*.json'):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # Обработка данных
```

**Вариант 2: Загрузка в БД**

См. раздел "Загрузка данных" в [`TECHNICAL_DOCS.md`](TECHNICAL_DOCS.md)

---

## Логирование

### Конфигурация

**Файл:** [`config/config.php`](config/config.php)

```php
define('LOGGING', true);      // Включить/выключить логирование
define('DEBUG_LOGGING', true); // Подробное логирование (отладка)
```

### Формат записей

**Файлы логов:** [`logs/app_YYYY-MM-DD.log`](logs/)

```
[2026-03-10 12:34:56] [INFO] Сообщение
[2026-03-10 12:34:57] [DEBUG] Отладочная информация
[2026-03-10 12:34:58] [WARNING] Предупреждение
[2026-03-10 12:34:59] [ERROR] Ошибка
[2026-03-10 12:35:00] [BACKUP] Резервное копирование
```

### Типы записей

| Уровень | Описание | Пример |
|---------|----------|--------|
| `INFO` | Обычные события | "Page 10 saved for respondent..." |
| `DEBUG` | Отладочная информация | "Backup directory: ..." |
| `WARNING` | Предупреждения | "File reported as created but doesn't exist" |
| `ERROR` | Ошибки | "file_put_contents failed" |
| `BACKUP` | Резервное копирование | "=== BACKUP START ===" |

### Функция логирования

**Файл:** [`config/config.php`](config/config.php)

```php
function log_event($message, $level = 'INFO') {
    if (!defined('LOGGING') || !LOGGING) {
        return; // Логирование выключено
    }
    
    $timestamp = date('Y-m-d H:i:s');
    $log_file = LOGS_PATH . '/app_' . date('Y-m-d') . '.log';
    $log_entry = "[$timestamp] [$level] $message" . PHP_EOL;
    file_put_contents($log_file, $log_entry, FILE_APPEND);
}
```

---


## Ссылки на файлы проекта

| Компонент | Файл | Описание |
|-----------|------|----------|
| **Конфигурация** | [`config/config.php`](config/config.php) | Настройки БД, логирования |
| **Ключи переменных** | [`config/variable_keys.php`](config/variable_keys.php) | Маппинг вопросов в ключи |
| **Маршрутизатор** | [`public/index.php`](public/index.php) | Обработка страниц 0-10 |
| **Сохранение** | [`public/save.php`](public/save.php) | AJAX сохранение ответов |
| **Бэкапы** | [`public/backup.php`](public/backup.php) | Функция резервного копирования |
| **Результаты** | [`public/results.php`](public/results.php) | Страница результатов |
| **База данных** | [`src/Database.php`](src/Database.php) | PDO wrapper |
| **Опросник** | [`src/Survey.php`](src/Survey.php) | Логика опросника |
| **Калькулятор** | [`src/Calculator.php`](src/Calculator.php) | Расчёт баллов шкал |
| **Создание БД** | [`sql/create_database_full.sql`](sql/create_database_full.sql) | Полный скрипт БД |
| **Индексы** | [`sql/011_add_indexes_for_percentiles.sql`](sql/011_add_indexes_for_percentiles.sql) | Индексы для полей |

---

**Контакты:** maxim.dorofeev@mnogosdelal.ru
