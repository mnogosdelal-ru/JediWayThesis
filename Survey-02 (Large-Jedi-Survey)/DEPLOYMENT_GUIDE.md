# Развёртывание базы данных с нуля

## Дата: 2026-03-08

## Варианты развёртывания

### Вариант 1: Новая база данных (рекомендуется)

**Для кого:** Новый проект или можно удалить старую БД

**Шаги:**

1. **Создайте базу данных:**
   ```sql
   CREATE DATABASE jedi_survey CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
   ```

2. **Примените полный скрипт:**
   ```bash
   mysql -u root -p jedi_survey < sql/create_database_full.sql
   ```

   **Или через phpMyAdmin:**
   - Откройте phpMyAdmin
   - Выберите `jedi_survey`
   - Вкладка "SQL"
   - Скопируйте содержимое `sql/create_database_full.sql`
   - Нажмите "Вперёд"

**Результат:**
- ✅ Таблица `respondents` со всеми полями
- ✅ Все индексы для производительности
- ✅ Поле `responses_detailed` для детализированных ответов
- ❌ Таблица `aggregates` НЕ создаётся (отказались от агрегатов)

---

### Вариант 2: Существующая база данных

**Для кого:** Проект уже работает, нужно обновить структуру

**Шаги:**

1. **Примените миграции по порядку:**

   ```bash
   # Миграция 009: Поле responses_detailed
   mysql -u root -p jedi_survey < sql/009_add_detailed_responses_field.sql
   
   # Миграция 010: Удаление таблицы aggregates
   mysql -u root -p jedi_survey < sql/010_drop_aggregates_table.sql
   
   # Миграция 011: Добавление индексов
   mysql -u root -p jedi_survey < sql/011_add_indexes_for_percentiles.sql
   ```

2. **Проверьте структуру:**
   ```sql
   SHOW TABLES;
   -- Ожидаемо: respondents
   -- НЕ ожидаемо: aggregates
   
   SHOW INDEX FROM respondents;
   -- Ожидаемо: 16 индексов
   ```

---

## Проверка после развёртывания

### 1. Проверьте таблицы

```sql
SHOW TABLES;
```

**Ожидаемо:**
```
respondents
```

**НЕ ожидаемо:**
```
aggregates  (удалена)
```

### 2. Проверьте индексы

```sql
SHOW INDEX FROM respondents;
```

**Ожидаемо (16 индексов):**
| Index name | Column name |
|------------|-------------|
| PRIMARY | id |
| idx_status | status |
| idx_completed_at | completed_at |
| idx_code | code |
| idx_mijs_total | mijs_total |
| idx_mbi_exhaustion | mbi_exhaustion_score |
| idx_mbi_cynicism | mbi_cynicism_score |
| idx_mbi_efficacy | mbi_efficacy_score |
| idx_swls_total | swls_total |
| idx_procrastination | procrastination_total |
| idx_practices_freq | practices_freq_total |
| idx_practices_quality | practices_quality_total |
| idx_vaccines | vaccines_total |
| idx_personal_urgent | personal_urgent_important |
| idx_work_urgent | work_urgent_important |
| idx_work_satisfaction | work_satisfaction |
| idx_age | age |
| idx_children_count | children_count |
| idx_remote_days | remote_days |
| idx_status_completed | status, completed_at |

### 3. Проверьте процентили

```sql
SET profiling = 1;

SELECT 
    COUNT(*) AS total_count,
    SUM(CASE WHEN mijs_total < 38 THEN 1 ELSE 0 END) AS mijs_less,
    SUM(CASE WHEN mijs_total = 38 THEN 1 ELSE 0 END) AS mijs_equal,
    SUM(CASE WHEN mijs_total > 38 THEN 1 ELSE 0 END) AS mijs_greater
FROM respondents 
WHERE status = 'completed';

SHOW PROFILES;
```

**Ожидаемо:** Duration < 100 ms

---

## Структура базы данных

### Таблица: respondents

**Основные поля:**
- `id`, `code`, `session_id` — идентификаторы
- `status`, `current_page`, `completed_at` — статус прохождения
- `age`, `gender`, `children_count`, и т.д. — демография
- `mijs_*`, `mbi_*`, `swls_*`, и т.д. — шкалы
- `practices_frequency`, `practices_quality` — практики (20 практик)
- `open_most_useful_practice`, `open_other_practices` — открытые вопросы
- `responses_detailed` — все ответы с ключами переменных (JSON)

**Индексы:** 16 индексов для ускорения COUNT запросов

---

## Отличия от старой версии

| Параметр | Старая версия | Новая версия |
|----------|---------------|--------------|
| Таблицы | respondents, aggregates | respondents |
| Индексы | 5 | 20 |
| Практик | 21 | 20 |
| MIJS item 8 | «Я не чувствую вины...» | «Я легко отсеиваю...» |
| Агрегаты | Есть | Нет (процентили в реальном времени) |
| responses_detailed | Нет | Есть (JSON) |

---

## Примечания

### 1. Отказ от агрегатов

Мы отказались от таблицы `aggregates` в пользу прямых COUNT запросов.

**Преимущества:**
- ✅ Актуальные данные (не нужно пересчитывать)
- ✅ Проще архитектура
- ✅ Меньше кода
- ✅ Выше производительность

**Недостатки:**
- ❌ Немного выше нагрузка на БД (но индексы компенсируют)

### 2. 20 практик вместо 21

Удалена практика 21 (дубль практики 18).

**Было:** 21 практика, диапазон 21-126  
**Стало:** 20 практик, диапазон 20-120

### 3. Обновлённая формулировка MIJS item 8

**Было:** «Я не чувствую вины, когда говорю "нет"...»  
**Стало:** «Я легко отсеиваю задачи, которые не служат...»

---

## Документация

| Файл | Описание |
|------|----------|
| `create_database_full.sql` | Полный скрипт создания БД |
| `009_add_detailed_responses_field.sql` | Миграция поля responses_detailed |
| `010_drop_aggregates_table.sql` | Миграция удаления aggregates |
| `011_add_indexes_for_percentiles.sql` | Миграция индексов |
| `MIGRATION_011_INSTRUCTIONS.md` | Инструкция по применению миграции 011 |
| `AGGREGATES_REMOVAL.md` | Описание отказа от агрегатов |
| `QUICK_START_NO_AGGREGATES.md` | Быстрый старт без агрегатов |

---

## Контакты

При проблемах:
- maxim.dorofeev@mnogosdelal.ru
