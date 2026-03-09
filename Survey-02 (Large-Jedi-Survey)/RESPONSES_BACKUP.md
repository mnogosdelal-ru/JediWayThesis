# Резервное копирование ответов респондентов

## Дата: 2026-03-09

## Описание

После завершения опроса (страница 10) полный JSON со всеми ответами респондента сохраняется в файл в папке `responses/`.

---

## Структура backup файла

**Файл:** `responses/{respondent_id}.json`

```json
{
  "respondent_id": "7cec08e7e860a00d82c3d550e3c8b2e7",
  "timestamp": "2026-03-09 21:53:37",
  "answers": {
    "consent_given": 1,
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
    "mijs_items": {
      "mijs_q1_ya_chuvstvuyu_chto_moi_resursy": 1,
      "mijs_q2_srochnye_dela_meshayut_mne_razvivatsya": 2,
      ...
    },
    "mijs_urgency_score": 15,
    "mijs_agency_score": 23,
    "mijs_total": 38,
    "swls_items": {
      "swls_q1_v_osnovnom_moya_zhizn_blizka_k_idealu": 5,
      ...
    },
    "swls_total": 26,
    "mbi_exhaustion_items": {...},
    "mbi_exhaustion_score": 28,
    "mbi_cynicism_items": {...},
    "mbi_cynicism_score": 12,
    "mbi_efficacy_items": {...},
    "mbi_efficacy_score": 35,
    "mbi_total": 75,
    "procrastination_items": {...},
    "procrastination_total": 22,
    "practices_frequency": {...},
    "practices_freq_total": 85,
    "practices_quality": {...},
    "practices_quality_total": 65,
    "vaccines": {...},
    "vaccines_total": 8,
    "open_most_useful_practice": "Планирование на неделю помогает снизить хаос",
    "open_other_practices": "Хочу внедрить зелёные задачи по утрам",
    "attention_check_passed": 1,
    "attention_check_freq_answer": 6,
    "attention_check_quality_answer": 4,
    "time_spent_seconds": 915
  }
}
```

---

## Преимущества

### 1. **Полнота данных**
Все ответы респондента в одном файле, включая:
- ✅ Все шкалы с ключами переменных
- ✅ Агрегированные баллы
- ✅ Открытые ответы
- ✅ Внимание check
- ✅ Время прохождения

### 2. **Независимость от БД**
- ✅ Бэкап хранится отдельно от базы данных
- ✅ Можно восстановить данные даже при потере БД
- ✅ Удобно для анализа в Python/R

### 3. **Чистота БД**
- ✅ Нет дублирования данных (удалено поле `responses_detailed`)
- ✅ Меньше размер таблицы
- ✅ Быстрее запросы

---

## Использование

### Просмотр бэкапа

```bash
# Просмотреть файл бэкапа
cat responses/7cec08e7e860a00d82c3d550e3c8b2e7.json

# Или через Python
python -c "import json; print(json.load(open('responses/7cec08e7e860a00d82c3d550e3c8b2e7.json')))"
```

### Восстановление из бэкапа

```python
import json
import mysql.connector

# Загружаем бэкап
with open('responses/7cec08e7e860a00d82c3d550e3c8b2e7.json') as f:
    backup = json.load(f)

# Подключаемся к БД
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='',
    database='jedi_survey'
)
cursor = conn.cursor()

# Восстанавливаем данные
answers = backup['answers']
cursor.execute("""
    UPDATE respondents SET
        age = %s,
        gender = %s,
        mijs_items = %s,
        mijs_total = %s,
        ...
    WHERE id = %s
""", (
    answers['age'],
    answers['gender'],
    answers['mijs_items'],
    answers['mijs_total'],
    ...
    backup['respondent_id']
))

conn.commit()
```

---

## Хранение

**Путь:** `responses/{respondent_id}.json`

**Размер файла:** ~5-10 КБ на респондента

**Примерный размер:**
- 100 респондентов = ~500-1000 КБ
- 1000 респондентов = ~5-10 МБ
- 10000 респондентов = ~50-100 МБ

---

## Безопасность

**Рекомендации:**
1. Регулярно копируйте папку `responses/` на внешний носитель
2. Шифруйте бэкапы при хранении
3. Не храните бэкапы в публичном доступе

---

## Отключение бэкапирования

Если нужно отключить сохранение в файлы, закомментируйте вызов в `save.php`:

```php
// if ($success && $page >= 10) {
//     saveRespondentBackup($respondent_id, $answers);
// }
```

---

**Готово!** Все ответы респондентов сохраняются в отдельные JSON файлы для надёжности.
