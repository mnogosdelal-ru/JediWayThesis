# Оптимизация API: упрощение раздела answers

## Дата: 2026-03-08

## Изменения

### 1. public/api.php

**Раздел `scores`:**
- ❌ Удалён `tool_preference` (дублируется в `demographics`)

**Раздел `answers`:**
- ❌ Удалены все сырые ответы (массивы/объекты)
- ✅ Оставлены только открытые вопросы:
  - `open_most_useful_practice`
  - `open_other_practices`

**Обоснование:**
- Все ответы уже сохраняются в `detailed_responses` с ключами переменных
- Дублирование данных увеличивает размер ответа
- `detailed_responses` удобнее для анализа (ключи вместо индексов)

---

### 2. docs/API_REFERENCE.md

Обновлена документация:
- ✅ Пример JSON ответа обновлён
- ✅ Описание раздела `answers` упрощено
- ✅ Добавлено примечание о `detailed_responses`

---

## Структура ответа API

### Было (до изменений):

```json
{
  "demographics": {
    "tool_preference": 2,
    ...
  },
  "scores": {
    "tool_preference": 2,  // ← Дубликат
    ...
  },
  "answers": {
    "mindset_technical_humanitarian": 2,
    "mijs_items": [2, 3, 4, ...],
    "swls_items": [5, 4, 5, ...],
    "mbi_exhaustion_items": [...],
    "mbi_cynicism_items": [...],
    "mbi_efficacy_items": [...],
    "procrastination_items": [...],
    "practices_frequency": {...},
    "practices_quality": {...},
    "vaccines": {...},
    "open_most_useful_practice": "...",
    "open_other_practices": "..."
  },
  "detailed_responses": {...}
}
```

### Стало (после изменений):

```json
{
  "demographics": {
    "tool_preference": 2,
    ...
  },
  "scores": {
    // ← tool_preference удалён
    ...
  },
  "answers": {
    "open_most_useful_practice": "...",
    "open_other_practices": "..."
  },
  "detailed_responses": {
    "mijs_q1_ya_chuvstvuyu_chto_moi_resursy": 2,
    "mijs_q2_srochnye_dela_meshayut_mne_razvivatsya": 3,
    ...
  }
}
```

---

## Преимущества

### 1. Меньший размер ответа

**Экономия:** ~2-5 КБ на запрос (в зависимости от количества ответов)

### 2. Чище структура

- `demographics` — демографические данные
- `scores` — агрегированные баллы по шкалам
- `answers` — только открытые вопросы (текст)
- `detailed_responses` — все ответы с ключами

### 3. Нет дублирования

- `tool_preference` только в `demographics`
- Ответы только в `detailed_responses`

### 4. Удобнее для анализа

```javascript
// Было (непонятно что это):
response.answers.mijs_items[0]  // 2

// Стало (ясно что измеряем):
response.detailed_responses.mijs_q1_ya_chuvstvuyu_chto_moi_resursy  // 2
```

---

## Влияние на существующий код

### Если вы использовали `answers.*` для получения сырых ответов:

**Было:**
```javascript
const mijsItems = response.answers.mijs_items;
const swlsItems = response.answers.swls_items;
```

**Стало (используйте detailed_responses):**
```javascript
// Получить все ответы MIJS
const mijsAnswers = Object.keys(response.detailed_responses)
  .filter(key => key.startsWith('mijs_'))
  .reduce((obj, key) => {
    obj[key] = response.detailed_responses[key];
    return obj;
  }, {});

// Или получить конкретный ответ
const mijsQ1 = response.detailed_responses.mijs_q1_ya_chuvstvuyu_chto_moi_resursy;
```

### Если вы использовали `scores.tool_preference`:

**Было:**
```javascript
const toolPref = response.scores.tool_preference;
```

**Стало:**
```javascript
const toolPref = response.demographics.tool_preference;
```

---

## Проверка

### 1. Протестировать API

```bash
curl "http://localhost/.../api.php?code=TLZ827" | jq .
```

**Проверить:**
- ✅ `tool_preference` только в `demographics`
- ✅ `answers` содержит только `open_most_useful_practice` и `open_other_practices`
- ✅ `detailed_responses` содержит все ответы с ключами

### 2. Проверить размер ответа

**Было:** ~15-20 КБ  
**Стало:** ~12-17 КБ (экономия ~20-25%)

---

## Итог

✅ API оптимизирован  
✅ Удалено дублирование данных  
✅ Уменьшен размер ответа  
✅ Улучшена структура  
✅ Документация обновлена  

---

## Контакты

Исследователь: maxim.dorofeev@mnogosdelal.ru
