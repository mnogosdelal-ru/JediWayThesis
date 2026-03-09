# Исправление: Использование ключей из variable_keys.php в Calculator.php

**Дата:** 2026-03-10  
**Задача:** Перейти на использование имён ключей из `config/variable_keys.php` вместо порядковых номеров вопросов во всех функциях расчёта шкал.

## Проблема

Ранее `Calculator.php` использовал порядковые номера вопросов (1, 2, 3...) или индексы массивов для доступа к ответам. Это могло привести к ошибкам, если:
- Порядок ключей в JSON изменялся
- Данные приходили в ассоциативном формате с пропусками (например, `[5=>val, 10=>val, ...]`)
- Ключи не совпадали с ожидаемыми позициями

## Решение

Все функции расчёта шкал теперь используют ключи из `config/variable_keys.php`:
- `$MIJS_KEYS` — для MIJS
- `$MBI_EXHAUSTION_KEYS`, `$MBI_CYNICISM_KEYS`, `$MBI_EFFICACY_KEYS` — для MBI
- `$SWLS_KEYS` — для SWLS
- `$PROCRASTINATION_KEYS` — для прокрастинации
- `$VACCINES_KEYS` — для вакцин
- `$PRACTICES_KEYS` — для практик (используется в `save.php`)

## Изменения

### 1. src/Calculator.php

**До:**
```php
public static function calculateMijs(array $items): array {
    if (count($items) < 12) {
        return ['urgency' => 0, 'agency' => 0, 'total' => 0];
    }

    // Items 1-5: напрямую
    $urgency = array_sum(array_slice($items, 0, 5));

    // Items 6-12: инвертируются
    $agency_items = array_slice($items, 5, 7);
    $agency = array_sum(array_map(fn($x) => 6 - $x, $agency_items));

    return [
        'urgency' => $urgency,
        'agency' => $agency,
        'total' => $urgency + $agency
    ];
}
```

**После:**
```php
public static function calculateMijs(array $items): array {
    global $MIJS_KEYS;

    if (empty($items)) {
        return ['urgency' => 0, 'agency' => 0, 'total' => 0];
    }

    // Items 1-5: напрямую
    $urgency = 0;
    for ($i = 1; $i <= 5; $i++) {
        $key = $MIJS_KEYS[$i] ?? null;
        if ($key !== null) {
            $urgency += (int)($items[$key] ?? 0);
        }
    }

    // Items 6-12: инвертируются
    $agency = 0;
    for ($i = 6; $i <= 12; $i++) {
        $key = $MIJS_KEYS[$i] ?? null;
        if ($key !== null) {
            $value = (int)($items[$key] ?? 0);
            $agency += (6 - $value);
        }
    }

    return [
        'urgency' => $urgency,
        'agency' => $agency,
        'total' => $urgency + $agency
    ];
}
```

### 2. MBI (эмоциональное истощение)

**До:**
```php
if ($isAssociative) {
    $exhaustion += ($exhaustion_items[1] ?? 0);
    $exhaustion += ($exhaustion_items[2] ?? 0);
    $exhaustion += ($exhaustion_items[3] ?? 0);
    $exhaustion += (6 - ($exhaustion_items[6] ?? 0));
    $exhaustion += ($exhaustion_items[8] ?? 0);
    // ...
}
```

**После:**
```php
foreach ($MBI_EXHAUSTION_KEYS as $questionNum => $key) {
    $value = (int)($exhaustion_items[$key] ?? 0);
    
    // Item 6 - обратный
    if ($questionNum === 6) {
        $exhaustion += (6 - $value);
    } else {
        $exhaustion += $value;
    }
}
```

### 3. Остальные шкалы

Аналогичные изменения внесены для:
- **SWLS** — цикл по `$SWLS_KEYS`
- **Прокрастинация** — цикл по `$PROCRASTINATION_KEYS` с инверсией item 1
- **Вакцины** — цикл по `$VACCINES_KEYS`
- **MBI Цинизм** — цикл по `$MBI_CYNICISM_KEYS`
- **MBI Эффективность** — цикл по `$MBI_EFFICACY_KEYS`

### 4. public/save.php

**Изменения:**
- Убран резервный генератор ключей `"q{$questionNumber}"` — теперь используются только ключи из `variable_keys.php`
- Если ключ не найден в `$keysMap`, он пропускается

**До:**
```php
$key = $keysMap[$questionNumber] ?? "q{$questionNumber}";
```

**После:**
```php
$key = $keysMap[$questionNumber] ?? null;
if ($key !== null) {
    $result[$key] = (int)$value;
}
```

## Преимущества

1. **Надёжность:** Ключи всегда соответствуют данным в БД
2. **Читаемость:** Код явно показывает, какие вопросы используются
3. **Поддержка:** При изменении ключей нужно обновить только `variable_keys.php`
4. **Консистентность:** Все части кода используют одни и те же ключи

## Примеры ключей

### MIJS
```php
$MIJS_KEYS = [
    1 => 'mijs_q1_ya_chuvstvuyu_chto_moi_resursy',
    2 => 'mijs_q2_srochnye_dela_meshayut_mne_razvivatsya',
    // ...
];
```

### MBI (эмоциональное истощение)
```php
$MBI_EXHAUSTION_KEYS = [
    1 => 'mbi_exh_q1_k_kontsu_rabochey_nedeli_ya_chuvstvuyu_sebya_emotsionalno_opustoshennym',
    2 => 'mbi_exh_q2_k_kontsu_rabochego_dnya_ya_chuvstvuyu_sebya_kak_vyzhatyy_limon',
    3 => 'mbi_exh_q3_ya_chuvstvuyu_sebya_ustalym_kogda_vstayu_utrom',
    6 => 'mbi_exh_q6_ya_chuvstvuyu_sebya_energichnym_i_emotsionalno_voodushevlyennym', // обратный
    // ...
];
```

## Тестирование

Проверьте:
1. ✅ Все шкалы рассчитываются корректно
2. ✅ Обратные пункты (MIJS 6-12, MBI item 6, Procrastination item 1) инвертируются правильно
3. ✅ Данные в БД сохраняются с полными ключами
4. ✅ Результаты в `results.php` отображаются верно

## Примечание

`Calculator.php` теперь требует подключения `config/variable_keys.php` через `require_once`. Это уже сделано в `save.php`.

Для `results.php` подключение не требуется, так как там используются готовые значения из БД.
