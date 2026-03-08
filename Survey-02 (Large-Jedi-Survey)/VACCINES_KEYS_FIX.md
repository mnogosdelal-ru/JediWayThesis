# Исправление ключей вакцин

## Дата: 2026-03-08

## Проблема

Ключи вакцин в `config/variable_keys.php` не соответствовали реальным названиям на странице 9 опросника.

### Было (неправильно):

```php
$VACCINES_KEYS = [
    1 => 'vacc_q1_peregulirovka_slozhnykh_zadach',      // ❌ Нет такой вакцины
    2 => 'vacc_q2_vneshnyaya_pamyat_dlya_idey',         // ❌ Нет такой вакцины
    3 => 'vacc_q3_pravilo_2_minut',                     // ❌ Нет такой вакцины
    4 => 'vacc_q4_zelenye_zadachi_po_utram',            // ❌ Нет такой вакцины
    5 => 'vacc_q5_voskresnoe_planirovanie'              // ❌ Нет такой вакцины
];
```

### Стало (правильно):

```php
$VACCINES_KEYS = [
    1 => 'vacc_q1_udalit_prilozheniya_sotsialnykh_setey',
    2 => 'vacc_q2_otklyuchit_krasnye_kruzhochki_na_ikonkakh',
    3 => 'vacc_q3_otklyuchit_opoveshcheniya_o_novykh_soobshcheniyakh_v_chatakh',
    4 => 'vacc_q4_otklyuchit_opoveshcheniya_o_novykh_soobshcheniyakh_v_pochte',
    5 => 'vacc_q5_otklyuchit_push_opoveshcheniya_ot_prilozheniy'
];
```

---

## Реальные вакцины (страница 9)

| № | Вакцина | Описание |
|---|---------|----------|
| 1 | Удалить приложения социальных сетей со своего смартфона | Полное удаление соцсетей. Можно пользоваться через браузер при необходимости. |
| 2 | Отключить красные кружочки на иконках приложений | Убрать уведомления-бейджи. Владельцы Android могут установить oLauncher. |
| 3 | Отключить оповещения о новых сообщениях в чатах | Полное отключение уведомлений из мессенджеров. Надо — позвонят. |
| 4 | Отключить оповещения о новых сообщениях в почте | Отключение push- и звуковых уведомлений о новых письмах. |
| 5 | Отключить push-оповещения от приложений (соцсети, магазины, заправки) | Отключение всех маркетинговых уведомлений, кроме звонков и банковских приложений. |

---

## Изменённые файлы

### 1. config/variable_keys.php

Обновлены ключи вакцин в соответствии с реальными названиями.

### 2. docs/VARIABLE_KEYS.md

Обновлена таблица вакцин с правильными ключами.

---

## Влияние на данные

### Существующие респонденты

Данные в БД используют числовые ключи (1, 2, 3, 4, 5), поэтому изменения **не затронут** существующие записи.

**Пример хранения:**
```json
{
  "vaccines": {"1": 3, "2": 2, "3": 3, "4": 1, "5": 2},
  "vaccines_total": 11
}
```

### Новые респонденты

Будут сохранять данные с правильными ключами:

**Пример BACKUP лога:**
```json
{
  "detailed_responses": {
    "vacc_q1_udalit_prilozheniya_sotsialnykh_setey": 3,
    "vacc_q2_otklyuchit_krasnye_kruzhochki_na_ikonkakh": 2,
    "vacc_q3_otklyuchit_opoveshcheniya_o_novykh_soobshcheniyakh_v_chatakh": 3,
    "vacc_q4_otklyuchit_opoveshcheniya_o_novykh_soobshcheniyakh_v_pochte": 1,
    "vacc_q5_otklyuchit_push_opoveshcheniya_ot_prilozheniy": 2
  }
}
```

---

## Проверка

### 1. Пройти опрос

1. Откройте `http://localhost/.../public/`
2. Дойдите до страницы 9 (Вакцины)
3. Проверьте, что названия вакцин совпадают с ключами в документации

### 2. Проверить лог

После сохранения проверьте BACKUP запись в логе:

```bash
powershell -Command "Get-Content logs\app_*.log | Select-String 'vacc_q' | Select-Object -Last 1"
```

**Ожидаемо:** Ключи начинаются с `vacc_q1_udalit_...`, `vacc_q2_otklyuchit_...` и т.д.

---

## Итог

✅ Ключи вакцин теперь соответствуют реальным названиям на странице 9

✅ Документация обновлена

✅ Существующие данные не затронуты

✅ Новые данные будут сохраняться с правильными ключами

---

## Контакты

Исследователь: maxim.dorofeev@mnogosdelal.ru
