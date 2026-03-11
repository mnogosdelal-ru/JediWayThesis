# API для получения результатов респондента

## Дата: 2026-03-08 (обновлено: поиск по любому статусу)

## Эндпоинт

```
GET /api.php?code=ABC123
```

## Параметры

| Параметр | Тип | Обязательный | Описание |
|----------|-----|--------------|----------|
| `code` | string | Да | Уникальный код респондента (6 знаков: 3 буквы + 3 цифры) |

**Примечание:** API ищет респондента по коду независимо от статуса (`in_progress`, `completed`, `abandoned`).

## Ответ

### Успешный запрос (200 OK)

```json
{
  "success": true,
  "respondent": {
    "id": "3acab4b5eb8685882d221382ef9792b2",
    "code": "TLZ827",
    "created_at": "2026-03-08 10:15:30",
    "completed_at": "2026-03-08 10:30:45",
    "time_spent_seconds": 915
  },
  "demographics": {
    "age": 35,
    "gender": "male",
    "children_count": 2,
    "work_experience_years": 12,
    "position": "middle_manager",
    "industry": "it",
    "industry_other": null,
    "remote_days": "3",
    "mindset_technical_humanitarian": 2,
    "tool_preference": 2
  },
  "scores": {
    "mijs_total": 38,
    "mijs_urgency": 15,
    "mijs_agency": 23,
    "mbi_exhaustion": 28,
    "mbi_cynicism": 12,
    "mbi_efficacy": 35,
    "mbi_total": 75,
    "swls_total": 26,
    "procrastination_total": 22,
    "practices_freq_total": 85,
    "practices_quality_total": 65,
    "vaccines_total": 8,
    "work_urgent_important": 4,
    "personal_urgent_important": 4,
    "work_satisfaction": 5
  },
  "answers": {
    "open_most_useful_practice": "Планирование на неделю помогает снизить хаос",
    "open_other_practices": "Хочу внедрить зелёные задачи по утрам"
  },
  "detailed_responses": {
    "mijs_q1_ya_chuvstvuyu_chto_moi_resursy": 2,
    "mijs_q2_srochnye_dela_meshayut_mne_razvivatsya": 3,
    "mijs_q3_ya_zhivu_v_rezhime_tusheniya_pozharov": 4,
    "mijs_q4_moy_potentsial_ne_realizuetsya_iz": 1,
    "mijs_q5_srochnye_dela_zabirayu_moyu_luchshuyu": 5,
    "mijs_q6_ya_uspevayu_zavershit_vazhnye_etapy": 4,
    "mijs_q7_kazhdyy_den_ya_vnoshu_khotya_nebolshoy": 3,
    "mijs_q8_ya_ne_chuvstvuyu_viny_kogda_govoryu": 2,
    "mijs_q9_ya_predpochitayu_delat_vazhnye_zadachi": 4,
    "mijs_q10_ya_dovozhu_vazhnye_dela_do_kontsa": 3,
    "mijs_q11_mne_udaetsya_sokhranit_kontsentratsiyu_na_vazhnom": 4,
    "mijs_q12_kazhdyy_den_ya_mogu_sformulirovat_kakoe": 5,
    "swls_q1_v_osnovnom_moya_zhizn_blizka_k_idealu": 5,
    "swls_q2_obstoyatelstva_moei_zhizni_isklyuchitelno_blagopriyatny": 4,
    "swls_q3_ya_polnostyu_udovletvoren_moei_zhiznyu": 5,
    "swls_q4_u_menya_est_v_zhizni_to": 6,
    "swls_q5_esli_by_mne_prishlos_zhit_esche_raz": 4,
    "mbi_exh_q1_ya_chuvstvuyu_sebya_emotsionalno_opustoshennym": 4,
    "mbi_exh_q2_posle_raboty_ya_chuvstvuyu_sebya_kak_vyzhatyy_limon": 3,
    "mbi_exh_q3_utrom_ya_chuvstvuyu_ustalost_i_nezhelanie_idti": 4,
    "mbi_exh_q6_posle_raboty_na_nekotoroe_vremya_khochetsya_uedinit_sya": 5,
    "mbi_exh_q8_ya_chuvstvuyu_ugnetennost_i_apatiyu": 6,
    "mbi_exh_q13_moya_rabota_vse_bolshe_menya_razocharovyvaet": 4,
    "mbi_exh_q14_mne_kazhetsya_chto_ya_slishkom_mnogo_rabotayu": 5,
    "mbi_exh_q16_mne_khochetsya_uedinit_sya_i_otdokhnut_ot_vsego": 3,
    "mbi_exh_q20_ya_chuvstvuyu_ravnodushie_i_poteryu_interesa": 4,
    "mbi_cyn_q5_ya_chuvstvuyu_chto_obshchayus_s_nekotorymi_podchinennymi": 3,
    "mbi_cyn_q10_v_poslednee_vremya_ya_stal_bolee_cherstvym_po_otnosheniyu": 4,
    "mbi_cyn_q11_ya_zamechayu_chto_moya_rabota_ozhestochaet_menya": 2,
    "mbi_cyn_q15_byvaet_chto_mne_deistvitelno_bezrazlichno_to": 3,
    "mbi_cyn_q22_v_poslednee_vremya_mne_kazhetsya_chto_kollegi_i_podchinennye": 5,
    "mbi_eff_q4_ya_khorosho_ponimayu_chto_chuvstvui_moi_podchinennye": 5,
    "mbi_eff_q7_ya_umeyu_nakhodit_pravilnoe_reshenie_v_konfliktnykh_situatsiyakh": 4,
    "mbi_eff_q9_ya_uveren_chto_moya_rabota_nuzhna_lyudyam": 6,
    "mbi_eff_q12_u_menya_mnogo_planov_na_budushchee_i_ya_veru": 5,
    "mbi_eff_q17_ya_legko_mogu_sozdat_atmosferu_dobrozhelatelnosti_i_sotrudnichestva": 4,
    "mbi_eff_q18_vo_vremya_raboty_ya_chuvstvuyu_priyatnoe_ozhivlenie": 5,
    "mbi_eff_q19_blagodarya_svoei_rabote_ya_uzhe_sdelal_v_zhizni": 6,
    "mbi_eff_q21_na_rabote_ya_spokoyno_spravlayus_s_emotsionalnymi_problemami": 4,
    "proc_q1_ya_chasto_zamechayu_chto_zaranee_vypolnyayu_zadachi": 2,
    "proc_q2_ya_zamechayu_chto_rabota_ne_vypolnyaetsya_v_techenie_neskolkih_dnei": 3,
    "proc_q3_kogda_mne_nuzhno_vypolnit_trudnuyu_rabotu_ya_zhdu": 3,
    "proc_q4_mne_obychno_prikhoditsya_speshit_chtoby_vypolnit_zadachi_vovremya": 4,
    "proc_q5_kogda_priblizhayutsya_sroki_zaversheniya_raboty_ya_chasto_teruyu": 3,
    "proc_q6_pri_podgotovke_k_vstreche_ya_chasto_lovlyu_sebya_na_tom": 2,
    "proc_q7_chasto_ya_dolgo_ne_prystupayu_k_vypolneniyu_zadachi_kotoruyu_mne_nuzhno": 3,
    "proc_q8_ya_chasto_govoryu_sdelayu_eto_zavtra": 4,
    "prac_q1_lezha_v_posteli_ne_polzovatsya_ustroistvami_freq": 4,
    "prac_q1_lezha_v_posteli_ne_polzovatsya_ustroistvami_qual": 3,
    "prac_q2_razgruzit_svoyu_pamyat_freq": 3,
    "prac_q2_razgruzit_svoyu_pamyat_qual": 4,
    ...
    "vacc_q1_peregulirovka_slozhnykh_zadach": 2,
    "vacc_q2_vneshnyaya_pamyat_dlya_idey": 3,
    "vacc_q3_pravilo_2_minut": 1,
    "vacc_q4_zelenye_zadachi_po_utram": 2,
    "vacc_q5_voskresnoe_planirovanie": 3
  }
}
```

### Ошибки

**400 Bad Request:**
```json
{
  "success": false,
  "error": "Code parameter is required"
}
```

**404 Not Found:**
```json
{
  "success": false,
  "error": "Респондент с кодом ABC123 не найден"
}
```

**500 Internal Server Error:**
```json
{
  "success": false,
  "error": "Internal server error"
}
```

---

## Структура ответа

### respondent
Основная информация о респонденте:
- `id` - Уникальный идентификатор (MD5 hash)
- `code` - Код для доступа (3 буквы + 3 цифры)
- `created_at` - Дата начала прохождения
- `completed_at` - Дата завершения
- `time_spent_seconds` - Время прохождения в секундах

### demographics
Демографические данные:
- `age` - Возраст
- `gender` - Пол
- `children_count` - Количество детей
- `work_experience_years` - Стаж работы
- `position` - Позиция
- `industry` - Отрасль
- `industry_other` - Другая отрасль (если указано)
- `remote_days` - Дней удалённой работы в неделю
- `mindset_technical_humanitarian` - Склад ума (1=технический, 5=гуманитарный)
- `tool_preference` - Предпочтение инструментов (1=электронные, 5=бумажные)

### scores
Агрегированные баллы по шкалам:
- `mijs_total`, `mijs_urgency`, `mijs_agency` - MIJS
- `mbi_exhaustion`, `mbi_cynicism`, `mbi_efficacy`, `mbi_total` - MBI
- `swls_total` - SWLS
- `procrastination_total` - Прокрастинация
- `practices_freq_total`, `practices_quality_total` - Практики
- `vaccines_total` - Вакцины
- `work_urgent_important`, `personal_urgent_important` - Срочное/важное
- `work_satisfaction` - Удовлетворённость работой

### answers

Открытые ответы респондента:
- `open_most_useful_practice` — Самая полезная практика (текст)
- `open_other_practices` — Другие практики для внедрения (текст)

**Примечание:** Все остальные ответы сохраняются в разделе `detailed_responses` с ключами переменных.

### detailed_responses

Все ответы респондента с ключами переменных в формате `{ключ: значение}`:
- Все вопросы MIJS, SWLS, MBI, прокрастинации
- Практики с суффиксами `_freq` и `_qual`
- Вакцины

**Преимущества:**
- Можно получить ответ на конкретный вопрос по ключу
- Ключ содержит номер вопроса и первые 4 слова текста
- Все ключи на латинице

---

## Примеры использования

### JavaScript (fetch)

```javascript
const code = 'TLZ827';
fetch(`/api.php?code=${code}`)
  .then(response => response.json())
  .then(data => {
    if (data.success) {
      console.log('MIJS Total:', data.scores.mijs_total);
      console.log('MBI Exhaustion:', data.scores.mbi_exhaustion);
      
      // Получить ответ на конкретный вопрос
      console.log('MIJS Q1:', data.detailed_responses.mijs_q1_ya_chuvstvuyu_chto_moi_resursy);
    }
  });
```

### Python (requests)

```python
import requests

code = 'TLZ827'
response = requests.get(f'http://localhost/api.php?code={code}')
data = response.json()

if data['success']:
    print(f"MIJS Total: {data['scores']['mijs_total']}")
    print(f"MBI Exhaustion: {data['scores']['mbi_exhaustion']}")
    
    # Получить ответ на конкретный вопрос
    q1_key = 'mijs_q1_ya_chuvstvuyu_chto_moi_resursy'
    print(f"MIJS Q1: {data['detailed_responses'].get(q1_key)}")
```

### R (httr)

```r
library(httr)
library(jsonlite)

code <- 'TLZ827'
response <- GET(paste0('http://localhost/api.php?code=', code))
data <- fromJSON(content(response, 'text'))

if (data$success) {
  cat('MIJS Total:', data$scores$mijs_total, '\n')
  cat('MBI Exhaustion:', data$scores$mbi_exhaustion, '\n')
  
  # Получить ответ на конкретный вопрос
  cat('MIJS Q1:', data$detailed_responses$mijs_q1_ya_chuvstvuyu_chto_moi_resursy, '\n')
}
```

---

## Примечания

1. **Детализированные ответы** доступны только для респондентов, прошедших опрос после 2026-03-08
2. Для старых респондентов поле `detailed_responses` будет `null`
3. Все тексты в ключах транслитерированы (кириллица → латиница)
4. Ключи практик имеют суффиксы `_freq` (частота) и `_qual` (качество)

---

## Контакты

Исследователь: maxim.dorofeev@mnogosdelal.ru
