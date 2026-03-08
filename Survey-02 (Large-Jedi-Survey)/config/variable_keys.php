<?php
/**
 * Конфигурация ключей переменных для сохранения ответов
 * 
 * Формат ключа: {префикс}_{номер вопроса}_{первые 4 слова транслитом}
 * Пример: mbi_q1_chuvstvuyu_sebya_emotsionalno
 */

// ============================================================================
// MIJS (Multi-Item Jedi Scale) - 12 items
// ============================================================================
$MIJS_KEYS = [
    1 => 'mijs_q1_ya_chuvstvuyu_chto_moi_resursy',      // «Я чувствую, что мои ресурсы (время и энергия) исчерпываются на срочное»
    2 => 'mijs_q2_srochnye_dela_meshayut_mne_razvivatsya', // «Срочные дела мешают мне развиваться в нужном мне направлении»
    3 => 'mijs_q3_ya_zhivu_v_rezhime_tusheniya_pozharov', // «Я живу в режиме "тушения пожаров" большую часть времени»
    4 => 'mijs_q4_moy_potentsial_ne_realizuetsya_iz',   // «Мой потенциал не реализуется из-за необходимости постоянно реагировать на срочное»
    5 => 'mijs_q5_srochnye_dela_zabirayu_moyu_luchshuyu', // «Срочные дела забирают мою лучшую энергию, важное получает остатки»
    6 => 'mijs_q6_ya_uspevayu_zavershit_vazhnye_etapy', // «Я успеваю завершить важные этапы проектов до того, как они превращаются в "горящие" задачи»
    7 => 'mijs_q7_kazhdyy_den_ya_vnoshu_khotya_nebolshoy', // «Каждый день я вношу хотя бы небольшой вклад в мои значимые цели»
    8 => 'mijs_q8_ya_ne_chuvstvuyu_viny_kogda_govoryu', // «Я не чувствую вины, когда говорю "нет" задачам, которые не соответствуют моим целям»
    9 => 'mijs_q9_ya_predpochitayu_delat_vazhnye_zadachi', // «Я предпочитаю делать важные задачи качественно, даже если это означает, что на срочный запрос я отвечу не первым»
    10 => 'mijs_q10_ya_dovozhu_vazhnye_dela_do_kontsa', // «Я довожу важные дела до конца, не бросая их на полпути из-за появления срочных дел»
    11 => 'mijs_q11_mne_udaetsya_sokhranit_kontsentratsiyu_na_vazhnom', // «Мне удается сохранять концентрацию на важном, даже если у коллег хаос и все "горит"»
    12 => 'mijs_q12_kazhdyy_den_ya_mogu_sformulirovat_kakoe' // «Каждый день я могу сформулировать, какое самое важное дело я сделал сегодня»
];

// ============================================================================
// SWLS (Satisfaction With Life Scale) - 5 items
// ============================================================================
$SWLS_KEYS = [
    1 => 'swls_q1_v_osnovnom_moya_zhizn_blizka_k_idealu', // «В основном моя жизнь близка к идеалу.»
    2 => 'swls_q2_obstoyatelstva_moei_zhizni_isklyuchitelno_blagopriyatny', // «Обстоятельства моей жизни исключительно благоприятны.»
    3 => 'swls_q3_ya_polnostyu_udovletvoren_moei_zhiznyu', // «Я полностью удовлетворен моей жизнью.»
    4 => 'swls_q4_u_menya_est_v_zhizni_to', // «У меня есть в жизни то, что мне по-настоящему нужно.»
    5 => 'swls_q5_esli_by_mne_prishlos_zhit_esche_raz' // «Если бы мне пришлось жить еще раз, я бы оставил все как есть.»
];

// ============================================================================
// MBI (Maslach Burnout Inventory) - 22 items
// ============================================================================

// Эмоциональное истощение (9 items): 1, 2, 3, 6, 8, 13, 14, 16, 20
$MBI_EXHAUSTION_KEYS = [
    1 => 'mbi_exh_q1_ya_chuvstvuyu_sebya_emotsionalno_opustoshennym', // «Я чувствую себя эмоционально опустошенным.»
    2 => 'mbi_exh_q2_posle_raboty_ya_chuvstvuyu_sebya_kak_vyzhatyy_limon', // «После работы я чувствую себя как «выжатый лимон».»
    3 => 'mbi_exh_q3_utrom_ya_chuvstvuyu_ustalost_i_nezhelanie_idti', // «Утром я чувствую усталость и нежелание идти на работу.»
    6 => 'mbi_exh_q6_posle_raboty_na_nekotoroe_vremya_khochetsya_uedinit_sya', // «После работы на некоторое время хочется уединиться от всех и всего.»
    8 => 'mbi_exh_q8_ya_chuvstvuyu_ugnetennost_i_apatiyu', // «Я чувствую угнетенность и апатию.»
    13 => 'mbi_exh_q13_moya_rabota_vse_bolshe_menya_razocharovyvaet', // «Моя работа все больше меня разочаровывает.»
    14 => 'mbi_exh_q14_mne_kazhetsya_chto_ya_slishkom_mnogo_rabotayu', // «Мне кажется, что я слишком много работаю.»
    16 => 'mbi_exh_q16_mne_khochetsya_uedinit_sya_i_otdokhnut_ot_vsego', // «Мне хочется уединиться и отдохнуть от всего и всех.»
    20 => 'mbi_exh_q20_ya_chuvstvuyu_ravnodushie_i_poteryu_interesa' // «Я чувствую равнодушие и потерю интереса ко многому, что радовало меня в моей работе.»
];

// Цинизм/Деперсонализация (5 items): 5, 10, 11, 15, 22
$MBI_CYNICISM_KEYS = [
    5 => 'mbi_cyn_q5_ya_chuvstvuyu_chto_obshchayus_s_nekotorymi_podchinennymi', // «Я чувствую, что общаюсь с некоторыми подчиненными и коллегами как с предметами (без теплоты и расположения к ним).»
    10 => 'mbi_cyn_q10_v_poslednee_vremya_ya_stal_bolee_cherstvym_po_otnosheniyu', // «В последнее время я стал более "черствым" по отношению к тем, с кем работаю.»
    11 => 'mbi_cyn_q11_ya_zamechayu_chto_moya_rabota_ozhestochaet_menya', // «Я замечаю, что моя работа ожесточает меня.»
    15 => 'mbi_cyn_q15_byvaet_chto_mne_deistvitelno_bezrazlichno_to', // «Бывает, что мне действительно безразлично то, что происходит с некоторыми моими подчиненными и коллегами.»
    22 => 'mbi_cyn_q22_v_poslednee_vremya_mne_kazhetsya_chto_kollegi_i_podchinennye' // «В последнее время мне кажется, что коллеги и подчиненные все чаще перекладывают на меня груз своих проблем и обязанностей.»
];

// Профессиональная эффективность (8 items, инвертируются): 4, 7, 9, 12, 17, 18, 19, 21
$MBI_EFFICACY_KEYS = [
    4 => 'mbi_eff_q4_ya_khorosho_ponimayu_chto_chuvstvui_moi_podchinennye', // «Я хорошо понимаю, что чувствуют мои подчиненные и коллеги, и стараюсь учитывать это в интересах дела.»
    7 => 'mbi_eff_q7_ya_umeyu_nakhodit_pravilnoe_reshenie_v_konfliktnykh_situatsiyakh', // «Я умею находить правильное решение в конфликтных ситуациях, возникающих при общении с коллегами.»
    9 => 'mbi_eff_q9_ya_uveren_chto_moya_rabota_nuzhna_lyudyam', // «Я уверен, что моя работа нужна людям.»
    12 => 'mbi_eff_q12_u_menya_mnogo_planov_na_budushchee_i_ya_veru', // «У меня много планов на будущее, и я верю в их осуществление.»
    17 => 'mbi_eff_q17_ya_legko_mogu_sozdat_atmosferu_dobrozhelatelnosti_i_sotrudnichestva', // «Я легко могу создать атмосферу доброжелательности и сотрудничества в коллективе.»
    18 => 'mbi_eff_q18_vo_vremya_raboty_ya_chuvstvuyu_priyatnoe_ozhivlenie', // «Во время работы я чувствую приятное оживление.»
    19 => 'mbi_eff_q19_blagodarya_svoei_rabote_ya_uzhe_sdelal_v_zhizni', // «Благодаря своей работе я уже сделал в жизни много действительно ценного.»
    21 => 'mbi_eff_q21_na_rabote_ya_spokoyno_spravlayus_s_emotsionalnymi_problemami' // «На работе я спокойно справляюсь с эмоциональными проблемами.»
];

// ============================================================================
// Прокрастинация - 8 items
// ============================================================================
$PROCRASTINATION_KEYS = [
    1 => 'proc_q1_ya_chasto_zamechayu_chto_zaranee_vypolnyayu_zadachi', // «Я часто замечаю, что заранее выполняю задачи, которые я собирался сделать в будущем»
    2 => 'proc_q2_ya_zamechayu_chto_rabota_ne_vypolnyaetsya_v_techenie_neskolkih_dnei', // «Я замечаю, что работа не выполняется в течение нескольких дней, даже если она не требует большего, чем просто сесть и сделать ее»
    3 => 'proc_q3_kogda_mne_nuzhno_vypolnit_trudnuyu_rabotu_ya_zhdu', // «Когда мне нужно выполнить трудную работу, я жду прилив вдохновения»
    4 => 'proc_q4_mne_obychno_prikhoditsya_speshit_chtoby_vypolnit_zadachi_vovremya', // «Мне обычно приходится спешить, чтобы выполнить задачи вовремя»
    5 => 'proc_q5_kogda_priblizhayutsya_sroki_zaversheniya_raboty_ya_chasto_teruyu', // «Когда приближаются сроки завершения работы, я часто теряю время, занимаясь другими делами»
    6 => 'proc_q6_pri_podgotovke_k_vstreche_ya_chasto_lovlyu_sebya_na_tom', // «При подготовке ко встрече я часто ловлю себя на том, что делаю что-то в последнюю минуту»
    7 => 'proc_q7_chasto_ya_dolgo_ne_prystupayu_k_vypolneniyu_zadachi_kotoruyu_mne_nuzhno', // «Часто я долго не приступаю к выполнению задачи, которую мне нужно решить»
    8 => 'proc_q8_ya_chasto_govoryu_sdelayu_eto_zavtra' // «Я часто говорю: "Сделаю это завтра"»
];

// ============================================================================
// Джедайские практики - 21 practice (частота + качество)
// ============================================================================
$PRACTICES_KEYS = [
    1 => 'prac_q1_lezha_v_posteli_ne_polzovatsya_ustroistvami',
    2 => 'prac_q2_razgruzit_svoyu_pamyat',
    3 => 'prac_q3_otdelyat_zadachi_ot_proektov',
    4 => 'prac_q4_delat_zadachi_tolko_zapisannye',
    5 => 'prac_q5_formulirovat_zadachi_kak_dlya_obezyanki',
    6 => 'prac_q6_sostavit_plan_na_den',
    7 => 'prac_q7_skhodit_na_trenirovku',
    8 => 'prac_q8_podvodit_itogi_dnya',
    9 => 'prac_q9_sostavit_plan_na_nedelyu',
    10 => 'prac_q10_sveritsya_otkorrektirovat_plan_na_nedelyu',
    11 => 'prac_q11_podvesti_itogi_nedeli',
    12 => 'prac_q12_poradovat_svoyu_obezyanku',
    13 => 'prac_q13_uvidet_dno_inboksa',
    14 => 'prac_q14_vypolnit_zadachu_do_chatov',
    15 => 'prac_q15_sootnesti_zadachi_s_tsyelyami',
    16 => 'prac_q16_vypolnit_regulyarnuyu_praktiku',
    17 => 'prac_q17_vypolnit_zelenuyu_zadachu',
    18 => 'prac_q18_zelenaya_zadacha_do_chatov',
    19 => 'prac_q19_inkubator_idey',
    20 => 'prac_q20_15_minut_naedine_s_myslyami'
];

// ============================================================================
// Джедайские вакцины - 5 items
// ============================================================================
$VACCINES_KEYS = [
    1 => 'vacc_q1_udalit_prilozheniya_sotsialnykh_setey', // «Удалить приложения социальных сетей со своего смартфона»
    2 => 'vacc_q2_otklyuchit_krasnye_kruzhochki_na_ikonkakh', // «Отключить красные кружочки на иконках приложений»
    3 => 'vacc_q3_otklyuchit_opoveshcheniya_o_novykh_soobshcheniyakh_v_chatakh', // «Отключить оповещения о новых сообщениях в чатах»
    4 => 'vacc_q4_otklyuchit_opoveshcheniya_o_novykh_soobshcheniyakh_v_pochte', // «Отключить оповещения о новых сообщениях в почте»
    5 => 'vacc_q5_otklyuchit_push_opoveshcheniya_ot_prilozheniy' // «Отключить push-оповещения от приложений (соцсети, магазины, заправки)»
];

// ============================================================================
// Функция для получения ключей по шкале
// ============================================================================

/**
 * Получить ассоциативный массив ключей для шкалы
 * 
 * @param string $scale Название шкалы (mijs, swls, mbi_exh, mbi_cyn, mbi_eff, proc, prac, vacc)
 * @return array
 */
function getScaleKeys($scale) {
    global $MIJS_KEYS, $SWLS_KEYS, $MBI_EXHAUSTION_KEYS, $MBI_CYNICISM_KEYS, $MBI_EFFICACY_KEYS, 
           $PROCRASTINATION_KEYS, $PRACTICES_KEYS, $VACCINES_KEYS;
    
    $maps = [
        'mijs' => $MIJS_KEYS,
        'swls' => $SWLS_KEYS,
        'mbi_exh' => $MBI_EXHAUSTION_KEYS,
        'mbi_cyn' => $MBI_CYNICISM_KEYS,
        'mbi_eff' => $MBI_EFFICACY_KEYS,
        'proc' => $PROCRASTINATION_KEYS,
        'prac' => $PRACTICES_KEYS,
        'vacc' => $VACCINES_KEYS
    ];
    
    return $maps[$scale] ?? [];
}

/**
 * Преобразовать русский текст в транслит
 * 
 * @param string $text Исходный текст
 * @return string Транслитерированный текст
 */
function translit($text) {
    $converter = [
        'а' => 'a',    'б' => 'b',    'в' => 'v',    'г' => 'g',    'д' => 'd',
        'е' => 'e',    'ё' => 'yo',   'ж' => 'zh',   'з' => 'z',    'и' => 'i',
        'й' => 'y',    'к' => 'k',    'л' => 'l',    'м' => 'm',    'н' => 'n',
        'о' => 'o',    'п' => 'p',    'р' => 'r',    'с' => 's',    'т' => 't',
        'у' => 'u',    'ф' => 'f',    'х' => 'kh',   'ц' => 'ts',   'ч' => 'ch',
        'ш' => 'sh',   'щ' => 'shch', 'ъ' => '',     'ы' => 'y',    'ь' => '',
        'э' => 'e',    'ю' => 'yu',   'я' => 'ya',
        ' ' => '_',    '.' => '',     ',' => '',     '!' => '',     '?' => '',
        '«' => '',     '»' => '',     '"' => '',     "'" => '',     ':' => '',
        ';' => '',     '(' => '',     ')' => '',     '-' => '',
    ];
    
    $text = mb_strtolower($text, 'UTF-8');
    $result = '';
    
    for ($i = 0; $i < mb_strlen($text, 'UTF-8'); $i++) {
        $char = mb_substr($text, $i, 1, 'UTF-8');
        $result .= $converter[$char] ?? $char;
    }
    
    // Удаляем повторяющиеся подчеркивания
    $result = preg_replace('/_+/', '_', $result);
    $result = trim($result, '_');
    
    return $result;
}

/**
 * Сгенерировать ключ для ответа
 * 
 * @param string $prefix Префикс шкалы (mijs, swls, mbi_exh, и т.д.)
 * @param int $questionNumber Номер вопроса
 * @param string $questionText Текст вопроса (для генерации транслита)
 * @return string
 */
function generateKey($prefix, $questionNumber, $questionText) {
    // Берем первые 4 слова
    $words = explode(' ', $questionText);
    $firstFourWords = implode(' ', array_slice($words, 0, 4));
    
    // Транслитерируем
    $translit = translit($firstFourWords);
    
    // Формируем ключ
    return "{$prefix}_q{$questionNumber}_{$translit}";
}
