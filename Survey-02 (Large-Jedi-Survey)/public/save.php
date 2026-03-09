<?php
/**
 * AJAX обработчик сохранения ответов
 *
 * Принимает POST запрос с ответами и сохраняет в БД
 */

// Старт сессии
session_start();

require_once __DIR__ . '/../config/config.php';
require_once __DIR__ . '/../config/variable_keys.php';
require_once __DIR__ . '/../src/Database.php';
require_once __DIR__ . '/../src/Survey.php';
require_once __DIR__ . '/../src/Calculator.php';

header('Content-Type: application/json');

// ============================================================================
// ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
// ============================================================================

/**
 * Сохранить ответы шкалы с расширенными ключами
 *
 * @param array $answers Массив ответов (будет дополнен)
 * @param array $items Ответы в виде массива значений
 * @param array $keysMap Ассоциативный массив ключей (номер => ключ)
 * @return void
 */
function saveScaleWithKeys(&$answers, $items, $keysMap) {
    foreach ($items as $index => $value) {
        $questionNumber = $index + 1;
        $key = $keysMap[$questionNumber] ?? "q{$questionNumber}";
        $answers[$key] = (int)$value;
    }
}

/**
 * Преобразовать массив ответов в объект с ключами
 * @param array $items Массив значений [1,2,3,...]
 * @param array $keysMap Ассоциативный массив ключей
 * @return array Объект {key1: value1, key2: value2, ...}
 */
function itemsToAssocArray($items, $keysMap) {
    $result = [];
    foreach ($items as $index => $value) {
        $questionNumber = $index + 1;
        $key = $keysMap[$questionNumber] ?? "q{$questionNumber}";
        $result[$key] = (int)$value;
    }
    return $result;
}

// ============================================================================
// ОСНОВНОЙ КОД
// ============================================================================

// Проверяем метод
if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    echo json_encode(['success' => false, 'error' => 'Invalid method']);
    exit;
}

// Получаем данные
$input = $_POST;

if (!$input) {
    echo json_encode(['success' => false, 'error' => 'No input data']);
    exit;
}

$respondent_id = $input['respondent_id'] ?? null;
$page = (int)($input['page'] ?? 0);
$next_page = (int)($input['next_page'] ?? $page + 1);

if (!$respondent_id) {
    echo json_encode(['success' => false, 'error' => 'No respondent ID']);
    exit;
}

try {
    // Собираем ответы в зависимости от страницы
    $answers = [];

    switch ($page) {
        case 0:
            $answers['consent_given'] = isset($input['consent_given']) ? 1 : 0;
            break;

        case 1:
            $answers['age'] = (int)($input['age'] ?? 0);
            $answers['gender'] = $input['gender'] ?? '';
            $answers['children_count'] = (int)($input['children_count'] ?? 0);
            $answers['position'] = $input['position'] ?? '';

            $not_working = in_array($answers['position'], ['unemployed', 'pensioner'], true);
            if (!$not_working) {
                $answers['work_experience_years'] = (int)($input['work_experience_years'] ?? 0);
                $answers['industry'] = $input['industry'] ?? '';
                $answers['industry_other'] = $input['industry_other'] ?? '';
                $answers['remote_days'] = $input['remote_days'] ?? '';
            } else {
                $answers['work_experience_years'] = null;
                $answers['industry'] = null;
                $answers['industry_other'] = null;
                $answers['remote_days'] = null;
            }

            $answers['mindset_technical_humanitarian'] = (int)($input['mindset_technical_humanitarian'] ?? 0);
            $answers['tool_preference'] = (int)($input['tool_preference'] ?? 0);
            break;

        case 2:
            $answers['personal_urgent_important'] = (int)($input['personal_urgent_important'] ?? 0);
            break;

        case 3:
            $answers['work_urgent_important'] = (int)($input['work_urgent_important'] ?? 0);
            $answers['work_satisfaction'] = (int)($input['work_satisfaction'] ?? 0);
            break;

        case 4:
            $mijs_items = json_decode($input['mijs_items'] ?? '[]', true);
            $mijs_assoc = itemsToAssocArray($mijs_items, $MIJS_KEYS);
            $answers['mijs_items'] = json_encode($mijs_assoc, JSON_UNESCAPED_UNICODE);

            $calculated = Calculator::calculateMijs($mijs_items);
            $answers['mijs_urgency_score'] = $calculated['urgency'];
            $answers['mijs_agency_score'] = $calculated['agency'];
            $answers['mijs_total'] = $calculated['total'];
            break;

        case 5:
            $swls_items = json_decode($input['swls_items'] ?? '[]', true);
            $swls_assoc = itemsToAssocArray($swls_items, $SWLS_KEYS);
            $answers['swls_items'] = json_encode($swls_assoc, JSON_UNESCAPED_UNICODE);
            $answers['swls_total'] = Calculator::calculateSwls($swls_items);
            break;

        case 6:
            $mbi_exhaustion = json_decode($input['mbi_exhaustion_items'] ?? '[]', true);
            $mbi_cynicism = json_decode($input['mbi_cynicism_items'] ?? '[]', true);
            $mbi_efficacy = json_decode($input['mbi_efficacy_items'] ?? '[]', true);

            $answers['mbi_exhaustion_items'] = json_encode(itemsToAssocArray($mbi_exhaustion, $MBI_EXHAUSTION_KEYS), JSON_UNESCAPED_UNICODE);
            $answers['mbi_cynicism_items'] = json_encode(itemsToAssocArray($mbi_cynicism, $MBI_CYNICISM_KEYS), JSON_UNESCAPED_UNICODE);
            $answers['mbi_efficacy_items'] = json_encode(itemsToAssocArray($mbi_efficacy, $MBI_EFFICACY_KEYS), JSON_UNESCAPED_UNICODE);

            $calculated = Calculator::calculateMbi($mbi_exhaustion, $mbi_cynicism, $mbi_efficacy);
            $answers['mbi_exhaustion_score'] = $calculated['exhaustion'];
            $answers['mbi_cynicism_score'] = $calculated['cynicism'];
            $answers['mbi_efficacy_score'] = $calculated['efficacy'];
            $answers['mbi_total'] = $calculated['total'];
            break;

        case 7:
            $procrastination = json_decode($input['procrastination_items'] ?? '[]', true);
            $proc_assoc = itemsToAssocArray($procrastination, $PROCRASTINATION_KEYS);
            $answers['procrastination_items'] = json_encode($proc_assoc, JSON_UNESCAPED_UNICODE);
            $answers['procrastination_total'] = Calculator::calculateProcrastination($procrastination);
            break;

        case 8:
            $frequency = json_decode($input['practices_frequency'] ?? '{}', true);
            $quality = json_decode($input['practices_quality'] ?? '{}', true);

            $freq_assoc = [];
            foreach ($frequency as $practiceNum => $value) {
                $key = $PRACTICES_KEYS[$practiceNum] ?? "prac_q{$practiceNum}";
                $freq_assoc[$key . '_freq'] = (int)$value;
            }
            $answers['practices_frequency'] = json_encode($freq_assoc, JSON_UNESCAPED_UNICODE);

            $qual_assoc = [];
            foreach ($quality as $practiceNum => $value) {
                $key = $PRACTICES_KEYS[$practiceNum] ?? "prac_q{$practiceNum}";
                $qual_assoc[$key . '_qual'] = (int)$value;
            }
            $answers['practices_quality'] = json_encode($qual_assoc, JSON_UNESCAPED_UNICODE);

            $answers['practices_freq_total'] = Calculator::calculatePracticesFreq($frequency);
            $answers['practices_quality_total'] = Calculator::calculatePracticesQuality($quality);
            break;

        case 9:
            $vaccines = json_decode($input['vaccines'] ?? '{}', true);
            $vacc_assoc = itemsToAssocArray($vaccines, $VACCINES_KEYS);
            $answers['vaccines'] = json_encode($vacc_assoc, JSON_UNESCAPED_UNICODE);
            $answers['vaccines_total'] = Calculator::calculateVaccines($vaccines);
            break;

        case 10:
            $answers['open_most_useful_practice'] = $input['open_most_useful_practice'] ?? '';
            $answers['open_other_practices'] = $input['open_other_practices'] ?? '';

            $respondent = Survey::getRespondent($input['respondent_id'] ?? '');
            if ($respondent && !empty($respondent['practices_frequency'])) {
                $frequency = json_decode($respondent['practices_frequency'], true);
                $quality = json_decode($respondent['practices_quality'], true);

                $practice_16_freq = $frequency['16'] ?? 0;
                $practice_16_quality = $quality['16'] ?? 0;

                $answers['attention_check_passed'] = ($practice_16_freq == 6 && $practice_16_quality == 4) ? 1 : 0;
                $answers['attention_check_freq_answer'] = $practice_16_freq;
                $answers['attention_check_quality_answer'] = $practice_16_quality;
            }
            break;
    }

    $time_spent = Survey::calculateTimeSpent($respondent_id);
    $answers['time_spent_seconds'] = $time_spent;

    log_event("Page $page saving for respondent $respondent_id", 'DEBUG');

    // Сохраняем ответы
    $success = Survey::savePage($respondent_id, $page, $answers);

    log_event("Page $page save result: " . ($success ? 'SUCCESS' : 'FAILED'), 'DEBUG');

    // Возвращаем URL следующей страницы
    echo json_encode([
        'success' => $success,
        'redirect' => "index.php?page=$next_page",
        'respondent_id' => $respondent_id
    ]);

} catch (\Throwable $e) {
    log_event("Save error: " . $e->getMessage(), 'ERROR');
    echo json_encode(['success' => false, 'error' => $e->getMessage()]);
}
