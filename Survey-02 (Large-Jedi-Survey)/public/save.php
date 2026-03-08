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

// Проверяем метод
if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    echo json_encode(['success' => false, 'error' => 'Invalid method']);
    exit;
}

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
        // Индекс в массиве начинается с 0, но номера вопросов с 1
        $questionNumber = $index + 1;
        $key = $keysMap[$questionNumber] ?? "q{$questionNumber}";
        $answers[$key] = (int)$value;
    }
}

// ============================================================================
// ОСНОВНОЙ КОД
// ============================================================================

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

// Проверяем существование респондента
$respondent = Survey::getRespondent($respondent_id);
log_event("Save page $page: respondent_id=$respondent_id, found=" . ($respondent ? 'yes' : 'no'));

// Если не найден - создаём нового (БД могла быть очищена, сессия истекла)
if (!$respondent) {
    log_event("WARNING: Respondent $respondent_id not found in DB, creating new one", 'WARNING');
    $respondent_id = Survey::createRespondent();
    $_SESSION['respondent_id'] = $respondent_id;
    log_event("Created new respondent after DB loss: $respondent_id");
}

try {
    // Получаем существующие detailed responses из БД
    $detailed_responses = [];
    if ($respondent && !empty($respondent['responses_detailed'])) {
        $detailed_responses = json_decode($respondent['responses_detailed'], true) ?? [];
    }
    
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
            
            // Если работает - сохраняем данные о работе
            $not_working = in_array($answers['position'], ['unemployed', 'pensioner'], true);
            if (!$not_working) {
                $answers['work_experience_years'] = (int)($input['work_experience_years'] ?? 0);
                $answers['industry'] = $input['industry'] ?? '';
                $answers['industry_other'] = $input['industry_other'] ?? '';
                $answers['remote_days'] = $input['remote_days'] ?? '';
            } else {
                // Для неработающих ставим null
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
            $answers['mijs_items'] = json_encode($mijs_items);
            
            // Сохраняем ответы с ключами ТОЛЬКО в detailed responses (JSON)
            saveScaleWithKeys($detailed_responses, $mijs_items, $MIJS_KEYS);

            // Рассчитываем MIJS
            $calculated = Calculator::calculateMijs($mijs_items);
            $answers['mijs_urgency_score'] = $calculated['urgency'];
            $answers['mijs_agency_score'] = $calculated['agency'];
            $answers['mijs_total'] = $calculated['total'];
            break;

        case 5:
            $swls_items = json_decode($input['swls_items'] ?? '[]', true);
            $answers['swls_items'] = json_encode($swls_items);
            
            // Сохраняем ответы с ключами ТОЛЬКО в detailed responses (JSON)
            saveScaleWithKeys($detailed_responses, $swls_items, $SWLS_KEYS);
            
            $answers['swls_total'] = Calculator::calculateSwls($swls_items);
            break;

        case 6:
            $mbi_exhaustion = json_decode($input['mbi_exhaustion_items'] ?? '[]', true);
            $mbi_cynicism = json_decode($input['mbi_cynicism_items'] ?? '[]', true);
            $mbi_efficacy = json_decode($input['mbi_efficacy_items'] ?? '[]', true);

            $answers['mbi_exhaustion_items'] = json_encode($mbi_exhaustion);
            $answers['mbi_cynicism_items'] = json_encode($mbi_cynicism);
            $answers['mbi_efficacy_items'] = json_encode($mbi_efficacy);
            
            // Для MBI используем специальные ключи с оригинальными номерами вопросов
            // Массивы ключей используют оригинальные номера (1,2,3,6,8...), а не индексы
            $mbiExhaustionKeys = [1, 2, 3, 6, 8, 13, 14, 16, 20];
            $mbiCynicismKeys = [5, 10, 11, 15, 22];
            $mbiEfficacyKeys = [4, 7, 9, 12, 17, 18, 19, 21];
            
            // Сохраняем с ключами для exhaustion
            foreach ($mbiExhaustionKeys as $index => $questionNum) {
                $key = $MBI_EXHAUSTION_KEYS[$questionNum] ?? "mbi_exh_q{$questionNum}";
                $value = $mbi_exhaustion[$index] ?? 0;
                $detailed_responses[$key] = (int)$value;
            }
            
            // Сохраняем с ключами для cynicism
            foreach ($mbiCynicismKeys as $index => $questionNum) {
                $key = $MBI_CYNICISM_KEYS[$questionNum] ?? "mbi_cyn_q{$questionNum}";
                $value = $mbi_cynicism[$index] ?? 0;
                $detailed_responses[$key] = (int)$value;
            }
            
            // Сохраняем с ключами для efficacy
            foreach ($mbiEfficacyKeys as $index => $questionNum) {
                $key = $MBI_EFFICACY_KEYS[$questionNum] ?? "mbi_eff_q{$questionNum}";
                $value = $mbi_efficacy[$index] ?? 0;
                $detailed_responses[$key] = (int)$value;
            }

            // Рассчитываем MBI
            $calculated = Calculator::calculateMbi($mbi_exhaustion, $mbi_cynicism, $mbi_efficacy);
            $answers['mbi_exhaustion_score'] = $calculated['exhaustion'];
            $answers['mbi_cynicism_score'] = $calculated['cynicism'];
            $answers['mbi_efficacy_score'] = $calculated['efficacy'];
            $answers['mbi_total'] = $calculated['total'];
            break;

        case 7:
            $procrastination = json_decode($input['procrastination_items'] ?? '[]', true);
            $answers['procrastination_items'] = json_encode($procrastination);
            
            // Сохраняем ответы с ключами ТОЛЬКО в detailed responses (JSON)
            saveScaleWithKeys($detailed_responses, $procrastination, $PROCRASTINATION_KEYS);
            
            $answers['procrastination_total'] = Calculator::calculateProcrastination($procrastination);
            break;

        case 8:
            $frequency = json_decode($input['practices_frequency'] ?? '{}', true);
            $quality = json_decode($input['practices_quality'] ?? '{}', true);

            $answers['practices_frequency'] = json_encode($frequency);
            $answers['practices_quality'] = json_encode($quality);
            
            // Сохраняем ответы с ключами ТОЛЬКО в detailed responses (JSON)
            foreach ($frequency as $practiceNum => $value) {
                $key = $PRACTICES_KEYS[$practiceNum] ?? "prac_q{$practiceNum}";
                $detailed_responses[$key . '_freq'] = (int)$value;
            }
            
            foreach ($quality as $practiceNum => $value) {
                $key = $PRACTICES_KEYS[$practiceNum] ?? "prac_q{$practiceNum}";
                $detailed_responses[$key . '_qual'] = (int)$value;
            }
            
            $answers['practices_freq_total'] = Calculator::calculatePracticesFreq($frequency);
            $answers['practices_quality_total'] = Calculator::calculatePracticesQuality($quality);
            break;

        case 9:
            $vaccines = json_decode($input['vaccines'] ?? '{}', true);
            $answers['vaccines'] = json_encode($vaccines);
            
            // Сохраняем ответы с ключами ТОЛЬКО в detailed responses (JSON)
            foreach ($vaccines as $vaccNum => $value) {
                $key = $VACCINES_KEYS[$vaccNum] ?? "vacc_q{$vaccNum}";
                $detailed_responses[$key] = (int)$value;
            }
            
            $answers['vaccines_total'] = Calculator::calculateVaccines($vaccines);
            break;
            
        case 10:
            $answers['open_most_useful_practice'] = $input['open_most_useful_practice'] ?? '';
            $answers['open_other_practices'] = $input['open_other_practices'] ?? '';
            
            // Добавляем открытые ответы в detailed responses для бэкапа
            $detailed_responses['open_most_useful_practice'] = $answers['open_most_useful_practice'];
            $detailed_responses['open_other_practices'] = $answers['open_other_practices'];

            // Attention check (практика 16) - данные уже сохранены на странице 8
            // Получаем их из БД
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
    
    // Сохраняем detailed responses в JSON поле
    $answers['responses_detailed'] = json_encode($detailed_responses);

    // Рассчитываем время прохождения
    $time_spent = Survey::calculateTimeSpent($respondent_id);
    $answers['time_spent_seconds'] = $time_spent;

    // ========================================================================
    // ЛОГИРОВАНИЕ ОТВЕТОВ ДЛЯ ВОССТАНОВЛЕНИЯ БАЗЫ
    // Логируем каждую страницу для полного резервного копирования
    // ========================================================================
    $logData = [
        'respondent_id' => $respondent_id,
        'page' => $page,
        'timestamp' => date('Y-m-d H:i:s'),
        'answers' => $answers,
        'detailed_responses' => $detailed_responses,
        'input_raw' => $input
    ];
    
    // Для страницы 10 добавляем открытые ответы в отдельное поле для удобства
    if ($page == 10) {
        $logData['open_answers'] = [
            'open_most_useful_practice' => $answers['open_most_useful_practice'] ?? '',
            'open_other_practices' => $answers['open_other_practices'] ?? ''
        ];
    }
    
    log_event("BACKUP: " . json_encode($logData, JSON_UNESCAPED_UNICODE), 'BACKUP');
    // ========================================================================

    // Сохраняем ответы
    $success = Survey::savePage($respondent_id, $page, $answers);

    if ($success) {
        log_event("Page $page saved successfully for respondent $respondent_id");

        // Если завершил - логируем
        if ($page >= 10) {
            log_event("Respondent $respondent_id completed survey");
        }

        // Возвращаем URL следующей страницы
        echo json_encode([
            'success' => true,
            'redirect' => "index.php?page=$next_page",
            'respondent_id' => $respondent_id
        ]);
    } else {
        echo json_encode(['success' => false, 'error' => 'Failed to save']);
    }
    
} catch (\Throwable $e) {
    log_event("Save error: " . $e->getMessage(), 'ERROR');
    echo json_encode(['success' => false, 'error' => $e->getMessage()]);
}
