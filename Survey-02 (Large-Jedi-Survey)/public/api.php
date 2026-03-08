<?php
/**
 * API для получения результатов респондента по коду
 *
 * GET /api.php?code=ABC123
 */

require_once __DIR__ . '/../config/config.php';
require_once __DIR__ . '/../config/variable_keys.php';
require_once __DIR__ . '/../src/Database.php';
require_once __DIR__ . '/../src/Survey.php';

header('Content-Type: application/json');

// Получаем код из запроса
$code = $_GET['code'] ?? null;

if (!$code) {
    http_response_code(400);
    echo json_encode([
        'success' => false,
        'error' => 'Code parameter is required'
    ]);
    exit;
}

try {
    // Ищем респондента по коду
    $respondent = Survey::getRespondentByCode($code);

    if (!$respondent) {
        http_response_code(404);
        echo json_encode([
            'success' => false,
            'error' => 'Респондент с кодом ' . htmlspecialchars($code) . ' не найден'
        ]);
        exit;
    }

    // Получаем детализированные ответы
    $detailed_responses = !empty($respondent['responses_detailed']) 
        ? json_decode($respondent['responses_detailed'], true) 
        : null;

    // Формируем ответ
    $response = [
        'success' => true,
        'respondent' => [
            'id' => $respondent['id'],
            'code' => $respondent['code'],
            'created_at' => $respondent['created_at'],
            'completed_at' => $respondent['completed_at'],
            'time_spent_seconds' => $respondent['time_spent_seconds']
        ],
        'demographics' => [
            'age' => $respondent['age'],
            'gender' => $respondent['gender'],
            'children_count' => $respondent['children_count'],
            'work_experience_years' => $respondent['work_experience_years'],
            'position' => $respondent['position'],
            'industry' => $respondent['industry'],
            'industry_other' => $respondent['industry_other'] ?? null,
            'remote_days' => $respondent['remote_days'],
            'mindset_technical_humanitarian' => $respondent['mindset_technical_humanitarian'],
            'tool_preference' => $respondent['tool_preference']
        ],
        'scores' => [
            'mijs_total' => $respondent['mijs_total'],
            'mijs_urgency' => $respondent['mijs_urgency_score'],
            'mijs_agency' => $respondent['mijs_agency_score'],
            'mbi_exhaustion' => $respondent['mbi_exhaustion_score'],
            'mbi_cynicism' => $respondent['mbi_cynicism_score'],
            'mbi_efficacy' => $respondent['mbi_efficacy_score'],
            'mbi_total' => $respondent['mbi_total'],
            'swls_total' => $respondent['swls_total'],
            'procrastination_total' => $respondent['procrastination_total'],
            'practices_freq_total' => $respondent['practices_freq_total'],
            'practices_quality_total' => $respondent['practices_quality_total'],
            'vaccines_total' => $respondent['vaccines_total'],
            'work_urgent_important' => $respondent['work_urgent_important'],
            'personal_urgent_important' => $respondent['personal_urgent_important'],
            'work_satisfaction' => $respondent['work_satisfaction']
        ],
        'answers' => [
            'open_most_useful_practice' => $respondent['open_most_useful_practice'],
            'open_other_practices' => $respondent['open_other_practices']
        ],
        'detailed_responses' => $detailed_responses
    ];

    echo json_encode($response, JSON_UNESCAPED_UNICODE);

} catch (Exception $e) {
    log_event("API error: " . $e->getMessage(), 'ERROR');
    http_response_code(500);
    echo json_encode([
        'success' => false,
        'error' => 'Internal server error'
    ]);
}
