<?php
/**
 * API пульс-опросов — сохранение ответа
 */
error_reporting(E_ALL);
ini_set('display_errors', 1);

header('Content-Type: application/json; charset=utf-8');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: POST, OPTIONS');
header('Access-Control-Allow-Headers: Content-Type');

if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') { http_response_code(200); exit; }

require_once 'db_config.php';

try {
    $pdo = getDbConnection();
} catch (Exception $e) {
    http_response_code(500);
    echo json_encode(['error' => 'Ошибка подключения к БД']);
    exit;
}

$sessionId = $_POST['session_id'] ?? '';
if (empty($sessionId)) {
    http_response_code(400);
    echo json_encode(['error' => 'Отсутствует session_id']);
    exit;
}

// Проверяем дубликат
$stmt = $pdo->prepare("SELECT id FROM pulse_responses_q4_2026 WHERE session_id = :sid");
$stmt->execute([':sid' => $sessionId]);
if ($stmt->fetch()) {
    http_response_code(409);
    echo json_encode(['error' => 'Этот ответ уже сохранён']);
    exit;
}

// Данные
$tgId = isset($_POST['tg_id']) && $_POST['tg_id'] !== '' ? $_POST['tg_id'] : null;
$week = isset($_POST['week']) && $_POST['week'] !== '' ? $_POST['week'] : null;
$groupId = isset($_POST['group_id']) && $_POST['group_id'] !== '' ? $_POST['group_id'] : null;
// Пол участника: s=m (мальчики, по умолчанию) / s=f (девочки)
$sex = isset($_POST['sex']) && $_POST['sex'] !== '' ? $_POST['sex'] : null;
$reactive = (int)($_POST['cubes_reactive'] ?? 0);
$proactive = (int)($_POST['cubes_proactive'] ?? 0);
$operational = (int)($_POST['cubes_operational'] ?? 0);
$pool = (int)($_POST['cubes_pool'] ?? 0);

$representative = isset($_POST['representative']) ? $_POST['representative'] : null;
$workLife = isset($_POST['work_life']) ? $_POST['work_life'] : null;
$satisfaction = isset($_POST['satisfaction']) ? $_POST['satisfaction'] : null;

// SIMEA: одно-пунктовая пиктограммная шкала энергии (Weigelt et al., 2022), 1-7
$simea = isset($_POST['simea']) && $_POST['simea'] !== '' ? $_POST['simea'] : null;

// Short PANAS (10-item, Q4 2026)
$panasPa1 = isset($_POST['panas_pa_1']) && $_POST['panas_pa_1'] !== '' ? $_POST['panas_pa_1'] : null;
$panasPa2 = isset($_POST['panas_pa_2']) && $_POST['panas_pa_2'] !== '' ? $_POST['panas_pa_2'] : null;
$panasPa3 = isset($_POST['panas_pa_3']) && $_POST['panas_pa_3'] !== '' ? $_POST['panas_pa_3'] : null;
$panasPa4 = isset($_POST['panas_pa_4']) && $_POST['panas_pa_4'] !== '' ? $_POST['panas_pa_4'] : null;
$panasPa5 = isset($_POST['panas_pa_5']) && $_POST['panas_pa_5'] !== '' ? $_POST['panas_pa_5'] : null;
$positiveAffect = isset($_POST['positive_affect']) && $_POST['positive_affect'] !== '' ? $_POST['positive_affect'] : null;
$panasNa1 = isset($_POST['panas_na_1']) && $_POST['panas_na_1'] !== '' ? $_POST['panas_na_1'] : null;
$panasNa2 = isset($_POST['panas_na_2']) && $_POST['panas_na_2'] !== '' ? $_POST['panas_na_2'] : null;
$panasNa3 = isset($_POST['panas_na_3']) && $_POST['panas_na_3'] !== '' ? $_POST['panas_na_3'] : null;
$panasNa4 = isset($_POST['panas_na_4']) && $_POST['panas_na_4'] !== '' ? $_POST['panas_na_4'] : null;
$panasNa5 = isset($_POST['panas_na_5']) && $_POST['panas_na_5'] !== '' ? $_POST['panas_na_5'] : null;
$negativeAffect = isset($_POST['negative_affect']) && $_POST['negative_affect'] !== '' ? $_POST['negative_affect'] : null;

// Производные метрики эмоционального фона (club.mnogosdelal.ru/post/3289):
//   emotion_intensity = max(PA_norm, NA_norm) * 100, где norm = (SUM - 5) / 20
//   positivity_percent = atan2(PA_norm, NA_norm) / (pi/2) * 100
$emotionIntensity = isset($_POST['emotion_intensity']) && $_POST['emotion_intensity'] !== '' ? $_POST['emotion_intensity'] : null;
$positivityPercent = isset($_POST['positivity_percent']) && $_POST['positivity_percent'] !== '' ? $_POST['positivity_percent'] : null;

$takeaway = trim($_POST['takeaway'] ?? '');
$comment = trim($_POST['comment'] ?? '');
$timeTotal = isset($_POST['time_total']) ? $_POST['time_total'] : null;

$ipHash = hash('sha256', $_SERVER['REMOTE_ADDR'] ?? '');
$ua = $_SERVER['HTTP_USER_AGENT'] ?? '';
$device = preg_match('/Mobile|Android|iPhone|iPad/i', $ua) ? 'mobile' : 'desktop';

$stmt = $pdo->prepare("
    INSERT INTO pulse_responses_q4_2026 (
        session_id, status, tg_id, week, group_id, sex,
        cubes_reactive, cubes_proactive, cubes_operational, cubes_pool,
        simea,
        panas_pa_1, panas_pa_2, panas_pa_3, panas_pa_4, panas_pa_5, positive_affect,
        panas_na_1, panas_na_2, panas_na_3, panas_na_4, panas_na_5, negative_affect,
        emotion_intensity, positivity_percent,
        representative, work_life, satisfaction,
        takeaway, comment, time_total,
        user_agent, ip_hash, device_type
    ) VALUES (
        :sid, 'completed', :tg_id, :week, :group_id, :sex,
        :r, :g, :o, :p,
        :simea,
        :pa1, :pa2, :pa3, :pa4, :pa5, :pa,
        :na1, :na2, :na3, :na4, :na5, :na,
        :ei, :pp,
        :rep, :wl, :sat,
        :takeaway, :comment, :tt,
        :ua, :ip, :device
    )
");

$stmt->execute([
    ':sid' => $sessionId,
    ':tg_id' => $tgId, ':week' => $week, ':group_id' => $groupId, ':sex' => $sex,
    ':r' => $reactive, ':g' => $proactive, ':o' => $operational, ':p' => $pool,
    ':simea' => $simea,
    ':pa1' => $panasPa1, ':pa2' => $panasPa2, ':pa3' => $panasPa3, ':pa4' => $panasPa4, ':pa5' => $panasPa5, ':pa' => $positiveAffect,
    ':na1' => $panasNa1, ':na2' => $panasNa2, ':na3' => $panasNa3, ':na4' => $panasNa4, ':na5' => $panasNa5, ':na' => $negativeAffect,
    ':ei' => $emotionIntensity, ':pp' => $positivityPercent,
    ':rep' => $representative, ':wl' => $workLife, ':sat' => $satisfaction,
    ':takeaway' => ($takeaway !== '' ? $takeaway : null),
    ':comment' => ($comment !== '' ? $comment : null),
    ':tt' => ($timeTotal !== null && $timeTotal !== '' ? $timeTotal : null),
    ':ua' => $ua, ':ip' => $ipHash, ':device' => $device
]);

// Сохранение в Google Sheets (не блокирует ответ, ошибки игнорируем)
ob_start();
try {
    require_once 'save_to_sheets.php';
    saveToGoogleSheets([
        'timestamp' => date('Y-m-d H:i:s'),
        'session_id' => $sessionId,
        'tg_id' => $tgId,
        'week' => $week,
        'group_id' => $groupId,
        'sex' => $sex,
        'cubes_reactive' => $reactive,
        'cubes_proactive' => $proactive,
        'cubes_operational' => $operational,
        'cubes_pool' => $pool,
        'simea' => $simea,
        'panas_pa_1' => $panasPa1,
        'panas_pa_2' => $panasPa2,
        'panas_pa_3' => $panasPa3,
        'panas_pa_4' => $panasPa4,
        'panas_pa_5' => $panasPa5,
        'positive_affect' => $positiveAffect,
        'panas_na_1' => $panasNa1,
        'panas_na_2' => $panasNa2,
        'panas_na_3' => $panasNa3,
        'panas_na_4' => $panasNa4,
        'panas_na_5' => $panasNa5,
        'negative_affect' => $negativeAffect,
        'emotion_intensity' => $emotionIntensity,
        'positivity_percent' => $positivityPercent,
        'satisfaction' => $satisfaction,
        'representative' => $representative,
        'work_life' => $workLife,
        'takeaway' => $takeaway,
        'comment' => $comment,
        'time_total' => $timeTotal,
        'device_type' => $device
    ]);
} catch (Exception $e) {
    // Игнорируем ошибки Google Sheets
}
ob_end_clean();

echo json_encode(['success' => true]);
