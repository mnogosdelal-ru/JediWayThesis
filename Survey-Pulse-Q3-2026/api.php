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
$stmt = $pdo->prepare("SELECT id FROM pulse_responses_q3_2026 WHERE session_id = :sid");
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
$reactive = (int)($_POST['cubes_reactive'] ?? 0);
$proactive = (int)($_POST['cubes_proactive'] ?? 0);
$operational = (int)($_POST['cubes_operational'] ?? 0);
$pool = (int)($_POST['cubes_pool'] ?? 0);

$representative = isset($_POST['representative']) ? $_POST['representative'] : null;
$workLife = isset($_POST['work_life']) ? $_POST['work_life'] : null;
$satisfaction = isset($_POST['satisfaction']) ? $_POST['satisfaction'] : null;

// PSS-4
$pss1 = isset($_POST['pss_1']) ? $_POST['pss_1'] : null;
$pss2 = isset($_POST['pss_2']) ? $_POST['pss_2'] : null;
$pss3 = isset($_POST['pss_3']) ? $_POST['pss_3'] : null;
$pss4 = isset($_POST['pss_4']) ? $_POST['pss_4'] : null;

$takeaway = trim($_POST['takeaway'] ?? '');
$comment = trim($_POST['comment'] ?? '');
$timeTotal = isset($_POST['time_total']) ? $_POST['time_total'] : null;

$ipHash = hash('sha256', $_SERVER['REMOTE_ADDR'] ?? '');
$ua = $_SERVER['HTTP_USER_AGENT'] ?? '';
$device = preg_match('/Mobile|Android|iPhone|iPad/i', $ua) ? 'mobile' : 'desktop';

$stmt = $pdo->prepare("
    INSERT INTO pulse_responses_q3_2026 (
        session_id, status, tg_id, week, group_id,
        cubes_reactive, cubes_proactive, cubes_operational, cubes_pool,
        representative, work_life, satisfaction,
        pss_1, pss_2, pss_3, pss_4,
        takeaway, comment, time_total,
        user_agent, ip_hash, device_type
    ) VALUES (
        :sid, 'completed', :tg_id, :week, :group_id,
        :r, :g, :o, :p,
        :rep, :wl, :sat,
        :pss1, :pss2, :pss3, :pss4,
        :takeaway, :comment, :tt,
        :ua, :ip, :device
    )
");

$stmt->execute([
    ':sid' => $sessionId,
    ':tg_id' => $tgId, ':week' => $week, ':group_id' => $groupId,
    ':r' => $reactive, ':g' => $proactive, ':o' => $operational, ':p' => $pool,
    ':rep' => $representative, ':wl' => $workLife, ':sat' => $satisfaction,
    ':pss1' => $pss1, ':pss2' => $pss2, ':pss3' => $pss3, ':pss4' => $pss4,
    ':takeaway' => $takeaway ?: null, ':comment' => $comment ?: null, ':tt' => $timeTotal ?: null,
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
        'cubes_reactive' => $reactive,
        'cubes_proactive' => $proactive,
        'cubes_operational' => $operational,
        'cubes_pool' => $pool,
        'satisfaction' => $satisfaction,
        'representative' => $representative,
        'work_life' => $workLife,
        'pss_1' => $pss1,
        'pss_2' => $pss2,
        'pss_3' => $pss3,
        'pss_4' => $pss4,
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
