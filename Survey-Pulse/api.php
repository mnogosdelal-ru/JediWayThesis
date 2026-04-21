<?php
/**
 * API пульс-опросов — сохранение ответа
 */
error_reporting(0);
ini_set('display_errors', 0);

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
$stmt = $pdo->prepare("SELECT id FROM pulse_responses WHERE session_id = :sid");
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

if ($reactive + $proactive + $operational !== 6) {
    http_response_code(400);
    echo json_encode(['error' => 'Сумма кубиков должна равняться 6']);
    exit;
}

$representative = isset($_POST['representative']) ? $_POST['representative'] : null;
$workLife = isset($_POST['work_life']) ? $_POST['work_life'] : null;
$energyDeficit = isset($_POST['energy_deficit']) ? $_POST['energy_deficit'] : null;
$mvr = isset($_POST['memory_vs_records']) ? $_POST['memory_vs_records'] : null;
$takeaway = trim($_POST['takeaway'] ?? '');
$comment = trim($_POST['comment'] ?? '');
$timeTotal = isset($_POST['time_total']) ? $_POST['time_total'] : null;

$ipHash = hash('sha256', $_SERVER['REMOTE_ADDR'] ?? '');
$ua = $_SERVER['HTTP_USER_AGENT'] ?? '';
$device = preg_match('/Mobile|Android|iPhone|iPad/i', $ua) ? 'mobile' : 'desktop';

$stmt = $pdo->prepare("
    INSERT INTO pulse_responses (
        session_id, status, tg_id, week, group_id,
        cubes_reactive, cubes_proactive, cubes_operational,
        representative, work_life, energy_deficit, memory_vs_records,
        takeaway, comment, time_total,
        user_agent, ip_hash, device_type
    ) VALUES (
        :sid, 'completed', :tg_id, :week, :group_id,
        :r, :g, :o,
        :rep, :wl, :def, :mvr,
        :takeaway, :comment, :tt,
        :ua, :ip, :device
    )
");

$stmt->execute([
    ':sid' => $sessionId,
    ':tg_id' => $tgId, ':week' => $week, ':group_id' => $groupId,
    ':r' => $reactive, ':g' => $proactive, ':o' => $operational,
    ':rep' => $representative, ':wl' => $workLife, ':def' => $energyDeficit, ':mvr' => $mvr,
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
        'memory_vs_records' => $mvr,
        'representative' => $representative,
        'work_life' => $workLife,
        'energy_deficit' => $energyDeficit,
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
