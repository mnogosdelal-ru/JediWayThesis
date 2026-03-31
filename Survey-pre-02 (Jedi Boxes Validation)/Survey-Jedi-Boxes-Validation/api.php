<?php
// Глобальный параметр отладки
define('DEBUG_MODE', true);

require_once 'db_config.php';

header('Content-Type: application/json; charset=utf-8');

$input = file_get_contents('php://input');
$data = json_decode($input, true);

if (!$data || !isset($data['action'])) {
    echo json_encode(['success' => false, 'error' => 'Invalid request']);
    exit;
}

$action = $data['action'];

// 1. Инициализация сессии
if ($action === 'init_session') {
    $force_new_session = defined('DEBUG_MODE') && DEBUG_MODE;
    
    if (!$force_new_session && isset($data['session_id']) && !empty($data['session_id'])) {
        $session_id = $data['session_id'];
        $stmt = $pdo->prepare("SELECT * FROM jedi_boxes_respondents WHERE session_id = ?");
        $stmt->execute([$session_id]);
        $existing = $stmt->fetch();

        if ($existing) {
            echo json_encode([
                'success' => true,
                'session_id' => $existing['session_id'],
                'debug_mode' => DEBUG_MODE
            ]);
            exit;
        }
    }

    // Новая сессия
    $session_id = bin2hex(random_bytes(16));

    try {
        $stmt = $pdo->prepare("INSERT INTO jedi_boxes_respondents (session_id, debug_mode) VALUES (?, ?)");
        $stmt->execute([$session_id, defined('DEBUG_MODE') && DEBUG_MODE ? 1 : 0]);

        echo json_encode([
            'success' => true,
            'session_id' => $session_id,
            'debug_mode' => defined('DEBUG_MODE') ? DEBUG_MODE : false
        ]);
    } catch (PDOException $e) {
        echo json_encode(['success' => false, 'error' => $e->getMessage()]);
    }
    exit;
}

// Все последующие действия требуют session_id
if (!isset($data['session_id'])) {
    echo json_encode(['success' => false, 'error' => 'Missing session_id']);
    exit;
}

$session_id = $data['session_id'];

// 2. Сохранение данных страницы
if ($action === 'save_page') {
    $pageData = $data['data'] ?? [];
    
    $allowed_fields = [
        // Page 0: Демография
        'status', 'age', 'gender', 'position', 'profession',
        'time_page0_start', 'time_page0_end',
        
        // Page 1: Кубики
        'time_page1_start', 'time_page1_end', 'time_page1_total',
        'cubes_reactive', 'cubes_proactive', 'cubes_operational',
        
        // Page 2: Контекстные вопросы
        'time_page2_start', 'time_page2_end', 'time_page2_total',
        'representative', 'work_life', 'energy_deficit', 
        'subjective_productivity', 'energy_level', 'memory_vs_records',
        
        // Page 3: Прокрастинация
        'time_page3_start', 'time_page3_end', 'time_page3_total',
        'proc_1', 'proc_2', 'proc_3', 'proc_4', 'proc_5', 'proc_6', 'proc_7', 'proc_8',
        
        // Page 4: SWLS
        'time_page4_start', 'time_page4_end', 'time_page4_total',
        'swls_1', 'swls_2', 'swls_3', 'swls_4', 'swls_5',
        
        // Page 5: MBI
        'time_page5_start', 'time_page5_end', 'time_page5_total',
        'mbi_1', 'mbi_2', 'mbi_3', 'mbi_4', 'mbi_5', 'mbi_6', 
        'mbi_7', 'mbi_8', 'mbi_9', 'mbi_10', 'mbi_11', 'mbi_12'
    ];
    
    $updateFields = [];
    $updateValues = [];
    
    foreach ($pageData as $key => $value) {
        if (in_array($key, $allowed_fields)) {
            $updateFields[] = "$key = ?";
            $updateValues[] = $value;
        }
    }
    
    if (empty($updateFields)) {
        echo json_encode(['success' => true, 'message' => 'No data to update']);
        exit;
    }
    
    $updateValues[] = $session_id;
    $sql = "UPDATE jedi_boxes_respondents SET " . implode(", ", $updateFields) . " WHERE session_id = ?";
    
    try {
        $stmt = $pdo->prepare($sql);
        $stmt->execute($updateValues);
        echo json_encode(['success' => true]);
    } catch (PDOException $e) {
        echo json_encode(['success' => false, 'error' => $e->getMessage()]);
    }
    exit;
}

// 3. Сохранение страницы с расчётом суммарных баллов
if ($action === 'save_page') {
    $pageData = $data['data'] ?? [];
    
    $allowed_fields = [
        // Page 0: Демография
        'status', 'age', 'gender', 'position', 'profession',
        'time_page0_start', 'time_page0_end',
        
        // Page 1: Кубики
        'time_page1_start', 'time_page1_end', 'time_page1_total',
        'cubes_reactive', 'cubes_proactive', 'cubes_operational',
        
        // Page 2: Контекстные вопросы
        'time_page2_start', 'time_page2_end', 'time_page2_total',
        'representative', 'work_life', 'energy_deficit', 
        'subjective_productivity', 'energy_level', 'memory_vs_records',
        
        // Page 3: Прокрастинация
        'time_page3_start', 'time_page3_end', 'time_page3_total',
        'proc_1', 'proc_2', 'proc_3', 'proc_4', 'proc_5', 'proc_6', 'proc_7', 'proc_8',
        
        // Page 4: SWLS
        'time_page4_start', 'time_page4_end', 'time_page4_total',
        'swls_1', 'swls_2', 'swls_3', 'swls_4', 'swls_5',
        
        // Page 5: MBI
        'time_page5_start', 'time_page5_end', 'time_page5_total',
        'mbi_1', 'mbi_2', 'mbi_3', 'mbi_4', 'mbi_5', 'mbi_6', 
        'mbi_7', 'mbi_8', 'mbi_9', 'mbi_10', 'mbi_11', 'mbi_12'
    ];
    
    $updateFields = [];
    $updateValues = [];
    
    foreach ($pageData as $key => $value) {
        if (in_array($key, $allowed_fields)) {
            $updateFields[] = "$key = ?";
            $updateValues[] = $value;
        }
    }
    
    // Расчёт суммарных баллов
    $proc_total = 0;
    for ($i = 1; $i <= 8; $i++) {
        if (isset($pageData["proc_$i"])) {
            $proc_total += (int)$pageData["proc_$i"];
        }
    }
    if ($proc_total > 0) {
        $updateFields[] = "proc_total = ?";
        $updateValues[] = $proc_total;
    }
    
    $swls_total = 0;
    for ($i = 1; $i <= 5; $i++) {
        if (isset($pageData["swls_$i"])) {
            $swls_total += (int)$pageData["swls_$i"];
        }
    }
    if ($swls_total > 0) {
        $updateFields[] = "swls_total = ?";
        $updateValues[] = $swls_total;
    }
    
    $mbi_total = 0;
    for ($i = 1; $i <= 12; $i++) {
        if (isset($pageData["mbi_$i"])) {
            $mbi_total += (int)$pageData["mbi_$i"];
        }
    }
    if ($mbi_total > 0) {
        $updateFields[] = "mbi_total = ?";
        $updateValues[] = $mbi_total;
    }
    
    if (empty($updateFields)) {
        echo json_encode(['success' => true, 'message' => 'No data to update']);
        exit;
    }
    
    $updateValues[] = $session_id;
    $sql = "UPDATE jedi_boxes_respondents SET " . implode(", ", $updateFields) . " WHERE session_id = ?";
    
    try {
        $stmt = $pdo->prepare($sql);
        $stmt->execute($updateValues);
        echo json_encode(['success' => true]);
    } catch (PDOException $e) {
        echo json_encode(['success' => false, 'error' => $e->getMessage()]);
    }
    exit;
}

// 4. Завершение опроса
if ($action === 'finish_survey') {
    $pageData = $data['data'] ?? [];
    
    $updateFields = ['status = ?', 'time_finished = ?'];
    $updateValues = ['completed', time()];
    
    $allowed_fields = [
        'time_total', 'time_page5_start', 'time_page5_end', 'time_page5_total',
        'mbi_1', 'mbi_2', 'mbi_3', 'mbi_4', 'mbi_5', 'mbi_6', 
        'mbi_7', 'mbi_8', 'mbi_9', 'mbi_10', 'mbi_11', 'mbi_12'
    ];
    
    // Расчёт mbi_total
    $mbi_total = 0;
    for ($i = 1; $i <= 12; $i++) {
        if (isset($pageData["mbi_$i"])) {
            $mbi_total += (int)$pageData["mbi_$i"];
        }
    }
    if ($mbi_total > 0) {
        $updateFields[] = "mbi_total = ?";
        $updateValues[] = $mbi_total;
    }
    
    foreach ($pageData as $key => $value) {
        if (in_array($key, $allowed_fields)) {
            $updateFields[] = "$key = ?";
            $updateValues[] = $value;
        }
    }
    
    $updateValues[] = $session_id;
    $sql = "UPDATE jedi_boxes_respondents SET " . implode(", ", $updateFields) . " WHERE session_id = ?";
    
    try {
        $stmt = $pdo->prepare($sql);
        $stmt->execute($updateValues);
        
        // Получаем все данные для JSON
        $stmt = $pdo->prepare("SELECT * FROM jedi_boxes_respondents WHERE session_id = ?");
        $stmt->execute([$session_id]);
        $record = $stmt->fetch();
        
        if ($record) {
            $backup_dir = __DIR__ . '/respondents';
            if (!is_dir($backup_dir)) {
                mkdir($backup_dir, 0777, true);
            }
            $filename = $backup_dir . '/respondent_' . $session_id . '.json';
            file_put_contents($filename, json_encode($record, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE));
        }
        
        echo json_encode(['success' => true]);
    } catch (PDOException $e) {
        echo json_encode(['success' => false, 'error' => $e->getMessage()]);
    }
    exit;
}

// 5. Получение результатов с процентами
if ($action === 'get_results') {
    try {
        // Получаем данные текущего респондента
        $stmt = $pdo->prepare("SELECT proc_total, swls_total, mbi_total FROM jedi_boxes_respondents WHERE session_id = ?");
        $stmt->execute([$session_id]);
        $current = $stmt->fetch();
        
        if (!$current) {
            echo json_encode(['success' => false, 'error' => 'Session not found']);
            exit;
        }
        
        $result = [
            'proc_total' => (int)$current['proc_total'],
            'swls_total' => (int)$current['swls_total'],
            'mbi_total' => (int)$current['mbi_total']
        ];
        
        // Получаем общее количество завершённых ответов (исключая текущего)
        $stmt = $pdo->query("SELECT COUNT(*) as total FROM jedi_boxes_respondents WHERE status = 'completed'");
        $total_count = $stmt->fetch()['total'];
        
        // Если нет других респондентов, возвращаем 50% для всех
        if ($total_count == 0) {
            $result['proc_percentile'] = 50;
            $result['swls_percentile'] = 50;
            $result['mbi_percentile'] = 50;
            $result['total_respondents'] = 1;
            $result['other_respondents'] = 0;
            echo json_encode(['success' => true, 'data' => $result]);
            exit;
        }
        
        // Процентиль для прокрастинации
        $stmt = $pdo->prepare("
            SELECT 
                COUNT(*) as cnt,
                SUM(CASE WHEN proc_total < ? THEN 1 ELSE 0 END) as below,
                SUM(CASE WHEN proc_total = ? THEN 1 ELSE 0 END) as equal,
                SUM(CASE WHEN proc_total > ? THEN 1 ELSE 0 END) as above
            FROM jedi_boxes_respondents 
            WHERE status = 'completed' AND proc_total > 0
        ");
        $stmt->execute([$current['proc_total'], $current['proc_total'], $current['proc_total']]);
        $proc_stats = $stmt->fetch();
        
        // Процентиль: (ниже + 0.5 * равно) / общее * 100
        $proc_percentile = 0;
        if ($proc_stats['cnt'] > 0) {
            $proc_percentile = round(($proc_stats['below'] + 0.5 * $proc_stats['equal']) / $proc_stats['cnt'] * 100);
        }
        
        // Процентиль для SWLS
        $stmt = $pdo->prepare("
            SELECT 
                COUNT(*) as cnt,
                SUM(CASE WHEN swls_total < ? THEN 1 ELSE 0 END) as below,
                SUM(CASE WHEN swls_total = ? THEN 1 ELSE 0 END) as equal,
                SUM(CASE WHEN swls_total > ? THEN 1 ELSE 0 END) as above
            FROM jedi_boxes_respondents 
            WHERE status = 'completed' AND swls_total > 0
        ");
        $stmt->execute([$current['swls_total'], $current['swls_total'], $current['swls_total']]);
        $swls_stats = $stmt->fetch();
        
        $swls_percentile = 0;
        if ($swls_stats['cnt'] > 0) {
            $swls_percentile = round(($swls_stats['below'] + 0.5 * $swls_stats['equal']) / $swls_stats['cnt'] * 100);
        }
        
        // Процентиль для MBI
        $stmt = $pdo->prepare("
            SELECT 
                COUNT(*) as cnt,
                SUM(CASE WHEN mbi_total < ? THEN 1 ELSE 0 END) as below,
                SUM(CASE WHEN mbi_total = ? THEN 1 ELSE 0 END) as equal,
                SUM(CASE WHEN mbi_total > ? THEN 1 ELSE 0 END) as above
            FROM jedi_boxes_respondents 
            WHERE status = 'completed' AND mbi_total > 0
        ");
        $stmt->execute([$current['mbi_total'], $current['mbi_total'], $current['mbi_total']]);
        $mbi_stats = $stmt->fetch();
        
        $mbi_percentile = 0;
        if ($mbi_stats['cnt'] > 0) {
            $mbi_percentile = round(($mbi_stats['below'] + 0.5 * $mbi_stats['equal']) / $mbi_stats['cnt'] * 100);
        }
        
        $result['proc_percentile'] = $proc_percentile;
        $result['proc_below'] = (int)$proc_stats['below'];
        $result['proc_equal'] = (int)$proc_stats['equal'];
        $result['proc_above'] = (int)$proc_stats['above'];
        
        $result['swls_percentile'] = $swls_percentile;
        $result['swls_below'] = (int)$swls_stats['below'];
        $result['swls_equal'] = (int)$swls_stats['equal'];
        $result['swls_above'] = (int)$swls_stats['above'];
        
        $result['mbi_percentile'] = $mbi_percentile;
        $result['mbi_below'] = (int)$mbi_stats['below'];
        $result['mbi_equal'] = (int)$mbi_stats['equal'];
        $result['mbi_above'] = (int)$mbi_stats['above'];
        
        $result['total_respondents'] = (int)$proc_stats['cnt'];
        $result['other_respondents'] = (int)$proc_stats['cnt'];
        
        echo json_encode(['success' => true, 'data' => $result]);
    } catch (PDOException $e) {
        echo json_encode(['success' => false, 'error' => $e->getMessage()]);
    }
    exit;
}

echo json_encode(['success' => false, 'error' => 'Unknown action']);
?>
