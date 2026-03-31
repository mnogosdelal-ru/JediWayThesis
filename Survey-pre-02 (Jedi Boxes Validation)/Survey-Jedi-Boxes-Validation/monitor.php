<?php
/**
 * Monitor - страница мониторинга хода исследования
 */

require_once 'db_config.php';

header('Content-Type: text/html; charset=utf-8');

// Обработка выгрузки CSV
if (isset($_GET['export']) && $_GET['export'] === 'csv') {
    exportCSV($pdo);
    exit;
}

function exportCSV($pdo) {
    $stmt = $pdo->query("SELECT * FROM jedi_boxes_respondents ORDER BY created_at DESC");
    $data = $stmt->fetchAll(PDO::FETCH_ASSOC);
    
    if (empty($data)) {
        echo "Нет данных для выгрузки";
        exit;
    }
    
    // Заголовки для скачивания
    header('Content-Type: text/csv; charset=utf-8');
    header('Content-Disposition: attachment; filename=jedi_boxes_results_' . date('Y-m-d_His') . '.csv');
    
    // Открываем поток вывода
    $output = fopen('php://output', 'w');
    
    // BOM для корректного отображения UTF-8 в Excel
    fprintf($output, chr(0xEF).chr(0xBB).chr(0xBF));
    
    // Заголовки
    fputcsv($output, array_keys($data[0]), ';');
    
    // Данные
    foreach ($data as $row) {
        fputcsv($output, $row, ';');
    }
    
    fclose($output);
    exit;
}

// Получаем статистику
$stats = getStats($pdo);

function getStats($pdo) {
    // Общее количество
    $stmt = $pdo->query("SELECT COUNT(*) as cnt FROM jedi_boxes_respondents");
    $total = $stmt->fetch()['cnt'];
    
    // Завершённые
    $stmt = $pdo->query("SELECT COUNT(*) as cnt FROM jedi_boxes_respondents WHERE status = 'completed'");
    $completed = $stmt->fetch()['cnt'];
    
    // Не начавшие (нет записей о возрасте - значит не прошли демографию)
    $stmt = $pdo->query("SELECT COUNT(*) as cnt FROM jedi_boxes_respondents WHERE age IS NULL OR age = 0");
    $notStarted = $stmt->fetch()['cnt'];
    
    // Дошли до страницы 1 (заполнили демографию)
    $stmt = $pdo->query("SELECT COUNT(*) as cnt FROM jedi_boxes_respondents WHERE age > 0 AND (cubes_reactive IS NULL OR cubes_reactive = 0)");
    $atPage1 = $stmt->fetch()['cnt'];
    
    // Дошли до страницы 2 (распределили кубики)
    $stmt = $pdo->query("SELECT COUNT(*) as cnt FROM jedi_boxes_respondents WHERE cubes_reactive > 0 OR cubes_proactive > 0 OR cubes_operational > 0");
    $atPage2 = $stmt->fetch()['cnt'];
    
    // Дошли до страницы 3 (ответили на контекстные вопросы)
    $stmt = $pdo->query("SELECT COUNT(*) as cnt FROM jedi_boxes_respondents WHERE representative > 0");
    $atPage3 = $stmt->fetch()['cnt'];
    
    // Дошли до страницы 4 (ответили на прокрастинацию)
    $stmt = $pdo->query("SELECT COUNT(*) as cnt FROM jedi_boxes_respondents WHERE proc_1 > 0");
    $atPage4 = $stmt->fetch()['cnt'];
    
    // Дошли до страницы 5 (ответили на SWLS)
    $stmt = $pdo->query("SELECT COUNT(*) as cnt FROM jedi_boxes_respondents WHERE swls_1 > 0");
    $atPage5 = $stmt->fetch()['cnt'];
    
    return [
        'total' => $total,
        'completed' => $completed,
        'in_progress' => $total - $completed,
        'pages' => [
            'started' => $total - $notStarted,
            'page1' => $atPage1,
            'page2' => $atPage2,
            'page3' => $atPage3,
            'page4' => $atPage4,
            'page5' => $atPage5,
            'completed' => $completed
        ]
    ];
}
?>
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Мониторинг исследования</title>
<style>
    * {
        box-sizing: border-box;
        margin: 0;
        padding: 0;
    }
    
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
        padding: 20px;
    }
    
    .container {
        max-width: 900px;
        margin: 0 auto;
    }
    
    h1 {
        color: white;
        text-align: center;
        margin-bottom: 30px;
        font-size: 2rem;
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 20px;
        margin-bottom: 30px;
    }
    
    .stat-card {
        background: white;
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .stat-card .value {
        font-size: 3rem;
        font-weight: bold;
        color: #3498db;
    }
    
    .stat-card.completed .value {
        color: #27ae60;
    }
    
    .stat-card .label {
        font-size: 0.95rem;
        color: #7f8c8d;
        margin-top: 5px;
    }
    
    .progress-section {
        background: white;
        border-radius: 16px;
        padding: 25px;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .progress-section h2 {
        margin-bottom: 20px;
        color: #2c3e50;
    }
    
    .page-bar {
        display: flex;
        align-items: center;
        margin-bottom: 15px;
    }
    
    .page-label {
        width: 150px;
        font-weight: 600;
        color: #2c3e50;
    }
    
    .page-track {
        flex: 1;
        height: 30px;
        background: #ecf0f1;
        border-radius: 15px;
        overflow: hidden;
        margin: 0 15px;
    }
    
    .page-fill {
        height: 100%;
        background: linear-gradient(90deg, #3498db, #2980b9);
        border-radius: 15px;
        transition: width 0.5s ease;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 10px;
        color: white;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    .page-fill.green {
        background: linear-gradient(90deg, #27ae60, #1e8449);
    }
    
    .page-count {
        width: 60px;
        text-align: right;
        color: #7f8c8d;
        font-weight: 600;
    }
    
    .btn-export {
        display: block;
        width: 100%;
        max-width: 300px;
        margin: 0 auto;
        padding: 15px 30px;
        background: #27ae60;
        color: white;
        border: none;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        text-decoration: none;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .btn-export:hover {
        background: #1e8449;
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(39, 174, 96, 0.4);
    }
    
    .btn-refresh {
        padding: 12px 25px;
        background: #3498db;
        color: white;
        border: none;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .btn-refresh:hover {
        background: #2980b9;
        transform: translateY(-2px);
    }
    
    .refresh-note {
        text-align: center;
        color: rgba(255,255,255,0.8);
        margin-top: 20px;
        font-size: 0.9rem;
    }
    
    @media (max-width: 600px) {
        .stats-grid {
            grid-template-columns: 1fr;
        }
        
        .page-bar {
            flex-direction: column;
            align-items: flex-start;
        }
        
        .page-label {
            width: 100%;
            margin-bottom: 5px;
        }
        
        .page-track {
            width: 100%;
            margin: 5px 0;
        }
        
        .page-count {
            width: 100%;
            text-align: left;
        }
    }
</style>
</head>
<body>

<div class="container">
    <h1>📊 Мониторинг исследования</h1>
    
    <div class="stats-grid">
        <div class="stat-card">
            <div class="value"><?= $stats['total'] ?></div>
            <div class="label">Всего респондентов</div>
        </div>
        <div class="stat-card completed">
            <div class="value"><?= $stats['completed'] ?></div>
            <div class="label">Прошли до конца</div>
        </div>
        <div class="stat-card">
            <div class="value"><?= $stats['in_progress'] ?></div>
            <div class="label">В процессе</div>
        </div>
    </div>
    
    <div class="progress-section">
        <h2>Распределение по страницам</h2>
        
        <?php 
        $maxCount = max($stats['total'], 1);
        $pagesList = [
            'started' => 'Начали (демография)',
            'page1' => 'Страница 1: Кубики',
            'page2' => 'Страница 2: Контекст',
            'page3' => 'Страница 3: Прокрастинация',
            'page4' => 'Страница 4: SWLS',
            'page5' => 'Страница 5: MBI',
            'completed' => 'Завершили'
        ];
        
        foreach ($pagesList as $key => $label):
            $count = isset($stats['pages'][$key]) ? $stats['pages'][$key] : ($key === 'completed' ? $stats['completed'] : 0);
            $percent = round(($count / $maxCount) * 100);
            $isLast = ($key === 'completed');
        ?>
        <div class="page-bar">
            <div class="page-label"><?= $label ?></div>
            <div class="page-track">
                <div class="page-fill<?= $isLast ? ' green' : '' ?>" style="width: <?= $percent ?>%">
                    <?= $percent > 10 ? $percent . '%' : '' ?>
                </div>
            </div>
            <div class="page-count"><?= $count ?></div>
        </div>
        <?php endforeach; ?>
    </div>
    
    <a href="?export=csv" class="btn-export">📥 Выгрузить CSV</a>
    
    <div style="text-align: center; margin-top: 20px;">
        <button onclick="location.reload()" class="btn-refresh">🔄 Обновить данные</button>
    </div>
    
    <p class="refresh-note">Автоматическое обновление отключено. Нажмите кнопку для обновления.</p>
</div>

</body>
</html>
