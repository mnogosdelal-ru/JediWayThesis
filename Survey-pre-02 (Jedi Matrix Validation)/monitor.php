<?php
/**
 * Страница мониторинга A/B теста
 * Показывает статистику и позволяет скачать данные в CSV
 */

require_once 'db_config.php';

header('Content-Type: text/html; charset=utf-8');

// Обработка скачивания CSV
if (isset($_GET['download']) && $_GET['download'] === 'csv') {
    downloadCSV($pdo);
    exit;
}

// Получаем статистику
$stats = getStats($pdo);

function getStats($pdo) {
    // Всего респондентов
    $stmt = $pdo->query("SELECT COUNT(*) as total FROM ab_respondents");
    $total = $stmt->fetch()['total'];

    // Завершили опрос
    $stmt = $pdo->query("SELECT COUNT(*) as completed FROM ab_respondents WHERE status = 'completed'");
    $completed = $stmt->fetch()['completed'];

    // Начатные (начали, но не завершили)
    $started = $total - $completed;

    // По группам
    $stmt = $pdo->query("SELECT group_id, COUNT(*) as count FROM ab_respondents GROUP BY group_id");
    $byGroup = $stmt->fetchAll();

    // По вариантам
    $stmt = $pdo->query("SELECT variant, COUNT(*) as count FROM ab_respondents GROUP BY variant");
    $byVariant = $stmt->fetchAll();

    // Среднее время прохождения (для завершённых)
    $stmt = $pdo->query("SELECT AVG(time_total) as avg_time FROM ab_respondents WHERE status = 'completed' AND time_total IS NOT NULL");
    $avgTime = $stmt->fetch()['avg_time'];

    return [
        'total' => $total,
        'completed' => $completed,
        'started' => $started,
        'byGroup' => $byGroup,
        'byVariant' => $byVariant,
        'avgTime' => $avgTime ? round($avgTime / 60, 1) : 0 // в минутах
    ];
}

function downloadCSV($pdo) {
    $stmt = $pdo->query("SELECT * FROM ab_respondents ORDER BY created_at DESC");
    $data = $stmt->fetchAll();

    if (empty($data)) {
        echo "Нет данных для выгрузки";
        exit;
    }

    // Заголовки
    $headers = array_keys($data[0]);

    // BOM для корректного отображения русских символов в Excel
    header('Content-Type: text/csv; charset=utf-8');
    header('Content-Disposition: attachment; filename="ab_test_respondents_' . date('Y-m-d_His') . '.csv"');
    echo "\xEF\xBB\xBF";

    // Вывод заголовков
    echo implode(';', $headers) . "\n";

    // Вывод данных
    foreach ($data as $row) {
        $row = array_map(function($val) {
            if (is_null($val)) return '';
            // Экранируем кавычки и разделяем точкой с запятой
            $val = str_replace(['"', "\n", "\r"], ['""', ' ', ' '], $val);
            return '"' . $val . '"';
        }, $row);
        echo implode(';', $row) . "\n";
    }

    exit;
}
?>

<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Мониторинг A/B теста</title>
<link rel="icon" type="image/svg+xml" href="favicon.svg">
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 40px 20px;
        }

        .container {
            max-width: 800px;
            margin: 0 auto;
        }

        h1 {
            color: white;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2rem;
        }

        .subtitle {
            color: rgba(255,255,255,0.8);
            text-align: center;
            margin-bottom: 30px;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .stat-card {
            background: white;
            border-radius: 16px;
            padding: 25px;
            text-align: center;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }

        .stat-card.total {
            grid-column: span 2;
        }

        @media (max-width: 500px) {
            .stat-card.total {
                grid-column: span 1;
            }
        }

        .stat-value {
            font-size: 3rem;
            font-weight: bold;
            color: #667eea;
            line-height: 1;
        }

        .stat-value.completed {
            color: #27ae60;
        }

        .stat-value.started {
            color: #f39c12;
        }

        .stat-label {
            color: #666;
            font-size: 0.9rem;
            margin-top: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .details {
            background: white;
            border-radius: 16px;
            padding: 25px;
            margin-bottom: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }

        .details h2 {
            color: #333;
            margin-bottom: 20px;
            font-size: 1.2rem;
        }

        .details-row {
            display: flex;
            justify-content: space-between;
            padding: 12px 0;
            border-bottom: 1px solid #eee;
        }

        .details-row:last-child {
            border-bottom: none;
        }

        .details-label {
            color: #666;
        }

        .details-value {
            font-weight: bold;
            color: #333;
        }

        .download-section {
            text-align: center;
        }

        .download-btn {
            display: inline-block;
            background: #27ae60;
            color: white;
            padding: 18px 40px;
            border-radius: 50px;
            text-decoration: none;
            font-size: 1.1rem;
            font-weight: bold;
            box-shadow: 0 4px 15px rgba(39, 174, 96, 0.4);
            transition: all 0.3s ease;
        }

        .download-btn:hover {
            background: #219a52;
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(39, 174, 96, 0.5);
        }

        .download-btn:active {
            transform: translateY(0);
        }

        .back-link {
            display: block;
            text-align: center;
            margin-top: 30px;
            color: rgba(255,255,255,0.8);
            text-decoration: none;
        }

        .back-link:hover {
            color: white;
        }

        .empty-state {
            text-align: center;
            color: rgba(255,255,255,0.7);
            padding: 40px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Мониторинг A/B теста</h1>
        <p class="subtitle">Контрол продуктивности</p>

        <?php if ($stats['total'] == 0): ?>
            <div class="empty-state">
                <p>Пока нет данных. Дождитесь, пока респонденты начнут проходить опрос.</p>
            </div>
        <?php else: ?>

            <div class="stats-grid">
                <div class="stat-card total">
                    <div class="stat-value"><?= $stats['total'] ?></div>
                    <div class="stat-label">Всего респондентов</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value completed"><?= $stats['completed'] ?></div>
                    <div class="stat-label">Завершили</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value started"><?= $stats['started'] ?></div>
                    <div class="stat-label">Начали (не завершили)</div>
                </div>
            </div>

            <div class="details">
                <h2>📈 Детализация по группам</h2>
                
                <div class="details-row">
                    <span class="details-label">Группа 1 (standard, life→work)</span>
                    <span class="details-value"><?= getGroupCount($stats['byGroup'], 1) ?></span>
                </div>
                <div class="details-row">
                    <span class="details-label">Группа 2 (standard, work→life)</span>
                    <span class="details-value"><?= getGroupCount($stats['byGroup'], 2) ?></span>
                </div>
                <div class="details-row">
                    <span class="details-label">Группа 3 (horizontal, life→work)</span>
                    <span class="details-value"><?= getGroupCount($stats['byGroup'], 3) ?></span>
                </div>
                <div class="details-row">
                    <span class="details-label">Группа 4 (horizontal, work→life)</span>
                    <span class="details-value"><?= getGroupCount($stats['byGroup'], 4) ?></span>
                </div>
            </div>

            <div class="details">
                <h2>📋 Варианты интерфейса</h2>
                <?php foreach ($stats['byVariant'] as $variant): ?>
                    <div class="details-row">
                        <span class="details-label"><?= $variant['variant'] === 'standard' ? 'Стандартный (вертикальный)' : 'Горизонтальный' ?></span>
                        <span class="details-value"><?= $variant['count'] ?></span>
                    </div>
                <?php endforeach; ?>
            </div>

            <?php if ($stats['avgTime'] > 0): ?>
                <div class="details">
                    <h2>⏱️ Время прохождения</h2>
                    <div class="details-row">
                        <span class="details-label">Среднее время (завершившие)</span>
                        <span class="details-value"><?= $stats['avgTime'] ?> мин</span>
                    </div>
                </div>
            <?php endif; ?>

            <div class="download-section">
                <a href="?download=csv" class="download-btn">📥 Скачать CSV</a>
            </div>

        <?php endif; ?>

        <a href="index.html" class="back-link">← Вернуться к опросу</a>
    </div>

    <?php
    function getGroupCount($byGroup, $groupId) {
        foreach ($byGroup as $g) {
            if ($g['group_id'] == $groupId) return $g['count'];
        }
        return 0;
    }
    ?>
</body>
</html>
