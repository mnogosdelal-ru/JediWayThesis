<?php
/**
 * Страница мониторинга сбора данных
 * 
 * Отображает статистику по респондентам и их прогресс
 */

require_once __DIR__ . '/../config/config.php';
require_once __DIR__ . '/../src/Database.php';

// Получаем общую статистику
$statsQuery = "
    SELECT 
        COUNT(*) as total,
        SUM(CASE WHEN created_at >= DATE_SUB(NOW(), INTERVAL 1 HOUR) THEN 1 ELSE 0 END) as last_hour,
        SUM(CASE WHEN created_at >= DATE_SUB(NOW(), INTERVAL 24 HOUR) THEN 1 ELSE 0 END) as last_24h,
        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
        SUM(CASE WHEN status = 'in_progress' THEN 1 ELSE 0 END) as in_progress,
        SUM(CASE WHEN status = 'abandoned' THEN 1 ELSE 0 END) as abandoned
    FROM respondents
";

$stats = Database::selectOne($statsQuery);

// Получаем распределение по страницам
$pagesQuery = "
    SELECT 
        current_page,
        COUNT(*) as count
    FROM respondents
    WHERE status = 'in_progress'
    GROUP BY current_page
    ORDER BY current_page
";

$pages = Database::select($pagesQuery);

// Создаём массив для всех страниц (0-10)
$pagesDistribution = [];
for ($i = 0; $i <= 10; $i++) {
    $pagesDistribution[$i] = 0;
}

foreach ($pages as $row) {
    $pageNum = (int)$row['current_page'];
    if ($pageNum >= 0 && $pageNum <= 10) {
        $pagesDistribution[$pageNum] = (int)$row['count'];
    }
}

// Названия страниц
$pageNames = [
    0 => 'Согласие',
    1 => 'Демография',
    2 => 'Личная жизнь',
    3 => 'Работа',
    4 => 'MIJS',
    5 => 'SWLS',
    6 => 'MBI',
    7 => 'Прокрастинация',
    8 => 'Практики',
    9 => 'Вакцины',
    10 => 'Финиш'
];

// Получаем последних респондентов
$recentQuery = "
    SELECT code, status, current_page, created_at, completed_at
    FROM respondents
	WHERE status = 'completed'
    ORDER BY created_at DESC
    LIMIT 10
";
$recent = Database::select($recentQuery);

$page_title = 'Мониторинг сбора данных';
?>
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title><?= htmlspecialchars($page_title) ?></title>
    <link rel="icon" type="image/svg+xml" href="favicon.svg">
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        h1 {
            font-size: 28px;
            margin-bottom: 30px;
            color: #2c3e50;
            text-align: center;
        }
        
        .total-block {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 30px;
        }
        
        .total-number {
            font-size: 72px;
            font-weight: 700;
            line-height: 1;
            margin-bottom: 10px;
        }
        
        .total-label {
            font-size: 18px;
            opacity: 0.9;
        }
        
        .total-details {
            margin-top: 15px;
            font-size: 16px;
            opacity: 0.85;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .stat-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #667eea;
        }
        
        .stat-card.completed {
            border-left-color: #27ae60;
            background: #e8f8f5;
        }
        
        .stat-card.in-progress {
            border-left-color: #3498db;
            background: #e8f4fd;
        }
        
        .stat-card.abandoned {
            border-left-color: #e74c3c;
            background: #fdedec;
        }
        
        .stat-number {
            font-size: 36px;
            font-weight: 700;
            color: #2c3e50;
        }
        
        .stat-label {
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }
        
        .table-container {
            overflow-x: auto;
            margin-bottom: 30px;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            background: white;
        }
        
        th {
            background: #3498db;
            color: white;
            padding: 12px 8px;
            text-align: center;
            font-weight: 600;
        }
        
        td {
            padding: 12px 8px;
            text-align: center;
            border-bottom: 1px solid #eee;
        }
        
        tr:hover {
            background: #f5f5f5;
        }
        
        .page-name {
            font-weight: 600;
            color: #2c3e50;
        }
        
        .count-cell {
            font-size: 18px;
            font-weight: 700;
            color: #667eea;
        }
        
        .completed-cell {
            background: #27ae60;
            color: white;
            font-weight: 700;
            font-size: 18px;
        }
        
        .recent-table {
            margin-top: 20px;
        }
        
        .recent-table th {
            background: #2c3e50;
        }
        
        .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }
        
        .status-in_progress {
            background: #3498db;
            color: white;
        }
        
        .status-completed {
            background: #27ae60;
            color: white;
        }
        
        .status-abandoned {
            background: #e74c3c;
            color: white;
        }
        
        .refresh-notice {
            text-align: center;
            color: #666;
            font-size: 14px;
            margin-top: 20px;
        }
        
        .auto-refresh {
            background: #fff3cd;
            border: 1px solid #ffc107;
            padding: 10px 20px;
            border-radius: 4px;
            text-align: center;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Мониторинг сбора данных</h1>
        
        <div class="auto-refresh">
            🔄 Страница обновляется автоматически каждые 30 секунд
        </div>
        
        <!-- Общая статистика -->
        <div class="total-block">
            <div class="total-number"><?= (int)$stats['total'] ?></div>
            <div class="total-label">Всего респондентов</div>
            <div class="total-details">
                (за час: <strong><?= (int)$stats['last_hour'] ?></strong>, 
                 за сутки: <strong><?= (int)$stats['last_24h'] ?></strong>)
            </div>
        </div>
        
        <!-- Карточки статистики -->
        <div class="stats-grid">
            <div class="stat-card completed">
                <div class="stat-number"><?= (int)$stats['completed'] ?></div>
                <div class="stat-label">Завершили опрос</div>
            </div>
            
            <div class="stat-card in-progress">
                <div class="stat-number"><?= (int)$stats['in_progress'] ?></div>
                <div class="stat-label">В процессе</div>
            </div>
            
            <div class="stat-card abandoned">
                <div class="stat-number"><?= (int)$stats['abandoned'] ?></div>
                <div class="stat-label">Забросили</div>
            </div>
            
            <?php 
            $completionRate = $stats['total'] > 0 
                ? round(($stats['completed'] / $stats['total']) * 100) 
                : 0;
            ?>
            <div class="stat-card">
                <div class="stat-number"><?= $completionRate ?>%</div>
                <div class="stat-label">Завершение</div>
            </div>
        </div>
        
        <!-- Таблица распределения по страницам -->
        <div class="table-container">
            <h3 style="margin-bottom: 15px; color: #2c3e50;">📈 Распределение по страницам (в процессе)</h3>
            <table>
                <thead>
                    <tr>
                        <?php for ($i = 0; $i <= 10; $i++): ?>
                        <th>
                            <div style="font-size: 11px; opacity: 0.9;">Стр. <?= $i ?></div>
                            <div style="font-size: 10px; opacity: 0.8; font-weight: normal;"><?= $pageNames[$i] ?></div>
                        </th>
                        <?php endfor; ?>
                        <th style="background: #27ae60;">✅ Завершили</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <?php for ($i = 0; $i <= 10; $i++): ?>
                        <td>
                            <div class="count-cell"><?= $pagesDistribution[$i] ?></div>
                        </td>
                        <?php endfor; ?>
                        <td class="completed-cell"><?= (int)$stats['completed'] ?></td>
                    </tr>
                </tbody>
            </table>
        </div>
        
        <!-- Последние респонденты -->
        <div class="table-container">
            <h3 style="margin-bottom: 15px; color: #2c3e50;">🕐 Последние респонденты</h3>
            <table class="recent-table">
                <thead>
                    <tr>
                        <th>Код</th>
                        <th>Статус</th>
                        <th>Страница</th>
                        <th>Начало</th>
                        <th>Завершение</th>
                    </tr>
                </thead>
                <tbody>
                    <?php foreach ($recent as $r): ?>
                    <tr>
                        <td style="font-family: monospace; font-size: 14px;">
                            <?php if ($r['status'] === 'completed'):  ?>
                                <a href="results.php?code=<?= htmlspecialchars($r['code']) ?>" target="_blank">
                                    <?= htmlspecialchars($r['code']) ?>
                                </a>
                            <?php else: ?>
                                —
                            <?php endif; ?>
                        </td>
                        <td>
                            <span class="status-badge status-<?= $r['status'] ?>">
                                <?= $r['status'] === 'in_progress' ? 'В процессе' : ($r['status'] === 'completed' ? 'Завершён' : 'Заброшен') ?>
                            </span>
                        </td>
                        <td><?= $r['current_page'] ?></td>
                        <td><?= date('d.m.Y H:i', strtotime($r['created_at'])) ?></td>
                        <td><?= $r['completed_at'] ? date('d.m.Y H:i', strtotime($r['completed_at'])) : '—' ?></td>
                    </tr>
                    <?php endforeach; ?>
                </tbody>
            </table>
        </div>
        
        <p class="refresh-notice">
            Данные актуальны на <?= date('d.m.Y H:i:s') ?>
        </p>
    </div>
    
    <!-- Автообновление страницы -->
    <script>
        // Обновлять каждые 30 секунд
        setTimeout(function() {
            location.reload();
        }, 30000);
    </script>
</body>
</html>
