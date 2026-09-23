<?php
/**
 * Монитор пульс-опросов
 * - Ответы по дням за последние 2 недели
 * - Выгрузка в CSV
 */
require_once 'db_config.php';

try {
    $pdo = getDbConnection();
} catch (Exception $e) {
    die('Ошибка подключения к БД: ' . htmlspecialchars($e->getMessage()));
}

// ── CSV экспорт (порядок столбцов = Google Sheets) ──
if (isset($_GET['csv'])) {
    header('Content-Type: text/csv; charset=utf-8');
    header('Content-Disposition: attachment; filename="pulse_results_' . date('Y-m-d') . '.csv"');

    $out = fopen('php://output', 'w');
    fwrite($out, "\xEF\xBB\xBF"); // BOM для Excel

    // Заголовки — точно как в Google Sheets
    fputcsv($out, [
        'timestamp',
        'session_id',
        'tg_id',
        'week',
        'group_id',
        'cubes_reactive',
        'cubes_proactive',
        'cubes_operational',
        'memory_vs_records',
        'representative',
        'work_life',
        'energy_deficit',
        'takeaway',
        'comment',
        'time_total',
        'device_type'
    ], ';');

    $stmt = $pdo->query("
        SELECT created_at as timestamp, session_id, tg_id, week, group_id,
               cubes_reactive, cubes_proactive, cubes_operational,
               memory_vs_records, representative, work_life, energy_deficit,
               takeaway, comment, time_total, device_type
        FROM pulse_responses
        WHERE status = 'completed'
        ORDER BY created_at DESC
    ");

    while ($r = $stmt->fetch()) {
        fputcsv($out, [
            $r['timestamp'],
            $r['session_id'],
            $r['tg_id'] ?? '',
            $r['week'] ?? '',
            $r['group_id'] ?? '',
            $r['cubes_reactive'],
            $r['cubes_proactive'],
            $r['cubes_operational'],
            $r['memory_vs_records'] ?? '',
            $r['representative'] ?? '',
            $r['work_life'] ?? '',
            $r['energy_deficit'] ?? '',
            $r['takeaway'] ?? '',
            $r['comment'] ?? '',
            $r['time_total'] ?? '',
            $r['device_type'] ?? ''
        ], ';');
    }

    fclose($out);
    exit;
}

// ── Ответы по дням за последние 2 недели ──
$stmt = $pdo->query("
    SELECT
        DATE(created_at) as day,
        COUNT(*) as total,
        SUM(status='completed') as completed,
        AVG(cubes_reactive) as avg_r,
        AVG(cubes_proactive) as avg_g,
        AVG(cubes_operational) as avg_o,
        AVG(representative) as avg_rep,
        AVG(work_life) as avg_wl,
        AVG(energy_deficit) as avg_def
    FROM pulse_responses
    WHERE created_at >= DATE_SUB(CURDATE(), INTERVAL 14 DAY)
    GROUP BY DATE(created_at)
    ORDER BY day DESC
");
$days = $stmt->fetchAll();

// ── Общая статистика ──
$stats = $pdo->query("
    SELECT
        COUNT(*) as total_all,
        COUNT(CASE WHEN status='completed' THEN 1 END) as completed_all,
        AVG(cubes_reactive) as avg_r_all,
        AVG(cubes_proactive) as avg_g_all,
        AVG(cubes_operational) as avg_o_all,
        AVG(representative) as avg_rep_all,
        AVG(work_life) as avg_wl_all,
        AVG(energy_deficit) as avg_def_all,
        AVG(time_total) as avg_time
    FROM pulse_responses
")->fetch();

// ── Последние текстовые ответы ──
$takeaways = $pdo->query("
    SELECT takeaway, comment, created_at
    FROM pulse_responses
    WHERE status='completed' AND (takeaway IS NOT NULL OR comment IS NOT NULL)
    ORDER BY created_at DESC
    LIMIT 30
")->fetchAll();

// ── Последние ответы ──
$recent = $pdo->query("
    SELECT created_at, tg_id, week, group_id, cubes_reactive, cubes_proactive, cubes_operational,
           representative, work_life, energy_deficit, time_total, device_type
    FROM pulse_responses
    WHERE status='completed'
    ORDER BY created_at DESC
    LIMIT 50
")->fetchAll();
?>
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Пульс-опрос — Монитор</title>
<style>
* { box-sizing: border-box; }
body { font-family: system-ui, -apple-system, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f0f2f5; color: #2c3e50; }
h1 { margin-bottom: 8px; }
h2 { margin: 24px 0 12px; font-size: 1.2rem; }
.subtitle { color: #7f8c8d; margin-bottom: 20px; }

/* Кнопки */
.actions { margin-bottom: 20px; display: flex; gap: 12px; flex-wrap: wrap; }
.btn { padding: 10px 20px; border: none; border-radius: 8px; font-size: 0.9rem; cursor: pointer; font-weight: 600; text-decoration: none; display: inline-block; }
.btn-csv { background: #27ae60; color: white; }
.btn-csv:hover { background: #219a52; }

/* Статистика */
.stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin-bottom: 24px; }
.stat-card { background: white; padding: 14px; border-radius: 8px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
.stat-card .value { font-size: 1.8rem; font-weight: 700; }
.stat-card .value.blue { color: #3498db; }
.stat-card .value.green { color: #27ae60; }
.stat-card .value.red { color: #e74c3c; }
.stat-card .value.orange { color: #e67e22; }
.stat-card .value.gray { color: #95a5a6; }
.stat-card .label { font-size: 0.8rem; color: #95a5a6; margin-top: 2px; }

/* Таблица по дням */
.days-table { width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
.days-table th, .days-table td { padding: 10px 12px; text-align: center; font-size: 0.85rem; }
.days-table th { background: #34495e; color: white; }
.days-table td:first-child { text-align: left; font-weight: 600; }
.days-table tr:not(:last-child) td { border-bottom: 1px solid #eee; }
.days-table tr:hover { background: #f8f9fa; }
.bar-cell { display: flex; gap: 4px; justify-content: center; }
.bar { height: 20px; border-radius: 3px; min-width: 4px; }
.bar-r { background: #e74c3c; }
.bar-g { background: #27ae60; }
.bar-o { background: #95a5a6; }

/* Текстовые ответы */
.text-answer { background: white; padding: 12px; border-radius: 8px; margin-bottom: 8px; border-left: 4px solid #3498db; }
.text-answer .date { font-size: 0.75rem; color: #bdc3c7; }
.text-answer .takeaway { font-weight: 600; margin: 4px 0; color: #2c3e50; }
.text-answer .comment { color: #7f8c8d; font-style: italic; }

/* Последние ответы */
.recent-table { width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
.recent-table th, .recent-table td { padding: 8px 10px; text-align: left; font-size: 0.8rem; }
.recent-table th { background: #34495e; color: white; }
.recent-table tr:not(:last-child) td { border-bottom: 1px solid #eee; }
.tg-id { font-size: 0.7rem; color: #bdc3c7; }

@media (max-width: 768px) {
    .stats-grid { grid-template-columns: repeat(2, 1fr); }
    .days-table { font-size: 0.75rem; }
    .days-table th, .days-table td { padding: 6px 4px; }
}
</style>
</head>
<body>

<h1>📊 Пульс-опрос — Монитор</h1>
<p class="subtitle">Ответы за последние 14 дней</p>

<div class="actions">
    <a href="?csv" class="btn btn-csv">⬇ Выгрузить CSV</a>
</div>

<!-- Общая статистика -->
<h2>Общая статистика (всё время)</h2>
<div class="stats-grid">
    <div class="stat-card">
        <div class="value blue"><?= $stats['completed_all'] ?></div>
        <div class="label">Завершённых</div>
    </div>
    <div class="stat-card">
        <div class="value red"><?= $stats['total_all'] ?></div>
        <div class="label">Всего записей</div>
    </div>
    <div class="stat-card">
        <div class="value red"><?= $stats['avg_r_all'] ? number_format($stats['avg_r_all'], 1) : '—' ?></div>
        <div class="label">🔴 Срочное (среднее)</div>
    </div>
    <div class="stat-card">
        <div class="value green"><?= $stats['avg_g_all'] ? number_format($stats['avg_g_all'], 1) : '—' ?></div>
        <div class="label">🟢 Целевое (среднее)</div>
    </div>
    <div class="stat-card">
        <div class="value gray"><?= $stats['avg_o_all'] ? number_format($stats['avg_o_all'], 1) : '—' ?></div>
        <div class="label">⚪ Операционное (среднее)</div>
    </div>
    <div class="stat-card">
        <div class="value blue"><?= $stats['avg_rep_all'] ? number_format($stats['avg_rep_all'], 1) : '—' ?></div>
        <div class="label">Типичность недели</div>
    </div>
    <div class="stat-card">
        <div class="value orange"><?= $stats['avg_wl_all'] ? number_format($stats['avg_wl_all'], 1) : '—' ?></div>
        <div class="label">Баланс работа/личное</div>
    </div>
    <div class="stat-card">
        <div class="value red"><?= $stats['avg_def_all'] ? number_format($stats['avg_def_all'], 1) : '—' ?></div>
        <div class="label">Энергодефицит</div>
    </div>
    <div class="stat-card">
        <div class="value blue"><?= $stats['avg_time'] ? round($stats['avg_time']) . 'с' : '—' ?></div>
        <div class="label">Среднее время</div>
    </div>
</div>

<!-- По дням -->
<h2>Ответы по дням (последние 14 дней)</h2>
<?php if (empty($days)): ?>
    <p style="color:#95a5a6;">Пока нет данных</p>
<?php else: ?>
<table class="days-table">
<tr>
    <th>Дата</th>
    <th>Всего</th>
    <th>Завершено</th>
    <th>🔴</th>
    <th>🟢</th>
    <th>⚪</th>
    <th>Типичн.</th>
    <th>Баланс</th>
    <th>Дефицит</th>
</tr>
<?php foreach ($days as $d): ?>
<tr>
    <td><?= date('d.m', strtotime($d['day'])) ?></td>
    <td><?= $d['total'] ?></td>
    <td><?= $d['completed'] ?></td>
    <td><?= $d['avg_r'] ? number_format($d['avg_r'], 1) : '—' ?></td>
    <td><?= $d['avg_g'] ? number_format($d['avg_g'], 1) : '—' ?></td>
    <td><?= $d['avg_o'] ? number_format($d['avg_o'], 1) : '—' ?></td>
    <td><?= $d['avg_rep'] ? number_format($d['avg_rep'], 1) : '—' ?></td>
    <td><?= $d['avg_wl'] ? number_format($d['avg_wl'], 1) : '—' ?></td>
    <td><?= $d['avg_def'] ? number_format($d['avg_def'], 1) : '—' ?></td>
</tr>
<?php endforeach; ?>
</table>
<?php endif; ?>

<!-- Текстовые ответы -->
<?php if (!empty($takeaways)): ?>
<h2>Текстовые ответы (<?= count($takeaways) ?>)</h2>
<?php foreach ($takeaways as $t): ?>
<div class="text-answer">
    <div class="date"><?= htmlspecialchars($t['created_at']) ?></div>
    <?php if ($t['takeaway']): ?>
    <div class="takeaway">💡 <?= htmlspecialchars($t['takeaway']) ?></div>
    <?php endif; ?>
    <?php if ($t['comment']): ?>
    <div class="comment">💬 <?= htmlspecialchars($t['comment']) ?></div>
    <?php endif; ?>
</div>
<?php endforeach; ?>
<?php endif; ?>

<!-- Последние ответы -->
<?php if (!empty($recent)): ?>
<h2>Последние ответы</h2>
<table class="recent-table">
<tr>
    <th>Дата</th>
    <th>TG ID</th>
    <th>Week</th>
    <th>Group</th>
    <th>🔴</th>
    <th>🟢</th>
    <th>⚪</th>
    <th>Типичн.</th>
    <th>Баланс</th>
    <th>Дефицит</th>
    <th>Время</th>
    <th>Устройство</th>
</tr>
<?php foreach ($recent as $i => $r): ?>
<tr>
    <td><?= $i + 1 ?></td>
    <td><?= htmlspecialchars($r['created_at']) ?></td>
    <td><span class="tg-id"><?= $r['tg_id'] ? htmlspecialchars($r['tg_id']) : '—' ?></span></td>
    <td><span class="tg-id"><?= $r['week'] ? htmlspecialchars($r['week']) : '—' ?></span></td>
    <td><span class="tg-id"><?= $r['group_id'] ? htmlspecialchars($r['group_id']) : '—' ?></span></td>
    <td><?= $r['cubes_reactive'] ?></td>
    <td><?= $r['cubes_proactive'] ?></td>
    <td><?= $r['cubes_operational'] ?></td>
    <td><?= $r['representative'] ?? '—' ?></td>
    <td><?= $r['work_life'] ?? '—' ?></td>
    <td><?= $r['energy_deficit'] ?? '—' ?></td>
    <td><?= $r['time_total'] ? $r['time_total'] . 'с' : '—' ?></td>
    <td><?= htmlspecialchars($r['device_type'] ?? '') ?></td>
</tr>
<?php endforeach; ?>
</table>
<?php endif; ?>

</body>
</html>
