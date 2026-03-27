<?php
/**
 * Jedi Boxes — Просмотр результатов
 * Простая админ-панель для просмотра собранных данных
 */

// Путь к файлу результатов
define('RESULTS_FILE', __DIR__ . '/data/results.json');

// Простая защита паролем (можно изменить)
define('ADMIN_PASSWORD', 'jedi2026'); // В продакшене используйте более надёжную защиту!

// Проверяем авторизацию
session_start();
$authorized = isset($_SESSION['jb_authorized']) && $_SESSION['authorized'] === true;

// Обработка входа
if (isset($_POST['action']) && $_POST['action'] === 'login') {
    if ($_POST['password'] === ADMIN_PASSWORD) {
        $_SESSION['jb_authorized'] = true;
        $authorized = true;
    } else {
        $error = 'Неверный пароль';
    }
}

// Обработка выхода
if (isset($_GET['logout'])) {
    session_destroy();
    header('Location: results.php');
    exit;
}
?>
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Jedi Boxes — Результаты</title>
    <style>
        :root {
            --color-urgent: #e74c3c;
            --color-main: #27ae60;
            --color-background: #95a5a6;
            --color-primary: #3498db;
            --color-text: #2c3e50;
            --color-white: #ffffff;
        }
        
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f7fa;
            padding: 20px;
            color: var(--color-text);
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: var(--color-white);
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #eee;
        }
        
        h1 { color: var(--color-text); }
        
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            text-decoration: none;
            display: inline-block;
            font-weight: 600;
        }
        
        .btn-primary { background: var(--color-primary); color: white; }
        .btn-danger { background: #e74c3c; color: white; }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .stat-card {
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        
        .stat-card.urgent { background: #fadbd8; }
        .stat-card.main { background: #d5f5e3; }
        .stat-card.background { background: #ebedef; }
        .stat-card.pool { background: #d6dbdf; }
        
        .stat-value {
            font-size: 2.5rem;
            font-weight: bold;
            color: var(--color-text);
        }
        
        .stat-label {
            color: #7f8c8d;
            margin-top: 5px;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        
        th {
            background: #f8f9fa;
            font-weight: 600;
        }
        
        tr:hover { background: #f8f9fa; }
        
        .bar-container {
            display: flex;
            height: 20px;
            border-radius: 10px;
            overflow: hidden;
            background: #eee;
        }
        
        .bar { height: 100%; }
        .bar.urgent { background: var(--color-urgent); }
        .bar.main { background: var(--color-main); }
        .bar.background { background: var(--color-background); }
        .bar.pool { background: var(--color-text-light); }
        
        .login-form {
            max-width: 400px;
            margin: 100px auto;
            padding: 30px;
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        
        .login-form input {
            width: 100%;
            padding: 12px;
            border: 2px solid #eee;
            border-radius: 8px;
            margin-bottom: 15px;
            font-size: 1rem;
        }
        
        .error {
            color: #e74c3c;
            margin-bottom: 15px;
        }
        
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #7f8c8d;
        }
    </style>
</head>
<body>
    <?php if (!$authorized): ?>
        <!-- Форма входа -->
        <div class="login-form">
            <h2 style="text-align: center; margin-bottom: 20px;">🔐 Вход для администратора</h2>
            <?php if (isset($error)): ?>
                <p class="error"><?= htmlspecialchars($error) ?></p>
            <?php endif; ?>
            <form method="POST">
                <input type="hidden" name="action" value="login">
                <input type="password" name="password" placeholder="Введите пароль" required autofocus>
                <button type="submit" class="btn btn-primary" style="width: 100%;">Войти</button>
            </form>
            <p style="text-align: center; margin-top: 20px; color: #7f8c8d;">
                <a href="index.php">← Вернуться к опросу</a>
            </p>
        </div>
    <?php else: ?>
        <!-- Админ-панель -->
        <div class="container">
            <header>
                <h1>📊 Jedi Boxes — Результаты</h1>
                <div>
                    <a href="index.php" class="btn btn-primary">К опросу</a>
                    <a href="?logout=1" class="btn btn-danger">Выйти</a>
                </div>
            </header>
            
            <?php
            // Читаем результаты
            $results = [];
            if (file_exists(RESULTS_FILE)) {
                $content = file_get_contents(RESULTS_FILE);
                if ($content) {
                    $results = json_decode($content, true) ?? [];
                }
            }
            
            // Сортируем по дате (новые первые)
            usort($results, function($a, $b) {
                return strtotime($b['timestamp']) - strtotime($a['timestamp']);
            });
            
            // Считаем статистику
            $total = count($results);
            $avgUrgent = $total > 0 ? array_sum(array_column($results, 'urgent')) / $total : 0;
            $avgMain = $total > 0 ? array_sum(array_column($results, 'main')) / $total : 0;
            $avgBackground = $total > 0 ? array_sum(array_column($results, 'background')) / $total : 0;
            $avgPool = $total > 0 ? array_sum(array_column($results, 'pool')) / $total : 0;
            ?>
            
            <?php if ($total === 0): ?>
                <div class="empty-state">
                    <h2>📭 Пока нет результатов</h2>
                    <p>Результаты опроса появятся здесь после того, как респонденты начнут проходить опрос.</p>
                </div>
            <?php else: ?>
                <!-- Статистика -->
                <div class="stats">
                    <div class="stat-card urgent">
                        <div class="stat-value"><?= round($avgUrgent, 2) ?></div>
                        <div class="stat-label">Среднее: Срочное</div>
                    </div>
                    <div class="stat-card main">
                        <div class="stat-value"><?= round($avgMain, 2) ?></div>
                        <div class="stat-label">Среднее: Главное</div>
                    </div>
                    <div class="stat-card background">
                        <div class="stat-value"><?= round($avgBackground, 2) ?></div>
                        <div class="stat-label">Среднее: Фоновое</div>
                    </div>
                    <div class="stat-card pool">
                        <div class="stat-value"><?= round($avgPool, 2) ?></div>
                        <div class="stat-label">Среднее: В пуле</div>
                    </div>
                </div>
                
                <p style="margin-bottom: 15px;">
                    <strong>Всего ответов:</strong> <?= $total ?>
                </p>
                
                <!-- Таблица результатов -->
                <table>
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Дата</th>
                            <th>Распределение</th>
                            <th>Срочное</th>
                            <th>Главное</th>
                            <th>Фоновое</th>
                            <th>В пуле</th>
                        </tr>
                    </thead>
                    <tbody>
                        <?php foreach ($results as $result): ?>
                            <?php 
                            $pool = $result['pool'] ?? (7 - ($result['urgent'] + $result['main'] + $result['background']));
                            ?>
                            <tr>
                                <td><code><?= htmlspecialchars(substr($result['id'], 0, 12)) ?>...</code></td>
                                <td><?= date('d.m.Y H:i', strtotime($result['timestamp'])) ?></td>
                                <td>
                                    <div class="bar-container" style="width: 200px;">
                                        <div class="bar urgent" style="width: <?= ($result['urgent'] / 7) * 100 ?>%"></div>
                                        <div class="bar main" style="width: <?= ($result['main'] / 7) * 100 ?>%"></div>
                                        <div class="bar background" style="width: <?= ($result['background'] / 7) * 100 ?>%"></div>
                                        <?php if ($pool > 0): ?>
                                        <div class="bar pool" style="width: <?= ($pool / 7) * 100 ?>%"></div>
                                        <?php endif; ?>
                                    </div>
                                </td>
                                <td><strong><?= $result['urgent'] ?></strong></td>
                                <td><strong><?= $result['main'] ?></strong></td>
                                <td><strong><?= $result['background'] ?></strong></td>
                                <td><strong><?= $pool ?></strong></td>
                            </tr>
                        <?php endforeach; ?>
                    </tbody>
                </table>
            <?php endif; ?>
        </div>
    <?php endif; ?>
</body>
</html>
