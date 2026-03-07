<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title><?= htmlspecialchars($page_title) ?> - Большое исследование джедайских практик</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="container">
        <?php if ($page < 10): ?>
        <div class="progress-container">
            <div class="progress-bar">
                <div class="progress-fill" style="width: <?= $progress ?>%"></div>
            </div>
            <div class="progress-text"><?= $progress ?>% завершено</div>
        </div>
        <?php endif; ?>

        <div class="restore-notice" id="restore-notice">
            ℹ️ Ваши предыдущие ответы восстановлены. Вы можете продолжить с того места, где остановились.
        </div>

        <h1><?= htmlspecialchars($page_title) ?></h1>
