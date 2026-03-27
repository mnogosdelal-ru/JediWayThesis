<?php
/**
 * Jedi Boxes — Распределение энергии
 * Прототип инструмента оценки продуктивности
 */

// Константа: количество единиц энергии (можно изменить)
define('ENERGY_UNITS', 7);

// Обработка отправки формы
$result = null;
if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['submit'])) {
    $urgent = (int)($_POST['urgent'] ?? 0);
    $main = (int)($_POST['main'] ?? 0);
    $background = (int)($_POST['background'] ?? 0);
    $pool = ENERGY_UNITS - ($urgent + $main + $background);

    // Валидация: сумма не больше ENERGY_UNITS
    if ($urgent + $main + $background <= ENERGY_UNITS && $urgent >= 0 && $main >= 0 && $background >= 0) {
        $result = [
            'urgent' => $urgent,
            'main' => $main,
            'background' => $background,
            'pool' => $pool, // Кубики, оставшиеся в пуле
            'timestamp' => date('Y-m-d H:i:s')
        ];
        // Сохраняем в файл результатов
        saveResult($result);
    }
}

/**
 * Сохранение результата в JSON-файл
 */
function saveResult($result) {
    $dataDir = __DIR__ . '/data';
    $resultsFile = $dataDir . '/results.json';
    
    // Создаем директорию, если не существует
    if (!file_exists($dataDir)) {
        mkdir($dataDir, 0755, true);
    }
    
    // Читаем существующие результаты
    $existingResults = [];
    if (file_exists($resultsFile)) {
        $content = file_get_contents($resultsFile);
        if ($content) {
            $existingResults = json_decode($content, true) ?? [];
        }
    }
    
    // Добавляем ID и метаданные
    $result['id'] = uniqid('jb_', true);
    $result['ip'] = $_SERVER['REMOTE_ADDR'] ?? 'unknown';
    
    // Добавляем новый результат
    $existingResults[] = $result;
    
    // Сохраняем
    file_put_contents($resultsFile, json_encode($existingResults, JSON_UNESCAPED_UNICODE | JSON_PRETTY_PRINT));
}
?>
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Jedi Boxes — Распределение энергии</title>
    <link rel="stylesheet" href="css/styles.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>🧊 Jedi Boxes</h1>
            <p class="subtitle">Куда ушла ваша основная энергия на прошлой неделе?</p>
        </header>

        <?php if ($result): ?>
            <div class="result-box">
                <h2>Ваш результат:</h2>
                <div class="result-grid">
                    <div class="result-item urgent">
                        <span class="label">Реактивное</span>
                        <span class="value"><?= $result['urgent'] ?></span>
                    </div>
                    <div class="result-item main">
                        <span class="label">Проактивное</span>
                        <span class="value"><?= $result['main'] ?></span>
                    </div>
                    <div class="result-item background">
                        <span class="label">Фоновое</span>
                        <span class="value"><?= $result['background'] ?></span>
                    </div>
                </div>
                <?php if (isset($result['pool']) && $result['pool'] > 0): ?>
                <div class="result-pool">
                    <span class="label">⚪ В пуле (не распределено)</span>
                    <span class="value"><?= $result['pool'] ?></span>
                </div>
                <?php endif; ?>
                <p class="result-text">
                    Из <?= ENERGY_UNITS ?> единиц энергии:
                    <strong><?= $result['urgent'] ?> в реактивном</strong>,
                    <strong><?= $result['main'] ?> в проактивном</strong>,
                    <strong><?= $result['background'] ?> в фоновом</strong>
                    <?php if (isset($result['pool']) && $result['pool'] > 0): ?>
                    , <strong><?= $result['pool'] ?> не распределено</strong>
                    <?php endif; ?>.
                </p>
                <a href="index.php" class="btn btn-secondary">Пройти заново</a>
            </div>
        <?php else: ?>
            <div class="instructions">
                <p>
                    <strong>Инструкция:</strong> Распределите <span class="highlight"><?= ENERGY_UNITS ?> кубиков энергии</span> 
                    между тремя ёмкостями в зависимости от того, куда уходила ваша энергия на прошлой неделе.
                </p>
            </div>

            <!-- Легенда зон (описания) -->
            <div class="zones-legend">
                <div class="legend-item urgent">
                    <span class="legend-icon">🔴</span>
                    <div>
                        <strong>Реактивное</strong>
                        <p>Задачи, которые требовали немедленного решения (дедлайны, проблемы, «тушение пожаров»)</p>
                    </div>
                </div>
                <div class="legend-item main">
                    <span class="legend-icon">🟢</span>
                    <div>
                        <strong>Проактивное</strong>
                        <p>Задачи, приближающие к долгосрочным целям (развитие, планирование, обучение, стратегия)</p>
                    </div>
                </div>
                <div class="legend-item background">
                    <span class="legend-icon">⚪</span>
                    <div>
                        <strong>Фоновое</strong>
                        <p>Рутинные задачи, паузы, восстановление, прокрастинация (не вредно, но и не продвигает вперёд)</p>
                    </div>
                </div>
            </div>

            <!-- Зона хранения (источник кубиков) -->
            <div class="pool-container" id="pool">
                <div class="pool-label">Ваша энергия</div>
                <div class="pool-slots">
                    <?php for ($i = 0; $i < ENERGY_UNITS; $i++): ?>
                        <div class="energy-cube" data-id="<?= $i ?>" draggable="true">
                            <span>⚡</span>
                        </div>
                    <?php endfor; ?>
                </div>
                <div class="pool-counter">
                    Осталось: <span id="pool-count"><?= ENERGY_UNITS ?></span>
                </div>
            </div>

            <!-- Форма отправки -->
            <form method="POST" id="energy-form" data-energy-units="<?= ENERGY_UNITS ?>">
                <input type="hidden" name="urgent" id="urgent-count" value="0">
                <input type="hidden" name="main" id="main-count" value="0">
                <input type="hidden" name="background" id="background-count" value="0">

                <!-- Три ёмкости -->
                <div class="zones-container">
                    <!-- Зона 1: Реактивное -->
                    <div class="zone urgent-zone" data-zone="urgent">
                        <div class="zone-header">
                            <span class="zone-icon">🔴</span>
                            <h3>Реактивное</h3>
                        </div>
                        <div class="zone-drop-area" id="urgent-zone">
                            <div class="zone-cubes" id="urgent-cubes"></div>
                        </div>
                        <div class="zone-counter">
                            <span id="urgent-counter">0</span> / <?= ENERGY_UNITS ?>
                        </div>
                    </div>

                    <!-- Зона 2: Проактивное -->
                    <div class="zone main-zone" data-zone="main">
                        <div class="zone-header">
                            <span class="zone-icon">🟢</span>
                            <h3>Проактивное</h3>
                        </div>
                        <div class="zone-drop-area" id="main-zone">
                            <div class="zone-cubes" id="main-cubes"></div>
                        </div>
                        <div class="zone-counter">
                            <span id="main-counter">0</span> / <?= ENERGY_UNITS ?>
                        </div>
                    </div>

                    <!-- Зона 3: Фоновое -->
                    <div class="zone background-zone" data-zone="background">
                        <div class="zone-header">
                            <span class="zone-icon">⚪</span>
                            <h3>Фоновое</h3>
                        </div>
                        <div class="zone-drop-area" id="background-zone">
                            <div class="zone-cubes" id="background-cubes"></div>
                        </div>
                        <div class="zone-counter">
                            <span id="background-counter">0</span> / <?= ENERGY_UNITS ?>
                        </div>
                    </div>
                </div>

                <!-- Кнопка отправки -->
                <div class="form-actions">
                    <button type="submit" class="btn btn-primary" id="submit-btn" disabled>
                        Готово
                    </button>
                    <button type="button" class="btn btn-secondary" id="reset-btn">
                        Сбросить
                    </button>
                </div>

                <p class="form-hint" id="form-hint">
                    Распределите все <?= ENERGY_UNITS ?> кубиков, чтобы продолжить
                </p>
            </form>
        <?php endif; ?>

        <footer>
            <p>Jedi Boxes v1.0 | Исследование продуктивности</p>
        </footer>
    </div>

    <script src="js/drag-drop.js"></script>
</body>
</html>
