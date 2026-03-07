<?php
// Сохраняем ответы если они есть
if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['answers'])) {
    require_once __DIR__ . '/../src/Survey.php';
    Survey::savePage($respondent_id, 10, $_POST['answers']);
}
?>

<div style="text-align: center; padding: 40px 0;">
    <div style="font-size: 72px; margin-bottom: 20px;">🎉</div>
    
    <h2 style="color: #27ae60; margin-bottom: 20px;">Спасибо за участие!</h2>
    
    <p style="font-size: 18px; margin-bottom: 30px;">
        Вы успешно завершили опросник.
    </p>
    
    <?php if ($respondent): ?>
    <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 30px;">
        <p style="font-size: 16px; margin-bottom: 15px;">
            <strong>🔑 Ваш код респондента:</strong><br>
            <span style="font-size: 32px; color: #3498db; font-family: monospace;"><?= htmlspecialchars($respondent['code']) ?></span>
        </p>
        <p class="help-text">
            Запомните его или сохраните ссылку на эту страницу. Если пойдете учиться на наших программах, этот код вам пригодится, чтобы заново не заполнять опросники.
        </p>
    </div>
    <?php endif; ?>
    
    <div style="margin-top: 30px;">
        <a href="api.php?code=<?= htmlspecialchars($respondent['code'] ?? '') ?>" class="btn btn-primary" target="_blank">
            📥 Скачать результаты (JSON)
        </a>
    </div>
    
    <div style="margin-top: 20px;">
        <a href="index.php" class="btn btn-secondary">
            🏠 На главную
        </a>
    </div>
</div>
