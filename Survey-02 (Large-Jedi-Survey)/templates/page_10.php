<?php
// Сохраняем ответы если они есть
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    require_once __DIR__ . '/../src/Survey.php';
    
    $answers = [
        'open_most_useful_practice' => $_POST['open_most_useful_practice'] ?? '',
        'open_other_practices' => $_POST['open_other_practices'] ?? ''
    ];
    
    Survey::savePage($respondent_id, 10, $answers);
}
?>

<form method="post" action="page_10.php" id="thanks-form">
    <div style="text-align: center; padding: 40px 0;">
        <div style="font-size: 72px; margin-bottom: 20px;">🎉</div>

        <h2 style="color: #27ae60; margin-bottom: 20px;">Спасибо за участие!</h2>

        <p style="font-size: 18px; margin-bottom: 30px;">
            Вы успешно завершили опросник.
        </p>

        <p style="font-size: 16px; margin-bottom: 20px;">
            Ещё 2 необязательных вопроса и в качестве награды вы получите некоторую сводку по вашим ответам.
        </p>

        <div style="text-align: left; max-width: 600px; margin: 0 auto 30px;">
            <div class="form-group">
                <label for="open_most_useful_practice">Какая из описанных практик была для вас наиболее полезной? Почему?</label>
                <textarea id="open_most_useful_practice" name="open_most_useful_practice" rows="4" placeholder="Напишите ваш ответ здесь..."></textarea>
            </div>

            <div class="form-group">
                <label for="open_other_practices">Какие еще практики помогают вам лучше справляться с делами, но не были упомянуты в этом опросе?</label>
                <textarea id="open_other_practices" name="open_other_practices" rows="4" placeholder="Напишите ваш ответ здесь..."></textarea>
            </div>
        </div>

        <button type="submit" class="btn btn-primary">Отправить ответы и получить результаты →</button>
    </div>
</form>

<div id="results-container" style="display: none;">
    <?php if ($respondent): ?>
    <div style="text-align: center; padding: 40px 0;">
        <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 30px;">
            <p style="font-size: 16px; margin-bottom: 15px;">
                <strong>🔑 Ваш код респондента:</strong><br>
                <span style="font-size: 32px; color: #3498db; font-family: monospace;"><?= htmlspecialchars($respondent['code']) ?></span>
            </p>
            <p class="help-text">
                Запомните его или сохраните ссылку на эту страницу. Если пойдете учиться на наших программах, этот код вам пригодится, чтобы заново не заполнять опросники.
            </p>
        </div>

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
    <?php endif; ?>
</div>

<script>
// Показываем результаты после отправки формы
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('thanks-form');
    const resultsContainer = document.getElementById('results-container');
    
    if (form) {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(form);
            
            fetch('page_10.php', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                // После сохранения показываем результаты
                form.style.display = 'none';
                resultsContainer.style.display = 'block';
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Ошибка при сохранении ответов');
            });
        });
    }
});
</script>
