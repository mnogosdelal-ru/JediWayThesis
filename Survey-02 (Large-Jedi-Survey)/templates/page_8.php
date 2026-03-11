<form method="post" action="save.php" id="survey-form">
    <input type="hidden" name="respondent_id" value="<?= htmlspecialchars($respondent_id) ?>">
    <input type="hidden" name="page" value="8">
    <input type="hidden" name="next_page" value="9">
    <input type="hidden" name="practices_frequency" id="practices_frequency">
    <input type="hidden" name="practices_quality" id="practices_quality">
    
    <div class="form-group">
        <p class="help-text">
            Для каждой практики укажите: <strong>как часто</strong> вы это делаете и <strong>как</strong> вы это делаете.
        </p>
    </div>
    
    <?php
    // Загружаем конфигурацию практик
    $practices = include __DIR__ . '/../config/practices.php';
    
    // Варианты ответа для частоты
    $freq_labels_weekly = ['Никогда', 'Редко (раз в 1-2 мес)', 'Иногда (1-2 раза в мес)', 'Часто (3-4 раза в мес)', 'Постоянно (каждую неделю)'];
    $freq_labels_daily = ['Никогда', '1-2 раза в месяц', '1-2 раза в неделю', '3-4 раза в неделю', '5-6 раз в неделю', 'Ежедневно'];
    $quality_labels = ['Ровно как написано', 'Почти как в описании', 'Не совсем так, но примерно', 'Совсем не так'];
    
    /**
     * Санитизация HTML: разрешает только безопасные теги
     * 
     * @param string $html HTML-строка для санитизации
     * @return string Санитизированный HTML
     */
    function sanitizePracticeHtml($html) {
        // Разрешённые теги (без атрибутов для безопасности)
        $allowed_tags = '<strong><b><em><i><br><ul><ol><li><p>';
        return strip_tags($html, $allowed_tags);
    }
    ?>
    
    <?php foreach ($practices as $i => $practice):
        $is_weekly = $practice['is_weekly'] ?? false;
        $freq_options = $is_weekly ? $freq_labels_weekly : $freq_labels_daily;
    ?>
    <div class="practice-item">
        <label class="practice-label"><?= htmlspecialchars($practice['name']) ?></label>
        <div class="help-text practice-desc"><?= sanitizePracticeHtml($practice['desc']) ?></div>

        <div>
            <label class="practice-section-label">Как часто вы это делаете?</label>
            <div class="radio-group-vertical">
                <?php foreach ($freq_options as $v => $label): ?>
                <div class="radio-option-vertical">
                    <label>
                        <input type="radio" id="freq_<?= $i ?>_<?= $v+1 ?>" name="practices_frequency_<?= $i ?>" value="<?= $v+1 ?>" required onchange="updatePracticesJson()">
                        <span><?= $label ?></span>
                    </label>
                </div>
                <?php endforeach; ?>
            </div>
        </div>

        <div class="practice-quality-section" id="quality_container_<?= $i ?>" style="display: none;">
            <label class="practice-section-label">Как вы это делаете?</label>
            <div class="scale-group">
                <div class="scale-options">
                    <?php foreach ($quality_labels as $v => $label): ?>
                    <div class="scale-option">
                        <input type="radio" id="qual_<?= $i ?>_<?= $v+1 ?>" name="practices_quality_<?= $i ?>" value="<?= $v+1 ?>" onchange="updatePracticesJson()">
                        <label for="qual_<?= $i ?>_<?= $v+1 ?>" style="font-size: 10px; display: block; margin-top: 5px; line-height: 1.1;"><?= $v+1 ?></label>
                        <span class="scale-caption"><?= $label ?></span>
                    </div>
                    <?php endforeach; ?>
                </div>
            </div>
        </div>
    </div>
    <?php endforeach; ?>
    
    <div class="buttons">
        <a href="index.php?page=7" class="btn btn-secondary">← Назад</a>
        <button type="submit" class="btn btn-primary">Продолжить →</button>
    </div>
</form>

<script>
function updatePracticesJson() {
    const frequency = {};
    const quality = {};
    
    for (let i = 1; i <= 21; i++) {
        const freq = document.querySelector(`input[name="practices_frequency_${i}"]:checked`);
        if (freq) {
            frequency[i] = parseInt(freq.value);
            
            // Показываем/скрываем качество
            const qualityContainer = document.getElementById(`quality_container_${i}`);
            if (parseInt(freq.value) > 1) {
                qualityContainer.style.display = 'block';
            } else {
                qualityContainer.style.display = 'none';
                quality[i] = 0;
            }
        }
        
        const qual = document.querySelector(`input[name="practices_quality_${i}"]:checked`);
        if (qual) {
            quality[i] = parseInt(qual.value);
        }
    }
    
    document.getElementById('practices_frequency').value = JSON.stringify(frequency);
    document.getElementById('practices_quality').value = JSON.stringify(quality);
}

// Инициализация при загрузке
document.addEventListener('DOMContentLoaded', function() {
    updatePracticesJson();
});
</script>
