<form method="post" action="save.php" id="survey-form">
    <input type="hidden" name="respondent_id" value="<?= htmlspecialchars($respondent_id) ?>">
    <input type="hidden" name="page" value="9">
    <input type="hidden" name="next_page" value="10">
    <input type="hidden" name="vaccines" id="vaccines">
    
    <div class="form-group">
        <p class="help-text">
            «Джедайские вакцины» — это одноразовые настройки для защиты вашего внимания. 
            Для каждой вакцины укажите, в какой мере вы это применили.
        </p>
    </div>
    
    <?php
    // Загружаем конфигурацию вакцин
    $vaccines = include __DIR__ . '/../config/vaccines.php';
    
    // Варианты ответа
    $vaccine_labels = [
        0 => 'Не применил(а)',
        1 => 'Применил(а), но не полностью',
        2 => 'Применил(а) с небольшими исключениями',
        3 => 'Применил(а) по максимуму'
    ];
    
    /**
     * Санитизация HTML: разрешает только безопасные теги
     * 
     * @param string $html HTML-строка для санитизации
     * @return string Санитизированный HTML
     */
    function sanitizeVaccineHtml($html) {
        // Разрешённые теги (без атрибутов для безопасности)
        $allowed_tags = '<strong><b><em><i><br><ul><ol><li><p>';
        return strip_tags($html, $allowed_tags);
    }
    ?>
    
    <?php foreach ($vaccines as $i => $vaccine): ?>
    <div class="vaccine-item">
        <label class="vaccine-label"><?= htmlspecialchars($vaccine['name']) ?></label>
        <div class="help-text vaccine-desc"><?= sanitizeVaccineHtml($vaccine['desc']) ?></div>

        <div class="radio-group-vertical">
            <?php foreach ($vaccine_labels as $v => $label): ?>
            <div class="radio-option-vertical">
                <label>
                    <input type="radio" name="vaccines_<?= $i ?>" value="<?= $v ?>" required onchange="updateVaccinesJson()">
                    <span><?= $label ?></span>
                </label>
            </div>
            <?php endforeach; ?>
        </div>
    </div>
    <?php endforeach; ?>
    
    <div class="buttons">
        <a href="index.php?page=8" class="btn btn-secondary">← Назад</a>
        <button type="submit" class="btn btn-primary">Завершить →</button>
    </div>
</form>

<script>
function updateVaccinesJson() {
    const vaccines = {};
    for (let i = 1; i <= 5; i++) {
        const selected = document.querySelector(`input[name="vaccines_${i}"]:checked`);
        if (selected) {
            vaccines[i] = parseInt(selected.value);
        }
    }
    document.getElementById('vaccines').value = JSON.stringify(vaccines);
}
</script>
