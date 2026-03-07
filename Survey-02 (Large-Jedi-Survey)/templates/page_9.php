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
    $vaccines = [
        1 => [
            'name' => 'Удалить приложения социальных сетей со своего смартфона',
            'desc' => 'Полное удаление соцсетей. Можно пользоваться через браузер при необходимости.'
        ],
        2 => [
            'name' => 'Отключить красные кружочки на иконках приложений',
            'desc' => 'Убрать уведомления-бейджи. Владельцы Android могут установить oLauncher.'
        ],
        3 => [
            'name' => 'Отключить оповещения о новых сообщениях в чатах',
            'desc' => 'Полное отключение уведомлений из мессенджеров. Надо — позвонят.'
        ],
        4 => [
            'name' => 'Отключить оповещения о новых сообщениях в почте',
            'desc' => 'Отключение push- и звуковых уведомлений о новых письмах.'
        ],
        5 => [
            'name' => 'Отключить push-оповещения от приложений (соцсети, магазины, заправки)',
            'desc' => 'Отключение всех маркетинговых уведомлений, кроме звонков и банковских приложений.'
        ]
    ];
    
    $vaccine_labels = [
        0 => 'Не применил(а)',
        1 => 'Применил(а), но не полностью',
        2 => 'Применил(а) с небольшими исключениями',
        3 => 'Применил(а) по максимуму'
    ];
    ?>
    
    <?php foreach ($vaccines as $i => $vaccine): ?>
    <div class="vaccine-item">
        <label class="vaccine-label"><?= htmlspecialchars($vaccine['name']) ?></label>
        <p class="help-text vaccine-desc"><?= htmlspecialchars($vaccine['desc']) ?></p>

        <div class="radio-group">
            <?php foreach ($vaccine_labels as $v => $label): ?>
            <div class="radio-option">
                <label>
                    <input type="radio" name="vaccines_<?= $i ?>" value="<?= $v ?>" required onchange="updateVaccinesJson()">
                    <?= $label ?>
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
