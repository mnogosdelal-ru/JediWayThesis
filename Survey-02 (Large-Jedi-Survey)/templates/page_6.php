<form method="post" action="save.php" id="survey-form">
    <input type="hidden" name="respondent_id" value="<?= htmlspecialchars($respondent_id) ?>">
    <input type="hidden" name="page" value="6">
    <input type="hidden" name="next_page" value="7">
    <input type="hidden" name="mbi_exhaustion_items" id="mbi_exhaustion_items">
    <input type="hidden" name="mbi_cynicism_items" id="mbi_cynicism_items">
    <input type="hidden" name="mbi_efficacy_items" id="mbi_efficacy_items">

    <div class="form-group">
        <p class="help-text">
            Оцените, насколько каждое утверждение соответствует вашему состоянию <strong>в связи с работой</strong>.
        </p>
    </div>

    <?php
    // MBI: 22 items, шкала 0-6 (7 баллов)
    // Субшкала 1: Эмоциональное истощение (9 items): 1, 2, 3, 6, 8, 13, 14, 16, 20
    // Item 6 - обратный
    $mbi_exhaustion = [
        1 => 'К концу рабочей недели я чувствую себя эмоционально опустошенным(ой).',
        2 => 'К концу рабочего дня я чувствую себя как выжатый лимон.',
        3 => 'Я чувствую себя усталым(ой), когда встаю утром и должен (должна) идти на работу.',
        6 => 'Я чувствую себя энергичным и эмоционально воодушевленным человеком.',
        8 => 'Я чувствую угнетенность и апатию.',
        13 => 'У меня много жизненных разочарований.',
        14 => 'Я чувствую равнодушие и потерю интереса ко многому, что радовало меня раньше.',
        16 => 'Мне хочется уединиться и отдохнуть от всего и всех.',
        20 => 'Я чувствую себя на пределе возможностей.'
    ];

    // Субшкала 2: Цинизм/Деперсонализация (5 items): 5, 10, 11, 15, 22
    $mbi_cynicism = [
        5 => 'Меня раздражают подчиненные / коллеги / клиенты, которые жалуются или высказывают свои претензии (недовольство).',
        10 => 'В последнее время я стал(а) более черствым(ой) (бесчувственным) во взаимоотношениях с подчиненными.',
        11 => 'Люди, с которыми мне приходится работать, неинтересны для меня. Они скорее утомляют, чем радуют меня.',
        15 => 'Мне безразлично, что происходит с моими подчиненными / коллегами. Я предпочитаю формальное общение с ними, без лишних эмоций и стремлюсь свести общение с ними до минимума.',
        22 => 'Я проявляю к подчиненным больше внимания и заботы, чем получаю от них признательности и благодарности.'
    ];

    // Субшкала 3: Профессиональная эффективность (8 items): 4, 7, 9, 12, 17, 18, 19, 21
    $mbi_efficacy = [
        4 => 'Результаты моей работы не стоят тех усилий, которые я затрачиваю.',
        7 => 'При разговоре с агрессивными или конфликтными работниками я умею договориться и избежать конфликтов.',
        9 => 'Я легко могу повлиять на продуктивность работы моих подчиненных.',
        12 => 'У меня много планов на будущее, и я верю в их осуществление.',
        17 => 'Я легко могу создать атмосферу доброжелательности и сотрудничества в коллективе.',
        18 => 'Я легко общаюсь на работе со всеми (подчиненными, коллегами, техническим персоналом), независимо от их амбиций, эмоционального состояния и манеры общения.',
        19 => 'Я доволен (довольна) своими жизненными успехами (достижениями).',
        21 => 'Я смогу еще много сделать в своей жизни.'
    ];

    $mbi_labels = [
        0 => 'Никогда',
        1 => 'Очень редко',
        2 => 'Редко',
        3 => 'Иногда',
        4 => 'Часто',
        5 => 'Очень часто',
        6 => 'Ежедневно'
    ];
    ?>

    <?php foreach ($mbi_exhaustion as $i => $text): ?>
    <div class="form-group">
        <label><?= htmlspecialchars($text) ?></label>
        <div class="scale-group">
            <div class="scale-options">
                <?php foreach ($mbi_labels as $v => $label): ?>
                <div class="scale-option">
                    <input type="radio" id="exh_<?= $i ?>_<?= $v ?>" name="mbi_exhaustion_<?= $i ?>" value="<?= $v ?>" required onchange="updateMbiJson()">
                    <label for="exh_<?= $i ?>_<?= $v ?>" style="font-size: 10px; display: block; margin-top: 5px; line-height: 1.1;"><?= $v ?></label>
                    <span class="scale-caption"><?= $label ?></span>
                </div>
                <?php endforeach; ?>
            </div>
        </div>
    </div>
    <?php endforeach; ?>

    <?php foreach ($mbi_cynicism as $i => $text): ?>
    <div class="form-group">
        <label><?= htmlspecialchars($text) ?></label>
        <div class="scale-group">
            <div class="scale-options">
                <?php foreach ($mbi_labels as $v => $label): ?>
                <div class="scale-option">
                    <input type="radio" id="cyn_<?= $i ?>_<?= $v ?>" name="mbi_cynicism_<?= $i ?>" value="<?= $v ?>" required onchange="updateMbiJson()">
                    <label for="cyn_<?= $i ?>_<?= $v ?>" style="font-size: 10px; display: block; margin-top: 5px; line-height: 1.1;"><?= $v ?></label>
                    <span class="scale-caption"><?= $label ?></span>
                </div>
                <?php endforeach; ?>
            </div>
        </div>
    </div>
    <?php endforeach; ?>

    <?php foreach ($mbi_efficacy as $i => $text): ?>
    <div class="form-group">
        <label><?= htmlspecialchars($text) ?></label>
        <div class="scale-group">
            <div class="scale-options">
                <?php foreach ($mbi_labels as $v => $label): ?>
                <div class="scale-option">
                    <input type="radio" id="eff_<?= $i ?>_<?= $v ?>" name="mbi_efficacy_<?= $i ?>" value="<?= $v ?>" required onchange="updateMbiJson()">
                    <label for="eff_<?= $i ?>_<?= $v ?>" style="font-size: 10px; display: block; margin-top: 5px; line-height: 1.1;"><?= $v ?></label>
                    <span class="scale-caption"><?= $label ?></span>
                </div>
                <?php endforeach; ?>
            </div>
        </div>
    </div>
    <?php endforeach; ?>

    <div class="buttons">
        <a href="index.php?page=5" class="btn btn-secondary">← Назад</a>
        <button type="submit" class="btn btn-primary">Продолжить →</button>
    </div>
</form>

<script>
function updateMbiJson() {
    const exhaustion = {};
    const cynicism = {};
    const efficacy = {};

    // Собираем ответы с оригинальными номерами вопросов
    const exhaustion_order = [1, 2, 3, 6, 8, 13, 14, 16, 20];
    const cynicism_order = [5, 10, 11, 15, 22];
    const efficacy_order = [4, 7, 9, 12, 17, 18, 19, 21];

    exhaustion_order.forEach(i => {
        const exh = document.querySelector(`input[name="mbi_exhaustion_${i}"]:checked`);
        exhaustion[i] = exh ? parseInt(exh.value) : 0;
    });

    cynicism_order.forEach(i => {
        const cyn = document.querySelector(`input[name="mbi_cynicism_${i}"]:checked`);
        cynicism[i] = cyn ? parseInt(cyn.value) : 0;
    });

    efficacy_order.forEach(i => {
        const eff = document.querySelector(`input[name="mbi_efficacy_${i}"]:checked`);
        efficacy[i] = eff ? parseInt(eff.value) : 0;
    });

    // Сохраняем как ассоциативные массивы с номерами вопросов
    // PHP функция itemsToAssocArray() правильно обработает эти ключи
    document.getElementById('mbi_exhaustion_items').value = JSON.stringify(exhaustion);
    document.getElementById('mbi_cynicism_items').value = JSON.stringify(cynicism);
    document.getElementById('mbi_efficacy_items').value = JSON.stringify(efficacy);
}
</script>
