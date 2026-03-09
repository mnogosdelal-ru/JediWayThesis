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
    // MBI: 22 items, 7-балльная шкала (1-7)
    // Субшкала 1: Эмоциональное истощение (9 items): 1, 2, 3, 6, 8, 13, 14, 16, 20
    $mbi_exhaustion = [
        1 => 'Я чувствую себя эмоционально опустошенным.',
        2 => 'После работы я чувствую себя как «выжатый лимон».',
        3 => 'Утром я чувствую усталость и нежелание идти на работу.',
        6 => 'После работы на некоторое время хочется уединиться от всех и всего.',
        8 => 'Я чувствую угнетенность и апатию.',
        13 => 'Моя работа все больше меня разочаровывает.',
        14 => 'Мне кажется, что я слишком много работаю.',
        16 => 'Мне хочется уединиться и отдохнуть от всего и всех.',
        20 => 'Я чувствую равнодушие и потерю интереса ко многому, что радовало меня в моей работе.'
    ];

    // Субшкала 2: Деперсонализация/Цинизм (5 items): 5, 10, 11, 15, 22
    $mbi_cynicism = [
        5 => 'Я чувствую, что общаюсь с некоторыми подчиненными и коллегами как с предметами (без теплоты и расположения к ним).',
        10 => 'В последнее время я стал более «черствым» по отношению к тем, с кем работаю.',
        11 => 'Я замечаю, что моя работа ожесточает меня.',
        15 => 'Бывает, что мне действительно безразлично то, что происходит с некоторыми моими подчиненными и коллегами.',
        22 => 'В последнее время мне кажется, что коллеги и подчиненные все чаще перекладывают на меня груз своих проблем и обязанностей.'
    ];

    // Субшкала 3: Профессиональная эффективность (8 items): 4, 7, 9, 12, 17, 18, 19, 21
    $mbi_efficacy = [
        4 => 'Я хорошо понимаю, что чувствуют мои подчиненные и коллеги, и стараюсь учитывать это в интересах дела.',
        7 => 'Я умею находить правильное решение в конфликтных ситуациях, возникающих при общении с коллегами.',
        9 => 'Я уверен, что моя работа нужна людям.',
        12 => 'У меня много планов на будущее, и я верю в их осуществление.',
        17 => 'Я легко могу создать атмосферу доброжелательности и сотрудничества в коллективе.',
        18 => 'Во время работы я чувствую приятное оживление.',
        19 => 'Благодаря своей работе я уже сделал в жизни много действительно ценного.',
        21 => 'На работе я спокойно справляюсь с эмоциональными проблемами.'
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

    // Сохраняем как массивы для совместимости со старым кодом расчёта баллов
    document.getElementById('mbi_exhaustion_items').value = JSON.stringify(Object.values(exhaustion));
    document.getElementById('mbi_cynicism_items').value = JSON.stringify(Object.values(cynicism));
    document.getElementById('mbi_efficacy_items').value = JSON.stringify(Object.values(efficacy));
}
</script>
