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
    $mbi_exhaustion = [
        1 => 'Я чувствую себя эмоционально истощённым(ой) из-за работы.',
        2 => 'К концу рабочего дня я чувствую себя как «выжатый лимон».',
        3 => 'Я чувствую усталость, когда встаю утром и должен(на) идти на работу.',
        4 => 'Я чувствую, что работаю с людьми слишком много.',
        5 => 'Я чувствую себя загнанным(ой) в угол работой.'
    ];

    $mbi_cynicism = [
        1 => 'Я стал(а) более циничным(ой) по поводу того, может ли моя работа приносить пользу.',
        2 => 'Я сомневаюсь в значимости своей работы.',
        3 => 'Я стал(а) менее заинтересованным(ой) в своей работе с тех пор, как начал(а) ею заниматься.',
        4 => 'Я не чувствую энтузиазма по поводу своей работы.',
        5 => 'Я разочаровался(ась) в своей работе.'
    ];

    $mbi_efficacy = [
        1 => 'Я могу эффективно решать проблемы, возникающие на работе.',
        2 => 'Я чувствую, что положительно влияю на жизнь других людей через свою работу.',
        3 => 'Я уверен(а) в своей способности понимать нужды коллег/клиентов.',
        4 => 'Я справляюсь с рабочими задачами так же хорошо, как и раньше.',
        5 => 'Я чувствую себя энергичным(ой) на работе.',
        6 => 'Я достиг(ла) многих стоящих целей в своей работе.'
    ];
    ?>

    <?php
    $mbi_labels = [
        0 => 'Никогда',
        1 => 'Очень редко',
        2 => 'Редко',
        3 => 'Иногда',
        4 => 'Часто',
        5 => 'Очень часто',
        6 => 'Каждый день'
    ];
    ?>

    <?php foreach ($mbi_exhaustion as $i => $text): ?>
    <div class="form-group">
        <label><?= htmlspecialchars($text) ?></label>
        <div class="radio-group-vertical" style="margin-top: 10px;">
            <?php foreach ($mbi_labels as $v => $label): ?>
            <div class="radio-option-vertical" style="margin-bottom: 8px;">
                <label style="display: flex; align-items: center; font-weight: normal; cursor: pointer;">
                    <input type="radio" id="exh_<?= $i ?>_<?= $v ?>" name="mbi_exhaustion_<?= $i ?>" value="<?= $v ?>" required onchange="updateMbiJson()" style="margin-right: 10px; width: 18px; height: 18px;">
                    <span><?= $label ?></span>
                </label>
            </div>
            <?php endforeach; ?>
        </div>
    </div>
    <?php endforeach; ?>

    <?php foreach ($mbi_cynicism as $i => $text): ?>
    <div class="form-group">
        <label><?= htmlspecialchars($text) ?></label>
        <div class="radio-group-vertical" style="margin-top: 10px;">
            <?php foreach ($mbi_labels as $v => $label): ?>
            <div class="radio-option-vertical" style="margin-bottom: 8px;">
                <label style="display: flex; align-items: center; font-weight: normal; cursor: pointer;">
                    <input type="radio" id="cyn_<?= $i ?>_<?= $v ?>" name="mbi_cynicism_<?= $i ?>" value="<?= $v ?>" required onchange="updateMbiJson()" style="margin-right: 10px; width: 18px; height: 18px;">
                    <span><?= $label ?></span>
                </label>
            </div>
            <?php endforeach; ?>
        </div>
    </div>
    <?php endforeach; ?>

    <?php foreach ($mbi_efficacy as $i => $text): ?>
    <div class="form-group">
        <label><?= htmlspecialchars($text) ?></label>
        <div class="radio-group-vertical" style="margin-top: 10px;">
            <?php foreach ($mbi_labels as $v => $label): ?>
            <div class="radio-option-vertical" style="margin-bottom: 8px;">
                <label style="display: flex; align-items: center; font-weight: normal; cursor: pointer;">
                    <input type="radio" id="eff_<?= $i ?>_<?= $v ?>" name="mbi_efficacy_<?= $i ?>" value="<?= $v ?>" required onchange="updateMbiJson()" style="margin-right: 10px; width: 18px; height: 18px;">
                    <span><?= $label ?></span>
                </label>
            </div>
            <?php endforeach; ?>
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
    const exhaustion = [];
    const cynicism = [];
    const efficacy = [];
    
    for (let i = 1; i <= 5; i++) {
        const exh = document.querySelector(`input[name="mbi_exhaustion_${i}"]:checked`);
        exhaustion.push(exh ? parseInt(exh.value) : 0);
    }
    
    for (let i = 1; i <= 5; i++) {
        const cyn = document.querySelector(`input[name="mbi_cynicism_${i}"]:checked`);
        cynicism.push(cyn ? parseInt(cyn.value) : 0);
    }
    
    for (let i = 1; i <= 6; i++) {
        const eff = document.querySelector(`input[name="mbi_efficacy_${i}"]:checked`);
        efficacy.push(eff ? parseInt(eff.value) : 0);
    }
    
    document.getElementById('mbi_exhaustion_items').value = JSON.stringify(exhaustion);
    document.getElementById('mbi_cynicism_items').value = JSON.stringify(cynicism);
    document.getElementById('mbi_efficacy_items').value = JSON.stringify(efficacy);
}
</script>
