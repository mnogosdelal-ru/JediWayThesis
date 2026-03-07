<form method="post" action="save.php" id="survey-form">
    <input type="hidden" name="respondent_id" value="<?= htmlspecialchars($respondent_id) ?>">
    <input type="hidden" name="page" value="4">
    <input type="hidden" name="next_page" value="5">
    <input type="hidden" name="mijs_items" id="mijs_items">
    
    <div class="form-group">
        <p class="help-text">
            Оцените, насколько каждое утверждение соответствует вашей ситуации <strong>за последний месяц</strong>.
        </p>
    </div>
    
    <?php
    $mijs_items = [
        1 => 'Я чувствую, что мои ресурсы (время и энергия) исчерпываются на срочное',
        2 => 'Срочные дела мешают мне развиваться в нужном мне направлении',
        3 => 'Я живу в режиме "тушения пожаров" большую часть времени',
        4 => 'Мой потенциал не реализуется из-за необходимости постоянно реагировать на срочное',
        5 => 'Срочные дела забирают мою лучшую энергию, важное получает остатки',
        6 => 'Я успеваю завершить важные этапы проектов до того, как они превращаются в «горящие» задачи',
        7 => 'Каждый день я вношу хотя бы небольшой вклад в мои значимые цели',
        8 => 'Я не чувствую вины, когда говорю «нет» задачам, которые не соответствуют моим целям',
        9 => 'Я предпочитаю делать важные задачи качественно, даже если это означает, что на срочный запрос я отвечу не первым',
        10 => 'Я довожу важные дела до конца, не бросая их на полпути из-за появления срочных дел',
        11 => 'Мне удается сохранять концентрацию на важном, даже если у коллег хаос и все «горит»',
        12 => 'Каждый день я могу сформулировать, какое самое важное дело я сделал сегодня'
    ];
    ?>
    
    <?php foreach ($mijs_items as $i => $text): ?>
    <div class="form-group">
        <label><?= $i ?>. <?= htmlspecialchars($text) ?></label>
        
        <div class="scale-group">
            <div class="scale-options">
                <?php for ($v = 1; $v <= 5; $v++): ?>
                <div class="scale-option">
                    <input type="radio" id="mijs_<?= $i ?>_<?= $v ?>" name="mijs_<?= $i ?>" value="<?= $v ?>" required onchange="updateMijsJson()">
                    <label for="mijs_<?= $i ?>_<?= $v ?>"><?= $v ?></label>
                </div>
                <?php endfor; ?>
            </div>
        </div>
        <div class="scale-labels">
            <span>1 — Совсем не соответствует</span>
            <span>5 — Полностью соответствует</span>
        </div>
    </div>
    <?php endforeach; ?>
    
    <div class="buttons">
        <a href="index.php?page=3" class="btn btn-secondary">← Назад</a>
        <button type="submit" class="btn btn-primary">Продолжить →</button>
    </div>
</form>

<script>
function updateMijsJson() {
    const items = [];
    for (let i = 1; i <= 12; i++) {
        const selected = document.querySelector(`input[name="mijs_${i}"]:checked`);
        items.push(selected ? parseInt(selected.value) : 0);
    }
    document.getElementById('mijs_items').value = JSON.stringify(items);
}
</script>
