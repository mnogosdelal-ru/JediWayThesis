<form method="post" action="save.php" id="survey-form">
    <input type="hidden" name="respondent_id" value="<?= htmlspecialchars($respondent_id) ?>">
    <input type="hidden" name="page" value="5">
    <input type="hidden" name="next_page" value="6">
    <input type="hidden" name="swls_items" id="swls_items">
    
    <div class="form-group">
        <p class="help-text">
            Ниже даны утверждения, с которыми Вы можете согласиться или не согласиться. 
            Выразите степень Вашего согласия с каждым из них.
        </p>
    </div>
    
    <?php
    $swls_items = [
        1 => 'В основном моя жизнь близка к идеалу.',
        2 => 'Обстоятельства моей жизни исключительно благоприятны.',
        3 => 'Я полностью удовлетворен моей жизнью.',
        4 => 'У меня есть в жизни то, что мне по-настоящему нужно.',
        5 => 'Если бы мне пришлось жить еще раз, я бы оставил все как есть.'
    ];
    ?>
    
    <?php foreach ($swls_items as $i => $text): ?>
    <div class="form-group">
        <label><?= $i ?>. <?= htmlspecialchars($text) ?></label>
        
        <div class="scale-group">
            <div class="scale-options">
                <?php for ($v = 1; $v <= 7; $v++): ?>
                <div class="scale-option">
                    <input type="radio" id="swls_<?= $i ?>_<?= $v ?>" name="swls_<?= $i ?>" value="<?= $v ?>" required onchange="updateSwlsJson()">
                    <label for="swls_<?= $i ?>_<?= $v ?>"><?= $v ?></label>
                </div>
                <?php endfor; ?>
            </div>
        </div>
    </div>
    <?php endforeach; ?>
    
    <div class="scale-labels" style="margin-top: 20px;">
        <span>1 — Полностью не согласен</span>
        <span>7 — Полностью согласен</span>
    </div>
    
    <div class="buttons">
        <a href="index.php?page=4" class="btn btn-secondary">← Назад</a>
        <button type="submit" class="btn btn-primary">Продолжить →</button>
    </div>
</form>

<script>
function updateSwlsJson() {
    const items = [];
    for (let i = 1; i <= 5; i++) {
        const selected = document.querySelector(`input[name="swls_${i}"]:checked`);
        items.push(selected ? parseInt(selected.value) : 0);
    }
    document.getElementById('swls_items').value = JSON.stringify(items);
}
</script>
