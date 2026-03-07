<form method="post" action="save.php" id="survey-form">
    <input type="hidden" name="respondent_id" value="<?= htmlspecialchars($respondent_id) ?>">
    <input type="hidden" name="page" value="7">
    <input type="hidden" name="next_page" value="8">
    <input type="hidden" name="procrastination_items" id="procrastination_items">
    
    <div class="form-group">
        <p class="help-text">
            Оцените, насколько каждое утверждение соответствует вашему поведению.
        </p>
    </div>
    
    <?php
    $procrastination_items = [
        1 => 'Я часто замечаю, что заранее выполняю задачи, которые я собирался сделать в будущем',
        2 => 'Я замечаю, что работа не выполняется в течение нескольких дней, даже если она не требует большего, чем просто сесть и сделать ее',
        3 => 'Когда мне нужно выполнить трудную работу, я жду прилив вдохновения',
        4 => 'Мне обычно приходится спешить, чтобы выполнить задачи вовремя',
        5 => 'Когда приближаются сроки завершения работы, я часто теряю время, занимаясь другими делами',
        6 => 'При подготовке ко встрече я часто ловлю себя на том, что делаю что-то в последнюю минуту',
        7 => 'Часто я долго не приступаю к выполнению задачи, которую мне нужно решить',
        8 => 'Я часто говорю: «Сделаю это завтра»'
    ];
    ?>
    
    <?php foreach ($procrastination_items as $i => $text): ?>
    <div class="form-group">
        <label><?= htmlspecialchars($text) ?></label>
        
        <div class="scale-group">
            <div class="scale-options">
                <?php for ($v = 1; $v <= 4; $v++): ?>
                <div class="scale-option">
                    <input type="radio" id="proc_<?= $i ?>_<?= $v ?>" name="procrastination_<?= $i ?>" value="<?= $v ?>" required onchange="updateProcrastinationJson()">
                    <label for="proc_<?= $i ?>_<?= $v ?>"><?= $v ?></label>
                </div>
                <?php endfor; ?>
            </div>
        </div>
    </div>
    <?php endforeach; ?>
    
    <div class="scale-labels" style="margin-top: 20px;">
        <span>1 — Совсем не соответствует</span>
        <span>4 — Полностью соответствует</span>
    </div>
    
    <div class="buttons">
        <a href="index.php?page=6" class="btn btn-secondary">← Назад</a>
        <button type="submit" class="btn btn-primary">Продолжить →</button>
    </div>
</form>

<script>
function updateProcrastinationJson() {
    const items = [];
    for (let i = 1; i <= 8; i++) {
        const selected = document.querySelector(`input[name="procrastination_${i}"]:checked`);
        items.push(selected ? parseInt(selected.value) : 0);
    }
    document.getElementById('procrastination_items').value = JSON.stringify(items);
}
</script>
