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
    // Чередуем негативные (1-5) и позитивные (6-12) items для предотвращения bias
    // Порядок: 1(N), 6(P), 2(N), 7(P), 3(N), 8(P), 4(N), 9(P), 5(N), 10(P), 11(P), 12(P)
    // N = negatively worded (суммируются напрямую), P = positively worded (инвертируются)
    $mijs_items = [
        1 => 'Я чувствую, что мои ресурсы (время и энергия) исчерпываются на срочное',
        6 => 'Я успеваю завершить важные этапы проектов до того, как они превращаются в «горящие» задачи',
        2 => 'Срочные дела мешают мне развиваться в нужном мне направлении',
        7 => 'Каждый день я вношу хотя бы небольшой вклад в мои значимые цели',
        3 => 'Я живу в режиме "тушения пожаров" большую часть времени',
        8 => 'Я не чувствую вины, когда говорю «нет» задачам, которые не соответствуют моим целям',
        4 => 'Мой потенциал не реализуется из-за необходимости постоянно реагировать на срочное',
        9 => 'Я предпочитаю делать важные задачи качественно, даже если это означает, что на срочный запрос я отвечу не первым',
        5 => 'Срочные дела забирают мою лучшую энергию, важное получает остатки',
        10 => 'Я довожу важные дела до конца, не бросая их на полпути из-за появления срочных дел',
        11 => 'Мне удается сохранять концентрацию на важном, даже если у коллег хаос и все «горит»',
        12 => 'Каждый день я могу сформулировать, какое самое важное дело я сделал сегодня'
    ];
    
    // Маппинг оригинальных номеров для сохранения правильного кодирования
    // Позиция в массиве → оригинальный номер item
    $mijs_original_order = [1, 6, 2, 7, 3, 8, 4, 9, 5, 10, 11, 12];
    ?>
    
    <?php foreach ($mijs_items as $i => $text): ?>
    <div class="form-group">
        <label><?= htmlspecialchars($text) ?></label>

        <div class="scale-group">
            <div class="scale-options">
                <?php for ($v = 1; $v <= 5; $v++): ?>
                <div class="scale-option">
                    <input type="radio" id="mijs_<?= $i ?>_<?= $v ?>" name="mijs_<?= $i ?>" value="<?= $v ?>" required onchange="updateMijsJson()">
                    <label for="mijs_<?= $i ?>_<?= $v ?>"><?= $v ?></label>
                    <?php if ($v == 1): ?>
                    <span class="scale-caption">Совсем не<br>соответствует</span>
                    <?php elseif ($v == 3): ?>
                    <span class="scale-caption">Нечто<br>среднее</span>
                    <?php elseif ($v == 5): ?>
                    <span class="scale-caption">Полностью<br>соответствует</span>
                    <?php endif; ?>
                </div>
                <?php endfor; ?>
            </div>
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
    // Сохраняем ответы в порядке оригинальных номеров items (1-12)
    // Даже если вопросы отображаются вперемешку
    const items = new Array(13).fill(0); // индекс 0 не используется
    
<?php foreach ($mijs_items as $original_id => $text): ?>
    const selected_<?= $original_id ?> = document.querySelector(`input[name="mijs_<?= $original_id ?>"]:checked`);
    items[<?= $original_id ?>] = selected_<?= $original_id ?> ? parseInt(selected_<?= $original_id ?>.value) : 0;
<?php endforeach; ?>
    
    // Отправляем только items 1-12 (без нулевого элемента)
    document.getElementById('mijs_items').value = JSON.stringify(items.slice(1));
}
</script>
