<form method="post" action="save.php" id="survey-form">
    <input type="hidden" name="respondent_id" value="<?= htmlspecialchars($respondent_id) ?>">
    <input type="hidden" name="page" value="2">
    <input type="hidden" name="next_page" value="3">
    
    <div class="form-group">
        <p class="help-text">
            Вспомните, чем вы занимались в своей <strong>ЛИЧНОЙ ЖИЗНИ</strong> за последний месяц.
        </p>
    </div>
    
    <div class="form-group">
        <label>Как в среднем распределялись ваши задачи между срочными и важными?</label>
        
        <p class="help-text">
            <strong>Срочные задачи</strong> — требуют немедленного решения, чтобы избежать негативных последствий.<br>
            <strong>Важные задачи</strong> — приближают вас к долгосрочным целям, но при отсрочке не наносят немедленный вред.
        </p>
        
        <div class="radio-group-vertical">
            <?php for ($v = 1; $v <= 5; $v++): ?>
            <div class="radio-option-vertical">
                <label>
                    <input type="radio" id="personal_<?= $v ?>" name="personal_urgent_important" value="<?= $v ?>" required>
                    <span><?= $v ?> — <?php
                        if ($v == 1) echo 'Практически все задачи были срочными';
                        elseif ($v == 2) echo 'Больше срочных, но были и важные';
                        elseif ($v == 3) echo 'Примерно поровну срочных и важных';
                        elseif ($v == 4) echo 'Больше важных, но были и срочные';
                        elseif ($v == 5) echo 'Практически все задачи были важными';
                    ?></span>
                </label>
            </div>
            <?php endfor; ?>
        </div>
    </div>

    <div class="help-box">
        <strong>Что такое личная жизнь?</strong><br>
        Личная жизнь — это всё, что не связано с вашей работой: семья, отношения, отдых, хобби, домашние обязанности, забота о себе.
    </div>

    <div class="buttons">
        <a href="index.php?page=1" class="btn btn-secondary">← Назад</a>
        <button type="submit" class="btn btn-primary">Продолжить →</button>
    </div>
</form>
