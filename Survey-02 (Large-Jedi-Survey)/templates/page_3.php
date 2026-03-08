<form method="post" action="save.php" id="survey-form">
    <input type="hidden" name="respondent_id" value="<?= htmlspecialchars($respondent_id) ?>">
    <input type="hidden" name="page" value="3">
    <input type="hidden" name="next_page" value="4">
    
    <div class="form-group">
        <p class="help-text">
            Вспомните, чем вы занимались на <strong>РАБОТЕ</strong> за последний месяц.
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
                    <input type="radio" id="work_ui_<?= $v ?>" name="work_urgent_important" value="<?= $v ?>" required>
                    <span><?php
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
        <strong>Что такое работа?</strong><br>
        Работа — это ваша профессиональная деятельность: выполнение служебных обязанностей, работа по найму, собственный бизнес, фриланс, любая занятость, приносящая доход.
    </div>


    <div class="form-group">
        <label for="work_satisfaction">На сколько вы удовлетворены своей работой?</label>

        <div class="scale-group">
            <div class="scale-options">
                <?php for ($i = 1; $i <= 7; $i++): ?>
                <div class="scale-option">
                    <input type="radio" id="work_sat_<?= $i ?>" name="work_satisfaction" value="<?= $i ?>" required>
                    <label for="work_sat_<?= $i ?>"><?= $i ?></label>
                    <?php if ($i == 1): ?>
                    <span class="scale-caption">Ненавижу<br>эту работу</span>
                    <?php elseif ($i == 4): ?>
                    <span class="scale-caption">Нечто<br>среднее</span>
                    <?php elseif ($i == 7): ?>
                    <span class="scale-caption">Работа<br>мечты</span>
                    <?php endif; ?>
                </div>
                <?php endfor; ?>
            </div>
        </div>
    </div>
    
    <div class="buttons">
        <a href="index.php?page=2" class="btn btn-secondary">← Назад</a>
        <button type="submit" class="btn btn-primary">Продолжить →</button>
    </div>
</form>
