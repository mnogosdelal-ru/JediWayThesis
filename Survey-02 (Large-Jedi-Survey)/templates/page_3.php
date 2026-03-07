<form method="post" action="save.php" id="survey-form">
    <input type="hidden" name="respondent_id" value="<?= htmlspecialchars($respondent_id) ?>">
    <input type="hidden" name="page" value="3">
    <input type="hidden" name="next_page" value="4">
    
    <div class="form-group">
        <p class="help-text">
            Вспомните, чем вы занимались на <strong>РАБОТЕ</strong> за последний месяц.
        </p>
        
        <p style="background: #f8f9fa; padding: 15px; border-radius: 4px; margin-bottom: 20px;">
            <strong>Что такое работа?</strong><br>
            Работа — это ваша профессиональная деятельность: выполнение служебных обязанностей, работа по найму, собственный бизнес, фриланс, любая занятость, приносящая доход.
        </p>
    </div>
    
    <div class="form-group">
        <label>Как в среднем распределялись ваши задачи между срочными и важными?</label>
        
        <p class="help-text">
            <strong>Срочные задачи</strong> — требуют немедленного решения, чтобы избежать негативных последствий.<br>
            <strong>Важные задачи</strong> — приближают вас к долгосрочным целям, но при отсрочке не наносят немедленный вред.
        </p>
        
        <div class="radio-group">
            <div class="radio-option">
                <label>
                    <input type="radio" name="work_urgent_important" value="0" required>
                    0 — Я сейчас не работаю
                </label>
            </div>
            <div class="radio-option">
                <label>
                    <input type="radio" name="work_urgent_important" value="1">
                    1 — Практически все задачи были срочными
                </label>
            </div>
            <div class="radio-option">
                <label>
                    <input type="radio" name="work_urgent_important" value="2">
                    2 — Больше срочных, но были и важные
                </label>
            </div>
            <div class="radio-option">
                <label>
                    <input type="radio" name="work_urgent_important" value="3">
                    3 — Примерно поровну срочных и важных
                </label>
            </div>
            <div class="radio-option">
                <label>
                    <input type="radio" name="work_urgent_important" value="4">
                    4 — Больше важных, но были и срочные
                </label>
            </div>
            <div class="radio-option">
                <label>
                    <input type="radio" name="work_urgent_important" value="5">
                    5 — Практически все задачи были важными
                </label>
            </div>
        </div>
    </div>
    
    <div class="form-group">
        <label for="work_satisfaction">На сколько вы удовлетворены своей работой?</label>
        
        <div class="scale-group">
            <div class="scale-options">
                <?php for ($i = 0; $i <= 10; $i++): ?>
                <div class="scale-option">
                    <input type="radio" id="work_sat_<?= $i ?>" name="work_satisfaction" value="<?= $i ?>" required>
                    <label for="work_sat_<?= $i ?>"><?= $i ?></label>
                </div>
                <?php endfor; ?>
            </div>
            <div class="scale-labels">
                <span>0 — Ненавижу эту "галеру"!</span>
                <span>10 — Не могу представить лучшей работы</span>
            </div>
        </div>
    </div>
    
    <div class="buttons">
        <a href="index.php?page=2" class="btn btn-secondary">← Назад</a>
        <button type="submit" class="btn btn-primary">Продолжить →</button>
    </div>
</form>
