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
        
        <div class="radio-group">
            <div class="radio-option">
                <label>
                    <input type="radio" name="personal_urgent_important" value="1" required>
                    1 — Практически все задачи были срочными
                </label>
            </div>
            <div class="radio-option">
                <label>
                    <input type="radio" name="personal_urgent_important" value="2">
                    2 — Больше срочных, но были и важные
                </label>
            </div>
            <div class="radio-option">
                <label>
                    <input type="radio" name="personal_urgent_important" value="3">
                    3 — Примерно поровну срочных и важных
                </label>
            </div>
            <div class="radio-option">
                <label>
                    <input type="radio" name="personal_urgent_important" value="4">
                    4 — Больше важных, но были и срочные
                </label>
            </div>
            <div class="radio-option">
                <label>
                    <input type="radio" name="personal_urgent_important" value="5">
                    5 — Практически все задачи были важными
                </label>
            </div>
        </div>
    </div>
    
    <p style="background: #f8f9fa; padding: 15px; border-radius: 4px; margin-bottom: 20px;">
        <strong>Что такое личная жизнь?</strong><br>
        Личная жизнь — это всё, что не связано с вашей работой: семья, отношения, отдых, хобби, домашние обязанности, забота о себе.
    </p>

    <div class="buttons">
        <a href="index.php?page=1" class="btn btn-secondary">← Назад</a>
        <button type="submit" class="btn btn-primary">Продолжить →</button>
    </div>
</form>
