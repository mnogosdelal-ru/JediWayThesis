<form method="post" action="save.php" id="survey-form">
    <input type="hidden" name="respondent_id" value="<?= htmlspecialchars($respondent_id) ?>">
    <input type="hidden" name="page" value="0">
    <input type="hidden" name="next_page" value="1">
    
    <div class="form-group">
        <p><strong>Участие в этом исследовании добровольное.</strong> Вы можете прекратить участие в любой момент.</p>
        
        <ul style="margin: 20px 0 20px 20px;">
            <li>Данные анонимны и будут использованы только в исследовательских целях</li>
            <li>Результаты будут опубликованы в агрегированном виде</li>
            <li>Контакт исследователя: maxim.dorofeev@mnogosdelal.ru</li>
        </ul>
        
        <p>После заполнения всех вопросов вы сможете ознакомиться с краткой интерпретацией своих результатов и также получите ссылку на общие статистические результаты, чтобы сравнить себя на фоне других респондентов.</p>
    </div>
    
    <div class="form-group">
        <label class="checkbox-option">
            <input type="checkbox" name="consent_given" value="1" required>
            <span>Я прочитал(а) и согласен(на) участвовать в исследовании</span>
        </label>
    </div>
    
    <div class="buttons">
        <div></div> <!-- Пустой div для выравнивания -->
        <button type="submit" class="btn btn-primary">Продолжить →</button>
    </div>
</form>
