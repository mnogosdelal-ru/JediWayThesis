    </div>
    
    <script>
        const RESPONDENT_ID = '<?= htmlspecialchars($respondent_id ?? '') ?>';
        const CURRENT_PAGE = <?= $page ?? 0 ?>;

        // Показать уведомление о восстановлении
        function showRestoreNotice() {
            const notice = document.getElementById('restore-notice');
            notice.classList.add('visible');
            setTimeout(() => {
                notice.classList.remove('visible');
            }, 5000);
        }
        
        // Сохранение данных в localStorage
        function saveToLocalStorage(page, formData) {
            const key = `survey_page_${page}`;
            localStorage.setItem(key, JSON.stringify(formData));
        }
        
        // Восстановление данных из localStorage
        function restoreFromLocalStorage(page) {
            const key = `survey_page_${page}`;
            const data = localStorage.getItem(key);
            if (!data) return null;
            return JSON.parse(data);
        }
        
        // Сохранить текущую форму перед переходом
        function saveCurrentForm() {
            const form = document.querySelector('#survey-form');
            if (!form) return;
            
            const formData = new FormData(form);
            const data = {};
            formData.forEach((value, key) => {
                data[key] = value;
            });
            
            saveToLocalStorage(CURRENT_PAGE, data);
        }
        
        // Восстановить форму из localStorage
        function restoreForm(page) {
            const data = restoreFromLocalStorage(page);
            if (!data) return false;
            
            let hasData = false;
            
            for (const [key, value] of Object.entries(data)) {
                // Пропускаем скрытые поля
                if (key.startsWith('_')) continue;
                
                const input = document.querySelector(`[name="${key}"]`);
                if (!input) continue;
                
                hasData = true;
                
                if (input.type === 'radio') {
                    const radio = document.querySelector(`[name="${key}"][value="${value}"]`);
                    if (radio) radio.checked = true;
                } else if (input.type === 'checkbox') {
                    input.checked = (value == '1' || value == 'true');
                } else {
                    input.value = value;
                }
            }
            
            // Восстановить скрытые JSON поля и UI
            if (page === 4 && data.mijs_items) {
                updateMijsJson();
            }
            if (page === 5 && data.swls_items) {
                updateSwlsJson();
            }
            if (page === 6) {
                updateMbiJson();
            }
            if (page === 7 && data.procrastination_items) {
                updateProcrastinationJson();
            }
            if (page === 8) {
                updatePracticesJson();
            }
            if (page === 9 && data.vaccines) {
                updateVaccinesJson();
            }
            
            return hasData;
        }
        
        // Отправка формы с автосохранением
        function submitForm() {
            const form = document.querySelector('#survey-form');
            if (!form) return Promise.reject('Form not found');

            const submitBtn = form.querySelector('button[type="submit"]');
            const originalText = submitBtn.textContent;
            submitBtn.disabled = true;
            submitBtn.textContent = 'Сохранение...';

            return new Promise((resolve, reject) => {
                const formData = new FormData(form);
                
                // Логируем для отладки
                console.log('Submitting form:', {
                    respondent_id: formData.get('respondent_id'),
                    page: formData.get('page'),
                    session_respondent_id: window.RESPONDENT_ID
                });

                fetch('save.php', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success && data.redirect) {
                        // Если пришёл новый respondent_id - обновляем
                        if (data.respondent_id) {
                            window.RESPONDENT_ID = data.respondent_id;
                        }
                        // Очистить localStorage для текущей страницы
                        localStorage.removeItem(`survey_page_${CURRENT_PAGE}`);
                        resolve(data);
                    } else {
                        reject(new Error(data.error || 'Ошибка сохранения'));
                    }
                })
                .catch(error => {
                    reject(error);
                })
                .finally(() => {
                    submitBtn.disabled = false;
                    submitBtn.textContent = originalText;
                });
            });
        }
        
        // Инициализация при загрузке
        document.addEventListener('DOMContentLoaded', function() {
            // Не применяем логику footer на странице 10 (Спасибо)
            if (CURRENT_PAGE >= 10) return;
            
            const form = document.querySelector('#survey-form');
            if (!form) return;
            
            // Восстановить данные при загрузке страницы
            const restored = restoreForm(CURRENT_PAGE);
            if (restored) {
                showRestoreNotice();
            }
            
            // Сохранить при нажатии на "Назад"
            const backButtons = document.querySelectorAll('.btn-secondary[href^="index.php?page="]');
            backButtons.forEach(button => {
                button.addEventListener('click', function(e) {
                    saveCurrentForm();
                    // Переход произойдёт после сохранения
                });
            });
            
            // Отправка формы
            form.addEventListener('submit', function(e) {
                e.preventDefault();
                
                submitForm()
                    .then(data => {
                        window.location.href = data.redirect;
                    })
                    .catch(error => {
                        alert('Ошибка сохранения: ' + error.message);
                    });
            });
            
            // Автосохранение при изменении полей (каждые 30 секунд)
            let autoSaveInterval;
            if (CURRENT_PAGE < 10) {
                autoSaveInterval = setInterval(() => {
                    saveCurrentForm();
                }, 30000);
                
                // Сохранить при закрытии вкладки
                window.addEventListener('beforeunload', () => {
                    clearInterval(autoSaveInterval);
                    saveCurrentForm();
                });
                
                // Сохранить при изменении любого поля
                form.addEventListener('change', () => {
                    saveCurrentForm();
                });
            }
        });
    </script>
</body>
</html>
