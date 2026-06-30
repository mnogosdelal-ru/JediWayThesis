/**
 * Пульс-опрос: одностраничное приложение
 */
document.addEventListener('DOMContentLoaded', () => {
    const sessionId = generateSessionId();
    const appStartTime = Date.now();

    // TG ID, week и group_id из URL
    const urlParams = new URLSearchParams(window.location.search);
    const tgId = urlParams.get('tg_id') || urlParams.get('user_id') || null;
    const week = urlParams.get('week') || null;
    const groupId = urlParams.get('group_id') || null;

    // Показать форму
    document.getElementById('survey-page').classList.add('active');
    document.getElementById('thank-you').classList.remove('active');

    // Кнопка отправки
    document.getElementById('btn-submit').addEventListener('click', async () => {
        const state = window.getCubeState ? window.getCubeState() : null;

        // Валидация обязательных radio-вопросов
        const requiredRadios = [
            { name: 'satisfaction', label: 'Я доволен своим прогрессом за неделю' },
            { name: 'representative', label: 'Показательность недели' },
            { name: 'work_life', label: 'Распределение энергии между работой и личной жизнью' },
            { name: 'pss_1', label: 'PSS-4: контроль жизни' },
            { name: 'pss_2', label: 'PSS-4: уверенность' },
            { name: 'pss_3', label: 'PSS-4: всё идёт как надо' },
            { name: 'pss_4', label: 'PSS-4: трудности' }
        ];

        for (const radio of requiredRadios) {
            const selected = document.querySelector(`input[name="${radio.name}"]:checked`);
            if (!selected) {
                alert(`Пожалуйста, ответьте на вопрос: "${radio.label}"`);
                return;
            }
        }

        const btn = document.getElementById('btn-submit');
        btn.disabled = true;
        btn.textContent = 'Отправляем...';

        try {
            const fd = new FormData();
            fd.append('session_id', sessionId);
            fd.append('tg_id', tgId || '');
            fd.append('week', week || '');
            fd.append('group_id', groupId || '');
            fd.append('cubes_reactive', state.reactive);
            fd.append('cubes_proactive', state.proactive);
            fd.append('cubes_operational', state.operational);
            fd.append('cubes_pool', state.pool);
            fd.append('time_total', Math.round((Date.now() - appStartTime) / 1000));

            // Радио
            const sat = document.querySelector('input[name="satisfaction"]:checked');
            if (sat) fd.append('satisfaction', sat.value);
            const rep = document.querySelector('input[name="representative"]:checked');
            if (rep) fd.append('representative', rep.value);
            const wl = document.querySelector('input[name="work_life"]:checked');
            if (wl) fd.append('work_life', wl.value);

            // PSS-4
            for (let i = 1; i <= 4; i++) {
                const pss = document.querySelector(`input[name="pss_${i}"]:checked`);
                if (pss) fd.append(`pss_${i}`, pss.value);
            }

            // Текст
            fd.append('takeaway', (document.getElementById('takeaway')?.value || '').trim());
            fd.append('comment', (document.getElementById('comment')?.value || '').trim());

            const resp = await fetch('api.php', { method: 'POST', body: fd });
            const text = await resp.text();

            let data;
            try {
                data = JSON.parse(text);
            } catch (e) {
                // Не JSON — возможно HTML-ошибка
                console.error('Non-JSON response:', text.substring(0, 200));
                data = { success: false, error: 'Ошибка сервера' };
            }

            if (data.success) {
                document.getElementById('survey-page').classList.remove('active');
                document.getElementById('thank-you').classList.add('active');
                window.scrollTo({ top: 0, behavior: 'smooth' });
            } else if (data.error && data.error.includes('уже сохранён')) {
                // Уже отправлено — показываем "спасибо"
                document.getElementById('survey-page').classList.remove('active');
                document.getElementById('thank-you').classList.add('active');
                window.scrollTo({ top: 0, behavior: 'smooth' });
            } else {
                alert('Ошибка сохранения: ' + (data.error || 'неизвестная'));
                btn.disabled = false;
                btn.textContent = 'Отправить';
            }
        } catch (e) {
            console.error(e);
            alert('Ошибка сети. Попробуйте ещё раз.');
            btn.disabled = false;
            btn.textContent = 'Отправить';
        }
    });
});

function generateSessionId() {
    const arr = new Uint8Array(16);
    crypto.getRandomValues(arr);
    return Array.from(arr, b => b.toString(16).padStart(2, '0')).join('');
}
