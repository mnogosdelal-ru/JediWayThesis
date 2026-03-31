/**
 * Jedi Boxes — Drag-and-Drop для опросника валидации
 * Поддержка Desktop (HTML5 DnD) и Mobile (Touch Events)
 * Оптимизировано для мобильных устройств
 */

(function() {
    'use strict';

    const TOTAL_CUBES = 6;

    // Состояние
    let state = {
        pool: TOTAL_CUBES,
        reactive: 0,
        proactive: 0,
        operational: 0
    };

    // DOM элементы
    let elements = {};

    // Для touch
    let draggedElement = null;
    let touchClone = null;
    let touchStartZone = null;

    /**
     * Инициализация
     */
    function init() {
        elements = {
            poolSlots: document.getElementById('pool-slots'),
            availableCount: document.getElementById('available-cubes'),
            zones: {
                reactive: document.querySelector('.zone[data-zone="reactive"]'),
                proactive: document.querySelector('.zone[data-zone="proactive"]'),
                operational: document.querySelector('.zone[data-zone="operational"]')
            },
            dropAreas: {
                reactive: document.getElementById('reactive-zone'),
                proactive: document.getElementById('proactive-zone'),
                operational: document.getElementById('operational-zone')
            },
            cubes: {
                reactive: document.getElementById('reactive-cubes'),
                proactive: document.getElementById('proactive-cubes'),
                operational: document.getElementById('operational-cubes')
            },
            counters: {
                reactive: document.querySelector('.zone[data-zone="reactive"] .count-value'),
                proactive: document.querySelector('.zone[data-zone="proactive"] .count-value'),
                operational: document.querySelector('.zone[data-zone="operational"] .count-value')
            }
        };

        // Создаём кубики
        createCubes();

        // Настраиваем события
        setupDesktopDragAndDrop();
        setupTouchDragAndDrop();

        updateUI();
    }

    /**
     * Создание кубиков в пуле
     */
    function createCubes() {
        elements.poolSlots.innerHTML = '';
        for (let i = 0; i < TOTAL_CUBES; i++) {
            const cube = document.createElement('div');
            cube.className = 'energy-cube';
            cube.dataset.id = i;
            cube.draggable = true;
            cube.innerHTML = '<span>⚡</span>';
            elements.poolSlots.appendChild(cube);
        }
    }

    /**
     * HTML5 Drag and Drop для Desktop
     */
    function setupDesktopDragAndDrop() {
        elements.poolSlots.addEventListener('dragstart', handleDragStart);
        elements.poolSlots.addEventListener('dragend', handleDragEnd);
        elements.poolSlots.addEventListener('dragover', handleDragOver);
        elements.poolSlots.addEventListener('dragleave', handleDragLeave);
        elements.poolSlots.addEventListener('drop', handleDrop);

        Object.keys(elements.zones).forEach(zoneName => {
            const zone = elements.zones[zoneName];
            const dropArea = elements.dropAreas[zoneName];

            dropArea.addEventListener('dragover', handleDragOver);
            dropArea.addEventListener('dragleave', handleDragLeave);
            dropArea.addEventListener('drop', handleDrop);

            // Кубики внутри зон тоже можно перетаскивать
            elements.cubes[zoneName].addEventListener('dragstart', handleDragStart);
            elements.cubes[zoneName].addEventListener('dragend', handleDragEnd);
        });
    }

    /**
     * Touch Events для Mobile
     */
    function setupTouchDragAndDrop() {
        elements.poolSlots.addEventListener('touchstart', handleTouchStart, { passive: false });
        elements.poolSlots.addEventListener('touchmove', handleTouchMove, { passive: false });

        Object.keys(elements.zones).forEach(zoneName => {
            const dropArea = elements.dropAreas[zoneName];
            dropArea.addEventListener('touchstart', handleZoneTouchStart, { passive: false });
            dropArea.addEventListener('touchmove', handleZoneTouchMove, { passive: false });
            dropArea.addEventListener('touchend', handleZoneTouchEnd);
            
            // Перетаскивание из зон
            elements.cubes[zoneName].addEventListener('touchstart', handleTouchStart, { passive: false });
            elements.cubes[zoneName].addEventListener('touchmove', handleTouchMove, { passive: false });
        });

        document.addEventListener('touchend', handleTouchEnd, { passive: false });
    }

    /**
     * Drag Start (Desktop)
     */
    function handleDragStart(e) {
        const cube = e.target.closest('.energy-cube');
        if (!cube) return;

        draggedElement = cube;
        cube.classList.add('dragging');
        e.dataTransfer.effectAllowed = 'move';
        e.dataTransfer.setData('text/plain', cube.dataset.id);

        setTimeout(() => {
            cube.style.visibility = 'hidden';
        }, 0);
    }

    /**
     * Drag End (Desktop)
     */
    function handleDragEnd(e) {
        if (draggedElement) {
            draggedElement.classList.remove('dragging');
            draggedElement.style.visibility = '';
            draggedElement = null;
        }
        
        clearHighlights();
    }

    /**
     * Drag Over (Desktop)
     */
    function handleDragOver(e) {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';

        const dropArea = e.target.closest('.zone-drop-area') || e.target.closest('.pool-slots');
        if (dropArea) {
            dropArea.classList.add('drag-over');
            const zone = dropArea.closest('.zone');
            if (zone) zone.classList.add('drag-over');
        }
    }

    /**
     * Drag Leave (Desktop)
     */
    function handleDragLeave(e) {
        const dropArea = e.target.closest('.zone-drop-area') || e.target.closest('.pool-slots');
        if (dropArea && !dropArea.contains(e.relatedTarget)) {
            dropArea.classList.remove('drag-over');
            dropArea.closest('.zone')?.classList.remove('drag-over');
        }
    }

    /**
     * Drop (Desktop)
     */
    function handleDrop(e) {
        e.preventDefault();

        const dropArea = e.target.closest('.zone-drop-area') || e.target.closest('.pool-slots');
        if (!dropArea || !draggedElement) return;

        clearHighlights();
        dropArea.classList.remove('drag-over');

        const sourceZoneEl = draggedElement.closest('.zone');
        const sourceZoneName = sourceZoneEl ? sourceZoneEl.dataset.zone : 'pool';
        
        const zone = dropArea.closest('.zone');
        const zoneName = zone ? zone.dataset.zone : 'pool';

        if (zoneName === sourceZoneName) {
            draggedElement.style.visibility = '';
            draggedElement.classList.remove('dragging');
            draggedElement = null;
            return;
        }

        draggedElement.style.visibility = '';
        moveCube(draggedElement, zoneName);
        draggedElement = null;
    }

    /**
     * Touch Start
     */
    function handleTouchStart(e) {
        const cube = e.target.closest('.energy-cube');
        if (!cube) return;

        e.preventDefault();
        
        clearHighlights();
        
        const touch = e.touches[0];
        draggedElement = cube;
        
        const sourceZone = cube.closest('.zone');
        touchStartZone = sourceZone ? sourceZone.dataset.zone : 'pool';
        
        draggedElement.style.visibility = 'hidden';
        
        // Создаём клон для визуализации
        touchClone = cube.cloneNode(true);
        touchClone.classList.add('dragging');
        
        const rect = cube.getBoundingClientRect();
        
        touchClone.style.position = 'fixed';
        touchClone.style.left = '0';
        touchClone.style.top = '0';
        touchClone.style.zIndex = '9999';
        touchClone.style.pointerEvents = 'none';
        touchClone.style.opacity = '0.9';
        touchClone.style.width = '60px';
        touchClone.style.height = '60px';
        touchClone.style.display = 'flex';
        touchClone.style.alignItems = 'center';
        touchClone.style.justifyContent = 'center';
        touchClone.style.transition = 'none';
        touchClone.style.willChange = 'transform';
        touchClone.style.transform = `translate(${touch.clientX - 30}px, ${touch.clientY - 30}px)`;
        touchClone.style.visibility = 'visible';

        document.body.appendChild(touchClone);
        
        highlightZoneUnderTouch(touch.clientX, touch.clientY);
    }

    /**
     * Touch Move
     */
    function handleTouchMove(e) {
        if (!touchClone) return;

        e.preventDefault();
        const touch = e.touches[0];
        touchClone.style.transform = `translate(${touch.clientX - 30}px, ${touch.clientY - 30}px)`;
        highlightZoneUnderTouch(touch.clientX, touch.clientY);
    }

    /**
     * Touch End
     */
    function handleTouchEnd(e) {
        if (!draggedElement || !touchClone) return;

        const touch = e.changedTouches[0];

        touchClone.style.visibility = 'hidden';
        const dropTarget = document.elementFromPoint(touch.clientX, touch.clientY);
        touchClone.style.visibility = 'visible';

        const zone = dropTarget?.closest('.zone');
        const poolArea = dropTarget?.closest('.pool-slots');

        if (zone) {
            const zoneName = zone.dataset.zone;
            if (zoneName !== touchStartZone) {
                draggedElement.style.visibility = '';
                moveCube(draggedElement, zoneName);
            } else {
                draggedElement.style.visibility = '';
            }
        } else if (poolArea) {
            if (touchStartZone !== 'pool') {
                draggedElement.style.visibility = '';
                moveCube(draggedElement, 'pool');
            } else {
                draggedElement.style.visibility = '';
            }
        } else {
            draggedElement.style.visibility = '';
        }

        clearHighlights();
        
        if (touchClone) {
            touchClone.remove();
            touchClone = null;
        }

        draggedElement = null;
        touchStartZone = null;
    }

    /**
     * Zone Touch Start
     */
    function handleZoneTouchStart(e) {
        if (!touchClone) return;
        
        const dropArea = e.target.closest('.zone-drop-area');
        if (dropArea) {
            dropArea.classList.add('drag-over');
            e.target.closest('.zone').classList.add('drag-over');
        }
    }

    /**
     * Zone Touch Move
     */
    function handleZoneTouchMove(e) {
        if (!touchClone) return;
        e.preventDefault();
    }

    /**
     * Zone Touch End
     */
    function handleZoneTouchEnd(e) {
        if (!touchClone) return;
        
        const dropArea = e.target.closest('.zone-drop-area');
        if (dropArea) {
            dropArea.classList.remove('drag-over');
            e.target.closest('.zone')?.classList.remove('drag-over');
        }
    }

    /**
     * Подсветка зоны под пальцем
     */
    function highlightZoneUnderTouch(x, y) {
        clearHighlights();

        if (touchClone) {
            touchClone.style.visibility = 'hidden';
        }
        
        const dropTarget = document.elementFromPoint(x, y);
        
        if (touchClone) {
            touchClone.style.visibility = 'visible';
        }

        const zone = dropTarget?.closest('.zone');
        const poolArea = dropTarget?.closest('.pool-slots');

        if (zone) {
            zone.classList.add('drag-over');
            const dropArea = zone.querySelector('.zone-drop-area');
            if (dropArea) dropArea.classList.add('drag-over');
        } else if (poolArea) {
            poolArea.classList.add('drag-over');
        }
    }

    /**
     * Очистка подсветки
     */
    function clearHighlights() {
        Object.values(elements.zones).forEach(zone => {
            zone.classList.remove('drag-over');
            const dropArea = zone.querySelector('.zone-drop-area');
            if (dropArea) dropArea.classList.remove('drag-over');
        });
        elements.poolSlots.classList.remove('drag-over');
    }

    /**
     * Перемещение кубика
     */
    function moveCube(cube, targetZoneName) {
        const sourceZone = cube.closest('.zone-drop-area') || elements.poolSlots;
        const sourceZoneEl = sourceZone.closest('.zone');
        const sourceName = sourceZoneEl ? sourceZoneEl.dataset.zone : 'pool';

        if (sourceName === targetZoneName) return;

        // Обновляем состояние
        if (sourceName === 'pool') {
            state.pool--;
        } else {
            state[sourceName]--;
        }
        
        if (targetZoneName === 'pool') {
            state.pool++;
        } else {
            state[targetZoneName]++;
        }

        // Перемещаем DOM
        if (targetZoneName === 'pool') {
            elements.poolSlots.appendChild(cube);
        } else {
            elements.dropAreas[targetZoneName].querySelector('.zone-cubes').appendChild(cube);
        }

        cube.classList.remove('dragging');
        cube.style.visibility = '';
        cube.style.pointerEvents = '';

        // Обновляем UI
        updateUI();
        
        // Обновляем глобальное состояние для app.js
        if (window.updateCubeCount) {
            window.updateCubeCount('reactive', state.reactive);
            window.updateCubeCount('proactive', state.proactive);
            window.updateCubeCount('operational', state.operational);
        }
    }

    /**
     * Обновление UI
     */
    function updateUI() {
        elements.availableCount.textContent = state.pool;
        
        Object.keys(elements.counters).forEach(zone => {
            elements.counters[zone].textContent = state[zone];
        });
    }

    // Глобальная функция для инициализации
    window.initDragDrop = init;

    // Запуск
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
