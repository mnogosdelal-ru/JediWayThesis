/**
 * ProductivityControl - виджет для операционализации понятия продуктивности
 * 
 * Создаёт интерактивную матрицу с бегунками для распределения задач по категориям.
 * 
 * Использование:
 *   const container = document.getElementById('my-container');
 *   const control = new ProductivityControl(container, {
 *     size: 300,
 *     onChange: (values) => console.log(values)
 *   });
 * 
 *   // Получить значения
 *   const values = control.getValues();
 * 
 *   // Установить значения
 *   control.setValues({ tl: 25, tr: 25, bl: 25, br: 25 });
 */
class ProductivityControl {
    constructor(container, options = {}) {
        this.container = container;
        this.options = {
            size: options.size || 300,
            onChange: options.onChange || null,
            initialValues: options.initialValues || { vSplit: 50, hSplitTop: 50, hSplitBottom: 50 }
        };

        // Состояние (в процентах 0-100)
        this.state = { ...this.options.initialValues };

        // DOM элементы (будут созданы в render)
        this.elements = {};

        this.render();
        this.bindEvents();
        this.updateUI();
    }

    /**
     * Создаёт DOM-структуру виджета
     */
    render() {
        const size = this.options.size;
        
        // Инъекция стилей (только один раз)
        if (!document.getElementById('productivity-control-styles')) {
            const style = document.createElement('style');
            style.id = 'productivity-control-styles';
            style.textContent = this.getStyles();
            document.head.appendChild(style);
        }

        // Основной контейнер
        this.container.innerHTML = `
            <div class="pc-widget" style="--pc-size: ${size}px;">
                <!-- Общая надпись над горизонтальными подписями -->
                <div class="pc-h-title-top">
                    <span>То, что я это сделал ...</span>
                </div>

                <!-- Подписи сверху -->
                <div class="pc-labels-top">
                    <span>ни на что не повлияло</span>
                    <span>приблизило меня к моим целям</span>
                </div>

                <!-- Основной layout -->
                <div class="pc-layout">
                    <!-- Левая вертикальная надпись -->
                    <div class="pc-v-title">
                        <span>Если бы я это не сделал(а), то ...</span>
                    </div>

                    <!-- Вертикальные подписи -->
                    <div class="pc-v-labels">
                        <div class="pc-v-label-top"><span>ничего бы не поменялось</span></div>
                        <div class="pc-v-label-bottom"><span>произошло бы что-то плохое</span></div>
                    </div>

                    <!-- Вертикальный бегунок -->
                    <div class="pc-v-slider" id="pc-v-slider">
                        <div class="pc-v-track"></div>
                        <div class="pc-v-handle" id="pc-v-handle"></div>
                    </div>

                    <!-- Матрица -->
                    <div class="pc-matrix-wrapper">
                        <!-- Горизонтальный бегунок сверху -->
                        <div class="pc-h-slider-top" id="pc-h-slider-top">
                            <div class="pc-h-track">
                                <div class="pc-h-thumb" id="pc-h-thumb-top"></div>
                            </div>
                        </div>

                        <!-- Квадрат матрицы -->
                        <div class="pc-matrix" id="pc-matrix">
                            <div class="pc-zone pc-zone-tl" id="pc-zone-tl"></div>
                            <div class="pc-zone pc-zone-tr" id="pc-zone-tr"></div>
                            <div class="pc-zone pc-zone-bl" id="pc-zone-bl"></div>
                            <div class="pc-zone pc-zone-br" id="pc-zone-br"></div>
                        </div>

                        <!-- Горизонтальный бегунок снизу -->
                        <div class="pc-h-slider-bottom" id="pc-h-slider-bottom">
                            <div class="pc-h-track">
                                <div class="pc-h-thumb" id="pc-h-thumb-bottom"></div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Легенда -->
                <div class="pc-legend">
                    <div class="pc-legend-item">
                        <span class="pc-legend-color pc-color-tl"></span>
                        <span>Рутина, «убийство» времени</span>
                        <span class="pc-legend-value" id="pc-val-tl">0%</span>
                    </div>
                    <div class="pc-legend-item">
                        <span class="pc-legend-color pc-color-tr"></span>
                        <span>Приближает к важным целям</span>
                        <span class="pc-legend-value" id="pc-val-tr">0%</span>
                    </div>
                    <div class="pc-legend-item">
                        <span class="pc-legend-color pc-color-bl"></span>
                        <span>Задачи для избегания проблем</span>
                        <span class="pc-legend-value" id="pc-val-bl">0%</span>
                    </div>
                    <div class="pc-legend-item">
                        <span class="pc-legend-color pc-color-br"></span>
                        <span>«Через угрозы к целям»</span>
                        <span class="pc-legend-value" id="pc-val-br">0%</span>
                    </div>
                </div>
            </div>
        `;

        // Сохраняем ссылки на элементы
        this.elements = {
            vSlider: this.container.querySelector('#pc-v-slider'),
            vHandle: this.container.querySelector('#pc-v-handle'),
            hSliderTop: this.container.querySelector('#pc-h-slider-top'),
            hThumbTop: this.container.querySelector('#pc-h-thumb-top'),
            hSliderBottom: this.container.querySelector('#pc-h-slider-bottom'),
            hThumbBottom: this.container.querySelector('#pc-h-thumb-bottom'),
            matrix: this.container.querySelector('#pc-matrix'),
            zones: {
                tl: this.container.querySelector('#pc-zone-tl'),
                tr: this.container.querySelector('#pc-zone-tr'),
                bl: this.container.querySelector('#pc-zone-bl'),
                br: this.container.querySelector('#pc-zone-br')
            },
            legendValues: {
                tl: this.container.querySelector('#pc-val-tl'),
                tr: this.container.querySelector('#pc-val-tr'),
                bl: this.container.querySelector('#pc-val-bl'),
                br: this.container.querySelector('#pc-val-br')
            }
        };
    }

    /**
     * Возвращает CSS стили виджета
     */
    getStyles() {
        return `
            /* === Переменные по умолчанию === */
            .pc-widget {
                --pc-size: 300px;
                --pc-v-title-width: 50px;
                --pc-v-labels-width: 40px;
                --pc-v-slider-width: 30px;
                --pc-h-slider-height: 25px;
                --pc-handle-size: 20px;
                --pc-color-tl: #e2e3e5;
                --pc-color-tr: #a8e6cf;
                --pc-color-bl: #ffb3ba;
                --pc-color-br: #ffcba4;
            }

            .pc-widget {
                display: flex;
                flex-direction: column;
                align-items: center;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            }

            /* Общая надпись над горизонтальными подписями */
            .pc-h-title-top {
                width: var(--pc-size);
                margin-left: calc(var(--pc-v-title-width) + var(--pc-v-labels-width) + var(--pc-v-slider-width));
                font-size: 12px;
                color: #495057;
                font-weight: 500;
                text-align: center;
                margin-bottom: 5px;
            }

            /* Подписи сверху */
            .pc-labels-top {
                display: flex;
                width: var(--pc-size);
                margin-left: calc(var(--pc-v-title-width) + var(--pc-v-labels-width) + var(--pc-v-slider-width));
                font-size: 12px;
                color: #495057;
                font-weight: 500;
                margin-bottom: 2px;
            }

            .pc-labels-top span {
                width: 50%;
                text-align: center;
            }

            /* Основной layout */
            .pc-layout {
                display: flex;
                align-items: stretch;
            }

            /* Левая вертикальная надпись */
            .pc-v-title {
                position: relative;
                width: var(--pc-v-title-width);
                height: var(--pc-size);
                margin-top: var(--pc-h-slider-height);
                display: flex;
                align-items: center;
                justify-content: center;
            }

            .pc-v-title span {
                writing-mode: vertical-rl;
                transform: rotate(180deg);
                text-align: center;
                font-size: 12px;
                color: #495057;
                font-weight: 500;
            }

            /* Вертикальные подписи */
            .pc-v-labels {
                position: relative;
                width: var(--pc-v-labels-width);
                height: var(--pc-size);
                margin-top: var(--pc-h-slider-height);
            }

            .pc-v-label-top, .pc-v-label-bottom {
                position: absolute;
                left: 0;
                width: var(--pc-v-labels-width);
                height: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                box-sizing: border-box;
            }

            .pc-v-label-top { top: 0; }
            .pc-v-label-bottom { top: 50%; }

            .pc-v-label-top span, .pc-v-label-bottom span {
                writing-mode: vertical-rl;
                transform: rotate(180deg);
                text-align: center;
                font-size: 12px;
                color: #495057;
                font-weight: 500;
                height: 100%;
                display: flex;
                align-items: center;
                justify-content: center;
            }

            /* Вертикальный бегунок */
            .pc-v-slider {
                position: relative;
                width: var(--pc-v-slider-width);
                height: var(--pc-size);
                margin-top: var(--pc-h-slider-height);
                cursor: ns-resize;
            }

            .pc-v-track {
                position: absolute;
                left: 50%;
                top: 0;
                bottom: 0;
                width: 4px;
                background: rgba(0,0,0,0.2);
                border-radius: 2px;
                transform: translateX(-50%);
            }

            .pc-v-handle {
                position: absolute;
                left: 50%;
                top: 50%;
                width: var(--pc-handle-size);
                height: var(--pc-handle-size);
                background: #333;
                border: 2px solid #fff;
                border-radius: 50%;
                transform: translate(-50%, -50%);
                box-shadow: 0 2px 5px rgba(0,0,0,0.4);
                pointer-events: none;
            }

            /* Матрица */
            .pc-matrix-wrapper {
                display: flex;
                flex-direction: column;
            }

            .pc-matrix {
                position: relative;
                width: var(--pc-size);
                height: var(--pc-size);
                background-color: #f8f9fa;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }

            /* Зоны */
            .pc-zone {
                position: absolute;
                transition: background-color 0.3s, width 0.1s, height 0.1s, top 0.1s, left 0.1s;
                border: 1px solid rgba(0,0,0,0.4);
                box-sizing: border-box;
            }

            .pc-zone-tl { background-color: var(--pc-color-tl); }
            .pc-zone-tr { background-color: var(--pc-color-tr); }
            .pc-zone-bl { background-color: var(--pc-color-bl); }
            .pc-zone-br { background-color: var(--pc-color-br); }

            /* Горизонтальные бегунки */
            .pc-h-slider-top, .pc-h-slider-bottom {
                position: relative;
                height: 20px;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: ew-resize;
            }

            .pc-h-slider-top { margin-bottom: 5px; }
            .pc-h-slider-bottom { margin-top: 5px; }

            .pc-h-track {
                position: relative;
                width: var(--pc-size);
                height: 4px;
                background: rgba(0,0,0,0.2);
                border-radius: 2px;
            }

            .pc-h-thumb {
                position: absolute;
                top: 50%;
                width: var(--pc-handle-size);
                height: var(--pc-handle-size);
                background: #333;
                border: 2px solid #fff;
                border-radius: 50%;
                transform: translate(-50%, -50%);
                box-shadow: 0 2px 4px rgba(0,0,0,0.4);
                pointer-events: none;
            }

            /* Легенда */
            .pc-legend {
                margin-top: 20px;
                display: flex;
                flex-direction: column;
                gap: 10px;
                width: var(--pc-size);
            }

            .pc-legend-item {
                display: flex;
                align-items: center;
                gap: 10px;
                font-size: 13px;
                color: #495057;
            }

            .pc-legend-color {
                width: 20px;
                height: 20px;
                border-radius: 4px;
                border: 1px solid rgba(0,0,0,0.1);
                flex-shrink: 0;
            }

            .pc-color-tl { background-color: var(--pc-color-tl); }
            .pc-color-tr { background-color: var(--pc-color-tr); }
            .pc-color-bl { background-color: var(--pc-color-bl); }
            .pc-color-br { background-color: var(--pc-color-br); }

            .pc-legend-value {
                margin-left: auto;
                font-weight: 600;
                font-size: 14px;
            }

            /* Отключение выделения текста при перетаскивании */
            .pc-dragging {
                user-select: none;
            }
        `;
    }

    /**
     * Привязывает события
     */
    bindEvents() {
        this.setupDrag(this.elements.vSlider, (val) => {
            this.state.vSplit = val;
            this.updateUI();
            this.emitChange();
        }, true, this.elements.vSlider);

        this.setupDrag(this.elements.hSliderTop, (val) => {
            this.state.hSplitTop = val;
            this.updateUI();
            this.emitChange();
        }, false, this.elements.matrix);

        this.setupDrag(this.elements.hSliderBottom, (val) => {
            this.state.hSplitBottom = val;
            this.updateUI();
            this.emitChange();
        }, false, this.elements.matrix);
    }

    /**
     * Настраивает перетаскивание для слайдера
     */
    setupDrag(element, callback, isVertical, referenceElement) {
        let isDragging = false;
        const size = this.options.size;

        const startDrag = (e) => {
            isDragging = true;
            document.body.classList.add('pc-dragging');
            handleMove(e);
        };

        const stopDrag = () => {
            isDragging = false;
            document.body.classList.remove('pc-dragging');
        };

        const handleMove = (e) => {
            if (!isDragging) return;

            const rect = referenceElement.getBoundingClientRect();
            let val;

            if (isVertical) {
                const y = e.clientY - rect.top;
                val = (y / size) * 100;
            } else {
                const x = e.clientX - rect.left;
                val = (x / size) * 100;
            }

            val = Math.max(0, Math.min(100, val));
            callback(val);
        };

        element.addEventListener('mousedown', startDrag);
        element.addEventListener('touchstart', startDrag);
        window.addEventListener('mousemove', handleMove);
        window.addEventListener('touchmove', handleMove);
        window.addEventListener('mouseup', stopDrag);
        window.addEventListener('touchend', stopDrag);
    }

    /**
     * Обновляет UI
     */
    updateUI() {
        const { vSplit, hSplitTop, hSplitBottom } = this.state;
        const { vHandle, hThumbTop, hThumbBottom, zones, legendValues } = this.elements;

        // Позиции ручек
        vHandle.style.top = `${vSplit}%`;
        hThumbTop.style.left = `${hSplitTop}%`;
        hThumbBottom.style.left = `${hSplitBottom}%`;

        // Размеры зон
        const topHeight = vSplit;
        const bottomHeight = 100 - vSplit;

        zones.tr.style.cssText = `top: 0; height: ${topHeight}%; left: ${hSplitTop}%; width: ${100 - hSplitTop}%;`;
        zones.tl.style.cssText = `top: 0; height: ${topHeight}%; left: 0; width: ${hSplitTop}%;`;
        zones.br.style.cssText = `top: ${topHeight}%; height: ${bottomHeight}%; left: ${hSplitBottom}%; width: ${100 - hSplitBottom}%;`;
        zones.bl.style.cssText = `top: ${topHeight}%; height: ${bottomHeight}%; left: 0; width: ${hSplitBottom}%;`;

        // Значения в легенде
        const areaTR = (topHeight / 100) * ((100 - hSplitTop) / 100) * 100;
        const areaTL = (topHeight / 100) * (hSplitTop / 100) * 100;
        const areaBR = (bottomHeight / 100) * ((100 - hSplitBottom) / 100) * 100;
        const areaBL = (bottomHeight / 100) * (hSplitBottom / 100) * 100;

        legendValues.tr.textContent = `${Math.round(areaTR)}%`;
        legendValues.tl.textContent = `${Math.round(areaTL)}%`;
        legendValues.br.textContent = `${Math.round(areaBR)}%`;
        legendValues.bl.textContent = `${Math.round(areaBL)}%`;
    }

    /**
     * Вызывает callback при изменении
     */
    emitChange() {
        if (this.options.onChange) {
            this.options.onChange(this.getValues());
        }
    }

    /**
     * Возвращает текущие значения
     * @returns {{ tl: number, tr: number, bl: number, br: number }}
     */
    getValues() {
        const { vSplit, hSplitTop, hSplitBottom } = this.state;
        const topHeight = vSplit;
        const bottomHeight = 100 - vSplit;

        return {
            tl: Math.round((topHeight / 100) * (hSplitTop / 100) * 100),
            tr: Math.round((topHeight / 100) * ((100 - hSplitTop) / 100) * 100),
            bl: Math.round((bottomHeight / 100) * (hSplitBottom / 100) * 100),
            br: Math.round((bottomHeight / 100) * ((100 - hSplitBottom) / 100) * 100)
        };
    }

    /**
     * Устанавливает значения (в процентах для каждой зоны)
     * @param {{ tl?: number, tr?: number, bl?: number, br?: number }} values
     */
    setValues(values) {
        const tl = values.tl ?? 0;
        const tr = values.tr ?? 0;
        const bl = values.bl ?? 0;
        const br = values.br ?? 0;

        const topSum = tl + tr;
        const bottomSum = bl + br;
        const total = topSum + bottomSum;

        if (total === 0) return;

        this.state.vSplit = (topSum / total) * 100;
        this.state.hSplitTop = topSum > 0 ? (tl / topSum) * 100 : 50;
        this.state.hSplitBottom = bottomSum > 0 ? (bl / bottomSum) * 100 : 50;

        this.updateUI();
        this.emitChange();
    }

    /**
     * Возвращает внутреннее состояние бегунков
     * @returns {{ vSplit: number, hSplitTop: number, hSplitBottom: number }}
     */
    getState() {
        return { ...this.state };
    }

    /**
     * Устанавливает внутреннее состояние бегунков
     * @param {{ vSplit?: number, hSplitTop?: number, hSplitBottom?: number }} state
     */
    setState(state) {
        if (state.vSplit !== undefined) this.state.vSplit = state.vSplit;
        if (state.hSplitTop !== undefined) this.state.hSplitTop = state.hSplitTop;
        if (state.hSplitBottom !== undefined) this.state.hSplitBottom = state.hSplitBottom;
        this.updateUI();
        this.emitChange();
    }
}

// Экспорт для использования в других модулях
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ProductivityControl;
}
