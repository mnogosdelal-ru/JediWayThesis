<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title><?= htmlspecialchars($page_title) ?> - Большое исследование джедайских практик</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        h1 {
            font-size: 24px;
            margin-bottom: 20px;
            color: #2c3e50;
        }
        
        .progress-container {
            margin-bottom: 30px;
        }
        
        .progress-bar {
            height: 10px;
            background: #e0e0e0;
            border-radius: 5px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: #3498db;
            transition: width 0.3s ease;
        }
        
        .progress-text {
            text-align: right;
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #2c3e50;
        }
        
        .help-text {
            font-size: 14px;
            color: #666;
            margin-bottom: 10px;
        }
        
        input[type="number"],
        input[type="text"],
        select,
        textarea {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 16px;
        }
        
        textarea {
            min-height: 100px;
            resize: vertical;
        }
        
        .radio-group,
        .checkbox-group {
            margin-top: 10px;
        }
        
        .radio-option,
        .checkbox-option {
            margin-bottom: 10px;
        }
        
        .radio-option label,
        .checkbox-option label {
            display: flex;
            align-items: center;
            font-weight: normal;
            cursor: pointer;
        }
        
        input[type="radio"],
        input[type="checkbox"] {
            margin-right: 10px;
            width: 18px;
            height: 18px;
        }
        
        .scale-group {
            margin-top: 15px;
        }
        
        .scale-options {
            display: flex;
            justify-content: space-between;
            margin-top: 10px;
            flex-wrap: wrap;
        }
        
        .scale-option {
            text-align: center;
            flex: 1;
            min-width: 60px;
        }
        
        .scale-option input {
            display: block;
            margin: 5px auto;
        }
        
        .scale-labels {
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }
        
        .buttons {
            display: flex;
            justify-content: space-between;
            margin-top: 30px;
            gap: 10px;
        }
        
        button,
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 4px;
            font-size: 16px;
            cursor: pointer;
            text-decoration: none;
            display: inline-block;
            transition: background 0.2s;
        }
        
        .btn-primary {
            background: #3498db;
            color: white;
        }
        
        .btn-primary:hover {
            background: #2980b9;
        }
        
        .btn-secondary {
            background: #95a5a6;
            color: white;
        }
        
        .btn-secondary:hover {
            background: #7f8c8d;
        }
        
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .error {
            color: #e74c3c;
            margin-top: 5px;
            font-size: 14px;
        }
        
        .required {
            color: #e74c3c;
        }
        
        /* Индикатор сохранения */
        .save-indicator {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 15px;
            background: #27ae60;
            color: white;
            border-radius: 4px;
            font-size: 14px;
            opacity: 0;
            transition: opacity 0.3s;
            z-index: 1000;
        }
        
        .save-indicator.visible {
            opacity: 1;
        }
        
        /* Индикатор восстановленных данных */
        .restore-notice {
            background: #fff3cd;
            border: 1px solid #ffc107;
            color: #856404;
            padding: 10px 15px;
            border-radius: 4px;
            margin-bottom: 20px;
            display: none;
        }
        
        .restore-notice.visible {
            display: block;
        }
    </style>
</head>
<body>
    <div class="save-indicator" id="save-indicator">✓ Сохранено</div>
    
    <div class="container">
        <?php if ($page < 10): ?>
        <div class="progress-container">
            <div class="progress-bar">
                <div class="progress-fill" style="width: <?= $progress ?>%"></div>
            </div>
            <div class="progress-text"><?= $progress ?>% завершено</div>
        </div>
        <?php endif; ?>
        
        <div class="restore-notice" id="restore-notice">
            ℹ️ Ваши предыдущие ответы восстановлены. Вы можете продолжить с того места, где остановились.
        </div>
        
        <h1><?= htmlspecialchars($page_title) ?></h1>
