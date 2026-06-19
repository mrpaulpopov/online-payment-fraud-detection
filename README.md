### IEEE-CIS Fraud Detection
Запускать просто как  
`docker compose up -d`

Что в проекте:
SQL
LightGBM
Метрики
SHAP values
MLflow

1. Парсинг csv, создаю код для создания таблиц с множеством столбцов. 
2. Из созданного кода создаю таблицы, затем делаю COPY из csv.
3. Объединяем train_transaction и train_identity по TransactionID.
4. FEATURE ENGINEERING: Т.к. нет единого user_id, его нужно предположительно воссоздать. Это делается 4 разными подходами uid. Теперь каждая транзакция подписана разными uid:
- uid1 = card1
- uid2 = card1 + card2
- uid3 = card1 + card2 + addr1 + P_emaildomain
- uid4 = card1 + card2 + addr1 + DeviceInfo
- Однако это было моей ошибкой, так как модель начала обучаться строго на uid. Я оставил только uid1.
5. Делаю агрегаты по uid1. Чтобы избежать data leakage (не делать подсчет средних значений по будущим значениям для транзакции), то я делаю rolling-window агрегаты, от первого появления до текущей.
6. Делаю несколько BEHAVIORAL ASSUMPTIONS: 
- число транзакций за последние 5 минут (и другие временные промежутки)
- time since last transaction
- amount from last hour
- ratio amount/average transaction per user
- time since last geo change
- novelty of the device (for each mobile and desktop type)
7. I'm loading final_features table into Pandas and doing slight preprocessing (drop columns with null ratio > 90%, sort by transaction time).
8. Train test split: time-based split was used to prevent temporal leakage
9. LightGBM: модель показывает огромный feature importance для uid3 и uid4. Я попробовал их убрать и посмотреть результат.


Итоговая схема таблиц:
transaction_features
    ↓
uid4_features (aggregates over history)
    ↓
uid4_time_features (rolling windows)
    ↓
final_train_dataset (JOIN all)



Accuracy - процент, когда модель вообще права.
"Но accuracy может быть обманчивой при дисбалансе классов.
Например:
97% нормальных транзакций,
3% fraud.
Тогда модель, которая всегда говорит "не fraud", уже получит ~97% accuracy.
Поэтому здесь довольно бесполезна.

Precision:
Это важно, если false positive дорогие:
блокировка карт,
ручная проверка,
раздражение клиентов.
Высокий precision - мало false alarms, мало раздраженных клиентов. При =1 нет ни одного ложного

Recall:
Это наоборот
если 0.31, то модель пропускает ~69% мошенничества.
Для fraud detection recall обычно критически важен.
При =1 ни один fraud не пропущен.

ROC-AUC (насколько хорошо разделяет классы)
У ROC-кривой оси: true positive rate / false positive rate.
0.5	случайное угадывание
0.9+	отличное


f1 - это Precision и Recall в одном значении. Однако это срез в одной точке при заданном threshold.
PR-AUC при этом показывает площадь под графиком, не зависит от threshold.


Recall@FPR
"Сколько fraud мы ловим, если разрешаем только 1% falae alarms?"
Очень полезная метрика для бизнеса, где бизнес сам задает этот процент.

Как читать SHAP values (beeswarm plot):
Значения вправо увеличивают результат (вероятность fraud), влево уменьшают.
серый цвет - категориальные features, тут просто редкие значения
D2 имеет красный влево, а синий вправо. Значит, увеличение D2 уменьшает вероятность fraud.


P_emaildomain - да, валидный признак: модель определяет анонимные домены почты.


Не запускать fastapi, пока база не поднята. Прописать это в docker-compose:

  fastapi_app:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      postgres_db:
        condition: service_healthy # Ждем, пока healthcheck базы не скажет "ОК"



AUTOENCODER

Pytorch: У меня единый пайплайн обучения и инференса, поэтому после обучения в pytorch модель из оперативной памяти
сразу же используется. Она дополнительно сохраняется в файл, впрочем


To prevent OpenMP threading conflicts on macOS and ensure stable CPU usage during the pipeline execution,
the maximum number of threads for LightGBM is explicity limited in config.yaml.

## Known Issues
**macOS: SIGSEGV when running LightGBM after PyTorch**

On macOS, PyTorch and LightGBM both ship their own OpenMP runtime (`libomp`),
which causes a segfault when both are used in the same process.

The pipeline automatically sets `num_threads=1` for LightGBM on macOS,
which disables the conflicting OpenMP initialization. Training will be
slower locally, but Docker (Linux) runs with full multi-threading.

## Known Issues

**macOS: SIGSEGV when running LightGBM after PyTorch**

On macOS, PyTorch and LightGBM both ship their own OpenMP runtime (`libomp`),
which causes a segfault when both are used in the same process.

The pipeline automatically sets `num_threads=1` for LightGBM on macOS,
which disables the conflicting OpenMP initialization. Training will be
slower locally, but Docker (Linux) runs with full multi-threading.

SHAP summary plot is also skipped on macOS for the same reason.
All visualizations are available when running via Docker.



MLFLOW: 5001 port

Чему я научился новому?
1. Высокое заполнение ram, поэтому периодическое ручное удаление тяжелых элементов и gc.collect. High cardinality. Конвертация float64-float32. Отслеживание потребляемого RAM
2. Проблема с портом 5000 на macOS, поэтому mapping выходного порта 5001:5000
3. Docker: порядок выполнения сервисов, conditions, причем выполнение своего тестового запроса.
4. Проблема OpenMP на macOS
5. Уровни logging
6. Чтение большого read_sql через chunksize
7. Ручное написание аналога train_test_split, работающего по временному ряду (через iloc)
8. Автоматическая сборка requirements.txt через pipreqs
9. Проверка fraud drift после разделения данных
10. Autoencoder (nn на основе самой себя)
11. Анализ SHAP
12. Динамический Dockerfile (с override)


docker-compose up --build
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up --build




Схема данных:
X, y = load_data()
X_train, y_train = train_split()

X_train_nn = pytorch_preprocessing(X_train)

X_train['anomaly_score'] = anomaly_scores
train_data = prepare_data_for_lgbm(X_train)


localhost:5001/#/experiments/1/runs