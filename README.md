### IEEE-CIS Fraud Detection

## Project Overview & Architecture
Online Payment Fraud Detection

## Technical Stack
- Infrastructure: Docker Compose, PostgreSQL, FastAPI
- ML: PyTorch, LightGBM
- MLOps & Tracking: MLflow, Optuna

## Data Pipeline & Feature Engineering
```
src/scripts/create_table_script.py
db/01_schema.sql
db/02_seed.sql
db/03_train_features.sql, db/04_test_features.sql
```

First of all, I copied columns information from .csv, then copied all data from .csv to sql-tables. I combined train_transaction и train_identity tables by TransactionID.
My first behavioral assumption was: card1 = unique user id, uid1.
I tried also to fingerprint users as uid2 = card1_card2, uid3 = card1_card2_addr1, uid4 = card1_card2_addr1_Pemaildomain.
But it leaded to extreme models overfitting.

Then I made the aggregates by uid1 with rolling-windows.
! For prevent data leakage, I made the aggregates with rolling-windows: from the first occurrence to the current.

### Behavioral Assumptions
I made a few behavioral assumptions, the aggregates based on uid1:
- Count of transactions for the last 5m, 1h, 24h, 7d,
- Time since last transaction,
- Amount of transactions for the last hour,
- Ratio amount/average transaction per user,
- Time since last geo change,
- Novelty of the device, for each mobile and desktop type.
Then I did slight preprocessing in Pandas (dropped columns with (null ratio > 90%), sorted by transaction time).

Train test split: time-based split was used to prevent temporal leakage.

### Simplified data flow schema
```
X, y = load_data()
X_train, y_train = train_split()
X_train_nn = pytorch_preprocessing(X_train)
X_train['anomaly_score'] = anomaly_scores
train_data = prepare_data_for_lgbm(X_train)
```

## Modeling: Autoencoder + LightGBM
My baseline model was LightGBM. However, to help him find anomaly patterns in transactions, I made unsupervised method
of autoencoding in PyTorch. It returns a new column `anomaly_score`, and then LightGBM train with it.
#### Autoencoder
I used bottleneck method with customizable `latent_dim` (the narrowest part).

## MLOps & Hyperparameter Tuning
I made hyperparameters optimization in that order:
1. PyTorch HPO. I found the best hyperparameters for PyTorch Autoencoder (including `latent_dim`)
2. LightGBM HPO. I found the best hyperparameters for LightGBM with anomaly_scores taken from already optimized PyTorch Autoencoder.
3. I made a comparison of metrics between PyTorch+LightGBM and LightGBM only (baseline pipeline).

## Threshold Optimization (Math vs. Business)
Threshold converts a probability to a boolean prediction. Using `precision_recall_curve`, I developed the two approaches to find it:
### Business-driven threshold
Business says: 'You detect fraud. We want that no more than 25% should be false alerts, because they are good customers
who will definitely be angry and call us.' - 

However, if business target is unreachable, mathematical threshold will be used as a fallback.

### Mathematical Threshold


## Model Evaluation, SHAP
PR-AUC
Key final metrics:
lgbm_cv_pr_auc
0.7307300546504649
-
lgbm_train_precision
0.9081603435934145
-
lgbm_train_recall
0.8726784977300867
-
lgbm_train_f1
0.8900659464010102
-
lgbm_train_pr_auc
0.940838526180892
-
lgbm_train_recall_at_fpr
0.8199889943596093
-
lgbm_val_accuracy
0.9745204953658234
-
lgbm_val_precision
0.6696930393428447
-
lgbm_val_recall
0.5092044707429323
-
lgbm_val_f1
0.5785247432306256
-
lgbm_val_roc_auc
0.9285942436920803
-
lgbm_val_pr_auc
0.6082407997366494
-
lgbm_val_recall_at_fpr
0.3136094674556213
-
lgbm_test_accuracy
0.9692033280274551
-
lgbm_test_precision
0.5689320388349515
-
lgbm_test_recall
0.4751865066493675
-
lgbm_test_f1
0.5178508306822198
-
lgbm_test_pr_auc
0.5435352678310291
-
lgbm_test_recall_at_fpr
0.2403503081414207
-

## API

## How to Run (Docker & GPU)
```
docker compose up -d --build training
docker logs -f fraud_training


docker compose down
docker-compose up -d --build

docker compose stop training
```

#### GPU launch
`docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up --build`

#### MLflow interface
`localhost:5001/#/experiments/1/runs`

#### FastAPI interface
`localhost:8000`

#### Optuna hyperparameters optimization
```
docker-compose run --rm training python src/scripts/tune_pytorch_script.py
docker-compose run --rm training python src/scripts/tune_lgbm_script.py
```

#### Tune evaluation standalone  script
```
docker-compose run --rm training python -m src.scripts.tune_evaluation_script
```

## Kaggle Results
This project is based on the Kaggle IEEE-CIS Fraud Detection dataset. The model achieved a score of **0.799174** on the public leaderboard.
However, the primary focus of this project was not to obtain a high score, but to build a complete, production-ready MLOps pipeline.
![kaggle.png](docs/kaggle.png)

## Limitations, Known Issues
#### macOS: SIGSEGV when running LightGBM after PyTorch

On macOS, PyTorch and LightGBM both ship their own OpenMP runtime (`libomp`),
which causes a segfault when both are used in the same process.

My first solution was to limit `num_threads=1` for LightGBM on macOS, which disables the conflicting OpenMP initialization.
However, I decided to not allow to launch this project locally, only Docker (Linux) runs allowed with full multi-threading.

#### Hyperparameter tuning assumption
In the ideal case, I should have done LightGBM HPO both before adding anomaly_scores and after it. 



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