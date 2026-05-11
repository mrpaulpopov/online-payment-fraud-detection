Запускать просто как  
`docker compose up -d`


1. Парсинг csv, создаю код для создания таблиц с множеством столбцов. 
2. Из созданного кода создаю таблицы, затем делаю COPY из csv.
3. Объединяем train_transaction и train_identity по TransactionID.
4. Т.к. нет единого user_id, его нужно предположительно воссоздать. Это делается 4 разными подходами uid. Теперь каждая транзакция подписана разными uid.
5. Делаю агрегаты по uid1 (uid1 уже включает в себя uid2, uid3, uid4). Чтобы избежать data leakage (не делать подсчет средних значений по будущим значениям для транзакции), то я делаю rolling-window агрегаты, от первого появления до текущей.
6. Делаю несколько BEHAVIORAL ASSUMPTIONS: 
- число транзакций за последние 5 минут (и другие временные промежутки)
- time since last transaction
- amount from last hour
- ratio amount/average transaction per user
- time since last geo change
- novelty of the device (for each mobile and desktop type)
7. I'm loading final_features table into Pandas and doing basic preprocessing (drop columns with null ratio > 90%).


Итоговая схема таблиц:
transaction_features
    ↓
uid4_features (aggregates over history)
    ↓
uid4_time_features (rolling windows)
    ↓
final_train_dataset (JOIN all)


Собирать features и агрегаты в sql, а потом просто сделать один раз
df = pd.read_sql("SELECT * FROM final_features", engine)







SELECT *
FROM transaction_features t
LEFT JOIN uid1_features u USING(uid1)
LEFT JOIN transaction_time_features w USING(TransactionID);