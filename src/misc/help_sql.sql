-- порядок:
--
-- FROM
-- JOIN
-- WHERE
-- GROUP BY
-- HAVING
-- ORDER BY
-- LIMIT

SELECT 'Hello world';

-- создание таблицы
CREATE TABLE users (id SERIAL PRIMARY KEY, name TEXT, age INT);

-- добавление данных
INSERT INTO users (name, age) VALUES
('Alice', 25),
('Bob', 30),
('Charlie', 35);

-- создание таблицы
CREATE TABLE user_scores (
    id SERIAL PRIMARY KEY, -- SERIAL - автоинкремент
    user_id INT,
    score FLOAT
);

-- добавление данных
INSERT INTO user_scores (user_id, score) VALUES
(1, 0.85),
(2, 0.92),
(3, 0.78);

-- просмотр данных
SELECT * FROM users;

-- JOIN (соединение двух таблиц по ключам)
SELECT users.name, user_scores.score
FROM users -- 1.берем эту таблицу как основу
JOIN user_scores -- 2.присоединяем эту таблицу
ON users.id = user_scores.user_id; -- правило соединения (сопоставление id, или "соедини данные по ключу")

-- выборка данных
SELECT * FROM users
WHERE age > 25;

-- более сложная выборка данных
SELECT * FROM users
WHERE age > 25 AND name = 'Bob';

-- более сложная выборка данных
SELECT * FROM users
WHERE age < 30 OR name = 'Charlie';

-- удаление таблицы
DROP TABLE Rooms;

-- GROUP BY
CREATE TABLE Rooms (id SERIAL PRIMARY KEY, home_type TEXT, has_tv BOOLEAN, price FLOAT);
INSERT INTO Rooms (home_type, has_tv, price) VALUES
('Private room', TRUE, 149),
('Entire home/apt', FALSE, 225),
('Private room', TRUE, 150),
('Entire home/apt', TRUE, 89),
('Entire home/apt', FALSE, 80),
('Entire home/apt', FALSE, 200),
('Private room', FALSE, 60),
('Private room', TRUE, 79),
('Private room', TRUE, 79),
('Entire home/apt', TRUE, 150),
('Entire home/apt', TRUE, 135),
('Private room', FALSE, 85),
('Private room', FALSE, 89),
('Private room', FALSE, 85),
('Entire home/apt', TRUE, 120),
('Shared room', TRUE, 40);

SELECT * FROM Rooms;
-- Выдать столбец home_type
SELECT home_type
FROM Rooms;
-- (Выдать только уникальные значения из столбца home_type)
SELECT DISTINCT home_type
FROM Rooms;
-- Сгруппировать (оставить уникальные значения) по значениям столбец home_type.
-- Отличие от DISTINCT в том, что с GROUP BY можно делать агрегаты.
SELECT home_type
FROM Rooms
GROUP BY home_type;

-- Применяем AVG по price к каждой группе отдельно и сохраняем результат в столбец avg_price.
-- Смысл: среднее значение по price для каждой группы.
-- Смысл агрегатных функций в том, что они выдают из нескольких значений одно значение, которое можно записать в одно поле.
SELECT home_type, AVG(price) as avg_price
FROM Rooms
GROUP BY home_type;

-- WHERE - это фильтрация строк, а
-- HAVING - это фильтрация групп.
-- Смысл: среднее значение по price для каждой группы + фильтрация групп по средней цене.
SELECT home_type, AVG(price) as avg_price
FROM Rooms
GROUP BY home_type
HAVING AVG(price) > 100;

-- Смысл: отфильтровать группы только с большим количеством элементов внутри (игнорировать редкие категории)
SELECT home_type, COUNT(*)
FROM Rooms
GROUP BY home_type
HAVING COUNT(*) > 2;

-- Соединяем WHERE и HAVING
SELECT home_type, AVG(price)
FROM Rooms
WHERE price > 50 -- фильтрация строк
GROUP BY home_type
HAVING AVG(price) > 80; -- фильтрация групп

-- Сортировка (по умолчанию она по возрастанию)
SELECT * FROM user_scores
ORDER BY score;

-- Сортировка по убыванию
SELECT * FROM user_scores
ORDER BY score DESC;

-- Топ 1: DESC+LIMIT
SELECT * FROM user_scores
ORDER BY score DESC
LIMIT 1;

-- Топ 3: DESC+LIMIT
SELECT * FROM rooms
ORDER BY price DESC
LIMIT 3;

-- Сортировка по группам
SELECT home_type, AVG(price) as avg_price
FROM rooms
GROUP BY home_type
ORDER BY avg_price;

-- Сортировка сначала по одному условию, затем по другому условию
SELECT *
FROM rooms
ORDER BY has_tv DESC, price ASC;

-- Соединяем несколько
SELECT users.name, AVG(user_scores.score) as avg_score
FROM users
JOIN user_scores
ON users.id = user_scores.user_id
GROUP BY users.name
ORDER BY avg_score DESC;

-----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------

-- раньше мы делали так. Строк столько, сколько групп.
SELECT user_id, AVG(score)
FROM user_scores
GROUP BY user_id;

-- То же самое, только количество строк такое же.
SELECT user_id, score, AVG(score) OVER (PARTITION BY user_id) as avg_score
FROM user_scores;

-- ROW NUMBER - это номер строки внутри группы-partitition.
-- Учитывая, что группа-partitition сортирована, это номер призового места в группе-partitition.
SELECT user_id, score, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY score DESC) as rn
FROM user_scores
ORDER BY user_id;

-- То же самое, но выбираем теперь 1 место.
-- Можно указать WHERE rn <= 3, чтобы получить топ-3.
SELECT *
FROM (
    SELECT user_id,
           score,
           ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY score DESC) as rn
    FROM user_scores
) t
WHERE rn = 1;

-- RANK
-- У нас две повторяющиеся строки score. ROW_NUMBER нумерует даже повторяющиеся строки по порядку,
-- а RANK нумерует повторяющиеся одинаково.
SELECT user_id,
       score,
       RANK() OVER (PARTITION BY user_id ORDER BY score DESC) as rnk
FROM user_scores;

-- DENSE_RANK делает это аккуратнее, по порядку
SELECT user_id,
       score,
       DENSE_RANK() OVER (PARTITION BY user_id ORDER BY score DESC) as drnk
FROM user_scores;


-- CREATE TABLE .. AS SELECT - сохранение результата select в таблицу

-- USING ("TransactionID") - это аналог ON t."TransactionID" = i."TransactionID", только более удобный.

-- ALTER TABLE - правка существующей таблицы.

-- UPDATE SET без WHERE правит все значения в определенной колонке определенной таблицы.