CREATE TABLE Employee (id SERIAL PRIMARY KEY, name TEXT, salary INT, departmentId INT);
CREATE TABLE Department (id SERIAL PRIMARY KEY, name TEXT);

INSERT INTO Employee (name, salary, departmentid) VALUES
('Joe', 85000, 1),
('Henry', 80000, 2),
('Sam', 60000, 2),
('Max', 90000, 1),
('Janet', 69000, 1),
('Randy', 85000, 1),
('Will', 70000, 1);


INSERT INTO Department (name) VALUES
('IT'),
('Sales');