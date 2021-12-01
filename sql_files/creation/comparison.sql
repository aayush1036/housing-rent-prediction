USE PREPROCESS;
DROP TABLE IF EXISTS preprocess_count;
CREATE TABLE preprocess_count(
id INTEGER PRIMARY KEY AUTO_INCREMENT,
city VARCHAR(20),
nrows INTEGER
);

INSERT INTO preprocess_count (city, nrows) VALUES 
('Ahemdabad', (SELECT COUNT(*) FROM AHEMDABAD)),
('Bangalore', (SELECT COUNT(*) FROM BANGALORE)),
('Chennai', (SELECT COUNT(*) FROM CHENNAI)),
('Delhi', (SELECT COUNT(*) FROM DELHI)),
('Hyderabad', (SELECT COUNT(*) FROM HYDERABAD)),
('Kolkata', (SELECT COUNT(*) FROM KOLKATA)),
('Mumbai', (SELECT COUNT(*) FROM MUMBAI)),
('Pune', (SELECT COUNT(*) FROM PUNE));

USE CLEAN;
DROP TABLE IF EXISTS clean_count;
CREATE TABLE clean_count(
id INTEGER PRIMARY KEY AUTO_INCREMENT,
city VARCHAR(20),
nrows INTEGER
);

INSERT INTO clean_count (city, nrows) VALUES 
('Ahemdabad', (SELECT COUNT(*) FROM AHEMDABAD)),
('Bangalore', (SELECT COUNT(*) FROM BANGALORE)),
('Chennai', (SELECT COUNT(*) FROM CHENNAI)),
('Delhi', (SELECT COUNT(*) FROM DELHI)),
('Hyderabad', (SELECT COUNT(*) FROM HYDERABAD)),
('Kolkata', (SELECT COUNT(*) FROM KOLKATA)),
('Mumbai', (SELECT COUNT(*) FROM MUMBAI)),
('Pune', (SELECT COUNT(*) FROM PUNE));

SELECT * FROM CLEAN.clean_count LEFT JOIN PREPROCESS.preprocess_count ON CLEAN.clean_count.id = PREPROCESS.preprocess_count.id;