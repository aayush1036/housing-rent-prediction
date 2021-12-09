import mysql.connector
import json
with open('config.json','r') as f:
    config = json.load(f)
conn = mysql.connector.connect(
    host=config.get('host'),
    port=config.get('port'),
    user=config.get('user'),
    password=config.get('password'),
    database=config.get('database')
    )
cursor = conn.cursor()
cursor.execute("INSERT INTO TEST VALUES (2,'123')")
conn.commit()
conn.close()