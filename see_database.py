import sqlite3

# Connect to the database
conn = sqlite3.connect("visitor_log.db")
cursor = conn.cursor()

# Print all rows in the visitors table
cursor.execute("SELECT * FROM visitors")
rows = cursor.fetchall()

print("ID | Timestamp           | Image Path")
print("---------------------------------------")
for row in rows:
    print(row)

conn.close()
