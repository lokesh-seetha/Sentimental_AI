import sqlite3
import os

db_path = os.path.join("instance", "hotel_reviews.db")
if not os.path.exists(db_path):
    # Fallback to root if not in instance (older flask versions or config)
    db_path = "hotel_reviews.db"

print(f"Inspecting database at: {db_path}")

if not os.path.exists(db_path):
    print("Database file not found!")
else:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("PRAGMA table_info(review)")
        columns = cursor.fetchall()
        print("Columns in 'review' table:")
        for col in columns:
            print(col)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()
