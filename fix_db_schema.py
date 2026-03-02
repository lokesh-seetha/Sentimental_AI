import sqlite3
import os

db_path = os.path.join("instance", "hotel_reviews.db")
if not os.path.exists(db_path):
    # Fallback to root if not in instance
    db_path = "hotel_reviews.db"

print(f"Fixing database at: {db_path}")

if not os.path.exists(db_path):
    print("Database file not found!")
else:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        # Add the missing column
        print("Adding 'user_id' column to 'review' table...")
        cursor.execute("ALTER TABLE review ADD COLUMN user_id INTEGER REFERENCES user(id)")
        conn.commit()
        print("Column added successfully.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()
