from app import app, db
import sys

print("Imported app and db")
try:
    with app.app_context():
        print("Creating all tables...")
        db.create_all()
        print("db init success")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
