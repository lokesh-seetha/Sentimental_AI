try:
    import flask_login
    print("flask_login is installed and importable.")
    print(f"Version: {flask_login.__version__}")
    print(f"Location: {flask_login.__file__}")
except ImportError as e:
    print(f"Error: {e}")
