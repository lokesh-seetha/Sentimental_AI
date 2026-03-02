from jinja2 import Environment, FileSystemLoader

env = Environment(loader=FileSystemLoader('templates'))
try:
    env.parse(env.loader.get_source(env, 'output.html')[0])
    print("Template syntax is valid.")
except Exception as e:
    print(f"Template syntax error: {e}")
