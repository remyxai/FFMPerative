import inspect
from . import tools as t

def generate_tools_mapping():
    tools_mapping = {}
    for name, obj in inspect.getmembers(t):
        if inspect.isclass(obj) and issubclass(obj, t.Tool) and obj is not t.Tool:
            tools_mapping[name] = obj()  # Instantiate the class
    return tools_mapping
