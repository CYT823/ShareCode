def analytic_module(description, return_type):
    def decorator(func):
        func.module_name = func.__name__
        func.description = description
        func.return_type = return_type
        return func
    return decorator

@analytic_module(description="Analyzes xxx", return_type="list")
def funcA(data: list, threshold: int):
    return ["AAA"]

@analytic_module(description="Analyzes yyy", return_type="string")
def funcB(data: list):
    return "BBB"

@analytic_module(description="Analyzes zzz", return_type="image")
def funcC(data: str):
    return "CCC"
