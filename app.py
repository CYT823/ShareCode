from flask import Flask, render_template, url_for, request, redirect, jsonify
from flask.wrappers import Request
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import inspect
import importlib
import sys

app=Flask(__name__)

@app.route("/api/getfunc", methods=["GET"])
def get_func():
    module_name = "analytic_module" # python檔案名稱
    package_name = "AnalyticModule" # 資料夾名稱
    analytic_modules = {}

    module = importlib.import_module(name=f".{module_name}", package=package_name)
    
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) and hasattr(obj, "module_name"):
            parameters = inspect.signature(obj).parameters
            parameter_info = {args_name: str(parameters[args_name].annotation) for args_name in parameters}
            analytic_modules[obj.module_name] = {"description": obj.description, "parameters": parameter_info, "return_type": obj.return_type}
    metadata =  analytic_modules
    return jsonify(metadata)

@app.route("/api/analyze", methods=["GET"])
def analyze():
    module_name = "analytic_module" # python檔案名稱
    package_name = "AnalyticModule" # 資料夾名稱
    func_name = request.args.get("func_name")

    module = importlib.import_module(name=f".{module_name}", package=package_name)
    
    args_dict = dict(request.args)  # Convert ImmutableMultiDict to a regular dictionary
    if "func_name" in args_dict:
        args_dict.pop("func_name")  # Remove function name from arguments

    func = getattr(module, func_name)
    result = func(*args_dict)
    return result

    

if __name__ == "__main__":
    app.run(debug=True)