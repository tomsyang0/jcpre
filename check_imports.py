import sys

def check_import(package_name):
    try:
        __import__(package_name)
        print(f"✓ {package_name} 已安装")
        return True
    except ImportError:
        print(f"✗ {package_name} 未安装")
        return False

print("检查项目依赖...")
packages = [
    "pandas",
    "numpy", 
    "sklearn",
    "xgboost",
    "lightgbm",
    "loguru",
    "joblib",
    "scipy"
]

results = []
for package in packages:
    results.append(check_import(package))

print(f"\n安装状态: {sum(results)}/{len(packages)}")
if sum(results) < len(packages):
    print("需要安装缺失的依赖: pip install -r requirements.txt")
