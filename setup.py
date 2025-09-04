from setuptools import setup, find_packages

setup(
    name="credit-risk-analysis",
    version="1.0.0",
    author="Sushmita Singh",
    description="Advanced Credit Risk Analysis - MSc Research",
    url="https://github.com/Sushmitha2701/Advanced-Credit-Risk-Analysis-with-Machine-Learning",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0", 
        "scikit-learn>=1.0.0",
        "tensorflow>=2.6.0",
        "matplotlib>=3.4.0",
    ],
)
