from setuptools import setup, find_packages

setup(
    name="openeyes-live",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "openeyes=cli.main:main",
        ],
    },
    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "llama-cpp-python>=0.2.0",
    ],
    python_requires=">=3.10",
)
