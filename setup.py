from setuptools import find_packages, setup

setup(
    name="openeyes-live",
    version="0.1.0",
    packages=find_packages(include=["src", "src.*"]),
    entry_points={
        "console_scripts": [
            "openeyes=src.cli.main:main",
        ],
    },
    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "llama-cpp-python>=0.2.0",
        "PyYAML>=6.0",
        "onnxruntime>=1.17.0",
        "sherpa-onnx>=1.10.0",
        "sounddevice>=0.4.6",
    ],
    python_requires=">=3.10",
)
