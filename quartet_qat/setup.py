from setuptools import setup, find_packages

setup(
    name="quartet-qat",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    author="Andrei Panferov",
    author_email="andrei@panferov.org",
    description="A Python library for Quartet QAT",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/IST-DASLab/Quartet",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=[
        "torch>=2.7.0",
        "fast_hadamard_transform>=1.0.4",
        "qutlass>=0.1.0",
    ],
) 