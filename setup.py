from setuptools import setup, find_packages

setup(
    name="CFLA",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "tqdm",
    ],
    author="HOUNSI Antoine",
    author_email="antoinehounsi3@gmail.com",
    description="A framework for federated learning algorithms",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Totorino02/CFLA/tree/main",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)