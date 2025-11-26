from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="alzhimer",
    version="1.0.0",
    author="Mohamad AlJasem, MD MPH MSc",
    author_email="mohamad@aljasem.eu.org",
    description="Early Alzheimer's Disease Detection using Machine Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/m-aljasem/alzhimer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.13.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "streamlit>=1.28.0",
        "Pillow>=10.0.0",
        "kaggle>=1.5.0",
    ],
)
