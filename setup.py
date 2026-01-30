from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cloudcost-optimizer",
    version="1.0.0",
    author="Brice Roméo Zemba Wendémi",
    author_email="bricezemba336@gmail.com",
    description="AI-powered cloud cost prediction and resource optimization for SaaS applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BriceZemba/cloudcost-optimizer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.15.0",
        "numpy>=1.24.0",
        "pandas>=2.1.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.8.0",
        "streamlit>=1.29.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.12.0",
            "flake8>=7.0.0",
        ],
    },
)
