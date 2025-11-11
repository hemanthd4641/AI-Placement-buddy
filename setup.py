from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ai-placement-mentor-bot",
    version="1.0.0",
    author="AI Placement Team",
    author_email="placement-bot@example.com",
    description="An AI-powered mentor bot to help students prepare for placements",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/ai-placement-mentor-bot",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Topic :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "placement-bot=app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "requirements.txt"],
    },
)