from setuptools import setup, find_packages

setup(
    name="robot_stand_up",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
    ],
    author="Sorina Lupu",
    description="A model-based reinforcement learning project for robot control using CEM",
    python_requires=">=3.11",
)
