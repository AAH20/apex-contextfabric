from setuptools import setup, find_packages

setup(
    name="apex-contextfabric",
    version="1.0.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "contextfabric=apex_contextfabric.cli:main",
        ],
    },
)
