import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")