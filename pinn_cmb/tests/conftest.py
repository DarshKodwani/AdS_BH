import pytest

from pinn_cmb import background as bgm


@pytest.fixture(scope="session")
def bg():
    return bgm.build_background()
