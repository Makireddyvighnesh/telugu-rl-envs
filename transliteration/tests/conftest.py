import pytest

from telugu_transliteration.data.catalog import Catalog
from telugu_transliteration.fixtures import make_fixture


@pytest.fixture
def catalog(tmp_path):
    make_fixture(tmp_path)
    return Catalog(tmp_path)
