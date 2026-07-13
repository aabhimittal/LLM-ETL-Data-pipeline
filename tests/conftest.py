import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "full_name": ["Alice Johnson", "Bob Smith", "Carlos Mendez"],
            "email": [" Alice.Johnson@Example.COM ", "BOB@example.com", "carlos@example.mx"],
            "phone": ["(415) 555-0100", "415.555.0111", "+52 55 1234 5678"],
            "country": ["us", "us", "mx"],
            "amount": [120.50, 89.00, 240.00],
        }
    )
