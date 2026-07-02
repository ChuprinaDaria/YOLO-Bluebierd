"""Back-compat shim — канонічний модуль переїхав у `pipelines/shared/composite.py`.

Камерна деградація винесена у Stage 5 `pipelines/shared/polish.py` (albumentations).
Лишено щоб старі імпорти і pipeline #1 не зламались. Новий код імпортуй із shared.
"""

from datasetforge.pipelines.shared.composite import (  # noqa: F401
    _luma,
    _sun_vec_from_meta,
    composite_one,
)
