"""Back-compat shim — канонічний модуль переїхав у `pipelines/shared/prompts.py`.

Лишено щоб старі імпорти (`from datasetforge.pipelines.inpaint.prompts import ...`)
і pipeline #1 не зламались. Новий код імпортуй із shared.
"""

from datasetforge.pipelines.shared.prompts import (  # noqa: F401
    azimuth_to_cardinal,
    build_prompt,
)
