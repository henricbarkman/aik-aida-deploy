"""Client for EPD Norge's digital EPD hub.

EPD Norge publishes through the same soda4LCA software as Environdec, at
epdnorway.lca-data.com, so the wire format is ILCD JSON and the detail parser
in ``environdec_client`` is reused unchanged. Measured 2026-09-14 against the
live hub: 12 579 EPDs in the PUBLIC stock, zero uuid overlap with Environdec's
18 849, so every row it contributes is new to the catalog.

Three differences from Environdec, all in the LIST endpoint, and each one is a
way the Environdec assumptions would have failed silently here:

1. No ``owner``. Not one of 12 579 rows carries it; the owner exists only in
   the detail response. ``nordic_supplier`` therefore cannot pre-filter on the
   index. What the index does carry is ``geo``, and unlike Environdec's
   GLO/RER, EPD Norge's geo is the producer's country in practice
   (NO 3292, SE 2071, DK 1378, DE 993).
2. No ``name`` for 3 491 rows, every one of them a dataset whose only language
   is no/da/sv, and 2 402 of them Nordic. Those rows do carry ``classific``
   ("Bygg / Takbelegg, membraner"), but candidate search still matches on
   name only, so these rows are not reached yet. Matching on classification
   is the known next step.
3. 2 066 rows predate EN 15804+A2 and declare one bare "Global warming
   potential (GWP)" with no fossil/biogenic split. The parser records that as
   GWP-total; the builder refuses total-only rows, so they leave with a log
   line instead of entering the fossil-basis median under the wrong label.
   The other 10 513 carry GWP-fossil/-biogenic/-luluc/-total exactly as
   Environdec does.

The hub is open and unauthenticated. Requests are spaced ``request_interval``
seconds apart (1.0 by default), which is the whole of the politeness contract.
"""

from __future__ import annotations

import logging
from pathlib import Path

from aida.data.environdec_client import EnvirondecClient

logger = logging.getLogger(__name__)

EPD_NORGE_HUB_URL = "https://epdnorway.lca-data.com/resource"
# The "PUBLIC" stock: "Data which is published". The root REFERENCE_DATA stock
# and InData_2018 are subsets and exchange stocks, not the catalogue.
EPD_NORGE_PUBLIC_STOCK = "91413340-7bf0-4f88-a952-0f91cba685df"
NORGE_INDEX_PATH = Path(__file__).parent / "epd_norge_index.json"


class EpdNorgeClient(EnvirondecClient):
    """Same protocol, different hub. See module docstring for what differs."""

    registry = "epd_norge"
    datastock = EPD_NORGE_PUBLIC_STOCK
    index_path = NORGE_INDEX_PATH
    index_page_sleep = 1.0

    def __init__(self, base_url: str = EPD_NORGE_HUB_URL, request_interval: float = 1.0):
        super().__init__(base_url=base_url, request_interval=request_interval)
