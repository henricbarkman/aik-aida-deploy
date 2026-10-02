"""Weight per metre of standard steel profiles, from standard tables (HENRIC-3363).

A steel beam or a steel stud is bought by the metre, while the steel EPDs in
the catalog are declared per kg (45 konstruktionsstål, 18 stålregel). Until
2026-10-01 a "Stålbalk HEA 200" in löpmeter met only the four stud EPDs that
are themselves declared per metre, and was told to give the quantity in kg:
the weight of a metre depends on the profile, and HENRIC-3290 refused to guess
it. For a standard profile nothing has to be guessed. EN 10365 fixes the
dimensions of an HEA 200 and its producers publish its mass, 42,3 kg/m; a
hollow section's mass follows from its outer size and wall thickness; a
thin-sheet stud's mass is in the manufacturer's price list. So the bridge is
the same one the timber side has (unit_conversion.member_volume_per_unit),
with a table where timber has a cross-section: the designation is read from
the component's name and nowhere else, and a name without one gets no figure.

What counts as a designation
----------------------------
- Hot-rolled I, H and U: HEA, HEB, IPE and UPE with the nominal height
  ("HEA 200", "IPE200", "HE 200 A"). Only the heights the table lists.
- Hollow sections: VKR (hot-finished, EN 10210) or KKR (cold-formed,
  EN 10219), outer size AND wall thickness ("VKR 100x100x5", "KKR 120x80x4").
  Without the wall thickness, or without saying VKR or KKR, no figure: the
  mass of a 100x100 tube is 11,9 to 27,4 kg/m depending on the wall, and a
  hot-finished and a cold-formed tube of the same size differ by their corner
  radii.
- Hot-rolled channels and the heavy H series (HENRIC-3390): UPN ("UPN 200",
  "UNP 200") and HE M ("HEM 200", "HE 200 M").
- Angles (HENRIC-3390): equal and unequal, all three sides ("L 50x50x5",
  "Vinkelstål 50x50x5", "VST 120x80x8"). Without the thickness, no figure.
- Circular hollow sections (HENRIC-3390): outer diameter and wall after VKR,
  KKR, KCKR or CHS ("VKR 114,3x5"). A round tube has no corner radius, so its
  nominal mass is the same hot-finished (EN 10210-2) and cold-formed
  (EN 10219-2), pi x (D - t) x t x 7,85 kg/dm3; the table is Tibnor's
  cold-formed one, the only one with 114,3, and serves both.
- Thin-sheet studs: a steel stud ("Stålregel", "C-regel", "C 70") of width
  45, 70, 95, 120, 145 or 160 mm. A förstärkningsregel (1,0 mm sheet) has its
  own row. A standard stud is 0,5 mm sheet; a name stating another thickness
  between 0,4 and 1,0 mm ("Stålregel 70 0,7 mm") gets the 0,5 mm stud's
  weight scaled by the thickness, said in the row: the profile's developed
  width is the same, and a thin-walled section's mass is that width times
  the sheet. No maker publishes a 0,7 mm stud's weight (Lindab's RdB7 had
  none and is no longer sold), so this is the figure there is.

Every profile also has a form, "öppen" (I, H, U, angle) or "rör" (hollow),
which the baseline uses to pick the EPDs of its own form (HENRIC-3390).

Sources, every row
------------------
- ArcelorMittal Europe, "Sections and Merchant bars, Sales programme 2024-1"
  (2024-03), dimensions to EN 10365:2017, column G (kg/m). IPE on pdf pages
  48-53, HE A and HE B on 54-61, UPE on 96-97. Read from the PDF text
  2026-10-01; the HE B and IPE rows agree with every other producer table, as
  they must, the dimensions being the standard's.
- Tibnor, "Tibnors konstruktionstabeller" (2023), column g "massa per m":
  VKR-rör kvadratiska and rektangulära S355J2H enligt SS-EN 10210 (pages
  20-25), KKR-rör kvadratiska and rektangulära enligt SS-EN 10219 (pages
  26-29). Read from the PDF 2026-10-01. Every row is checked in
  test_stalprofiler against the nominal-mass formula of EN 10210-2 and
  EN 10219-2 (cross-section with the standards' calculation corner radii,
  7,85 kg/dm3), which is how Tibnor's table is computed; a row that misreads
  the PDF fails that check.
- Norgips, "Produktkatalog april 2023", column VIKT KG/M: C 45 regel 0,49
  (page 20), C 70 0,59 (page 22), C 95 0,68 (page 24), C 120 0,78 (page 26);
  C 145 dB+ 0,88 and C 160 dB+ 0,94 (page 28, Norgips makes 145 and 160 only
  as dB+); förstärkningsregel CF 45 1,13, CF 70 1,62, CF 95 1,85, CF 120
  1,71, CF 145 1,93, CF 160 2,10, godstjocklek 1,0 mm. The same pages in the
  February and May 2024 editions. The traditional stud has flange 35/37 mm,
  the one every Swedish maker sells under its own name (Lindab RE, Gyproc
  ER), at slightly different weights: Lindab RE 70 is 0,536 kg/m by its own
  article weights (Construline EPD list, 2026), Gyproc ER 0,46 mm sheet.
  Sheet 0,5 mm: Norgips "Produktdatablad C profiler dB+" 01/2020, column
  TJOCKLEK, 0,5 for every width 45 to 160 at the catalogue's weights. (Until
  2026-10-02 this said Lindab's RE 70 was listed at 0,59 kg/m; no Lindab
  source says so.)
- ArcelorMittal 2024-1 as above: UPN on pdf page 100, HE M on 54-61, equal
  angles (EN 10056-1:2017) on 102-109, unequal on 110. Read 2026-10-02.
- Tibnor 2023 as above: KCKR-rör, runda kallformade, SS-EN 10219, pages
  30-31; vinkelstång liksidig, pages 40-43, for the small and odd sizes
  ArcelorMittal does not roll. Read 2026-10-02.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# The two stomme families a profile belongs to (build_epd_alternatives
# EPD_SUBCATEGORY_KEYWORDS["stomme"], palats_client.SUBCATEGORY_KEYWORDS).
STRUCTURAL = "konstruktionsstål"
STUD = "stålregel"

SOURCES: dict[str, str] = {
    "am-ipe": "ArcelorMittal, Sections and Merchant bars 2024-1, EN 10365, s. 48-53",
    "am-he": "ArcelorMittal, Sections and Merchant bars 2024-1, EN 10365, s. 54-61",
    "am-upe": "ArcelorMittal, Sections and Merchant bars 2024-1, EN 10365, s. 96-97",
    "tibnor-vkr-kvadrat": "Tibnors konstruktionstabeller 2023, VKR-rör EN 10210, s. 20-21",
    "tibnor-vkr-rektangel": "Tibnors konstruktionstabeller 2023, VKR-rör EN 10210, s. 22-23",
    "tibnor-vkr-rektangel-2": "Tibnors konstruktionstabeller 2023, VKR-rör EN 10210, s. 24-25",
    "tibnor-kkr-kvadrat": "Tibnors konstruktionstabeller 2023, KKR-rör EN 10219, s. 26-27",
    "tibnor-kkr-rektangel": "Tibnors konstruktionstabeller 2023, KKR-rör EN 10219, s. 28-29",
    "norgips-c45": "Norgips produktkatalog april 2023, standardregel C 45, s. 20",
    "norgips-c70": "Norgips produktkatalog april 2023, standardregel C 70, s. 22",
    "norgips-c95": "Norgips produktkatalog april 2023, standardregel C 95, s. 24",
    "norgips-cf45": "Norgips produktkatalog april 2023, förstärkningsregel CF 45, s. 20",
    "norgips-cf70": "Norgips produktkatalog april 2023, förstärkningsregel CF 70, s. 22",
    "norgips-cf95": "Norgips produktkatalog april 2023, förstärkningsregel CF 95, s. 24",
    "am-upn": "ArcelorMittal, Sections and Merchant bars 2024-1, EN 10365, s. 100-101",
    "am-vinkel-liksidig": "ArcelorMittal, Sections and Merchant bars 2024-1, EN 10056-1, s. 102-109",
    "am-vinkel-oliksidig": "ArcelorMittal, Sections and Merchant bars 2024-1, EN 10056-1, s. 110-111",
    "tibnor-vinkel": "Tibnors konstruktionstabeller 2023, vinkelstång liksidig, s. 40-43",
    "tibnor-kckr": ("Tibnors konstruktionstabeller 2023, KCKR-rör EN 10219, s. 30-31 "
                    "(runt rör, samma nominella vikt varm- och kallformat)"),
    "norgips-c120": "Norgips produktkatalog april 2023, standardregel C 120, s. 26",
    "norgips-c145": "Norgips produktkatalog april 2023, C 145 dB+, s. 28",
    "norgips-c160": "Norgips produktkatalog april 2023, C 160 dB+, s. 28",
    "norgips-cf120": "Norgips produktkatalog april 2023, förstärkningsregel CF 120, s. 26",
    "norgips-cf145": "Norgips produktkatalog april 2023, förstärkningsregel CF 145, s. 28",
    "norgips-cf160": "Norgips produktkatalog april 2023, förstärkningsregel CF 160, s. 28",
}

# Hot-rolled sections, kg/m by nominal height, one source per series.
HOT_ROLLED: dict[str, tuple[str, dict[int, float]]] = {
    "HEA": ("am-he", {
        100: 16.7, 120: 19.9, 140: 24.7, 160: 30.4, 180: 35.5, 200: 42.3,
        220: 50.5, 240: 60.3, 260: 68.2, 280: 76.4, 300: 88.3, 320: 97.6,
        340: 105, 360: 112, 400: 125, 450: 140, 500: 155, 550: 166, 600: 178,
        650: 190, 700: 204, 800: 224, 900: 252, 1000: 272,
    }),
    "HEB": ("am-he", {
        100: 20.4, 120: 26.7, 140: 33.7, 160: 42.6, 180: 51.2, 200: 61.3,
        220: 71.5, 240: 83.2, 260: 93.0, 280: 103, 300: 117, 320: 127,
        340: 134, 360: 142, 400: 155, 450: 171, 500: 187, 550: 199, 600: 212,
        650: 225, 700: 241, 800: 262, 900: 291, 1000: 314,
    }),
    "IPE": ("am-ipe", {
        80: 6.0, 100: 8.1, 120: 10.4, 140: 12.9, 160: 15.8, 180: 18.8,
        200: 22.4, 220: 26.2, 240: 30.7, 270: 36.1, 300: 42.2, 330: 49.1,
        360: 57.1, 400: 66.3, 450: 77.6, 500: 90.7, 550: 106, 600: 122,
    }),
    "UPE": ("am-upe", {
        80: 7.9, 100: 9.8, 120: 12.1, 140: 14.5, 160: 17.0, 180: 19.7,
        200: 22.8, 220: 26.6, 240: 30.2, 270: 35.2, 300: 44.4, 330: 53.2,
        360: 61.2, 400: 72.2,
    }),
    "HEM": ("am-he", {
        100: 41.8, 120: 52.1, 140: 63.2, 160: 76.2, 180: 88.9, 200: 103,
        220: 117, 240: 157, 260: 172, 280: 189, 300: 238, 320: 245, 340: 248,
        360: 250, 400: 256, 450: 263, 500: 270, 550: 278, 600: 285, 650: 293,
        700: 301, 800: 317, 900: 333, 1000: 349,
    }),
    "UPN": ("am-upn", {
        50: 5.6, 65: 7.1, 80: 8.6, 100: 10.6, 120: 13.4, 140: 16.0, 160: 18.8,
        180: 22.0, 200: 25.3, 220: 29.4, 240: 33.2, 260: 37.9, 280: 41.8,
        300: 46.2, 320: 59.5, 350: 60.6, 380: 63.1, 400: 71.8,
    }),
}

# Hollow sections, kg/m by (b, h, t) in mm, grouped by the table page they
# were read from. Rectangular sections are listed with the long side first;
# lookup tries both orders.
HOLLOW: dict[str, dict[tuple[int, int, float], float]] = {
    "tibnor-kkr-kvadrat": {
        (25, 25, 3): 1.89,
        (30, 30, 3): 2.36,
        (40, 40, 2.5): 2.82,
        (40, 40, 3): 3.3,
        (40, 40, 4): 4.2,
        (50, 50, 3): 4.25,
        (50, 50, 4): 5.45,
        (50, 50, 5): 6.56,
        (60, 60, 3): 5.19,
        (60, 60, 4): 6.71,
        (60, 60, 5): 8.13,
        (70, 70, 3): 6.13,
        (70, 70, 4): 7.97,
        (70, 70, 5): 9.7,
        (80, 80, 3): 7.07,
        (80, 80, 4): 9.22,
        (80, 80, 5): 11.3,
        (80, 80, 6): 13.2,
        (90, 90, 3): 8.01,
        (90, 90, 4): 10.5,
        (90, 90, 5): 12.8,
        (90, 90, 6): 15.1,
        (100, 100, 3): 8.96,
        (100, 100, 4): 11.7,
        (100, 100, 5): 14.4,
        (100, 100, 6): 17,
        (100, 100, 8): 21.4,
        (120, 120, 4): 14.2,
        (120, 120, 5): 17.5,
        (120, 120, 6): 20.7,
        (120, 120, 8): 26.4,
        (120, 120, 10): 31.8,
        (140, 140, 5): 20.7,
        (140, 140, 6): 24.5,
        (140, 140, 8): 31.4,
        (140, 140, 10): 38.1,
        (150, 150, 5): 22.3,
        (150, 150, 6): 26.4,
        (150, 150, 8): 33.9,
        (150, 150, 10): 41.3,
        (160, 160, 6): 28.3,
        (160, 160, 8): 36.5,
        (160, 160, 10): 44.4,
        (180, 180, 6): 32.1,
        (180, 180, 8): 41.5,
        (180, 180, 10): 50.7,
        (200, 200, 5): 30.1,
        (200, 200, 6): 35.8,
        (200, 200, 8): 46.5,
        (200, 200, 10): 57,
        (200, 200, 12.5): 68.3,
        (220, 220, 10): 63.2,
        (250, 250, 6): 45.2,
        (250, 250, 8): 59.1,
        (250, 250, 10): 72.7,
        (250, 250, 12.5): 88,
        (300, 300, 12.5): 108,
    },
    "tibnor-kkr-rektangel": {
        (50, 30, 2.5): 2.82,
        (50, 30, 3): 3.3,
        (50, 30, 4): 4.2,
        (60, 40, 3): 4.25,
        (60, 40, 4): 5.45,
        (60, 40, 5): 6.56,
        (80, 40, 3): 5.19,
        (80, 40, 4): 6.71,
        (80, 40, 5): 8.13,
        (80, 60, 4): 7.97,
        (90, 50, 4): 7.97,
        (90, 50, 5): 9.7,
        (100, 40, 4): 7.97,
        (100, 50, 3): 6.6,
        (100, 50, 4): 8.59,
        (100, 50, 5): 10.5,
        (100, 50, 6): 12.3,
        (100, 60, 4): 9.22,
        (100, 60, 5): 11.3,
        (100, 60, 6): 13.2,
        (100, 80, 4): 10.5,
        (120, 60, 5): 12.8,
        (120, 60, 6): 15.1,
        (120, 80, 4): 11.7,
        (120, 80, 5): 14.4,
        (120, 80, 6): 17,
        (120, 80, 8): 21.4,
        (140, 70, 4): 12.4,
        (140, 70, 5): 15.2,
        (140, 80, 4): 13,
        (140, 80, 6): 18.9,
        (150, 100, 4): 14.9,
        (150, 100, 5): 18.3,
        (150, 100, 6): 21.7,
        (150, 100, 8): 27.7,
        (150, 100, 10): 33.4,
        (160, 80, 4): 14.2,
        (160, 80, 5): 17.5,
        (160, 80, 6): 20.7,
        (160, 80, 8): 26.4,
        (180, 100, 6): 24.5,
        (180, 100, 8): 31.4,
        (180, 100, 10): 38.1,
        (200, 100, 5): 22.3,
        (200, 100, 6): 26.4,
        (200, 100, 8): 33.9,
        (200, 100, 10): 41.3,
        (200, 100, 12.5): 48.7,
        (200, 120, 6): 28.3,
        (200, 120, 8): 36.5,
        (200, 120, 10): 44.4,
        (250, 150, 6): 35.8,
        (250, 150, 8): 46.5,
        (250, 150, 10): 57,
        (250, 150, 12.5): 68.3,
        (300, 200, 6): 45.2,
        (300, 200, 8): 59.1,
        (300, 200, 10): 72.7,
        (300, 200, 12.5): 88,
        (400, 200, 6): 54.7,
        (400, 200, 8): 71.6,
        (400, 200, 10): 88.4,
        (400, 200, 12.5): 108,
    },
    "tibnor-vkr-kvadrat": {
        (40, 40, 3): 3.41,
        (40, 40, 4): 4.39,
        (50, 50, 3): 4.35,
        (50, 50, 4): 5.64,
        (50, 50, 5): 6.85,
        (60, 60, 3): 5.29,
        (60, 60, 4): 6.9,
        (60, 60, 5): 8.42,
        (70, 70, 3.6): 7.4,
        (70, 70, 4): 8.15,
        (70, 70, 5): 9.99,
        (80, 80, 3.6): 8.53,
        (80, 80, 4): 9.41,
        (80, 80, 5): 11.6,
        (80, 80, 6.3): 14.2,
        (80, 80, 7.1): 15.8,
        (90, 90, 3.6): 9.66,
        (90, 90, 4): 10.7,
        (90, 90, 5): 13.1,
        (90, 90, 6.3): 16.2,
        (100, 100, 4): 11.9,
        (100, 100, 5): 14.7,
        (100, 100, 6.3): 18.2,
        (100, 100, 8): 22.6,
        (100, 100, 10): 27.4,
        (120, 120, 4.5): 16.1,
        (120, 120, 5): 17.8,
        (120, 120, 6.3): 22.2,
        (120, 120, 8): 27.6,
        (120, 120, 10): 33.7,
        (140, 140, 5): 21,
        (140, 140, 6.3): 26.1,
        (140, 140, 8): 32.6,
        (140, 140, 10): 40,
        (150, 150, 5): 22.6,
        (150, 150, 6.3): 28.1,
        (150, 150, 8): 35.1,
        (150, 150, 10): 43.1,
        (160, 160, 6.3): 30.1,
        (160, 160, 8): 37.6,
        (160, 160, 10): 46.3,
        (180, 180, 6.3): 34,
        (180, 180, 8): 42.7,
        (180, 180, 10): 52.5,
        (200, 200, 6.3): 38,
        (200, 200, 8): 47.7,
        (200, 200, 10): 58.8,
        (200, 200, 12.5): 72.3,
        (200, 200, 16): 90.3,
        (220, 220, 6.3): 41.9,
        (220, 220, 10): 65.1,
        (250, 250, 6.3): 47.9,
        (250, 250, 8): 60.3,
        (250, 250, 10): 74.5,
        (250, 250, 12.5): 91.9,
        (250, 250, 16): 115,
        (300, 300, 10): 90.2,
        (300, 300, 12.5): 112,
        (300, 300, 16): 141,
        (350, 350, 10): 106,
        (350, 350, 12.5): 131,
        (350, 350, 16): 166,
        (400, 400, 10): 122,
        (400, 400, 12.5): 151,
        (400, 400, 16): 191,
    },
    "tibnor-vkr-rektangel": {
        (50, 30, 4): 4.39,
        (60, 40, 3): 4.35,
        (60, 40, 4): 5.64,
        (70, 40, 4): 6.27,
        (80, 40, 4): 6.9,
        (80, 40, 5): 8.42,
        (90, 50, 3.6): 7.4,
        (90, 50, 4): 8.15,
        (90, 50, 5): 9.99,
        (100, 50, 3): 6.71,
        (100, 50, 4): 8.78,
        (100, 50, 5): 10.8,
        (100, 50, 5.6): 11.9,
        (100, 50, 6.3): 13.3,
        (100, 50, 8): 16.3,
        (100, 60, 3.6): 8.53,
        (100, 60, 4): 9.41,
        (100, 60, 5): 11.6,
        (100, 60, 5.6): 12.8,
        (100, 60, 6.3): 14.2,
        (120, 60, 3.6): 9.66,
        (120, 60, 4): 10.7,
        (120, 60, 5): 13.1,
        (120, 60, 6.3): 16.2,
        (120, 80, 4): 11.9,
        (120, 80, 5): 14.7,
        (120, 80, 6.3): 18.2,
        (120, 80, 8): 22.6,
        (140, 70, 4): 12.6,
        (140, 70, 5): 15.5,
        (140, 70, 6.3): 19.2,
        (140, 80, 4): 13.2,
        (140, 80, 6.3): 20.2,
        (150, 100, 5): 18.6,
        (150, 100, 6.3): 23.1,
        (150, 100, 8): 28.9,
        (150, 100, 10): 35.3,
        (160, 80, 4): 14.4,
        (160, 80, 5): 17.8,
        (160, 80, 6.3): 22.2,
        (160, 80, 8): 27.6,
        (160, 80, 10): 33.7,
        (160, 90, 5): 18.6,
        (160, 90, 7.1): 25.9,
        (160, 90, 8): 28.9,
        (180, 100, 5.6): 23.4,
        (180, 100, 6.3): 26.1,
        (180, 100, 8): 32.6,
        (180, 100, 10): 40,
        (200, 100, 5): 22.6,
        (200, 100, 6.3): 28.1,
        (200, 100, 8): 35.1,
        (200, 100, 10): 43.1,
        (200, 100, 12.5): 52.7,
        (200, 120, 6.3): 30.1,
        (200, 120, 8): 37.6,
        (200, 120, 10): 46.3,
        (220, 120, 6.3): 32,
        (220, 120, 8): 40.2,
        (220, 120, 10): 49.4,
    },
    "tibnor-vkr-rektangel-2": {
        (250, 150, 6.3): 38,
        (250, 150, 8): 47.7,
        (250, 150, 10): 58.8,
        (260, 140, 6.3): 38,
        (260, 140, 8): 47.7,
        (260, 140, 12.5): 72.3,
        (300, 200, 6.3): 47.9,
        (300, 200, 8): 60.3,
        (300, 200, 10): 74.5,
        (300, 200, 12.5): 91.9,
        (300, 200, 16): 115,
        (400, 200, 10): 90.2,
        (400, 200, 12.5): 112,
        (400, 200, 16): 141,
        (450, 250, 10): 106,
        (450, 250, 12.5): 131,
        (450, 250, 16): 166,
    },
}

# Angles, kg/m by (a, b, t) in mm, long leg first; lookup tries both orders.
ANGLES: dict[str, dict[tuple[int, int, float], float]] = {
    "am-vinkel-liksidig": {
        (45, 45, 4): 2.74, (50, 50, 4): 3.06, (50, 50, 5): 3.77,
        (50, 50, 6): 4.47, (60, 60, 4): 3.7, (60, 60, 5): 4.57,
        (60, 60, 6): 5.42, (60, 60, 8): 7.09, (75, 75, 4): 4.65,
        (75, 75, 5): 5.76, (75, 75, 6): 6.85, (75, 75, 7): 7.93,
        (75, 75, 8): 8.99, (75, 75, 10): 11.1, (80, 80, 5): 6.17,
        (80, 80, 6): 7.34, (80, 80, 7): 8.49, (80, 80, 8): 9.63,
        (80, 80, 10): 11.9, (90, 90, 6): 8.3, (90, 90, 7): 9.61,
        (90, 90, 8): 10.9, (90, 90, 9): 12.2, (90, 90, 10): 13.4,
        (90, 90, 11): 14.7, (100, 100, 7): 10.7, (100, 100, 8): 12.2,
        (100, 100, 10): 15, (100, 100, 12): 17.8, (120, 120, 8): 14.7,
        (120, 120, 10): 18.2, (120, 120, 11): 19.9, (120, 120, 12): 21.6,
        (120, 120, 13): 23.3, (120, 120, 15): 26.6, (130, 130, 10): 19.8,
        (130, 130, 11): 21.7, (130, 130, 12): 23.6, (130, 130, 13): 25.4,
        (130, 130, 14): 27.2, (130, 130, 15): 29, (130, 130, 16): 30.8,
        (140, 140, 9): 19.3, (140, 140, 10): 21.4, (140, 140, 11): 23.4,
        (140, 140, 12): 25.4, (140, 140, 13): 27.5, (140, 140, 14): 29.4,
        (140, 140, 15): 31.4, (140, 140, 16): 33.3, (140, 140, 18): 37.2,
        (150, 150, 10): 23, (150, 150, 12): 27.3, (150, 150, 13): 29.5,
        (150, 150, 14): 31.6, (150, 150, 15): 33.8, (150, 150, 16): 35.9,
        (150, 150, 18): 40.1, (150, 150, 20): 44.2, (160, 160, 12): 29.3,
        (160, 160, 14): 33.9, (160, 160, 15): 36.2, (160, 160, 16): 38.4,
        (160, 160, 17): 40.7, (160, 160, 18): 42.9, (160, 160, 19): 45.1,
        (160, 160, 20): 47.3, (180, 180, 13): 35.7, (180, 180, 14): 38.3,
        (180, 180, 15): 40.9, (180, 180, 16): 43.5, (180, 180, 17): 46,
        (180, 180, 18): 48.6, (180, 180, 19): 51.1, (180, 180, 20): 53.7,
        (180, 180, 22): 58.6, (200, 200, 13): 39.8, (200, 200, 15): 45.6,
        (200, 200, 16): 48.5, (200, 200, 17): 51.4, (200, 200, 18): 54.2,
        (200, 200, 19): 57.1, (200, 200, 20): 59.9, (200, 200, 21): 62.8,
        (200, 200, 22): 65.6, (200, 200, 23): 68.3, (200, 200, 24): 71.1,
        (200, 200, 25): 73.9, (200, 200, 26): 76.6, (200, 200, 28): 82,
        (250, 250, 17): 64.4, (250, 250, 18): 68.1, (250, 250, 19): 71.7,
        (250, 250, 20): 75.3, (250, 250, 21): 78.9, (250, 250, 22): 82.5,
        (250, 250, 23): 86.1, (250, 250, 24): 89.7, (250, 250, 25): 93.2,
        (250, 250, 26): 96.7, (250, 250, 27): 101, (250, 250, 28): 104,
        (250, 250, 29): 107, (250, 250, 30): 111, (250, 250, 31): 114,
        (250, 250, 32): 118, (250, 250, 33): 121, (250, 250, 34): 124,
        (250, 250, 35): 128, (300, 300, 27): 121, (300, 300, 28): 125,
        (300, 300, 29): 129, (300, 300, 30): 133, (300, 300, 31): 138,
        (300, 300, 32): 142, (300, 300, 33): 146, (300, 300, 34): 150,
        (300, 300, 35): 154,
    },
    "tibnor-vinkel": {
        (15, 15, 3): 0.64, (20, 20, 3): 0.88, (20, 20, 4): 1.14,
        (25, 25, 3): 1.12, (25, 25, 4): 1.45, (25, 25, 5): 1.77,
        (30, 30, 3): 1.36, (30, 30, 4): 1.78, (30, 30, 5): 2.18,
        (35, 35, 3): 1.6, (35, 35, 4): 2.1, (35, 35, 5): 2.57,
        (40, 40, 3): 1.84, (40, 40, 4): 2.42, (40, 40, 5): 2.97,
        (40, 40, 6): 3.52, (40, 40, 8): 4.55, (45, 45, 5): 3.38,
        (45, 45, 7): 4.6, (50, 50, 7): 5.15, (50, 50, 8): 5.82,
        (55, 55, 6): 4.95, (60, 60, 10): 8.69, (65, 65, 7): 6.83,
        (70, 70, 7): 7.38, (70, 70, 9): 9.34, (80, 80, 12): 14.1,
        (100, 100, 14): 20.6, (110, 110, 10): 16.6, (110, 110, 12): 19.7,
        (140, 140, 13): 27.5, (180, 180, 18): 48.6, (200, 200, 18): 54.3,
    },
    "am-vinkel-oliksidig": {
        (120, 80, 8): 12.2, (120, 80, 10): 15, (120, 80, 12): 17.8,
        (200, 100, 10): 23, (200, 100, 12): 27.3, (200, 100, 14): 31.6,
        (200, 100, 15): 33.7, (200, 100, 16): 35.9, (250, 90, 12): 31.2,
        (250, 90, 14): 36.1, (250, 90, 16): 40.9,
    },
}

# Circular hollow sections, kg/m by (D, t) in mm.
CHS: dict[tuple[float, float], float] = {
    (42.4, 3): 2.91, (42.4, 4): 3.79, (48.3, 3): 3.35, (48.3, 4): 4.37,
    (60.3, 3): 4.24, (60.3, 4): 5.55, (76.1, 4): 7.11, (76.1, 5): 8.77,
    (88.9, 4): 8.38, (88.9, 5): 10.3, (101.6, 4): 9.63, (101.6, 5): 11.9,
    (101.6, 6): 14.1, (114.3, 4): 10.9, (114.3, 5): 13.5, (139.7, 4): 13.4,
    (139.7, 6): 19.8, (139.7, 8): 26, (168.3, 4): 16.2, (168.3, 6): 24,
    (168.3, 8): 31.6, (193.7, 6): 27.8, (193.7, 8): 36.6, (193.7, 10): 45.3,
    (193.7, 12.5): 55.9, (219.1, 6): 31.5, (219.1, 8): 41.6, (219.1, 10): 51.6,
    (219.1, 12.5): 63.7, (244.5, 6): 35.3, (244.5, 8): 46.7, (244.5, 10): 57.8,
    (244.5, 12.5): 71.5, (273, 8): 52.3, (273, 10): 64.9, (273, 12.5): 80.3,
    (323.9, 10): 77.4, (323.9, 12.5): 96,
}
CHS_SOURCE = "tibnor-kckr"

# Which standard each hollow table is: VKR is hot-finished (EN 10210), KKR
# cold-formed (EN 10219).
HOLLOW_KIND = {key: ("VKR" if "-vkr-" in key else "KKR") for key in HOLLOW}

# Thin-sheet studs: (kg/m, source, sheet thickness in mm or None when the
# source does not state it), by kind and web width.
STUDS: dict[tuple[str, int], tuple[float, str, float | None]] = {
    ("regel", 45): (0.49, "norgips-c45", 0.5),
    ("regel", 70): (0.59, "norgips-c70", 0.5),
    ("regel", 95): (0.68, "norgips-c95", 0.5),
    ("regel", 120): (0.78, "norgips-c120", 0.5),
    ("regel", 145): (0.88, "norgips-c145", 0.5),
    ("regel", 160): (0.94, "norgips-c160", 0.5),
    ("förstärkningsregel", 45): (1.13, "norgips-cf45", 1.0),
    ("förstärkningsregel", 70): (1.62, "norgips-cf70", 1.0),
    ("förstärkningsregel", 95): (1.85, "norgips-cf95", 1.0),
    ("förstärkningsregel", 120): (1.71, "norgips-cf120", 1.0),
    ("förstärkningsregel", 145): (1.93, "norgips-cf145", 1.0),
    ("förstärkningsregel", 160): (2.10, "norgips-cf160", 1.0),
}
STUD_WIDTHS = "45, 70, 95, 120, 145 eller 160 mm"
# The sheet a standard stud's weight may be scaled to (see the docstring).
SCALABLE_SHEET_MM = (0.4, 1.0)

OPEN = "öppen"
HOLLOW_FORM = "rör"


@dataclass(frozen=True)
class ProfileMass:
    """The weight of one metre of a named profile, and where it comes from."""

    designation: str  # as AIda writes it: "HEA 200", "VKR 100x100x5", "C 70"
    family: str  # STRUCTURAL or STUD
    kg_per_m: float
    source: str  # the citation, in Swedish, from SOURCES
    form: str = ""  # OPEN or HOLLOW_FORM for structural steel, "" for a stud

    @property
    def label(self) -> str:
        """'HEA 200: 42,3 kg/m (ArcelorMittal ...)' for a row's text."""
        return f"{self.designation}: {_sv(self.kg_per_m)} kg/m ({self.source})"


def _sv(x: float) -> str:
    return f"{x:g}".replace(".", ",")


def _fold(text: str) -> str:
    """Lowercase, composed, and without the dot a Turkish capital I leaves
    behind when lowercased. å, ä and ö stay: the patterns below are Swedish."""
    return unicodedata.normalize("NFC", (text or "").lower().replace("̇", ""))


_NUM = r"(\d{1,2}(?:[.,]\d{1,2})?)"
# The end of a number: not more digits, not a decimal ("200,5"), not the
# first side of a size ("200x"). A comma before a space ends it ("HEA 200, 6 m").
_END = r"(?!\d|[.,]\d|\s*[x×*]\s*\d)"
# "HEA 200", "HEA200", "HEA-200", "IPE 200". Not inside a word. An "x" after
# it is a length ("HEA 200 x 6 m"): no hot-rolled designation has a second
# dimension, so only a decimal or more digits end the match (review 2026-10-01).
_HOT_RE = re.compile(
    r"(?<![a-zåäö0-9])(hea|heb|hem|ipe|upe|upn|unp)\s*-?\s*(\d{2,4})(?!\d|[.,]\d)")
# Ipe (ipé) is also a tropical hardwood sold as decking: "Trall ipe 28 mm" and
# "Ipe 80 trall" are wood. An IPE next to one of these words is not read.
_WOOD_RE = re.compile(r"trall|trä|virke|decking|bräd|panel|golv|wood|timber|hardwood|lumber")
# "HE 200 A", "HE200B", the designation EN 10365 itself uses.
_HE_RE = re.compile(r"(?<![a-zåäö0-9])he\s*-?\s*(\d{3,4})\s*([abm])(?![a-zåäö0-9])")
# "VKR 100x100x5", "KKR-rör 120x80x4,0", "VKR 100x100 t=5", "VKR 100x100 t5".
_HOLLOW_RE = re.compile(
    r"(?<![a-zåäö0-9])(vkr|kkr)(?:\s*-?\s*rör)?\s*(\d{2,3})\s*[x×*]\s*(\d{2,3})"
    r"(?:\s*[x×*]\s*" + _NUM + r"|\s*,?\s*t\s*=?\s*" + _NUM + r")?" + _END)
_HOLLOW_WORD_RE = re.compile(r"(?<![a-zåäö0-9])(vkr|kkr)(?![a-zåäö])")
# A round tube: "VKR 114,3x5", "KCKR 114,3x5,0", "CHS 168,3x8". Only a
# diameter the table lists counts as one, so "VKR 100x50" (a rectangle missing
# its wall) is still the rectangular reader's to answer.
_CHS_RE = re.compile(
    r"(?<![a-zåäö0-9])(vkr|kkr|kckr|vckr|chs|cfchs|hfchs)(?:\s*-?\s*rör)?\s*"
    r"(\d{2,3}(?:[.,]\d)?)\s*[x×*]\s*" + _NUM + _END)
# "L 50x50x5", "Vinkelstål 120x80x8", "VST 50x50x5". The bare "L" needs all
# three numbers; the words alone are enough to be asked for the thickness.
_ANGLE_RE = re.compile(
    r"(?<![a-zåäö0-9])(l|vst|vinkelstål|vinkelstång|vinkeljärn)\s*-?\s*(\d{2,3})"
    r"\s*[x×*]\s*(\d{2,3})(?:\s*[x×*]\s*" + _NUM + r")?" + _END)
_ANGLE_WORD_RE = re.compile(r"vinkelstål|vinkelstång|vinkeljärn|(?<![a-zåäö])vst(?![a-zåäö])")
# A stud's web width, alone or after its maker's letters ("C70", "CF 70",
# "RE 70"). Not a cross-section ("45x70"), not a decimal ("2,7"), and not a
# count or a length ("12 st", "3 m", "2700").
_STUD_WIDTH_RE = re.compile(
    r"(?<![\d.,x×*])(\d{2,3})(?:\s*mm)?" + _END
    + r"(?!\s*(?:st|styck|m|meter|lm|löpmeter|%)(?![a-zåäö]))")
# A steel stud named as one: the stomme family's own words, plus the
# one-letter profile names.
_STEEL_STUD_RE = re.compile(
    r"stålregel|stålreglar|(?<![a-zåäö])c\s*-?\s*(?:regel|reglar|profil)"
    r"|(?<![a-zåäö])cf\s*-?\s*\d")
_STIFFENER_RE = re.compile(r"förstärkningsregel|förstärkningsreglar|(?<![a-zåäö])cf\s*-?\s*\d")
# A sheet thickness: "0,5 mm", "1,0 mm", "t=0,7".
_SHEET_RE = re.compile(r"(?<![\d.,])([0-2][.,]\d{1,2})\s*mm|\bt\s*=\s*([0-2][.,]\d{1,2})")
_SECTION_RE = re.compile(r"\d{2,4}\s*[x×*]\s*\d{2,4}")


def names_steel_profile(text: str) -> bool:
    """True when `text` carries a hot-rolled or hollow-section designation
    ("HEA 200", "VKR 100x100x5"), the thing itself and so stomme."""
    t = _fold(text)
    return bool(_hot_matches(t) or _HE_RE.search(t) or _HOLLOW_WORD_RE.search(t)
                or _chs_matches(t) or _ANGLE_WORD_RE.search(t)
                or any(m.group(4) for m in _ANGLE_RE.finditer(t)))


def _hot_matches(t: str) -> list[re.Match]:
    """_HOT_RE's matches, without an IPE that is the decking wood."""
    wood = bool(_WOOD_RE.search(t))
    return [m for m in _HOT_RE.finditer(t) if not (wood and m.group(1) == "ipe")]


def _num(s: str) -> float:
    return float(s.replace(",", "."))


def _hot_rolled(t: str) -> tuple[ProfileMass | None, str]:
    found = {(m.group(1).upper(), int(m.group(2))) for m in _hot_matches(t)}
    found |= {("HE" + m.group(2).upper(), int(m.group(1))) for m in _HE_RE.finditer(t)}
    if not found:
        return None, ""
    if len(found) > 1:
        names = ", ".join(f"{s} {h}" for s, h in sorted(found))
        return None, f"Namnet anger flera profiler ({names}), så vikten per meter är inte entydig."
    series, height = found.pop()
    series = "UPN" if series == "UNP" else series
    source_key, table = HOT_ROLLED[series]
    kg = table.get(height)
    if kg is None:
        return None, (f"{series} {height} finns inte i standardtabellen (EN 10365), "
                      f"så vikten per meter är okänd.")
    return ProfileMass(f"{series} {height}", STRUCTURAL, kg, SOURCES[source_key], OPEN), ""


def _hollow(t: str) -> tuple[ProfileMass | None, str]:
    matches = list(_HOLLOW_RE.finditer(t))
    if not matches:
        if _chs_matches(t):
            return None, ""  # a round tube, _chs answers it
        if _HOLLOW_WORD_RE.search(t):
            return None, ("Ett VKR- eller KKR-rör behöver ytterdimension och väggtjocklek "
                          "i namnet, till exempel \"VKR 100x100x5\".")
        return None, ""
    found = set()
    for m in matches:
        wall = m.group(4) or m.group(5)
        found.add((m.group(1).upper(), int(m.group(2)), int(m.group(3)),
                   _num(wall) if wall else None))
    if len(found) > 1:
        return None, "Namnet anger flera rördimensioner, så vikten per meter är inte entydig."
    kind, b, h, wall = found.pop()
    if wall is None:
        return None, (f"{kind} {b}x{h} saknar väggtjocklek, och vikten per meter beror på "
                      f"den. Ange den i namnet, till exempel \"{kind} {b}x{h}x5\".")
    for key, table in HOLLOW.items():
        if HOLLOW_KIND[key] != kind:
            continue
        kg = table.get((b, h, wall)) or table.get((h, b, wall))
        if kg is not None:
            return ProfileMass(f"{kind} {b}x{h}x{_sv(wall)}", STRUCTURAL, kg, SOURCES[key],
                               HOLLOW_FORM), ""
    return None, (f"{kind} {b}x{h}x{_sv(wall)} finns inte i tabellen över standardrör, "
                  f"så vikten per meter är okänd.")


_CHS_DIAMETERS = {d for d, _ in CHS}


def _chs_matches(t: str) -> set[tuple[str, float, float]]:
    found = set()
    for m in _CHS_RE.finditer(t):
        d = _num(m.group(2))
        if d in _CHS_DIAMETERS:
            found.add((m.group(1).upper(), d, _num(m.group(3))))
    return found


def _chs(t: str) -> tuple[ProfileMass | None, str]:
    found = _chs_matches(t)
    if not found:
        return None, ""
    if len({(d, w) for _, d, w in found}) > 1:
        return None, "Namnet anger flera rördimensioner, så vikten per meter är inte entydig."
    kind, d, wall = found.pop()
    kg = CHS.get((d, wall))
    designation = f"{kind} {_sv(d)}x{_sv(wall)}"
    if kg is None:
        return None, (f"{designation} finns inte i tabellen över runda standardrör, "
                      f"så vikten per meter är okänd.")
    return ProfileMass(designation, STRUCTURAL, kg, SOURCES[CHS_SOURCE], HOLLOW_FORM), ""


def _angle(t: str) -> tuple[ProfileMass | None, str]:
    matches = [m for m in _ANGLE_RE.finditer(t)
               if m.group(4) or m.group(1) != "l"]
    if not matches:
        if _ANGLE_WORD_RE.search(t):
            return None, ("Ett vinkelstål behöver skänklar och godstjocklek i namnet, "
                          "till exempel \"Vinkelstål 50x50x5\".")
        return None, ""
    found = {(int(m.group(2)), int(m.group(3)), _num(m.group(4)) if m.group(4) else None)
             for m in matches}
    if len(found) > 1:
        return None, "Namnet anger flera vinkelstål, så vikten per meter är inte entydig."
    a, b, wall = found.pop()
    if wall is None:
        return None, (f"Vinkelstål {a}x{b} saknar godstjocklek, och vikten per meter beror "
                      f"på den. Ange den i namnet, till exempel \"L {a}x{b}x5\".")
    for key, table in ANGLES.items():
        kg = table.get((a, b, wall)) or table.get((b, a, wall))
        if kg is not None:
            long, short = max(a, b), min(a, b)
            return ProfileMass(f"L {long}x{short}x{_sv(wall)}", STRUCTURAL, kg, SOURCES[key],
                               OPEN), ""
    return None, (f"L {a}x{b}x{_sv(wall)} finns inte i tabellen över standardvinklar, "
                  f"så vikten per meter är okänd.")


def _stud(t: str) -> tuple[ProfileMass | None, str]:
    if not _STEEL_STUD_RE.search(t):
        return None, ""
    if _SECTION_RE.search(t):
        # "45x95" is a timber section; a steel stud named with one is not a
        # designation this table knows.
        return None, ""
    widths = {int(m.group(1)) for m in _STUD_WIDTH_RE.finditer(t)}
    kind = "förstärkningsregel" if _STIFFENER_RE.search(t) else "regel"
    known = {w for w in widths if (kind, w) in STUDS}
    if len(widths) != 1 or not known:
        if not widths:
            return None, ("Namnet anger inte regelns bredd. Ange den, till exempel "
                          f"\"Stålregel 70\" ({STUD_WIDTHS}).")
        return None, ("Regelns bredd går inte att läsa entydigt ur namnet, eller är ingen "
                      f"av standardbredderna {STUD_WIDTHS}.")
    width = known.pop()
    kg, source_key, sheet = STUDS[(kind, width)]
    prefix = "CF" if kind == "förstärkningsregel" else "C"
    stated = {_num(a or b) for a, b in _SHEET_RE.findall(t)}
    if not stated or (sheet is not None and stated == {sheet}):
        return ProfileMass(f"{prefix} {width}", STUD, kg, SOURCES[source_key]), ""
    low, high = SCALABLE_SHEET_MM
    if len(stated) == 1 and kind == "regel" and sheet:
        thickness = next(iter(stated))
        if low <= thickness <= high:
            # Same profile, other sheet: the developed width times the sheet.
            scaled = round(kg * thickness / sheet, 3)
            source = (f"{SOURCES[source_key]}, {_sv(kg)} kg/m vid {_sv(sheet)} mm plåt, "
                      f"räknat om till {_sv(thickness)} mm")
            return ProfileMass(f"{prefix} {width} {_sv(thickness)} mm", STUD, scaled, source), ""
    return None, ("Namnet anger en plåttjocklek som tabellen inte kan bekräfta, så "
                  "vikten per meter är okänd.")


def profile_mass(name: str) -> tuple[ProfileMass | None, str]:
    """(ProfileMass, "") for a name carrying a known designation, else
    (None, reason). The reason is "" when the name names no steel profile at
    all, and otherwise a Swedish sentence saying what is missing."""
    t = _fold(name)
    answers = [a for a in (reader(t) for reader in (_hot_rolled, _hollow, _chs, _angle, _stud))
               if a[0] or a[1]]
    if len(answers) > 1:
        # "HEA 200 + VKR 100x100x5": two kinds of profile in one component,
        # and no single weight per metre (review 2026-10-01).
        return None, "Namnet anger flera profiler, så vikten per meter är inte entydig."
    return answers[0] if answers else (None, "")


# What an EPD in konstruktionsstål declares, by its name (HENRIC-3390). Open
# sections are mostly rolled from scrap in an electric furnace, hollow
# sections mostly formed from hot-rolled coil, so the two sit apart in the
# catalog: upper-half medians 2,05 and 2,78 kg CO2e/kg on 2026-10-02, the
# family 2,63. A name naming both, or neither (fabricated components,
# "structural steel" as such, galvanized assemblies), has no form and counts
# only in the family.
_EPD_HOLLOW_RE = re.compile(r"hollow|tube|\brör\b|pipe", re.I)
_EPD_OPEN_RE = re.compile(
    r"beam|merchant bar|angle|channel|i-section|\bh, i, u\b|steel profiles|steel sections",
    re.I)


def epd_form(name: str) -> str:
    """OPEN, HOLLOW_FORM or "" for a structural steel EPD's name."""
    hollow = bool(_EPD_HOLLOW_RE.search(name or ""))
    open_ = bool(_EPD_OPEN_RE.search(name or ""))
    if hollow == open_:
        return ""
    return HOLLOW_FORM if hollow else OPEN


_FORM_LABEL = {OPEN: "öppna profiler", HOLLOW_FORM: "rör"}


def form_subcategory(subcategory: str, form: str) -> str:
    """The typvärde key of a family's form: "konstruktionsstål (öppna profiler)".
    Read as it stands in a row's text, so it is written for a reader."""
    return f"{subcategory} ({_FORM_LABEL[form]})"
