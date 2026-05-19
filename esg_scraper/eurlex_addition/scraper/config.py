"""Registry of company → scraper class.

Add a new company by:
  1. Creating extractors/newco.py with a class subclassing BaseScraper
  2. Importing and registering it here
"""
from .extractors.totalenergies import TotalEnergiesScraper
from .extractors.engie import EngieScraper
from .extractors.bnp_paribas import BNPParibasScraper
from .extractors.iberdrola import IberdrolaScraper
from .extractors.airbus import AirbusScraper
from .extractors.danone import DanoneScraper
from .extractors.loreal import LOrealScraper
from .extractors.volkswagen import VolkswagenScraper
from .extractors.siemens import SiemensScraper
from .extractors.schneider import SchneiderScraper
from .extractors.enel import EnelScraper
from .extractors.eurlex import EURLexScraper

SCRAPERS = {
    "totalenergies":    TotalEnergiesScraper,
    "engie":            EngieScraper,
    "bnp_paribas":      BNPParibasScraper,
    "iberdrola":        IberdrolaScraper,
    "airbus":           AirbusScraper,
    "danone":           DanoneScraper,
    "loreal":           LOrealScraper,
    "volkswagen":       VolkswagenScraper,
    "siemens":          SiemensScraper,
    "schneider":        SchneiderScraper,
    "enel":             EnelScraper,
    "eu_regulators":    EURLexScraper,
}
