"""Rebuild the three corpora behind `_SHOUTED_MIN_STOPWORD` and `_SHOUTED_MIN_WORD`.

`laya.lang` holds an all-caps line to an evidence bar before it will name it as foreign prose:
a matched stopword of `_SHOUTED_MIN_STOPWORD` letters (or non-English diacritics), and one word
of `_SHOUTED_MIN_WORD` letters. Those two constants were chosen against three corpora, and the
corpora were not committed, so the trade behind them could not be rechecked when someone wanted
to move them. This script rebuilds all three from pinned sources and sweeps the bar over them.

    prose    MASSIVE test split @ REVISION, every row upper-cased, keeping the rows the
             *unguarded* rule names as their own locale. Genuine shouted prose: it must survive.
    acronym  `ACRONYM_LLINES` lines of 4-8 tokens drawn from `ACRONYMS` below, a hand-checked
             list of real abbreviations. A false positive is a line the bar lets through.
    address  `ADDRESS_LINES` all-caps US address and signage lines built from `PLACES`,
             `STREETS` and `SIGNAGE` below. Same false-positive definition.

Deterministic: `random.Random(SEED)` and nothing else. Run from the repository root:

    python research/evals/shouted_bar_corpora.py            # the table
    python research/evals/shouted_bar_corpora.py --out x.json

The pools here are smaller than the ones the bar was originally chosen against (that run used
5,670 acronyms and 18,718 place names, neither committed, so its absolute counts are not
reproducible and are not quoted anywhere). What this script pins is the shape of the trade: the
ordering of the rows, and the marginal cost in real prose of each step of the bar.

No model weights, no network beyond the dataset download `massive_route_comparison.py` already
does, and no dependency on the current value of either constant -- the bar is applied here from
the swept parameters, so the table stays meaningful after the constants move.
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from laya.lang import (  # noqa: E402
    _CODE_LINE, _JOINED, _LETTER_RUN, _STOP, _WORD, NON_EN_DIACRITIC_RATE, latin_profile)

DATASET = "mteb/amazon_massive_intent"
REVISION = "940fd47a81eaa7f2cc7b129674d945d618ac38c2"
SEED = 0
ACRONYM_LINES = 500_000
ADDRESS_LINES = 50_000

# The 51 MASSIVE locales, as the snapshot names its test files.
LOCALES = [
    "af", "am", "ar", "az", "bn", "cy", "da", "de", "el", "en", "es", "fa", "fi", "fr", "he",
    "hi", "hu", "hy", "id", "is", "it", "ja", "jv", "ka", "km", "kn", "ko", "lv", "ml", "mn",
    "ms", "my", "nb", "nl", "pl", "pt", "ro", "ru", "sl", "sq", "sv", "sw", "ta", "te", "th",
    "tl", "tr", "ur", "vi", "zh-CN", "zh-TW",
]

# Real abbreviations, upper-cased as they are written. Chosen for breadth, and deliberately
# including the ones that collide with a Romance or Germanic stopword -- MON, DES, EST, LA, COM,
# DOS, LOS, LAS, EL, QUE, LES, UNO, SON, VAN, DAS -- because those are what the bar exists for.
ACRONYMS = """
ABS ACH ACL ADA ADD ADHD ADSL AFK AGI AIDS AKA ALS AMA AMD AML ANSI AOL API APR APT ARM ARP
ASAP ASCII ASIC ASL ASM ASP ATM ATP AVI AWOL AWS BBC BBQ BBS BCC BIOS BLT BMI BMW BPM BRB BSD
BTU BTW CAD CAM CAPTCHA CBC CBS CCTV CDC CDN CEO CFO CGI CIA CIO CLI CMS CNC CNN COD COM CPA
CPI CPR CPU CRM CRT CSI CSS CSV CTO CUDA CV DAB DARPA DAW DBA DDL DDOS DEA DEF DEI DES DHCP
DHL DIY DJ DLC DLL DMA DMV DNA DNS DOB DOC DOD DOE DOJ DOS DPI DRAM DSL DST DUI DVD DVI EAP
EBCDIC ECG ECU EDI EDT EEG EEOC EFT EKG EMI EMS EMT EOD EOF EOL EPA EPS ERP ESA ESL ESP ESPN
EST ETA ETC ETF ETL EULA EUR EVP FAA FAQ FBI FCC FDA FDIC FEMA FIFA FIFO FM FOIA FPGA FPS FTC
FTP FWD FYI GAAP GB GBP GDP GIF GIS GMO GMT GNU GOP GPA GPS GPU GSM GUI HBO HDD HDMI HIPAA
HIV HMO HOA HP HR HTML HTTP HUD HVAC IBM ICBM ICU IDE IEEE IETF IMAP IMDB IMF IMO INC IP IPO
IQ IRA IRC IRS ISBN ISO ISP ISS IT IV JDK JPEG JPG JSON JVM KB KFC KGB KPI KYC LAN LAPD LAS
LATAM LCD LED LES LIFO LLC LLM LMS LNG LOL LOS LPG LSAT LTD LTE LTS MAC MBA MBPS MCAT MIA MIB
MIDI MIT MLB MLS MMO MMS MON MPA MPEG MPG MPH MRI MRP MSDS MSG MSRP MTA MTV MVP NAACP NAFTA
NAS NASA NASCAR NATO NBA NBC NCAA NDA NFC NFL NGO NHL NHS NIC NIH NIST NLP NOAA NPR NRA NSA
NSF NTP NTSB NVME NYC NYPD NYSE OCD OCR ODBC OECD OEM OLED OPEC ORM OSHA OTC OTP PAC PAN PBS
PCB PCI PDA PDF PDT PEM PGA PGP PHD PHP PIN PLC PLZ PMS PNG PO POP POS PPE PPI PPM PPO PPP
PPT PSA PSI PST PTA PTO PTSD PVC PWD QA QC QED QOS QUE RAID RAM RBI RCA RDBMS REIT REM RFC
RFID RGB RIP RNA ROI ROM RPG RPM RSA RSS RSVP RTF RTO RV SAAS SAE SAN SAS SAT SCSI SDK SEC
SEO SFO SGML SHA SIM SKU SLA SMB SMS SMTP SNMP SOAP SOC SOP SOS SPF SQL SRAM SSD SSH SSID SSL
SSN SSO STD STEM SUV SVG SWAT SYN TARP TBA TBD TCP TDD TED TGIF TIFF TLC TLD TLS TMI TNT TSA
TTL TTY TV TXT UAE UAV UDP UEFA UFC UFO UHD UI UK UML UNESCO UNICEF UNO UPC UPS URI URL USA
USB USD USDA USGS USPS UTC UTF UUID UV UX VA VAT VGA VHF VHS VIN VIP VLAN VOIP VPN VPS VR WAN
WAV WBC WEP WHO WIFI WMA WPA WTO WWF WWII WWW XHTML XML XSD XSS YMCA YOLO YTD ZIP
LA EL AL AZ CA CO CT DC DE FL GA HI IA ID IL IN KS KY MA MD ME MI MN MO MS MT NC ND NE NH NJ
NM NV NY OH OK OR PA RI SC SD TN TX UT VT WA WI WV WY AB BC MB NB NL NS NT NU ON PE QC SK YT
ATL BOS CDG DEN DFW DTW EWR FRA HKG IAD IAH JFK LAX LGA LHR MCO MDW MEX MIA MSP MUC NRT ORD
PDX PHL PHX SAN SEA SFO SJC SLC STL SYD YUL YVR YYZ ZRH
""".split()

# Real US place names, written as a sign would write them. The Spanish-derived ones are the
# point: `LOS`, `LAS` and `EL` are what named Spanish in the defect this bar closed.
PLACES = """
ALBUQUERQUE ALEXANDRIA ALLENTOWN AMARILLO ANAHEIM ANCHORAGE ANN_ARBOR ANTIOCH APPLETON ARLINGTON
ARVADA ASHEVILLE ATHENS ATLANTA AUGUSTA AURORA AUSTIN BAKERSFIELD BALTIMORE BATON_ROUGE BEAUMONT
BELLEVUE BERKELEY BETHLEHEM BILLINGS BIRMINGHAM BLOOMINGTON BOISE BOSTON BOULDER BRIDGEPORT
BROCKTON BROKEN_ARROW BROWNSVILLE BUFFALO BURBANK CAMBRIDGE CAMDEN CANTON CAPE_CORAL CARLSBAD
CARROLLTON CARY CEDAR_RAPIDS CHANDLER CHARLESTON CHARLOTTE CHATTANOOGA CHESAPEAKE CHEYENNE
CHICAGO CHULA_VISTA CINCINNATI CLARKSVILLE CLEARWATER CLEVELAND CLOVIS COLLEGE_STATION
COLORADO_SPRINGS COLUMBIA COLUMBUS CONCORD CORAL_SPRINGS CORONA CORPUS_CHRISTI COSTA_MESA
DALLAS DALY_CITY DANBURY DAVENPORT DAYTON DEARBORN DENTON DENVER DES_MOINES DETROIT DOWNEY
DULUTH DURHAM EL_CAJON EL_MONTE EL_PASO EL_SEGUNDO ELGIN ELIZABETH ELK_GROVE ERIE ESCONDIDO
EUGENE EVANSVILLE EVERETT FAIRFIELD FARGO FAYETTEVILLE FLINT FONTANA FORT_COLLINS FORT_LAUDERDALE
FORT_WAYNE FORT_WORTH FREMONT FRESNO FRISCO FULLERTON GAINESVILLE GARDEN_GROVE GARLAND GARY
GILBERT GLENDALE GRAND_PRAIRIE GRAND_RAPIDS GREELEY GREEN_BAY GREENSBORO GREENVILLE HAMPTON
HARTFORD HAYWARD HENDERSON HIALEAH HIGH_POINT HOLLYWOOD HONOLULU HOUSTON HUNTINGTON_BEACH
HUNTSVILLE INDEPENDENCE INDIANAPOLIS INGLEWOOD IRVINE IRVING JACKSON JACKSONVILLE JERSEY_CITY
JOLIET KANSAS_CITY KENOSHA KILLEEN KNOXVILLE LA_CROSSE LA_HABRA LA_MESA LA_PORTE LAFAYETTE
LAKEWOOD LANCASTER LANSING LAREDO LAS_CRUCES LAS_VEGAS LEWISVILLE LEXINGTON LINCOLN LITTLE_ROCK
LIVONIA LONG_BEACH LONGMONT LOS_ANGELES LOS_BANOS LOUISVILLE LOWELL LUBBOCK LYNN MACON MADISON
MANCHESTER MCALLEN MCKINNEY MEMPHIS MERIDIAN MESA MESQUITE MIAMI MIDLAND MILWAUKEE MINNEAPOLIS
MIRAMAR MOBILE MODESTO MONTGOMERY MORENO_VALLEY MURFREESBORO MURRIETA NAPERVILLE NASHUA NASHVILLE
NEW_HAVEN NEW_ORLEANS NEW_YORK NEWARK NEWPORT_NEWS NORFOLK NORMAN NORTH_LAS_VEGAS NORWALK OAKLAND
OCEANSIDE ODESSA OKLAHOMA_CITY OLATHE OMAHA ONTARIO ORANGE ORLANDO OVERLAND_PARK OXNARD PALM_BAY
PALMDALE PALO_ALTO PASADENA PATERSON PEMBROKE_PINES PEORIA PHILADELPHIA PHOENIX PITTSBURGH PLANO
POMONA PORT_ST_LUCIE PORTLAND PROVIDENCE PROVO PUEBLO RALEIGH RANCHO_CUCAMONGA READING RENO
RICHARDSON RICHMOND RIVERSIDE ROANOKE ROCHESTER ROCKFORD ROSEVILLE ROUND_ROCK SACRAMENTO SAGINAW
SALEM SALINAS SALT_LAKE_CITY SAN_ANTONIO SAN_BERNARDINO SAN_DIEGO SAN_FRANCISCO SAN_JOSE
SAN_MATEO SANDY_SPRINGS SANTA_ANA SANTA_BARBARA SANTA_CLARA SANTA_CLARITA SANTA_FE SANTA_MARIA
SANTA_MONICA SANTA_ROSA SAVANNAH SCOTTSDALE SEATTLE SHREVEPORT SIMI_VALLEY SIOUX_CITY SIOUX_FALLS
SOUTH_BEND SPARKS SPOKANE SPRINGFIELD ST_LOUIS ST_PAUL ST_PETERSBURG STAMFORD STERLING_HEIGHTS
STOCKTON SUNNYVALE SURPRISE SYRACUSE TACOMA TALLAHASSEE TAMPA TEMECULA TEMPE THORNTON THOUSAND_OAKS
TOLEDO TOPEKA TORRANCE TRENTON TUCSON TULSA TUSCALOOSA TYLER VACAVILLE VALLEJO VANCOUVER VENTURA
VICTORVILLE VIRGINIA_BEACH VISALIA WACO WARREN WASHINGTON WATERBURY WEST_COVINA WEST_JORDAN
WESTMINSTER WICHITA WILMINGTON WINSTON_SALEM WORCESTER YONKERS YUMA
""".split()

STREETS = """
AVE BLVD CIR CT DR EXPY HWY LN PKWY PL RD ROUTE ST TER TRL WAY
""".split()

SIGNAGE = """
ACCESS ALL ARRIVALS CLOSED CUSTOMER DAILY DELIVERIES DELIVERY DEPARTURES DEPOT DOCK ENTRANCE
EXIT FREIGHT GATE HOURS LOADING LOT NO OFFICE ONLY OPEN PARKING PERMIT PICKUP PLANT RECEIVING
RETURNS SERVICE SHIP SHIPPING STAFF STORE STORES SUITE TERMINAL TO UNIT VEHICLES VISITOR
WAREHOUSE YARD
""".split()


def name_unguarded(segment: str) -> Optional[Tuple[str, List[str], float]]:
    """`_named_prose_language` with the all-caps evidence bar left off.

    Returns (language, tokens, diacritic_rate) when the rule would name the segment foreign
    with no bar at all, else None. Reimplemented here rather than imported so the sweep below
    does not depend on the value of either constant it is sweeping.
    """
    if not segment.strip() or _CODE_LINE.search(segment):
        return None
    prose = " ".join(tok for tok in segment.split() if not _JOINED.search(tok))
    shouted = any(ch.isupper() for ch in prose) and not any(ch.islower() for ch in prose)
    if not shouted:
        prose = _LETTER_RUN.sub(lambda m: " " if m.group().isupper() else m.group(), prose)
    tokens = _WORD.findall(prose)
    if len(tokens) < 4:
        return None
    prof = latin_profile(prose)
    lang = prof["language"]
    if lang in (None, "en"):
        return None
    if len({w.lower() for w in tokens} & _STOP.get(lang, set())) < 2:
        return None
    return lang, tokens, float(prof["diacritic_rate"])


def accepts(named: Tuple[str, List[str], float], sw_bar: int, wd_bar: int, dia: bool) -> bool:
    """Whether a bar of (`sw_bar`, `wd_bar`, diacritic escape `dia`) lets a named line through."""
    lang, tokens, rate = named
    if wd_bar and not any(len(t) >= wd_bar for t in tokens):
        return False
    if dia and rate >= NON_EN_DIACRITIC_RATE:
        return True
    if not sw_bar:
        return True
    matched = {t.lower() for t in tokens} & _STOP.get(lang, set())
    return any(len(w) >= sw_bar for w in matched)


def massive_rows() -> Dict[str, List[str]]:
    """The pinned test split, from the hub or from an already-downloaded snapshot of it."""
    from huggingface_hub import hf_hub_download
    rows: Dict[str, List[str]] = {}
    for loc in LOCALES:
        name = "test/%s.json.gz" % loc
        try:
            path = hf_hub_download(DATASET, name, revision=REVISION, repo_type="dataset")
        except Exception:  # offline: the same revision, already in the local cache
            path = (Path.home() / ".cache/huggingface/hub"
                    / ("datasets--%s" % DATASET.replace("/", "--"))
                    / "snapshots" / REVISION / name)
            if not path.exists():
                raise
        with gzip.open(str(path), "rt", encoding="utf-8") as fh:
            rows[loc] = [json.loads(line)["text"] for line in fh if line.strip()]
    return rows


def build_prose(rows: Dict[str, List[str]]) -> List[Tuple[str, Tuple]]:
    """Upper-cased MASSIVE rows the unguarded rule names as their own locale."""
    out = []
    for loc, texts in rows.items():
        for text in texts:
            named = name_unguarded(text.upper())
            if named is not None and named[0] == loc:
                out.append((text.upper(), named))
    return out


def build_acronyms(rng: random.Random, n: int) -> List[Tuple[str, Optional[Tuple]]]:
    out = []
    for _ in range(n):
        k = rng.randint(4, 8)
        line = " ".join(rng.choice(ACRONYMS) for _ in range(k))
        out.append((line, name_unguarded(line)))
    return out


def build_addresses(rng: random.Random, n: int) -> List[Tuple[str, Optional[Tuple]]]:
    """All-caps address and signage lines. `_` in a place name stands for a space."""
    place = lambda: rng.choice(PLACES).replace("_", " ")
    out = []
    for _ in range(n):
        shape = rng.randint(0, 4)
        if shape == 0:      # a store list
            line = "%s %s %s %s %s" % (rng.choice(SIGNAGE), place(), place(), place(),
                                       rng.choice(SIGNAGE))
        elif shape == 1:    # a shipping label
            line = "%s %s %d %s %s %s" % (rng.choice(SIGNAGE), rng.choice(SIGNAGE),
                                          rng.randint(1, 9999), rng.choice(STREETS), place(),
                                          place())
        elif shape == 2:    # a depot sign
            line = "%s %s %s %s %s" % (place(), rng.choice(SIGNAGE), rng.choice(SIGNAGE),
                                       rng.choice(SIGNAGE), place())
        elif shape == 3:    # a route board
            line = "%s %d %s %s %s %s" % (rng.choice(STREETS), rng.randint(1, 99),
                                          place(), place(), place(), rng.choice(SIGNAGE))
        else:               # a terminal board
            line = "%s %s %s %s %s %s" % (rng.choice(SIGNAGE), place(), place(),
                                          rng.choice(SIGNAGE), place(), rng.choice(SIGNAGE))
        out.append((line, name_unguarded(line)))
    return out


BARS = [
    ("none (no bar at all)", 0, 0, False),
    ("diacritics only", 0, 0, True),
    ("word >= 5 only", 0, 5, True),
    ("stopword >= 3 only", 3, 0, True),
    ("stopword >= 4 only", 4, 0, True),
    ("stopword >= 5 only", 5, 0, True),
    ("stopword >= 3 AND word >= 5", 3, 5, True),
    ("stopword >= 4 AND word >= 5", 4, 5, True),
    ("stopword >= 5 AND word >= 5", 5, 5, True),
    ("stopword >= 4 AND word >= 6", 4, 6, True),
    ("stopword >= 4 AND word >= 7", 4, 7, True),
    ("stopword >= 4 AND word >= 5, no diacritic escape", 4, 5, False),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", help="write the table and corpus sizes as JSON")
    ap.add_argument("--acronym-lines", type=int, default=ACRONYM_LINES)
    ap.add_argument("--address-lines", type=int, default=ADDRESS_LINES)
    args = ap.parse_args()

    rng = random.Random(SEED)
    prose = build_prose(massive_rows())
    acr = build_acronyms(rng, args.acronym_lines)
    adr = build_addresses(rng, args.address_lines)
    total = len(prose)
    print("dataset %s @ %s   seed %d" % (DATASET, REVISION, SEED))
    print("pools: %d acronyms, %d place names, %d street types, %d signage words"
          % (len(ACRONYMS), len(PLACES), len(STREETS), len(SIGNAGE)))
    print("corpora: prose %d (upper-cased MASSIVE rows named as their own locale), "
          "acronym %d lines, address %d lines\n" % (total, len(acr), len(adr)))
    fmt = "%-48s %8s %8s %11s %11s"
    print(fmt % ("rule", "kept", "pct", "acronym FP", "address FP"))
    table = []
    for label, sw, wd, dia in BARS:
        kept = sum(1 for _, n in prose if accepts(n, sw, wd, dia))
        afp = sum(1 for _, n in acr if n is not None and accepts(n, sw, wd, dia))
        dfp = sum(1 for _, n in adr if n is not None and accepts(n, sw, wd, dia))
        print(fmt % (label, kept, "%.2f%%" % (100.0 * kept / total), afp, dfp))
        table.append({"rule": label, "stopword_bar": sw, "word_bar": wd, "diacritic_escape": dia,
                      "prose_kept": kept, "prose_pct": round(100.0 * kept / total, 2),
                      "acronym_fp": afp, "address_fp": dfp})
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"dataset": DATASET, "revision": REVISION, "seed": SEED,
             "pools": {"acronyms": len(ACRONYMS), "places": len(PLACES),
                       "streets": len(STREETS), "signage": len(SIGNAGE)},
             "corpora": {"prose": total, "acronym": len(acr), "address": len(adr)},
             "table": table}, indent=2) + "\n", encoding="utf-8")
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
