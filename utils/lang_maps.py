import os

ROOT: str = os.path.join(
    '/home', 'drd92', 'mozilla-asr-challenge', 'mcv-sps-st-09-2025'
)

HR_ROOT: str = os.path.join(
    '/home', 'drd92', 'mozilla-asr-challenge', 'cv-corpus-23.0-2025-09-05'
)

LANGUAGES: set[str] = {
    'bxk',   # Bukusu (Bantu, Zone J)
    'cgg',   # Chiga (Bantu, Zone J)
    'koo',   # Konzo (Bantu, Zone J)
    'lke',   # Kenyi (Bantu, Zone J)
    'ruc',   # Ruuli (Bantu, Zone J)
    'ttj',   # Rutoro (Bantu, Zone J)
    'rwm',   # Amba (Bantu, Zone D)

    'kcn',   # Nubi (AA, Semitic creole)

    'led',   # Lendu (NS, Eastern Sudanic)
    'lth',   # Thur (NS, Eastern Sudanic)
    'ukv',   # Kuku (NS, Eastern Sudanic)

    'hch',   # Wixárika (Uto-Aztecan, Corachol)

    'meh',   # SW Tlaxiaco Mixtec (Oto-Manguean, Mixtecan)
    'mmc',   # Michoacán Mazahua (Oto-Manguean, Otomian)

    'top',   # Papantla Totonac (Totonacan)

    'tob',   # Toba Qom (Guaicuruan)

    'aln',   # Gheg Albanian (IE, Albanian)
    'el-CY', # Cypriot Greek (IE, Greek)
    'sco',   # Scots (IE, Germanic)

    'bew',   # Betawi (Austronesian, Malayo-Polynesian creole)
    'pne',   # Western Penan (Austronesian, Malayo-Polynesian)
}

HIGH_RESOURCE: set[str] = {
    'lg',    # Luganda (Bantu, Zone J)

    'mt',    # Maltese (AA, Latin script)

    'luo',   # Dholuo (NS, Eastern Sudanic)
    'kln',   # Kalenjin (NS, Eastern Sudanic)

    'ncx',   # Central Puebla Nahuatl (Uto-Aztecan, Southern, Aztecan)
    'nhi',   # Tetelancingo Nahuatl (Uto-Aztecan, Southern, Aztecan)
    'nlv',   # Orizaba Nahuatl (Uto-Aztecan, Southern, Aztecan)
    'yaq',   # Yaqui (Uto-Aztecan, Southern)
    'tar',   # Central Tarahumara (Uto-Aztecan, Southern)
    'var',   # Huarijio (Uto-Aztecan, Southern)

    'cut',   # Teutila Cuicatec (Oto-Manguaean, Mixtecan)
    'cux',   # Tepeuixila Cuicatec (Oto-Manguean, Mixtecan)
    'mau',   # Huautla Mazatec (Oto-Manguean, Popolocan)

    'qup',   # Southern Pastaza Quechua
    'quy',   # Quechua Chanka
    'qxa',   # Quechua Chiquián
    'qux',   # Quechua Yauyos
    'qvl',   # Quechua Cajatambo
    'qxu',   # Quechua Arequipa-La Unión
    'qxw',   # Quechua Jauja Wanka
    'qur',   # Quechua Yanahuanca
    'qus',   # Quechua Santiago del Estero
    'qva',   # Quechua Ambo-Pasco
    'qwa',   # Quechua Corongo Ancash
    'qws',   # Quechua Sihuas Ancash
    'qxt',   # Quechua Pasco Santa Ana de Tusi

    'sq',    # Albanian (IE, Albanian)
    'el',    # Greek (IE, Greek)
    'ur',    # Urdu (IE, Indo-Aryan, Arabic script)

    'ab',    # Abkhaz (NW Caucasian)

    'id',    # Indonesian (Austronesian, Malayo-Polynesian)
    'ms',    # Malay (Austronesian, Malayo-Polynesian)
    'msi',   # Sabah Malay (Austronesian, Malayo-Polynesian creole)
}

TEST_ONLY: set[str] = {
    'bas', # Basaa (Bantu, Zone A)
    
    'qxp', # Puno Quechua (Quechuan)

    'ush', # Ushojo (IE, Indo-Aryan)

    'ady', # Adyghe (NW Caucasian)
    'kbd', # Kabardian (NW Caucasian)
}

ALL_TARGETS = LANGUAGES.union(TEST_ONLY)

HR_MAP: dict[str, list[str]] = {
    'bxk': ['lg'],          # Bukusu (Bantu, Zone J)
    'cgg': ['lg'],          # Chiga (Bantu, Zone J)
    'koo': ['lg'],          # Konzo (Bantu, Zone J)
    'lke': ['lg'],          # Kenyi (Bantu, Zone J)
    'ruc': ['lg'],          # Ruuli (Bantu, Zone J)
    'ttj': ['lg'],          # Rutoro (Bantu, Zone J)
    'rwm': ['lg'],          # Amba (Bantu, Zone D)
    'bas': ['lg'],          # Basaa (Bantu, Zone A)

    'kcn': ['mt'],          # Nubi (AA, Semitic creole)

    'led': ['luo', 'kln'],  # Lendu (NS, Eastern Sudanic)
    'lth': ['luo', 'kln'],  # Thur (NS, Eastern Sudanic)
    'ukv': ['luo', 'kln'],  # Kuku (NS, Eastern Sudanic)

    'hch': ['ncx', 'nhi', 'nlv', 'yaq', 'tar', 'var'],   # Wixárika (Uto-Aztecan, Corachol)

    'meh': ['cut', 'cux', 'mau'], # SW Tlaxiaco Mixtec (Oto-Manguean, Mixtecan)
    'mmc': ['cut', 'cux', 'mau'], # Michoacán Mazahua (Oto-Manguean, Otomian)

    'top': [], # Papantla Totonac (Totonacan)

    'qxp': [ # Puno Quechua (Quechuan)
        'qup', 'quy', 'qxa', 'qux', 'qvl', 'qxu', 'qxw', 'qur', 'qus', 'qva', 'qwa', 'qws', 'qxt'
    ],

    'tob': [],       # Toba Qom (Guaicuruan)

    'aln': ['sq'],   # Gheg Albanian (IE, Albanian)
    'el-CY': ['el'], # Cypriot Greek (IE, Greek)
    'sco': [],       # Scots (IE, Germanic)
    'ush': ['ur'],   # Ushojo (IE, Indo-Aryan, Arabic script)

    'ady': ['ab'],   # Adyghe (NW Caucasian)
    'kbd': ['ab'],   # Kabardian (NW Caucasian)

    'bew': ['id', 'ms'], # Betawi (Austronesian, Malayo-Polynesian creole)
    'pne': ['id', 'ms'], # Western Penan (Austronesian, Malayo-Polynesian)
}

def validate():
    for key, values in HR_MAP.items():
        if key not in LANGUAGES:
            assert key in TEST_ONLY
            assert os.path.exists(os.path.join(HR_ROOT, key)), key
        if key not in TEST_ONLY:
            assert key in LANGUAGES
            assert os.path.exists(os.path.join(ROOT, f'sps-corpus-1.0-2025-09-05-{key}')), key
        
        for value in values:
            assert value in HIGH_RESOURCE, value
            assert os.path.exists(os.path.join(HR_ROOT, value)), value