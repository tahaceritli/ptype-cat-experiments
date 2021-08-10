from enum import Enum

TYPE_SYNONYMS = {
    "categorical": [
        "binary",
        "boolean",
        "boolean-valued",
        "logical",
        "categorical",
        "discrete",
        "enum",
        "enumerated",
        "list",
        "list/logical",
        "nominal",
        "ordinal",
    ],
    "date": ["date", "time", "date/time"],
    "numeric": [
        "continuous",
        "float",
        "real",
        "numeric real",
        "percentage",
        "int",
        "integer",
        "positive integer",
        "number",
        "numeric",
        "numerical",
    ],
    "string": ["char", "character", "string", "text"],
}

TYPE_MAPPING = {
    "boolean": "categorical",
    "date": "date",
    "float": "numeric",
    "integer": "numeric",
    "string": "string",
}

BOT_TYPES_MAPPING = {
    "bool": "categorical",
    "cat_int": "categorical",
    "cat_int64": "categorical",
    "cat_string": "categorical",
    "date": "date",
    "int64": "integer",
    "float64": "float",
    "string": "string",
}


ADDITIONAL_FEATURES = {
    "u_ratio": 4,
    "u_ratio_clean": 5,
    "U": 6,
    "U_clean": 7,
    "N": 8,
    "N_clean": 9,
}


class FEATURE(Enum):
    DATE = 0
    FLOAT = 1
    INTEGER = 2
    STRING = 3
    U_RATIO = 4
    U_RATIO_CLEAN = 5
    U = 6
    U_CLEAN = 7
    N = 8
    N_CLEAN = 9


CANONICAL_TYPES_SORTED = ["categorical", "date", "numeric", "string"]
PTYPE_TYPES_SORTED = ["date", "float", "integer", "string"]
DATA2PANDAS_TYPES_SORTED = ["categorical", "date", "float", "integer", "string"]
LR_TYPES_SORTED = ["categorical", "integer", "string"]
ANNOTATION_PATH = "../../annotations/annotations-fixed.npy"

PREDICTION_PATH = "data/type_predictions.json"
POSTERIOR_PATH = "data/type_posteriors.npy"
FEATURE_PATH = "data/additional_features.npy"

DATA_ROOT = "../sequence_classification/inputs/cv/"
OUTPUT_PATH = "outputs/"

PLOT_COLORS = {
    "boolean": "r",
    "categorical": "r",
    "date": "y",
    "float": "darkblue",
    "integer": "lightblue",
    "numeric": "blue",
    "string": "g",
}
FOLD = 5

method_name_abbs = {
    "bot": "Bot",
    "ptype": "ptype",
    "LogisticRegression": "LR",
    "DecisionTree": "DT",
    "keyword-search": "KS",
    "bi-lstm": "Bi-LSTM",
    "bi-lstm-attention": "Bi-LSTM-Attention",
    "roberta": "RoBERTa",
    "DT+LR": "DT+LR",
    "DT+LR*": "DT+LR*",
    "Bot+DT+LR": "Bot+DT+LR",
}

MAP_OUTPUTS = {
    "bool": "categorical",
    "cat_int": "categorical",
    "cat_int64": "categorical",
    "cat_string": "categorical",
    "date": "date",
    "int64": "numeric",
    "float64": "numeric",
    "string": "string",
}

DATASETS = {
    "Australian": ("utf-8", "infer"),
    "KDDCup99": ("utf-8", "infer"),
    "KEGGMetabolicReactionNetwork": ("utf-8", "infer"),
    "MagicTelescope": ("utf-8", "infer"),
    "SPECT": ("utf-8", "infer"),
    "abalone": ("ISO-8859-1", "infer"),
    "ada_prior": ("utf-8", "infer"),
    "anneal": ("utf-8", "infer"),
    "atari-head": ("utf-8", "infer"),
    "autoMpg": ("utf-8", "infer"),
    "autos": ("utf-8", "infer"),
    "bank-marketing": ("utf-8", "infer"),
    "banknote-authentication": ("utf-8", "infer"),
    "cleanEHR": ("utf-8", "infer"),
    "cmc": ("utf-8", "infer"),
    "cylinder-bands": ("utf-8", "infer"),
    "eucalyptus": ("utf-8", "infer"),
    "jm1": ("utf-8", "infer"),
    "sick": ("utf-8", "infer"),
    "kr-vs-kp": ("utf-8", "infer"),
    "geoplaces2": ("utf-8", "infer"),
    "nomao": ("utf-8", "infer"),
    "ozone-level-8hr": ("utf-8", "infer"),
    "nyc-home-parks": ("utf-8", "infer"),
    "rodents": ("utf-8", "infer"),
    "wholesale-customers": ("utf-8", "infer"),
    "squash-stored": ("utf-8", "infer"),
    "ringnorm": ("utf-8", "infer"),
    "wall-robot-navigation": ("utf-8", "infer"),
    "electricity_prices_ICON": ("utf-8", "infer"),
    "ldpa": ("utf-8", "infer"),
    "poker-hand": ("utf-8", "infer"),
    "page-blocks": ("utf-8", "infer"),
    "thoracic-surgery": ("utf-8", "infer"),
    "white-clover": ("utf-8", "infer"),
    "pasture": ("utf-8", "infer"),
    "echocardiogram-uci": ("utf-8", "infer"),
    "zoo": ("utf-8", "infer"),
    "young-people-survey": ("utf-8", "infer"),
    "student-alcohol-consumption": ("utf-8", "infer"),
    "womens-clothing-reviews": ("utf-8", "infer"),
    "consolidation-centres": ("utf-8", "infer"),
    "wikipedia-adventure": ("utf-8", "infer"),
    "river-ice": ("utf-8", "infer"),
    "grub-damage": ("utf-8", "infer"),
    "tae": ("utf-8", "infer"),
    "kick": ("utf-8", "infer"),
    "vinnie": ("utf-8", "infer"),
    "slump": ("utf-8", "infer"),
    "Click_prediction_small": ("utf-8", "infer"),
    "backache": ("utf-8", "infer"),
    "volcanoes-b6": ("utf-8", "infer"),
    "volcanoes-d4": ("utf-8", "infer"),
    "molecular-biology_promoters": ("utf-8", "infer"),
    "rabe_176": ("utf-8", "infer"),
    "analcatdata_broadwaymult": ("utf-8", "infer"),
    "colleges_aaup": ("utf-8", "infer"),
    "iris": ("utf-8", "infer"),
    "jungle_chess_2pcs_endgame_lion_elephant": ("utf-8", "infer"),
    "pharynx": ("utf-8", "infer"),
    "chscase_geyser1": ("utf-8", "infer"),
    "fruitfly": ("utf-8", "infer"),
    "fri_c4_250_25": ("utf-8", "infer"),
    "ParkinsonSpeechDatasetwithMultipleTypesofSoundRecordings": ("utf-8", "infer"),
    "sleuth_case1202": ("utf-8", "infer"),
    "philippine": ("utf-8", "infer"),
    "covertype": ("utf-8", "infer"),
    "forest_fires": ("utf-8", "infer"),
    "Diabetes(scikit-learn)": ("utf-8", "infer"),
    "fri_c2_1000_50": ("utf-8", "infer"),
    "analcatdata_homerun": ("utf-8", "infer"),
    "wdbc": ("utf-8", "infer"),
    "Satellite": ("utf-8", "infer"),
    "MIP-2016-PAR10-classification": ("utf-8", "infer"),
    "usp05": ("utf-8", "infer"),
    "Midwest_Survey_nominal": ("utf-8", "infer"),
    "Premier_League_odds_and_prob": ("utf-8", "infer"),
    "mofn-3-7-10": ("utf-8", "infer"),
    "jannis": ("utf-8", "infer"),
    "emotions": ("utf-8", "infer"),
    "hypothyroid": ("utf-8", "infer"),
    "image": ("utf-8", "infer"),
    "LED(50000)": ("utf-8", "infer"),
    "mc1": ("utf-8", "infer"),
    "scene": ("utf-8", "infer"),
    "Weather": ("utf-8", "infer")
}

EPS = 1e-10
