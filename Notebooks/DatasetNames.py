dataset_class_name_dict = {
        "DermaMNIST-2": {
        '0': 'actinic keratoses and intraepithelial carcinoma',
        '1': 'basal cell carcinoma',
        # '2': 'benign keratosis-like lesions',
        # '3': 'dermatofibroma',
        # '4': 'melanoma',
        # '5': 'melanocytic nevi',
        # '6': 'vascular lesions'
    },

    "DermaMNIST-4A": {
        # '0': 'actinic keratoses and intraepithelial carcinoma',
        # '1': 'basal cell carcinoma',
        # '2': 'benign keratosis-like lesions',
        '3': 'dermatofibroma',
        '4': 'melanoma',
        '5': 'melanocytic nevi',
        '6': 'vascular lesions'
    },
        "DermaMNIST-4": {
        '0': 'actinic keratoses and intraepithelial carcinoma',
        '1': 'basal cell carcinoma',
        '2': 'benign keratosis-like lesions',
        '3': 'dermatofibroma',
        # '4': 'melanoma',
        # '5': 'melanocytic nevi',
        # '6': 'vascular lesions'
    },

    "DermaMNIST": {
        '0': 'actinic keratoses and intraepithelial carcinoma',
        '1': 'basal cell carcinoma',
        '2': 'benign keratosis-like lesions',
        '3': 'dermatofibroma',
        '4': 'melanoma',
        '5': 'melanocytic nevi',
        '6': 'vascular lesions'
    },

    "OctMNIST-2A": {
        # "0": "choroidal neovascularization",
        # "1": "diabetic macular edema",
        "2": "drusen",
        "3": "normal",
    },

    "OctMNIST-2": {
        "0": "choroidal neovascularization",
        "1": "diabetic macular edema",
        # "2": "drusen",
        # "3": "normal",
    },

    "OctMNIST": {
        "0": "choroidal neovascularization",
        "1": "diabetic macular edema",
        "2": "drusen",
        "3": "normal",
    },

    'PneumoniaMNIST': {
        "0": "normal", 
        "1": "pneumonia"
        },

     "BloodMNIST": {
            "0": "basophil",
            "1": "eosinophil",
            "2": "erythroblast",
            "3": "immature granulocytes",
            "4": "lymphocyte",
            "5": "monocyte",
            "6": "neutrophil",
            "7": "platelet",
        },

    "RetinaMNIST": {
        "0": "retinal artery",
        "1": "retinal vein",
        "2": "normal"
    },

    "PathMNIST": {
            "0": "adipose",
            "1": "background",
            "2": "debris",
            "3": "lymphocytes",
            "4": "mucus",
            "5": "smooth muscle",
            "6": "normal colon mucosa",
            "7": "cancer-associated stroma",
            "8": "colorectal adenocarcinoma epithelium",
        },

        "OrganCMNIST": {
            "0": "bladder",
            "1": "femur-left",
            "2": "femur-right",
            "3": "heart",
            "4": "kidney-left",
            "5": "kidney-right",
            "6": "liver",
            "7": "lung-left",
            "8": "lung-right",
            "9": "pancreas",
            "10": "spleen",
        }
}

def get_dataset_class_name(dataset_name: str) -> dict:
    return dataset_class_name_dict[dataset_name]