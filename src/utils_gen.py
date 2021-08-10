def get_canonical_type(metadata_type, type_synonyms):
    converted_type = None
    for canonical_type in type_synonyms:
        if metadata_type in type_synonyms[canonical_type]:
            converted_type = canonical_type.split("-")[0]
    return converted_type


def map_type(ptype_type, type_mapping):
    return type_mapping[ptype_type]
