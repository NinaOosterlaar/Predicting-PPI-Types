import jellyfish


def levenshtein_distance(s1, s2):
    """Return the Levenshtein distance between two strings.
        s1, s2: strings
        return the Levenshtein distance"""
    return jellyfish.levenshtein_distance(s1, s2)

def damereu_levenshtein_distance(s1, s2):
    """Return the Damerau-Levenshtein distance between two strings.
        s1, s2: strings
        return the Damerau-Levenshtein distance"""
    return jellyfish.damerau_levenshtein_distance(s1, s2)

