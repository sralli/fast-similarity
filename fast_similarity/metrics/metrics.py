from scipy import spatial
import math



def cosine_distance(a, b):
    cosine_distance = float(spatial.distance.cosine(a, b))
    return cosine_distance


def cosine_similarity(a, b):
    cosine_similarity = 1 - cosine_distance(a, b)
    return cosine_similarity


def angular_distance(a, b):
    cosine_similarity = cosine_similarity(a, b)
    angular_distance = math.acos(cosine_similarity) / math.pi
    return angular_distance


def angular_similarity(a, b):
    angular_similarity = 1 - angular_distance(a, b)
    return angular_similarity
