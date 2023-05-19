####
# DEFINED RELATIONS
#####

relation_opp_dict = { # 6 * 2 = 12
    "part of":"has part",
    "has part":"part of",
    "inside":"contains", # = encloses in Halo relations
    "contains":"inside",
    "in front of":"behind",
    "behind":"in front of",
    "above":"below",
    "below":"above",
    "surrounds":"surrounded by",
    "surrounded by":"surrounds",
    "requires":"required by", # functional dependencies
    "required by":"requires",
    }

functional_dep_plus_directly_connected = ["requires", "required by",\
                                          "part of", "has part",\
                                          "connects", "next to"]
mutual_relations = ("connects", "next to") # 2
all_relations_lst = list(relation_opp_dict.keys()) + list(mutual_relations)

