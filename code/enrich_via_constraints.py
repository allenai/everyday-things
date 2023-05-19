from relations import *
####
# OUTPUT FILES
#####
enrich_logfile = open("enriched_mms/enrich_log.txt", "w")
enrich_cnts_logfile = open("enriched_mms/enrich_cnts_log.txt", "w")
added_relations_logfile = open("enriched_mms/added_relations_log.tsv", "w")

mini_ET_dataset = open("enriched_mms/full-ET-dataset.tsv", "w")


####
# For enriching annotations based on rules
#####

def get_original_relations(relations, verbose = False):
    '''
    Input: relations is a list of lists, where inner lists are of length 3 
            like ['comb', 'above', 'eye']
    Output: original relations, list of tuples that contain the triplets and labels 
            like (('comb', 'above', 'eye'), True)
    '''
    original_relations = []
    orig_cnt = 0
    for p1, rln, p2 in relations:
        original_relation = ((p1, rln, p2), True)
        original_relations.append(original_relation)
        if verbose:
            print("Original annotated:", original_relation)
        enrich_logfile.write("Recorded original annotated: " + str(original_relation) + "\n")
        orig_cnt += 1
        
    assert len(relations) == len(original_relations)
    enrich_cnts_logfile.write("Original cnt: " + str(orig_cnt) + "\n")
    return original_relations


def enrich_using_contraints123(enriched_relations, verbose = False):
    '''
    Input: original relations, list of tuples that contain the triplets and labels 
            like (('comb', 'above', 'eye'), True)
    Output: enriched relations, list of tuples that contain the triplets and labels also
            like (('comb', 'above', 'eye'), True)
    '''
    further_enriched_relations = []
    add_cnt = 0
    for triplet, label in enriched_relations:
        p1, rln, p2 = triplet
        original_relation = (triplet, label)
        if verbose:
            print("Trying to enrich for original annotated:", original_relation)
        enrich_logfile.write("Trying to enrich for original annotated: " + str(original_relation) + "\n")
        

        if rln in mutual_relations:

            #[CONSTRAINT TYPE1] Symmetric (mutual_relations): x rln y <-> y rln x 
            # x rln y -> y rln x, y rln x -> x rln y
            # they must be same judgement
            # if either one is annotated (therefore is True), add the other (also must be True)
            added_relation = ((p2, rln, p1), label)
            if added_relation not in enriched_relations and added_relation not in further_enriched_relations :
                further_enriched_relations.append(added_relation)
                if verbose:
                    print("Constraint 1: x rln y <-> y rln x", added_relation)
                enrich_logfile.write("Constraint 1: x rln y <-> y rln x " + str(added_relation) + "\n")
                added_relations_logfile.write("\t".join(["Constraint 1: x rln y <-> y rln x", str(original_relation), str(added_relation)]) + "\n")
                add_cnt += 1

        elif rln in relation_opp_dict:
            #[CONSTRAINT TYPE2] Asymmetric (relation_opp_dict): NOT(x rln y) OR NOT (y rln x)
            # x rln y -> NOT(y rln x)
            # they cannot both be True
            # for each x rln y annotated (therefore is True), mark y rln x as False
            if label:
                added_relation = ((p2, rln, p1), False)
                if added_relation not in enriched_relations and added_relation not in further_enriched_relations :
                    further_enriched_relations.append(added_relation)
                    if verbose:
                        print("Constraint 2: x rln y -> NOT(y rln x)", added_relation)
                    enrich_logfile.write("Constraint 2: x rln y -> NOT(y rln x) " + str(added_relation) + "\n")
                    added_relations_logfile.write("\t".join(["Constraint 2: x rln y -> NOT(y rln x)", str(original_relation), str(added_relation)]) + "\n")
                    add_cnt += 1

            # [CONSTRAINT TYPE3] Inverse (relation_opp_dict):  x rln y <-> y inverse(rln) x
            # they must be same judgement
            # if either one is annotated (therefore is True), add the other (also must be True)
            added_relation = ((p2, relation_opp_dict[rln], p1), label)
            if added_relation not in enriched_relations and added_relation not in further_enriched_relations :
                further_enriched_relations.append(added_relation)
                if verbose:
                    print("Constraint 3: x rln y <-> y inverse(rln) x", added_relation)
                enrich_logfile.write("Constraint 3: x rln y <-> y inverse(rln) x " + str(added_relation) + "\n")
                added_relations_logfile.write("\t".join(["Constraint 3: x rln y <-> y inverse(rln) x", str(original_relation), str(added_relation)]) + "\n")
                add_cnt += 1

    enrich_cnts_logfile.write("Additions from constraints 1-3: " + str(add_cnt) + "\n")
    assert add_cnt == len(further_enriched_relations)
    return add_cnt, enriched_relations + further_enriched_relations


def enrich_using_contraint4(enriched_relations, verbose = False):
    # [CONSTRAINT TYPE4] Transitivity (transitive_relations): x rln y ^ y rln z -> x rln z 
    # if x rln y AND y rln z are both annotated (therefore are True), add x rln z (also must be True)
    transitive_relations = [x for x in all_relations_lst if x not in functional_dep_plus_directly_connected]

    # (x, rln) as key
    x_rln_propositions = {}
    for triplet, label in enriched_relations:
        if label == False:
            continue
        x_rln = (triplet[0], triplet[1])
        if x_rln in x_rln_propositions:
            assert triplet not in x_rln_propositions[x_rln]
            x_rln_propositions[x_rln] += [triplet]
        else:
            x_rln_propositions[x_rln] = [triplet]

    # look at all True relations
    further_enriched_relations = []
    transitivity_adds = 0
    for x_rln in x_rln_propositions:
        x_rln_relations = x_rln_propositions[x_rln]
        for p1, rln, p2 in x_rln_relations: # for each x rln y
            if rln in transitive_relations and (p2, rln) in x_rln_propositions: # if it is a transitive relation
                for p1_2, rln_2, p2_2 in x_rln_propositions[(p2, rln)]: # y rln z1, y rln z2 ...
                    # exclude the case where x = z                                      
                    if p1 == p2_2:
                        continue
                    assert p2 == p1_2 and rln == rln_2
                    # x rln y (A) ^ y rln z (B) True
                    added_relation = ((p1, rln, p2_2), True)
                    if added_relation not in enriched_relations and added_relation not in further_enriched_relations :
                        further_enriched_relations.append(added_relation)
                        if verbose:
                            print("Constraint 4:", (p1, rln, p2), (p1_2, rln_2, p2_2), "->", (p1, rln, p2_2))
                        enrich_logfile.write("Constraint 4:" + str((p1, rln, p2)) + str((p1_2, rln_2, p2_2)) + "->" + str((p1, rln, p2_2)) + "\n")
                        added_relations_logfile.write("\t".join(["Constraint 4 (True antecedent)", str((p1, rln, p2)) + str((p1_2, rln_2, p2_2)), str(added_relation)]) + "\n")
                        transitivity_adds += 1

    enrich_cnts_logfile.write("Additions from constraint 4: " + str(transitivity_adds) + "\n")
    assert transitivity_adds == len(further_enriched_relations)
    return transitivity_adds, enriched_relations + further_enriched_relations


def get_all_relations_with_labels(relations, verbose = False):
    enriched_relations0 = get_original_relations(relations, verbose)
    additions = -1
    round_cnt = 0
    total_additions = 0

    while additions != 0:
        round_cnt += 1
        enrich_cnts_logfile.write("Round " + str(round_cnt) + " ...\n")
        added_relations_logfile.write("Round " + str(round_cnt) + " ...\n")
        
        adds1, enriched_relations1 = enrich_using_contraints123(enriched_relations0, verbose)
        adds2, enriched_relations2 = enrich_using_contraint4(enriched_relations1, verbose)

        additions = adds1 + adds2
        total_additions += additions
        enriched_relations0 = enriched_relations2


    assert len(enriched_relations2) == total_additions + len(relations)
    enrich_cnts_logfile.write("Total relations: {} ({} + {})".format(len(enriched_relations2), len(relations), total_additions) + "\n\n")
    return enriched_relations2
