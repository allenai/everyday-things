from relations import *
####
# For getting % of consistency violations
#####

def get_symmetric_constraint_violations(result_dict_props, verbose=True):
    #[CONSTRAINT TYPE1] Symmetric (mutual_relations): x rln y <-> y rln x [A <-> B]
    # they must be same judgement
    symrelationship_instances_dict = {}
    for triplet in result_dict_props:
        p1, relation, p2 = triplet
        if relation in mutual_relations:
            if (p2, relation, p1) in symrelationship_instances_dict:
                symrelationship_instances_dict[(p2, relation, p1)] += [triplet]
            else:
                symrelationship_instances_dict[triplet] = [triplet]

    ###
    # Denominator: # applicable constraints
    #             number of times symmetric relationship occurred between pairs of objects 
    #             take x rln y and y rln x as *one occurence* of the relationship but *2 instances*
    # Numerator: # constraint violations
    #             number of times symmetric relationship is not symmetric
    #             e.g. x rln y is considered True but not y rln x 
    ###
    denominator = 0
    numerator = 0
    for symrelationship in symrelationship_instances_dict:
        denominator += 1
        # count violations -- not symmetric
        instances_cnt = len(symrelationship_instances_dict[symrelationship])
        if instances_cnt != 2:
            assert instances_cnt == 1
            numerator += 1
            if verbose:
                print("Mutual relation not symmetric!")
                print(symrelationship_instances_dict[symrelationship])


    print("[CONSTRAINT TYPE1]")
    if denominator == 0:
        print("Consistency violation: {numerator}/{denominator}".format(numerator=numerator,denominator=denominator))
        print("Not applicable for given mental model!")
    else:
        print("Consistency violation: {numerator}/{denominator} ({percentage}))".format(numerator=numerator,\
            denominator=denominator, percentage=round(numerator/denominator,2)))


    return numerator, denominator

def get_asymmetric_constraint_violations(result_dict_props, verbose=True):
    #[CONSTRAINT TYPE2] Asymmetric (relation_opp_dict): NOT(x rln y) OR NOT (y rln x) [-A OR -B]
    # they cannot both be True
    asymrelationship_instances_dict = {}
    for triplet in result_dict_props:
        p1, relation, p2 = triplet
        if relation in relation_opp_dict:
            if (p2, relation, p1) in asymrelationship_instances_dict:
                asymrelationship_instances_dict[(p2, relation, p1)] += [triplet]
            else:
                asymrelationship_instances_dict[triplet] = [triplet]

    ###
    # Denominator: # applicable constraints
    #             number of times asymmetric relationship occured between pairs of objects 
    #             take x rln y and y rln x as *one occurence* of the relationship but *2 instances*
    # Numerator: # constraint violations
    #             number of times asymmetric relationship is symmetric
    #             i.e. x rln y is considered True AND y rln x also True
    ###
    denominator = 0
    numerator = 0
    for asymrelationship in asymrelationship_instances_dict:
        denominator += 1
        # count violations -- asymmetric relation symmetric
        instances_cnt = len(asymrelationship_instances_dict[asymrelationship])
        if instances_cnt != 1:
            assert instances_cnt == 2
            numerator += 1
            if verbose:
                print("Asymmetric relation symmetric!")
                print(asymrelationship_instances_dict[asymrelationship])


    print("[CONSTRAINT TYPE2]")
    if denominator == 0:
        print("Consistency violation: {numerator}/{denominator}".format(numerator=numerator,denominator=denominator))
        print("Not applicable for given mental model!")
    else:
        print("Consistency violation: {numerator}/{denominator} ({percentage}))".format(numerator=numerator,\
            denominator=denominator, percentage=round(numerator/denominator,2)))


    return numerator, denominator

def get_inverse_constraint_violations(result_dict_props, verbose=True):
    # [CONSTRAINT TYPE3] Inverse (relation_opp_dict):  x rln y (A) <-> y inverse(rln) x (B) [A <-> B]
    # they must be same judgement
    inverse_relationship_instances_dict = {}
    for triplet in result_dict_props:
        p1, relation, p2 = triplet
        if relation in relation_opp_dict:
            if (p2,  relation_opp_dict[relation], p1) in inverse_relationship_instances_dict:
                inverse_relationship_instances_dict[(p2,  relation_opp_dict[relation], p1)] += [triplet]
            else:
                inverse_relationship_instances_dict[triplet] = [triplet]

    ###
    # Denominator: # applicable constraints
    #             number of times inverse relationship occured between pairs of objects 
    #             take x rln y and y inverse(rln) x as *one occurence* of the relationship but *2 instances*
    # Numerator: # constraint violations
    #             number of times  x rln y,  y inverse(rln) x not have the same judgement
    ###
    denominator = 0
    numerator = 0
    for inverse_relationship in inverse_relationship_instances_dict:
        denominator += 1
        # count violations --   x rln y,  y inverse(rln) x not have the same judgement
        instances_cnt = len(inverse_relationship_instances_dict[inverse_relationship])
        if instances_cnt != 2:
            assert instances_cnt == 1
            numerator += 1
            if verbose:
                print("Inverse relation not interpreted as expected!")
                print(inverse_relationship_instances_dict[inverse_relationship])
    #         else:
    #             print("Inverse relation CORRECT!")
    #             print(inverse_relationship_instances_dict[inverse_relationship])


    print("[CONSTRAINT TYPE3]")
    if denominator == 0:
        print("Consistency violation: {numerator}/{denominator}".format(numerator=numerator,denominator=denominator))
        print("Not applicable for given mental model!")
    else:
        print("Consistency violation: {numerator}/{denominator} ({percentage}))".format(numerator=numerator,\
            denominator=denominator, percentage=round(numerator/denominator,2)))


    return numerator, denominator

def get_transitivity_constraint_violations(result_dict_props, verbose=True):
    # [CONSTRAINT TYPE4] Transitivity (transitive_relations): x rln y (A) ^ y rln z (B) -> x rln z (C) [A ^ B -> C]

    all_relations_lst = list(relation_opp_dict.keys()) + list(mutual_relations)
    transitive_relations = [x for x in all_relations_lst if x not in functional_dep_plus_directly_connected]
    
    # (x, rln) as key
    x_rln_propositions = {}
    for triplet in result_dict_props:
        x_rln = (triplet[0], triplet[1])
        if x_rln in x_rln_propositions:
            assert triplet not in x_rln_propositions[x_rln]
            x_rln_propositions[x_rln] += [triplet]
        else:
            x_rln_propositions[x_rln] = [triplet]
            
    ###
    # Denominator: # applicable constraints
    #             number of times x rln y (A) ^ y rln z (B) is True
    # Numerator: # constraint violations
    #             number of times x rln y (A) ^ y rln z (B) is True AND x rln z (C) is False
    ###
    denominator = 0
    numerator = 0

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
                    denominator += 1
                    # count violations --  x rln z (C) False
                    if (p1, rln, p2_2) not in  x_rln_relations: # x rln z1, x rln z2 ...
                        numerator += 1
                        if verbose:
                            print("Transitivity violated!")
                            print((p1, rln, p2), (p1_2, rln_2, p2_2), "!->", (p1, rln, p2_2))
    #                     else:
    #                         print((p1, rln, p2), (p1_2, rln_2, p2_2), "->", (p1, rln, p2_2), "[CORRECT!]")



    print("[CONSTRAINT TYPE4]")
    if denominator == 0:
        print("Consistency violation: {numerator}/{denominator}".format(numerator=numerator,denominator=denominator))
        print("Not applicable for given mental model!")
    else:
        print("Consistency violation: {numerator}/{denominator} ({percentage}))".format(numerator=numerator,\
            denominator=denominator, percentage=round(numerator/denominator,2)))


    return numerator, denominator

def get_all_constraint_violations(result_dict_props, verbose=False, max_sat_applied=False):
    n1, d1 = get_symmetric_constraint_violations(result_dict_props, verbose)
    n2, d2 = get_asymmetric_constraint_violations(result_dict_props, verbose)
    n3, d3 = get_inverse_constraint_violations(result_dict_props, verbose)
    n4, d4 = get_transitivity_constraint_violations(result_dict_props, verbose)
    
    n = n1 + n2 + n3 + n4
    d = d1 + d2 + d3 + d4
    
    if max_sat_applied:
        assert n == 0
    

    print("[===OVERALL==]")
    if d == 0:
        print("Consistency violation: {numerator}/{denominator}".format(numerator=n,denominator=d))
        print("Not applicable for given mental model!")
    else:
        print("Consistency violation: {numerator}/{denominator} ({percentage}))".format(numerator=n,\
            denominator=d, percentage=round(n/d,2)))
    if verbose:
        return n, d, ((n1, d1), (n2, d2), (n3, d3), (n4, d4))
    else:
        return n, d
