from relations import *
from utils import *
####
# Imports needed for Macaw-ImagineADevice-CSP-Viz-v5 pipeline
#####

import os
import pickle
import spacy
import math
import requests
from tqdm import tqdm
import itertools
import graphviz
from pysat.examples.rc2 import RC2, RC2Stratified
from pysat.formula import WCNF


####
# OUTPUT FILES
#####
macaw_getMM_logfile = open("saved_logs/pipeline_step1_macaw_getMM_log.txt", "w")
impose_contraints_logfile = open("saved_logs/pipeline_step2_impose_contraints_log.txt", "w")




####
# Macaw-ImagineADevice-CSP-Viz-v5 pipeline
#####

def ask_macaw(question, mc_list = [], mc_str =""):
    '''
    Input: question in the form of string
    Output: answer
    Use this function to get answer from macaw without probabilities (faster)
    '''
    if mc_list:
        mcoptions = " ".join(["(" + chr(i+65) + ") " + word for i, word in enumerate(mc_list)])
        response = requests.get('http://aristo-server1:8502/api', \
            params={"input":"Q: " + question + "\nM: " + mcoptions + "\nA"})
    elif mc_str:
        response = requests.get('http://aristo-server1:8502/api', \
            params={"input":"Q: " + question + "\nM: " + mc_str + "\nA"})
    else:
        response = requests.get('http://aristo-server1:8502/api', params={"input":"Q: " + question + "\nA"})
    #print (question, response.json()['output_slots_list'][0]['answer'], response.json()['output_slots_list'])
    #print (question, response.json()['output_slots_list'][0]['answer'])
    return response.json()['output_slots_list'][0]['answer']

def ask_macaw_conf(question, mc_list = ["yes", "no"], mc_str ="", get_all=False):
    '''
    input: question in the form of string
    output: answer
    default MC options is Yes/No
    Use this function to get answer from macaw with probabilities (slower queries)
    '''
    if mc_list:
        mcoptions = " ".join(["(" + chr(i+65) + ") " + word for i, word in enumerate(mc_list)])
        response = requests.get('http://aristo-server1:8502/api', \
            params={"input":"Q: " + question + "\nM: " + mcoptions + "\nX: " + mcoptions + "\nA"})
    elif mc_str:
        response = requests.get('http://aristo-server1:8502/api', \
            params={"input":"Q: " + question + "\nM: " + mc_str + "\nX: " + mc_str + "\nA"})

    if get_all:
         ans_score =  [(x['output_text'], x['output_prob']) for x in response.json()['explicit_outputs']]
    else:
        ans_score = [(x['output_text'], x['output_prob']) for x in response.json()['explicit_outputs'] if x['output_text'] == response.json()['output_slots_list'][0]['answer']][0]
    
    
    return ans_score

### Pipeline

nlp = spacy.load("en_core_web_sm")

### Step1: Prompt the model to get its MM of a device
def get_parts_perm(device, parts=[]):
    """
    Input: everyday thing, (optional) list of parts
    Output: permutation of list of parts
    """
    macaw_getMM_logfile.write("Everything processed: " + device + "\n")
    if not parts:
        qns_parts = "What are the parts of a/an {}?".format(device)
        ans_parts = ask_macaw(qns_parts)
        macaw_getMM_logfile.write("\t".join(["[===QA===] ", str(qns_parts), str(ans_parts)]) + "\n")
        parts = list(set([part.strip() for part in ans_parts.strip().split(",")]))
    macaw_getMM_logfile.write("List of parts: \n" + str(parts) + "\n")
    perm = list(itertools.permutations(parts, 2))
    macaw_getMM_logfile.write("Permutation of parts: \n" + str(perm) + "\n")
    
    return perm

def get_determiner(word):
    a_or_an = "a"
    vowels = ["a", "e", "i", "o", "u"]
    if word.startswith(tuple(vowels)):
        a_or_an = "an"
    return a_or_an

def is_plural(noun):
    """
    Input: the entity (string)
    Output: whether the entity is plural (True/False)
    """
    doc = nlp(noun)
    lemma = " ".join([token.lemma_ for token in doc])
    plural = True if (noun is not lemma and noun.endswith("s")) else False
    return plural

    
def triplet2statement(triplet):
    relation_singular_plural_dict = {"has part": "have part", \
                                    "contains": "contain", \
                                    "surrounds": "surround", \
                                    "requires": "require"
                                    }

    num = "plural" if is_plural(triplet[0]) else "singular"

    if triplet[1] == "connects":
        linking_phrase = "directly connected to"
    else:
        linking_phrase = triplet[1] # default is singular

    if linking_phrase in relation_singular_plural_dict:
        # special treatment of linking phrase for plural
        if num == "plural":
            linking_phrase = relation_singular_plural_dict[linking_phrase]
    else:
        # + is/are
        add = "are" if num == "plural" else "is"
        linking_phrase = add + " " + linking_phrase

    if "has part" == linking_phrase or "have part" == linking_phrase:
        phrase_p1, phrase_p2 = linking_phrase.split(" ")
        its_their = "their" if is_plural(triplet[0]) else "its"
        statement = " ".join([triplet[0], phrase_p1, "the", triplet[2], "as", its_their, phrase_p2])
    else:
        statement = " ".join([triplet[0], linking_phrase, "the", triplet[2]])
    return statement

def get_p_statement_and_p_neg_statement(device, statement):
    '''
    Output: direct query answer, p_statement, p_neg_statement
    '''
    probs_dict = {"T_given_statement": 0, "T_given_neg_statement": 0}

    mc_list = ("True", "False")
    #mcoptions = " ".join(["(" + chr(i+65) + ") " + option for i, option in enumerate(mc_list)])
    
    compiled_qns = "Judge whether this statement is true or false: In a/an {device}, {statement}.".format( \
                    device = device, statement=statement)
    ans_conf_all = ask_macaw_conf(compiled_qns, mc_list, get_all=True) # T_given_statement, F_given_statement
    ans = ans_conf_all[0][0] if ans_conf_all[0][1] > ans_conf_all[1][1] else ans_conf_all[1][0]
    macaw_getMM_logfile.write("\t".join(["[===QA===] ", compiled_qns, ans, str(ans_conf_all)]) + "\n")
    for entry in ans_conf_all:
        if entry[0] == "True":
            probs_dict["T_given_statement"] = entry[1]

    neg_compiled_qns = "Judge whether this statement is true or false: In a/an {device}, it is not the case that {statement}.".format( \
                    device = device, statement=statement)
    neg_ans_conf_all = ask_macaw_conf(neg_compiled_qns, mc_list, get_all=True) # T_given_neg_statement, F_given_neg_statement
    macaw_getMM_logfile.write("\t".join(["[===QA===] ", neg_compiled_qns, str(neg_ans_conf_all)]) + "\n")
    for entry in neg_ans_conf_all:
        if entry[0] == "True":
            probs_dict["T_given_neg_statement"] = entry[1]

    #return ans, probs_dict["T_given_statement"], probs_dict["T_given_neg_statement"]
    return ans, probs_dict["T_given_statement"], 1 - probs_dict["T_given_statement"]

def query_macaw_statements(device, perm):
    '''
    Input: everyday thing, list of tuples for permutation of list of parts
    Output: triplet_ans_conf_lst - contains [triplet,  ans, p_statement]
            neg_ans_conf_lst - list of p_neg_statement
    '''

    triplet_ans_conf_lst = [] # list of list
    neg_ans_conf_lst = [] # list
    for entry in perm:
        for rln in tqdm(all_relations_lst):
            
            triplet = (entry[0], rln, entry[1])
            statement = triplet2statement(triplet)
            
            ans, p_statement, p_neg_statement = get_p_statement_and_p_neg_statement(device, statement)
            triplet_ans_conf_lst.append([triplet,  ans, p_statement])
            neg_ans_conf_lst.append(p_neg_statement)
            
    return triplet_ans_conf_lst, neg_ans_conf_lst

def get_statements_that_macaw_believesT(triplet_ans_conf_lst):
    triplet_ans_conf_lst_true = []
    for triplet, ans, ans_conf in triplet_ans_conf_lst:
        if ans == "True":
            triplet_ans_conf_lst_true.append([triplet,  ans, ans_conf])
    return triplet_ans_conf_lst_true

# query macaw on "everyday thing"
def run_query_macaw_everyday_thing(device, parts):
    # get parts
    perm = get_parts_perm(device, parts)
    # get macaw judgment
    triplet_ans_conf_lst, neg_ans_conf_lst = query_macaw_statements(device, perm)
    triplet_ans_conf_lst_true = get_statements_that_macaw_believesT(triplet_ans_conf_lst)
    return triplet_ans_conf_lst, triplet_ans_conf_lst_true, neg_ans_conf_lst


### Step2: Impose constraints using maxsat
# def get_propositions_input(triplet_ans_conf_lst, neg_ans_conf_lst):
#     '''
#     1) Getting WCNF format... Processing propositions as believed by model...
#     We "reason" (impose 4 types of constraints) later based on the original statements,
#     but we also assert negations of these statements and 
#     constrain that orginal and negation cannot be true at the same time.
#     '''
#     impose_contraints_logfile.write("1) Getting WCNF format... Processing propositions as believed by model...\n")
#     #print(neg_ans_conf_lst)
#     wcnf_lines_ind = []
#     wcnf_lines_ind_negations = []
#     wcnf_lines_cons_TF = []
#     object_parts_propositions = {} #{('stem', 'leaves'): {('stem', 'part of', 'leaves'): {'conf': 0, 'idx': 1}
#     prop_idx = 0
#     true_cnt = 0
#     false_cnt = 0
#     for idx_triplet, (triplet, ans, ans_conf) in enumerate(triplet_ans_conf_lst):
#         part1, _, part2 = triplet
#         prediction_confidence = round(ans_conf * 100)
#         #neg_prediction_confidence = 100 - prediction_confidence
#         neg_prediction_confidence = round(neg_ans_conf_lst[idx_triplet] * 100)
#         assert ans == "True" or ans == "False"
#         if ans == "True":
#             true_cnt += 1
#         elif ans == "False":
#             false_cnt += 1
#             # neg_ans_conf

#         prop_idx += 1
#         if (part1, part2) in object_parts_propositions:
#             assert triplet not in object_parts_propositions[(part1, part2)]
#             object_parts_propositions[(part1, part2)][triplet] = {"conf": prediction_confidence, "idx": prop_idx}
#         else:
#             object_parts_propositions[(part1, part2)]= {triplet: {"conf": prediction_confidence, "idx": prop_idx}}

#         ###
#         # ===Individual beliefs===
#         # confidence from macaw, in format: <weight> <belief-id> 0
#         ###
#         # print("{} {} 0".format(prediction_confidence,prop_idx)) # 1-based idx
#         wcnf_lines_ind.append("{} {} 0".format(prediction_confidence,prop_idx))
#         # negation
#         prop_idx += 1
#         wcnf_lines_ind_negations.append("{} {} 0".format(neg_prediction_confidence,prop_idx))
#         wcnf_lines_cons_TF.append("10000 {} {} 0".format(-(prop_idx - 1), -prop_idx))

#     assert prop_idx == (true_cnt + false_cnt) * 2
    
#     return wcnf_lines_ind, wcnf_lines_ind_negations, wcnf_lines_cons_TF, object_parts_propositions
    

## WE TRY; WHAT HAPPENS IF WE REMOVE NEGATION?
def get_propositions_input(triplet_ans_conf_lst, neg_ans_conf_lst):
    '''
    1) Getting WCNF format... Processing propositions as believed by model...
    We "reason" (impose 4 types of constraints) later based on the original statements,
    but we also assert negations of these statements and 
    constrain that orginal and negation cannot be true at the same time.
    '''
    impose_contraints_logfile.write("1) Getting WCNF format... Processing propositions as believed by model...\n")
    #print(neg_ans_conf_lst)
    wcnf_lines_ind = []
    wcnf_lines_ind_negations = []
    wcnf_lines_cons_TF = []
    object_parts_propositions = {} #{('stem', 'leaves'): {('stem', 'part of', 'leaves'): {'conf': 0, 'idx': 1}
    prop_idx = 0
    true_cnt = 0
    false_cnt = 0
    for idx_triplet, (triplet, ans, ans_conf) in enumerate(triplet_ans_conf_lst):
        part1, _, part2 = triplet
        prediction_confidence = round(ans_conf * 100)
        #neg_prediction_confidence = 100 - prediction_confidence
        # neg_prediction_confidence = round(neg_ans_conf_lst[idx_triplet] * 100)
        assert ans == "True" or ans == "False"
        if ans == "True":
            true_cnt += 1
        elif ans == "False":
            false_cnt += 1
            # neg_ans_conf

        prop_idx += 1
        if (part1, part2) in object_parts_propositions:
            assert triplet not in object_parts_propositions[(part1, part2)]
            object_parts_propositions[(part1, part2)][triplet] = {"conf": prediction_confidence, "idx": prop_idx}
        else:
            object_parts_propositions[(part1, part2)]= {triplet: {"conf": prediction_confidence, "idx": prop_idx}}

        ###
        # ===Individual beliefs===
        # confidence from macaw, in format: <weight> <belief-id> 0
        ###
        # print("{} {} 0".format(prediction_confidence,prop_idx)) # 1-based idx
        wcnf_lines_ind.append("{} {} 0".format(prediction_confidence,prop_idx))
    #     # negation
    #     prop_idx += 1
    #     wcnf_lines_ind_negations.append("{} {} 0".format(neg_prediction_confidence,prop_idx))
    #     wcnf_lines_cons_TF.append("10000 {} {} 0".format(-(prop_idx - 1), -prop_idx))

    # assert prop_idx == (true_cnt + false_cnt) * 2
    
    return wcnf_lines_ind, wcnf_lines_ind_negations, wcnf_lines_cons_TF, object_parts_propositions

def preprocess_propositions_input(object_parts_propositions):
    # (x,y) and (y,x) under same key now
    no_order_object_parts_propositions = {}
    for pair in object_parts_propositions:
        pair_no_order = tuple(sorted(list(pair)))
        if pair_no_order in no_order_object_parts_propositions:
            no_order_object_parts_propositions[pair_no_order][pair] = object_parts_propositions[pair]
        else:
            no_order_object_parts_propositions[pair_no_order] =  {pair: object_parts_propositions[pair]}

    # (x, rln) as key
    x_rln_propositions = {}
    for pair in object_parts_propositions:
        propositions_conf_idx_dict = object_parts_propositions[pair]
        for proposition in propositions_conf_idx_dict:
            x_rln = (proposition[0], proposition[1])
            if x_rln in x_rln_propositions:
                assert proposition not in x_rln_propositions[x_rln]
                x_rln_propositions[x_rln][proposition] = propositions_conf_idx_dict[proposition]
            else:
                x_rln_propositions[x_rln] = {proposition: propositions_conf_idx_dict[proposition]}
                
    return no_order_object_parts_propositions, x_rln_propositions


def get_constraints(no_order_object_parts_propositions, x_rln_propositions, verbose = False):
    '''
    2) Getting WCNF format... Processing constraints...
    Constraints:
    •Symmetric 
        mutual_relations("directly connected to", "next to") [both sides of op. mean the same]
        x rln y <-> y rln x

    •Asymmetric (x above y → NOT y above x) (y above x → NOT x above y) [one must be false, nand]
        relation_opp_dict
        x rln y -> NOT (y rln x) <->
        y rln x -> NOT (x rln y) <->
        NOT(x rln y) OR NOT (y rln x)

    •Inverse (x above y <-> y below x) [both sides of op. mean the same]
        relation_opp_dict
        x rln y <-> y inverse(rln) x

    •Transitivity (x above y, y above z → x above z)
        all relations expect functional dependencies and "directly connected to"
        all_relations_lst - functional_dep_plus_directly_connected
        x rln y ^ y rln z -> x rln z

    '''         

    impose_contraints_logfile.write("2) Getting WCNF format... Processing constraints...\n")
    wcnf_lines_cons = []

    for unordered_pair in no_order_object_parts_propositions:
        unordered_pair_relations = no_order_object_parts_propositions[unordered_pair]
        # take sorted order as the x y order
        x_y_relations = unordered_pair_relations[unordered_pair]
        y_x_relations = unordered_pair_relations[(unordered_pair[1], unordered_pair[0])]
        for p1, rln, p2 in x_y_relations:
            
            # assert judgement avail for both x rln y (A), y rln x (B)
            assert (p2, rln, p1) in y_x_relations
            a_statement_idx = x_y_relations[(p1, rln, p2)]['idx']
            b_statement_idx = y_x_relations[(p2, rln, p1)]['idx']
            # [CONSTRAINT TYPE1] Symmetric (mutual_relations): x rln y <-> y rln x [A <-> B]
            if rln in mutual_relations:
                # they must be same judgement
                if verbose:
                    impose_contraints_logfile.write("[CONSTRAINT TYPE1] Symmetric (mutual_relations): x rln y <-> y rln x [A <-> B]" + "\n")
                    impose_contraints_logfile.write(str((p1, rln, p2)) + str((p2, rln, p1)) + "\n")
                    impose_contraints_logfile.write("10000 {} {} 0".format(-a_statement_idx, b_statement_idx) + "\n") # -A OR B
                    impose_contraints_logfile.write("10000 {} {} 0".format(-b_statement_idx, a_statement_idx) + "\n") # -B OR A
                wcnf_lines_cons.append("10000 {} {} 0".format(-a_statement_idx, b_statement_idx))
                wcnf_lines_cons.append("10000 {} {} 0".format(-b_statement_idx, a_statement_idx))

            # [CONSTRAINT TYPE2] Asymmetric (relation_opp_dict): NOT(x rln y) OR NOT (y rln x) [-A OR -B]
            if rln in relation_opp_dict:
                # they cannot both be True
                if verbose:
                    impose_contraints_logfile.write("[CONSTRAINT TYPE2] Asymmetric (relation_opp_dict): NOT(x rln y) OR NOT (y rln x) [-A OR -B]" + "\n")
                    impose_contraints_logfile.write(str((p1, rln, p2)) + str((p2, rln, p1)) + "\n")
                    impose_contraints_logfile.write("10000 {} {} 0".format(-a_statement_idx, -b_statement_idx) + "\n") # -A OR -B
                wcnf_lines_cons.append("10000 {} {} 0".format(-a_statement_idx, -b_statement_idx))

            # If model made judgement for both  x rln y (A), y inverse(rln) x (B)
            # [CONSTRAINT TYPE3] Inverse (relation_opp_dict):  x rln y (A) <-> y inverse(rln) x (B) [A <-> B]
            if rln in relation_opp_dict:
                assert (p2, relation_opp_dict[rln], p1) in y_x_relations
                # they must be same judgement
                a_statement_idx = x_y_relations[(p1, rln, p2)]['idx']
                b_statement_idx = y_x_relations[(p2, relation_opp_dict[rln], p1)]['idx']
                if verbose:
                    impose_contraints_logfile.write("[CONSTRAINT TYPE3] Inverse (relation_opp_dict):  x rln y (A) <-> y inverse(rln) x (B) [A <-> B]" + "\n")
                    impose_contraints_logfile.write(str((p1, rln, p2)) + str((p2, relation_opp_dict[rln], p1)) + "\n")
                    impose_contraints_logfile.write("10000 {} {} 0".format(-a_statement_idx, b_statement_idx) + "\n") # -A OR B
                    impose_contraints_logfile.write("10000 {} {} 0".format(-b_statement_idx, a_statement_idx) + "\n") # -B OR A
                wcnf_lines_cons.append("10000 {} {} 0".format(-a_statement_idx, b_statement_idx))
                wcnf_lines_cons.append("10000 {} {} 0".format(-b_statement_idx, a_statement_idx))

    transitive_relations = [x for x in all_relations_lst if x not in functional_dep_plus_directly_connected]
    for x_rln in x_rln_propositions:
        x_rln_relations = x_rln_propositions[x_rln]
        for p1, rln, p2 in x_rln_relations: # for each x rln y
            # [CONSTRAINT TYPE4] Transitivity (transitive_relations): x rln y (A) ^ y rln z (B) -> x rln z (C) [A ^ B -> C]
            if rln in transitive_relations: # if it is a transitive relation
                assert (p2, rln) in x_rln_propositions
                for p1_2, rln_2, p2_2 in x_rln_propositions[(p2, rln)]: # y rln z1, y rln z2 ...
                    assert p2 == p1_2 and rln == rln_2
                    if p1 == p2_2:
                        continue
                    assert (p1, rln, p2_2) in  x_rln_relations # x rln z1, x rln z2 ...
        
                    # checked model made judgement for all 3 statements, it must satisfy A ^ B -> C
                    a_statement_idx = x_rln_propositions[(p1, rln)][(p1, rln, p2)]['idx']
                    b_statement_idx = x_rln_propositions[(p1_2, rln_2)][(p1_2, rln_2, p2_2)]['idx']
                    c_statement_idx = x_rln_propositions[(p1, rln)][(p1, rln, p2_2)]['idx']
                    if verbose:
                        impose_contraints_logfile.write("[CONSTRAINT TYPE4] Transitivity (transitive_relations): x rln y (A) ^ y rln z (B) -> x rln z (C) [A ^ B -> C]" + "\n")
                        impose_contraints_logfile.write(str((p1, rln, p2)) + str((p1_2, rln_2, p2_2)) + str((p1, rln, p2_2)) + "\n")
                        impose_contraints_logfile.write("10000 {} {} {} 0".format(-a_statement_idx, -b_statement_idx, c_statement_idx) + "\n") # -A OR -B OR C
                    wcnf_lines_cons.append("10000 {} {} {} 0".format(-a_statement_idx, -b_statement_idx, c_statement_idx))

    impose_contraints_logfile.write("Update: Getting WCNF format... Processing constraints... DONE!\n")
    return wcnf_lines_cons

def store_wcnf(wcnf_dir, device, turker, wcnf_lines_ind, wcnf_lines_cons):
    '''
    3) Store the WCNF for reference
    '''
    impose_contraints_logfile.write("3) Saving WCNF format to file..." + "\n")
    # Write WCNFs to a file 
    make_sure_dir_exists(wcnf_dir)
    wcnf_filename = "{wcnf_dir}{device}_{turker}.wcnf".format(wcnf_dir=wcnf_dir, device=device.replace(" ", "-"), turker=turker)
    with open(wcnf_filename, "w") as outfile:
        outfile.write("\n".join(["p wcnf {variables} {clauses} 10000".format(variables=len(wcnf_lines_ind), clauses=len(wcnf_lines_ind) + len(wcnf_lines_cons))] + wcnf_lines_cons + wcnf_lines_ind) + "\n")

    impose_contraints_logfile.write("Update: Saving WCNF format to file... DONE!" + "\n")
    return wcnf_filename

def solve_rc2(wcnf_filename):
    '''
    4) Compute using solver
    '''
    impose_contraints_logfile.write("4) Solving with RC2 solver..." + "\n")
    wcnf = WCNF(from_file=wcnf_filename)
    # # # Just for our info. Enumerate top MaxSAT solutions (from best to worst).
    # with RC2(wcnf) as rc2:
        # for m in rc2.enumerate():
        #     impose_contraints_logfile.write('model {0} has cost {1}'.format(m, rc2.cost)  + "\n")

    # For computing an assignment satisfying all hard clauses of the input formula and 
    # maximizing the sum of weights of satisfied soft clauses.
    with RC2(wcnf) as rc2:
    #with RC2Stratified(wcnf, blo='full') as rc2:
        model = rc2.compute()
        impose_contraints_logfile.write('model {0} has cost {1}'.format(model, rc2.cost) + "\n")

    remaining_props = [num for num in model if num > 0]
    impose_contraints_logfile.write("Update: Solving with RC2 solver ===COMPLETED!===\n" + "\n")

    return remaining_props

def fuzzy_fixed_mm(object_parts_propositions, wcnf_lines_ind, remaining_props, triplet_ans_conf_lst, neg_ans_conf_lst):
    total_statements_cnt = 0
    maxsat_selected_props = {}
    model_believe_true_props = {}

    model_believe_true_triplets = []
    for idx_triplet, (triplet, ans, ans_conf) in enumerate(triplet_ans_conf_lst):
        if ans_conf > neg_ans_conf_lst[idx_triplet]:
             model_believe_true_triplets.append(triplet)
    
    #model_believe_true_triplets = [x[0] for x in triplet_ans_conf_lst_true ] neg_ans_conf_lst
    for pair in object_parts_propositions:
        for triplet in object_parts_propositions[pair]:
            # going over all statements 
            total_statements_cnt += 1

            if object_parts_propositions[pair][triplet]['idx'] in remaining_props:
                maxsat_selected_props[triplet] = object_parts_propositions[pair][triplet]
            if triplet in model_believe_true_triplets:
                model_believe_true_props[triplet] = object_parts_propositions[pair][triplet]

    # check all props that were input to maxsat are processed here
    assert total_statements_cnt == len(wcnf_lines_ind)
    return model_believe_true_props, maxsat_selected_props

def run_maxsat(device, turker, wcnf_dir, triplet_ans_conf_lst, neg_ans_conf_lst, triplet_ans_conf_lst_true, use_only_model_true_props = False):
    impose_contraints_logfile.write(device + " " + turker + "\n")

    # get individual beliefs
    if use_only_model_true_props: ### NOT IMPLEMENTED YET for neg_ans_conf_lst
        input_lst = triplet_ans_conf_lst_true
    else:
        input_lst = triplet_ans_conf_lst
    wcnf_lines_ind, wcnf_lines_ind_negations, wcnf_lines_cons_TF, object_parts_propositions = get_propositions_input(input_lst, neg_ans_conf_lst)

    print("Getting constraints...")
    # impose constraints (based on true versions of statements and associated conf.)
    no_order_object_parts_propositions, x_rln_propositions = preprocess_propositions_input(object_parts_propositions)
    wcnf_lines_cons = get_constraints(no_order_object_parts_propositions, x_rln_propositions, verbose = False) # TRUE for detailed printout

    print("Store wcnf...")
    # store and compute (with negated version and nand constraints)
    wcnf_filename = store_wcnf(wcnf_dir, device, turker, wcnf_lines_ind + wcnf_lines_ind_negations, wcnf_lines_cons_TF + wcnf_lines_cons)

    print("Call solver...")
    remaining_props = solve_rc2(wcnf_filename)
    print("Solver finished running!")

    # fuzzy vs. fixed mm
    model_believe_true_props, maxsat_selected_props = fuzzy_fixed_mm(object_parts_propositions, wcnf_lines_ind, remaining_props, triplet_ans_conf_lst, neg_ans_conf_lst)

    return model_believe_true_props, maxsat_selected_props

### Step3: Model confidence-based filtering (optional)
def filter_props(props, threshold):
    filtered_props = {}
    for prop in props:
        if props[prop]['conf'] > threshold:
            filtered_props[prop] = props[prop]
            
    return filtered_props


### Step4: Visualize the model's MM of device before and after
def generate_graph_png(device, turker, props_info_dict, plots_folder, tag =""): # based on relations
    device_str = device.replace(" ", "-")
    g = graphviz.Digraph(device_str, filename=device_str+ "_" + turker + "_" + tag.replace(" ", "-"), directory=plots_folder, format='png')
    # get dot code from the relations stored
    graphviz_code = [] 
    for triplet in props_info_dict:
        #conf = props_info_dict[triplet]['conf']
        #g.edge(triplet[0], triplet[-1], label="{position} ({confidence})".format(position = triplet[1], confidence = conf))
        g.edge(triplet[0], triplet[-1], label="{position}".format(position = triplet[1]))
        
    g.render()
