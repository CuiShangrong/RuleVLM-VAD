EXPRESSION_LIST = ["certain", "almost certain", "highly likely", "very good chance", "we believe",
 "probably", "probable", "likely", "better than even", "about even", "probably not", "we doubt",
 "unlikely", "little chance", "chances are slight", "improbable", "highly unlikely", "almost no chance", "impossible"]

def induction_messages(objects):
    return [
      {"role":"system","content":"As a surveillance monitor for urban safety using the ShanghaiTech dataset, my job is derive rules for detect abnormal human activities or environmental object."},
      {"role":"user","content":"Based on the assumption that the given frame description are normal, Please derive rules for normal, start from an abstract concept and then generalize to concrete activities (e.g., walking) or objects."},
      {"role":"assistant","content":"**Rules for Normal Human Activities:\n1. \n**Rules for Normal Environmental Objects:\n1."},
      {"role":"user","content":"Compared with the above rules for normal, can you provide potential rules for anomaly? Please start from an abstract concept then generalize to concrete activities (e.g., non-walking such as riding a bicycle, scooting, skateboarding.) or objects, compared with normal ones."},
      {"role":"assistant","content":"**Rules for Anomaly Human Activities:\n1. \n**Rules for Anomaly Environmental Objects:\n1."},
      {"role":"user","content":f"Now you are given {objects}. What are the Normal and Anomaly rules you got? Think step by step. Reply following the above format, start from an abstract concept and then generalize to concrete activities or objects. List them using short terms, not an entire sentence."}]

def correction_messages(rule_list,n):
    return [
      {"role":"system","content":"As a surveillance monitor for urban safety using the ShanghaiTech dataset, my job is to organize rules for detect abnormal activities and objeacts."},
      {"role":"user","content":f"You are given {n} independent sets of rules for Normal and Anomaly. For the organized normal Rules, list the given normal rules with high-frenquency elementsFor the organized anomaly Rules, list all the given anomaly rules"},
      {"role":"assistant","content":"**Rules for Anomaly Human Activities:\n1. Non-walking movement: riding a bicycle, scooting, skateboarding.\n2.\n**Rules for Anomaly Environmental Objects:\n1. Ground transportation: vehicles, motorcycles.\n2.\n**Rules for Normal Human Activities:\n1. Walking with common objects: a backpack, bag, umbrella.\n2.\n**Rules for Normal Environmental Objects:\n1. Architectural structures: building, bridges.\n2."},
      {"role":"user","content":f"Now you are given {n} independent sets of rules as the sublists of {rule_list}. What rules for Anomaly and Normal do you get? Think step by step, reply following the above examples."}]

def deduction_prompt(description,rule,initial_pred,anomaly_keyword=None):
    if initial_pred:
        initial=f"Anomaly, since I found {anomaly_keyword}."
        third=f"Third, are the human activities or environmental objects anomaly? Answer: anomaly, if you also find {anomaly_keyword} or ANY anomaly rule (even if only one, no matter human activities or environmental objects) matches, otherwise answer: normal."
    else:
        initial="Normal."
        third="Third, are the human activities or environmental objects anomaly? Answer: anomaly, if ANY anomaly rule (even if only one, no matter human activities or environmental objects) matches, otherwise answer: normal."
    return f"""You will be given an description of scene, you task is to double check my initial anomaly detection result based on the rules. The rules are:
{rule}

My initial result is {initial}
First, if human activity present, which rule is matching? List the rule category, e.g., normal or anomaly, with number.

Second, if environmental object present, which rule is matching? List the rule category, e.g., normal or anomaly, with number.

{third}

Now you are given the scene {description}, think step by step."""
