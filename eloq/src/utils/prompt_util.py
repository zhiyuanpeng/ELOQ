from eloq.src.api.llmlib import LLM
from eloq.src.utils import doc_util as utils
import os, re, json

document_transforms = None
question_generation = None
rag_confusion_check = None
examples_of_questions = None

def read_prompts(folder):

    global document_transforms
    with open(os.path.join(folder, "document-transforms.json"), "r") as f:
        document_transforms_raw = json.load(f)
    document_transforms = {}
    for key, prompts_raw in document_transforms_raw.items():
        assert {"system", "user_reduce"}.issubset(prompts_raw.keys())
        prompts = {}
        for p_type in prompts_raw.keys():
            if isinstance(prompts_raw[p_type], str) or prompts_raw[p_type] is None:
                prompts[p_type] = prompts_raw[p_type]
            else:
                prompts[p_type] = "".join(prompts_raw[p_type])
        document_transforms[key] = prompts

    global question_generation
    with open(os.path.join(folder, "question-generation.json"), "r") as f:
        question_generation_raw = json.load(f)
    question_generation = {}
    for key, prompts_raw in question_generation_raw.items():
        assert {"system", "user_orig", "user_conf"}.issubset(prompts_raw.keys())
        prompts = {}
        for p_type in prompts_raw.keys():
            if isinstance(prompts_raw[p_type], str) or prompts_raw[p_type] is None:
                prompts[p_type] = prompts_raw[p_type]
            else:
                prompts[p_type] = "".join(prompts_raw[p_type])
        question_generation[key] = prompts

    global rag_confusion_check
    with open(os.path.join(folder, "rag-confusion-check.json"), "r") as f:
        rag_confusion_check_raw = json.load(f)
    rag_confusion_check = {}
    for key, prompts_raw in rag_confusion_check_raw.items():
        assert {"system", "user_rag"}.issubset(prompts_raw.keys())
        prompts = {}
        for p_type in prompts_raw.keys():
            if isinstance(prompts_raw[p_type], str) or prompts_raw[p_type] is None:
                prompts[p_type] = prompts_raw[p_type]
            else:
                prompts[p_type] = "".join(prompts_raw[p_type])
        rag_confusion_check[key] = prompts

    global examples_of_questions
    with open(os.path.join(folder, "examples-of-questions.json"), "r") as f:
        examples_of_questions_raw = json.load(f)
    examples_of_questions = {}
    for key, example_raw in examples_of_questions_raw.items():
        assert {"document", "source", "orig_questions", "conf_questions"}.issubset(example_raw.keys())
        example = {}
        if isinstance(example_raw["document"], str):
            example["document"] = example_raw["document"]
        else:
            example["document"] = "".join(example_raw["document"])
        assert isinstance(example_raw["orig_questions"], list)
        assert isinstance(example_raw["conf_questions"], list)
        example["num_q"] = len(example_raw["orig_questions"])
        # assert example["num_q"] == len(example_raw["conf_questions"])
        example["orig_questions"] = example_raw["orig_questions"]
        example["conf_questions"] = example_raw["conf_questions"]
        if "facts" in example_raw:
            example["facts"] = example_raw["facts"]
        examples_of_questions[key] = example

def reduce_document(llm, document, num_fact, prompt_key):
    prompt = []
    if document_transforms[prompt_key]["system"]:
        prompt.append({
            "role" : "system",
            "content" : document_transforms[prompt_key]["system"]
        })
    prompt.append({
        "role" : "user",
        "content" : document_transforms[prompt_key]["user_reduce"].format(document = document, num_fact=num_fact)
    })
    reduce_doc = LLM.get(llm)(prompt)
    return reduce_doc


def modify_reduced_document(llm, document, reduce_doc, num_fact, prompt_key):
    doc_0 = reduce_doc
    doc_1 = suppress_facts(doc_0, lambda i: (i % 3 == 2))
    doc_2 = impute_facts(llm, doc_1, num_fact, prompt_key)
    doc_3 = suppress_facts(doc_2, lambda i: (i % 3 == 1))
    doc_4 = impute_facts(llm, doc_3, num_fact, prompt_key)
    doc_5 = suppress_facts(doc_4, lambda i: (i % 3 == 0))
    doc_6 = impute_facts(llm, doc_5, num_fact, prompt_key)

    doc_7 = suppress_facts(doc_6, lambda i: (i % 3 == 2))
    doc_8 = impute_facts(llm, doc_7, num_fact, prompt_key)
    doc_9 = suppress_facts(doc_8, lambda i: (i % 3 == 1))
    doc_10 = impute_facts(llm, doc_9, num_fact, prompt_key)
    doc_11 = suppress_facts(doc_10, lambda i: (i % 3 == 0))
    doc_12 = impute_facts(llm, doc_11, num_fact, prompt_key)

    doc_13 = suppress_facts(doc_12, lambda i: (i % 3 == 2))
    doc_14 = impute_facts(llm, doc_13, num_fact, prompt_key)
    doc_15 = suppress_facts(doc_14, lambda i: (i % 3 == 1))
    doc_16 = impute_facts(llm, doc_15, num_fact, prompt_key)
    doc_17 = suppress_facts(doc_16, lambda i: (i % 3 == 0))
    doc_18 = impute_facts(llm, doc_17, num_fact, prompt_key)

    modify_doc = doc_18
    remained_facts = remove_facts(llm, document, reduce_doc, modify_doc, num_fact, prompt_key)
    return remained_facts

def impute_facts(llm, missing_facts_doc, num_fact, prompt_key):
    prompt = []
    if document_transforms[prompt_key]["system"]:
        prompt.append({
            "role" : "system",
            "content" : document_transforms[prompt_key]["system"]
        })
    prompt.append({
        "role" : "user",
        "content" : document_transforms[prompt_key]["user_modify"].format(document = missing_facts_doc, num_fact=num_fact)
    })
    imputed_facts_doc = LLM.get(llm)(prompt)
    lines = imputed_facts_doc.splitlines()
    if "list of facts" in lines[0].lower():
        imputed_facts_doc = "\n".join(lines[1:])
    return imputed_facts_doc

def remove_facts(llm, document, ori_facts, hallucinated_facts, num_fact, prompt_key):
    example_document = examples_of_questions["z-sport-5-5"]["document"]
    true_facts_list = examples_of_questions["z-sport-5-5"]["facts"]["true_facts"]
    false_facts_list = examples_of_questions["z-sport-5-5"]["facts"]["exp_hallucinated_facts"]
    true_facts = utils.enum_list(true_facts_list[:4])
    false_facts = "\n".join(false_facts_list)
    remained_facts = "\n".join(examples_of_questions["z-sport-5-5"]["facts"]["remained_hallucinated_facts"])
    prompt = []
    if document_transforms[prompt_key]["system"]:
        prompt.append({
            "role" : "system",
            "content" : document_transforms[prompt_key]["system"]
        })
    prompt.append({
        "role" : "user",
        "content" : document_transforms[prompt_key]["user_remove"].format(document = example_document, ori_facts = true_facts, hallucinated_facts = false_facts, num_true_fact=len(true_facts_list[:4]), num_false_fact=len(false_facts_list))
    })

    prompt.append({
        "role" : "assistant",
        "content" : remained_facts
    })

    prompt.append({
        "role" : "user",
        "content" : document_transforms[prompt_key]["user_remove"].format(document = document, ori_facts = ori_facts, hallucinated_facts = hallucinated_facts, num_true_fact=num_fact, num_false_fact=num_fact)
    })
    remained_facts = LLM.get(llm)(prompt)
    return remained_facts

def suppress_facts(text, suppress):
    raw_lines = text.splitlines()
    lines = [line.strip() for line in raw_lines]
    facts = []
    for line in lines:
        if len(line) > 0:
            x = re.search(r"^\d+[:\.]\s+", line)
            if x:
                facts.append(line[x.span()[1]:])
            else:
                x = re.search(r"^[:\.\-\*\+]\s+", line)
                if x:
                    facts.append(line[x.span()[1]:])
                else:
                    facts.append(line)
    for i in range(len(facts)):
        if suppress(i):
            facts[i] = "(missing)"
    return utils.enum_list(facts)

def generate_questions(llm, document, num_q, prompt_key = "q-z-1"):
    exp_document = examples_of_questions["z-sport-5-5"]["document"]
    exp_ori_questions = []
    for t in examples_of_questions["z-sport-5-5"]["orig_questions"]:
        exp_ori_questions.append(t["question"])
    exp_num_q = len(exp_ori_questions)
    exp_ori_questions = utils.enum_list(exp_ori_questions)
    prompt = []
    if question_generation[prompt_key]["system"]:
        prompt.append({
            "role" : "system",
            "content" : question_generation[prompt_key]["system"]
        })
    prompt.append({
            "role" : "user",
            "content" : question_generation[prompt_key]["user_orig"].format(num_q = exp_num_q, document = exp_document)
        })
    prompt.append({
        "role" : "assistant",
        "content" : exp_ori_questions
    })
    prompt.append({
        "role" : "user",
        "content" : question_generation[prompt_key]["user_orig"].format(num_q = num_q, document = document)
    })
    raw_questions = LLM.get(llm)(prompt)
    questions = utils.parse_numbered_questions(raw_questions)
    return questions

def confuse_questions_v2(llm, document, hallucinated_facts, prompt_key = "q-z-1"):
    '''
    Convert hallucinated facts into questions that can't be answered by the original document
    '''
    prompt = []
    if question_generation[prompt_key]["system"]:
        prompt.append({
            "role" : "system",
            "content" : question_generation[prompt_key]["system"]
        })
    prompt.append({
        "role" : "user",
        "content" : question_generation[prompt_key]["user_conf"].format(document = document, hallucinated_facts = hallucinated_facts)
    })
    raw_questions = LLM.get(llm)(prompt)
    questions = utils.parse_numbered_questions(raw_questions)
    return questions

def generate_response(llm, document, question, prompt_key = "r-z-1"):
    prompt = []
    if rag_confusion_check[prompt_key]["system"]:
        prompt.append({
            "role" : "system",
            "content" : rag_confusion_check[prompt_key]["system"]
        })
    prompt.append({
        "role" : "user",
        "content" : rag_confusion_check[prompt_key]["user_rag"].format(document = document, question = question)
    })
    response = LLM.get(llm)(prompt)
    return response

def generate_response_2shot(llm, document, question, prompt_key = "r-z-1"):
    example_document = examples_of_questions["z-sport-5-5"]["document"]
    example_questions = examples_of_questions["z-sport-5-5"]["facts"]["two_shot_questions"]["questions"]
    example_answers = examples_of_questions["z-sport-5-5"]["facts"]["two_shot_questions"]["responses"]
    example_questions = utils.enum_list(example_questions)
    example_answers = utils.enum_list(example_answers)
    prompt = []
    if rag_confusion_check[prompt_key]["system"]:
        prompt.append({
            "role" : "system",
            "content" : rag_confusion_check[prompt_key]["system"]
        })
    prompt.append({
        "role" : "user",
        "content" : rag_confusion_check[prompt_key]["user_rag"].format(document = example_document, question = example_questions)
    })
    prompt.append({
        "role" : "assistant",
        "content" : example_answers
    })
    prompt.append({
        "role" : "user",
        "content" : rag_confusion_check[prompt_key]["user_rag"].format(document = document, question = question)
    })
    response = LLM.get(llm)(prompt)
    return response

def generate_response_cot(llm, document, question, prompt_key = "r-z-1"):
    prompt = []
    if rag_confusion_check[prompt_key]["system"]:
        prompt.append({
            "role" : "system",
            "content" : rag_confusion_check[prompt_key]["system"]
        })
    prompt.append({
        "role" : "user",
        "content" : rag_confusion_check[prompt_key]["user_rag_cot"].format(document = document, question = question)
    })
    response = LLM.get(llm)(prompt)
    return response
    
def find_confusion(llm, document, question, n, prompt_key = "r-z-1"):
    exp_document = examples_of_questions["z-sport-5-5"]["document"]
    exp_ori_questions, exp_ori_reasonings, exp_conf_questions, exp_conf_reasonings = [], [], [], []
    for t in examples_of_questions["z-sport-5-5"]["orig_questions"]:
        exp_ori_questions.append(t["question"])
        exp_ori_reasonings.append(t["explanation"]+ " This answer is No.")
    for t in examples_of_questions["z-sport-5-5"]["conf_questions"]:
        exp_conf_questions.append(t["question"])
        exp_conf_reasonings.append(t["explanation"]+ " This answer is Yes.")
    exp_questions = utils.enum_list(exp_ori_questions + exp_conf_questions)
    exp_reasonings = "\n\n" + utils.enum_list(exp_ori_reasonings + exp_conf_reasonings)
    prompt = []
    if rag_confusion_check[prompt_key]["system"]:
        prompt.append({
            "role" : "system",
            "content" : rag_confusion_check[prompt_key]["system"]
        })
    prompt.append({
        "role" : "user",
        "content" : rag_confusion_check[prompt_key]["user_conf_rag_example"].format(document = exp_document, question = exp_questions)
    })
    prompt.append({
        "role" : "assistant",
        "content" : exp_reasonings
    })
    prompt.append({
        "role" : "user",
        "content" : rag_confusion_check[prompt_key]["user_conf_rag_example"].format(document = document, question = question)
    })
    confusion = LLM.get(llm)(prompt, n=n)
    if n != 1:
        answers = []
        for conf in confusion:
            if (
                conf.lower().endswith("no") or
                conf.lower().endswith("answer: no") or
                conf.lower().endswith("the answer is: no") or
                conf.lower().endswith("the answer is \"no\"") or
                conf.lower().endswith("no.") or
                conf.lower().endswith("answer: no.") or
                conf.lower().endswith("the answer is: no.") or
                conf.lower().endswith("the answer is \"no.\"")
            ):
                answers.append("none")
            else:
                answers.append("yes")
        majority = max(answers, key = answers.count)
        return majority
    if (
            confusion.lower().endswith("no") or
            confusion.lower().endswith("answer: no") or
            confusion.lower().endswith("the answer is: no") or
            confusion.lower().endswith("the answer is \"no\"") or
            confusion.lower().endswith("no.") or
            confusion.lower().endswith("answer: no.") or
            confusion.lower().endswith("the answer is: no.") or
            confusion.lower().endswith("the answer is \"no.\"")
        ):
        return "none"
    else:
        return confusion

def check_response_for_defusion(llm, document, question, response, n, shot, prompt_key = "r-z-1"):
    prompt = []
    if rag_confusion_check[prompt_key]["system"]:
        prompt.append({
            "role" : "system",
            "content" : rag_confusion_check[prompt_key]["system"]
        })
    example_document = examples_of_questions["z-sport-5-5"]["document"]
    example_questions = examples_of_questions["z-sport-5-5"]["facts"]["defuse_scope"]["question"]
    example_answers = examples_of_questions["z-sport-5-5"]["facts"]["defuse_scope"]["responses"]
    example_reasonings = examples_of_questions["z-sport-5-5"]["facts"]["defuse_scope"]["reasonings"]
    assert len(example_answers) == len(example_reasonings)

    exp_questions_extra = examples_of_questions["z-sport-5-5"]["facts"]["defuse_extra"]["question"]
    exp_answers_extra = examples_of_questions["z-sport-5-5"]["facts"]["defuse_extra"]["responses"]
    exp_reasonings_extra = examples_of_questions["z-sport-5-5"]["facts"]["defuse_extra"]["reasonings"]
    assert len(exp_answers_extra) == len(exp_reasonings_extra)

    example_questions = utils.enum_list(example_questions*len(example_answers) + exp_questions_extra*len(exp_answers_extra))
    example_answers = utils.enum_list(example_answers + exp_answers_extra)
    example_reasonings = "\n\n" + utils.enum_list(example_reasonings + exp_reasonings_extra)
    prompt.append({
        "role" : "user",
        "content" : rag_confusion_check[prompt_key]["user_rag"].format(document = example_document, question = example_questions)
    })
    prompt.append({
        "role" : "assistant",
        "content" : example_answers
    })
    prompt.append({
        "role" : "user",
        "content" : rag_confusion_check[prompt_key]["user_def_check"]
    })
    prompt.append({
        "role" : "assistant",
        "content" : example_reasonings
    })
    prompt.append({
        "role" : "user",
        "content" : rag_confusion_check[prompt_key]["user_rag"].format(document = document, question = question)
    })
    prompt.append({
        "role" : "assistant",
        "content" : response
    })
    prompt.append({
        "role" : "user",
        "content" : rag_confusion_check[prompt_key]["user_def_check"]
    })
    defusion = LLM.get(llm)(prompt, n=n)
    if n != 1:
        answers = []
        for defu in defusion:
            if (
                defu.lower().endswith("no") or
                defu.lower().endswith("answer: no") or
                defu.lower().endswith("the answer is: no") or
                defu.lower().endswith("the answer is \"no\"") or
                defu.lower().endswith("no.") or
                defu.lower().endswith("answer: no.") or
                defu.lower().endswith("the answer is: no.") or
                defu.lower().endswith("the answer is \"no.\"")
            ):
                answers.append("no")
                no_defu = defu
            elif (
                defu.lower().endswith("yes") or
                defu.lower().endswith("answer: yes") or
                defu.lower().endswith("the answer is: yes") or
                defu.lower().endswith("the answer is \"yes\"") or
                defu.lower().endswith("yes.") or
                defu.lower().endswith("answer: yes.") or
                defu.lower().endswith("the answer is: yes.") or
                defu.lower().endswith("the answer is \"yes.\"")
            ):
                answers.append("yes") 
                yes_defu = defu
            else:
                answers.append("unsure")
                unsure_defu = defu
        majority = max(answers, key = answers.count)
        if majority == "no":
            return no_defu, majority
        elif majority == "yes":
            return yes_defu, majority
        else:
            return unsure_defu, majority
    if (
            defusion.lower().endswith("no") or
            defusion.lower().endswith("answer: no") or
            defusion.lower().endswith("the answer is: no") or
            defusion.lower().endswith("the answer is \"no\"") or
            defusion.lower().endswith("no.") or
            defusion.lower().endswith("answer: no.") or
            defusion.lower().endswith("the answer is: no.") or
            defusion.lower().endswith("the answer is \"no.\"")
        ):
        is_defused = "no"
    elif (
            defusion.lower().endswith("yes") or
            defusion.lower().endswith("answer: yes") or
            defusion.lower().endswith("the answer is: yes") or
            defusion.lower().endswith("the answer is \"yes\"") or
            defusion.lower().endswith("yes.") or
            defusion.lower().endswith("answer: yes.") or
            defusion.lower().endswith("the answer is: yes.") or
            defusion.lower().endswith("the answer is \"yes.\"")
        ):
        is_defused = "yes"
    else:
        is_defused = "unsure"
    return defusion, is_defused