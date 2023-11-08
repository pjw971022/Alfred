
PICK_TARGETS = {
  "blue block": None,
  "red block": None,
  "green block": None,
  "yellow block": None,
}

COLORS = {
    "blue":   (78/255,  121/255, 167/255, 255/255),
    "red":    (255/255,  87/255,  89/255, 255/255),
    "green":  (89/255,  169/255,  79/255, 255/255),
    "yellow": (237/255, 201/255,  72/255, 255/255),
}

PLACE_TARGETS = {
  "blue block": None,
  "red block": None,
  "green block": None,
  "yellow block": None,

  "blue bowl": None,
  "red bowl": None,
  "green bowl": None,
  "yellow bowl": None,

  "top left corner":     (-0.3 + 0.05, -0.2 - 0.05, 0),
  "top right corner":    (0.3 - 0.05,  -0.2 - 0.05, 0),
  "middle":              (0,           -0.5,        0),
  "bottom left corner":  (-0.3 + 0.05, -0.8 + 0.05, 0),
  "bottom right corner": (0.3 - 0.05,  -0.8 + 0.05, 0),
}

overwrite_cache = True
if overwrite_cache:
  LLM_CACHE = {}


from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

import torch

from PIL import Image

def blip_call(state, text_query):
    model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
    processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    inputs = processor(images=state, text=text_query, return_tensors="pt").to(device)
    outputs = model.forward(    
        **inputs,
                            )
    # outputs = model.generate(
    #     **inputs,
    #     do_sample=False,
    #     num_beams=5,
    #     max_length=256,
    #     min_length=1,
    #     top_p=0.9,
    #     repetition_penalty=1.5,
    #     length_penalty=1.0,
    #     temperature=1,
    # )
    return outputs

def blip_scoring(state, text_query, task_options, constraint_options, option_start="\n", limit_num_options=None, ):
    if limit_num_options:
        task_options = task_options[:limit_num_options]
    # verbose and print("Scoring", len(options), "options")
    blip_prompt_options = [text_query + task_option for task_option in task_options]
    
    
    response = blip_call(
        state,
        blip_prompt_options,
    )
    # response = gpt3_call(
    #     engine=engine,
    #     prompt=gpt3_prompt_options,
    #     max_tokens=0,
    #     logprobs=1,
    #     temperature=0,
    #     echo=True, )
    # desc = "Scoring " + str(len(options)) + " options\n"

    scores = {}
    for option, choice in zip(options, response["logits"]): # @
        tokens = choice["logprobs"]["tokens"]
        token_logprobs = choice["logprobs"]["token_logprobs"]

        total_logprob = 0
        for token, token_logprob in zip(reversed(tokens), reversed(token_logprobs)):
            # print_tokens and print(token, token_logprob)
            if option_start is None and not token in option:
                break
            if token == option_start:
                break
            total_logprob += token_logprob
        scores[option] = total_logprob

    for i, option in enumerate(sorted(scores.items(), key=lambda x: -x[1])):
        # verbose and print(option[1], "\t", option[0])
        desc = desc + str(option[1]) + "\t" + str(option[0]) + "\n"
        if i >= 10:
            break

    return scores, desc


def make_options(pick_targets=None, place_targets=None, options_in_api_form=True, termination_string="done()"):
    if not pick_targets:
        pick_targets = PICK_TARGETS
    if not place_targets:
        place_targets = PLACE_TARGETS
    options = []
    for pick in pick_targets:
        for place in place_targets:
            if options_in_api_form:
                option = "robot.pick_and_place({}, {})".format(pick, place)
            else:
                option = "Pick the {} and place it on the {}.".format(pick, place)
            options.append(option)

    options.append(termination_string)
    print("Considering", len(options), "options")
    return options



def affordance_score(state, task_options, constraint_options):
    return None


if __name__ == "__main__":
    for idx, description_and_solution in enumerate(initial_state_description):
        # termination_string = "Finish"
        description = description_and_solution[0]
        solution = description_and_solution[1]
        text_query = system_prompt + description

        # options = make_options(PICK_TARGETS, PLACE_TARGETS, termination_string=termination_string)
        task_options = [
            "[pick up red block]",
            "[pick up blue block]",
            "[pick up yellow block]",
            "[close the box]",
            "[open the box]",
            "[place the block to the desk]",
            "[place the block in the box]",
            "[Finish]"
        ]
        constraint_options = []
    
        log = ""
        for i in range(7):
            # llm_scores, _, desc = gpt3_scoring(query, options, verbose=True, engine=ENGINE)
            vlm_scores, _, desc = blip_scoring(state, text_query, task_options, constraint_options)
            max_score = -1e+6
            max_option = None
            vlm_scores[solution[i]] += affordance_score(state, task_options, constraint_options)

            for option in vlm_scores.keys():
                if max_score < vlm_scores[option]:
                    max_score = vlm_scores[option]
                    max_option = option

            print(max_option)
            log = log + desc + "Solution: " + solution[i] + "\n\n"
            query = query + '\n' + max_option

        with open("./results/saycan/result_{}.txt".format(idx), 'w') as f:
            f.write(log + query + "\n\n" + "Solution: \n" + "\n".join(solution))

        break
