import openai
openai.api_key =
ENGINE = "gpt-3.5-turbo"  # "text-ada-001"

from initial_state import initial_state_description, system_prompt


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

def gpt3_call(engine="text-ada-001", prompt="", max_tokens=128, temperature=0,
              logprobs=1, echo=False):
  full_query = ""
  for p in prompt:
    full_query += p
  id = tuple((engine, full_query, max_tokens, temperature, logprobs, echo))
  if id in LLM_CACHE.keys():
    print('cache hit, returning')
    response = LLM_CACHE[id]
  else:
    response = openai.ChatCompletion.create(engine=engine,
                                        prompt=prompt,
                                        max_tokens=max_tokens,
                                        temperature=temperature,
                                        logprobs=logprobs,
                                        echo=echo)
    LLM_CACHE[id] = response
  return response


def gpt3_scoring(query, options, engine="text-ada-001", limit_num_options=None, option_start="\n", verbose=False, print_tokens=False):
    if limit_num_options:
        options = options[:limit_num_options]
    verbose and print("Scoring", len(options), "options")
    gpt3_prompt_options = [query + option for option in options]
    response = gpt3_call(
        engine=engine,
        prompt=gpt3_prompt_options,
        max_tokens=0,
        logprobs=1,
        temperature=0,
        echo=True, )
    desc = "Scoring " + str(len(options)) + " options\n"

    scores = {}
    for option, choice in zip(options, response["choices"]):
        tokens = choice["logprobs"]["tokens"]
        token_logprobs = choice["logprobs"]["token_logprobs"]

        total_logprob = 0
        for token, token_logprob in zip(reversed(tokens), reversed(token_logprobs)):
            print_tokens and print(token, token_logprob)
            if option_start is None and not token in option:
                break
            if token == option_start:
                break
            total_logprob += token_logprob
        scores[option] = total_logprob

    for i, option in enumerate(sorted(scores.items(), key=lambda x: -x[1])):
        verbose and print(option[1], "\t", option[0])
        desc = desc + str(option[1]) + "\t" + str(option[0]) + "\n"
        if i >= 10:
            break

    return scores, response, desc


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

if __name__ == "__main__":

    for idx, description_and_solution in enumerate(initial_state_description):
        termination_string = "Finish"
        description = description_and_solution[0]
        solution = description_and_solution[1]
        query = system_prompt + description

        # options = make_options(PICK_TARGETS, PLACE_TARGETS, termination_string=termination_string)
        options = [
            "[pick up red block]",
            "[pick up blue block]",
            "[pick up yellow block]",
            "[close the box]",
            "[open the box]",
            "[place the block to the desk]",
            "[place the block in the box]",
            "[Finish]"
        ]

        log = ""
        for i in range(7):
            raw_llm_scores, _, desc = gpt3_scoring(query, options, verbose=True, engine=ENGINE)

            max_score = -1e+6
            max_option = None
            llm_scores = {_key: _score for _key, _score in zip(possible_actions, raw_llm_scores)}
            combined_scores = {_action: llm_scores[_action] * affordance_scores[_action] for _action in possible_actions}
            combined_scores = normalize_scores(combined_scores)

            ## Select action
            command = max(combined_scores, key=combined_scores.get)
            actions.append(command)
            prompt += command + "\n"

        with open("./results/saycan/result_{}.txt".format(idx), 'w') as f:
            f.write(log + query + "\n\n" + "Solution: \n" + "\n".join(solution))

        break
