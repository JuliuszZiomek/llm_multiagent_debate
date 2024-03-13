import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
import json
import numpy as np
import time
import pickle
from tqdm import tqdm



def parse_bullets(sentence):
    bullets_preprocess = sentence.split("\n")
    bullets = []

    for bullet in bullets_preprocess:
        try:
            idx = bullet.find(next(filter(str.isalpha, bullet)))
        except:
            continue

        bullet = bullet[idx:]

        if len(bullet) != 0:
            bullets.append(bullet)

    return bullets


def generate_answer(answer_context, model, tokenizer):

    model_inputs = tokenizer.apply_chat_template(answer_context, return_tensors="pt").to("cuda")
    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    completion = tokenizer.batch_decode(generated_ids)[0]

    return completion


def construct_message(agents, question, idx, stubborn=False):

    # Use introspection in the case in which there are no other agents.
    if len(agents) == 0:
        return {"role": "user", "content": "Can you verify that your answer is correct. Please reiterate your answer, making sure to state your answer at the end of the response."}

    prefix_string = "These are the recent/updated opinions from other agents: "

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent response: ```{}```".format(agent_response)

        prefix_string = prefix_string + response
    
    if stubborn:
        prefix_string = prefix_string + "\n\n Use these opinions carefully as additional advice, however, keep in my mind that they can be wrong and your original answer is likely to be correct. Can you provide an updated answer? Make sure to state your answer at the end of the response.".format(question)
    else:
        prefix_string = prefix_string + "\n\n Use these opinions carefully as additional advice, can you provide an updated answer? Make sure to state your answer at the end of the response.".format(question)
    return {"role": "user", "content": prefix_string}


def construct_assistant_message(completion):
    content = completion
    return {"role": "assistant", "content": content}

def parse_answer(sentence):
    sentence = sentence.replace("\n", " ")
    parts = sentence.split(" ")

    for part in parts[::-1]:
        if part.endswith("."):
            part = part[:-1]
        try:
            answer = float(part)
            return answer
        except:
            continue


def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        current_frequency = List.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_agents", default=2)
    parser.add_argument("--n_rounds", default=2)
    parser.add_argument("--model", default='both')
    parser.add_argument("--stubborn", action="store_true")
    args = parser.parse_args()

    if args.model == "mistral":
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    elif args.model == "llama":
        model_name = "meta-llama/Llama-2-7b-chat-hf"
    elif args.model == "both":
        model_name = ["mistralai/Mistral-7B-Instruct-v0.2", "meta-llama/Llama-2-7b-chat-hf"]
    else:
        raise ValueError()

    if type(model_name) == str:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        model = [AutoModelForCausalLM.from_pretrained(mn, device_map="cpu", torch_dtype=torch.float16) for mn in model_name]
        tokenizer = [AutoTokenizer.from_pretrained(mn) for mn in model_name]

    answer = parse_answer("My answer is the same as the other agents and AI language model: the result of 12+28*19+6 is 550.")

    agents = int(args.n_agents)
    rounds = int(args.n_rounds)

    if agents > 1:
        assert rounds > 1

    evaluation_round = 100
    scores = []

    generated_description = {}

    for round in tqdm(range(evaluation_round)):
        a, b, c, d, e, f = np.random.randint(0, 30, size=6)

        answer = a + b * c + d - e * f
        agent_contexts = [[{"role": "user", "content": """What is the result of {}+{}*{}+{}-{}*{}? Make sure to state your answer at the end of the response.""".format(a, b, c, d, e, f)}] for agent in range(agents)]

        content = agent_contexts[0][0]['content']
        question_prompt = "We seek to find the result of {}+{}*{}+{}-{}*{}?".format(a, b, c, d, e, f)

        for round in range(rounds):
            for i, agent_context in enumerate(agent_contexts):

                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    message = construct_message(agent_contexts_other, question_prompt, 2*round - 1, stubborn=args.stubborn)
                    agent_context.append(message)

                    print("message: ", message)

                if type(model_name) == str:
                    completion = generate_answer(agent_context, model, tokenizer)
                else:
                    model[i%2].to("cuda")
                    completion = generate_answer(agent_context, model[i%2], tokenizer[i%2])
                    model[i%2].to("cpu")

                assistant_message = construct_assistant_message(completion)
                agent_context.append(assistant_message)
                print(completion)

        text_answers = []

        for agent_context in agent_contexts:
            text_answer = string =  agent_context[-1]['content']
            text_answer = text_answer.replace(",", ".")

            if text_answer.endswith("</s>"):
                text_answer = text_answer[:-4]
            
            if text_answer.endswith("."):
                text_answer = text_answer[:-1]

            text_answer = text_answer.replace("**", " ")

            text_answer = parse_answer(text_answer)

            if text_answer is None:
                continue

            text_answers.append(text_answer)

        generated_description[(a, b, c, d, e, f)] = (agent_contexts, answer)

        try:
            print("Answers:", text_answers)
            text_answer = most_frequent(text_answers)
            if text_answer == answer:
                scores.append(1)
            else:
                scores.append(0)
        except:
            continue

        print("performance:", np.mean(scores), np.std(scores) / (len(scores) ** 0.5))

    prefix = "stubborn_" if args.stubborn else ""

    pickle.dump(generated_description, open(prefix + "{}_math_agents{}_rounds{}.p".format(args.model, agents, rounds), "wb"))
    results_file = open(prefix + f"{args.model}_math_agents{agents}_rounds{rounds}.txt", "w")
    results_file.write(f"performance:, {np.mean(scores)}, {np.std(scores) / (len(scores) ** 0.5)}")
    results_file.close()
