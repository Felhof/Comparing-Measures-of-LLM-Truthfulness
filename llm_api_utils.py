import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

endpoint_exposed = False
endpoint = None


def get_tokens_as_tuple(tokenizer, word):
    tokens = tuple(tokenizer([word], add_special_tokens=False).input_ids[0])
    return tuple([tokens[0]])

def model_to_repo_owner(name):
    if "llama" in name.lower():
        return "meta-llm"
    if "mistral" in name.lower():
        return "mistralai"
    raise ValueError("Unkown model. Only Llama and Mistral are supported.")


class LLMAPI:
    def __init__(self, name="Llama-2-7b-chat-hf", logit_bias=None, **model_kwargs) -> None:
        self.name = name

        repo_owner = model_to_repo_owner(name)
        self.tokenizer = AutoTokenizer.from_pretrained(f"{repo_owner}/{name}", padding_side="left")
        if "mistral" in name.lower():
            print("Setting Mistral pad token id")
            self.tokenizer.pad_token_id = 0
        self.model = AutoModelForCausalLM.from_pretrained(
            f"{repo_owner}/{name}",
            **model_kwargs    
        )

        if logit_bias is not None:
            sequence_bias = {}
            for word, bias in logit_bias.items():
                sequence_bias[get_tokens_as_tuple(self.tokenizer, word)] = bias
        else:
            sequence_bias = None

        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            sequence_bias=sequence_bias,
            **model_kwargs
        )

    def __call__(self, prompts, max_tokens=25, stop=None, return_logprobs=False, *args, **kwargs):

        if stop is not None:
            stop_token_id = self.tokenizer.convert_tokens_to_ids(
                [stop]
            )
        else:
            stop_token_id = [0]

        if type(prompts) is not list:
            prompts = [prompts]

        if return_logprobs:
            logprobtokens = [self.get_top_tokens(prompt) for prompt in prompts]

            return_dict = {'choices': [{
                "text": token[0],
                "logprobs": {
                    "tokens": token[0],
                    "top_logprobs": [token[1]]
            }} for token in logprobtokens]}
            # the 'logprobs' item should have a 'tokens' element and a 'top_logprobs' element. The first will contain
            # the produced token while the second will need to be a dictionary with the top 5 tokens and their logprobs.

            return return_dict

        if "sequence_bias" in kwargs:
            if "type" in kwargs:
                sequence_bias = kwargs["sequence_bias"][kwargs["type"]]
            else:
                sequence_bias = kwargs["sequence_bias"]
        else:
            sequence_bias = None
        
        out = self.generator(
            prompts,
            max_new_tokens=max_tokens,
            return_full_text=False,
            eos_token_id=stop_token_id,
            sequence_bias=sequence_bias
        )

        return {"choices": [{"text": o[0]["generated_text"].split(stop)[0]} for o in out]}

    def get_top_tokens(self, prompt):
        # Encode the prompt text
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

        # Perform a forward pass through the model and obtain the logits
        with torch.no_grad():
            outputs = self.model(input_ids.to('cuda:0'))
            logits = outputs.logits

        # logits contains the logits after each input token, so we only need to keep that after the last one in the
        # sequence. This are the logits for the first token in the completion:
        last_token_logits = logits[:, -1, :]  #

        # Perform a softmax to obtain the log-probabilities for each token
        last_token_logprobs = torch.nn.functional.log_softmax(last_token_logits, dim=-1)

        # Get the top 5 most likely tokens and their log probabilities
        top_k = 5
        top_log_probs, top_indices = torch.topk(last_token_logprobs, top_k, dim=-1)

        tokens = {}
        for i in range(top_k):
            token = self.tokenizer.decode(top_indices[0][i].item())
            log_prob = top_log_probs[0][i].item()
            # print(f"{i+1}. Token: {token}, Log Probability: {log_prob}")
            tokens[token] = log_prob

        out = max(tokens, key=tokens.get)

        return out, tokens
    

def establish_endpoint(name="Llama-2-7b-chat-hf", logit_bias=None, **model_kwargs):
    global endpoint_exposed
    global endpoint

    endpoint_exposed = True
    endpoint = LLMAPI(name, logit_bias=logit_bias, **model_kwargs)

    return endpoint


def get_llm_sequence_bias(bias):
    assert endpoint is not None, "Can only get sequence bias after initializing endpoint"
    sequence_bias = {}
    for word, bias in bias.items():
        sequence_bias[get_tokens_as_tuple(endpoint.tokenizer, word)] = bias
    return sequence_bias