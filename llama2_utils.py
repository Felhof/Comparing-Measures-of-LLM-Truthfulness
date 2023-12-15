import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, pipeline

llama_endpoint_exposed = False
llama_endpoint = None


class Llama2API:
    def __init__(self, name="Llama-2-7b-chat-hf") -> None:
        self.name = name

        self.tokenizer = LlamaTokenizer.from_pretrained(f"meta-llama/{name}")
        self.model = LlamaForCausalLM.from_pretrained(f"meta-llama/{name}")

        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device="cuda:0",
        )

    def __call__(self, prompts, max_tokens=25, stop=None, return_logprobs=False, *args, **kwargs):

        if stop is not None:
            stop_token_id = llama_endpoint.tokenizer.convert_tokens_to_ids(
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

        out = self.generator(
            prompts,
            max_new_tokens=max_tokens,
            return_full_text=False,
            eos_token_id=stop_token_id,
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


def establish_llama2_endpoint(name="Llama-2-7b-chat-hf"):
    global llama_endpoint_exposed
    global llama_endpoint

    llama_endpoint_exposed = True
    llama_endpoint = Llama2API(name)

    return llama_endpoint