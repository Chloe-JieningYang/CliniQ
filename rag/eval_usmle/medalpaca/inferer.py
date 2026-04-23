import sys
import json
from pathlib import Path
import torch
from peft import PeftModel
from transformers import GenerationConfig, BitsAndBytesConfig, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
from .handler import DataHandler

assert torch.cuda.is_available(), "No cuda device detected"


class Inferer:
    """
    A basic inference class for accessing medAlpaca models programmatically.

    This class provides methods for loading supported medAlpaca models, tokenizing inputs,
    and generating outputs based on the specified model and configurations.

    Attributes:
        available_models (dict): A dictionary containing the supported models and their configurations.

    Args:
        model_name (str): The name of the medAlpaca model to use for inference.
        prompt_template (str): The path to the JSON file containing the prompt template.
        base_model (str, optional): If LoRA is used, this should point to the bases model weigts
        model_max_length: (int, optional): Number of input tokens to the model. Default is 512.
        load_in_8bit (bool, optional): Wether a quantized model should be loaded. Default is False
        torch_dtype (torch.dtype, optional): The torch datatype to load the base model. Default is float16
        peft (bool, optional): If the model was trainied in 8bit or with LoRA, PEFT library should be used
            to load the model. Default is False. 

    Example:

        medalpaca = medAlapaca("medalpaca/medalapca-7b", "prompts/alpaca.json")
        response = medalpaca(input="What is Amoxicillin?")
    """
        
    def __init__(
        self, 
        model_name: str, 
        prompt_template: str,
        base_model: str = None,
        model_max_length: int = 512,
        load_in_8bit: bool = False, 
        torch_dtype: torch.dtype = torch.float16, 
        peft: bool = False
    ) -> None:
        
        self.model, self.tokenizer = self._load_model(
            model_name, 
            load_in_8bit=load_in_8bit, 
            load_in_4bit=False, 
        )
        
        self.data_handler = DataHandler(
            self.tokenizer,
            prompt_template = prompt_template, 
            model_max_length = model_max_length,
            train_on_inputs = False,
        )
    
    def _load_model(self, path: str, load_in_4bit: bool = False, load_in_8bit: bool = False):
        ### Copied from evaluate_dpo.py ###

        path = str(Path(path).resolve())
        is_lora = (Path(path) / "adapter_config.json").exists()

        bnb = None
        if load_in_4bit:
            bnb = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            )
        elif load_in_8bit:
            bnb = BitsAndBytesConfig(load_in_8bit=True)

        dtype = torch.float16 if not (load_in_4bit or load_in_8bit) else None

        if is_lora:
            from peft import PeftModel
            from accelerate import dispatch_model, infer_auto_device_map

            cfg = json.loads((Path(path) / "adapter_config.json").read_text())
            base = cfg.get("base_model_name_or_path", "")
            if not base:
                sys.exit(f"Cannot infer base model from {path}/adapter_config.json")
            base = str(Path(base).resolve()) if base.startswith(".") else base

            model = AutoModelForCausalLM.from_pretrained(
                base, quantization_config=bnb, torch_dtype=dtype,
                device_map=None, trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(model, path, is_trainable=False)
            model = model.merge_and_unload()

            # Now it is a plain nn.Module — safe to dispatch across GPUs/CPU.
            # Pass class names as plain strings; avoids the set-hashing bug in
            # accelerate <= 0.29 where sets were passed instead of lists.
            no_split = [
                "LlamaDecoderLayer", "MistralDecoderLayer",
                "Qwen2DecoderLayer", "FalconDecoderLayer",
                "GPTNeoXLayer", "BloomBlock",
            ]
            device_map = infer_auto_device_map(
                model, no_split_module_classes=no_split,
            )
            model = dispatch_model(model, device_map=device_map)

            tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                path, quantization_config=bnb, torch_dtype=dtype,
                device_map="auto", trust_remote_code=True,
            )
            tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        model.eval()
        return model, tok
    
    def _generate_batch(self, model, tokenizer, prompts: list,
        max_new_tokens: int = 200,
        temperature: float = 0.1,
        top_p: float = 0.75,
    ) -> list:
        """Generate responses for a batch of prompts in a single forward pass."""
        enc = tokenizer(
            prompts, return_tensors="pt",
            padding=True, truncation=True, max_length=1024,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        input_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        # Decode only the newly generated tokens for each item in the batch
        return [
            tokenizer.decode(out[i, input_len:], skip_special_tokens=True).strip()
            for i in range(len(prompts))
        ]
            
    def __call__(
        self,
        input: str,
        instruction: str = None,
        output: str = None,
        max_new_tokens: int = 128,
        verbose: bool = False,
        **generation_kwargs,
    ) -> str:
        """
        Generate a response from the medAlpaca model using the given input and instruction.

        Args:
            input (str):
                The input text to provide to the model.
            instruction (str, optional):
                An optional instruction to guide the model's response.
            output (str, optional): 
                Prepended to the models output, e.g. for 1-shot prompting
            max_new_tokens (int, optional): 
                How many new tokens the model can generate
            verbose (bool, optional): 
                If True, print the prompt before generating a response.
            **generation_kwargs:
                Keyword arguments to passed to the `GenerationConfig`.
                See here for possible arguments: https://huggingface.co/docs/transformers/v4.20.1/en/main_classes/text_generation

        Returns:
            str: The generated response from the medAlpaca model.
        """

        prompt = self.data_handler.generate_prompt(instruction = instruction, input = input, output = output)
        if verbose:
            print(prompt)

        response = self._generate_batch(self.model, self.tokenizer, [prompt],
                          max_new_tokens, temperature=0.1, top_p=0.75)[0]
        return response