import requests
import time
from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.model_handler.model_style import ModelStyle
from bfcl_eval.model_handler.utils import (
    func_doc_language_specific_pre_processing,
    default_decode_ast_prompting,
    default_decode_execute_prompting,
)
from overrides import override


class SOHandler(BaseHandler):
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        self.model_style = ModelStyle.OpenAI_Responses
        self.base_url = "http://localhost:8000"

    @override
    def _query_prompting(self, inference_data: dict):
        messages = inference_data["message"]
        functions = inference_data["function"]
        
        tools = [{"type": "function", "name": f["name"], "description": f["description"], "parameters": f["parameters"]} for f in functions]
        
        request_data = {"model": "gpt-4", "type": "start", "input": messages}
        if tools:
            request_data["tools"] = tools
            
        start_time = time.time()
        response = requests.post(f"{self.base_url}/v1/chat/completions", json=request_data, timeout=30)
        end_time = time.time()
        
        return response, end_time - start_time

    @override
    def _parse_query_response_prompting(self, api_response) -> dict:
        response_json = api_response.json()
        content = response_json["choices"][0]["message"]["content"] if "choices" in response_json else str(response_json)
        
        return {
            "model_responses": content,
            "input_token": response_json.get("usage", {}).get("prompt_tokens", 0),
            "output_token": response_json.get("usage", {}).get("completion_tokens", 0),
        }

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions = func_doc_language_specific_pre_processing(test_entry["function"], test_entry["id"].rsplit("_", 1)[0])
        return {"message": [], "function": functions}

    @override
    def add_first_turn_message_prompting(self, inference_data: dict, first_turn_message: list[dict]) -> dict:
        inference_data["message"].extend(first_turn_message)
        return inference_data

    @override
    def _add_next_turn_user_message_prompting(self, inference_data: dict, user_message: list[dict]) -> dict:
        inference_data["message"].extend(user_message)
        return inference_data

    @override
    def _add_assistant_message_prompting(self, inference_data: dict, model_response_data: dict) -> dict:
        inference_data["message"].append({"role": "assistant", "content": model_response_data["model_responses"]})
        return inference_data

    @override
    def _add_execution_results_prompting(self, inference_data: dict, execution_results: list[str], model_response_data: dict) -> dict:
        for result, decoded_response in zip(execution_results, model_response_data["model_responses_decoded"]):
            inference_data["message"].append({"role": "tool", "name": decoded_response, "content": result})
        return inference_data

    @override
    def decode_ast(self, result, language="Python"):
        return default_decode_ast_prompting(result, language)

    @override
    def decode_execute(self, result):
        return default_decode_execute_prompting(result)

    # FC methods - minimal implementations
    @override
    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        inference_data["message"] = []
        return inference_data

    @override
    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        functions = func_doc_language_specific_pre_processing(test_entry["function"], test_entry["id"].rsplit("_", 1)[0])
        tools = [{"type": "function", "name": f["name"], "description": f["description"], "parameters": f["parameters"]} for f in functions]
        inference_data["tools"] = tools
        return inference_data

    @override
    def _query_FC(self, inference_data: dict):
        messages = inference_data["message"]
        tools = inference_data.get("tools", [])
        
        # Tools only sent on first request (type: "start")
        is_first_request = len(messages) <= 1
        request_data = {
            "model": "gpt-4", 
            "type": "start" if is_first_request else "continue",
            "input": messages
        }
        
        if is_first_request and tools:
            request_data["tools"] = tools
            
        start_time = time.time()
        response = requests.post(f"{self.base_url}/v1/chat/completions", json=request_data, timeout=30)
        end_time = time.time()
        
        return response, end_time - start_time

    @override
    def _parse_query_response_FC(self, api_response) -> dict:
        return self._parse_query_response_prompting(api_response)

    @override
    def add_first_turn_message_FC(self, inference_data: dict, first_turn_message: list[dict]) -> dict:
        return self.add_first_turn_message_prompting(inference_data, first_turn_message)

    @override
    def _add_next_turn_user_message_FC(self, inference_data: dict, user_message: list[dict]) -> dict:
        return self._add_next_turn_user_message_prompting(inference_data, user_message)

    @override
    def _add_assistant_message_FC(self, inference_data: dict, model_response_data: dict) -> dict:
        return self._add_assistant_message_prompting(inference_data, model_response_data)

    @override
    def _add_execution_results_FC(self, inference_data: dict, execution_results: list[str], model_response_data: dict) -> dict:
        return self._add_execution_results_prompting(inference_data, execution_results, model_response_data)