import torch
import numpy as np


coefficients = {
    "DEFAULT": [-1.12343328e+02,  1.50680483e+02, -5.15023303e+01,  6.24892431e+00, 6.85022158e-02],
}


class TeaCache:
    def __init__(self, num_inference_steps, model_name, treshold=0.3, start_step_treshold=0.1, end_step_treshold=0.9):
        self.input_bank = []
        self.current_step = 0
        self.accumulated_distance = 0.0
        self.num_inference_steps = num_inference_steps * 2
        self.start_step_teacache = int(num_inference_steps * start_step_treshold) * 2
        self.end_step_teacache = int(num_inference_steps * end_step_treshold) * 2
        self.treshold = treshold # [0.3, 0.5, 0.7, 0.9]
        self.coefficients = coefficients[model_name]
        self.step_name = "even"
        self.init_memory()

    def init_memory(self):
        self.accumulated_distance = {
            "even": 0.0,
            "odd": 0.0,
        }
        self.flow_direction = {
            "even": None,
            "odd": None,
        }
        self.previous_modulated_input = {
            "even": None,
            "odd": None,
        }
        # print("TEACACHE MEMORY HAS BEEN CREATED")

    def check_for_using_cached_value(self, modulated_input):
        use_tea_cache = (self.treshold > 0.0) and (self.start_step_teacache <= self.current_step < self.end_step_teacache)
        self.step_name = "even" if self.current_step % 2 == 0 else "odd"

        use_cached_value = False 
        if use_tea_cache:
            rescale_func = np.poly1d(self.coefficients)
            current_disntace = rescale_func(
                self.calculate_distance(modulated_input, self.previous_modulated_input[self.step_name])
            )
            self.accumulated_distance[self.step_name] += current_disntace
            
            if self.accumulated_distance[self.step_name] < self.treshold:
                use_cached_value = True
            else:
                use_cached_value = False
                self.accumulated_distance[self.step_name] = 0.0
        
        if self.step_name == "even":
            self.input_bank.append(modulated_input.cpu())

        self.previous_modulated_input[self.step_name] = modulated_input.clone()
        # if use_tea_cache:
        #     print(f"[ STEP:{self.current_step} | USE CACHED VALUE: {use_cached_value} | ACCUMULATED DISTANCE: {self.accumulated_distance} | CURRENT DISTANCE: {current_disntace} ]")
        return use_cached_value
    
    def use_cache(self, hidden_states):
        return hidden_states + self.flow_direction[self.step_name].to(device=hidden_states.device)

    def calculate_distance(self, previous_tensor, current_tensor):
        relative_l1_distance = torch.abs(
            previous_tensor - current_tensor
        ).mean() / torch.abs(previous_tensor).mean()
        return relative_l1_distance.to(torch.float32).cpu().item()

    def update(self, flow_direction):
        self.flow_direction[self.step_name] = flow_direction
        self.current_step += 1
        if self.current_step == self.num_inference_steps:
            self.current_step = 0
            self.init_memory()