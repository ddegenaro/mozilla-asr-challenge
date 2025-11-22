import torch
"""
	Code from https://github.com/mlfoundations/task_vectors/blob/main/src/task_vectors.py 
	From the original ICLR 2023 paper Editing Models with Task Arithmetic, by Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Suchin Gururangan, Ludwig Schmidt, Hannaneh Hajishirzi and Ali Farhadi
""" 
class TaskVector():
    def __init__(self, pretrained_model=None, finetuned_model=None, vector=None):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.
        
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_model is not None and finetuned_model is not None
            with torch.no_grad():
                pretrained_state_dict = pretrained_model.state_dict()
                finetuned_state_dict = finetuned_model.state_dict()
                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]
    
    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVector(vector=new_vector)

    def apply_to(self, pretrained_model, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
        pretrained_model.load_state_dict(new_state_dict, strict=False)
        return pretrained_model


    def tv_to_vector(self):
        """
            turns task vector into a flattened vector
        """
        
        # Extract all parameter tensors and flatten each one
        flattened_tensors = [torch.flatten(param) for param in self.vector.values()]

        # Concatenate all flattened tensors into a single 1D vector
        single_vector = torch.cat(flattened_tensors)

        return single_vector
    
    def tv_to_layer_wise(self):
        """
            returns matrix of weights grouped layer-wise
        """
        layer_groupings = []
        layer_strings = []
        curr_layer_str = ""
        curr_layer = []
        for key in self.vector.keys():
            if len(self.vector[key]) > 0:
                split_key = key.split(".")
                if split_key[0] == "model":
                    if split_key[1] == "encoder" or split_key[1] == "decoder":
                        if split_key[2] != "layers":
                            if split_key[2] == curr_layer_str:
                                curr_layer.append(torch.flatten(self.vector[key]))
                            else:
                                if len(curr_layer) > 0:
                                    layer_groupings.append(torch.cat(curr_layer))
                                    layer_strings.append(curr_layer_str)
                                curr_layer = []
                                curr_layer_str = split_key[2] 
                                curr_layer.append(torch.flatten(self.vector[key]))
                        else:
                            if split_key[3] == curr_layer_str:
                                curr_layer.append(torch.flatten(self.vector[key]))
                            else:
                                if len(curr_layer) > 0:
                                    layer_groupings.append(torch.cat(curr_layer))
                                    layer_strings.append(curr_layer_str)

                                curr_layer = []
                                curr_layer_str = split_key[3] 
                                curr_layer.append(torch.flatten(self.vector[key]))
                    
                else:
                    if len(curr_layer) > 0:
                        layer_groupings.append(torch.cat(curr_layer))
                        layer_strings.append(curr_layer_str)
                    curr_layer = []
                    curr_layer_str =  key    
                    curr_layer.append(torch.flatten(self.vector[key]))
        layer_groupings.append(torch.cat(curr_layer))
        layer_strings.append(curr_layer_str)
        return layer_groupings, layer_strings
        
