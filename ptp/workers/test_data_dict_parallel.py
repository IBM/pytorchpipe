
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import time

from ptp.components.problems.problem import Problem
from ptp.components.models.model import Model

from ptp.data_types.data_streams import DataStreams
from ptp.data_types.data_definition import DataDefinition
from ptp.utils.data_streams_parallel import DataStreamsParallel


class RandomDataset(Problem):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):

        # Return data_streams.
        data_streams = DataStreams({"index": None})
        data_streams["index"] = self.data[index]

        return data_streams

        #return self.data[index]

    def __len__(self):
        return self.len

    def output_data_definitions(self):
        return {"index": DataDefinition(1,1,"str")}

    def collate_fn(self, batch):
        print("Collate!")
        return DataStreams({key: torch.utils.data.dataloader.default_collate([sample[key] for sample in batch]) for key in batch[0]})


class TestModel1(Model):
    # Our model
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, datadict):
        input = datadict["index"]
        print("Dummy Model: input size {}, device: {}\n".format(input.size(), input.device))
        output = self.fc(input)
        print("Dummy Model: output size {}\n".format(output.size()))

        datadict.extend({"middle": output})
        #print("saved to output : ",type(output))
        #return output

    def input_data_definitions(self):
        return {"index": DataDefinition(1,1,"str")}

    def output_data_definitions(self):
        return {"middle": DataDefinition(1,1,"str")}


class TestModel2(Model):
    # Our model
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, datadict):
        input = datadict["middle"]

        print("Dummy Model: input size {}, device: {}\n".format(input.size(), input.device))
        output = self.fc(input)
        print("Dummy Model: output size {}\n".format(output.size()))

        datadict.extend({"output": output})

    def input_data_definitions(self):
        return {"middle": DataDefinition(1,1,"str")}

    def output_data_definitions(self):
        return {"output": DataDefinition(1,1,"str")}



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')

    use_dataparallel = False

    # Parameters and DataLoaders
    input_size = 5
    middle_size = 2
    output_size = 3

    batch_size = 10
    data_size = 100000

    model1 = TestModel1(input_size, middle_size)
    model2 = TestModel2(middle_size, output_size)
    print("Models DONE!!")
    #time.sleep(2)

    dataset = RandomDataset(input_size, data_size)
    rand_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    print("Dataloader DONE!!")
    #time.sleep(2)


    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model1 = DataStreamsParallel(model1) 
        model2 = DataStreamsParallel(model2) 
        use_dataparallel = True
    # Move to desired device.
    model1.to(device)
    model2.to(device)
    print("DataParallel DONE!!")
    #time.sleep(2)

    #datadict1 = {}#DataStreams({"index":None,"output":None})
    for datadict in rand_loader:
        print("!!!!!  Got object from loader: {}".format(type(datadict)))
        datadict.to(device)

        print("datadict before model1: ",datadict)
        if use_dataparallel:
            outputs = model1(datadict)
            # Postprocessing: copy only the outputs of the wrapped model.
            for key in model1.module.output_data_definitions().keys():
                datadict.extend({key: outputs[key]})
        else:
            model1(datadict)

        # Let's see what will happen here!
        datadict.to(device)

        print("datadict before model2: ",datadict)
        if use_dataparallel:
            outputs = model2(datadict)
            # Postprocessing: copy only the outputs of the wrapped model.
            for key in model2.module.output_data_definitions().keys():
                datadict.extend({key: outputs[key]})
        else: 
            model2(datadict)

        print("datadict after model2: ",datadict)

        output = datadict["output"]
        loss = output.sum()
        loss.backward()

        #print(type(output))
        #output = datadict2["output"]
        #print("For: after model: output_size ", output.size(),"\n")

