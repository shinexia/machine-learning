#!/usr/bin/env python3

import argparse
import json
import logging
import os
import torch
import gradio as gr
import torchvision.transforms as transforms
from collections import OrderedDict
from typing import List

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)s - %(message)s",
    level=logging.INFO,
)


class App(object):
    def __init__(self, model: torch.nn.Module, classes: List[str], topk: int,
                 device: torch.device):
        self.model = model
        self.classes = classes
        self.topk = topk
        self.device = device
        self.preprocessor = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def predict(self, img):
        img = self.preprocessor(img)
        batch = torch.unsqueeze(img, 0).to(self.device)

        with torch.no_grad():
            outputs = self.model(batch)
            outputs = torch.nn.functional.softmax(outputs, dim=1)

        output = outputs[0]
        indices = torch.topk(output, k=self.topk).indices
        confidences = {self.classes[i]: float(output[i]) for i in indices}

        return confidences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workdir', '-w', default='', help='')
    parser.add_argument('--model', '-m', default='models/model.pt', help='')
    parser.add_argument('--topk', '-k', default=10, help='')
    FLAGS = parser.parse_args()
    logging.info('FLAGS: %s', json.dumps(FLAGS.__dict__, ensure_ascii=False))
    workdir = FLAGS.workdir
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logging.info('using device: %s', device)

    model_file: str = FLAGS.model
    if not model_file.startswith(('/', './', '../')):
        model_file = os.path.join(workdir, model_file)
    logging.info('loading model: %s', model_file)
    model = torch.jit.load(model_file)
    model = model.eval().to(device)

    classes_file = os.path.join(workdir, 'classes.json')
    logging.info("loading classes: %s", classes_file)
    with open(classes_file, 'rb') as f:
        classes = [str(v) for v in OrderedDict(json.loads(f.read())).values()]

    inference = App(model=model,
                    classes=classes,
                    topk=FLAGS.topk,
                    device=device)
    examples = [
        os.path.join(workdir, a)
        for a in ['tench.jpeg', 'goldfish.jpeg', 'shark.jpeg']
    ]
    server = gr.Interface(
        fn=inference.predict,
        inputs=gr.components.Image(type="pil"),
        outputs=gr.components.Label(num_top_classes=5),
        title='Gradio',
        description="Gradio demo for image classification",
        article="",
        examples=examples,
        analytics_enabled=False,
    )

    logging.info('launching server at 0.0.0.0:8000')
    server.launch(debug=True,
                  enable_queue=True,
                  server_name='0.0.0.0',
                  server_port=8000)


if __name__ == '__main__':
    main()
