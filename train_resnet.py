import os
import numpy as np
from argparse import ArgumentParser
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
from evaluate import load as load_metric
from PIL import Image
from sklearn.model_selection import train_test_split

def create_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cpu",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed used for training"
    )
    parser.add_argument(
        "--hf-token", type=str, default=None, help="Hugging face access token"
    )

    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to initial dataset"
    )

    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[512, 512],
        help="Size of images",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
    )
    parser.add_argument("--num-workers", type=int, default=1)

    parser.add_argument(
        "--save-dir", type=str, required=True, help="Where to save the model"
    )
    parser.add_argument(
        "--num-epochs", type=int, default=5, help="Number of epochs"
    )
    return parser.parse_args()

def load_images(image_dir, label, size=(224, 224)):
    images = []
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(image_dir, filename)
            image = Image.open(img_path).convert('RGB')
            image = image.resize(size, Image.LANCZOS)
            images.append({'image': image, 'label': label})
    return images

def main(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    dataset_dir = args.dataset
    real_images_dir = os.path.join(dataset_dir, 'reals')
    fake_images_dir = os.path.join(dataset_dir, 'fakes')

    real_images = load_images(real_images_dir, 0)
    fake_images = load_images(fake_images_dir, 1)
    
    all_images = real_images + fake_images

    # Split dataset with 82% for training and 18% for testing
    train_images, test_images = train_test_split(all_images, test_size=0.2, random_state=args.seed)

    # Convert to Dataset
    train_dataset = Dataset.from_dict({
        'image': [img['image'] for img in train_images],
        'label': [img['label'] for img in train_images],
    })

    test_dataset = Dataset.from_dict({
        'image': [img['image'] for img in test_images],
        'label': [img['label'] for img in test_images],
    })

    dataset = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })

    processor = AutoImageProcessor.from_pretrained('microsoft/resnet-50', trust_remote_code=True)
    model = AutoModelForImageClassification.from_pretrained('microsoft/resnet-50', num_labels=2, ignore_mismatched_sizes=True, trust_remote_code=True)

    def transform(example_batch):
        inputs = processor(images=example_batch['image'], return_tensors='pt')
        inputs['label'] = torch.tensor(example_batch['label'])
        return inputs

    dataset = dataset.map(transform, batched=True)

    accuracy = load_metric("accuracy")

    def compute_metrics(p):
        return accuracy.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

    training_args = TrainingArguments(
        output_dir=args.save_dir,
        eval_strategy="epoch",  # Thay đổi từ evaluation_strategy thành eval_strategy
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        save_steps=10_000,
        save_total_limit=2,
        seed=args.seed
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Đánh giá mô hình trên tập test
    results = trainer.evaluate(eval_dataset=dataset['test'])

    # Lưu kết quả đánh giá
    with open(os.path.join(args.save_dir, "test_results.txt"), "w") as writer:
        for key, value in results.items():
            writer.write(f"{key}: {value}\n")

if __name__ == "__main__":
    args = create_args()
    main(args)
