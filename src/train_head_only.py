"""Fine-tune only the YOLOv8 Detect module on expanded rendered data."""
import argparse
from pathlib import Path

from ultralytics import YOLO

WEIGHTS = Path("yolov8n.pt")
DATASET = Path("data/yolo_ycbv_expanded/dataset.yaml")

def verify_trainable_parameters(trainer):
    trainable = [name for name, parameter in trainer.model.named_parameters()
                 if parameter.requires_grad]
    if not trainable or any(not name.startswith('model.22.') for name in trainable):
        raise RuntimeError('Expected only Detect module parameters to be trainable.')
    count = sum(parameter.numel() for parameter in trainer.model.parameters()
                if parameter.requires_grad)
    print(f'Head-only verification passed: {count:,} trainable parameters.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true', help='Inspect freezing without training.')
    args = parser.parse_args()
    model = YOLO(str(WEIGHTS))
    layers = model.model.model
    if len(layers) != 23 or type(layers[22]).__name__ != 'Detect':
        raise RuntimeError('Unexpected architecture; recheck freeze indices.')

    if args.check:
        total = sum(parameter.numel() for parameter in model.model.parameters())
        trainable = 0
        for name, parameter in model.model.named_parameters():
            enabled = name.startswith('model.22.') and '.dfl' not in name
            parameter.requires_grad_(enabled)
            if enabled:
                trainable += parameter.numel()
        print('Checkpoint:', WEIGHTS)
        print('Frozen: modules 0–21 (backbone and neck)')
        print('Trainable: module 22 (Detect), except fixed DFL projection')
        print(f'Trainable parameters: {trainable:,} / {total:,}')
        return

    if not DATASET.exists():
        raise FileNotFoundError(DATASET)

    model.add_callback('on_train_start', verify_trainable_parameters)
    model.train(
        data=str(DATASET.resolve()),
        freeze=22,
        epochs=50,
        patience=10,
        imgsz=640,
        batch=8,
        device=0,
        workers=2,
        cache=False,
        optimizer='AdamW',
        lr0=0.0004,
        momentum=0.9,
        weight_decay=0.0005,
        seed=42,
        close_mosaic=10,
        project=str(Path('runs/ycbv').resolve()),
        name='coco_head_only_9800',
        plots=True,
        save_period=-1,
    )


if __name__ == '__main__':
    main()
