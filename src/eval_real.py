from pathlib import Path
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import ops
from ultralytics.utils.plotting import plot_images
import torch


class ReadableDetectionValidator(DetectionValidator):
    """Hide confidence in batch previews without changing evaluation."""

    def plot_predictions(self, batch, preds, ni, max_det=None):
        if not preds:
            return
        limit = max_det or self.args.max_det
        combined = {
            key: torch.cat([prediction[key][:limit] for prediction in preds], dim=0)
            for key in preds[0]
            if key != "batch_idx"
        }
        combined["batch_idx"] = torch.cat([
            torch.full_like(prediction["conf"][:limit], index)
            for index, prediction in enumerate(preds)
        ])
        combined["bboxes"] = ops.xyxy2xywh(combined["bboxes"])
        plot_images(
            images=batch["img"],
            labels=combined,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names={key: f"ID: {key + 1}" for key in self.names},
            on_plot=self.on_plot,
            show_conf=False,
        )


def main():
    model = YOLO("runs/ycbv/coco_head_only_9800/weights/best.pt")

    metrics = model.val(
        validator=ReadableDetectionValidator,
        data="data/yolo_ycbv_real/dataset.yaml",
        split="val",
        imgsz=640,
        batch=8,
        device=0,
        workers=2,
        conf=0.001,
        iou=0.50,
        plots=True,
        project=str(Path("runs/ycbv").resolve()),
        name="real_eval_coco_head_only_9800",
    )

    print("\nReal-image development results")
    print("mAP@0.50:", metrics.box.map50)
    print("mAP@0.50:0.95:", metrics.box.map)


if __name__ == "__main__":
    main()

