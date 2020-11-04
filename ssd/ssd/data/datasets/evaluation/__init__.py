from ssd.data.datasets import VOCDataset, COCODataset, VisDroneDataset
from .coco import coco_evaluation
from .voc import voc_evaluation
from .visdrone import visdrone_evaluation


def evaluate(dataset, predictions, output_dir, **kwargs):
    """evaluate dataset using different methods based on dataset type.
    Args:
        dataset: Dataset object
        predictions(list[(boxes, labels, scores)]): Each item in the list represents the
            prediction results for one image. And the index should match the dataset index.
        output_dir: output folder, to save evaluation files or results.
    Returns:
        evaluation result
    """
    args = dict(
        dataset=dataset, predictions=predictions, output_dir=output_dir, **kwargs,
    )
    if isinstance(dataset, VOCDataset):
        return voc_evaluation(**args)
    elif isinstance(dataset, COCODataset):
        return coco_evaluation(**args)
    elif isinstance(dataset, VisDroneDataset):
        return visdrone_evaluation(**args)
    else:
        raise NotImplementedError
