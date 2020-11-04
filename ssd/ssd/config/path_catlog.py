import os


class DatasetCatalog:
    # DATA_DIR = 'datasets'
    # DATA_DIR = 'F:/datas/VOC/VOC_ROOT'
    # DATA_DIR = 'F:/datas/VOC/VisDrone_ROOT'
    DATA_DIR = '/mnt/data/vehicle_reid/ssd/data/voc/VisDrone_ROOT/'
    # DATA_DIR = '/home/chenmingsong/coding_data/coco/VOC_ROOT/VOC2012'
    DATASETS = {
        'VisDrone_2019_train': {
            "data_dir": "DET2019",
            "split": "train"
        },
        'VisDrone_2019__val': {
            "data_dir": "DET2019",
            "split": "val"
        },
        'VisDrone_2019__trainval': {
            "data_dir": "DET2019",
            "split": "trainval"
        },
        'VisDrone_2019__test': {
            "data_dir": "DET2019",
            "split": "test"
        },
        'voc_2007_train': {
            "data_dir": "VOC2007",
            "split": "train"
        },
        'voc_2007_val': {
            "data_dir": "VOC2007",
            "split": "val"
        },
        'voc_2007_trainval': {
            "data_dir": "VOC2007",
            "split": "trainval"
        },
        'voc_2007_test': {
            "data_dir": "VOC2007",
            "split": "test"
        },
        'voc_2012_train': {
            "data_dir": "VOC2012",
            "split": "train"
        },
        'voc_2012_val': {
            "data_dir": "VOC2012",
            "split": "val"
        },
        'voc_2012_trainval': {
            "data_dir": "VOC2012",
            "split": "trainval"
        },
        'voc_2012_test': {
            "data_dir": "VOC2012",
            "split": "test"
        },
        'coco_2014_valminusminival': {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_valminusminival2014.json"
        },
        'coco_2014_minival': {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_minival2014.json"
        },
        'coco_2014_train': {
            "data_dir": "train2014",
            "ann_file": "annotations/instances_train2014.json"
        },
        'coco_2014_val': {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_val2014.json"
        },
    }

    @staticmethod
    def get(name):
        if 'VisDrone' in name:
            visdron_root = DatasetCatalog.DATA_DIR
            # voc_root = DatasetCatalog.DATA_DIR
            if 'VisDrone_ROOT' in os.environ:
                voc_root = os.environ['VisDrone_ROOT']
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(visdron_root, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(factory="VisDroneDataset", args=args)

        elif "voc" in name:
            voc_root = DatasetCatalog.DATA_DIR
            if 'VOC_ROOT' in os.environ:
                voc_root = os.environ['VOC_ROOT']
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(voc_root, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(factory="VOCDataset", args=args)

        elif "coco" in name:
            coco_root = DatasetCatalog.DATA_DIR
            if 'COCO_ROOT' in os.environ:
                coco_root = os.environ['COCO_ROOT']

            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(coco_root, attrs["data_dir"]),
                ann_file=os.path.join(coco_root, attrs["ann_file"]),
            )
            return dict(factory="COCODataset", args=args)

        raise RuntimeError("Dataset not available: {}".format(name))
