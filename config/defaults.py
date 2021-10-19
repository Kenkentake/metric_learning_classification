from yacs.config import CfgNode


_C = CfgNode()

# Train Params
_C.TRAIN = CfgNode()
_C.TRAIN.BATCH_SIZE = 256
_C.TRAIN.LEARNING_RATE = 1e-4
_C.TRAIN.MODEL_TYPE = ''
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.NUM_EPOCHS = 1
_C.TRAIN.NUM_GPUS = 1
_C.TRAIN.OPTIMIZER_TYPE = ''
_C.TRAIN.RUN_NAME
_C.TRAIN.SEED = 42

_C.DATA = CfgNode()
_C.DATA.IMG_SIZE = (32, 32)
_C.DATA.ROOT_PATH = './dataset'
_C.DATA.TRANSFORM_LIST = ['resize', 'to_tensor']
_C.DATA.NUM_WORKERS = 32
