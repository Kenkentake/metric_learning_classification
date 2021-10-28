from models.arcface_model import ArcfaceModel
from models.cnn_model import CNNModel
from models.triplet_net_model import TripletNetModel

def select_model(args, device):
    model_type = args.TRAIN.MODEL_TYPE
    model_dict = {
            "arcface": ArcfaceModel,
            "cnn": CNNModel,
            "triplet_net": TripletNetModel,
            }
    
    if model_type not in model_dict:
        print("Selected model type is not in model_dict")
        raise NotImplementedError()

    model = model_dict[model_type](args, device)
    return model
