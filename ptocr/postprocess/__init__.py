from .db_postprocess import DBPostProcess
import copy

def build_post_processing(config):
    config = copy.deepcopy(config)
    post_processing_name = config.pop("name")
    try:
        post_processing = eval(post_processing_name)(config)
        return post_processing
    except:
        return None
