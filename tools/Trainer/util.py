def init_logger(log_file_path: str = None):
    import logging
    logging._warn_preinit_stderr = 0
    # 第一步，创建一个logger
    logger = logging.getLogger('PyTorchOCR')
    logger.setLevel(logging.DEBUG)  # Log等级总开关
    # 第二步，创建一个handler，用于写入日志文件
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if log_file_path is not None:
        file_handle = logging.FileHandler(log_file_path)
        file_handle.setFormatter(formatter)
        logger.addHandler(file_handle)
    logger.setLevel(logging.DEBUG)
    return logger
