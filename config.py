class Settings:
    _base_model_route = "params/{}/{}"
    _base_data_route = "data/{}"

    MODEL_POINT_DECODER = _base_model_route.format("point_decoder_vith.pth")
    MODEL_POINT_DECODER_CNN = _base_data_route.format("point_decoder_cnn_vith.pth")
    MODEL_CLIP_ZERO_SHOT = _base_model_route.format("MLP_small_box_w1_zeroshot.tar")

    TRAIN_IMAGE = _base_data_route.format("train", "images")
    TRAIN_GT = _base_data_route.format("train", "ground_truth")
    TEST_IMAGE = _base_data_route.format("test", "images")
    TEST_GT = _base_data_route.format("test", "ground_truth")


settings = Settings()
