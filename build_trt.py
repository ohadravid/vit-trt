import tensorrt as trt


def trt_from_onnx(onnx_path, trt_path):
    print(f"Converting ONNX to TRT: {onnx_path} -> {trt_path}")
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)

    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        parser.parse(f.read())

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)

    # Disable opts for quicker builds by setting to 0 or 1.
    config.builder_optimization_level = 3

    engine = builder.build_serialized_network(network, config)
    with open(trt_path, "wb") as f:
        f.write(engine)


if __name__ == "__main__":
    trt_from_onnx("./model.onnx", "./model.trt")
