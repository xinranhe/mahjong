PARAMETERS = {
    # Input params
    "training_data_patterns": [
        "/media/data/transformer-bert/tfrecord/201801*",
        "/media/data/transformer-bert/tfrecord/201802*",
        "/media/data/transformer-bert/tfrecord/201803*",
        "/media/data/transformer-bert/tfrecord/201804*",
        "/media/data/transformer-bert/tfrecord/201805*",
        "/media/data/transformer-bert/tfrecord/201806*",
        "/media/data/transformer-bert/tfrecord/201807*",
        "/media/data/transformer-bert/tfrecord/201808*",
        "/media/data/transformer-bert/tfrecord/201809*",
        "/media/data/transformer-bert/tfrecord/201810*",
        "/media/data/transformer-bert/tfrecord/201811*"
    ],
    "evaluation_data_patterns":[
        "/media/data/transformer-bert/tfrecord/201812*"
    ],
    # Training data loader properties
    "buffer_size": 10000,
    "num_parsing_threads": 16,
    "num_parallel_readers": 4,
    "prefetch_buffer_size": 1,
    "compression_type": "GZIP",

    # Model params
    "initializer_gain": 1.0,  # Used in trainable variable initialization.
    "hidden_size": 32, # Model dimension in the hidden layers, input embedding dimension
    "num_hidden_layers": 3, # Number of layers in the encoder stacks.
    "num_heads": 4,
    "filter_size": 256,
    "feature_hidden_size": [32,32],

    # Dropout values (only used when training)
    "layer_postprocess_dropout": 0.1,
    "attention_dropout": 0.1,
    "relu_dropout": 0.1,

    # Training params
    "learning_rate": 0.1,
    "learning_rate_decay_rate": 1.0,
    "learning_rate_warmup_steps": 16000,

    # Optimizer params
    "optimizer_adam_beta1": 0.9,
    "optimizer_adam_beta2": 0.997,
    "optimizer_adam_epsilon": 1e-09,

    # batch size
    "batch_size": 1024,

    # Training and evaluation parameters
    "num_gpus": 4,
    "model_dir": "/tmp/model",
    "num_train_steps": 2000000,
    "num_eval_steps": 1000,

    # Params for transformer TPU
    "allow_ffn_pad": True,
}
