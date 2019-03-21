PARAMETERS = {
    # Input params
    "training_data_patterns": [
        "data/tfrecord/201801*",
        "data/tfrecord/201802*",
        "data/tfrecord/201803*",
        "data/tfrecord/201804*",
        "data/tfrecord/201805*",
        "data/tfrecord/201806*",
        "data/tfrecord/201807*",
        "data/tfrecord/201808*",
        "data/tfrecord/201809*",
        "data/tfrecord/201810*",
        "data/tfrecord/201811*",
    ],
    "evaluation_data_patterns":[
        "data/tfrecord/201812*"
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
    "num_hidden_layers": 6, # Number of layers in the encoder stacks.
    "num_heads": 8,
    "filter_size": 512,
    "feature_hidden_size": [32,32],

    # Dropout values (only used when training)
    "layer_postprocess_dropout": 0.1,
    "attention_dropout": 0.1,
    "relu_dropout": 0.1,

    # Training params
    "learning_rate": 0.01,
    "learning_rate_decay_rate": 1.0,
    "learning_rate_warmup_steps": 16000,

    # Optimizer params
    "optimizer_adam_beta1": 0.9,
    "optimizer_adam_beta2": 0.997,
    "optimizer_adam_epsilon": 1e-09,

    # batch size
    "batch_size": 256,

    # Training and evaluation parameters
    "model_dir": "/home/xinran/GitProject/mahjong/training/20190319",
    "num_train_steps": 1000000,
    "num_eval_steps": 1000,

    # Params for transformer TPU
    "allow_ffn_pad": True,
}
