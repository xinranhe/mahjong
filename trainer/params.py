PARAMETERS = {
    # Input params
    "training_data_patterns": [
        "data/tfrecord/20180101.gz"
    ],
    "evaluation_data_patterns":[
        "data/tfrecord/20180101.gz"
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
    "learning_rate": 2.0,
    "learning_rate_decay_rate": 1.0,
    "learning_rate_warmup_steps": 16000,

    # Optimizer params
    "optimizer_adam_beta1": 0.9,
    "optimizer_adam_beta2": 0.997,
    "optimizer_adam_epsilon": 1e-09,

    # batch size
    "batch_size": 32,

    # Training and evaluation parameters
    "model_dir": "/tmp/model",
    "num_train_steps": 10000,
    "num_eval_steps": 1000,

    # Params for transformer TPU
    "allow_ffn_pad": True,
}
