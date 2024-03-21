from transformers import AutoModel, AutoTokenizer, BertConfig

def load_model(model_name, hyperparameters):
    # Create a config with the hyperparameters
    config = BertConfig.from_pretrained(model_name,
                                        **hyperparameters)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load the model with the config
    model = AutoModel.from_pretrained(model_name,
                                      config=config)

    return tokenizer, model

# Define the hyperparameters
hyperparameters = 
    'num_attention_heads': 12,
    'num_hidden_layers': 12,
    'hidden_size': 768,
    # Add more hyperparameters as needed
}

# Use the function to load a model with the hyperparameters
tokenizer, model = load_model('bert-base-uncased' hyperparameters)
