from transformers import AutoConfig

model_name = "nvidia/Alpamayo-R1-10B"
try:
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    print(f"Config for {model_name}:")
    
    # Check text config
    if hasattr(config, "text_config"):
        print(f"Text Config Hidden Size: {getattr(config.text_config, 'hidden_size', 'N/A')}")
    elif hasattr(config, "hidden_size"):
        print(f"Hidden Size: {config.hidden_size}")
        
    # Check vision config
    if hasattr(config, "vision_config"):
        print(f"Vision Config Hidden Size: {getattr(config.vision_config, 'hidden_size', 'N/A')}")
        
except Exception as e:
    print(f"Error loading config: {e}")
