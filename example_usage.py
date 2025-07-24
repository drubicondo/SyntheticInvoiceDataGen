from synthetic_invoice_payment_generator import SyntheticDataGenerator, DEFAULT_CONFIG

# Configure your Azure OpenAI credentials
AZURE_ENDPOINT = "https://your-resource.openai.azure.com/"
# Ensure AZURE_OPENAI_API_KEY is set in your environment, e.g.:
# export AZURE_OPENAI_API_KEY="your-api-key"

# Initialize generator
generator = SyntheticDataGenerator(DEFAULT_CONFIG, AZURE_ENDPOINT)

# Generate dataset
dataset = generator.generate_dataset()

# Export in multiple formats
generator.export_dataset(dataset, "./synthetic_data")
