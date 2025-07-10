from synthetic_invoice_payment_generator import SyntheticDataGenerator, DEFAULT_CONFIG

# Configure your Azure OpenAI credentials
AZURE_ENDPOINT = "https://your-resource.openai.azure.com/"
API_KEY = "your-api-key"

# Initialize generator
generator = SyntheticDataGenerator(DEFAULT_CONFIG, AZURE_ENDPOINT, API_KEY)

# Generate dataset
dataset = generator.generate_dataset(total_size=5000)

# Export in multiple formats
generator.export_dataset(dataset, "./synthetic_data")