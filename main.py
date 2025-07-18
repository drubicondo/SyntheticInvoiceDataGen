#!/usr/bin/env python3
"""Main entry point for FLOPayments ML synthetic data generation"""

import logging
import os
from dotenv import load_dotenv

from flopayments_ml import SyntheticDataGenerator, DEFAULT_CONFIG
from flopayments_ml.core.exceptions import EnvironmentConfigError
from flopayments_ml.utils.file_utils import check_write_permission

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # Validate write permissions
        assert check_write_permission('output/invoices.csv')
        assert check_write_permission('output/payments.csv')
        assert check_write_permission('output/ground_truth.csv')
        
        # Load environment
        success = load_dotenv()
        if not success:
            raise EnvironmentConfigError('Cannot load .env file')
        
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentConfigError('AZURE_OPENAI_API_KEY not found')
        
        # Initialize generator
        generator = SyntheticDataGenerator(
            config=DEFAULT_CONFIG,
            azure_endpoint="https://iason-gpt-4.openai.azure.com"
        )
        
        # Generate dataset
        logger.info("Generating synthetic dataset...")
        dataset = generator.generate_dataset()
        
        # Export dataset
        generator.export_dataset(dataset, output_dir="output")
        
        logger.info("Dataset generation completed!")
        
    except Exception as e:
        logger.error(f"Error during dataset generation: {e}")
        raise

if __name__ == "__main__":
    main()